"""SQLite persistence layer with WAL mode and NumPy float32 embedding serialization."""

from __future__ import annotations

from pathlib import Path
import sqlite3
import threading

import numpy as np


class WathiqStorage:
    """Thread-safe SQLite storage backend for documents, chunks, embeddings, and audit logs."""

    def __init__(self, db_path: str) -> None:
        self.db_path = db_path
        if db_path != ":memory:":
            Path(db_path).parent.mkdir(parents=True, exist_ok=True)
        self._lock = threading.RLock()
        self._conn = sqlite3.connect(db_path, check_same_thread=False)
        self._conn.row_factory = sqlite3.Row
        with self._lock:
            self._conn.execute("PRAGMA journal_mode=WAL;")
            self._conn.execute("PRAGMA foreign_keys=ON;")

    def init_schema(self) -> None:
        """Create database tables and indexes if they do not exist."""
        with self._lock, self._conn:
            self._conn.executescript(
                """
                CREATE TABLE IF NOT EXISTS documents (
                    id              INTEGER PRIMARY KEY AUTOINCREMENT,
                    title           TEXT NOT NULL,
                    source_type     TEXT NOT NULL DEFAULT 'text',
                    content_hash    TEXT NOT NULL UNIQUE,
                    sensitivity     TEXT NOT NULL DEFAULT 'INTERNAL',
                    pii_redacted    INTEGER NOT NULL DEFAULT 0,
                    page_count      INTEGER NOT NULL DEFAULT 1,
                    chunk_count     INTEGER NOT NULL DEFAULT 0,
                    created_at      TEXT NOT NULL DEFAULT (strftime('%Y-%m-%dT%H:%M:%SZ', 'now'))
                );

                CREATE TABLE IF NOT EXISTS document_chunks (
                    id                  INTEGER PRIMARY KEY AUTOINCREMENT,
                    document_id         INTEGER NOT NULL REFERENCES documents(id) ON DELETE CASCADE,
                    chunk_index         INTEGER NOT NULL,
                    page_number         INTEGER NOT NULL DEFAULT 1,
                    content             TEXT NOT NULL,
                    normalized_content  TEXT NOT NULL,
                    token_count         INTEGER NOT NULL DEFAULT 0,
                    char_count          INTEGER NOT NULL DEFAULT 0,
                    chunk_hash          TEXT NOT NULL,
                    embedding           BLOB NOT NULL,
                    UNIQUE(document_id, chunk_index)
                );

                CREATE INDEX IF NOT EXISTS idx_chunks_doc_id
                    ON document_chunks(document_id);

                CREATE TABLE IF NOT EXISTS query_audit_logs (
                    id                  INTEGER PRIMARY KEY AUTOINCREMENT,
                    query_text          TEXT NOT NULL,
                    pii_redacted        INTEGER NOT NULL DEFAULT 0,
                    guardrail_blocked   INTEGER NOT NULL DEFAULT 0,
                    blocked_reason      TEXT,
                    retrieved_count     INTEGER NOT NULL DEFAULT 0,
                    faithfulness_score  REAL NOT NULL DEFAULT 0.0,
                    latency_ms          REAL NOT NULL DEFAULT 0.0,
                    created_at          TEXT NOT NULL DEFAULT (strftime('%Y-%m-%dT%H:%M:%SZ', 'now'))
                );
                """
            )

    def document_exists_by_hash(self, content_hash: str) -> dict | None:
        """Return existing document metadata if a document with content_hash already exists."""
        with self._lock:
            row = self._conn.execute(
                """
                SELECT id, title, source_type, content_hash, sensitivity,
                       pii_redacted, page_count, chunk_count, created_at
                FROM documents
                WHERE content_hash = ?
                """,
                (content_hash,),
            ).fetchone()
            return dict(row) if row else None

    def save_document_with_chunks(
        self,
        title: str,
        source_type: str,
        content_hash: str,
        sensitivity: str,
        pii_redacted: int,
        page_count: int,
        chunks: list[dict],
        embeddings: list[np.ndarray],
    ) -> dict:
        """Persist a document and all its chunks/embeddings atomically."""
        if len(chunks) != len(embeddings):
            raise ValueError("chunks and embeddings must have identical lengths")

        with self._lock, self._conn:
            cur = self._conn.execute(
                """
                INSERT INTO documents (
                    title, source_type, content_hash, sensitivity,
                    pii_redacted, page_count, chunk_count
                )
                VALUES (?, ?, ?, ?, ?, ?, ?)
                """,
                (
                    title,
                    source_type,
                    content_hash,
                    sensitivity,
                    pii_redacted,
                    page_count,
                    len(chunks),
                ),
            )
            doc_id = int(cur.lastrowid)

            chunk_rows = [
                (
                    doc_id,
                    int(chunk["chunk_index"]),
                    int(chunk["page_number"]),
                    chunk["content"],
                    chunk["normalized_content"],
                    int(chunk["token_count"]),
                    int(chunk["char_count"]),
                    chunk["chunk_hash"],
                    np.asarray(emb, dtype=np.float32).tobytes(),
                )
                for chunk, emb in zip(chunks, embeddings)
            ]

            self._conn.executemany(
                """
                INSERT INTO document_chunks (
                    document_id, chunk_index, page_number, content,
                    normalized_content, token_count, char_count,
                    chunk_hash, embedding
                )
                VALUES (?, ?, ?, ?, ?, ?, ?, ?, ?)
                """,
                chunk_rows,
            )

            created = self._conn.execute(
                """
                SELECT id, title, source_type, content_hash, sensitivity,
                       pii_redacted, page_count, chunk_count, created_at
                FROM documents
                WHERE id = ?
                """,
                (doc_id,),
            ).fetchone()
            return dict(created)

    def list_documents(self) -> list[dict]:
        """List all ingested documents ordered newest first."""
        with self._lock:
            rows = self._conn.execute(
                """
                SELECT id, title, source_type, content_hash, sensitivity,
                       pii_redacted, page_count, chunk_count, created_at
                FROM documents
                ORDER BY id DESC
                """
            ).fetchall()
            return [dict(r) for r in rows]

    def get_all_chunks(self) -> list[dict]:
        """Load all chunks along with parent document title and float32 embedding vectors."""
        with self._lock:
            rows = self._conn.execute(
                """
                SELECT c.id AS chunk_id,
                       c.document_id AS doc_id,
                       d.title AS doc_title,
                       d.sensitivity AS sensitivity,
                       c.chunk_index,
                       c.page_number,
                       c.content,
                       c.normalized_content,
                       c.token_count,
                       c.char_count,
                       c.chunk_hash,
                       c.embedding
                FROM document_chunks c
                JOIN documents d ON d.id = c.document_id
                ORDER BY c.id ASC
                """
            ).fetchall()

        results: list[dict] = []
        for row in rows:
            item = dict(row)
            blob = item.pop("embedding")
            item["embedding"] = np.frombuffer(blob, dtype=np.float32).copy()
            results.append(item)
        return results

    def log_query(
        self,
        query_text: str,
        pii_redacted: int,
        guardrail_blocked: bool,
        blocked_reason: str | None,
        retrieved_count: int,
        faithfulness_score: float,
        latency_ms: float,
    ) -> int:
        """Insert a query audit log entry and return its ID."""
        with self._lock, self._conn:
            cur = self._conn.execute(
                """
                INSERT INTO query_audit_logs (
                    query_text, pii_redacted, guardrail_blocked, blocked_reason,
                    retrieved_count, faithfulness_score, latency_ms
                )
                VALUES (?, ?, ?, ?, ?, ?, ?)
                """,
                (
                    query_text,
                    int(pii_redacted),
                    1 if guardrail_blocked else 0,
                    blocked_reason,
                    int(retrieved_count),
                    float(faithfulness_score),
                    float(latency_ms),
                ),
            )
            return int(cur.lastrowid)

    def get_metrics(self, limit: int = 20) -> dict:
        """Compute aggregate corpus and audit telemetry metrics."""
        with self._lock:
            doc_stats = self._conn.execute(
                """
                SELECT COUNT(*) AS total_documents,
                       COALESCE(SUM(chunk_count), 0) AS total_chunks,
                       COALESCE(SUM(pii_redacted), 0) AS doc_pii_redactions
                FROM documents
                """
            ).fetchone()

            query_stats = self._conn.execute(
                """
                SELECT COUNT(*) AS total_queries,
                       COALESCE(AVG(latency_ms), 0.0) AS avg_latency_ms,
                       COALESCE(AVG(CASE WHEN guardrail_blocked = 0 THEN faithfulness_score END), 0.0) AS avg_faithfulness,
                       COALESCE(SUM(guardrail_blocked), 0) AS guardrail_blocks,
                       COALESCE(SUM(pii_redacted), 0) AS query_pii_redactions
                FROM query_audit_logs
                """
            ).fetchone()

            recent_rows = self._conn.execute(
                """
                SELECT id, query_text, pii_redacted, guardrail_blocked,
                       blocked_reason, retrieved_count, faithfulness_score,
                       latency_ms, created_at
                FROM query_audit_logs
                ORDER BY id DESC
                LIMIT ?
                """,
                (int(limit),),
            ).fetchall()

        return {
            "total_documents": int(doc_stats["total_documents"]),
            "total_chunks": int(doc_stats["total_chunks"]),
            "total_queries": int(query_stats["total_queries"]),
            "avg_latency_ms": round(float(query_stats["avg_latency_ms"]), 2),
            "avg_faithfulness": round(float(query_stats["avg_faithfulness"]), 4),
            "guardrail_blocks": int(query_stats["guardrail_blocks"]),
            "pii_redactions": int(doc_stats["doc_pii_redactions"]) + int(query_stats["query_pii_redactions"]),
            "recent_queries": [
                {
                    **dict(r),
                    "guardrail_blocked": bool(r["guardrail_blocked"]),
                }
                for r in recent_rows
            ],
        }

    def close(self) -> None:
        """Close the SQLite database connection."""
        with self._lock:
            self._conn.close()
