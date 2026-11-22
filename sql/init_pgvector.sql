-- PostgreSQL 16 + pgvector schema for Wathiq Enterprise Document ETL & Hybrid RAG Engine
CREATE EXTENSION IF NOT EXISTS vector;

CREATE TABLE IF NOT EXISTS documents (
    id              BIGSERIAL PRIMARY KEY,
    title           TEXT NOT NULL,
    source_type     TEXT NOT NULL DEFAULT 'text',
    content_hash    TEXT NOT NULL UNIQUE,
    sensitivity     TEXT NOT NULL DEFAULT 'INTERNAL',
    pii_redacted    INTEGER NOT NULL DEFAULT 0,
    page_count      INTEGER NOT NULL DEFAULT 1,
    chunk_count     INTEGER NOT NULL DEFAULT 0,
    created_at      TIMESTAMPTZ NOT NULL DEFAULT NOW()
);

CREATE TABLE IF NOT EXISTS document_chunks (
    id                  BIGSERIAL PRIMARY KEY,
    document_id         BIGINT NOT NULL REFERENCES documents(id) ON DELETE CASCADE,
    chunk_index         INTEGER NOT NULL,
    page_number         INTEGER NOT NULL DEFAULT 1,
    content             TEXT NOT NULL,
    normalized_content  TEXT NOT NULL,
    token_count         INTEGER NOT NULL DEFAULT 0,
    char_count          INTEGER NOT NULL DEFAULT 0,
    chunk_hash          TEXT NOT NULL,
    embedding           vector(256) NOT NULL,
    UNIQUE (document_id, chunk_index)
);

CREATE INDEX IF NOT EXISTS idx_document_chunks_embedding_hnsw
    ON document_chunks
    USING hnsw (embedding vector_cosine_ops)
    WITH (m = 16, ef_construction = 64);

CREATE INDEX IF NOT EXISTS idx_document_chunks_fts
    ON document_chunks
    USING gin (to_tsvector('simple', normalized_content));

CREATE TABLE IF NOT EXISTS query_audit_logs (
    id                  BIGSERIAL PRIMARY KEY,
    query_text          TEXT NOT NULL,
    pii_redacted        INTEGER NOT NULL DEFAULT 0,
    guardrail_blocked   BOOLEAN NOT NULL DEFAULT FALSE,
    blocked_reason      TEXT,
    retrieved_count     INTEGER NOT NULL DEFAULT 0,
    faithfulness_score  DOUBLE PRECISION NOT NULL DEFAULT 0.0,
    latency_ms          DOUBLE PRECISION NOT NULL DEFAULT 0.0,
    created_at          TIMESTAMPTZ NOT NULL DEFAULT NOW()
);
