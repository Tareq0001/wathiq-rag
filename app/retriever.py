"""Hybrid Retriever combining Okapi BM25, Dense Cosine Similarity, and Reciprocal Rank Fusion."""

from __future__ import annotations

from collections import Counter
import math

import numpy as np

from app.etl import compute_embedding, tokenize


class HybridRetriever:
    """Exact Okapi BM25 + Dense Cosine Similarity fused via Reciprocal Rank Fusion (RRF)."""

    def __init__(
        self,
        k1: float = 1.5,
        b: float = 0.75,
        rrf_k: int = 60,
        w_dense: float = 1.0,
        w_bm25: float = 1.0,
    ) -> None:
        self.k1 = k1
        self.b = b
        self.rrf_k = rrf_k
        self.w_dense = w_dense
        self.w_bm25 = w_bm25

    def _compute_bm25_scores(self, query_tokens: list[str], doc_token_lists: list[list[str]]) -> list[float]:
        """Compute exact Okapi BM25 scores for all chunks."""
        n_docs = len(doc_token_lists)
        if n_docs == 0 or not query_tokens:
            return [0.0] * n_docs

        doc_lengths = [len(toks) for toks in doc_token_lists]
        avg_dl = sum(doc_lengths) / n_docs if n_docs > 0 else 1.0
        if avg_dl == 0.0:
            avg_dl = 1.0

        # Document frequency for query terms
        unique_q_terms = set(query_tokens)
        df: dict[str, int] = {term: 0 for term in unique_q_terms}
        tf_per_doc: list[Counter[str]] = []

        for toks in doc_token_lists:
            counts = Counter(toks)
            tf_per_doc.append(counts)
            for term in unique_q_terms:
                if counts[term] > 0:
                    df[term] += 1

        scores: list[float] = []
        for idx, counts in enumerate(tf_per_doc):
            dl = doc_lengths[idx]
            score = 0.0
            for term in query_tokens:
                tf = counts.get(term, 0)
                if tf == 0:
                    continue
                n_q = df[term]
                idf = math.log((n_docs - n_q + 0.5) / (n_q + 0.5) + 1.0)
                denom = tf + self.k1 * (1.0 - self.b + self.b * (dl / avg_dl))
                score += idf * ((tf * (self.k1 + 1.0)) / denom)
            scores.append(score)
        return scores

    def retrieve(self, query: str, chunks: list[dict], top_k: int = 5) -> list[dict]:
        """Rank corpus chunks using BM25 + Dense Cosine RRF and return top_k hits."""
        if not chunks or top_k <= 0:
            return []

        query_tokens = tokenize(query)
        q_vec = compute_embedding(query)

        doc_token_lists = [tokenize(chunk["normalized_content"]) for chunk in chunks]
        bm25_scores = self._compute_bm25_scores(query_tokens, doc_token_lists)

        dense_scores: list[float] = []
        for chunk in chunks:
            d_vec = chunk["embedding"]
            score = float(np.dot(q_vec, d_vec)) if np.linalg.norm(q_vec) > 0 and np.linalg.norm(d_vec) > 0 else 0.0
            dense_scores.append(score)

        n = len(chunks)
        dense_order = sorted(range(n), key=lambda i: (-dense_scores[i], chunks[i]["chunk_id"]))
        bm25_order = sorted(range(n), key=lambda i: (-bm25_scores[i], chunks[i]["chunk_id"]))

        dense_ranks = [0] * n
        bm25_ranks = [0] * n
        for rank, idx in enumerate(dense_order, start=1):
            dense_ranks[idx] = rank
        for rank, idx in enumerate(bm25_order, start=1):
            bm25_ranks[idx] = rank

        fused: list[dict] = []
        for idx, chunk in enumerate(chunks):
            d_rank = dense_ranks[idx]
            b_rank = bm25_ranks[idx]
            rrf_score = (self.w_dense / (self.rrf_k + d_rank)) + (self.w_bm25 / (self.rrf_k + b_rank))
            fused.append(
                {
                    "chunk_id": int(chunk["chunk_id"]),
                    "doc_id": int(chunk["doc_id"]),
                    "doc_title": chunk["doc_title"],
                    "page_number": int(chunk["page_number"]),
                    "content": chunk["content"],
                    "rrf_score": round(float(rrf_score), 6),
                    "dense_score": round(float(dense_scores[idx]), 6),
                    "bm25_score": round(float(bm25_scores[idx]), 6),
                    "dense_rank": d_rank,
                    "bm25_rank": b_rank,
                }
            )

        fused.sort(
            key=lambda item: (
                -item["rrf_score"],
                -(1 if item["bm25_score"] > 0 else 0),
                -item["dense_score"],
                item["chunk_id"],
            )
        )
        return fused[:top_k]
