"""RAG synthesis, citation grounding, faithfulness scoring, and optional Gemini LLM integration."""

from __future__ import annotations

import os
import re
import time

import httpx

from app.etl import tokenize
from app.guardrails import GuardrailReport, inspect_query
from app.retriever import HybridRetriever
from app.storage import WathiqStorage

_SENTENCE_SPLIT_RE = re.compile(r"(?<=[.!؟؛\n])\s+")
_CITATION_TAG_RE = re.compile(r"\[[^\]]+\]")
_STOPWORDS = frozenset(
    {
        "the", "a", "an", "and", "or", "of", "to", "in", "on", "for", "with", "is", "are",
        "was", "were", "be", "by", "as", "at", "from", "that", "this", "it",
        "في", "من", "على", "الى", "إلى", "عن", "مع", "او", "أو", "و", "ان", "أن", "لا",
        "ما", "هو", "هي", "هذا", "هذه", "ذلك", "تلك", "التي", "الذي", "بين", "عند",
    }
)


class GuardrailBlockedError(Exception):
    """Raised when a query violates security guardrails."""

    def __init__(self, report: GuardrailReport) -> None:
        self.report = report
        super().__init__(report.blocked_reason or "Query blocked by guardrail")


class WathiqRAGService:
    """Orchestrates guardrail checks, hybrid retrieval, grounded synthesis, and audit logging."""

    def __init__(self, storage: WathiqStorage, retriever: HybridRetriever | None = None) -> None:
        self.storage = storage
        self.retriever = retriever or HybridRetriever()

    @staticmethod
    def _extract_sentences(text: str) -> list[str]:
        parts = _SENTENCE_SPLIT_RE.split(text.strip())
        return [p.strip() for p in parts if len(p.strip()) >= 12]

    def _synthesize_extractive(self, query: str, hits: list[dict]) -> str:
        """Deterministic extractive synthesis selecting highest-relevance sentences with citations."""
        if not hits:
            return "لم يتم العثور على مقاطع مطابقة في قاعدة المعرفة. / No matching knowledge chunks were found."

        query_terms = [t for t in tokenize(query) if t not in _STOPWORDS]
        query_term_set = set(query_terms)

        candidates: list[tuple[float, str, str]] = []
        seen_sentences: set[str] = set()

        for rank_idx, hit in enumerate(hits):
            citation = f"[{hit['doc_title']} - p.{hit['page_number']}]"
            sentences = self._extract_sentences(hit["content"])
            if not sentences:
                sentences = [hit["content"].strip()]

            rank_bonus = 1.0 / (rank_idx + 1)
            for sent in sentences:
                norm_key = " ".join(tokenize(sent))
                if not norm_key or norm_key in seen_sentences:
                    continue
                seen_sentences.add(norm_key)

                sent_tokens = set(tokenize(sent))
                overlap = len(query_term_set & sent_tokens)
                score = overlap * 2.0 + rank_bonus + hit["rrf_score"] * 10.0
                candidates.append((score, sent, citation))

        candidates.sort(key=lambda x: -x[0])
        selected = candidates[:3]
        if not selected:
            top = hits[0]
            return f"{top['content']} [{top['doc_title']} - p.{top['page_number']}]"

        return " ".join(f"{sent} {citation}" for _, sent, citation in selected)

    def _synthesize_with_gemini(self, query: str, hits: list[dict]) -> str | None:
        """Optional Gemini API synthesis enabled only when GEMINI_API_KEY and USE_EXTERNAL_LLM=1."""
        api_key = os.getenv("GEMINI_API_KEY", "").strip()
        if not api_key or os.getenv("USE_EXTERNAL_LLM") != "1":
            return None

        context_blocks = "\n".join(
            f"- [{h['doc_title']} - p.{h['page_number']}]: {h['content']}"
            for h in hits
        )
        prompt = (
            "Answer the user question strictly using the provided context. "
            "Include inline citations formatted exactly as [DocTitle - p.X].\n\n"
            f"Context:\n{context_blocks}\n\nQuestion: {query}"
        )
        url = (
            "https://generativelanguage.googleapis.com/v1beta/models/"
            f"gemini-2.5-flash:generateContent?key={api_key}"
        )
        try:
            with httpx.Client(timeout=12.0) as client:
                resp = client.post(
                    url,
                    json={"contents": [{"parts": [{"text": prompt}]}]},
                )
                resp.raise_for_status()
                data = resp.json()
                text = data["candidates"][0]["content"]["parts"][0]["text"].strip()
                return text or None
        except Exception:
            return None

    @staticmethod
    def compute_faithfulness(answer: str, hits: list[dict]) -> float:
        """Measure ratio of content tokens in the answer verifiable in retrieved chunks."""
        if not answer or not hits:
            return 0.0

        clean_answer = _CITATION_TAG_RE.sub(" ", answer)
        answer_tokens = [t for t in tokenize(clean_answer) if t not in _STOPWORDS]
        if not answer_tokens:
            return 0.0

        context_tokens: set[str] = set()
        for hit in hits:
            context_tokens.update(tokenize(hit["content"]))

        supported = sum(1 for tok in answer_tokens if tok in context_tokens)
        score = supported / len(answer_tokens)
        return round(max(0.0, min(1.0, score)), 4)

    def answer_query(self, query: str, top_k: int = 5) -> dict:
        """Execute guardrails, hybrid retrieval, grounded synthesis, and audit logging."""
        t0 = time.perf_counter()
        report = inspect_query(query)

        if not report.passed:
            latency_ms = round((time.perf_counter() - t0) * 1000.0, 2)
            self.storage.log_query(
                query_text=report.redacted_text,
                pii_redacted=report.pii_redacted_count,
                guardrail_blocked=True,
                blocked_reason=report.blocked_reason,
                retrieved_count=0,
                faithfulness_score=0.0,
                latency_ms=latency_ms,
            )
            raise GuardrailBlockedError(report)

        chunks = self.storage.get_all_chunks()
        hits = self.retriever.retrieve(report.redacted_text, chunks, top_k=top_k)

        answer = self._synthesize_with_gemini(report.redacted_text, hits)
        if not answer:
            answer = self._synthesize_extractive(report.redacted_text, hits)

        faithfulness = self.compute_faithfulness(answer, hits)
        latency_ms = round((time.perf_counter() - t0) * 1000.0, 2)

        citations: list[dict] = []
        seen_cites: set[tuple[int, int]] = set()
        for hit in hits:
            key = (hit["doc_id"], hit["page_number"])
            if key not in seen_cites:
                seen_cites.add(key)
                citations.append(
                    {
                        "doc_id": hit["doc_id"],
                        "doc_title": hit["doc_title"],
                        "page_number": hit["page_number"],
                        "label": f"[{hit['doc_title']} - p.{hit['page_number']}]",
                    }
                )

        self.storage.log_query(
            query_text=report.redacted_text,
            pii_redacted=report.pii_redacted_count,
            guardrail_blocked=False,
            blocked_reason=None,
            retrieved_count=len(hits),
            faithfulness_score=faithfulness,
            latency_ms=latency_ms,
        )

        return {
            "query": report.redacted_text,
            "answer": answer,
            "citations": citations,
            "retrieved_chunks": hits,
            "latency_ms": latency_ms,
            "faithfulness_score": faithfulness,
            "pii_redacted": report.pii_redacted_count,
            "sensitivity_level": report.sensitivity_level,
        }
