"""Document ETL pipeline: PDF extraction, Arabic/English normalization, chunking, and embeddings."""

from __future__ import annotations

import hashlib
import re

import fitz
import numpy as np

_TATWEEL_RE = re.compile(r"\u0640+")
_DIACRITICS_RE = re.compile(r"[\u064B-\u065F\u0670]")
_ALEF_RE = re.compile(r"[أإآٱ]")
_WHITESPACE_RE = re.compile(r"\s+")
_TOKEN_RE = re.compile(r"[a-z0-9\u0621-\u064a]+")


def sha256_hash(text: str) -> str:
    """Return lowercase hexadecimal SHA-256 digest for UTF-8 text."""
    return hashlib.sha256(text.encode("utf-8")).hexdigest()


def extract_pdf_pages(pdf_bytes: bytes) -> list[dict]:
    """Extract per-page text from raw PDF bytes using PyMuPDF."""
    if not pdf_bytes:
        return []

    pages: list[dict] = []
    with fitz.open(stream=pdf_bytes, filetype="pdf") as doc:
        for idx, page in enumerate(doc, start=1):
            raw_text = page.get_text("text") or ""
            cleaned = raw_text.strip()
            if cleaned:
                pages.append({"page_number": idx, "text": cleaned})
    return pages


def normalize_text(text: str) -> str:
    """Normalize Arabic orthographic variants and whitespace for consistent indexing."""
    if not text:
        return ""
    cleaned = _TATWEEL_RE.sub("", text)
    cleaned = _DIACRITICS_RE.sub("", cleaned)
    cleaned = _ALEF_RE.sub("ا", cleaned)
    cleaned = cleaned.replace("ة", "ه").replace("ى", "ي")
    cleaned = _WHITESPACE_RE.sub(" ", cleaned).strip()
    return cleaned


def tokenize(text: str) -> list[str]:
    """Extract lowercased English and normalized Arabic tokens."""
    normalized = normalize_text(text).lower()
    return _TOKEN_RE.findall(normalized)


def _slice_on_word_boundary(text: str, start: int, max_len: int) -> int:
    """Find an end offset <= start + max_len aligned to a whitespace boundary when possible."""
    end = min(len(text), start + max_len)
    if end == len(text):
        return end

    # Prefer splitting at whitespace in the second half of the window to preserve whole words
    split_pos = text.rfind(" ", start + max_len // 2, end + 1)
    if split_pos > start:
        return split_pos
    return end


def chunk_document(
    pages: list[dict],
    chunk_size: int = 450,
    overlap: int = 90,
) -> list[dict]:
    """Split extracted pages into overlapping word-boundary-aware chunks with SHA-256 hashes."""
    if chunk_size <= 0:
        raise ValueError("chunk_size must be positive")
    if overlap < 0 or overlap >= chunk_size:
        raise ValueError("overlap must satisfy 0 <= overlap < chunk_size")

    chunks: list[dict] = []
    global_idx = 0

    for page in pages:
        page_num = int(page.get("page_number", 1))
        raw_text = _WHITESPACE_RE.sub(" ", str(page.get("text", ""))).strip()
        if not raw_text:
            continue

        start = 0
        text_len = len(raw_text)
        while start < text_len:
            end = _slice_on_word_boundary(raw_text, start, chunk_size)
            chunk_text = raw_text[start:end].strip()
            if chunk_text:
                norm_text = normalize_text(chunk_text).lower()
                tokens = _TOKEN_RE.findall(norm_text)
                chunks.append(
                    {
                        "chunk_index": global_idx,
                        "page_number": page_num,
                        "content": chunk_text,
                        "normalized_content": norm_text,
                        "token_count": len(tokens),
                        "char_count": len(chunk_text),
                        "chunk_hash": sha256_hash(chunk_text),
                    }
                )
                global_idx += 1

            if end >= text_len:
                break

            next_start = max(start + 1, end - overlap)
            # Align overlap start forward to the next word start if inside a word
            if next_start < text_len and raw_text[next_start - 1] != " ":
                space_pos = raw_text.find(" ", next_start)
                if space_pos != -1 and space_pos < end:
                    next_start = space_pos + 1
            start = next_start

    return chunks


def _hash_feature(feature: str, dim: int, weight: float, vec: np.ndarray) -> None:
    """Project a single string feature into the dense vector via signed blake2b hashing."""
    digest = hashlib.blake2b(feature.encode("utf-8"), digest_size=8).digest()
    val = int.from_bytes(digest, "little")
    bucket = val % dim
    sign = 1.0 if ((val >> 63) & 1) == 0 else -1.0
    vec[bucket] += sign * weight


def compute_embedding(text: str, dim: int = 256) -> np.ndarray:
    """Compute a deterministic L2-normalized float32 vector over word and char n-grams."""
    vec = np.zeros(dim, dtype=np.float32)
    tokens = tokenize(text)
    if not tokens:
        return vec

    # Word unigrams carry primary semantic weight
    for tok in tokens:
        _hash_feature(f"w1:{tok}", dim, 1.5, vec)

    # Word bigrams capture local phrase order
    for i in range(len(tokens) - 1):
        _hash_feature(f"w2:{tokens[i]}_{tokens[i + 1]}", dim, 1.0, vec)

    # Character trigrams improve robustness to Arabic morphology and typos
    joined = " ".join(tokens)
    if len(joined) >= 3:
        for i in range(len(joined) - 2):
            _hash_feature(f"c3:{joined[i : i + 3]}", dim, 0.35, vec)

    norm = float(np.linalg.norm(vec))
    if norm > 0.0:
        vec /= norm
    return vec.astype(np.float32)
