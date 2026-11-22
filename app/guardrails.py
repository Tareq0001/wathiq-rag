"""Saudi PDPL PII masking and prompt/SQL injection guardrails."""

from __future__ import annotations

from dataclasses import dataclass, field
import re

# Ordered from most specific structural formats to general 10-digit IDs
_PII_PATTERNS: tuple[tuple[str, re.Pattern[str], str], ...] = (
    (
        "SAUDI_IBAN",
        re.compile(r"\bSA\d{22}\b", re.IGNORECASE),
        "[REDACTED_IBAN]",
    ),
    (
        "SAUDI_PHONE",
        re.compile(r"(?<!\d)(?:\+9665|05)\d{8}(?!\d)"),
        "[REDACTED_PHONE]",
    ),
    (
        "SAUDI_NATIONAL_ID",
        re.compile(r"(?<!\d)[12]\d{9}(?!\d)"),
        "[REDACTED_SAUDI_ID]",
    ),
    (
        "EMAIL",
        re.compile(r"\b[A-Za-z0-9._%+-]+@[A-Za-z0-9.-]+\.[A-Za-z]{2,}\b"),
        "[REDACTED_EMAIL]",
    ),
)

_INJECTION_PATTERNS: tuple[tuple[str, re.Pattern[str]], ...] = (
    ("PROMPT_INJECTION", re.compile(r"ignore\s+(?:all\s+)?previous\s+instructions", re.IGNORECASE)),
    ("PROMPT_INJECTION", re.compile(r"disregard\s+(?:all\s+)?previous\s+instructions", re.IGNORECASE)),
    ("PROMPT_INJECTION", re.compile(r"reveal\s+(?:the\s+)?system\s+prompt", re.IGNORECASE)),
    ("PROMPT_INJECTION", re.compile(r"\bsystem\s+prompt\b", re.IGNORECASE)),
    ("SQL_INJECTION", re.compile(r"\bdrop\s+table\b", re.IGNORECASE)),
    ("SQL_INJECTION", re.compile(r"\bunion\s+select\b", re.IGNORECASE)),
    ("SQL_INJECTION", re.compile(r"\bdelete\s+from\b", re.IGNORECASE)),
    ("ARABIC_PROMPT_INJECTION", re.compile(r"تجاهل\s+(?:كل\s+|كافة\s+)?التعليمات\s+السابقة")),
    ("ARABIC_PROMPT_INJECTION", re.compile(r"اكشف\s+(?:لي\s+)?(?:عن\s+)?التعليمات")),
    ("ARABIC_PROMPT_INJECTION", re.compile(r"اعرض\s+موجه\s+النظام")),
)


@dataclass(slots=True)
class GuardrailReport:
    """Result of PII masking and security inspection."""

    passed: bool
    pii_redacted_count: int
    redacted_text: str
    blocked_reason: str | None = None
    sensitivity_level: str = "INTERNAL"
    detected_entities: list[str] = field(default_factory=list)


def _mask_pii(text: str) -> tuple[str, int, list[str]]:
    """Mask Saudi PDPL regulated PII entities and return redacted text, count, and entity labels."""
    redacted = text
    total_count = 0
    entities: list[str] = []

    for label, pattern, replacement in _PII_PATTERNS:
        redacted, count = pattern.subn(replacement, redacted)
        if count > 0:
            total_count += count
            entities.append(label)

    return redacted, total_count, entities


def inspect_document(text: str) -> GuardrailReport:
    """Inspect document text and mask Saudi PDPL PII before storage or indexing."""
    redacted, pii_count, entities = _mask_pii(text)
    sensitivity = "CONFIDENTIAL_PII" if pii_count > 0 else "INTERNAL"
    return GuardrailReport(
        passed=True,
        pii_redacted_count=pii_count,
        redacted_text=redacted,
        blocked_reason=None,
        sensitivity_level=sensitivity,
        detected_entities=entities,
    )


def inspect_query(query: str) -> GuardrailReport:
    """Inspect user query for prompt/SQL injection and mask any PII."""
    redacted, pii_count, entities = _mask_pii(query)
    sensitivity = "CONFIDENTIAL_PII" if pii_count > 0 else "INTERNAL"

    for threat_label, pattern in _INJECTION_PATTERNS:
        if pattern.search(query):
            return GuardrailReport(
                passed=False,
                pii_redacted_count=pii_count,
                redacted_text=redacted,
                blocked_reason=f"Blocked by security guardrail ({threat_label})",
                sensitivity_level=sensitivity,
                detected_entities=entities,
            )

    return GuardrailReport(
        passed=True,
        pii_redacted_count=pii_count,
        redacted_text=redacted,
        blocked_reason=None,
        sensitivity_level=sensitivity,
        detected_entities=entities,
    )
