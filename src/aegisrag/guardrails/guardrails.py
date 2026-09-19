"""Guardrails (§16 of the design doc): input / retrieval / output checkpoints.

Document content and user queries are always treated as DATA here, never as instructions —
that's the actual defense against prompt injection, not a prompt asking the model to behave.
"""
from __future__ import annotations

import re
from dataclasses import dataclass

MAX_QUERY_CHARS = 2000

# Patterns that indicate an attempt to override the system's instructions — deliberately broad;
# false positives here just mean "flagged for review," not silently blocked.
_INJECTION_PATTERNS = [
    r"ignore (all |any )?(previous|prior|above) instructions",
    r"disregard (all |any )?(previous|prior|above) instructions",
    r"you are now",
    r"system prompt",
    r"reveal your (instructions|prompt|system prompt)",
    r"act as (?!an? (assistant|helper))",
    r"new instructions:",
]
_INJECTION_RE = re.compile("|".join(_INJECTION_PATTERNS), re.IGNORECASE)


@dataclass
class GuardrailResult:
    allowed: bool
    reason: str = ""
    flagged: bool = False


def check_input(query: str) -> GuardrailResult:
    if not query or not query.strip():
        return GuardrailResult(allowed=False, reason="empty query")
    if len(query) > MAX_QUERY_CHARS:
        return GuardrailResult(allowed=False, reason=f"query exceeds {MAX_QUERY_CHARS} characters")
    if _INJECTION_RE.search(query):
        return GuardrailResult(
            allowed=True, flagged=True, reason="possible prompt-injection pattern in user query"
        )
    return GuardrailResult(allowed=True)


def check_retrieved_content(passages: list[str]) -> GuardrailResult:
    """Retrieved document text is scanned too — a PDF can contain injection attempts just as
    easily as a user query. Flagged passages are still usable as DATA by the agents (the
    synthesis prompt already tells the model to treat evidence as data, never commands) but the
    flag is recorded so a reviewer can see it happened."""
    flagged = any(_INJECTION_RE.search(p) for p in passages)
    return GuardrailResult(allowed=True, flagged=flagged, reason="pattern found in retrieved text" if flagged else "")


def check_output(answer: str, confidence: float, qualify_threshold: float) -> GuardrailResult:
    if not answer or not answer.strip():
        return GuardrailResult(allowed=False, reason="empty answer")
    if confidence < qualify_threshold:
        return GuardrailResult(
            allowed=False, reason=f"confidence {confidence:.2f} below qualify threshold {qualify_threshold}"
        )
    return GuardrailResult(allowed=True)
