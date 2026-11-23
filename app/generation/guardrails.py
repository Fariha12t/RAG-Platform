"""Prompt-injection defences. Document text is *untrusted data*: we (1) flag suspicious
chunks at ingestion, (2) redact instruction-like sentences before they reach the model,
(3) neutralise our own prompt delimiters, and (4) tell the model to never obey source text."""
from __future__ import annotations

import re

from app.textutils import split_sentences

INJECTION_PATTERNS = [
    r"ignore\s+(all\s+|any\s+|the\s+)*(previous|prior|above|earlier|preceding)\s+(instructions?|prompts?|messages?|rules?)",
    r"disregard\s+(all\s+|any\s+|the\s+)*(previous|prior|above|earlier|preceding)",
    r"forget\s+(everything|all|your)\s+(you|instructions?|rules?|previous)",
    r"(reveal|print|show|repeat|leak)\s+(the\s+|your\s+)?(system|hidden|secret|initial)\s+(prompt|instructions?)",
    r"you\s+are\s+now\s+(a|an|in|the)\b",
    r"\bnew\s+instructions?\s*:",
    r"\b(system|developer|assistant)\s*(prompt|message)\s*:",
    r"do\s+not\s+(tell|inform|mention)\s+(this\s+to\s+)?the\s+user",
    r"tell\s+the\s+user\s+(that|to)\b",
    r"<\s*/?\s*(system|assistant|instructions?)\s*>",
    r"\[/?INST\]|<\|im_(start|end)\|>",
    r"\boverride\s+(your|the|all)\s+(safety|instructions?|rules?|guidelines)",
]
_COMPILED = [re.compile(p, re.IGNORECASE) for p in INJECTION_PATTERNS]
REDACTION = "[redacted: possible prompt-injection text]"


def scan(text: str) -> list[str]:
    """Return the patterns that match (empty list == clean)."""
    return [p.pattern for p in _COMPILED if p.search(text)]


def sanitize_source(text: str) -> tuple[str, bool]:
    """Redact instruction-like sentences and neutralise prompt delimiters.
    Returns (clean_text, was_modified)."""
    modified = False
    kept = []
    for sent in split_sentences(text) or [text]:
        if any(p.search(sent) for p in _COMPILED):
            kept.append(REDACTION)
            modified = True
        else:
            kept.append(sent)
    clean = " ".join(kept)
    neutral = re.sub(r"<\s*(/?)\s*(source|sources|question)\b", r"‹\1\2", clean, flags=re.IGNORECASE)
    return neutral, modified or neutral != clean
