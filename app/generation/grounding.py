"""Citation enforcement: keep only cited sentences; detect refusals.

This is citation *presence* enforcement, not entailment. A sentence is kept only
if it carries at least one in-range ``[n]`` marker. Coverage is the fraction of the
model's sentences that survived. The entailment verifier (does the cited span
actually support the sentence?) is Phase 6.
"""

from __future__ import annotations

import re
from dataclasses import dataclass

from app.generation.prompt import REFUSAL_SENTINEL

_SENTENCE = re.compile(r"(?<=[.!?])\s+")
_CITATION = re.compile(r"\[(\d+)\]")


@dataclass(frozen=True)
class CitationParse:
    """The enforced answer text, the markers used, and citation coverage."""

    text: str
    markers: list[int]
    coverage: float


def is_refusal(raw: str) -> bool:
    """True if the model declined (empty output or the refusal sentinel)."""
    stripped = raw.strip()
    return not stripped or stripped.upper().startswith(REFUSAL_SENTINEL)


def enforce_citations(raw: str, num_chunks: int) -> CitationParse:
    """Drop uncited sentences and report which markers were used and coverage."""
    sentences = [s.strip() for s in _SENTENCE.split(raw.strip()) if s.strip()]
    if not sentences:
        return CitationParse(text="", markers=[], coverage=0.0)

    kept: list[str] = []
    used: set[int] = set()
    for sentence in sentences:
        valid = [
            marker
            for marker in (int(m) for m in _CITATION.findall(sentence))
            if 1 <= marker <= num_chunks
        ]
        if valid:
            kept.append(sentence)
            used.update(valid)

    coverage = len(kept) / len(sentences)
    return CitationParse(text=" ".join(kept), markers=sorted(used), coverage=coverage)
