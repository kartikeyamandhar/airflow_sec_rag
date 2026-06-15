"""Citation parsing and enforcement.

`parse_answer` splits the model's output into sentences and keeps those that carry
at least one in-range ``[n]`` citation, as `Claim`s (sentence + its markers).
`enforce_citations` is the Phase 5 view (joined text, used markers, coverage),
re-expressed on top of `parse_answer`. The Phase 6 verifier consumes `Claim`s to
check entailment. This is citation *presence*; entailment is the verifier's job.
"""

from __future__ import annotations

import re
from dataclasses import dataclass

from app.generation.prompt import REFUSAL_SENTINEL

_SENTENCE = re.compile(r"(?<=[.!?])\s+")
_CITATION = re.compile(r"\[(\d+)\]")


@dataclass(frozen=True)
class Claim:
    """A single answer sentence and the in-range markers it cites."""

    text: str
    markers: list[int]


@dataclass(frozen=True)
class ParsedAnswer:
    """Cited claims plus the total sentence count (for coverage)."""

    claims: list[Claim]
    total_sentences: int


@dataclass(frozen=True)
class CitationParse:
    """The Phase 5 view: joined cited text, used markers, and coverage."""

    text: str
    markers: list[int]
    coverage: float


def is_refusal(raw: str) -> bool:
    """True if the model declined (empty output or the refusal sentinel)."""
    stripped = raw.strip()
    return not stripped or stripped.upper().startswith(REFUSAL_SENTINEL)


def parse_answer(raw: str, num_chunks: int) -> ParsedAnswer:
    """Split into sentences; keep those with an in-range citation as claims."""
    sentences = [s.strip() for s in _SENTENCE.split(raw.strip()) if s.strip()]
    claims: list[Claim] = []
    for sentence in sentences:
        markers = sorted(
            {
                marker
                for marker in (int(m) for m in _CITATION.findall(sentence))
                if 1 <= marker <= num_chunks
            }
        )
        if markers:
            claims.append(Claim(text=sentence, markers=markers))
    return ParsedAnswer(claims=claims, total_sentences=len(sentences))


def enforce_citations(raw: str, num_chunks: int) -> CitationParse:
    """Drop uncited sentences; report joined text, used markers, and coverage."""
    parsed = parse_answer(raw, num_chunks)
    if parsed.total_sentences == 0:
        return CitationParse(text="", markers=[], coverage=0.0)
    markers = sorted({m for claim in parsed.claims for m in claim.markers})
    text = " ".join(claim.text for claim in parsed.claims)
    coverage = len(parsed.claims) / parsed.total_sentences
    return CitationParse(text=text, markers=markers, coverage=coverage)
