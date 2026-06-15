"""Value objects for the generation layer."""

from __future__ import annotations

from dataclasses import dataclass, field
from datetime import date


@dataclass(frozen=True)
class Citation:
    """A source span backing a sentence in the answer.

    ``marker`` is the ``[n]`` number used inline. ``char_start``/``char_end`` index
    into the source passage; ``as_of`` is the filing's period of report.
    """

    marker: int
    accession: str
    ticker: str | None
    form: str
    section: str
    char_start: int
    char_end: int
    text: str
    as_of: date | None


@dataclass(frozen=True)
class Answer:
    """A grounded answer, or a refusal when ``refused`` is true."""

    question: str
    text: str
    citations: list[Citation] = field(default_factory=list)
    confidence: float = 0.0
    refused: bool = False
    reason: str | None = None
