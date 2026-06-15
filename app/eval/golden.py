"""The golden set and the evaluation harness.

A golden item is a question plus expectations: whether it should refuse, which
filing(s) a good answer cites, and substrings a good answer contains. The harness
runs the answerer over the set and reports refusal accuracy, citation hit rate,
answer-substring rate, and mean faithfulness.
"""

from __future__ import annotations

from dataclasses import dataclass, field
from pathlib import Path
from typing import Any, Protocol

import yaml

from app.generation.models import Answer


class Answerable(Protocol):
    """The answer surface the harness evaluates (satisfied by Answerer)."""

    def answer(
        self,
        question: str,
        *,
        ticker: str | None = None,
        form: str | None = None,
        section: str | None = None,
    ) -> Answer: ...


@dataclass(frozen=True)
class GoldenItem:
    question: str
    ticker: str | None = None
    should_refuse: bool = False
    expect_accessions: list[str] = field(default_factory=list)
    expect_substrings: list[str] = field(default_factory=list)


@dataclass(frozen=True)
class EvalReport:
    total: int
    refusal_accuracy: float
    citation_hit_rate: float
    substring_hit_rate: float
    mean_faithfulness: float


def load_golden(path: Path) -> list[GoldenItem]:
    """Load golden items from a YAML list."""
    raw: Any = yaml.safe_load(path.read_text(encoding="utf-8")) or []
    return [
        GoldenItem(
            question=str(item["question"]),
            ticker=item.get("ticker"),
            should_refuse=bool(item.get("should_refuse", False)),
            expect_accessions=[str(a) for a in item.get("expect_accessions", [])],
            expect_substrings=[str(s) for s in item.get("expect_substrings", [])],
        )
        for item in raw
    ]


def run_golden(answerer: Answerable, items: list[GoldenItem]) -> EvalReport:
    """Run the answerer over the golden set and aggregate metrics."""
    if not items:
        return EvalReport(0, 1.0, 1.0, 1.0, 1.0)

    refusal_correct = 0
    answered = 0
    citation_hits = 0
    substring_hits = 0
    faithfulness_sum = 0.0
    faithfulness_count = 0

    for item in items:
        answer = answerer.answer(item.question, ticker=item.ticker)
        if answer.refused == item.should_refuse:
            refusal_correct += 1
        if item.should_refuse or answer.refused:
            continue

        answered += 1
        accessions = {citation.accession for citation in answer.citations}
        if not item.expect_accessions or accessions & set(item.expect_accessions):
            citation_hits += 1
        if not item.expect_substrings or all(
            s.lower() in answer.text.lower() for s in item.expect_substrings
        ):
            substring_hits += 1
        if answer.faithfulness is not None:
            faithfulness_sum += answer.faithfulness
            faithfulness_count += 1

    return EvalReport(
        total=len(items),
        refusal_accuracy=refusal_correct / len(items),
        citation_hit_rate=citation_hits / answered if answered else 0.0,
        substring_hit_rate=substring_hits / answered if answered else 0.0,
        mean_faithfulness=(faithfulness_sum / faithfulness_count if faithfulness_count else 0.0),
    )
