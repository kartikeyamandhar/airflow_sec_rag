"""The deploy gate: block a deploy when metrics regress below thresholds.

Pure logic so it is unit-tested without any model calls. Fails closed: each metric
below its threshold is a recorded failure, and any failure fails the gate.
"""

from __future__ import annotations

from dataclasses import dataclass

from app.eval.golden import EvalReport


@dataclass(frozen=True)
class Thresholds:
    min_refusal_accuracy: float
    min_citation_hit_rate: float
    min_faithfulness: float


@dataclass(frozen=True)
class GateResult:
    passed: bool
    failures: list[str]


def evaluate_gate(report: EvalReport, thresholds: Thresholds) -> GateResult:
    """Return whether the report clears every threshold, with reasons on failure."""
    failures: list[str] = []
    if report.refusal_accuracy < thresholds.min_refusal_accuracy:
        failures.append(
            f"refusal_accuracy {report.refusal_accuracy:.2f} "
            f"< {thresholds.min_refusal_accuracy:.2f}"
        )
    if report.citation_hit_rate < thresholds.min_citation_hit_rate:
        failures.append(
            f"citation_hit_rate {report.citation_hit_rate:.2f} "
            f"< {thresholds.min_citation_hit_rate:.2f}"
        )
    if report.mean_faithfulness < thresholds.min_faithfulness:
        failures.append(
            f"mean_faithfulness {report.mean_faithfulness:.2f} < {thresholds.min_faithfulness:.2f}"
        )
    return GateResult(passed=not failures, failures=failures)
