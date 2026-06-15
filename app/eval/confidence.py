"""Combined confidence from citation coverage and entailment faithfulness.

Confidence is the weakest link of coverage (fraction of sentences cited) and
faithfulness (fraction of cited sentences the judge confirmed), halved if a number
in the answer could not be verified against the evidence.
"""

from __future__ import annotations


def combine_confidence(coverage: float, faithfulness: float, numeric_ok: bool) -> float:
    """Return a confidence in [0, 1]."""
    base = min(max(0.0, coverage), max(0.0, faithfulness))
    if not numeric_ok:
        base *= 0.5
    return round(min(1.0, base), 3)
