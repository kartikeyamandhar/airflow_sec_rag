"""Confidence scoring.

For Phase 5 the confidence is the citation coverage (the fraction of answer
sentences grounded in a retrieved passage), clamped to [0, 1] and zero when there
are no citations. Phase 6 will enrich this with entailment and retrieval signals.
"""

from __future__ import annotations


def compute_confidence(coverage: float, num_citations: int) -> float:
    """Return a confidence in [0, 1] from grounding coverage."""
    if num_citations == 0:
        return 0.0
    return round(max(0.0, min(1.0, coverage)), 3)
