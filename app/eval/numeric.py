"""Numeric consistency: every figure in the answer must appear in the evidence.

A pure check (no model). We extract financial-looking figures (those with a
currency sign, thousands separator, decimal, or percent), normalize away ``$``,
``,`` and ``%``, and confirm each answer figure also appears in the cited evidence.
Anything that does not is flagged as an unverified number. Cross-checking against
the XBRL ``numeric_facts`` table is a later enhancement.
"""

from __future__ import annotations

import re
from dataclasses import dataclass, field

_FIGURE = re.compile(r"\$?\s?\d[\d,]*\.?\d*\s?%?")


@dataclass(frozen=True)
class NumericCheck:
    consistent: bool
    unverified: list[str] = field(default_factory=list)


def _is_financial(token: str) -> bool:
    return any(ch in token for ch in (",", ".", "$", "%"))


def _normalize(token: str) -> str:
    return token.replace("$", "").replace(",", "").replace("%", "").replace(" ", "")


def extract_financial_numbers(text: str) -> set[str]:
    """Return normalized financial figures found in ``text``."""
    numbers: set[str] = set()
    for token in _FIGURE.findall(text):
        if not _is_financial(token):
            continue
        normalized = _normalize(token).rstrip(".")
        if normalized and normalized != ".":
            numbers.add(normalized)
    return numbers


def check_numeric_consistency(answer_text: str, evidence_text: str) -> NumericCheck:
    """Flag answer figures that do not appear in the cited evidence."""
    answer_numbers = extract_financial_numbers(answer_text)
    evidence_numbers = extract_financial_numbers(evidence_text)
    unverified = sorted(n for n in answer_numbers if n not in evidence_numbers)
    return NumericCheck(consistent=not unverified, unverified=unverified)
