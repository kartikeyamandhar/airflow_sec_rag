"""Decompose comparison questions into per-company sub-queries.

A heuristic for Phase 4: if a question names two or more known companies, split it
into one sub-query per company (each filtered to that company), so "compare Apple
and Microsoft" retrieves from both rather than blending them. LLM-based
decomposition is deferred to Phase 5.
"""

from __future__ import annotations

import re
from dataclasses import dataclass


@dataclass(frozen=True)
class SubQuery:
    """A query plus an optional company (ticker) filter."""

    text: str
    ticker: str | None


def aliases_from_companies(companies: list[tuple[str, str]]) -> dict[str, str]:
    """Map recognizable strings to tickers from (ticker, name) pairs.

    Each company contributes its ticker and the first significant word of its name
    (e.g. 'Apple Inc.' -> 'APPLE'), both uppercased.
    """
    aliases: dict[str, str] = {}
    for ticker, name in companies:
        aliases[ticker.upper()] = ticker
        words = name.split()
        if words and len(words[0]) >= 3:
            aliases[words[0].upper()] = ticker
    return aliases


def decompose(question: str, aliases: dict[str, str]) -> list[SubQuery]:
    """Split a question into sub-queries by the companies it names."""
    upper = question.upper()
    matched: list[str] = []
    for alias, ticker in aliases.items():
        if ticker in matched:
            continue
        if re.search(rf"\b{re.escape(alias)}\b", upper):
            matched.append(ticker)
    if len(matched) >= 2:
        return [SubQuery(text=question, ticker=ticker) for ticker in matched]
    return [SubQuery(text=question, ticker=matched[0] if matched else None)]
