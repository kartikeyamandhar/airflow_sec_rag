"""Decompose comparison questions into per-company sub-queries.

A heuristic for Phase 4: if a question names two or more known companies, split it
into one sub-query per company (each filtered to that company), so "compare Apple
and Microsoft" retrieves from both rather than blending them. LLM-based
decomposition is deferred to Phase 5.
"""

from __future__ import annotations

import re
from dataclasses import dataclass

# Optional possessive: an apostrophe followed by "s" (e.g. Apple's). A curly
# apostrophe is normalized to a straight one before matching.
_POSSESSIVE = "(?:'s)?"


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


def _strip_aliases(text: str, aliases: dict[str, str]) -> str:
    """Remove company names (and possessives) from a query, collapsing whitespace.

    Once a sub-query is filtered to one company by ticker, the company names in the
    text only dilute the embedding. A comparison phrased "How do Microsoft and
    Alphabet describe AI risk?" retrieves Alphabet's risk factors far better as the
    topical "How do describe AI risk?" with a GOOGL filter than with both names left
    in, which pull in generic business-overview passages instead.
    """
    cleaned = text.replace(chr(0x2019), "'")
    for alias in aliases:
        cleaned = re.sub(rf"\b{re.escape(alias)}{_POSSESSIVE}\b", " ", cleaned, flags=re.IGNORECASE)
    return re.sub(r"\s+", " ", cleaned).strip()


def decompose(question: str, aliases: dict[str, str]) -> list[SubQuery]:
    """Split a question into sub-queries by the companies it names.

    Each matched company yields a sub-query filtered to its ticker, with company
    names stripped from the text so retrieval keys on the topic, not the names.
    """
    upper = question.upper()
    matched: list[str] = []
    for alias, ticker in aliases.items():
        if ticker in matched:
            continue
        if re.search(rf"\b{re.escape(alias)}\b", upper):
            matched.append(ticker)
    if not matched:
        return [SubQuery(text=question, ticker=None)]
    topic = _strip_aliases(question, aliases) or question
    return [SubQuery(text=topic, ticker=ticker) for ticker in matched]
