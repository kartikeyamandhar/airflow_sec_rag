"""Chunk sections into parent and child units with citation offsets.

Each section becomes one ``parent`` chunk (the whole section text) plus a series of
overlapping ``child`` chunks. A child's ``char_start``/``char_end`` index into its
parent's text, so the exact citable span is ``parent.text[char_start:char_end]``.
Token counts are estimated by word count (model-exact tokenization is deferred to
the embedding phase; see ADR 0005).
"""

from __future__ import annotations

import re
from collections.abc import Sequence

from app.domain.models import Chunk
from app.narrative.extract import Section

_WORD = re.compile(r"\S+")


def estimate_tokens(text: str) -> int:
    """Approximate token count by counting whitespace-delimited words."""
    return len(_WORD.findall(text))


def chunk_sections(
    sections: Sequence[Section],
    *,
    accession: str,
    cik: int,
    ticker: str | None,
    form: str,
    child_tokens: int = 350,
    overlap_tokens: int = 50,
) -> list[Chunk]:
    """Produce parent and child chunks for a filing's sections."""
    chunks: list[Chunk] = []
    index = 0
    for section in sections:
        parent_index = index
        chunks.append(
            Chunk(
                accession=accession,
                cik=cik,
                ticker=ticker,
                form=form,
                section=section.name,
                kind="parent",
                chunk_index=index,
                parent_index=None,
                text=section.text,
                char_start=0,
                char_end=len(section.text),
                token_estimate=estimate_tokens(section.text),
            )
        )
        index += 1
        for start, end in _window_offsets(section.text, child_tokens, overlap_tokens):
            child_text = section.text[start:end]
            chunks.append(
                Chunk(
                    accession=accession,
                    cik=cik,
                    ticker=ticker,
                    form=form,
                    section=section.name,
                    kind="child",
                    chunk_index=index,
                    parent_index=parent_index,
                    text=child_text,
                    char_start=start,
                    char_end=end,
                    token_estimate=estimate_tokens(child_text),
                )
            )
            index += 1
    return chunks


def _window_offsets(text: str, child_tokens: int, overlap_tokens: int) -> list[tuple[int, int]]:
    words = list(_WORD.finditer(text))
    if not words:
        return []
    step = max(1, child_tokens - overlap_tokens)
    spans: list[tuple[int, int]] = []
    start_word = 0
    total = len(words)
    while start_word < total:
        window = words[start_word : start_word + child_tokens]
        spans.append((window[0].start(), window[-1].end()))
        if start_word + child_tokens >= total:
            break
        start_word += step
    return spans
