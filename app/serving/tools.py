"""Serialize domain objects to JSON-ready dicts (shared by HTTP and MCP)."""

from __future__ import annotations

from app.generation.models import Answer, Citation
from app.retrieval.retriever import RetrievedChunk

_EXCERPT = 280


def citation_to_dict(citation: Citation) -> dict[str, object]:
    return {
        "marker": citation.marker,
        "accession": citation.accession,
        "ticker": citation.ticker,
        "form": citation.form,
        "section": citation.section,
        "char_start": citation.char_start,
        "char_end": citation.char_end,
        "as_of": citation.as_of.isoformat() if citation.as_of else None,
        "excerpt": citation.text[:_EXCERPT],
    }


def answer_to_dict(answer: Answer) -> dict[str, object]:
    return {
        "question": answer.question,
        "refused": answer.refused,
        "reason": answer.reason,
        "text": answer.text,
        "confidence": answer.confidence,
        "coverage": answer.coverage,
        "faithfulness": answer.faithfulness,
        "numeric_ok": answer.numeric_ok,
        "unverified_numbers": answer.unverified_numbers,
        "citations": [citation_to_dict(c) for c in answer.citations],
    }


def chunk_to_dict(chunk: RetrievedChunk) -> dict[str, object]:
    return {
        "accession": chunk.accession,
        "ticker": chunk.ticker,
        "form": chunk.form,
        "section": chunk.section,
        "score": chunk.score,
        "char_start": chunk.char_start,
        "char_end": chunk.char_end,
        "excerpt": chunk.text[:_EXCERPT],
    }


def chunks_to_dicts(chunks: list[RetrievedChunk]) -> list[dict[str, object]]:
    return [chunk_to_dict(chunk) for chunk in chunks]
