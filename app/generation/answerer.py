"""The answerer: retrieve, refuse-or-generate, enforce citations, score."""

from __future__ import annotations

from collections.abc import Callable
from datetime import date
from typing import Protocol

from app.generation.confidence import compute_confidence
from app.generation.grounding import enforce_citations, is_refusal
from app.generation.llm import LLMClient
from app.generation.models import Answer, Citation
from app.generation.prompt import build_answer_prompt
from app.retrieval.retriever import RetrievedChunk

# Maps a set of accessions to their period-of-report dates (for "as of").
ReportDateLookup = Callable[[set[str]], dict[str, date]]


class ChunkRetriever(Protocol):
    """The retrieval surface the answerer depends on (satisfied by Retriever)."""

    def retrieve(
        self,
        question: str,
        *,
        ticker: str | None = None,
        form: str | None = None,
        section: str | None = None,
        limit: int | None = None,
    ) -> list[RetrievedChunk]: ...


class Answerer:
    """Compose retrieval and generation into a grounded, cited answer."""

    def __init__(
        self,
        *,
        retriever: ChunkRetriever,
        llm: LLMClient,
        max_tokens: int = 1024,
        context_chunks: int = 8,
        min_coverage: float = 0.5,
        report_date_lookup: ReportDateLookup | None = None,
    ) -> None:
        self._retriever = retriever
        self._llm = llm
        self._max_tokens = max_tokens
        self._context_chunks = context_chunks
        self._min_coverage = min_coverage
        self._report_date_lookup = report_date_lookup

    def answer(
        self,
        question: str,
        *,
        ticker: str | None = None,
        form: str | None = None,
        section: str | None = None,
    ) -> Answer:
        chunks = self._retriever.retrieve(
            question,
            ticker=ticker,
            form=form,
            section=section,
            limit=self._context_chunks,
        )
        if not chunks:
            return self._refuse(question, "No supporting passages were found.")

        system, user = build_answer_prompt(question, chunks)
        raw = self._llm.complete(system=system, user=user, max_tokens=self._max_tokens)
        if is_refusal(raw):
            return self._refuse(
                question, "The filings do not contain enough information to answer."
            )

        parse = enforce_citations(raw, len(chunks))
        if not parse.markers or parse.coverage < self._min_coverage:
            return self._refuse(question, "The answer could not be grounded in the filings.")

        citations = self._build_citations(parse.markers, chunks)
        confidence = compute_confidence(parse.coverage, len(citations))
        return Answer(
            question=question,
            text=parse.text,
            citations=citations,
            confidence=confidence,
            refused=False,
        )

    def _build_citations(self, markers: list[int], chunks: list[RetrievedChunk]) -> list[Citation]:
        accessions = {chunks[marker - 1].accession for marker in markers}
        dates = self._report_date_lookup(accessions) if self._report_date_lookup else {}
        citations: list[Citation] = []
        for marker in markers:
            chunk = chunks[marker - 1]
            citations.append(
                Citation(
                    marker=marker,
                    accession=chunk.accession,
                    ticker=chunk.ticker,
                    form=chunk.form,
                    section=chunk.section,
                    char_start=chunk.char_start,
                    char_end=chunk.char_end,
                    text=chunk.text,
                    as_of=dates.get(chunk.accession),
                )
            )
        return citations

    def _refuse(self, question: str, reason: str) -> Answer:
        return Answer(question=question, text="", confidence=0.0, refused=True, reason=reason)
