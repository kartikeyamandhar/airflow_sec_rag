"""The answerer: retrieve, refuse-or-generate, enforce citations, verify, score."""

from __future__ import annotations

from collections.abc import Callable
from datetime import date
from typing import Protocol

from app.eval.confidence import combine_confidence
from app.eval.numeric import check_numeric_consistency
from app.eval.verifier import GroundingVerifier
from app.generation.confidence import compute_confidence
from app.generation.grounding import Claim, is_refusal, parse_answer
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
    """Compose retrieval and generation into a grounded, verified, cited answer."""

    def __init__(
        self,
        *,
        retriever: ChunkRetriever,
        llm: LLMClient,
        max_tokens: int = 1024,
        context_chunks: int = 8,
        min_coverage: float = 0.5,
        report_date_lookup: ReportDateLookup | None = None,
        verifier: GroundingVerifier | None = None,
        numeric_check: bool = False,
    ) -> None:
        self._retriever = retriever
        self._llm = llm
        self._max_tokens = max_tokens
        self._context_chunks = context_chunks
        self._min_coverage = min_coverage
        self._report_date_lookup = report_date_lookup
        self._verifier = verifier
        self._numeric_check = numeric_check

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

        parsed = parse_answer(raw, len(chunks))
        coverage = len(parsed.claims) / parsed.total_sentences if parsed.total_sentences else 0.0
        if not parsed.claims or coverage < self._min_coverage:
            return self._refuse(
                question, "The answer could not be grounded in the filings.", coverage
            )

        claims = parsed.claims
        faithfulness: float | None = None
        if self._verifier is not None:
            claims, faithfulness = self._verify(parsed.claims, chunks)
            if not claims:
                return self._refuse(
                    question,
                    "No claim was entailed by its cited filing.",
                    coverage,
                )

        numeric_ok: bool | None = None
        unverified: list[str] = []
        if self._numeric_check:
            numeric_ok, unverified = self._check_numbers(claims, chunks)

        markers = sorted({m for claim in claims for m in claim.markers})
        citations = self._build_citations(markers, chunks)
        text = " ".join(claim.text for claim in claims)
        if faithfulness is not None:
            confidence = combine_confidence(
                coverage, faithfulness, numeric_ok if numeric_ok is not None else True
            )
        else:
            confidence = compute_confidence(coverage, len(citations))

        return Answer(
            question=question,
            text=text,
            citations=citations,
            confidence=confidence,
            refused=False,
            coverage=round(coverage, 3),
            num_context=len(chunks),
            faithfulness=faithfulness,
            numeric_ok=numeric_ok,
            unverified_numbers=unverified,
        )

    def _verify(
        self, claims: list[Claim], chunks: list[RetrievedChunk]
    ) -> tuple[list[Claim], float]:
        assert self._verifier is not None
        evidences = [self._evidence_for(claim, chunks) for claim in claims]
        result = self._verifier.verify(claims, evidences)
        supported = {v.index for v in result.verdicts if v.supported}
        kept = [claim for i, claim in enumerate(claims, start=1) if i in supported]
        return kept, result.faithfulness

    def _check_numbers(
        self, claims: list[Claim], chunks: list[RetrievedChunk]
    ) -> tuple[bool, list[str]]:
        answer_text = " ".join(claim.text for claim in claims)
        evidence = " ".join(self._evidence_for(claim, chunks) for claim in claims)
        result = check_numeric_consistency(answer_text, evidence)
        return result.consistent, result.unverified

    def _evidence_for(self, claim: Claim, chunks: list[RetrievedChunk]) -> str:
        return " ".join(chunks[marker - 1].text for marker in claim.markers)

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

    def _refuse(self, question: str, reason: str, coverage: float = 0.0) -> Answer:
        return Answer(
            question=question,
            text="",
            confidence=0.0,
            refused=True,
            reason=reason,
            coverage=round(coverage, 3),
        )
