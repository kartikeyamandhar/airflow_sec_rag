"""Generation: prompt, citation enforcement, confidence, and the answerer."""

from datetime import date

from app.generation.answerer import Answerer, ReportDateLookup
from app.generation.confidence import compute_confidence
from app.generation.grounding import enforce_citations, is_refusal
from app.generation.prompt import REFUSAL_SENTINEL, build_answer_prompt
from app.retrieval.retriever import RetrievedChunk
from tests.fakes import FakeLLMClient, FakeRetriever

_ACCESSION = "0000320193-23-000106"


def _chunk(text: str, *, section: str = "Item 1A") -> RetrievedChunk:
    return RetrievedChunk(
        accession=_ACCESSION,
        cik=320193,
        ticker="AAPL",
        form="10-K",
        section=section,
        chunk_index=1,
        parent_index=0,
        char_start=10,
        char_end=10 + len(text),
        text=text,
        score=1.0,
    )


def test_build_answer_prompt_numbers_chunks() -> None:
    chunks = [_chunk("Apple relies on overseas suppliers."), _chunk("Tariffs are a risk.")]
    system, user = build_answer_prompt("supply chain?", chunks)
    assert REFUSAL_SENTINEL in system
    assert "[1]" in user
    assert "[2]" in user
    assert "supply chain?" in user
    assert "Apple relies on overseas suppliers." in user


def test_is_refusal() -> None:
    assert is_refusal("NO_ANSWER")
    assert is_refusal("  no_answer  ")
    assert is_refusal("")
    assert not is_refusal("Apple is large [1].")


def test_enforce_citations_keeps_cited_drops_uncited() -> None:
    raw = "Apple relies on suppliers [1]. An aside without a cite. Tariffs are a risk [2]."
    parse = enforce_citations(raw, num_chunks=2)
    assert parse.markers == [1, 2]
    assert "aside" not in parse.text
    assert abs(parse.coverage - 2 / 3) < 1e-9


def test_enforce_citations_ignores_out_of_range_marker() -> None:
    parse = enforce_citations("Claim with a bad marker [9].", num_chunks=2)
    assert parse.markers == []
    assert parse.coverage == 0.0


def test_compute_confidence() -> None:
    assert compute_confidence(1.0, 2) == 1.0
    assert compute_confidence(0.5, 1) == 0.5
    assert compute_confidence(0.9, 0) == 0.0


def _answerer(
    response: str,
    chunks: list[RetrievedChunk],
    *,
    report_date_lookup: ReportDateLookup | None = None,
    min_coverage: float = 0.5,
) -> Answerer:
    return Answerer(
        retriever=FakeRetriever(chunks),
        llm=FakeLLMClient(response),
        report_date_lookup=report_date_lookup,
        min_coverage=min_coverage,
    )


def test_answerer_returns_grounded_cited_answer() -> None:
    chunks = [_chunk("Apple depends on overseas manufacturing partners.")]
    answerer = _answerer(
        "Apple depends on overseas manufacturing partners [1].",
        chunks,
        report_date_lookup=lambda accessions: {a: date(2023, 9, 30) for a in accessions},
    )
    answer = answerer.answer("supply chain risk?", ticker="AAPL")
    assert not answer.refused
    assert answer.confidence == 1.0
    assert len(answer.citations) == 1
    citation = answer.citations[0]
    assert citation.accession == _ACCESSION
    assert citation.section == "Item 1A"
    assert citation.as_of == date(2023, 9, 30)


def test_answerer_refuses_when_no_chunks() -> None:
    answer = _answerer("anything [1].", []).answer("q")
    assert answer.refused
    assert answer.confidence == 0.0


def test_answerer_refuses_on_sentinel() -> None:
    answer = _answerer("NO_ANSWER", [_chunk("A passage.")]).answer("q")
    assert answer.refused


def test_answerer_refuses_on_low_coverage() -> None:
    answer = _answerer(
        "Apple is great. It is large. Nothing is cited here.", [_chunk("A passage.")]
    ).answer("q")
    assert answer.refused
