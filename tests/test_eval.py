"""Evaluation: verifier, numeric check, confidence, gate, golden, verify path."""

from app.eval.confidence import combine_confidence
from app.eval.gate import Thresholds, evaluate_gate
from app.eval.golden import EvalReport, GoldenItem, run_golden
from app.eval.numeric import check_numeric_consistency, extract_financial_numbers
from app.eval.verifier import GroundingVerifier
from app.generation.answerer import Answerer
from app.generation.grounding import Claim
from app.generation.models import Answer, Citation
from app.retrieval.retriever import RetrievedChunk
from tests.fakes import FakeAnswerable, FakeLLMClient, FakeRetriever

_ACCESSION = "0000320193-23-000106"


def _chunk(text: str) -> RetrievedChunk:
    return RetrievedChunk(
        accession=_ACCESSION,
        cik=320193,
        ticker="AAPL",
        form="10-K",
        section="Item 1A",
        chunk_index=1,
        parent_index=0,
        char_start=0,
        char_end=len(text),
        text=text,
        score=1.0,
    )


def test_verifier_parses_verdicts_and_faithfulness() -> None:
    verifier = GroundingVerifier(FakeLLMClient("1: SUPPORTED\n2: UNSUPPORTED"))
    claims = [Claim("a [1].", [1]), Claim("b [2].", [2])]
    result = verifier.verify(claims, ["evidence a", "evidence b"])
    assert result.faithfulness == 0.5
    assert {v.index: v.supported for v in result.verdicts} == {1: True, 2: False}


def test_verifier_missing_verdict_is_unsupported() -> None:
    verifier = GroundingVerifier(FakeLLMClient("1: SUPPORTED"))
    result = verifier.verify([Claim("a [1].", [1]), Claim("b [2].", [2])], ["e1", "e2"])
    assert result.faithfulness == 0.5
    assert result.verdicts[1].supported is False


def test_extract_and_check_numeric() -> None:
    numbers = extract_financial_numbers("Revenue was $383,285 million, up 5.2%.")
    assert "383285" in numbers
    assert "5.2" in numbers

    ok = check_numeric_consistency("Revenue was $383,285 [1].", "revenue of $383,285")
    assert ok.consistent

    bad = check_numeric_consistency("Revenue was $999,999 [1].", "revenue of $383,285")
    assert not bad.consistent
    assert "999999" in bad.unverified


def test_combine_confidence() -> None:
    assert combine_confidence(1.0, 1.0, True) == 1.0
    assert combine_confidence(1.0, 0.5, True) == 0.5
    assert combine_confidence(1.0, 1.0, False) == 0.5


def test_gate_passes_and_fails() -> None:
    thresholds = Thresholds(
        min_refusal_accuracy=1.0, min_citation_hit_rate=0.8, min_faithfulness=0.8
    )
    strong = EvalReport(3, 1.0, 0.9, 0.9, 0.9)
    assert evaluate_gate(strong, thresholds).passed

    weak = EvalReport(3, 0.5, 0.9, 0.9, 0.5)
    result = evaluate_gate(weak, thresholds)
    assert not result.passed
    assert len(result.failures) == 2


def test_run_golden_scores_items() -> None:
    citation = Citation(
        marker=1,
        accession=_ACCESSION,
        ticker="AAPL",
        form="10-K",
        section="Item 1A",
        char_start=0,
        char_end=5,
        text="t",
        as_of=None,
    )
    answers = {
        "good": Answer(
            question="good",
            text="Apple depends on outsourcing partners.",
            citations=[citation],
            refused=False,
            faithfulness=1.0,
        ),
        "offtopic": Answer(question="offtopic", text="", refused=True),
    }
    items = [
        GoldenItem(question="good", should_refuse=False, expect_substrings=["outsourcing"]),
        GoldenItem(question="offtopic", should_refuse=True),
    ]
    report = run_golden(FakeAnswerable(answers), items)
    assert report.refusal_accuracy == 1.0
    assert report.substring_hit_rate == 1.0
    assert report.mean_faithfulness == 1.0


def test_answerer_verify_drops_unsupported_claim() -> None:
    chunks = [_chunk("Apple depends on overseas partners."), _chunk("Tariffs raise costs.")]
    model = FakeLLMClient("Apple depends on overseas partners [1]. Tariffs raise costs [2].")
    judge = FakeLLMClient("1: SUPPORTED\n2: UNSUPPORTED")
    answerer = Answerer(
        retriever=FakeRetriever(chunks),
        llm=model,
        verifier=GroundingVerifier(judge),
        numeric_check=True,
        min_coverage=0.5,
    )
    answer = answerer.answer("supply chain?")
    assert not answer.refused
    assert answer.faithfulness == 0.5
    assert "overseas partners" in answer.text
    assert "Tariffs" not in answer.text
    assert len(answer.citations) == 1
    assert answer.confidence == 0.5
