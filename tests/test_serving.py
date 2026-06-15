"""Serving: FastAPI routes via TestClient with fakes, serializers, MCP import."""

from datetime import date

from fastapi.testclient import TestClient

from app.generation.models import Answer, Citation
from app.retrieval.retriever import RetrievedChunk
from app.serving.api import create_app
from app.serving.deps import get_answerer, get_retriever
from app.serving.tools import answer_to_dict, chunks_to_dicts
from tests.fakes import FakeAnswerable, FakeRetriever

_ACCESSION = "0000320193-23-000106"


def _citation() -> Citation:
    return Citation(
        marker=1,
        accession=_ACCESSION,
        ticker="AAPL",
        form="10-K",
        section="Item 1A",
        char_start=0,
        char_end=5,
        text="evidence",
        as_of=date(2023, 9, 30),
    )


def _chunk() -> RetrievedChunk:
    return RetrievedChunk(
        accession=_ACCESSION,
        cik=320193,
        ticker="AAPL",
        form="10-K",
        section="Item 1A",
        chunk_index=1,
        parent_index=0,
        char_start=0,
        char_end=4,
        text="text",
        score=0.9,
    )


def _client(
    answerer: FakeAnswerable | None = None, retriever: FakeRetriever | None = None
) -> TestClient:
    app = create_app()
    if answerer is not None:
        app.dependency_overrides[get_answerer] = lambda: answerer
    if retriever is not None:
        app.dependency_overrides[get_retriever] = lambda: retriever
    return TestClient(app)


def test_health() -> None:
    assert _client().get("/health").json() == {"status": "ok"}


def test_index_returns_html() -> None:
    response = _client().get("/")
    assert response.status_code == 200
    assert "text/html" in response.headers["content-type"]
    assert "SEC RAG" in response.text


def test_answer_grounded() -> None:
    answer = Answer(
        question="risks?",
        text="Apple depends on partners [1].",
        citations=[_citation()],
        confidence=1.0,
        refused=False,
        faithfulness=1.0,
    )
    client = _client(answerer=FakeAnswerable({"risks?": answer}))
    body = client.post("/answer", json={"question": "risks?", "ticker": "AAPL"}).json()
    assert body["refused"] is False
    assert body["text"].startswith("Apple depends")
    assert body["confidence"] == 1.0
    assert body["citations"][0]["as_of"] == "2023-09-30"


def test_answer_refused() -> None:
    answer = Answer(question="weather?", text="", refused=True, reason="off topic")
    client = _client(answerer=FakeAnswerable({"weather?": answer}))
    body = client.post("/answer", json={"question": "weather?"}).json()
    assert body["refused"] is True
    assert body["citations"] == []


def test_answer_missing_question_is_422() -> None:
    assert _client(answerer=FakeAnswerable({})).post("/answer", json={}).status_code == 422


def test_search() -> None:
    client = _client(retriever=FakeRetriever([_chunk()]))
    body = client.post("/search", json={"question": "q", "ticker": "AAPL"}).json()
    assert len(body["results"]) == 1
    assert body["results"][0]["ticker"] == "AAPL"


def test_metrics_endpoint() -> None:
    answer = Answer(question="risks?", text="X [1].", citations=[_citation()], refused=False)
    client = _client(answerer=FakeAnswerable({"risks?": answer}))
    client.post("/answer", json={"question": "risks?"})
    metrics = client.get("/metrics")
    assert metrics.status_code == 200
    assert "sec_rag_answers_total" in metrics.text


def test_serializers() -> None:
    answer = Answer(
        question="q", text="t [1].", citations=[_citation()], confidence=0.8, refused=False
    )
    payload = answer_to_dict(answer)
    assert payload["refused"] is False
    assert payload["citations"][0]["as_of"] == "2023-09-30"  # type: ignore[index]
    chunks = chunks_to_dicts([_chunk()])
    assert chunks[0]["accession"] == _ACCESSION


def test_mcp_server_imports() -> None:
    from app.serving import mcp_server

    assert mcp_server.mcp is not None
