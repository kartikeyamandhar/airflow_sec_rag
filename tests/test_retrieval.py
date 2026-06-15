"""Retrieval: filters, decomposition, reranking, and the end-to-end retriever."""

from qdrant_client import QdrantClient

from app.retrieval.decompose import aliases_from_companies, decompose
from app.retrieval.filters import build_filter
from app.retrieval.reranker import PassthroughReranker
from app.retrieval.retriever import Retriever
from app.vectorstore.qdrant_index import ChunkPoint, QdrantIndex
from tests.fakes import FakeEmbedder, FakeReranker, FakeSparseEmbedder

_AAPL_ACC = "0000320193-23-000106"
_MSFT_ACC = "0000789019-23-000001"
_DIM = 8


def test_build_filter_combines_constraints() -> None:
    assert build_filter() is None
    combined = build_filter(ticker="AAPL", form="10-K")
    assert combined is not None
    assert isinstance(combined.must, list)
    assert len(combined.must) == 2


def test_decompose_single_comparison_and_none() -> None:
    aliases = aliases_from_companies([("AAPL", "Apple Inc."), ("MSFT", "Microsoft Corporation")])
    one = decompose("What are Apple's risks?", aliases)
    assert len(one) == 1
    assert one[0].ticker == "AAPL"

    # The company name (and possessive) is stripped from the sub-query text, since
    # the ticker filter already pins the company; the text keys on the topic.
    assert "Apple" not in one[0].text

    two = decompose("Compare Apple and Microsoft revenue", aliases)
    assert {sub.ticker for sub in two} == {"AAPL", "MSFT"}
    # Both names are stripped so each sub-query is a clean topical query.
    for sub in two:
        assert "Apple" not in sub.text and "Microsoft" not in sub.text
        assert "revenue" in sub.text

    none = decompose("What is the outlook?", aliases)
    assert len(none) == 1
    assert none[0].ticker is None
    assert none[0].text == "What is the outlook?"


def test_passthrough_reranker_preserves_order() -> None:
    scores = PassthroughReranker().rerank("q", ["a", "b", "c"])
    assert scores[0] > scores[1] > scores[2]


def _seed_index() -> QdrantIndex:
    index = QdrantIndex(QdrantClient(location=":memory:"), "ret_test", dimension=_DIM)
    index.ensure_collection()
    embedder = FakeEmbedder(dimension=_DIM)
    sparse = FakeSparseEmbedder()

    def make_point(
        accession: str, chunk_index: int, ticker: str, section: str, text: str
    ) -> ChunkPoint:
        return ChunkPoint(
            accession=accession,
            chunk_index=chunk_index,
            dense=embedder.embed([text])[0],
            sparse=sparse.embed_sparse([text])[0],
            payload={
                "accession": accession,
                "cik": 320193 if ticker == "AAPL" else 789019,
                "ticker": ticker,
                "form": "10-K",
                "section": section,
                "chunk_index": chunk_index,
                "parent_index": 0,
                "char_start": 0,
                "char_end": len(text),
                "text": text,
            },
        )

    index.upsert_chunks(
        [
            make_point(
                _AAPL_ACC,
                1,
                "AAPL",
                "Item 1A",
                "Apple depends on supply chain partners outside the United States",
            ),
            make_point(_AAPL_ACC, 2, "AAPL", "Item 7", "Apple revenue grew"),
            make_point(_MSFT_ACC, 1, "MSFT", "Item 1A", "Microsoft faces cloud competition"),
        ]
    )
    return index


def _retriever(index: QdrantIndex, *, aliases: dict[str, str] | None = None) -> Retriever:
    return Retriever(
        embedder=FakeEmbedder(dimension=_DIM),
        sparse_embedder=FakeSparseEmbedder(),
        index=index,
        reranker=FakeReranker(),
        top_k=10,
        top_n=5,
        aliases=aliases or {},
    )


def test_retriever_filters_by_ticker() -> None:
    results = _retriever(_seed_index()).retrieve("supply chain risks", ticker="AAPL")
    assert results
    assert all(chunk.ticker == "AAPL" for chunk in results)


def test_retriever_returns_citation_fields() -> None:
    results = _retriever(_seed_index()).retrieve("supply chain", ticker="AAPL")
    top = results[0]
    assert top.accession
    assert top.section
    assert top.text
    assert top.char_end > top.char_start


def test_retriever_decomposes_comparison_question() -> None:
    aliases = aliases_from_companies([("AAPL", "Apple Inc."), ("MSFT", "Microsoft Corporation")])
    results = _retriever(_seed_index(), aliases=aliases).retrieve(
        "Compare Apple and Microsoft risk factors"
    )
    tickers = {chunk.ticker for chunk in results}
    assert "AAPL" in tickers
    assert "MSFT" in tickers
