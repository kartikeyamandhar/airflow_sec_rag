"""QdrantIndex hybrid (dense + BM25 sparse) against an in-memory Qdrant."""

from qdrant_client import QdrantClient

from app.embedding.sparse import SparseVector
from app.retrieval.filters import build_filter
from app.vectorstore.qdrant_index import ChunkPoint, QdrantIndex, point_id

_ACCESSION = "0000320193-23-000106"


def _index(dimension: int = 4) -> QdrantIndex:
    client = QdrantClient(location=":memory:")
    index = QdrantIndex(client, "test_collection", dimension=dimension)
    index.ensure_collection()
    return index


def _point(
    chunk_index: int,
    dense: list[float],
    sparse_indices: list[int],
    payload: dict[str, object],
) -> ChunkPoint:
    return ChunkPoint(
        accession=_ACCESSION,
        chunk_index=chunk_index,
        dense=dense,
        sparse=SparseVector(indices=sparse_indices, values=[1.0] * len(sparse_indices)),
        payload=payload,
    )


def test_point_id_is_stable_and_unique() -> None:
    assert point_id(_ACCESSION, 1) == point_id(_ACCESSION, 1)
    assert point_id(_ACCESSION, 1) != point_id(_ACCESSION, 2)


def test_hybrid_upsert_count_and_search() -> None:
    index = _index()
    index.upsert_chunks(
        [
            _point(1, [1.0, 0.0, 0.0, 0.0], [3, 7], {"section": "Item 1"}),
            _point(2, [0.0, 1.0, 0.0, 0.0], [7, 11], {"section": "Item 1A"}),
        ]
    )
    assert index.count() == 2
    results = index.hybrid_search(
        [1.0, 0.0, 0.0, 0.0], SparseVector(indices=[3, 7], values=[1.0, 1.0]), limit=2
    )
    assert results
    payload = results[0].payload
    assert payload is not None
    assert payload["section"] in {"Item 1", "Item 1A"}


def test_hybrid_filter_narrows_results() -> None:
    index = _index()
    index.upsert_chunks(
        [
            _point(1, [1.0, 0.0, 0.0, 0.0], [3], {"ticker": "AAPL"}),
            _point(2, [1.0, 0.0, 0.0, 0.0], [3], {"ticker": "MSFT"}),
        ]
    )
    results = index.hybrid_search(
        [1.0, 0.0, 0.0, 0.0],
        SparseVector(indices=[3], values=[1.0]),
        limit=10,
        query_filter=build_filter(ticker="AAPL"),
    )
    assert len(results) == 1
    assert (results[0].payload or {})["ticker"] == "AAPL"


def test_upsert_is_idempotent() -> None:
    index = _index()
    point = _point(1, [1.0, 0.0, 0.0, 0.0], [3], {"text": "alpha"})
    index.upsert_chunks([point])
    index.upsert_chunks([point])
    assert index.count() == 1


def test_upsert_empty_is_noop() -> None:
    index = _index()
    index.upsert_chunks([])
    assert index.count() == 0
