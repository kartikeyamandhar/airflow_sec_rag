"""QdrantIndex against an in-memory Qdrant (no server, no Docker)."""

from qdrant_client import QdrantClient

from app.vectorstore.qdrant_index import ChunkPoint, QdrantIndex, point_id

_ACCESSION = "0000320193-23-000106"


def _index(dimension: int = 4) -> QdrantIndex:
    client = QdrantClient(location=":memory:")
    index = QdrantIndex(client, "test_collection", dimension=dimension)
    index.ensure_collection()
    return index


def test_point_id_is_stable_and_unique() -> None:
    assert point_id(_ACCESSION, 1) == point_id(_ACCESSION, 1)
    assert point_id(_ACCESSION, 1) != point_id(_ACCESSION, 2)


def test_upsert_count_and_search() -> None:
    index = _index()
    index.upsert_chunks(
        [
            ChunkPoint(
                accession=_ACCESSION,
                chunk_index=1,
                vector=[1.0, 0.0, 0.0, 0.0],
                payload={"section": "Item 1", "text": "alpha"},
            ),
            ChunkPoint(
                accession=_ACCESSION,
                chunk_index=2,
                vector=[0.0, 1.0, 0.0, 0.0],
                payload={"section": "Item 1A", "text": "beta"},
            ),
        ]
    )
    assert index.count() == 2
    results = index.search([1.0, 0.0, 0.0, 0.0], limit=1)
    assert len(results) == 1
    payload = results[0].payload
    assert payload is not None
    assert payload["section"] == "Item 1"


def test_upsert_is_idempotent() -> None:
    index = _index()
    point = ChunkPoint(
        accession=_ACCESSION,
        chunk_index=1,
        vector=[1.0, 0.0, 0.0, 0.0],
        payload={"text": "alpha"},
    )
    index.upsert_chunks([point])
    index.upsert_chunks([point])
    assert index.count() == 1


def test_upsert_empty_is_noop() -> None:
    index = _index()
    index.upsert_chunks([])
    assert index.count() == 0
