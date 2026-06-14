"""Qdrant collection management and chunk upsert/search."""

from __future__ import annotations

import uuid
from collections.abc import Sequence
from dataclasses import dataclass

from qdrant_client import QdrantClient, models

# Fixed namespace so the same (accession, chunk_index) always maps to the same id.
_POINT_NAMESPACE = uuid.UUID("8b9d5b3e-3c2a-4e1f-9a6b-1c2d3e4f5061")


def point_id(accession: str, chunk_index: int) -> str:
    """Deterministic Qdrant point id for a chunk (idempotent upsert key)."""
    return str(uuid.uuid5(_POINT_NAMESPACE, f"{accession}:{chunk_index}"))


@dataclass(frozen=True)
class ChunkPoint:
    """A vector plus its metadata payload, ready to upsert."""

    accession: str
    chunk_index: int
    vector: list[float]
    payload: dict[str, object]


class QdrantIndex:
    """Manage a Qdrant collection of chunk vectors."""

    def __init__(self, client: QdrantClient, collection: str, dimension: int) -> None:
        self._client = client
        self._collection = collection
        self._dimension = dimension

    def ensure_collection(self) -> None:
        """Create the collection with the configured dimension if it is absent."""
        if self._client.collection_exists(self._collection):
            return
        self._client.create_collection(
            collection_name=self._collection,
            vectors_config=models.VectorParams(
                size=self._dimension, distance=models.Distance.COSINE
            ),
        )

    def upsert_chunks(self, points: Sequence[ChunkPoint]) -> None:
        if not points:
            return
        self._client.upsert(
            collection_name=self._collection,
            points=[
                models.PointStruct(
                    id=point_id(p.accession, p.chunk_index),
                    vector=p.vector,
                    payload=p.payload,
                )
                for p in points
            ],
        )

    def search(
        self,
        vector: list[float],
        *,
        limit: int = 5,
        query_filter: models.Filter | None = None,
    ) -> list[models.ScoredPoint]:
        response = self._client.query_points(
            collection_name=self._collection,
            query=vector,
            limit=limit,
            query_filter=query_filter,
            with_payload=True,
        )
        return response.points

    def count(self) -> int:
        return self._client.count(collection_name=self._collection).count
