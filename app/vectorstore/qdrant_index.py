"""Qdrant collection management, hybrid upsert, and hybrid search.

The collection holds two vectors per chunk: a dense bge vector (semantic) and a
sparse BM25 vector (lexical, with an IDF modifier so term weighting is correct
across the collection). Search fuses both with Reciprocal Rank Fusion server-side.
Point ids are uuid5 of (accession, chunk_index), so re-indexing overwrites.
"""

from __future__ import annotations

import uuid
from collections.abc import Sequence
from dataclasses import dataclass

from qdrant_client import QdrantClient, models

from app.embedding.sparse import SparseVector

_POINT_NAMESPACE = uuid.UUID("8b9d5b3e-3c2a-4e1f-9a6b-1c2d3e4f5061")
_DENSE = "dense"
_SPARSE = "bm25"


def point_id(accession: str, chunk_index: int) -> str:
    """Deterministic Qdrant point id for a chunk (idempotent upsert key)."""
    return str(uuid.uuid5(_POINT_NAMESPACE, f"{accession}:{chunk_index}"))


@dataclass(frozen=True)
class ChunkPoint:
    """A dense + sparse vector pair plus metadata, ready to upsert."""

    accession: str
    chunk_index: int
    dense: list[float]
    sparse: SparseVector
    payload: dict[str, object]


class QdrantIndex:
    """Manage a hybrid Qdrant collection of chunk vectors."""

    def __init__(self, client: QdrantClient, collection: str, dimension: int) -> None:
        self._client = client
        self._collection = collection
        self._dimension = dimension

    def ensure_collection(self) -> None:
        """Create the hybrid collection if it is absent."""
        if self._client.collection_exists(self._collection):
            return
        self._client.create_collection(
            collection_name=self._collection,
            vectors_config={
                _DENSE: models.VectorParams(size=self._dimension, distance=models.Distance.COSINE)
            },
            sparse_vectors_config={
                _SPARSE: models.SparseVectorParams(modifier=models.Modifier.IDF)
            },
        )

    def upsert_chunks(self, points: Sequence[ChunkPoint]) -> None:
        if not points:
            return
        self._client.upsert(
            collection_name=self._collection,
            points=[
                models.PointStruct(
                    id=point_id(p.accession, p.chunk_index),
                    vector={
                        _DENSE: p.dense,
                        _SPARSE: models.SparseVector(
                            indices=p.sparse.indices, values=p.sparse.values
                        ),
                    },
                    payload=p.payload,
                )
                for p in points
            ],
        )

    def hybrid_search(
        self,
        dense: list[float],
        sparse: SparseVector,
        *,
        limit: int = 10,
        query_filter: models.Filter | None = None,
    ) -> list[models.ScoredPoint]:
        """Dense + sparse search fused by Reciprocal Rank Fusion."""
        response = self._client.query_points(
            collection_name=self._collection,
            prefetch=[
                models.Prefetch(query=dense, using=_DENSE, limit=limit, filter=query_filter),
                models.Prefetch(
                    query=models.SparseVector(indices=sparse.indices, values=sparse.values),
                    using=_SPARSE,
                    limit=limit,
                    filter=query_filter,
                ),
            ],
            query=models.FusionQuery(fusion=models.Fusion.RRF),
            limit=limit,
            with_payload=True,
        )
        return response.points

    def count(self) -> int:
        return self._client.count(collection_name=self._collection).count
