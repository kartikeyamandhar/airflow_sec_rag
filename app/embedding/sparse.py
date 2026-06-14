"""Sparse (lexical) embedding for the BM25 half of hybrid search.

A sparse vector is a list of (term-id, weight) pairs. Combined with the dense
vector and Qdrant's IDF modifier, it gives BM25-style lexical matching that dense
vectors miss (exact tickers, rare terms).
"""

from __future__ import annotations

from dataclasses import dataclass
from typing import Protocol

from fastembed import SparseTextEmbedding


@dataclass(frozen=True)
class SparseVector:
    """A sparse vector as parallel index/value lists."""

    indices: list[int]
    values: list[float]


class SparseEmbedder(Protocol):
    """Turns texts into sparse (lexical) vectors."""

    def embed_sparse(self, texts: list[str]) -> list[SparseVector]:
        """Return one sparse vector per input text, in order."""
        ...


class FastembedSparseEmbedder:
    """A ``SparseEmbedder`` backed by a fastembed BM25 model."""

    def __init__(self, model_name: str) -> None:
        self._model = SparseTextEmbedding(model_name=model_name)

    def embed_sparse(self, texts: list[str]) -> list[SparseVector]:
        return [
            SparseVector(
                indices=[int(i) for i in sparse.indices],
                values=[float(v) for v in sparse.values],
            )
            for sparse in self._model.embed(texts)
        ]
