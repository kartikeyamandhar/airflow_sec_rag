"""Cross-encoder reranking for retrieval precision.

A `Reranker` scores each candidate against the query so the most on-topic chunks
rise to the top. The fastembed cross-encoder runs on CPU; a passthrough is used
when reranking is disabled; a fake is used in the test gate.
"""

from __future__ import annotations

from typing import Protocol


class Reranker(Protocol):
    """Scores candidate documents against a query (higher is more relevant)."""

    def rerank(self, query: str, documents: list[str]) -> list[float]:
        """Return one score per document, in the input order."""
        ...


class PassthroughReranker:
    """A no-op reranker that preserves the input order (descending scores)."""

    def rerank(self, query: str, documents: list[str]) -> list[float]:
        return [float(len(documents) - i) for i in range(len(documents))]


class FastembedReranker:
    """A `Reranker` backed by a fastembed cross-encoder (CPU)."""

    def __init__(self, model_name: str) -> None:
        from fastembed.rerank.cross_encoder import TextCrossEncoder

        self._model = TextCrossEncoder(model_name=model_name)

    def rerank(self, query: str, documents: list[str]) -> list[float]:
        if not documents:
            return []
        return [float(score) for score in self._model.rerank(query, documents)]


def build_reranker(backend: str, model_name: str) -> Reranker:
    """Return the configured reranker, or a passthrough when disabled."""
    if backend == "none":
        return PassthroughReranker()
    return FastembedReranker(model_name=model_name)
