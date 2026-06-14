"""Build the configured embedding backend from settings."""

from __future__ import annotations

from app.embedding.base import Embedder
from app.embedding.fastembed_embedder import FastembedEmbedder
from app.embedding.runpod_embedder import RunPodEmbedder
from app.embedding.sparse import FastembedSparseEmbedder, SparseEmbedder
from configs.settings import Settings


def build_embedder(settings: Settings) -> Embedder:
    """Return the ``Embedder`` selected by ``settings.embedding_backend``."""
    if settings.embedding_backend == "runpod":
        return RunPodEmbedder(
            endpoint_id=settings.runpod_endpoint_id,
            api_key=settings.runpod_api_key.get_secret_value(),
            dimension=settings.embedding_dimension,
        )
    return FastembedEmbedder(
        model_name=settings.embedding_model,
        dimension=settings.embedding_dimension,
    )


def build_sparse_embedder(settings: Settings) -> SparseEmbedder:
    """Return the configured sparse (BM25) embedder."""
    return FastembedSparseEmbedder(model_name=settings.sparse_embedding_model)
