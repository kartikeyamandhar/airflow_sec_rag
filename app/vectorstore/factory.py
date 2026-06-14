"""Build a QdrantIndex from settings."""

from __future__ import annotations

from qdrant_client import QdrantClient

from app.vectorstore.qdrant_index import QdrantIndex
from configs.settings import Settings


def build_qdrant_index(settings: Settings) -> QdrantIndex:
    """Connect to the configured Qdrant and return an index handle."""
    api_key = settings.qdrant_api_key.get_secret_value() or None
    client = QdrantClient(url=settings.qdrant_url, api_key=api_key)
    return QdrantIndex(client, settings.qdrant_collection, settings.embedding_dimension)
