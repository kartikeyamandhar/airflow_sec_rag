"""Opt-in test of the real fastembed backend (downloads a model).

Skipped unless ``RUN_FASTEMBED=1`` so the default gate needs no model download.
Run with: ``RUN_FASTEMBED=1 uv run pytest tests/test_fastembed.py``
"""

import os

import pytest

from app.embedding.fastembed_embedder import FastembedEmbedder

pytestmark = pytest.mark.skipif(
    os.environ.get("RUN_FASTEMBED") != "1",
    reason="set RUN_FASTEMBED=1 to run the real fastembed embedder",
)


def test_fastembed_produces_real_vectors() -> None:
    embedder = FastembedEmbedder(model_name="BAAI/bge-small-en-v1.5", dimension=384)
    vectors = embedder.embed(["hello world", "second text"])
    assert len(vectors) == 2
    assert all(len(vector) == 384 for vector in vectors)
    assert embedder.dimension == 384
