"""CPU embedding backend using fastembed (ONNX bge models).

The default backend: real bge vectors, no GPU, no cloud, no API key. Enough for
the MVP scale; the RunPod backend takes over for large backfills.
"""

from __future__ import annotations

from fastembed import TextEmbedding


class FastembedEmbedder:
    """An ``Embedder`` backed by a local fastembed model."""

    def __init__(self, model_name: str, dimension: int) -> None:
        self._model = TextEmbedding(model_name=model_name)
        self._dimension = dimension

    @property
    def dimension(self) -> int:
        return self._dimension

    def embed(self, texts: list[str]) -> list[list[float]]:
        return [[float(value) for value in vector] for vector in self._model.embed(texts)]
