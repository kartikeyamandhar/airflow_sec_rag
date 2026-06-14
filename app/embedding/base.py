"""The embedding interface shared by all backends."""

from __future__ import annotations

from typing import Protocol


class Embedder(Protocol):
    """Turns texts into fixed-dimension dense vectors."""

    @property
    def dimension(self) -> int:
        """The length of every vector this embedder produces."""
        ...

    def embed(self, texts: list[str]) -> list[list[float]]:
        """Return one vector per input text, in order."""
        ...
