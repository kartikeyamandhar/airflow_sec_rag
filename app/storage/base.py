"""The storage interface shared by all backends."""

from __future__ import annotations

from typing import Protocol, runtime_checkable


@runtime_checkable
class RawStore(Protocol):
    """A content store for raw bytes addressed by string keys.

    Implementations must be idempotent: ``put`` of the same key overwrites and is
    safe to repeat, which is what makes acquisition re-runs safe.
    """

    def put(self, key: str, data: bytes, *, content_type: str) -> str:
        """Write ``data`` at ``key`` and return its storage URI."""
        ...

    def exists(self, key: str) -> bool:
        """Return whether an object exists at ``key``."""
        ...

    def get(self, key: str) -> bytes:
        """Read and return the bytes at ``key``. Raises if absent."""
        ...

    def uri_for(self, key: str) -> str:
        """Return the storage URI for ``key`` without performing IO."""
        ...
