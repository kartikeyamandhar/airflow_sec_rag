"""Local filesystem storage backend (dev and tests)."""

from __future__ import annotations

from pathlib import Path


class LocalRawStore:
    """A :class:`~app.storage.base.RawStore` backed by a local directory.

    Keys map to paths under ``root``. A resolved-path check rejects any key that
    would escape ``root`` (defense in depth on top of key validation).
    """

    def __init__(self, root: Path) -> None:
        self._root = root.resolve()

    def _path(self, key: str) -> Path:
        path = (self._root / key).resolve()
        if path != self._root and self._root not in path.parents:
            raise ValueError(f"Key escapes storage root: {key!r}")
        return path

    def put(self, key: str, data: bytes, *, content_type: str) -> str:
        path = self._path(key)
        path.parent.mkdir(parents=True, exist_ok=True)
        path.write_bytes(data)
        return self.uri_for(key)

    def exists(self, key: str) -> bool:
        return self._path(key).is_file()

    def get(self, key: str) -> bytes:
        return self._path(key).read_bytes()

    def uri_for(self, key: str) -> str:
        return self._path(key).as_uri()
