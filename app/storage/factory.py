"""Build the configured storage backend from settings."""

from __future__ import annotations

from app.storage.base import RawStore
from app.storage.local import LocalRawStore
from app.storage.s3 import S3RawStore
from configs.settings import Settings


def build_raw_store(settings: Settings) -> RawStore:
    """Return the ``RawStore`` selected by ``settings.storage_backend``."""
    if settings.storage_backend == "local":
        return LocalRawStore(settings.local_storage_dir)
    return S3RawStore(
        bucket=settings.r2_bucket,
        endpoint_url=settings.r2_endpoint_url,
        access_key=settings.r2_access_key_id.get_secret_value(),
        secret_key=settings.r2_secret_access_key.get_secret_value(),
    )
