"""Storage backends and key derivation."""

from pathlib import Path

import boto3
import pytest
from moto import mock_aws

from app.storage.keys import company_facts_key, primary_document_key
from app.storage.local import LocalRawStore
from app.storage.s3 import S3RawStore


def test_local_roundtrip(tmp_path: Path) -> None:
    store = LocalRawStore(tmp_path)
    assert store.exists("a/b/c.txt") is False
    uri = store.put("a/b/c.txt", b"hello", content_type="text/plain")
    assert uri.startswith("file://")
    assert store.exists("a/b/c.txt") is True
    assert store.get("a/b/c.txt") == b"hello"


def test_local_put_is_idempotent_overwrite(tmp_path: Path) -> None:
    store = LocalRawStore(tmp_path)
    store.put("k.txt", b"first", content_type="text/plain")
    store.put("k.txt", b"second", content_type="text/plain")
    assert store.get("k.txt") == b"second"


def test_local_rejects_traversal(tmp_path: Path) -> None:
    store = LocalRawStore(tmp_path)
    with pytest.raises(ValueError):
        store.put("../escape.txt", b"x", content_type="text/plain")


def test_storage_keys() -> None:
    assert (
        primary_document_key(320193, "0000320193-23-000106", "aapl.htm")
        == "filings/0000320193/000032019323000106/aapl.htm"
    )
    assert company_facts_key(320193) == "companyfacts/0000320193.json"


@mock_aws
def test_s3_roundtrip() -> None:
    bucket = "sec-rag-test"
    boto3.client("s3", region_name="us-east-1").create_bucket(Bucket=bucket)
    store = S3RawStore(
        bucket=bucket,
        endpoint_url="https://s3.amazonaws.com",
        access_key="testing",
        secret_key="testing",
        region="us-east-1",
    )
    assert store.exists("companyfacts/0000320193.json") is False
    uri = store.put(
        "companyfacts/0000320193.json", b'{"facts": 1}', content_type="application/json"
    )
    assert uri == "s3://sec-rag-test/companyfacts/0000320193.json"
    assert store.exists("companyfacts/0000320193.json") is True
    assert store.get("companyfacts/0000320193.json") == b'{"facts": 1}'
