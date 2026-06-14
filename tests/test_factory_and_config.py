"""Storage factory selection and run-config loading."""

from datetime import date
from pathlib import Path

import pytest

from app.storage.factory import build_raw_store
from app.storage.local import LocalRawStore
from app.storage.s3 import S3RawStore
from configs.run_config import RunConfig, load_run_config
from configs.settings import Settings


def test_factory_returns_local_backend(monkeypatch: pytest.MonkeyPatch, tmp_path: Path) -> None:
    monkeypatch.chdir(tmp_path)  # avoid reading the real .env
    settings = Settings(storage_backend="local", local_storage_dir=tmp_path)
    assert isinstance(build_raw_store(settings), LocalRawStore)


def test_factory_returns_s3_backend(monkeypatch: pytest.MonkeyPatch, tmp_path: Path) -> None:
    monkeypatch.chdir(tmp_path)
    settings = Settings(
        storage_backend="s3",
        r2_endpoint_url="https://acct.r2.cloudflarestorage.com",
        r2_bucket="bucket",
    )
    assert isinstance(build_raw_store(settings), S3RawStore)


def test_load_run_config(tmp_path: Path) -> None:
    path = tmp_path / "run.yaml"
    path.write_text(
        'tickers: [AAPL, MSFT]\nforms: ["10-K"]\nstart_date: 2023-01-01\nend_date: 2023-12-31\n',
        encoding="utf-8",
    )
    config = load_run_config(path)
    assert config.tickers == ["AAPL", "MSFT"]
    assert config.forms == ["10-K"]
    assert config.start_date == date(2023, 1, 1)


def test_run_config_defaults_forms() -> None:
    config = RunConfig(tickers=["AAPL"], start_date=date(2023, 1, 1), end_date=date(2023, 12, 31))
    assert config.forms == ["10-K", "10-Q"]
