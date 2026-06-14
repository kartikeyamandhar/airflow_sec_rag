"""Tests for the typed configuration surface (``configs.settings``).

These cover the Phase 0 testing-plan requirements: load with no ``.env`` (use
defaults, do not crash), tolerate an unexpected env key, and confirm secrets are
redacted in ``repr``/``str``.
"""

from pathlib import Path

import pytest
from pydantic import SecretStr

from configs.settings import Settings, get_settings

# Every environment variable the model declares. Cleared before each test so a
# value in the developer's real shell cannot leak into assertions.
_DECLARED_ENV_KEYS = (
    "EDGAR_IDENTITY",
    "EDGAR_MAX_REQUESTS_PER_SECOND",
    "EDGAR_REQUEST_TIMEOUT_SECONDS",
    "RUNPOD_API_KEY",
    "R2_ACCOUNT_ID",
    "R2_ACCESS_KEY_ID",
    "R2_SECRET_ACCESS_KEY",
    "R2_BUCKET",
    "R2_ENDPOINT_URL",
    "STORAGE_BACKEND",
    "LOCAL_STORAGE_DIR",
    "DATABASE_URL",
    "QDRANT_URL",
    "LLM_API_KEY",
)


@pytest.fixture
def isolated_env(monkeypatch: pytest.MonkeyPatch, tmp_path: Path) -> None:
    """Isolate settings loading from the host environment.

    Clears the declared keys and changes into an empty temp directory so the
    relative ``.env`` lookup finds nothing, so the model must use its defaults.
    Also clears the ``get_settings`` cache so each test loads fresh.
    """
    for key in _DECLARED_ENV_KEYS:
        monkeypatch.delenv(key, raising=False)
    monkeypatch.chdir(tmp_path)
    get_settings.cache_clear()


def test_settings_use_defaults_without_env(isolated_env: None) -> None:
    settings = Settings()
    assert settings.edgar_identity == ""
    assert settings.r2_bucket == "sec-rag-raw"
    assert settings.qdrant_url == "http://localhost:6333"
    # Secret fields default to an empty secret, not None.
    assert settings.llm_api_key.get_secret_value() == ""


def test_settings_read_from_environment(
    isolated_env: None, monkeypatch: pytest.MonkeyPatch
) -> None:
    monkeypatch.setenv("EDGAR_IDENTITY", "Tester tester@example.com")
    monkeypatch.setenv("QDRANT_URL", "http://qdrant:6333")
    settings = Settings()
    assert settings.edgar_identity == "Tester tester@example.com"
    assert settings.qdrant_url == "http://qdrant:6333"


def test_settings_ignore_unexpected_env_keys(
    isolated_env: None, monkeypatch: pytest.MonkeyPatch
) -> None:
    monkeypatch.setenv("SOME_TOTALLY_UNRELATED_KEY", "value")
    # Must not raise despite the unexpected key (extra="ignore").
    settings = Settings()
    assert not hasattr(settings, "some_totally_unrelated_key")


def test_secret_fields_are_redacted(isolated_env: None) -> None:
    settings = Settings(llm_api_key=SecretStr("supersecret-token"))
    # The plaintext must not leak via repr()/str().
    assert "supersecret-token" not in repr(settings)
    assert "supersecret-token" not in str(settings)
    assert str(settings.llm_api_key) == "**********"
    # ...but it is retrievable explicitly at the point of use.
    assert settings.llm_api_key.get_secret_value() == "supersecret-token"


def test_get_settings_is_cached(isolated_env: None) -> None:
    assert get_settings() is get_settings()


def test_acquisition_defaults(isolated_env: None) -> None:
    settings = Settings()
    assert settings.edgar_max_requests_per_second == 9.0
    assert settings.edgar_request_timeout_seconds == 30.0
    assert settings.storage_backend == "s3"
    assert str(settings.local_storage_dir) == "data/raw"
    assert settings.database_url.startswith("postgresql+psycopg://")
