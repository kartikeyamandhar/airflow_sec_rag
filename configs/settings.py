"""Typed application configuration loaded from the environment / ``.env``.

This module declares the *entire* configuration surface of SEC RAG as a single
``pydantic-settings`` model. The goals:

* **No secrets in code.** Values come from the environment or a local ``.env``
  (git-ignored). Only ``.env.example`` (no values) is committed.
* **No secrets in logs.** Secret fields use ``SecretStr``, whose ``repr()`` /
  ``str()`` render as ``**********``. Read the real value with
  ``.get_secret_value()`` at the point of use only.
* **Safe defaults.** Every field has a default, so the model loads without a
  ``.env`` (useful in tests and CI). Defaults are non-secret and inert; a missing
  required credential surfaces where it is *used*, not at import time.
* **Forward-compatible.** Keys for services introduced in later phases (RunPod,
  R2, Qdrant, LLM) are declared now to fix the contract, but carry no validation
  logic for subsystems that do not yet exist.

Usage::

    from configs.settings import get_settings

    settings = get_settings()
    edgar_ua = settings.edgar_identity
"""

from __future__ import annotations

from functools import lru_cache

from pydantic import Field, SecretStr
from pydantic_settings import BaseSettings, SettingsConfigDict


class Settings(BaseSettings):
    """Application settings, populated from environment variables / ``.env``.

    Field names map to upper-cased environment variables (e.g. ``edgar_identity``
    reads ``EDGAR_IDENTITY``). Unknown environment keys are ignored rather than
    raising, so the process does not crash on an unrelated variable in the shell.
    """

    model_config = SettingsConfigDict(
        env_file=".env",
        env_file_encoding="utf-8",
        case_sensitive=False,
        # Ignore env vars we do not declare. Defined behavior for the "unexpected
        # extra key" edge case (Phase 0 testing plan): tolerate, do not crash.
        extra="ignore",
    )

    # --- EDGAR -----------------------------------------------------------------
    edgar_identity: str = Field(
        default="",
        description=(
            "EDGAR User-Agent identifying the requester, e.g. "
            "'Your Name your-email@example.com'. Required by SEC or requests 403."
        ),
    )

    # --- RunPod (embedding compute) -------------------------------------------
    runpod_api_key: SecretStr = Field(
        default=SecretStr(""),
        description="RunPod API key for backfill GPU pod and serverless endpoint.",
    )

    # --- Cloudflare R2 (raw filing object storage) ----------------------------
    r2_account_id: str = Field(default="", description="Cloudflare R2 account id.")
    r2_access_key_id: SecretStr = Field(default=SecretStr(""), description="R2 access key id.")
    r2_secret_access_key: SecretStr = Field(
        default=SecretStr(""), description="R2 secret access key."
    )
    r2_bucket: str = Field(default="sec-rag-raw", description="R2 bucket holding raw filings.")

    # --- Qdrant (vector store) ------------------------------------------------
    qdrant_url: str = Field(
        default="http://localhost:6333",
        description="Qdrant endpoint (local Docker in dev, Cloud for the demo).",
    )

    # --- LLM (answer model + eval judge) --------------------------------------
    llm_api_key: SecretStr = Field(
        default=SecretStr(""),
        description="API key for the answer model and the evaluation judge.",
    )


@lru_cache(maxsize=1)
def get_settings() -> Settings:
    """Return a process-wide cached ``Settings`` instance.

    Caching avoids re-reading ``.env`` on every access and gives a single source
    of truth. Tests that need a fresh load can either construct ``Settings(...)``
    directly or call ``get_settings.cache_clear()``.
    """
    return Settings()
