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
from pathlib import Path
from typing import Literal

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
    edgar_max_requests_per_second: float = Field(
        default=9.0,
        gt=0,
        description="Client-side rate cap. SEC allows 10/s per IP; stay under it.",
    )
    edgar_request_timeout_seconds: float = Field(
        default=30.0,
        gt=0,
        description="Per-request timeout for raw EDGAR fetches.",
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
    r2_endpoint_url: str = Field(
        default="",
        description=(
            "R2 S3-compatible endpoint, e.g. "
            "'https://<account-id>.r2.cloudflarestorage.com'. No bucket, no trailing slash."
        ),
    )

    # --- Object storage backend -----------------------------------------------
    storage_backend: Literal["s3", "local"] = Field(
        default="s3",
        description="'s3' for R2/S3 (default); 'local' filesystem for tests and offline dev.",
    )
    local_storage_dir: Path = Field(
        default=Path("data/raw"),
        description="Root dir for the local storage backend (storage_backend='local').",
    )

    # --- Index / checkpoint database ------------------------------------------
    database_url: str = Field(
        default="postgresql+psycopg://sec_rag:sec_rag@localhost:5432/sec_rag",
        description="SQLAlchemy URL for the Postgres filing index and checkpoints.",
    )

    # --- Parsing / chunking ---------------------------------------------------
    parse_max_document_bytes: int = Field(
        default=50_000_000,
        gt=0,
        description="Reject documents larger than this when parsing (OOM guard).",
    )
    chunk_child_tokens: int = Field(
        default=350, gt=0, description="Target token estimate per child chunk."
    )
    chunk_overlap_tokens: int = Field(
        default=50, ge=0, description="Token overlap between adjacent child chunks."
    )
    chunk_min_section_chars: int = Field(
        default=200,
        ge=0,
        description="Sections shorter than this are dropped as likely table-of-contents.",
    )

    # --- Embedding ------------------------------------------------------------
    embedding_backend: Literal["fastembed", "runpod"] = Field(
        default="fastembed",
        description="'fastembed' (CPU, default) or 'runpod' (GPU serverless endpoint).",
    )
    embedding_model: str = Field(
        default="BAAI/bge-small-en-v1.5",
        description="Embedding model name (bge family).",
    )
    embedding_dimension: int = Field(
        default=384,
        gt=0,
        description="Vector dimension; must match the model and the Qdrant collection.",
    )
    embedding_batch_size: int = Field(default=64, gt=0, description="Texts per embedding call.")
    runpod_endpoint_id: str = Field(
        default="",
        description="RunPod serverless endpoint id (used when embedding_backend='runpod').",
    )

    # --- Qdrant (vector store) ------------------------------------------------
    qdrant_url: str = Field(
        default="http://localhost:6333",
        description="Qdrant endpoint (local Docker in dev, Cloud for the demo).",
    )
    qdrant_api_key: SecretStr = Field(
        default=SecretStr(""), description="Qdrant Cloud API key (empty for local)."
    )
    qdrant_collection: str = Field(
        default="sec_filings", description="Qdrant collection holding chunk vectors."
    )

    # --- Retrieval ------------------------------------------------------------
    sparse_embedding_model: str = Field(
        default="Qdrant/bm25",
        description="Sparse (lexical) model for the BM25 half of hybrid search.",
    )
    reranker_backend: Literal["fastembed", "none"] = Field(
        default="fastembed",
        description="'fastembed' cross-encoder rerank, or 'none' to skip reranking.",
    )
    reranker_model: str = Field(
        default="Xenova/ms-marco-MiniLM-L-6-v2",
        description="Cross-encoder model used to rerank candidates.",
    )
    retrieval_top_k: int = Field(
        default=20, gt=0, description="Candidates pulled from hybrid search per query."
    )
    rerank_top_n: int = Field(default=5, gt=0, description="Results kept after reranking.")

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
