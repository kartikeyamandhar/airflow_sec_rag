"""Core domain value objects, shared across storage, edgar, and index layers.

All models are frozen (immutable, hashable) value objects. They carry no
behavior beyond validation. Provenance fields (source URL, fetched_at, sha256)
are first-class because the product promise is span-cited, dated answers.
"""

from __future__ import annotations

from datetime import date, datetime
from typing import Literal

from pydantic import BaseModel, ConfigDict, Field, field_validator

from app.domain.identifiers import normalize_accession

# What an artifact is attached to: a single filing, or a whole company (CIK).
RefType = Literal["filing", "company"]
# The kind of raw artifact stored.
ArtifactKind = Literal["primary_document", "company_facts"]
# A chunk is either a section-level parent or a token-window child within it.
ChunkKind = Literal["parent", "child"]


class Company(BaseModel):
    """A company identified by its SEC CIK."""

    model_config = ConfigDict(frozen=True)

    cik: int = Field(ge=0)
    ticker: str
    name: str


class FilingRef(BaseModel):
    """A reference to a single filing and where its primary document lives.

    This is acquisition metadata, not parsed content. ``accession`` is always the
    canonical dashed form. ``report_date`` (period of report) is what later "as
    of" answers key on; ``filing_date`` is merely when it was submitted.
    """

    model_config = ConfigDict(frozen=True)

    cik: int = Field(ge=0)
    ticker: str | None = None
    accession: str
    form: str
    filing_date: date
    report_date: date | None = None
    primary_document: str
    primary_doc_url: str
    is_amendment: bool = False
    amends_accession: str | None = None

    @field_validator("accession", "amends_accession")
    @classmethod
    def _normalize_accession(cls, value: str | None) -> str | None:
        if value is None:
            return None
        return normalize_accession(value)


class StoredArtifact(BaseModel):
    """A raw artifact that has been written to object storage, with provenance."""

    model_config = ConfigDict(frozen=True)

    ref_type: RefType
    ref_key: str
    kind: ArtifactKind
    storage_uri: str
    sha256: str
    size_bytes: int = Field(ge=0)
    content_type: str
    source_url: str
    fetched_at: datetime


class NumericFact(BaseModel):
    """A single XBRL-tagged numeric fact (the numbers plane).

    Instant facts (e.g. Assets at a date) have only ``period_end``; duration facts
    (e.g. Revenues over a quarter) also have ``period_start``.
    """

    model_config = ConfigDict(frozen=True)

    cik: int = Field(ge=0)
    taxonomy: str
    concept: str
    label: str | None = None
    unit: str
    value: float
    period_start: date | None = None
    period_end: date
    fiscal_year: int | None = None
    fiscal_period: str | None = None
    form: str
    accession: str
    frame: str | None = None

    @field_validator("accession")
    @classmethod
    def _normalize_accession(cls, value: str) -> str:
        return normalize_accession(value)


class Chunk(BaseModel):
    """A retrieval unit of narrative text with citation offsets.

    A ``parent`` chunk holds a whole section's normalized text. A ``child`` chunk is
    a token-window slice of its parent; its ``char_start``/``char_end`` index into
    the parent's text, so ``parent.text[child.char_start:child.char_end]`` is the
    exact citable span.
    """

    model_config = ConfigDict(frozen=True)

    accession: str
    cik: int = Field(ge=0)
    ticker: str | None = None
    form: str
    section: str
    kind: ChunkKind
    chunk_index: int = Field(ge=0)
    parent_index: int | None = None
    text: str
    char_start: int = Field(ge=0)
    char_end: int = Field(ge=0)
    token_estimate: int = Field(ge=0)

    @field_validator("accession")
    @classmethod
    def _normalize_accession(cls, value: str) -> str:
        return normalize_accession(value)
