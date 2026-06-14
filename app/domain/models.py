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
