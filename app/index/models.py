"""SQLAlchemy models for the filing index and checkpoint store.

SQLAlchemy 2.0 typed ORM (``Mapped`` / ``mapped_column``). One company row per
CIK, one filing row per accession (the idempotency key), and an artifact row per
stored object. Status on ``Filing`` is the checkpoint.
"""

from __future__ import annotations

import datetime

from sqlalchemy import (
    BigInteger,
    Boolean,
    Date,
    DateTime,
    Float,
    Integer,
    String,
    Text,
    UniqueConstraint,
    func,
)
from sqlalchemy.orm import DeclarativeBase, Mapped, mapped_column

# Filing checkpoint states.
STATUS_DISCOVERED = "discovered"
STATUS_STORED = "stored"
STATUS_FAILED = "failed"
STATUS_PARSED = "parsed"
STATUS_PARSE_FAILED = "parse_failed"


class Base(DeclarativeBase):
    """Declarative base for all index tables."""


class Company(Base):
    __tablename__ = "companies"

    cik: Mapped[int] = mapped_column(BigInteger, primary_key=True, autoincrement=False)
    ticker: Mapped[str] = mapped_column(String(16))
    name: Mapped[str] = mapped_column(String(512))
    updated_at: Mapped[datetime.datetime] = mapped_column(
        DateTime(timezone=True), server_default=func.now(), onupdate=func.now()
    )


class Filing(Base):
    __tablename__ = "filings"

    id: Mapped[int] = mapped_column(primary_key=True)
    cik: Mapped[int] = mapped_column(BigInteger, index=True)
    ticker: Mapped[str | None] = mapped_column(String(16), nullable=True)
    accession: Mapped[str] = mapped_column(String(20), unique=True)
    form: Mapped[str] = mapped_column(String(16), index=True)
    filing_date: Mapped[datetime.date] = mapped_column(Date)
    report_date: Mapped[datetime.date | None] = mapped_column(Date, nullable=True)
    primary_document: Mapped[str] = mapped_column(String(256))
    primary_doc_url: Mapped[str] = mapped_column(String(1024))
    is_amendment: Mapped[bool] = mapped_column(Boolean)
    amends_accession: Mapped[str | None] = mapped_column(String(20), nullable=True)
    status: Mapped[str] = mapped_column(String(16), index=True)
    discovered_at: Mapped[datetime.datetime] = mapped_column(
        DateTime(timezone=True), server_default=func.now()
    )
    updated_at: Mapped[datetime.datetime] = mapped_column(
        DateTime(timezone=True), server_default=func.now(), onupdate=func.now()
    )


class Artifact(Base):
    __tablename__ = "artifacts"
    __table_args__ = (UniqueConstraint("ref_type", "ref_key", "kind", name="uq_artifact_ref"),)

    id: Mapped[int] = mapped_column(primary_key=True)
    ref_type: Mapped[str] = mapped_column(String(16))
    ref_key: Mapped[str] = mapped_column(String(64))
    kind: Mapped[str] = mapped_column(String(32))
    storage_uri: Mapped[str] = mapped_column(String(1024))
    sha256: Mapped[str] = mapped_column(String(64))
    size_bytes: Mapped[int] = mapped_column(BigInteger)
    content_type: Mapped[str] = mapped_column(String(128))
    source_url: Mapped[str] = mapped_column(String(1024))
    fetched_at: Mapped[datetime.datetime] = mapped_column(DateTime(timezone=True))


class NumericFact(Base):
    __tablename__ = "numeric_facts"

    id: Mapped[int] = mapped_column(primary_key=True)
    cik: Mapped[int] = mapped_column(BigInteger, index=True)
    taxonomy: Mapped[str] = mapped_column(String(32))
    concept: Mapped[str] = mapped_column(String(128), index=True)
    label: Mapped[str | None] = mapped_column(String(512), nullable=True)
    unit: Mapped[str] = mapped_column(String(32))
    # Double precision. Financial magnitudes fit; exactness for huge integers is a
    # known limitation (see PROGRESS debt), acceptable for the MVP numbers plane.
    value: Mapped[float] = mapped_column(Float)
    period_start: Mapped[datetime.date | None] = mapped_column(Date, nullable=True)
    period_end: Mapped[datetime.date] = mapped_column(Date)
    fiscal_year: Mapped[int | None] = mapped_column(Integer, nullable=True)
    fiscal_period: Mapped[str | None] = mapped_column(String(4), nullable=True)
    form: Mapped[str] = mapped_column(String(16))
    accession: Mapped[str] = mapped_column(String(20), index=True)
    frame: Mapped[str | None] = mapped_column(String(32), nullable=True)


class Chunk(Base):
    __tablename__ = "chunks"
    __table_args__ = (
        UniqueConstraint("accession", "chunk_index", name="uq_chunk_accession_index"),
    )

    id: Mapped[int] = mapped_column(primary_key=True)
    accession: Mapped[str] = mapped_column(String(20), index=True)
    cik: Mapped[int] = mapped_column(BigInteger, index=True)
    ticker: Mapped[str | None] = mapped_column(String(16), nullable=True)
    form: Mapped[str] = mapped_column(String(16))
    section: Mapped[str] = mapped_column(String(64), index=True)
    kind: Mapped[str] = mapped_column(String(8))
    chunk_index: Mapped[int] = mapped_column(Integer)
    parent_index: Mapped[int | None] = mapped_column(Integer, nullable=True)
    text: Mapped[str] = mapped_column(Text)
    char_start: Mapped[int] = mapped_column(Integer)
    char_end: Mapped[int] = mapped_column(Integer)
    token_estimate: Mapped[int] = mapped_column(Integer)
