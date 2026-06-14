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
    String,
    UniqueConstraint,
    func,
)
from sqlalchemy.orm import DeclarativeBase, Mapped, mapped_column

# Filing checkpoint states.
STATUS_DISCOVERED = "discovered"
STATUS_STORED = "stored"
STATUS_FAILED = "failed"


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
