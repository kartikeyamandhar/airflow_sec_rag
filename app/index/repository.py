"""Data-access functions for the filing index.

All writes are idempotent upserts keyed on the natural key (CIK, accession, or
the artifact triple). Crucially, ``upsert_filing`` updates metadata on conflict
but never downgrades a ``stored`` filing back to ``discovered`` so a re-run of
discovery cannot undo acquisition progress.
"""

from __future__ import annotations

from sqlalchemy import func, select, update
from sqlalchemy.dialects.postgresql import insert as pg_insert
from sqlalchemy.orm import Session

from app.domain.models import Company as CompanyModel
from app.domain.models import FilingRef, StoredArtifact
from app.index.models import (
    STATUS_DISCOVERED,
    STATUS_FAILED,
    STATUS_STORED,
    Artifact,
    Company,
    Filing,
)


def upsert_company(session: Session, company: CompanyModel) -> None:
    stmt = pg_insert(Company).values(cik=company.cik, ticker=company.ticker, name=company.name)
    stmt = stmt.on_conflict_do_update(
        index_elements=[Company.cik],
        set_={
            "ticker": stmt.excluded.ticker,
            "name": stmt.excluded.name,
            "updated_at": func.now(),
        },
    )
    session.execute(stmt)


def upsert_filing(session: Session, ref: FilingRef) -> None:
    stmt = pg_insert(Filing).values(
        cik=ref.cik,
        ticker=ref.ticker,
        accession=ref.accession,
        form=ref.form,
        filing_date=ref.filing_date,
        report_date=ref.report_date,
        primary_document=ref.primary_document,
        primary_doc_url=ref.primary_doc_url,
        is_amendment=ref.is_amendment,
        amends_accession=ref.amends_accession,
        status=STATUS_DISCOVERED,
    )
    # Update metadata on conflict, but never touch ``status`` (do not undo a prior
    # acquisition by resetting stored -> discovered).
    stmt = stmt.on_conflict_do_update(
        index_elements=[Filing.accession],
        set_={
            "ticker": stmt.excluded.ticker,
            "form": stmt.excluded.form,
            "filing_date": stmt.excluded.filing_date,
            "report_date": stmt.excluded.report_date,
            "primary_document": stmt.excluded.primary_document,
            "primary_doc_url": stmt.excluded.primary_doc_url,
            "is_amendment": stmt.excluded.is_amendment,
            "amends_accession": stmt.excluded.amends_accession,
            "updated_at": func.now(),
        },
    )
    session.execute(stmt)


def upsert_artifact(session: Session, artifact: StoredArtifact) -> None:
    stmt = pg_insert(Artifact).values(
        ref_type=artifact.ref_type,
        ref_key=artifact.ref_key,
        kind=artifact.kind,
        storage_uri=artifact.storage_uri,
        sha256=artifact.sha256,
        size_bytes=artifact.size_bytes,
        content_type=artifact.content_type,
        source_url=artifact.source_url,
        fetched_at=artifact.fetched_at,
    )
    stmt = stmt.on_conflict_do_update(
        constraint="uq_artifact_ref",
        set_={
            "storage_uri": stmt.excluded.storage_uri,
            "sha256": stmt.excluded.sha256,
            "size_bytes": stmt.excluded.size_bytes,
            "content_type": stmt.excluded.content_type,
            "source_url": stmt.excluded.source_url,
            "fetched_at": stmt.excluded.fetched_at,
        },
    )
    session.execute(stmt)


def filings_by_status(session: Session, status: str) -> list[Filing]:
    return list(session.scalars(select(Filing).where(Filing.status == status)))


def filings_pending_acquire(session: Session) -> list[Filing]:
    """Filings not yet stored (includes failed, so a re-run retries them)."""
    return list(session.scalars(select(Filing).where(Filing.status != STATUS_STORED)))


def distinct_ciks(session: Session) -> list[int]:
    return list(session.scalars(select(Company.cik).order_by(Company.cik)))


def mark_filing_stored(session: Session, accession: str) -> None:
    session.execute(
        update(Filing)
        .where(Filing.accession == accession)
        .values(status=STATUS_STORED, updated_at=func.now())
    )


def mark_filing_failed(session: Session, accession: str) -> None:
    session.execute(
        update(Filing)
        .where(Filing.accession == accession)
        .values(status=STATUS_FAILED, updated_at=func.now())
    )


def artifact_exists(session: Session, ref_type: str, ref_key: str, kind: str) -> bool:
    found = session.scalar(
        select(Artifact.id).where(
            Artifact.ref_type == ref_type,
            Artifact.ref_key == ref_key,
            Artifact.kind == kind,
        )
    )
    return found is not None
