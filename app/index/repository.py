"""Data-access functions for the filing index.

All writes are idempotent upserts keyed on the natural key (CIK, accession, or
the artifact triple). Crucially, ``upsert_filing`` updates metadata on conflict
but never downgrades a ``stored`` filing back to ``discovered`` so a re-run of
discovery cannot undo acquisition progress.
"""

from __future__ import annotations

import datetime

from sqlalchemy import delete, func, select, update
from sqlalchemy.dialects.postgresql import insert as pg_insert
from sqlalchemy.orm import Session

from app.domain.models import Chunk as ChunkModel
from app.domain.models import Company as CompanyModel
from app.domain.models import FilingRef, StoredArtifact
from app.domain.models import NumericFact as NumericFactModel
from app.index.models import (
    STATUS_DISCOVERED,
    STATUS_EMBED_FAILED,
    STATUS_EMBEDDED,
    STATUS_FAILED,
    STATUS_PARSE_FAILED,
    STATUS_PARSED,
    STATUS_STORED,
    Artifact,
    Chunk,
    Company,
    Filing,
    NumericFact,
    QueryLog,
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


def all_companies(session: Session) -> list[tuple[str, str]]:
    """Return (ticker, name) pairs for every known company (for query aliases)."""
    rows = session.execute(select(Company.ticker, Company.name)).all()
    return [(row[0], row[1]) for row in rows]


def report_dates_for_accessions(session: Session, accessions: set[str]) -> dict[str, datetime.date]:
    """Return the period-of-report date for each accession that has one."""
    if not accessions:
        return {}
    rows = session.execute(
        select(Filing.accession, Filing.report_date).where(Filing.accession.in_(accessions))
    ).all()
    return {row[0]: row[1] for row in rows if row[1] is not None}


def log_query(
    session: Session,
    *,
    question: str,
    ticker: str | None,
    num_context: int,
    coverage: float,
    faithfulness: float | None,
    confidence: float,
    refused: bool,
) -> None:
    """Append one online query-log row."""
    session.add(
        QueryLog(
            question=question,
            ticker=ticker,
            num_context=num_context,
            coverage=coverage,
            faithfulness=faithfulness,
            confidence=confidence,
            refused=refused,
        )
    )


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


# --- Phase 2: parsing checkpoints, facts, and chunks -------------------------


def filings_to_parse(session: Session) -> list[Filing]:
    """Filings that are stored but not yet parsed."""
    return list(session.scalars(select(Filing).where(Filing.status == STATUS_STORED)))


def mark_filing_parsed(session: Session, accession: str) -> None:
    session.execute(
        update(Filing)
        .where(Filing.accession == accession)
        .values(status=STATUS_PARSED, updated_at=func.now())
    )


def mark_filing_parse_failed(session: Session, accession: str) -> None:
    session.execute(
        update(Filing)
        .where(Filing.accession == accession)
        .values(status=STATUS_PARSE_FAILED, updated_at=func.now())
    )


def ciks_with_facts_artifact(session: Session) -> list[int]:
    """CIKs that have a stored CompanyFacts artifact."""
    rows = session.scalars(
        select(Artifact.ref_key).where(
            Artifact.ref_type == "company", Artifact.kind == "company_facts"
        )
    )
    return sorted({int(ref_key) for ref_key in rows})


def numeric_fact_count_for_cik(session: Session, cik: int) -> int:
    count = session.scalar(
        select(func.count()).select_from(NumericFact).where(NumericFact.cik == cik)
    )
    return count or 0


def replace_numeric_facts(session: Session, cik: int, facts: list[NumericFactModel]) -> None:
    """Replace all numeric facts for a CIK (idempotent re-parse)."""
    session.execute(delete(NumericFact).where(NumericFact.cik == cik))
    session.add_all(
        [
            NumericFact(
                cik=fact.cik,
                taxonomy=fact.taxonomy,
                concept=fact.concept,
                label=fact.label,
                unit=fact.unit,
                value=fact.value,
                period_start=fact.period_start,
                period_end=fact.period_end,
                fiscal_year=fact.fiscal_year,
                fiscal_period=fact.fiscal_period,
                form=fact.form,
                accession=fact.accession,
                frame=fact.frame,
            )
            for fact in facts
        ]
    )


def replace_chunks(session: Session, accession: str, chunks: list[ChunkModel]) -> None:
    """Replace all chunks for a filing (idempotent re-parse)."""
    session.execute(delete(Chunk).where(Chunk.accession == accession))
    session.add_all(
        [
            Chunk(
                accession=chunk.accession,
                cik=chunk.cik,
                ticker=chunk.ticker,
                form=chunk.form,
                section=chunk.section,
                kind=chunk.kind,
                chunk_index=chunk.chunk_index,
                parent_index=chunk.parent_index,
                text=chunk.text,
                char_start=chunk.char_start,
                char_end=chunk.char_end,
                token_estimate=chunk.token_estimate,
            )
            for chunk in chunks
        ]
    )


def numeric_fact_count(session: Session) -> int:
    count = session.scalar(select(func.count()).select_from(NumericFact))
    return count or 0


def chunk_count_for_accession(session: Session, accession: str) -> int:
    count = session.scalar(
        select(func.count()).select_from(Chunk).where(Chunk.accession == accession)
    )
    return count or 0


# --- Phase 3: embedding checkpoints and chunk reads --------------------------


def filings_to_embed(session: Session) -> list[Filing]:
    """Filings that are parsed but not yet embedded."""
    return list(session.scalars(select(Filing).where(Filing.status == STATUS_PARSED)))


def child_chunks_for_filing(session: Session, accession: str) -> list[Chunk]:
    """The child (retrieval-unit) chunks of a filing, in order."""
    return list(
        session.scalars(
            select(Chunk)
            .where(Chunk.accession == accession, Chunk.kind == "child")
            .order_by(Chunk.chunk_index)
        )
    )


def mark_filing_embedded(session: Session, accession: str) -> None:
    session.execute(
        update(Filing)
        .where(Filing.accession == accession)
        .values(status=STATUS_EMBEDDED, updated_at=func.now())
    )


def mark_filing_embed_failed(session: Session, accession: str) -> None:
    session.execute(
        update(Filing)
        .where(Filing.accession == accession)
        .values(status=STATUS_EMBED_FAILED, updated_at=func.now())
    )
