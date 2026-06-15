"""Repository idempotency and checkpoint behavior (against real Postgres)."""

from datetime import UTC, date, datetime

from sqlalchemy import func, select
from sqlalchemy.orm import Session

from app.domain.models import (
    ArtifactKind,
    Company,
    FilingRef,
    RefType,
    StoredArtifact,
)
from app.index import repository as repo
from app.index.models import STATUS_DISCOVERED, STATUS_STORED
from app.index.models import Artifact as ArtifactRow
from app.index.models import Company as CompanyRow
from app.index.models import Filing as FilingRow
from app.index.models import QueryLog as QueryLogRow

_ACCESSION = "0000320193-23-000106"


def _filing_ref(
    accession: str = _ACCESSION,
    form: str = "10-K",
    *,
    is_amendment: bool = False,
    amends_accession: str | None = None,
) -> FilingRef:
    return FilingRef(
        cik=320193,
        ticker="AAPL",
        accession=accession,
        form=form,
        filing_date=date(2023, 11, 3),
        report_date=date(2023, 9, 30),
        primary_document="aapl.htm",
        primary_doc_url="https://www.sec.gov/Archives/edgar/data/320193/x/aapl.htm",
        is_amendment=is_amendment,
        amends_accession=amends_accession,
    )


def _artifact(ref_key: str = _ACCESSION, kind: ArtifactKind = "primary_document") -> StoredArtifact:
    ref_type: RefType = "filing" if kind == "primary_document" else "company"
    return StoredArtifact(
        ref_type=ref_type,
        ref_key=ref_key,
        kind=kind,
        storage_uri="s3://sec-rag-raw/x",
        sha256="a" * 64,
        size_bytes=10,
        content_type="text/html",
        source_url="https://www.sec.gov/x",
        fetched_at=datetime(2024, 1, 1, tzinfo=UTC),
    )


def _count(session: Session, model: type[object]) -> int:
    value = session.scalar(select(func.count()).select_from(model))
    return value or 0


def test_upsert_company_is_idempotent_and_updates(db_session: Session) -> None:
    repo.upsert_company(db_session, Company(cik=320193, ticker="AAPL", name="Apple"))
    repo.upsert_company(db_session, Company(cik=320193, ticker="AAPL", name="Apple Inc."))
    assert _count(db_session, CompanyRow) == 1
    row = db_session.scalar(select(CompanyRow).where(CompanyRow.cik == 320193))
    assert row is not None
    assert row.name == "Apple Inc."


def test_upsert_filing_is_idempotent(db_session: Session) -> None:
    repo.upsert_filing(db_session, _filing_ref())
    repo.upsert_filing(db_session, _filing_ref())
    assert _count(db_session, FilingRow) == 1
    row = db_session.scalar(select(FilingRow).where(FilingRow.accession == _ACCESSION))
    assert row is not None
    assert row.status == STATUS_DISCOVERED


def test_rerun_discovery_does_not_downgrade_stored(db_session: Session) -> None:
    repo.upsert_filing(db_session, _filing_ref())
    repo.mark_filing_stored(db_session, _ACCESSION)
    # A later discovery re-run upserts the same filing again.
    repo.upsert_filing(db_session, _filing_ref())
    row = db_session.scalar(select(FilingRow).where(FilingRow.accession == _ACCESSION))
    assert row is not None
    assert row.status == STATUS_STORED


def test_filings_pending_acquire_excludes_stored(db_session: Session) -> None:
    repo.upsert_filing(db_session, _filing_ref(accession="0000320193-23-000106"))
    repo.upsert_filing(db_session, _filing_ref(accession="0000320193-23-000077"))
    repo.mark_filing_stored(db_session, "0000320193-23-000106")
    pending = repo.filings_pending_acquire(db_session)
    assert {f.accession for f in pending} == {"0000320193-23-000077"}


def test_upsert_artifact_is_idempotent(db_session: Session) -> None:
    repo.upsert_artifact(db_session, _artifact())
    repo.upsert_artifact(db_session, _artifact())
    assert _count(db_session, ArtifactRow) == 1
    assert repo.artifact_exists(db_session, "filing", _ACCESSION, "primary_document")
    assert not repo.artifact_exists(db_session, "company", "320193", "company_facts")


def test_log_query_writes_a_row(db_session: Session) -> None:
    repo.log_query(
        db_session,
        question="what are the risks?",
        ticker="AAPL",
        num_context=8,
        coverage=0.9,
        faithfulness=0.8,
        confidence=0.7,
        refused=False,
    )
    assert _count(db_session, QueryLogRow) == 1
