"""End-to-end indexing: parsed chunks to Qdrant, idempotently.

Uses a deterministic fake embedder and an in-memory Qdrant, with a real
testcontainers Postgres for the chunk rows and status checkpoint.
"""

from datetime import date

from qdrant_client import QdrantClient
from sqlalchemy import select
from sqlalchemy.orm import Session, sessionmaker

from app.domain.models import Company, FilingRef
from app.index import repository as repo
from app.index.db import session_scope
from app.index.models import STATUS_EMBEDDED
from app.index.models import Filing as FilingRow
from app.narrative.chunk import chunk_sections
from app.narrative.extract import Section
from app.vectorstore.qdrant_index import QdrantIndex
from scripts.index_chunks import run_index
from tests.fakes import FakeEmbedder, FakeSparseEmbedder

_ACCESSION = "0000320193-23-000106"
_CIK = 320193


def _seed_parsed_filing(session_factory: sessionmaker[Session]) -> int:
    section = Section("Item 1", " ".join(f"word{i}" for i in range(300)))
    chunks = chunk_sections(
        [section],
        accession=_ACCESSION,
        cik=_CIK,
        ticker="AAPL",
        form="10-K",
        child_tokens=100,
        overlap_tokens=20,
    )
    child_count = sum(1 for c in chunks if c.kind == "child")
    with session_scope(session_factory) as session:
        repo.upsert_company(session, Company(cik=_CIK, ticker="AAPL", name="Apple Inc."))
        repo.upsert_filing(
            session,
            FilingRef(
                cik=_CIK,
                ticker="AAPL",
                accession=_ACCESSION,
                form="10-K",
                filing_date=date(2023, 11, 3),
                report_date=date(2023, 9, 30),
                primary_document="aapl.htm",
                primary_doc_url="https://www.sec.gov/x/aapl.htm",
            ),
        )
        repo.mark_filing_stored(session, _ACCESSION)
        repo.replace_chunks(session, _ACCESSION, chunks)
        repo.mark_filing_parsed(session, _ACCESSION)
    return child_count


def test_index_then_reindex_is_idempotent(
    pg_session_factory: sessionmaker[Session],
) -> None:
    child_count = _seed_parsed_filing(pg_session_factory)
    assert child_count > 1

    embedder = FakeEmbedder(dimension=8)
    sparse = FakeSparseEmbedder()
    index = QdrantIndex(QdrantClient(location=":memory:"), "sec_test", dimension=8)

    summary = run_index(embedder, sparse, index, pg_session_factory, batch_size=4)
    assert summary.filings_embedded == 1
    assert summary.points == child_count
    assert summary.failed == 0
    assert index.count() == child_count

    with pg_session_factory() as session:
        filing = session.scalar(select(FilingRow).where(FilingRow.accession == _ACCESSION))
        assert filing is not None
        assert filing.status == STATUS_EMBEDDED

    # Re-run: the embedded filing is no longer pending, and the index is unchanged.
    rerun = run_index(embedder, sparse, index, pg_session_factory, batch_size=4)
    assert rerun.filings_embedded == 0
    assert rerun.points == 0
    assert index.count() == child_count
