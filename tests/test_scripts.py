"""End-to-end discovery and acquisition against a fake EDGAR client.

Uses a real Postgres (testcontainers) and a local filesystem store, but no
network. Proves the index transitions and that a re-run is idempotent/resumable.
"""

from datetime import date
from pathlib import Path

from sqlalchemy import func, select
from sqlalchemy.orm import Session, sessionmaker

from app.domain.models import Company, FilingRef
from app.index.models import STATUS_STORED
from app.index.models import Artifact as ArtifactRow
from app.index.models import Filing as FilingRow
from app.storage.local import LocalRawStore
from configs.run_config import RunConfig
from scripts.acquire_filings import run_acquire
from scripts.discover_filings import run_discovery
from tests.fakes import FakeEdgarClient

_ACCESSION = "0000320193-23-000106"
_DOC_URL = "https://www.sec.gov/Archives/edgar/data/320193/000032019323000106/aapl.htm"


def _fake_client() -> FakeEdgarClient:
    company = Company(cik=320193, ticker="AAPL", name="Apple Inc.")
    ref = FilingRef(
        cik=320193,
        ticker="AAPL",
        accession=_ACCESSION,
        form="10-K",
        filing_date=date(2023, 11, 3),
        report_date=date(2023, 9, 30),
        primary_document="aapl.htm",
        primary_doc_url=_DOC_URL,
    )
    return FakeEdgarClient(
        companies={"AAPL": company},
        filings={320193: [ref]},
        documents={_DOC_URL: b"<html>10-K</html>"},
        facts={320193: b'{"facts": true}'},
    )


def test_discover_then_acquire_is_idempotent(
    pg_session_factory: sessionmaker[Session], tmp_path: Path
) -> None:
    client = _fake_client()
    config = RunConfig(
        tickers=["AAPL"],
        forms=["10-K"],
        start_date=date(2023, 1, 1),
        end_date=date(2023, 12, 31),
    )
    store = LocalRawStore(tmp_path)

    discovery = run_discovery(client, pg_session_factory, config)
    assert discovery.companies == 1
    assert discovery.filings == 1

    acquire = run_acquire(client, store, pg_session_factory)
    assert acquire.documents_stored == 1
    assert acquire.facts_stored == 1
    assert acquire.failed == 0

    with pg_session_factory() as session:
        filing = session.scalar(select(FilingRow).where(FilingRow.accession == _ACCESSION))
        assert filing is not None
        assert filing.status == STATUS_STORED
        assert session.scalar(select(func.count()).select_from(ArtifactRow)) == 2

    assert store.exists("filings/0000320193/000032019323000106/aapl.htm")
    assert store.get("companyfacts/0000320193.json") == b'{"facts": true}'

    # Re-run acquisition: nothing new is fetched or stored. The stored filing has
    # dropped out of the pending query entirely (so it is not re-iterated); only
    # the per-CIK facts check is skipped after confirming the artifact still exists.
    rerun = run_acquire(client, store, pg_session_factory)
    assert rerun.documents_stored == 0
    assert rerun.facts_stored == 0
    assert rerun.skipped == 1
    with pg_session_factory() as session:
        filing = session.scalar(select(FilingRow).where(FilingRow.accession == _ACCESSION))
        assert filing is not None
        assert filing.status == STATUS_STORED
        assert session.scalar(select(func.count()).select_from(ArtifactRow)) == 2


def test_discovery_reports_unknown_ticker(
    pg_session_factory: sessionmaker[Session],
) -> None:
    config = RunConfig(
        tickers=["ZZZZ"],
        start_date=date(2023, 1, 1),
        end_date=date(2023, 12, 31),
    )
    summary = run_discovery(FakeEdgarClient(), pg_session_factory, config)
    assert summary.unknown_tickers == ["ZZZZ"]
    assert summary.companies == 0
