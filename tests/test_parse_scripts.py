"""End-to-end parse CLI: stored artifacts to facts and chunks, idempotently."""

import json
from datetime import UTC, date, datetime
from pathlib import Path
from typing import Any

import pytest
from sqlalchemy import select
from sqlalchemy.orm import Session, sessionmaker

from app.domain.models import Company, FilingRef, StoredArtifact
from app.index import repository as repo
from app.index.db import session_scope
from app.index.models import STATUS_PARSED
from app.index.models import Filing as FilingRow
from app.storage.keys import company_facts_key, primary_document_key
from app.storage.local import LocalRawStore
from configs.settings import Settings
from scripts.parse_filings import run_parse

_ACCESSION = "0000320193-23-000106"
_CIK = 320193
_DOC = "aapl.htm"

_HTML = (
    b"<html><body><p>Item 1. Business</p><p>"
    + (b"We make great products. " * 50)
    + b"</p></body></html>"
)

_FACTS: dict[str, Any] = {
    "cik": _CIK,
    "facts": {
        "us-gaap": {
            "Revenues": {
                "label": "Revenues",
                "units": {
                    "USD": [
                        {
                            "start": "2022-09-25",
                            "end": "2023-09-30",
                            "val": 383285000000,
                            "accn": _ACCESSION,
                            "fy": 2023,
                            "fp": "FY",
                            "form": "10-K",
                        }
                    ]
                },
            }
        },
        "dei": {
            "EntityCommonStockSharesOutstanding": {
                "label": "Shares",
                "units": {
                    "shares": [
                        {
                            "end": "2023-10-20",
                            "val": 15634232000,
                            "accn": _ACCESSION,
                            "fy": 2023,
                            "fp": "FY",
                            "form": "10-K",
                        }
                    ]
                },
            }
        },
    },
}


def _seed_stored(session_factory: sessionmaker[Session]) -> None:
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
                primary_document=_DOC,
                primary_doc_url="https://www.sec.gov/x/aapl.htm",
            ),
        )
        repo.mark_filing_stored(session, _ACCESSION)
        repo.upsert_artifact(
            session,
            StoredArtifact(
                ref_type="company",
                ref_key=str(_CIK),
                kind="company_facts",
                storage_uri="s3://x",
                sha256="a" * 64,
                size_bytes=10,
                content_type="application/json",
                source_url="https://x",
                fetched_at=datetime(2024, 1, 1, tzinfo=UTC),
            ),
        )


def test_parse_then_reparse_is_idempotent(
    pg_session_factory: sessionmaker[Session],
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    monkeypatch.chdir(tmp_path)  # avoid the real .env; use Settings defaults
    settings = Settings()
    store = LocalRawStore(tmp_path)
    store.put(primary_document_key(_CIK, _ACCESSION, _DOC), _HTML, content_type="text/html")
    store.put(
        company_facts_key(_CIK),
        json.dumps(_FACTS).encode("utf-8"),
        content_type="application/json",
    )
    _seed_stored(pg_session_factory)

    summary = run_parse(store, pg_session_factory, settings)
    assert summary.companies_facts_parsed == 1
    assert summary.facts_total == 2
    assert summary.filings_parsed == 1
    assert summary.chunks_total >= 2

    with pg_session_factory() as session:
        filing = session.scalar(select(FilingRow).where(FilingRow.accession == _ACCESSION))
        assert filing is not None
        assert filing.status == STATUS_PARSED
        assert repo.numeric_fact_count(session) == 2
        assert repo.chunk_count_for_accession(session, _ACCESSION) >= 2

    rerun = run_parse(store, pg_session_factory, settings)
    assert rerun.filings_parsed == 0  # parsed filing is no longer pending
    assert rerun.companies_facts_parsed == 0  # facts already present
    assert rerun.skipped >= 1
    with pg_session_factory() as session:
        assert repo.numeric_fact_count(session) == 2  # not duplicated
