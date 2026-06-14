"""Fetch and store raw filing artifacts; checkpoint the index.

For each company in the index, fetch its CompanyFacts JSON (the numbers plane,
once per CIK). For each filing not yet stored, fetch its primary document (the
narrative plane). Bytes are written to object storage, then the index is updated:
storage first, DB commit last, so a crash mid-fetch is always safe to re-run.

Run: ``uv run python -m scripts.acquire_filings``
"""

from __future__ import annotations

import argparse
import hashlib
from dataclasses import dataclass
from datetime import UTC, datetime

from sqlalchemy.orm import Session, sessionmaker

from app.domain.identifiers import cik_padded
from app.domain.models import StoredArtifact
from app.edgar.client import EdgarClient, EdgartoolsClient
from app.index import repository as repo
from app.index.db import create_all, make_engine, make_session_factory, session_scope
from app.logging import configure_logging, get_logger
from app.storage.base import RawStore
from app.storage.factory import build_raw_store
from app.storage.keys import company_facts_key, primary_document_key
from configs.settings import get_settings

logger = get_logger("scripts.acquire_filings")

_FACTS_URL = "https://data.sec.gov/api/xbrl/companyfacts/CIK{cik}.json"


@dataclass
class AcquireSummary:
    documents_stored: int = 0
    facts_stored: int = 0
    facts_absent: int = 0
    skipped: int = 0
    failed: int = 0


def _sha256(data: bytes) -> str:
    return hashlib.sha256(data).hexdigest()


def _utcnow() -> datetime:
    return datetime.now(UTC)


def _acquire_company_facts(
    client: EdgarClient,
    store: RawStore,
    session_factory: sessionmaker[Session],
    summary: AcquireSummary,
) -> None:
    with session_scope(session_factory) as session:
        ciks = repo.distinct_ciks(session)
    for cik in ciks:
        key = company_facts_key(cik)
        with session_scope(session_factory) as session:
            recorded = repo.artifact_exists(session, "company", str(cik), "company_facts")
        if recorded and store.exists(key):
            summary.skipped += 1
            continue
        data = client.fetch_company_facts(cik)
        if data is None:
            summary.facts_absent += 1
            continue
        uri = store.put(key, data, content_type="application/json")
        artifact = StoredArtifact(
            ref_type="company",
            ref_key=str(cik),
            kind="company_facts",
            storage_uri=uri,
            sha256=_sha256(data),
            size_bytes=len(data),
            content_type="application/json",
            source_url=_FACTS_URL.format(cik=cik_padded(cik)),
            fetched_at=_utcnow(),
        )
        with session_scope(session_factory) as session:
            repo.upsert_artifact(session, artifact)
        summary.facts_stored += 1
        logger.info("stored_company_facts", cik=cik)


def _acquire_documents(
    client: EdgarClient,
    store: RawStore,
    session_factory: sessionmaker[Session],
    summary: AcquireSummary,
) -> None:
    with session_scope(session_factory) as session:
        pending = [
            (f.accession, f.cik, f.primary_document, f.primary_doc_url)
            for f in repo.filings_pending_acquire(session)
        ]
    for accession, cik, primary_document, url in pending:
        key = primary_document_key(cik, accession, primary_document)
        with session_scope(session_factory) as session:
            recorded = repo.artifact_exists(session, "filing", accession, "primary_document")
        if recorded and store.exists(key):
            with session_scope(session_factory) as session:
                repo.mark_filing_stored(session, accession)
            summary.skipped += 1
            continue
        try:
            data = client.fetch_document(url)
            uri = store.put(key, data, content_type="text/html")
            artifact = StoredArtifact(
                ref_type="filing",
                ref_key=accession,
                kind="primary_document",
                storage_uri=uri,
                sha256=_sha256(data),
                size_bytes=len(data),
                content_type="text/html",
                source_url=url,
                fetched_at=_utcnow(),
            )
            with session_scope(session_factory) as session:
                repo.upsert_artifact(session, artifact)
                repo.mark_filing_stored(session, accession)
            summary.documents_stored += 1
            logger.info("stored_document", accession=accession)
        except Exception as exc:
            logger.error("acquire_failed", accession=accession, error=str(exc))
            with session_scope(session_factory) as session:
                repo.mark_filing_failed(session, accession)
            summary.failed += 1


def run_acquire(
    client: EdgarClient,
    store: RawStore,
    session_factory: sessionmaker[Session],
) -> AcquireSummary:
    """Acquire facts and documents for everything pending in the index."""
    summary = AcquireSummary()
    _acquire_company_facts(client, store, session_factory, summary)
    _acquire_documents(client, store, session_factory, summary)
    return summary


def main(argv: list[str] | None = None) -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--log-level", default="INFO")
    args = parser.parse_args(argv)

    configure_logging(args.log_level)
    settings = get_settings()

    engine = make_engine(settings.database_url)
    create_all(engine)
    session_factory = make_session_factory(engine)
    client = EdgartoolsClient.from_settings(settings)
    store = build_raw_store(settings)

    summary = run_acquire(client, store, session_factory)
    logger.info(
        "acquire_complete",
        documents_stored=summary.documents_stored,
        facts_stored=summary.facts_stored,
        facts_absent=summary.facts_absent,
        skipped=summary.skipped,
        failed=summary.failed,
    )
    print(
        f"Stored {summary.documents_stored} documents, {summary.facts_stored} facts; "
        f"skipped {summary.skipped}, absent facts {summary.facts_absent}, "
        f"failed {summary.failed}."
    )
    return 1 if summary.failed else 0


if __name__ == "__main__":
    raise SystemExit(main())
