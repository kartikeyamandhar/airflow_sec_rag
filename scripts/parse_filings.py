"""Parse stored raw artifacts into structured facts and chunks.

For each company with stored CompanyFacts, parse the JSON into numeric facts (once
per CIK). For each stored-but-unparsed filing, extract and chunk its primary
document and advance status to ``parsed``. Idempotent: facts are skipped if already
present, parsed filings drop out of the queue, and re-parse replaces rows cleanly.

Run: ``uv run python -m scripts.parse_filings``
"""

from __future__ import annotations

import argparse
from dataclasses import dataclass

from sqlalchemy.orm import Session, sessionmaker

from app.index import repository as repo
from app.index.db import create_all, make_engine, make_session_factory, session_scope
from app.logging import configure_logging, get_logger
from app.narrative.chunk import chunk_sections
from app.narrative.extract import html_to_text, segment_sections
from app.storage.base import RawStore
from app.storage.factory import build_raw_store
from app.storage.keys import company_facts_key, primary_document_key
from app.xbrl.parse import parse_company_facts
from configs.settings import Settings, get_settings

logger = get_logger("scripts.parse_filings")


@dataclass
class ParseSummary:
    companies_facts_parsed: int = 0
    facts_total: int = 0
    filings_parsed: int = 0
    chunks_total: int = 0
    skipped: int = 0
    failed: int = 0


def _parse_facts(
    store: RawStore,
    session_factory: sessionmaker[Session],
    summary: ParseSummary,
) -> None:
    with session_scope(session_factory) as session:
        ciks = repo.ciks_with_facts_artifact(session)
    for cik in ciks:
        with session_scope(session_factory) as session:
            already = repo.numeric_fact_count_for_cik(session, cik) > 0
        if already:
            summary.skipped += 1
            continue
        key = company_facts_key(cik)
        try:
            data = store.get(key)
        except Exception as exc:
            logger.warning("facts_bytes_missing", cik=cik, error=str(exc))
            continue
        facts = parse_company_facts(data)
        with session_scope(session_factory) as session:
            repo.replace_numeric_facts(session, cik, facts)
        summary.companies_facts_parsed += 1
        summary.facts_total += len(facts)
        logger.info("parsed_company_facts", cik=cik, facts=len(facts))


def _parse_documents(
    store: RawStore,
    session_factory: sessionmaker[Session],
    settings: Settings,
    summary: ParseSummary,
) -> None:
    with session_scope(session_factory) as session:
        pending = [
            (f.accession, f.cik, f.ticker, f.form, f.primary_document)
            for f in repo.filings_to_parse(session)
        ]
    for accession, cik, ticker, form, primary_document in pending:
        key = primary_document_key(cik, accession, primary_document)
        try:
            data = store.get(key)
            text = html_to_text(data, max_bytes=settings.parse_max_document_bytes)
            sections = segment_sections(text, min_section_chars=settings.chunk_min_section_chars)
            chunks = chunk_sections(
                sections,
                accession=accession,
                cik=cik,
                ticker=ticker,
                form=form,
                child_tokens=settings.chunk_child_tokens,
                overlap_tokens=settings.chunk_overlap_tokens,
            )
            with session_scope(session_factory) as session:
                repo.replace_chunks(session, accession, chunks)
                repo.mark_filing_parsed(session, accession)
            summary.filings_parsed += 1
            summary.chunks_total += len(chunks)
            logger.info(
                "parsed_filing",
                accession=accession,
                sections=len(sections),
                chunks=len(chunks),
            )
        except Exception as exc:
            logger.error("parse_failed", accession=accession, error=str(exc))
            with session_scope(session_factory) as session:
                repo.mark_filing_parse_failed(session, accession)
            summary.failed += 1


def run_parse(
    store: RawStore,
    session_factory: sessionmaker[Session],
    settings: Settings,
) -> ParseSummary:
    """Parse facts and documents for everything pending in the index."""
    summary = ParseSummary()
    _parse_facts(store, session_factory, summary)
    _parse_documents(store, session_factory, settings, summary)
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
    store = build_raw_store(settings)

    summary = run_parse(store, session_factory, settings)
    logger.info(
        "parse_complete",
        companies_facts_parsed=summary.companies_facts_parsed,
        facts_total=summary.facts_total,
        filings_parsed=summary.filings_parsed,
        chunks_total=summary.chunks_total,
        skipped=summary.skipped,
        failed=summary.failed,
    )
    print(
        f"Parsed {summary.filings_parsed} filings into {summary.chunks_total} chunks; "
        f"{summary.facts_total} facts across {summary.companies_facts_parsed} companies; "
        f"skipped {summary.skipped}, failed {summary.failed}."
    )
    return 1 if summary.failed else 0


if __name__ == "__main__":
    raise SystemExit(main())
