"""Discover target filings for the configured universe and record them.

Idempotent: resolves each ticker, enumerates its filings in the configured
window, and upserts companies and filings into the index with status
``discovered``. Re-running adds nothing new and never downgrades a stored filing.

Run: ``uv run python -m scripts.discover_filings --config configs/universe.dev.yaml``
"""

from __future__ import annotations

import argparse
from dataclasses import dataclass, field
from pathlib import Path

from sqlalchemy.orm import Session, sessionmaker

from app.edgar.client import EdgarClient, EdgartoolsClient
from app.edgar.discovery import discover
from app.index import repository as repo
from app.index.db import create_all, make_engine, make_session_factory, session_scope
from app.logging import configure_logging, get_logger
from configs.run_config import RunConfig, load_run_config
from configs.settings import get_settings

logger = get_logger("scripts.discover_filings")


@dataclass
class DiscoverSummary:
    companies: int = 0
    filings: int = 0
    unknown_tickers: list[str] = field(default_factory=list)


def run_discovery(
    client: EdgarClient,
    session_factory: sessionmaker[Session],
    config: RunConfig,
) -> DiscoverSummary:
    """Discover and index filings for every ticker in ``config``."""
    summary = DiscoverSummary()
    for ticker in config.tickers:
        result = discover(
            client,
            ticker,
            forms=config.forms,
            start=config.start_date,
            end=config.end_date,
        )
        if result is None:
            summary.unknown_tickers.append(ticker)
            logger.warning("discovery_unknown_ticker", ticker=ticker)
            continue
        company, refs = result
        with session_scope(session_factory) as session:
            repo.upsert_company(session, company)
            for ref in refs:
                repo.upsert_filing(session, ref)
        summary.companies += 1
        summary.filings += len(refs)
        logger.info("discovered_company", ticker=ticker, cik=company.cik, filings=len(refs))
    return summary


def main(argv: list[str] | None = None) -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--config", type=Path, default=Path("configs/universe.dev.yaml"))
    parser.add_argument("--log-level", default="INFO")
    args = parser.parse_args(argv)

    configure_logging(args.log_level)
    settings = get_settings()
    config = load_run_config(args.config)

    engine = make_engine(settings.database_url)
    create_all(engine)
    session_factory = make_session_factory(engine)
    client = EdgartoolsClient.from_settings(settings)

    summary = run_discovery(client, session_factory, config)
    logger.info(
        "discovery_complete",
        companies=summary.companies,
        filings=summary.filings,
        unknown=summary.unknown_tickers,
    )
    print(
        f"Discovered {summary.filings} filings across {summary.companies} companies. "
        f"Unknown tickers: {summary.unknown_tickers or 'none'}."
    )
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
