"""Discovery: resolve a ticker and enumerate its target filings.

Pure orchestration over an ``EdgarClient`` (so it is testable with a fake). It
stamps the ticker onto each filing and flags amendment supersedence: an ``X/A``
amendment is linked to the base ``X`` filing covering the same period of report.
That linkage lets later phases prefer the latest version of a filing.
"""

from __future__ import annotations

from collections.abc import Sequence
from datetime import date

from app.domain.models import Company, FilingRef
from app.edgar.client import EdgarClient


def discover(
    client: EdgarClient,
    ticker: str,
    *,
    forms: Sequence[str],
    start: date,
    end: date,
) -> tuple[Company, list[FilingRef]] | None:
    """Resolve ``ticker`` and return its company and filings, or None if unknown."""
    company = client.resolve_company(ticker)
    if company is None:
        return None
    filings = client.list_filings(company.cik, forms=forms, start=start, end=end)
    stamped = [f.model_copy(update={"ticker": company.ticker}) for f in filings]
    return company, _flag_amendments(stamped)


def _base_form(form: str) -> str:
    """Strip an amendment suffix: '10-K/A' -> '10-K'."""
    return form.removesuffix("/A")


def _flag_amendments(filings: list[FilingRef]) -> list[FilingRef]:
    """Link each amendment to the base filing of the same form and period."""
    base_accession: dict[tuple[str, date | None], str] = {}
    for filing in filings:
        if not filing.is_amendment:
            base_accession[(filing.form, filing.report_date)] = filing.accession

    result: list[FilingRef] = []
    for filing in filings:
        if filing.is_amendment:
            key = (_base_form(filing.form), filing.report_date)
            result.append(filing.model_copy(update={"amends_accession": base_accession.get(key)}))
        else:
            result.append(filing)
    return result
