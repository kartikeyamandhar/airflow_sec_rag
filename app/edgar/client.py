"""The EDGAR client: discovery via edgartools, raw bytes via polite httpx.

``EdgarClient`` is the Protocol the rest of the system depends on. The concrete
``EdgartoolsClient`` is the only place that performs network IO. Raw byte fetches
go through a rate limiter and a bounded retry loop that backs off on 429/5xx and
transport errors but never retries a 404.
"""

from __future__ import annotations

import time
from collections.abc import Callable, Sequence
from datetime import date, datetime
from typing import Protocol

import edgar
import httpx

from app.domain.identifiers import accession_no_dashes, cik_padded, normalize_accession
from app.domain.models import Company, FilingRef
from app.edgar.ratelimit import RateLimiter
from app.logging import get_logger
from configs.settings import Settings

logger = get_logger("app.edgar.client")

_ARCHIVE_BASE = "https://www.sec.gov/Archives/edgar/data"
_FACTS_BASE = "https://data.sec.gov/api/xbrl/companyfacts"
_RETRYABLE_STATUS = frozenset({429, 500, 502, 503, 504})


class EdgarClient(Protocol):
    """The seam between the pipeline and EDGAR."""

    def resolve_company(self, ticker: str) -> Company | None:
        """Resolve a ticker to a ``Company`` (CIK + name), or None if unknown."""
        ...

    def list_filings(
        self, cik: int, *, forms: Sequence[str], start: date, end: date
    ) -> list[FilingRef]:
        """List filings of the given forms within the inclusive date range."""
        ...

    def fetch_document(self, url: str) -> bytes:
        """Fetch raw document bytes from an EDGAR archive URL."""
        ...

    def fetch_company_facts(self, cik: int) -> bytes | None:
        """Fetch raw CompanyFacts JSON bytes, or None if the company has none."""
        ...


def _as_date(value: object) -> date:
    if isinstance(value, datetime):
        return value.date()
    if isinstance(value, date):
        return value
    if isinstance(value, str) and value:
        return date.fromisoformat(value[:10])
    raise ValueError(f"Cannot parse date from {value!r}")


def _as_optional_date(value: object) -> date | None:
    if value is None or value == "":
        return None
    return _as_date(value)


class EdgartoolsClient:
    """Concrete :class:`EdgarClient` backed by edgartools and httpx."""

    def __init__(
        self,
        *,
        identity: str,
        max_requests_per_second: float = 9.0,
        timeout: float = 30.0,
        http_client: httpx.Client | None = None,
        sleep: Callable[[float], None] = time.sleep,
        max_attempts: int = 5,
    ) -> None:
        if not identity:
            raise ValueError("EDGAR identity is required (set EDGAR_IDENTITY).")
        edgar.set_identity(identity)
        self._rate = RateLimiter(max_requests_per_second)
        self._sleep = sleep
        self._max_attempts = max_attempts
        self._http = http_client or httpx.Client(
            headers={"User-Agent": identity},
            timeout=timeout,
            follow_redirects=True,
        )

    @classmethod
    def from_settings(cls, settings: Settings) -> EdgartoolsClient:
        return cls(
            identity=settings.edgar_identity,
            max_requests_per_second=settings.edgar_max_requests_per_second,
            timeout=settings.edgar_request_timeout_seconds,
        )

    def resolve_company(self, ticker: str) -> Company | None:
        try:
            company = edgar.Company(ticker)
        except Exception:  # edgartools raises various errors for unknown tickers
            logger.warning("ticker_not_resolved", ticker=ticker)
            return None
        cik = getattr(company, "cik", None)
        if cik is None:
            logger.warning("ticker_not_resolved", ticker=ticker)
            return None
        return Company(cik=int(cik), ticker=ticker.upper(), name=str(company.name))

    def list_filings(
        self, cik: int, *, forms: Sequence[str], start: date, end: date
    ) -> list[FilingRef]:
        company = edgar.Company(cik)
        filings = company.get_filings(form=list(forms))
        refs: list[FilingRef] = []
        for filing in filings:
            filing_date = _as_date(filing.filing_date)
            if filing_date < start or filing_date > end:
                continue
            accession = normalize_accession(str(filing.accession_no))
            form = str(filing.form)
            primary_document = str(filing.primary_document)
            url = f"{_ARCHIVE_BASE}/{cik}/{accession_no_dashes(accession)}/{primary_document}"
            refs.append(
                FilingRef(
                    cik=cik,
                    ticker=None,
                    accession=accession,
                    form=form,
                    filing_date=filing_date,
                    report_date=_as_optional_date(getattr(filing, "report_date", None)),
                    primary_document=primary_document,
                    primary_doc_url=url,
                    is_amendment=form.endswith("/A"),
                    amends_accession=None,
                )
            )
        return refs

    def fetch_document(self, url: str) -> bytes:
        return self._get_bytes(url)

    def fetch_company_facts(self, cik: int) -> bytes | None:
        url = f"{_FACTS_BASE}/CIK{cik_padded(cik)}.json"
        try:
            return self._get_bytes(url)
        except httpx.HTTPStatusError as exc:
            if exc.response.status_code == 404:
                logger.info("company_facts_absent", cik=cik)
                return None
            raise

    def _get_bytes(self, url: str) -> bytes:
        last_exc: Exception | None = None
        for attempt in range(self._max_attempts):
            self._rate.acquire()
            try:
                response = self._http.get(url)
                response.raise_for_status()
                return response.content
            except httpx.HTTPStatusError as exc:
                if exc.response.status_code not in _RETRYABLE_STATUS:
                    raise
                last_exc = exc
                logger.warning(
                    "edgar_retryable_status",
                    url=url,
                    status=exc.response.status_code,
                    attempt=attempt + 1,
                )
            except httpx.TransportError as exc:
                last_exc = exc
                logger.warning("edgar_transport_error", url=url, attempt=attempt + 1)
            if attempt + 1 < self._max_attempts:
                self._sleep(_backoff_seconds(attempt))
        assert last_exc is not None  # loop ran at least once
        raise last_exc


def _backoff_seconds(attempt: int) -> float:
    """Exponential backoff: 1s, 2s, 4s, ... capped at 30s."""
    return min(2.0**attempt, 30.0)
