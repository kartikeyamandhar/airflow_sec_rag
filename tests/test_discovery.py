"""Discovery orchestration over a fake EDGAR client."""

from datetime import date

from app.domain.models import Company, FilingRef
from app.edgar.client import EdgarClient
from app.edgar.discovery import discover
from tests.fakes import FakeEdgarClient


def _ref(
    accession: str,
    form: str,
    filing_date: date,
    report_date: date | None,
    *,
    is_amendment: bool = False,
) -> FilingRef:
    return FilingRef(
        cik=320193,
        accession=accession,
        form=form,
        filing_date=filing_date,
        report_date=report_date,
        primary_document="doc.htm",
        primary_doc_url="https://www.sec.gov/Archives/edgar/data/320193/x/doc.htm",
        is_amendment=is_amendment,
    )


def test_discover_unknown_ticker_returns_none() -> None:
    client: EdgarClient = FakeEdgarClient()
    result = discover(client, "ZZZZ", forms=["10-K"], start=date(2020, 1, 1), end=date(2024, 1, 1))
    assert result is None


def test_discover_stamps_ticker_and_filters_by_date() -> None:
    company = Company(cik=320193, ticker="AAPL", name="Apple Inc.")
    in_range = _ref("0000320193-23-000106", "10-K", date(2023, 11, 3), date(2023, 9, 30))
    client: EdgarClient = FakeEdgarClient(companies={"AAPL": company}, filings={320193: [in_range]})
    result = discover(
        client, "AAPL", forms=["10-K"], start=date(2023, 1, 1), end=date(2023, 12, 31)
    )
    assert result is not None
    company_out, refs = result
    assert company_out == company
    assert [r.ticker for r in refs] == ["AAPL"]


def test_discover_links_amendment_to_base_filing() -> None:
    company = Company(cik=320193, ticker="AAPL", name="Apple Inc.")
    base = _ref("0000320193-23-000106", "10-K", date(2023, 11, 3), date(2023, 9, 30))
    amendment = _ref(
        "0000320193-24-000010",
        "10-K/A",
        date(2024, 1, 15),
        date(2023, 9, 30),
        is_amendment=True,
    )
    client: EdgarClient = FakeEdgarClient(
        companies={"AAPL": company}, filings={320193: [base, amendment]}
    )
    result = discover(
        client,
        "AAPL",
        forms=["10-K", "10-K/A"],
        start=date(2023, 1, 1),
        end=date(2024, 12, 31),
    )
    assert result is not None
    _, refs = result
    amend_ref = next(r for r in refs if r.is_amendment)
    assert amend_ref.amends_accession == "0000320193-23-000106"
