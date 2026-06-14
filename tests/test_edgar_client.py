"""EdgartoolsClient: raw-byte fetching, retries, and filing conversion.

Network is mocked with respx; edgartools' ``Company`` is monkeypatched for the
discovery-conversion test. Real-network behavior is covered by the opt-in live
smoke test (see tests/test_live_edgar.py), not here.
"""

from datetime import date

import httpx
import pytest
import respx

from app.edgar.client import EdgartoolsClient

_IDENTITY = "Tester tester@example.com"


def _client() -> EdgartoolsClient:
    return EdgartoolsClient(
        identity=_IDENTITY,
        max_requests_per_second=1000.0,
        sleep=lambda _seconds: None,
        max_attempts=4,
    )


@respx.mock
def test_fetch_document_returns_bytes() -> None:
    url = "https://www.sec.gov/Archives/edgar/data/320193/x/aapl.htm"
    respx.get(url).mock(return_value=httpx.Response(200, content=b"<html>ok</html>"))
    assert _client().fetch_document(url) == b"<html>ok</html>"


@respx.mock
def test_fetch_company_facts_404_returns_none() -> None:
    url = "https://data.sec.gov/api/xbrl/companyfacts/CIK0000320193.json"
    respx.get(url).mock(return_value=httpx.Response(404))
    assert _client().fetch_company_facts(320193) is None


@respx.mock
def test_fetch_retries_on_429_then_succeeds() -> None:
    url = "https://data.sec.gov/api/xbrl/companyfacts/CIK0000320193.json"
    route = respx.get(url).mock(
        side_effect=[httpx.Response(429), httpx.Response(200, content=b"{}")]
    )
    assert _client().fetch_company_facts(320193) == b"{}"
    assert route.call_count == 2


@respx.mock
def test_fetch_raises_after_exhausting_retries() -> None:
    url = "https://www.sec.gov/x.htm"
    respx.get(url).mock(return_value=httpx.Response(503))
    with pytest.raises(httpx.HTTPStatusError):
        _client().fetch_document(url)


class _FakeFiling:
    def __init__(
        self,
        accession_no: str,
        form: str,
        filing_date: date,
        primary_document: str,
        report_date: date | None,
    ) -> None:
        self.accession_no = accession_no
        self.form = form
        self.filing_date = filing_date
        self.primary_document = primary_document
        self.report_date = report_date


class _FakeEdgarCompany:
    def __init__(self, _ident: object) -> None:
        pass

    def get_filings(self, form: list[str]) -> list[_FakeFiling]:
        return [
            _FakeFiling(
                "0000320193-23-000106",
                "10-K",
                date(2023, 11, 3),
                "aapl.htm",
                date(2023, 9, 30),
            ),
            _FakeFiling("0000320193-19-000100", "10-K", date(2019, 1, 1), "old.htm", None),
        ]


def test_list_filings_filters_by_date_and_builds_url(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    monkeypatch.setattr("app.edgar.client.edgar.Company", _FakeEdgarCompany)
    refs = _client().list_filings(
        320193, forms=["10-K"], start=date(2023, 1, 1), end=date(2023, 12, 31)
    )
    assert len(refs) == 1
    ref = refs[0]
    assert ref.accession == "0000320193-23-000106"
    assert ref.report_date == date(2023, 9, 30)
    assert ref.primary_doc_url == (
        "https://www.sec.gov/Archives/edgar/data/320193/000032019323000106/aapl.htm"
    )
