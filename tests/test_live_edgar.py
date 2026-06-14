"""Opt-in live smoke test against the real EDGAR API.

Skipped unless ``RUN_LIVE_EDGAR=1``. It exercises the real ``EdgartoolsClient``
path (which the mocked tests cannot), so a developer can confirm the live wiring
with ``RUN_LIVE_EDGAR=1 uv run pytest tests/test_live_edgar.py``. Needs a valid
``EDGAR_IDENTITY`` in the environment. CI does not run it.
"""

import os
from datetime import date

import pytest

from app.edgar.client import EdgartoolsClient
from configs.settings import get_settings

pytestmark = pytest.mark.skipif(
    os.environ.get("RUN_LIVE_EDGAR") != "1",
    reason="set RUN_LIVE_EDGAR=1 to run the live EDGAR smoke test",
)

_APPLE_CIK = 320193


def _client() -> EdgartoolsClient:
    return EdgartoolsClient.from_settings(get_settings())


def test_live_company_facts() -> None:
    data = _client().fetch_company_facts(_APPLE_CIK)
    assert data is not None
    assert len(data) > 0


def test_live_list_filings() -> None:
    refs = _client().list_filings(
        _APPLE_CIK, forms=["10-K"], start=date(2022, 1, 1), end=date(2024, 12, 31)
    )
    assert refs
    assert all(r.form == "10-K" for r in refs)
    assert all(r.primary_doc_url.startswith("https://www.sec.gov/") for r in refs)
