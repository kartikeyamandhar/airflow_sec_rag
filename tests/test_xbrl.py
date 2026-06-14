"""CompanyFacts JSON parsing into numeric facts."""

import json
from datetime import date
from typing import Any

from app.xbrl.parse import parse_company_facts


def _encode(payload: dict[str, Any]) -> bytes:
    return json.dumps(payload).encode("utf-8")


def test_parse_company_facts_basic() -> None:
    payload = {
        "cik": 320193,
        "entityName": "Apple Inc.",
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
                                "accn": "0000320193-23-000106",
                                "fy": 2023,
                                "fp": "FY",
                                "form": "10-K",
                                "frame": "CY2023",
                            }
                        ]
                    },
                }
            },
            "dei": {
                "EntityCommonStockSharesOutstanding": {
                    "label": "Shares Outstanding",
                    "units": {
                        "shares": [
                            {
                                "end": "2023-10-20",
                                "val": 15634232000,
                                "accn": "0000320193-23-000106",
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
    facts = parse_company_facts(_encode(payload))
    assert len(facts) == 2

    revenue = next(f for f in facts if f.concept == "Revenues")
    assert revenue.taxonomy == "us-gaap"
    assert revenue.unit == "USD"
    assert revenue.value == 383285000000.0
    assert revenue.period_start == date(2022, 9, 25)
    assert revenue.period_end == date(2023, 9, 30)
    assert revenue.fiscal_year == 2023
    assert revenue.accession == "0000320193-23-000106"

    shares = next(f for f in facts if f.taxonomy == "dei")
    assert shares.period_start is None  # instant fact has no start
    assert shares.unit == "shares"


def test_parse_skips_incomplete_entries() -> None:
    payload = {
        "cik": 1,
        "facts": {
            "us-gaap": {
                "X": {
                    "units": {
                        "USD": [
                            {"val": 1},  # missing end and accn
                            {"end": "2023-01-01", "accn": "0000000001-23-000001"},  # no val
                        ]
                    }
                }
            }
        },
    }
    assert parse_company_facts(_encode(payload)) == []


def test_parse_empty_inputs() -> None:
    assert parse_company_facts(b"{}") == []
    assert parse_company_facts(b'{"cik": 1, "facts": {}}') == []
