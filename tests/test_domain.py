"""Domain identifier helpers and model validation."""

from datetime import date

import pytest

from app.domain.identifiers import (
    accession_no_dashes,
    cik_padded,
    is_dashed_accession,
    normalize_accession,
    validate_document_name,
)
from app.domain.models import Company, FilingRef


def test_cik_padded() -> None:
    assert cik_padded(320193) == "0000320193"


def test_cik_padded_rejects_negative() -> None:
    with pytest.raises(ValueError):
        cik_padded(-1)


def test_normalize_accession_from_bare() -> None:
    assert normalize_accession("000032019323000106") == "0000320193-23-000106"


def test_normalize_accession_from_dashed() -> None:
    assert normalize_accession("0000320193-23-000106") == "0000320193-23-000106"


@pytest.mark.parametrize("bad", ["123", "0000320193-23-00010", "abcdefghij-23-000106"])
def test_normalize_accession_rejects_bad(bad: str) -> None:
    with pytest.raises(ValueError):
        normalize_accession(bad)


def test_accession_no_dashes() -> None:
    assert accession_no_dashes("0000320193-23-000106") == "000032019323000106"


def test_is_dashed_accession() -> None:
    assert is_dashed_accession("0000320193-23-000106") is True
    assert is_dashed_accession("000032019323000106") is False


@pytest.mark.parametrize("bad", ["../etc/passwd", "a/b", "a\\b", "", "..", "a b.htm"])
def test_validate_document_name_rejects(bad: str) -> None:
    with pytest.raises(ValueError):
        validate_document_name(bad)


def test_validate_document_name_accepts() -> None:
    assert validate_document_name("aapl-20230930.htm") == "aapl-20230930.htm"


def test_filing_ref_normalizes_accession() -> None:
    ref = FilingRef(
        cik=320193,
        accession="000032019323000106",
        form="10-K",
        filing_date=date(2023, 11, 3),
        primary_document="aapl.htm",
        primary_doc_url="https://www.sec.gov/Archives/edgar/data/320193/x/aapl.htm",
    )
    assert ref.accession == "0000320193-23-000106"


def test_company_rejects_negative_cik() -> None:
    with pytest.raises(ValueError):
        Company(cik=-1, ticker="X", name="Y")
