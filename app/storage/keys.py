"""Derivation of object-storage keys from validated identifiers.

Keys are derived, never taken raw from EDGAR. The identifier helpers validate
CIK, accession, and document name first, so a hostile filename cannot escape its
prefix. Layout::

    filings/{cik:010d}/{accession_no_dashes}/{document}
    companyfacts/{cik:010d}.json
"""

from __future__ import annotations

from app.domain.identifiers import (
    accession_no_dashes,
    cik_padded,
    validate_document_name,
)


def primary_document_key(cik: int, accession: str, document: str) -> str:
    """Storage key for a filing's primary document (the narrative plane)."""
    return (
        f"filings/{cik_padded(cik)}/{accession_no_dashes(accession)}/"
        f"{validate_document_name(document)}"
    )


def company_facts_key(cik: int) -> str:
    """Storage key for a company's raw CompanyFacts JSON (the numbers plane)."""
    return f"companyfacts/{cik_padded(cik)}.json"
