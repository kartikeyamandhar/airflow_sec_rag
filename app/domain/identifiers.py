"""EDGAR identifier parsing, normalization, and validation.

These pure functions are the single place that decides what a valid CIK,
accession number, or document filename looks like. Storage-key derivation depends
on them, which is what keeps attacker-influenced filenames from escaping their
intended prefix (path traversal). See Phase 1 security review.
"""

from __future__ import annotations

import re

# Canonical accession form: 10 digits, 2 digits, 6 digits, dash-separated.
_ACCESSION_DASHED = re.compile(r"^\d{10}-\d{2}-\d{6}$")
# A conservative allowlist for stored document filenames. No path separators, no
# parent-dir tokens, no surprises.
_SAFE_DOCUMENT_NAME = re.compile(r"^[A-Za-z0-9][A-Za-z0-9._-]*$")


def cik_padded(cik: int) -> str:
    """Return the CIK zero-padded to 10 digits, as EDGAR URLs require."""
    if cik < 0:
        raise ValueError(f"CIK must be non-negative, got {cik}")
    if cik > 9_999_999_999:
        raise ValueError(f"CIK exceeds 10 digits: {cik}")
    return f"{cik:010d}"


def normalize_accession(raw: str) -> str:
    """Normalize an accession number to the canonical dashed form.

    Accepts the dashed form (``0000320193-23-000106``) or the bare 18-digit form
    (``000032019323000106``). Raises ``ValueError`` on anything else.
    """
    digits = raw.replace("-", "")
    if len(digits) != 18 or not digits.isdigit():
        raise ValueError(f"Invalid accession number: {raw!r}")
    return f"{digits[0:10]}-{digits[10:12]}-{digits[12:18]}"


def accession_no_dashes(accession: str) -> str:
    """Return the accession in the no-dash form used in EDGAR archive paths."""
    canonical = normalize_accession(accession)
    return canonical.replace("-", "")


def is_dashed_accession(value: str) -> bool:
    """True if ``value`` is already in canonical dashed accession form."""
    return bool(_ACCESSION_DASHED.match(value))


def validate_document_name(name: str) -> str:
    """Validate a filing document filename for safe use as a storage key segment.

    Rejects empty names, path separators, parent-dir tokens, and any character
    outside a conservative allowlist. Returns the name unchanged if valid.
    """
    if not name or "/" in name or "\\" in name or ".." in name:
        raise ValueError(f"Unsafe document name: {name!r}")
    if not _SAFE_DOCUMENT_NAME.match(name):
        raise ValueError(f"Document name has disallowed characters: {name!r}")
    return name
