"""Parse EDGAR CompanyFacts JSON into typed numeric facts.

CompanyFacts shape (abridged)::

    {"cik": 320193, "entityName": "Apple Inc.",
     "facts": {"us-gaap": {"Revenues": {"label": "...",
        "units": {"USD": [{"start": "...", "end": "...", "val": 1, "accn": "...",
                           "fy": 2023, "fp": "FY", "form": "10-K", "frame": "..."}]}}}}}

Bad or incomplete entries are skipped rather than raising, so one malformed fact
cannot abort ingestion of a whole company.
"""

from __future__ import annotations

import json
from datetime import date
from typing import Any

from pydantic import ValidationError

from app.domain.models import NumericFact
from app.logging import get_logger

logger = get_logger("app.xbrl.parse")


def parse_company_facts(data: bytes) -> list[NumericFact]:
    """Parse CompanyFacts JSON bytes into a list of numeric facts."""
    payload: Any = json.loads(data)
    if not isinstance(payload, dict):
        return []
    cik = _as_int(payload.get("cik"))
    if cik is None:
        return []
    facts_section = payload.get("facts")
    if not isinstance(facts_section, dict):
        return []

    results: list[NumericFact] = []
    for taxonomy, concepts in facts_section.items():
        if not isinstance(concepts, dict):
            continue
        for concept, body in concepts.items():
            if not isinstance(body, dict):
                continue
            label = body.get("label")
            units = body.get("units")
            if not isinstance(units, dict):
                continue
            for unit, entries in units.items():
                if not isinstance(entries, list):
                    continue
                for entry in entries:
                    fact = _entry_to_fact(cik, str(taxonomy), str(concept), label, str(unit), entry)
                    if fact is not None:
                        results.append(fact)
    return results


def _entry_to_fact(
    cik: int,
    taxonomy: str,
    concept: str,
    label: Any,
    unit: str,
    entry: Any,
) -> NumericFact | None:
    if not isinstance(entry, dict):
        return None
    end = entry.get("end")
    val = entry.get("val")
    accn = entry.get("accn")
    if end is None or val is None or accn is None:
        return None
    try:
        period_end = date.fromisoformat(str(end))
        start_raw = entry.get("start")
        period_start = date.fromisoformat(str(start_raw)) if start_raw else None
        value = float(val)
    except (ValueError, TypeError):
        return None

    frame = entry.get("frame")
    fp = entry.get("fp")
    try:
        return NumericFact(
            cik=cik,
            taxonomy=taxonomy,
            concept=concept,
            label=str(label) if label is not None else None,
            unit=unit,
            value=value,
            period_start=period_start,
            period_end=period_end,
            fiscal_year=_as_int(entry.get("fy")),
            fiscal_period=str(fp) if fp is not None else None,
            form=str(entry.get("form", "")),
            accession=str(accn),
            frame=str(frame) if frame is not None else None,
        )
    except ValidationError:
        return None


def _as_int(value: Any) -> int | None:
    if isinstance(value, bool):
        return None
    if isinstance(value, int):
        return value
    if isinstance(value, float):
        return int(value)
    return None
