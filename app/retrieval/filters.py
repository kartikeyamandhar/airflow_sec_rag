"""Build Qdrant metadata filters from typed constraints.

Only typed fields ever reach the filter; raw query text is never used to build
filter logic. A missing constraint simply widens the search.
"""

from __future__ import annotations

from qdrant_client import models


def build_filter(
    *,
    ticker: str | None = None,
    form: str | None = None,
    section: str | None = None,
    cik: int | None = None,
) -> models.Filter | None:
    """Return a Qdrant filter matching all provided constraints, or None."""
    conditions: list[models.FieldCondition] = []
    if ticker is not None:
        conditions.append(
            models.FieldCondition(key="ticker", match=models.MatchValue(value=ticker))
        )
    if form is not None:
        conditions.append(models.FieldCondition(key="form", match=models.MatchValue(value=form)))
    if section is not None:
        conditions.append(
            models.FieldCondition(key="section", match=models.MatchValue(value=section))
        )
    if cik is not None:
        conditions.append(models.FieldCondition(key="cik", match=models.MatchValue(value=cik)))
    if not conditions:
        return None
    return models.Filter(must=list(conditions))
