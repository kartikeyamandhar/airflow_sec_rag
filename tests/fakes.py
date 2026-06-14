"""In-memory test doubles implementing the project's Protocols."""

from __future__ import annotations

import hashlib
from collections.abc import Sequence
from datetime import date

from app.domain.models import Company, FilingRef
from app.embedding.sparse import SparseVector


class FakeEdgarClient:
    """An in-memory ``EdgarClient`` driven by fixture data (no network)."""

    def __init__(
        self,
        *,
        companies: dict[str, Company] | None = None,
        filings: dict[int, list[FilingRef]] | None = None,
        documents: dict[str, bytes] | None = None,
        facts: dict[int, bytes | None] | None = None,
    ) -> None:
        self._companies = companies or {}
        self._filings = filings or {}
        self._documents = documents or {}
        self._facts = facts or {}

    def resolve_company(self, ticker: str) -> Company | None:
        return self._companies.get(ticker.upper())

    def list_filings(
        self, cik: int, *, forms: Sequence[str], start: date, end: date
    ) -> list[FilingRef]:
        wanted = set(forms)
        return [
            f
            for f in self._filings.get(cik, [])
            if f.form in wanted and start <= f.filing_date <= end
        ]

    def fetch_document(self, url: str) -> bytes:
        return self._documents[url]

    def fetch_company_facts(self, cik: int) -> bytes | None:
        return self._facts.get(cik)


class FakeEmbedder:
    """A deterministic ``Embedder`` for tests (no model, no network)."""

    def __init__(self, dimension: int = 8) -> None:
        self._dimension = dimension

    @property
    def dimension(self) -> int:
        return self._dimension

    def embed(self, texts: list[str]) -> list[list[float]]:
        return [self._vector(text) for text in texts]

    def _vector(self, text: str) -> list[float]:
        digest = hashlib.sha256(text.encode("utf-8")).digest()
        return [digest[i % len(digest)] / 255.0 for i in range(self._dimension)]


class FakeSparseEmbedder:
    """A deterministic ``SparseEmbedder`` for tests (term ids from word hashes)."""

    def embed_sparse(self, texts: list[str]) -> list[SparseVector]:
        return [self._sparse(text) for text in texts]

    def _sparse(self, text: str) -> SparseVector:
        indices = sorted(
            {
                int(hashlib.sha256(word.encode("utf-8")).hexdigest(), 16) % 1000
                for word in text.split()[:16]
            }
        )
        return SparseVector(indices=indices, values=[1.0] * len(indices))


class FakeReranker:
    """A deterministic ``Reranker`` scoring by query/document word overlap."""

    def rerank(self, query: str, documents: list[str]) -> list[float]:
        query_words = set(query.lower().split())
        return [float(len(query_words & set(doc.lower().split()))) for doc in documents]
