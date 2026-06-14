"""The retriever: embed, hybrid-search (filtered), rerank, and merge sub-queries."""

from __future__ import annotations

from dataclasses import dataclass, replace

from qdrant_client import models

from app.embedding.base import Embedder
from app.embedding.sparse import SparseEmbedder
from app.retrieval.decompose import SubQuery, decompose
from app.retrieval.filters import build_filter
from app.retrieval.reranker import Reranker
from app.vectorstore.qdrant_index import QdrantIndex


@dataclass(frozen=True)
class RetrievedChunk:
    """A retrieved chunk with its citation metadata and relevance score."""

    accession: str
    cik: int
    ticker: str | None
    form: str
    section: str
    chunk_index: int
    parent_index: int | None
    char_start: int
    char_end: int
    text: str
    score: float


class Retriever:
    """Hybrid retrieval with metadata filtering, reranking, and decomposition."""

    def __init__(
        self,
        *,
        embedder: Embedder,
        sparse_embedder: SparseEmbedder,
        index: QdrantIndex,
        reranker: Reranker,
        top_k: int = 20,
        top_n: int = 5,
        aliases: dict[str, str] | None = None,
    ) -> None:
        self._embedder = embedder
        self._sparse = sparse_embedder
        self._index = index
        self._reranker = reranker
        self._top_k = top_k
        self._top_n = top_n
        self._aliases = aliases or {}

    def retrieve(
        self,
        question: str,
        *,
        ticker: str | None = None,
        form: str | None = None,
        section: str | None = None,
        limit: int | None = None,
    ) -> list[RetrievedChunk]:
        """Return the top reranked chunks for the question, merged across sub-queries.

        An explicit ``ticker`` pins the search to one company; otherwise the question
        is decomposed by the companies it names.
        """
        top_n = limit if limit is not None else self._top_n
        if ticker is not None:
            sub_queries = [SubQuery(text=question, ticker=ticker)]
        else:
            sub_queries = decompose(question, self._aliases)

        results: list[RetrievedChunk] = []
        for sub_query in sub_queries:
            query_filter = build_filter(ticker=sub_query.ticker, form=form, section=section)
            dense = self._embedder.embed([sub_query.text])[0]
            sparse = self._sparse.embed_sparse([sub_query.text])[0]
            points = self._index.hybrid_search(
                dense, sparse, limit=self._top_k, query_filter=query_filter
            )
            candidates = [self._to_chunk(point) for point in points]
            results.extend(self._rerank(sub_query.text, candidates)[:top_n])
        return results

    def _rerank(self, query: str, candidates: list[RetrievedChunk]) -> list[RetrievedChunk]:
        if not candidates:
            return []
        scores = self._reranker.rerank(query, [c.text for c in candidates])
        ordered = sorted(
            zip(candidates, scores, strict=True), key=lambda pair: pair[1], reverse=True
        )
        return [replace(chunk, score=score) for chunk, score in ordered]

    def _to_chunk(self, point: models.ScoredPoint) -> RetrievedChunk:
        payload = point.payload or {}
        ticker = payload.get("ticker")
        parent_index = payload.get("parent_index")
        return RetrievedChunk(
            accession=str(payload.get("accession", "")),
            cik=int(payload.get("cik", 0)),
            ticker=str(ticker) if ticker is not None else None,
            form=str(payload.get("form", "")),
            section=str(payload.get("section", "")),
            chunk_index=int(payload.get("chunk_index", 0)),
            parent_index=int(parent_index) if parent_index is not None else None,
            char_start=int(payload.get("char_start", 0)),
            char_end=int(payload.get("char_end", 0)),
            text=str(payload.get("text", "")),
            score=float(point.score),
        )
