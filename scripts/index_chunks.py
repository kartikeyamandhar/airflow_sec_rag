"""Embed parsed chunks (dense + sparse) and load them into the Qdrant index.

For each parsed filing, embed its child chunks with both the dense (bge) and sparse
(BM25) embedders and upsert them into Qdrant with their metadata payload, then
advance status to ``embedded``. Idempotent: point ids are derived from (accession,
chunk_index), so re-indexing overwrites rather than duplicates.

Run: ``uv run python -m scripts.index_chunks``
"""

from __future__ import annotations

import argparse
from dataclasses import dataclass

from sqlalchemy.orm import Session, sessionmaker

from app.embedding.base import Embedder
from app.embedding.factory import build_embedder, build_sparse_embedder
from app.embedding.sparse import SparseEmbedder, SparseVector
from app.index import repository as repo
from app.index.db import create_all, make_engine, make_session_factory, session_scope
from app.logging import configure_logging, get_logger
from app.vectorstore.factory import build_qdrant_index
from app.vectorstore.qdrant_index import ChunkPoint, QdrantIndex
from configs.settings import get_settings

logger = get_logger("scripts.index_chunks")


@dataclass
class IndexSummary:
    filings_embedded: int = 0
    points: int = 0
    failed: int = 0


def _embed_in_batches(embedder: Embedder, texts: list[str], batch_size: int) -> list[list[float]]:
    vectors: list[list[float]] = []
    for start in range(0, len(texts), batch_size):
        vectors.extend(embedder.embed(texts[start : start + batch_size]))
    return vectors


def _embed_sparse_in_batches(
    embedder: SparseEmbedder, texts: list[str], batch_size: int
) -> list[SparseVector]:
    vectors: list[SparseVector] = []
    for start in range(0, len(texts), batch_size):
        vectors.extend(embedder.embed_sparse(texts[start : start + batch_size]))
    return vectors


def run_index(
    embedder: Embedder,
    sparse_embedder: SparseEmbedder,
    index: QdrantIndex,
    session_factory: sessionmaker[Session],
    *,
    batch_size: int = 64,
) -> IndexSummary:
    """Embed and index all parsed-but-unembedded filings."""
    index.ensure_collection()
    summary = IndexSummary()

    with session_scope(session_factory) as session:
        accessions = [f.accession for f in repo.filings_to_embed(session)]

    for accession in accessions:
        with session_scope(session_factory) as session:
            rows: list[tuple[int, str, dict[str, object]]] = [
                (
                    c.chunk_index,
                    c.text,
                    {
                        "accession": accession,
                        "cik": c.cik,
                        "ticker": c.ticker,
                        "form": c.form,
                        "section": c.section,
                        "chunk_index": c.chunk_index,
                        "parent_index": c.parent_index,
                        "char_start": c.char_start,
                        "char_end": c.char_end,
                        "text": c.text,
                    },
                )
                for c in repo.child_chunks_for_filing(session, accession)
            ]
        try:
            if rows:
                texts = [text for _index, text, _payload in rows]
                dense = _embed_in_batches(embedder, texts, batch_size)
                sparse = _embed_sparse_in_batches(sparse_embedder, texts, batch_size)
                points = [
                    ChunkPoint(
                        accession=accession,
                        chunk_index=chunk_index,
                        dense=dense_vector,
                        sparse=sparse_vector,
                        payload=payload,
                    )
                    for (chunk_index, _text, payload), dense_vector, sparse_vector in zip(
                        rows, dense, sparse, strict=True
                    )
                ]
                index.upsert_chunks(points)
                summary.points += len(points)
            with session_scope(session_factory) as session:
                repo.mark_filing_embedded(session, accession)
            summary.filings_embedded += 1
            logger.info("embedded_filing", accession=accession, points=len(rows))
        except Exception as exc:
            logger.error("embed_failed", accession=accession, error=str(exc))
            with session_scope(session_factory) as session:
                repo.mark_filing_embed_failed(session, accession)
            summary.failed += 1

    return summary


def main(argv: list[str] | None = None) -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--log-level", default="INFO")
    args = parser.parse_args(argv)

    configure_logging(args.log_level)
    settings = get_settings()

    engine = make_engine(settings.database_url)
    create_all(engine)
    session_factory = make_session_factory(engine)
    embedder = build_embedder(settings)
    sparse_embedder = build_sparse_embedder(settings)
    index = build_qdrant_index(settings)

    summary = run_index(
        embedder,
        sparse_embedder,
        index,
        session_factory,
        batch_size=settings.embedding_batch_size,
    )
    logger.info(
        "index_complete",
        filings_embedded=summary.filings_embedded,
        points=summary.points,
        failed=summary.failed,
    )
    print(
        f"Embedded {summary.filings_embedded} filings into {summary.points} vectors; "
        f"failed {summary.failed}."
    )
    return 1 if summary.failed else 0


if __name__ == "__main__":
    raise SystemExit(main())
