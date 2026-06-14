"""Query the vector index and print cited chunks (Phase 4 retrieval demo).

Run: ``uv run python -m scripts.search "what are Apple's supply chain risks?" --ticker AAPL``
"""

from __future__ import annotations

import argparse

from app.embedding.factory import build_embedder, build_sparse_embedder
from app.index import repository as repo
from app.index.db import make_engine, make_session_factory, session_scope
from app.logging import configure_logging, get_logger
from app.retrieval.decompose import aliases_from_companies
from app.retrieval.reranker import build_reranker
from app.retrieval.retriever import Retriever
from app.vectorstore.factory import build_qdrant_index
from configs.settings import get_settings

logger = get_logger("scripts.search")


def main(argv: list[str] | None = None) -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("question", help="The question to search for.")
    parser.add_argument("--ticker", default=None, help="Restrict to one company.")
    parser.add_argument("--form", default=None, help="Restrict to a form, e.g. 10-K.")
    parser.add_argument("--section", default=None, help="Restrict to a section, e.g. 'Item 1A'.")
    parser.add_argument("--limit", type=int, default=None, help="Results to return.")
    parser.add_argument("--log-level", default="WARNING")
    args = parser.parse_args(argv)

    configure_logging(args.log_level)
    settings = get_settings()

    engine = make_engine(settings.database_url)
    session_factory = make_session_factory(engine)
    with session_scope(session_factory) as session:
        aliases = aliases_from_companies(repo.all_companies(session))

    retriever = Retriever(
        embedder=build_embedder(settings),
        sparse_embedder=build_sparse_embedder(settings),
        index=build_qdrant_index(settings),
        reranker=build_reranker(settings.reranker_backend, settings.reranker_model),
        top_k=settings.retrieval_top_k,
        top_n=settings.rerank_top_n,
        aliases=aliases,
    )

    results = retriever.retrieve(
        args.question,
        ticker=args.ticker,
        form=args.form,
        section=args.section,
        limit=args.limit,
    )

    if not results:
        print("No results.")
        return 0
    for rank, chunk in enumerate(results, start=1):
        excerpt = chunk.text[:240].replace("\n", " ")
        print(
            f"\n[{rank}] score={chunk.score:.3f} {chunk.ticker} {chunk.form} "
            f"{chunk.section} ({chunk.accession} chars {chunk.char_start}-{chunk.char_end})"
        )
        print(f"    {excerpt}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
