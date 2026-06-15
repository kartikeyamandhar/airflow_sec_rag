"""Ask a grounded question over the filings; print the cited answer or a refusal.

Run: ``uv run python -m scripts.answer "what are Apple's supply chain risks?" --ticker AAPL``
"""

from __future__ import annotations

import argparse
from datetime import date

from app.embedding.factory import build_embedder, build_sparse_embedder
from app.generation.answerer import Answerer
from app.generation.llm import build_llm_client
from app.index import repository as repo
from app.index.db import make_engine, make_session_factory, session_scope
from app.logging import configure_logging, get_logger
from app.retrieval.decompose import aliases_from_companies
from app.retrieval.reranker import build_reranker
from app.retrieval.retriever import Retriever
from app.vectorstore.factory import build_qdrant_index
from configs.settings import get_settings

logger = get_logger("scripts.answer")


def main(argv: list[str] | None = None) -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("question", help="The question to answer.")
    parser.add_argument("--ticker", default=None, help="Restrict to one company.")
    parser.add_argument("--form", default=None, help="Restrict to a form, e.g. 10-K.")
    parser.add_argument("--section", default=None, help="Restrict to a section.")
    parser.add_argument("--log-level", default="WARNING")
    args = parser.parse_args(argv)

    configure_logging(args.log_level)
    settings = get_settings()

    engine = make_engine(settings.database_url)
    session_factory = make_session_factory(engine)
    with session_scope(session_factory) as session:
        aliases = aliases_from_companies(repo.all_companies(session))

    def report_dates(accessions: set[str]) -> dict[str, date]:
        with session_scope(session_factory) as session:
            return repo.report_dates_for_accessions(session, accessions)

    retriever = Retriever(
        embedder=build_embedder(settings),
        sparse_embedder=build_sparse_embedder(settings),
        index=build_qdrant_index(settings),
        reranker=build_reranker(settings.reranker_backend, settings.reranker_model),
        top_k=settings.retrieval_top_k,
        top_n=settings.rerank_top_n,
        aliases=aliases,
    )
    answerer = Answerer(
        retriever=retriever,
        llm=build_llm_client(settings),
        max_tokens=settings.llm_max_tokens,
        context_chunks=settings.answer_context_chunks,
        min_coverage=settings.answer_min_citation_coverage,
        report_date_lookup=report_dates,
    )

    answer = answerer.answer(
        args.question, ticker=args.ticker, form=args.form, section=args.section
    )

    if answer.refused:
        print(f"Not supported by any filing. ({answer.reason})")
        return 0

    print(answer.text)
    print(f"\nConfidence: {answer.confidence:.2f}")
    print("Citations:")
    for citation in answer.citations:
        as_of = f", as of {citation.as_of}" if citation.as_of else ""
        print(
            f"  [{citation.marker}] {citation.ticker} {citation.form} "
            f"{citation.section} ({citation.accession}{as_of}) "
            f"chars {citation.char_start}-{citation.char_end}"
        )
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
