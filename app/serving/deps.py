"""Lazy, cached construction of the Retriever and Answerer for the service.

Building these needs Qdrant, Postgres, embedders, and the LLM, so the providers are
lazy (built on first use) and cached. In tests, the FastAPI providers
(`get_retriever`/`get_answerer`) are overridden with fakes, so the service is
testable without any backend.
"""

from __future__ import annotations

from datetime import date
from functools import lru_cache

from sqlalchemy.orm import Session, sessionmaker

from app.embedding.factory import build_embedder, build_sparse_embedder
from app.eval.verifier import GroundingVerifier
from app.generation.answerer import Answerer
from app.generation.llm import build_judge_client, build_llm_client
from app.index import repository as repo
from app.index.db import create_all, make_engine, make_session_factory, session_scope
from app.retrieval.decompose import aliases_from_companies
from app.retrieval.reranker import build_reranker
from app.retrieval.retriever import Retriever
from app.vectorstore.factory import build_qdrant_index
from configs.settings import Settings, get_settings


@lru_cache(maxsize=1)
def _session_factory() -> sessionmaker[Session]:
    engine = make_engine(get_settings().database_url)
    create_all(engine)
    return make_session_factory(engine)


def build_retriever(settings: Settings, session_factory: sessionmaker[Session]) -> Retriever:
    with session_scope(session_factory) as session:
        aliases = aliases_from_companies(repo.all_companies(session))
    return Retriever(
        embedder=build_embedder(settings),
        sparse_embedder=build_sparse_embedder(settings),
        index=build_qdrant_index(settings),
        reranker=build_reranker(settings.reranker_backend, settings.reranker_model),
        top_k=settings.retrieval_top_k,
        top_n=settings.rerank_top_n,
        aliases=aliases,
    )


def build_answerer(
    settings: Settings, session_factory: sessionmaker[Session], *, verify: bool
) -> Answerer:
    def report_dates(accessions: set[str]) -> dict[str, date]:
        with session_scope(session_factory) as session:
            return repo.report_dates_for_accessions(session, accessions)

    verifier = (
        GroundingVerifier(build_judge_client(settings), max_tokens=settings.judge_max_tokens)
        if verify
        else None
    )
    return Answerer(
        retriever=build_retriever(settings, session_factory),
        llm=build_llm_client(settings),
        max_tokens=settings.llm_max_tokens,
        context_chunks=settings.answer_context_chunks,
        min_coverage=settings.answer_min_citation_coverage,
        report_date_lookup=report_dates,
        verifier=verifier,
        numeric_check=verify,
    )


@lru_cache(maxsize=1)
def get_retriever() -> Retriever:
    """FastAPI dependency: the shared retriever (override in tests)."""
    return build_retriever(get_settings(), _session_factory())


@lru_cache(maxsize=1)
def get_answerer() -> Answerer:
    """FastAPI dependency: the shared answerer (override in tests)."""
    return build_answerer(get_settings(), _session_factory(), verify=False)
