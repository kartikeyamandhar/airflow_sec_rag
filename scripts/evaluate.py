"""Run the golden set and apply the deploy gate.

Builds the full answerer with entailment verification on, runs every golden item,
prints the metrics, and applies the threshold gate. Exits non-zero if any metric
regresses below its threshold (so a deploy job can block on it). This calls the
answer and judge models, so run it where an LLM_API_KEY and a budget exist, not in
the default PR CI.

Run: ``uv run python -m scripts.evaluate --golden configs/golden.yaml``
"""

from __future__ import annotations

import argparse
from datetime import date
from pathlib import Path

from app.embedding.factory import build_embedder, build_sparse_embedder
from app.eval.gate import Thresholds, evaluate_gate
from app.eval.golden import load_golden, run_golden
from app.eval.verifier import GroundingVerifier
from app.generation.answerer import Answerer
from app.generation.llm import build_judge_client, build_llm_client
from app.index import repository as repo
from app.index.db import make_engine, make_session_factory, session_scope
from app.logging import configure_logging, get_logger
from app.retrieval.decompose import aliases_from_companies
from app.retrieval.reranker import build_reranker
from app.retrieval.retriever import Retriever
from app.vectorstore.factory import build_qdrant_index
from configs.settings import get_settings

logger = get_logger("scripts.evaluate")


def main(argv: list[str] | None = None) -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--golden", type=Path, default=Path("configs/golden.yaml"))
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

    answerer = Answerer(
        retriever=Retriever(
            embedder=build_embedder(settings),
            sparse_embedder=build_sparse_embedder(settings),
            index=build_qdrant_index(settings),
            reranker=build_reranker(settings.reranker_backend, settings.reranker_model),
            top_k=settings.retrieval_top_k,
            top_n=settings.rerank_top_n,
            aliases=aliases,
        ),
        llm=build_llm_client(settings),
        max_tokens=settings.llm_max_tokens,
        context_chunks=settings.answer_context_chunks,
        min_coverage=settings.answer_min_citation_coverage,
        report_date_lookup=report_dates,
        verifier=GroundingVerifier(
            build_judge_client(settings), max_tokens=settings.judge_max_tokens
        ),
        numeric_check=True,
    )

    report = run_golden(answerer, load_golden(args.golden))
    print(
        f"items={report.total} "
        f"refusal_accuracy={report.refusal_accuracy:.2f} "
        f"citation_hit_rate={report.citation_hit_rate:.2f} "
        f"substring_hit_rate={report.substring_hit_rate:.2f} "
        f"mean_faithfulness={report.mean_faithfulness:.2f}"
    )

    thresholds = Thresholds(
        min_refusal_accuracy=settings.eval_min_refusal_accuracy,
        min_citation_hit_rate=settings.eval_min_citation_hit_rate,
        min_faithfulness=settings.eval_min_faithfulness,
    )
    result = evaluate_gate(report, thresholds)
    if result.passed:
        print("GATE PASSED")
        return 0
    print("GATE FAILED:")
    for failure in result.failures:
        print(f"  - {failure}")
    return 1


if __name__ == "__main__":
    raise SystemExit(main())
