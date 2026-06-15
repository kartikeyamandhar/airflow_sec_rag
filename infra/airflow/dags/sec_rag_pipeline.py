"""Airflow DAG: the SEC RAG ingestion pipeline (control plane only).

This DAG only shells out to the standalone, idempotent CLIs - it never parses or
embeds inside an Airflow worker. Because every step checkpoints by filing status, a
scheduled run is incremental: discovery picks up new filings and acquire/parse/index
process only the not-yet-done ones. That status checkpoint is the freshness
mechanism; no separate last-seen-accession store is needed.

For scale, set EMBEDDING_BACKEND=runpod in the worker environment so the index step
triggers GPU embedding on RunPod instead of running it on the Airflow worker.

Deploy: mount this file into the Airflow `dags/` folder and run from an image that
has uv and this repo (see infra/airflow/docker-compose.yml). Settings come from the
environment (.env), exactly as for the CLIs. This file is infrastructure - it is not
imported by the app, mypy, or the test suite.
"""

from __future__ import annotations

import pendulum
from airflow.models.baseoperator import chain
from airflow.models.dag import DAG
from airflow.operators.bash import BashOperator

# Where the repo is mounted inside the Airflow image.
PROJECT_DIR = "/opt/sec_rag"
_RUN = f"cd {PROJECT_DIR} && uv run python -m"

default_args = {
    "retries": 2,
    "retry_delay": pendulum.duration(minutes=5),
}

with DAG(
    dag_id="sec_rag_pipeline",
    description="Discover, acquire, parse, and index SEC filings (incremental).",
    schedule="@daily",
    start_date=pendulum.datetime(2026, 1, 1, tz="UTC"),
    catchup=False,
    default_args=default_args,
    tags=["sec-rag"],
) as dag:
    discover = BashOperator(
        task_id="discover",
        bash_command=f"{_RUN} scripts.discover_filings --config configs/universe.dev.yaml",
    )
    acquire = BashOperator(
        task_id="acquire",
        bash_command=f"{_RUN} scripts.acquire_filings",
    )
    parse = BashOperator(
        task_id="parse",
        bash_command=f"{_RUN} scripts.parse_filings",
    )
    index = BashOperator(
        task_id="index",
        bash_command=f"{_RUN} scripts.index_chunks",
    )

    chain(discover, acquire, parse, index)
