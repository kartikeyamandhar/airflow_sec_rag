"""The Airflow DAG compiles and only shells out to the pipeline CLIs.

A structural check (no Airflow import): the DAG file must be valid Python, chain the
four standalone CLIs via BashOperator, disable catchup, and never import the app
(control plane only - it must not parse or embed in-process).
"""

import py_compile
from pathlib import Path

_DAG = Path(__file__).resolve().parents[1] / "infra" / "airflow" / "dags" / "sec_rag_pipeline.py"


def test_dag_file_compiles() -> None:
    py_compile.compile(str(_DAG), doraise=True)


def test_dag_shells_out_to_each_cli() -> None:
    source = _DAG.read_text(encoding="utf-8")
    for module in (
        "scripts.discover_filings",
        "scripts.acquire_filings",
        "scripts.parse_filings",
        "scripts.index_chunks",
    ):
        assert module in source
    assert "BashOperator" in source
    assert "catchup=False" in source
    assert "schedule=" in source
    assert "chain(discover, acquire, parse, index)" in source


def test_dag_is_control_plane_only() -> None:
    source = _DAG.read_text(encoding="utf-8")
    # It orchestrates the CLIs; it must not import the app and run work in-process.
    assert "import app" not in source
    assert "from app" not in source
