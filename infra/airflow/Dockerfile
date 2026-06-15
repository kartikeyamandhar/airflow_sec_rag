# Airflow image with uv available, so DAG tasks can `uv run` the SEC RAG CLIs.
# The repo is mounted at /opt/sec_rag at runtime (see docker-compose.yml); the
# project venv is created there on first task run.
FROM apache/airflow:2.10.4-python3.12

USER root
COPY --from=ghcr.io/astral-sh/uv:latest /uv /uvx /bin/
USER airflow
