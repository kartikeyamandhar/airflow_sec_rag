"""CLI entrypoints for SEC RAG pipeline jobs.

Each module here is an idempotent, resumable job runnable with
``uv run python -m scripts.<name>``. Business logic lives in ``app`` and is
imported here; scripts only parse arguments, wire dependencies, and report.
"""
