"""Filing index and job checkpoint store (Postgres via SQLAlchemy).

The index is the source of truth for what has been discovered and what has been
stored. It is what makes acquisition idempotent and resumable: a re-run reads
status from here and skips completed work.
"""
