"""Structured logging configuration (structlog).

Call :func:`configure_logging` once at process start (the CLI entrypoints do).
Use :func:`get_logger` everywhere else. Output is structured key-value data, so
logs are machine-parseable. Never pass a secret as a log value; secrets are
``SecretStr`` in :mod:`configs.settings` precisely so they cannot be logged by
accident.
"""

from __future__ import annotations

import logging
from typing import cast

import structlog
from structlog.typing import FilteringBoundLogger


def configure_logging(level: str = "INFO", *, json_logs: bool = False) -> None:
    """Configure structlog process-wide.

    Args:
        level: Minimum level to emit (e.g. ``"INFO"``, ``"DEBUG"``).
        json_logs: If true, render JSON (for production log aggregation);
            otherwise a human-friendly console renderer (for local dev).
    """
    log_level = getattr(logging, level.upper(), logging.INFO)
    processors: list[structlog.typing.Processor] = [
        structlog.contextvars.merge_contextvars,
        structlog.processors.add_log_level,
        structlog.processors.TimeStamper(fmt="iso", utc=True),
        structlog.processors.StackInfoRenderer(),
        structlog.processors.format_exc_info,
    ]
    renderer: structlog.typing.Processor = (
        structlog.processors.JSONRenderer() if json_logs else structlog.dev.ConsoleRenderer()
    )
    structlog.configure(
        processors=[*processors, renderer],
        wrapper_class=structlog.make_filtering_bound_logger(log_level),
        logger_factory=structlog.PrintLoggerFactory(),
        cache_logger_on_first_use=True,
    )


def get_logger(name: str | None = None) -> FilteringBoundLogger:
    """Return a structlog logger bound to ``name``."""
    return cast(FilteringBoundLogger, structlog.get_logger(name))
