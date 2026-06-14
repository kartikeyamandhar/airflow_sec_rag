"""The run configuration: which companies, forms, and date window to acquire.

Loaded from a YAML file so the universe is data, not code. Scaling from 5 to
hundreds of companies is an edit to this file, not a code change.
"""

from __future__ import annotations

from datetime import date
from pathlib import Path
from typing import Any

import yaml
from pydantic import BaseModel, Field


class RunConfig(BaseModel):
    """A discovery/acquisition run definition."""

    tickers: list[str]
    forms: list[str] = Field(default_factory=lambda: ["10-K", "10-Q"])
    start_date: date
    end_date: date


def load_run_config(path: Path) -> RunConfig:
    """Load and validate a run config from a YAML file."""
    raw: Any = yaml.safe_load(path.read_text(encoding="utf-8"))
    return RunConfig.model_validate(raw)
