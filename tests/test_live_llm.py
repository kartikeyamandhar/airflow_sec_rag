"""Opt-in live test of the real Anthropic answer model.

Skipped unless ``RUN_LIVE_LLM=1`` so the default gate spends no credit. Needs a
valid ``LLM_API_KEY`` in the environment. Uses a tiny ``max_tokens`` to stay cheap.
Run: ``RUN_LIVE_LLM=1 uv run pytest tests/test_live_llm.py``
"""

import os

import pytest

from app.generation.llm import build_llm_client
from configs.settings import get_settings

pytestmark = pytest.mark.skipif(
    os.environ.get("RUN_LIVE_LLM") != "1",
    reason="set RUN_LIVE_LLM=1 to run the live Anthropic test",
)


def test_live_llm_completion() -> None:
    client = build_llm_client(get_settings())
    text = client.complete(
        system="You are terse.", user="Reply with the single word: ok", max_tokens=16
    )
    assert isinstance(text, str)
    assert text.strip() != ""
