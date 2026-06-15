"""The answer-model client.

Only this module talks to the LLM. The client sends no sampling parameters
(``temperature``/``top_p``/``top_k``) or ``thinking``/``effort`` config, so the
same call is valid across Haiku, Sonnet, and Opus (Opus 4.8 rejects sampling
params). A safety ``refusal`` stop reason is surfaced as empty output, which the
answerer treats as a refusal.
"""

from __future__ import annotations

from typing import Protocol

import anthropic
from anthropic.types import MessageParam, TextBlock

from configs.settings import Settings


class LLMClient(Protocol):
    """Single-shot text completion from a system + user prompt."""

    def complete(self, *, system: str, user: str, max_tokens: int) -> str:
        """Return the model's text response."""
        ...


class AnthropicClient:
    """An ``LLMClient`` backed by the Anthropic Messages API."""

    def __init__(self, *, api_key: str, model: str) -> None:
        if not api_key:
            raise ValueError("LLM_API_KEY is required for the Anthropic client.")
        self._client = anthropic.Anthropic(api_key=api_key)
        self._model = model

    def complete(self, *, system: str, user: str, max_tokens: int) -> str:
        messages: list[MessageParam] = [{"role": "user", "content": user}]
        message = self._client.messages.create(
            model=self._model,
            max_tokens=max_tokens,
            system=system,
            messages=messages,
        )
        if message.stop_reason == "refusal":
            return ""
        return "".join(block.text for block in message.content if isinstance(block, TextBlock))


def build_llm_client(settings: Settings) -> LLMClient:
    """Build the configured answer-model client."""
    return AnthropicClient(
        api_key=settings.llm_api_key.get_secret_value(), model=settings.llm_model
    )
