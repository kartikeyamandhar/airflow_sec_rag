"""The strict grounding prompt and the refusal sentinel."""

from __future__ import annotations

from collections.abc import Sequence

from app.retrieval.retriever import RetrievedChunk

# The model emits this exact token when the passages cannot support an answer.
REFUSAL_SENTINEL = "NO_ANSWER"

_SYSTEM = (
    "You answer questions about SEC filings using ONLY the numbered context "
    "passages provided. Rules:\n"
    "- Use only information found in the passages. Never use outside knowledge.\n"
    "- After each sentence, cite the passage(s) it draws from with bracketed "
    "numbers, e.g. [1] or [2][3].\n"
    "- Every sentence must carry at least one citation.\n"
    f"- If the passages do not contain enough information, reply with exactly: "
    f"{REFUSAL_SENTINEL}\n"
    "- Be concise and factual. Do not add commentary or preamble."
)


def build_answer_prompt(question: str, chunks: Sequence[RetrievedChunk]) -> tuple[str, str]:
    """Return (system, user) prompts for a grounded answer over ``chunks``."""
    blocks: list[str] = []
    for index, chunk in enumerate(chunks, start=1):
        header = (
            f"[{index}] {chunk.ticker or ''} {chunk.form} {chunk.section} "
            f"(accession {chunk.accession})"
        ).strip()
        blocks.append(f"{header}\n{chunk.text}")
    context = "\n\n".join(blocks)
    user = f"Question: {question}\n\nContext passages:\n{context}\n\nAnswer (cite every sentence):"
    return _SYSTEM, user
