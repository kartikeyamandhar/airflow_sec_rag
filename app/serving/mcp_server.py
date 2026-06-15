"""MCP server exposing retrieval and answering as tools (the agentic bridge).

An external agent (e.g. Claude) can call ``search_filings`` and ``answer_question``
to use this grounded engine as a tool. The tools delegate to the same lazily-built
services as the HTTP API. Run over stdio with ``python -m app.serving.mcp_server``.
"""

from __future__ import annotations

from typing import Any

from mcp.server.fastmcp import FastMCP

from app.serving.deps import get_answerer, get_retriever
from app.serving.tools import answer_to_dict, chunks_to_dicts

mcp = FastMCP("sec-rag")


@mcp.tool()
def search_filings(question: str, ticker: str | None = None) -> list[dict[str, Any]]:
    """Hybrid-search SEC filings and return the most relevant cited chunks."""
    return chunks_to_dicts(get_retriever().retrieve(question, ticker=ticker))


@mcp.tool()
def answer_question(question: str, ticker: str | None = None) -> dict[str, Any]:
    """Answer a question over SEC filings: a grounded, span-cited answer or a refusal."""
    return answer_to_dict(get_answerer().answer(question, ticker=ticker))


def main() -> None:
    mcp.run()


if __name__ == "__main__":
    main()
