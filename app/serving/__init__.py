"""Serving: a FastAPI HTTP service (with a thin UI and metrics) and an MCP server.

The HTTP and MCP surfaces both delegate to the same lazily-built Retriever and
Answerer, so the engine is reachable by a person, by other software, and by an agent.
"""
