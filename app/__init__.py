"""SEC RAG: a grounded answer engine over SEC filings.

This is the library package: clients, parsing, chunking, embedding, retrieval,
generation, and evaluation logic live here. IO sits at the edges; business logic
does not import the orchestrator (see CLAUDE.md Section 7).
"""

__version__ = "0.1.0"
