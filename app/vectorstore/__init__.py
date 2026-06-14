"""Vector store: a thin wrapper over Qdrant.

Holds the dense chunk vectors plus their metadata, with first-class metadata
filtering for later retrieval. Point ids are derived from (accession, chunk_index)
so re-indexing a filing overwrites its points rather than duplicating them.
"""
