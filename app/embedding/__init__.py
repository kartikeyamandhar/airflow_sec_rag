"""Embedding backends.

An ``Embedder`` turns text into dense vectors. The pipeline depends on the
Protocol, so the backend (CPU fastembed by default, RunPod GPU at scale, or a
deterministic fake in tests) is swappable without touching the indexing logic.
"""
