"""RunPod serverless handler for bge embeddings (the scale path).

Deploy this as a RunPod serverless endpoint, then set ``EMBEDDING_BACKEND=runpod``
and ``RUNPOD_ENDPOINT_ID`` in ``.env`` to route embedding through the GPU.

Contract (matches app/embedding/runpod_embedder.py):
    input:  {"texts": ["...", "..."]}
    output: {"embeddings": [[...], [...]]}

Build the image from this directory's Dockerfile, push it to a registry RunPod can
pull, and create a serverless endpoint from it. This file is infrastructure, not
part of the importable app package or the test gate.
"""

from __future__ import annotations

import os
from typing import Any

import runpod  # provided by the RunPod serverless base image
from fastembed import TextEmbedding

_MODEL_NAME = os.environ.get("EMBEDDING_MODEL", "BAAI/bge-small-en-v1.5")
_model = TextEmbedding(model_name=_MODEL_NAME)


def handler(event: dict[str, Any]) -> dict[str, Any]:
    texts = event.get("input", {}).get("texts", [])
    embeddings = [[float(value) for value in vector] for vector in _model.embed(texts)]
    return {"embeddings": embeddings}


runpod.serverless.start({"handler": handler})
