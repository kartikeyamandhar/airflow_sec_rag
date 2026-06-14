"""GPU embedding backend calling a RunPod serverless endpoint.

The scale path. It POSTs texts to the user's deployed endpoint (see
``infra/runpod/``) and expects vectors back. Contract::

    POST https://api.runpod.ai/v2/{endpoint_id}/runsync
    {"input": {"texts": ["...", "..."]}}
    -> {"output": {"embeddings": [[...], [...]]}}

This path is opt-in (``embedding_backend='runpod'``) and is covered by a contract
test, not a live call, since the endpoint is deployed by the human.
"""

from __future__ import annotations

import httpx


class RunPodEmbedder:
    """An ``Embedder`` backed by a RunPod serverless endpoint."""

    def __init__(
        self,
        *,
        endpoint_id: str,
        api_key: str,
        dimension: int,
        timeout: float = 120.0,
        http_client: httpx.Client | None = None,
    ) -> None:
        if not endpoint_id:
            raise ValueError("RunPod endpoint id is required for the runpod backend.")
        self._url = f"https://api.runpod.ai/v2/{endpoint_id}/runsync"
        self._headers = {
            "Authorization": f"Bearer {api_key}",
            "Content-Type": "application/json",
        }
        self._dimension = dimension
        self._http = http_client or httpx.Client(timeout=timeout)

    @property
    def dimension(self) -> int:
        return self._dimension

    def embed(self, texts: list[str]) -> list[list[float]]:
        response = self._http.post(
            self._url, headers=self._headers, json={"input": {"texts": texts}}
        )
        response.raise_for_status()
        payload = response.json()
        embeddings = payload["output"]["embeddings"]
        return [[float(value) for value in vector] for vector in embeddings]
