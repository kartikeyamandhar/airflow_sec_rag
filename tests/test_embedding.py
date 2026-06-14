"""Embedding backends: deterministic fake, and the RunPod request contract."""

import httpx
import respx

from app.embedding.runpod_embedder import RunPodEmbedder
from tests.fakes import FakeEmbedder


def test_fake_embedder_is_deterministic_and_shaped() -> None:
    embedder = FakeEmbedder(dimension=8)
    first = embedder.embed(["hello", "world"])
    second = embedder.embed(["hello", "world"])
    assert first == second
    assert len(first) == 2
    assert all(len(vector) == 8 for vector in first)
    assert embedder.embed(["hello"]) != embedder.embed(["different"])


@respx.mock
def test_runpod_embedder_request_and_response_contract() -> None:
    endpoint = "https://api.runpod.ai/v2/abc123/runsync"
    route = respx.post(endpoint).mock(
        return_value=httpx.Response(200, json={"output": {"embeddings": [[0.1, 0.2], [0.3, 0.4]]}})
    )
    embedder = RunPodEmbedder(endpoint_id="abc123", api_key="key", dimension=2)
    vectors = embedder.embed(["a", "b"])

    assert vectors == [[0.1, 0.2], [0.3, 0.4]]
    assert route.called
    sent = route.calls.last.request
    assert sent.headers["Authorization"] == "Bearer key"
    import json

    assert json.loads(sent.content) == {"input": {"texts": ["a", "b"]}}
