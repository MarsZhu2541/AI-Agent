from fastapi.testclient import TestClient
from langchain_core.messages import AIMessageChunk

from agent.api import app, graph


def test_chat_endpoint_streams_model_text(monkeypatch) -> None:
    async def fake_stream(*_args, **_kwargs):
        yield (AIMessageChunk(content="pon"), {})
        yield (AIMessageChunk(content="g"), {})

    monkeypatch.setattr(graph, "astream", fake_stream)

    with TestClient(app) as client:
        response = client.post(
            "/api/chat",
            json={"messages": [{"role": "user", "content": "ping"}]},
        )

    assert response.status_code == 200
    assert response.text == "pong"
