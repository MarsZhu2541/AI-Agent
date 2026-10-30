import pytest
from fastapi.testclient import TestClient
from langchain_core.messages import AIMessage, AIMessageChunk, HumanMessage, SystemMessage

from agent import history
from agent.api import ChatRequest, app, graph, to_langchain, transcript_of


def _convert(messages: list[dict]) -> list:
    return to_langchain(ChatRequest.model_validate({"messages": messages}).messages)


@pytest.fixture(autouse=True)
def memory_history(monkeypatch: pytest.MonkeyPatch) -> list[dict]:
    """Keep unit tests off the real MySQL database."""
    stored: list[dict] = []

    async def init_db() -> None:
        return None

    async def replace_messages(messages: list[dict]) -> None:
        stored.clear()
        stored.extend(messages)

    async def list_messages() -> list[dict]:
        return [dict(message) for message in stored]

    monkeypatch.setattr(history, "init_db", init_db)
    monkeypatch.setattr(history, "replace_messages", replace_messages)
    monkeypatch.setattr(history, "list_messages", list_messages)
    return stored

PNG = (
    "data:image/png;base64,"
    "iVBORw0KGgoAAAANSUhEUgAAAAEAAAABCAYAAAAfFcSJAAAADUlEQVR42mP8z8BQDwAEhQGAhKmMIQAAAABJRU5ErkJggg=="
)


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


def test_to_langchain_keeps_text_messages() -> None:
    messages = _convert(
        [
            {"role": "system", "content": "be brief"},
            {"role": "user", "content": "ping"},
            {"role": "assistant", "content": "pong"},
        ]
    )

    assert isinstance(messages[0], SystemMessage)
    assert messages[0].content == "be brief"
    assert isinstance(messages[1], HumanMessage)
    assert messages[1].content == "ping"
    assert isinstance(messages[2], AIMessage)
    assert messages[2].content == "pong"


def test_to_langchain_builds_image_blocks() -> None:
    messages = _convert(
        [
            {
                "role": "user",
                "content": [
                    {"type": "text", "text": "这是什么"},
                    {"type": "image", "image": PNG},
                ],
            }
        ]
    )

    content = messages[0].content
    assert content[0] == {"type": "text", "text": "这是什么"}
    assert content[1]["type"] == "image_url"
    assert content[1]["image_url"]["url"].startswith("data:image/png;base64,")


def test_to_langchain_prompts_when_the_image_has_no_text() -> None:
    messages = _convert(
        [{"role": "user", "content": [{"type": "image", "image": PNG}]}]
    )

    assert messages[0].content[0] == {"type": "text", "text": "请识别这张图片。"}
    assert messages[0].content[1]["type"] == "image_url"


def test_to_langchain_rejects_unsupported_images() -> None:
    with pytest.raises(ValueError, match="不支持的图片格式"):
        _convert(
            [
                {
                    "role": "user",
                    "content": [
                        {
                            "type": "image",
                            "image": "data:image/bmp;base64,Qk0=",
                        }
                    ],
                }
            ]
        )


def test_chat_rejects_unsupported_images() -> None:
    with TestClient(app) as client:
        response = client.post(
            "/api/chat",
            json={
                "messages": [
                    {
                        "role": "user",
                        "content": [
                            {
                                "type": "image",
                                "image": "data:image/bmp;base64,Qk0=",
                            }
                        ],
                    }
                ]
            },
        )

    assert response.status_code == 400
    assert "不支持的图片格式" in response.json()["detail"]


def test_chat_forwards_image_blocks(monkeypatch) -> None:
    seen: dict = {}

    async def fake_stream(state, *_args, **_kwargs):
        seen["messages"] = state["messages"]
        yield (AIMessageChunk(content="一只猫"), {})

    monkeypatch.setattr(graph, "astream", fake_stream)

    with TestClient(app) as client:
        response = client.post(
            "/api/chat",
            json={
                "messages": [
                    {
                        "role": "user",
                        "content": [
                            {"type": "text", "text": "这是什么"},
                            {"type": "image", "image": PNG},
                        ],
                    }
                ]
            },
        )

    assert response.status_code == 200
    assert response.text == "一只猫"
    assert seen["messages"][0].content[1]["image_url"]["url"].startswith(
        "data:image/png;base64,"
    )


def test_messages_are_empty_before_any_chat() -> None:
    with TestClient(app) as client:
        response = client.get("/api/messages")

    assert response.status_code == 200
    assert response.json() == {"messages": []}


def test_chat_saves_the_full_thread(memory_history: list[dict], monkeypatch) -> None:
    async def fake_stream(*_args, **_kwargs):
        yield (AIMessageChunk(content="pon"), {})
        yield (AIMessageChunk(content="g"), {})

    monkeypatch.setattr(graph, "astream", fake_stream)

    with TestClient(app) as client:
        first = client.post(
            "/api/chat",
            json={"messages": [{"role": "user", "content": "ping"}]},
        )
        assert first.status_code == 200
        assert first.text == "pong"
        saved = client.get("/api/messages")

        second = client.post(
            "/api/chat",
            json={
                "messages": [
                    {"role": "user", "content": "ping"},
                    {"role": "assistant", "content": "pong"},
                    {"role": "user", "content": "again"},
                ]
            },
        )

    assert saved.json()["messages"] == [
        {"role": "user", "content": "ping"},
        {"role": "assistant", "content": "pong"},
    ]
    assert second.status_code == 200
    assert memory_history == [
        {"role": "user", "content": "ping"},
        {"role": "assistant", "content": "pong"},
        {"role": "user", "content": "again"},
        {"role": "assistant", "content": "pong"},
    ]


def test_chat_saves_image_content(memory_history: list[dict], monkeypatch) -> None:
    async def fake_stream(*_args, **_kwargs):
        yield (AIMessageChunk(content="一只猫"), {})

    monkeypatch.setattr(graph, "astream", fake_stream)
    sent = [
        {
            "role": "user",
            "content": [
                {"type": "text", "text": "这是什么"},
                {"type": "image", "image": PNG},
            ],
        }
    ]

    with TestClient(app) as client:
        response = client.post("/api/chat", json={"messages": sent})

    assert response.status_code == 200
    assert memory_history[0]["content"] == sent[0]["content"]
    assert memory_history[1] == {"role": "assistant", "content": "一只猫"}


def test_transcript_skips_an_empty_reply() -> None:
    body = ChatRequest.model_validate(
        {"messages": [{"role": "user", "content": "ping"}]}
    )

    assert transcript_of(body, "") == [{"role": "user", "content": "ping"}]
