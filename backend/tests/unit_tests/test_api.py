import base64
import json

import pytest
from fastapi.testclient import TestClient
from langchain_core.messages import (
    AIMessage,
    AIMessageChunk,
    HumanMessage,
    SystemMessage,
)

from agent import history
from agent.api import ChatRequest, agent_reply_text, app, to_langchain, transcript_of
from agent.downloads import prepare_download


def test_agent_reply_text_hides_the_route_decision() -> None:
    hidden = (AIMessageChunk(content="no"), {"langgraph_node": "__start__"})
    visible = (AIMessageChunk(content="好"), {"langgraph_node": "agent"})

    assert agent_reply_text(hidden) == ""
    assert agent_reply_text(visible) == "好"


def _chat(response) -> tuple[str, list[dict]]:
    text: list[str] = []
    files: list[dict] = []
    for line in response.text.splitlines():
        if not line.strip():
            continue
        event = json.loads(line)
        if event["type"] == "text":
            text.append(event["text"])
        elif event["type"] == "file":
            files.append(event)
        else:
            raise AssertionError(event)
    return "".join(text), files


SESSION = "112031122990911488"


def _convert(messages: list[dict]) -> list:
    return to_langchain(
        ChatRequest.model_validate(
            {"session_id": SESSION, "messages": messages}
        ).messages
    )


def _chat_body(messages: list[dict]) -> dict:
    return {"session_id": SESSION, "messages": messages}


@pytest.fixture(autouse=True)
def memory_history(monkeypatch: pytest.MonkeyPatch) -> list[dict]:
    """Keep unit tests off the real PostgreSQL database."""
    stored: list[dict] = []

    async def init_db() -> None:
        return None

    async def replace_messages(messages: list[dict], session_id: str) -> None:
        assert session_id == SESSION
        stored.clear()
        stored.extend(messages)

    async def list_messages(session_id: str) -> list[dict]:
        assert session_id == SESSION
        return [dict(message) for message in stored]

    async def get_image(image_id: int) -> tuple[str, bytes] | None:
        return None

    async def image_data_url(image_id: int, session_id: str) -> str | None:
        return None

    monkeypatch.setattr(history, "init_db", init_db)
    monkeypatch.setattr(history, "replace_messages", replace_messages)
    monkeypatch.setattr(history, "list_messages", list_messages)
    monkeypatch.setattr(history, "get_image", get_image)
    monkeypatch.setattr(history, "image_data_url", image_data_url)
    return stored


PNG = (
    "data:image/png;base64,"
    "iVBORw0KGgoAAAANSUhEUgAAAAEAAAABCAYAAAAfFcSJAAAADUlEQVR42mP8z8BQDwAEhQGAhKmMIQAAAABJRU5ErkJggg=="
)


def test_chat_endpoint_streams_model_text(monkeypatch) -> None:
    async def fake_stream(*_args, **_kwargs):
        yield (AIMessageChunk(content="pon"), {"langgraph_node": "agent"})
        yield (AIMessageChunk(content="g"), {"langgraph_node": "agent"})

    monkeypatch.setattr("agent.api.stream_agent", fake_stream)

    with TestClient(app) as client:
        response = client.post(
            "/aiapi/chat",
            json=_chat_body([{"role": "user", "content": "ping"}]),
        )

    assert response.status_code == 200
    assert _chat(response) == ("pong", [])


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
            "/aiapi/chat",
            json=_chat_body(
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
            ),
        )

    assert response.status_code == 400
    assert "不支持的图片格式" in response.json()["detail"]


def test_chat_forwards_image_blocks(monkeypatch) -> None:
    seen: dict = {}

    async def fake_stream(state, *_args, **_kwargs):
        seen["messages"] = state["messages"]
        yield (AIMessageChunk(content="一只猫"), {"langgraph_node": "agent"})

    monkeypatch.setattr("agent.api.stream_agent", fake_stream)

    with TestClient(app) as client:
        response = client.post(
            "/aiapi/chat",
            json=_chat_body(
                [
                    {
                        "role": "user",
                        "content": [
                            {"type": "text", "text": "这是什么"},
                            {"type": "image", "image": PNG},
                        ],
                    }
                ]
            ),
        )

    assert response.status_code == 200
    assert _chat(response)[0] == "一只猫"
    assert (
        seen["messages"][0]
        .content[1]["image_url"]["url"]
        .startswith("data:image/png;base64,")
    )


def test_messages_are_empty_before_any_chat() -> None:
    with TestClient(app) as client:
        response = client.get(f"/aiapi/messages/{SESSION}")

    assert response.status_code == 200
    assert response.json() == {"messages": []}


def test_chat_saves_the_full_thread(memory_history: list[dict], monkeypatch) -> None:
    async def fake_stream(*_args, **_kwargs):
        yield (AIMessageChunk(content="pon"), {"langgraph_node": "agent"})
        yield (AIMessageChunk(content="g"), {"langgraph_node": "agent"})

    monkeypatch.setattr("agent.api.stream_agent", fake_stream)

    with TestClient(app) as client:
        first = client.post(
            "/aiapi/chat",
            json=_chat_body([{"role": "user", "content": "ping"}]),
        )
        assert first.status_code == 200
        assert _chat(first)[0] == "pong"
        saved = client.get(f"/aiapi/messages/{SESSION}")

        second = client.post(
            "/aiapi/chat",
            json=_chat_body(
                [
                    {"role": "user", "content": "ping"},
                    {"role": "assistant", "content": "pong"},
                    {"role": "user", "content": "again"},
                ]
            ),
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
        yield (AIMessageChunk(content="一只猫"), {"langgraph_node": "agent"})

    monkeypatch.setattr("agent.api.stream_agent", fake_stream)
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
        response = client.post("/aiapi/chat", json=_chat_body(sent))

    assert response.status_code == 200
    assert memory_history[0]["content"] == sent[0]["content"]
    assert memory_history[1] == {"role": "assistant", "content": "一只猫"}


def test_chat_streams_a_file_for_confirmation(
    memory_history: list[dict], monkeypatch
) -> None:
    async def fake_stream(*_args, **_kwargs):
        message = await prepare_download.ainvoke(
            {"filename": "note.txt", "content": "hello"}
        )
        assert "已生成" in message
        yield (AIMessageChunk(content="请确认下载"), {"langgraph_node": "agent"})

    monkeypatch.setattr("agent.api.stream_agent", fake_stream)

    with TestClient(app) as client:
        response = client.post(
            "/aiapi/chat",
            json=_chat_body([{"role": "user", "content": "写个文件"}]),
        )

    text, files = _chat(response)
    assert response.status_code == 200
    assert text == "请确认下载"
    assert files == [
        {
            "type": "file",
            "filename": "note.txt",
            "content": "hello",
            "mime": "text/plain",
        }
    ]
    assert memory_history[1]["content"] == [
        {"type": "text", "text": "请确认下载"},
        {
            "type": "file",
            "filename": "note.txt",
            "content": "hello",
            "mime": "text/plain",
        },
    ]


def test_transcript_skips_an_empty_reply() -> None:
    body = ChatRequest.model_validate(
        {"session_id": SESSION, "messages": [{"role": "user", "content": "ping"}]}
    )

    assert transcript_of(body, "") == [{"role": "user", "content": "ping"}]


def test_image_endpoint_returns_bytes(monkeypatch) -> None:
    png = base64.b64decode(PNG.split(",", 1)[1])

    async def get_image(image_id: int) -> tuple[str, bytes] | None:
        assert image_id == 9
        return "image/png", png

    monkeypatch.setattr(history, "get_image", get_image)

    with TestClient(app) as client:
        response = client.get("/aiapi/images/9")

    assert response.status_code == 200
    assert response.headers["content-type"].startswith("image/png")
    assert response.content == png


def test_chat_resolves_stored_image_urls(monkeypatch) -> None:
    seen: dict = {}

    async def fake_stream(state, *_args, **_kwargs):
        seen["messages"] = state["messages"]
        yield (AIMessageChunk(content="一只猫"), {"langgraph_node": "agent"})

    async def image_data_url(image_id: int, session_id: str) -> str:
        assert image_id == 4
        assert session_id == SESSION
        return PNG

    monkeypatch.setattr("agent.api.stream_agent", fake_stream)
    monkeypatch.setattr(history, "image_data_url", image_data_url)

    with TestClient(app) as client:
        response = client.post(
            "/aiapi/chat",
            json=_chat_body(
                [
                    {
                        "role": "user",
                        "content": [
                            {"type": "text", "text": "这是什么"},
                            {"type": "image", "image": "/aiapi/images/4"},
                        ],
                    }
                ]
            ),
        )

    assert response.status_code == 200
    assert (
        seen["messages"][0]
        .content[1]["image_url"]["url"]
        .startswith("data:image/png;base64,")
    )
