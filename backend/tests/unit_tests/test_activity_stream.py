"""Activity events emitted while the graph retrieves lore or searches the web."""

import sys

import pytest
from langchain_core.messages import AIMessage, HumanMessage

from agent.graph import stream_agent

graph_module = sys.modules["agent.graph"]


class ScriptedModel:
    """Answer the lore router, then either call a tool or finish the turn."""

    def __init__(self, *, search: bool) -> None:
        self.search = search

    def bind_tools(self, _tools):
        return self

    async def ainvoke(self, messages):
        system = getattr(messages[0], "content", "")
        if isinstance(system, str) and "只回答 yes 或 no" in system:
            return AIMessage(content="yes" if not self.search else "no")
        if any(getattr(message, "type", None) == "tool" for message in messages):
            return AIMessage(content="今天多云。")
        if self.search:
            return AIMessage(
                content="",
                tool_calls=[
                    {
                        "name": "search_web",
                        "args": {"query": "杭州天气"},
                        "id": "call_weather",
                        "type": "tool_call",
                    }
                ],
            )
        return AIMessage(content="亚索追着风走。")


def _activities(events: list[object]) -> list[dict]:
    return [
        event
        for event in events
        if isinstance(event, dict) and event.get("type") == "activity"
    ]


@pytest.mark.anyio
async def test_stream_reports_retrieved_champions(monkeypatch: pytest.MonkeyPatch) -> None:
    def search(query: str, limit: int = 4):
        del limit
        assert "亚索" in query
        return [{"name": "亚索", "title": "疾风剑豪", "text": "他追着风走。"}]

    monkeypatch.setattr(graph_module.embedding_model, "search", search)
    monkeypatch.setattr(graph_module, "model", ScriptedModel(search=False))

    events = []
    async for item in stream_agent(
        {"messages": [HumanMessage(content="亚索是谁")]},
        {"configurable": {"session_id": "activity-lore"}},
    ):
        events.append(item)

    activities = _activities(events)
    assert [item["status"] for item in activities] == ["running", "complete"]
    assert activities[0]["name"] == "retrieve_lore"
    assert activities[0]["title"].startswith("正在检索资料")
    assert activities[1]["title"] == "检索到 亚索（疾风剑豪）"
    assert "他追着风走。" in activities[1]["detail"]


@pytest.mark.anyio
async def test_stream_reports_web_search(monkeypatch: pytest.MonkeyPatch) -> None:
    async def search(query: str) -> str:
        assert query == "杭州天气"
        return "1. 杭州\n多云"

    monkeypatch.setattr("agent.tools._search", search)
    monkeypatch.setattr(graph_module, "model", ScriptedModel(search=True))

    events = []
    async for item in stream_agent(
        {"messages": [HumanMessage(content="杭州今天天气")]},
        {"configurable": {"session_id": "activity-search"}},
    ):
        events.append(item)

    activities = _activities(events)
    assert [item["status"] for item in activities] == ["running", "complete"]
    assert activities[0]["id"] == "call_weather"
    assert activities[0]["title"] == "正在搜索：杭州天气"
    assert activities[1]["title"] == "已搜索：杭州天气"
    assert "多云" in activities[1]["detail"]
