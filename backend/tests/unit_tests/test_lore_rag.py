import sys

import pytest
from langchain_core.messages import HumanMessage

from agent.graph import (
    GENERAL_SYSTEM,
    LORE_SYSTEM,
    format_lore,
    latest_user_text,
    lore_route_answer,
    retrieve,
    route_question,
    system_of,
)


def test_system_prompt_matches_whether_lore_was_retrieved() -> None:
    lore_prompt = system_of({}, "检索到的英雄传记片段：\n\n1. 亚托克斯")
    general_prompt = system_of({}, "")

    assert lore_prompt.startswith(LORE_SYSTEM)
    assert "《英雄联盟》背景故事百科全书" in lore_prompt
    assert general_prompt == GENERAL_SYSTEM
    assert "背景故事百科全书" not in general_prompt
    assert "与《英雄联盟》背景故事无关" in general_prompt


def test_latest_user_text_reads_text_blocks() -> None:
    messages = [
        HumanMessage(content="上一句"),
        HumanMessage(
            content=[
                {"type": "text", "text": "亚托克斯是谁"},
                {"type": "image_url", "image_url": {"url": "data:image/png;base64,aa"}},
            ]
        ),
    ]

    assert latest_user_text(messages) == "亚托克斯是谁"


def test_system_of_appends_retrieved_biographies() -> None:
    lore = format_lore(
        [
            {
                "name": "亚托克斯",
                "title": "暗裔剑魔",
                "text": "他被锁进了巨剑。",
            }
        ]
    )

    prompt = system_of({}, lore)

    assert prompt.startswith(LORE_SYSTEM)
    assert "只依据这些传记" in prompt
    assert "亚托克斯（暗裔剑魔）" in prompt
    assert "他被锁进了巨剑。" in prompt


@pytest.mark.anyio
async def test_retrieve_stores_matching_biographies(monkeypatch: pytest.MonkeyPatch) -> None:
    def search(query: str, limit: int = 4):
        del limit
        if query != "亚托克斯":
            return []
        return [{"name": "亚托克斯", "title": "暗裔剑魔", "text": "被封印在剑中"}]

    monkeypatch.setattr(sys.modules["agent.graph"].embedding_model, "search", search)

    result = await retrieve({"messages": [HumanMessage(content="亚托克斯")]}, {})

    assert "被封印在剑中" in result["lore_context"]


def test_lore_route_answer_accepts_only_yes() -> None:
    assert lore_route_answer("yes")
    assert lore_route_answer("是")
    assert not lore_route_answer("no")
    assert not lore_route_answer("这个问题和英雄联盟无关")


class _RouteModel:
    def __init__(self, answer: str) -> None:
        self.answer = answer

    async def ainvoke(self, _messages):
        class Reply:
            content = self.answer

        return Reply()


@pytest.mark.anyio
async def test_route_question_skips_unrelated_questions(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    monkeypatch.setattr(sys.modules["agent.graph"], "model", _RouteModel("no"))

    route = await route_question({"messages": [HumanMessage(content="杭州今天天气")]})

    assert route == "skip_retrieve"


@pytest.mark.anyio
async def test_route_question_retrieves_lore_questions(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    monkeypatch.setattr(sys.modules["agent.graph"], "model", _RouteModel("yes"))

    route = await route_question(
        {"messages": [HumanMessage(content="亚托克斯为什么会被锁进剑里")]}
    )

    assert route == "retrieve"


@pytest.mark.anyio
async def test_retrieve_skips_an_empty_question() -> None:
    result = await retrieve({"messages": [HumanMessage(content="   ")]}, {})

    assert result == {"lore_context": ""}
