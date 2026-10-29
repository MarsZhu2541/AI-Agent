import os

import pytest

from agent.tools import format_results, search_web


def test_format_results_prefers_tavily_answer_then_organic_hits() -> None:
    formatted = format_results(
        [
            {
                "title": "Tavily Search API",
                "url": "https://tavily.com/",
                "snippet": "杭州今天多云，气温 22 到 28 度",
            },
            {
                "title": "杭州天气预报",
                "url": "https://example.com/weather",
                "snippet": "未来三天以多云为主",
            },
        ]
    )

    assert formatted.startswith("摘要：杭州今天多云，气温 22 到 28 度")
    assert "1. 杭州天气预报" in formatted
    assert "https://example.com/weather" in formatted
    assert "未来三天以多云为主" in formatted


def test_format_results_returns_fallback_when_empty() -> None:
    assert format_results([]) == "没有搜到相关结果"
    assert format_results(None) == "没有搜到相关结果"


@pytest.mark.anyio
async def test_search_web_requires_api_key(monkeypatch: pytest.MonkeyPatch) -> None:
    monkeypatch.delenv("TAVILY_API_KEY", raising=False)

    assert await search_web.ainvoke({"query": "杭州天气"}) == "联网搜索未配置 tavily.api_key"
    assert await search_web.ainvoke({"query": " "}) == "搜索关键词为空"


@pytest.mark.anyio
async def test_search_web_live_against_tavily() -> None:
    if not os.getenv("TAVILY_API_KEY", "").strip():
        pytest.skip("未配置 TAVILY_API_KEY")

    result = await search_web.ainvoke({"query": "杭州今天天气"})
    print(result)

    assert "联网搜索未配置" not in result
    assert "联网搜索失败" not in result
    assert result != "没有搜到相关结果"
    assert len(result) > 20
