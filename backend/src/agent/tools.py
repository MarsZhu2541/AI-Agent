"""Tools the chat agent can call."""

import logging
import os
from datetime import datetime, timezone
from pathlib import Path
from typing import Annotated, Any

import httpx
from dotenv import load_dotenv
from langchain_core.tools import tool

load_dotenv(Path(__file__).resolve().parents[2] / ".env")

logger = logging.getLogger(__name__)

MAX_RESULTS = 5
TAVILY_ANSWER_URL = "https://tavily.com/"
TAVILY_SEARCH_URL = "https://api.tavily.com/search"


def _text(value: Any) -> str:
    """Return a trimmed string, or an empty string when the value is missing."""
    if value is None:
        return ""
    return str(value).strip()


def format_results(results: list[dict[str, Any]] | None) -> str:
    """Format Tavily hits the same way as the Java WebSearchTools."""
    if not results:
        return "没有搜到相关结果"
    lines: list[str] = []
    index = 1
    for result in results:
        if not result:
            continue
        url = _text(result.get("url"))
        snippet = _text(result.get("snippet"))
        if url == TAVILY_ANSWER_URL:
            if snippet:
                lines.append(f"摘要：{snippet}")
            continue
        lines.append(f"{index}. {_text(result.get('title'))}")
        index += 1
        if snippet:
            lines.append(snippet)
        if url:
            lines.append(url)
    formatted = "\n".join(lines).strip()
    return formatted or "没有搜到相关结果"


def _api_key() -> str:
    return os.getenv("TAVILY_API_KEY", "").strip()


def _to_results(payload: Any) -> list[dict[str, Any]]:
    if not isinstance(payload, dict):
        return []
    organic: list[dict[str, Any]] = []
    answer = _text(payload.get("answer"))
    if answer:
        organic.append(
            {
                "title": "Tavily Search API",
                "url": TAVILY_ANSWER_URL,
                "snippet": answer,
            }
        )
    for item in payload.get("results") or []:
        if not isinstance(item, dict):
            continue
        organic.append(
            {
                "title": _text(item.get("title")),
                "url": _text(item.get("url")),
                "snippet": _text(item.get("content")),
            }
        )
    return organic


async def _search(query: str) -> str:
    api_key = _api_key()
    if not api_key:
        return "联网搜索未配置 tavily.api_key"
    try:
        async with httpx.AsyncClient(timeout=30.0) as client:
            response = await client.post(
                TAVILY_SEARCH_URL,
                headers={"Authorization": f"Bearer {api_key}"},
                json={
                    "query": query,
                    "max_results": MAX_RESULTS,
                    "include_answer": True,
                },
            )
            response.raise_for_status()
            payload = response.json()
    except Exception as exc:
        logger.exception("Tavily search failed, query=%s", query)
        return f"联网搜索失败：{exc}"
    return format_results(_to_results(payload))


@tool(
    description="当需要实时信息、新闻、天气、近期事件、事实核验或你不确定的外部知识时，搜索互联网",
)
async def search_web(query: Annotated[str, "搜索关键词，尽量具体"]) -> str:
    """Search the web for real-time information."""
    cleaned = _text(query)
    if not cleaned:
        return "搜索关键词为空"
    return await _search(cleaned)


tools = [search_web]
