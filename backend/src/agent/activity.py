"""Visible steps the chat UI shows while retrieval and search run."""

from __future__ import annotations

import re
from typing import Any

from langgraph.config import get_stream_writer

MAX_TITLE = 200
MAX_DETAIL = 4000
ACTIVITY_NAMES = frozenset({"retrieve_lore", "search_web", "get_weather"})
ACTIVITY_STATUSES = frozenset({"running", "complete", "error"})
_ID = re.compile(r"^[A-Za-z0-9_.:-]{1,80}$")
_SEARCH_FAILURES = ("联网搜索失败", "联网搜索未配置")
_WEATHER_FAILURES = ("未能获取", "未找到地区", "搜索关键词为空")


def clip(value: object, limit: int) -> str:
    """Trim text to ``limit`` characters."""
    text = str(value or "").strip()
    if len(text) <= limit:
        return text
    if limit <= 1:
        return text[:limit]
    return text[: limit - 1] + "…"


def one_line(value: object, limit: int) -> str:
    """Collapse whitespace and trim text for a single-line label."""
    return clip(" ".join(str(value or "").split()), limit)


def safe_id(identifier: object, fallback: str) -> str:
    """Return a stream id, or ``fallback`` when the candidate is unusable."""
    cleaned = str(identifier or "").strip()
    if _ID.fullmatch(cleaned):
        return cleaned
    return fallback


def make_activity(
    identifier: object,
    name: str,
    status: str,
    title: object,
    detail: object = "",
) -> dict[str, str]:
    """Build one activity event."""
    return {
        "type": "activity",
        "id": safe_id(identifier, name),
        "name": name,
        "status": status,
        "title": one_line(title, MAX_TITLE),
        "detail": clip(detail, MAX_DETAIL),
    }


def publish_activity(event: dict[str, Any]) -> None:
    """Write one activity into the graph custom stream.

    Calls outside a streaming run are ignored.
    """
    try:
        get_stream_writer()(event)
    except Exception:
        return


def stream_activity(payload: object) -> dict[str, str] | None:
    """Return a validated activity event, or None when ``payload`` is not one."""
    if not isinstance(payload, dict) or payload.get("type") != "activity":
        return None
    name = payload.get("name")
    status = payload.get("status")
    if name not in ACTIVITY_NAMES or status not in ACTIVITY_STATUSES:
        return None
    identifier = safe_id(payload.get("id"), "")
    if not identifier:
        return None
    return make_activity(
        identifier,
        str(name),
        str(status),
        payload.get("title"),
        payload.get("detail"),
    )


def activity_for_history(payload: object) -> dict[str, str] | None:
    """Return an activity ready to store. A still-running step is marked unfinished."""
    event = stream_activity(payload)
    if event is None:
        return None
    if event["status"] == "running":
        event["status"] = "error"
        if not event["detail"]:
            event["detail"] = "未完成"
    return event


def lore_running(query: str) -> dict[str, str]:
    """Describe a biography search that has started."""
    short = one_line(query, 40)
    title = f"正在检索资料：{short}" if short else "正在检索资料"
    return make_activity("retrieve_lore", "retrieve_lore", "running", title, "")


def lore_found(hits: list[dict[str, Any]]) -> dict[str, str]:
    """Describe the biographies a search returned."""
    labels = [_hit_label(hit) for hit in hits[:4]]
    title = f"检索到 {'、'.join(labels)}" if labels else "检索到相关资料"
    if len(hits) > len(labels):
        title = f"{title} 等 {len(hits)} 篇"
    return make_activity(
        "retrieve_lore",
        "retrieve_lore",
        "complete",
        title,
        _lore_detail(hits),
    )


def lore_missed() -> dict[str, str]:
    """Describe a biography search that found nothing."""
    return make_activity(
        "retrieve_lore",
        "retrieve_lore",
        "complete",
        "没有检索到相关资料",
        "",
    )


def lore_failed() -> dict[str, str]:
    """Describe a biography search that failed."""
    return make_activity(
        "retrieve_lore",
        "retrieve_lore",
        "error",
        "资料检索失败",
        "检索没有完成",
    )


def search_running(identifier: str, query: str) -> dict[str, str]:
    """Describe a web search that has started."""
    title = f"正在搜索：{query}" if query else "正在搜索"
    return make_activity(identifier, "search_web", "running", title, "")


def search_finished(identifier: str, query: str, body: str) -> dict[str, str]:
    """Describe a finished web search, including a failed lookup."""
    failed = body.startswith(_SEARCH_FAILURES)
    if failed:
        title = f"搜索失败：{query}" if query else "搜索失败"
        status = "error"
    else:
        title = f"已搜索：{query}" if query else "已搜索"
        status = "complete"
    return make_activity(identifier, "search_web", status, title, body)


def search_unfinished(identifier: str, query: str) -> dict[str, str]:
    """Describe a web search that stopped before returning results."""
    title = f"搜索失败：{query}" if query else "搜索失败"
    return make_activity(identifier, "search_web", "error", title, "搜索没有完成")


def weather_running(identifier: str, query: str) -> dict[str, str]:
    """Describe a weather lookup that has started."""
    title = f"正在查询天气：{query}" if query else "正在查询天气"
    return make_activity(identifier, "get_weather", "running", title, "")


def weather_finished(identifier: str, query: str, body: str) -> dict[str, str]:
    """Describe a finished weather lookup, including a failed lookup."""
    failed = body.startswith(_WEATHER_FAILURES)
    if failed:
        title = f"天气查询失败：{query}" if query else "天气查询失败"
        status = "error"
    else:
        title = f"已查询天气：{query}" if query else "已查询天气"
        status = "complete"
    return make_activity(identifier, "get_weather", status, title, body)


def weather_unfinished(identifier: str, query: str) -> dict[str, str]:
    """Describe a weather lookup that stopped before returning a forecast."""
    title = f"天气查询失败：{query}" if query else "天气查询失败"
    return make_activity(identifier, "get_weather", "error", title, "天气查询没有完成")


def _hit_label(hit: dict[str, Any]) -> str:
    name = str(hit.get("name") or "").strip()
    title = str(hit.get("title") or "").strip()
    if name and title:
        return f"{name}（{title}）"
    return name or title or str(hit.get("id") or "资料")


def _lore_detail(hits: list[dict[str, Any]]) -> str:
    blocks: list[str] = []
    for hit in hits[:4]:
        excerpt = one_line(hit.get("text"), 160)
        label = _hit_label(hit)
        blocks.append(f"{label}\n{excerpt}" if excerpt else label)
    return "\n\n".join(blocks)
