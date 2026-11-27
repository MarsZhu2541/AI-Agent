"""Chat graph backed by OpenAI."""

from __future__ import annotations

import asyncio
import logging
from typing import Any

from langchain_core.messages import HumanMessage, RemoveMessage, SystemMessage
from langchain_openai import ChatOpenAI
from langgraph.checkpoint.memory import InMemorySaver
from langgraph.constants import START
from langgraph.graph import END, StateGraph
from langgraph.prebuilt import ToolNode

from agent.activity import (
    lore_failed,
    lore_found,
    lore_missed,
    lore_running,
    publish_activity,
    weather_finished,
    weather_running,
    weather_unfinished,
)
from agent.embeddings import embedding_model
from agent.state import AgentState
from agent.outfit.tools import tools

logger = logging.getLogger(__name__)

model = ChatOpenAI(model="deepseek-v4.1-flash")


LORE_SYSTEM = (
    "请参考以下穿搭风格资料给出穿搭建议"
    "文档没有记载的情节不要编造。"
    "回复消息中不要明说或者暗示检索到文档的存在。"
)

GENERAL_SYSTEM = (
    "你是智能穿搭助手。"
    "用户询问某地天气，或需要穿搭建议时，必须先调用 get_weather 或get_weather_by_ip 获取该地实时天气，"
    "有指定地区时，需要调用通过地址获取天气的工具"
    "有ip地址时，需要调用通过ip地址获取天气的工具"
    "不要凭记忆回答天气。"
    "拿到天气后，再结合用户衣橱和穿搭资料给出穿搭推荐。"
)

LORE_LIMIT = 4
WEATHER_TOOLS = {"get_weather", "get_weather_by_ip"}


DEFAULT_THREAD_ID = "default"

MAX_MESSAGES = 20

def thread_id_of(config) -> str:
    session_id = config.get("configurable").get("session_id")
    if session_id is None or str(session_id).strip() == "":
        return DEFAULT_THREAD_ID
    return str(session_id)


def max_messages_of(config) -> int:
    value = config.get("configurable").get("max_messages")
    try:
        value = int(value)
    except (TypeError, ValueError):
        return MAX_MESSAGES
    return value if value > 0 else MAX_MESSAGES


def should_continue(state):
    messages = state["messages"]
    last_message = messages[-1]
    if not last_message.tool_calls:
        return END
    else:
        return "tools"


def get_tools():
    return tools


def latest_user_text(messages: list[Any]) -> str:
    """Return the text of the latest user message."""
    for message in reversed(messages):
        if getattr(message, "type", None) != "human":
            continue
        content = message.content
        if isinstance(content, str):
            return content.strip()
        if isinstance(content, list):
            parts: list[str] = []
            for block in content:
                if isinstance(block, str):
                    parts.append(block)
                elif isinstance(block, dict) and block.get("type") == "text":
                    parts.append(str(block.get("text", "")))
            return "\n".join(part.strip() for part in parts if part.strip())
        return ""
    return ""


def format_lore(hits: list[dict[str, Any]]) -> str:
    """Format retrieved biographies for the system prompt."""
    if not hits:
        return ""
    blocks = ["检索到的穿搭知识："]
    for index, hit in enumerate(hits, start=1):
        name = str(hit.get("name") or "")
        title = str(hit.get("title") or "")
        heading = f"{name}（{title}）" if name and title else name or title or str(hit.get("id") or "")
        blocks.append(f"{index}. {heading}\n{hit.get('text') or ''}")
    return "\n\n".join(blocks)


def system_of() -> str:
    return GENERAL_SYSTEM + LORE_SYSTEM



def message_text(message: Any) -> str:
    """Return the text content of one model message."""
    content = message.content
    if isinstance(content, str):
        return content
    if isinstance(content, list):
        parts: list[str] = []
        for block in content:
            if isinstance(block, str):
                parts.append(block)
            elif isinstance(block, dict) and block.get("type") == "text":
                parts.append(str(block.get("text", "")))
        return "".join(parts)
    return str(content)

async def retrieve(state):
    """Embed the latest user message and store the blended biographies."""
    query = latest_user_text(state.get("messages") or [])
    if not query:
        return {"lore_context": ""}
    publish_activity(lore_running(query))
    try:
        hits = await asyncio.to_thread(embedding_model.outfit_search, query, LORE_LIMIT)
    except Exception:
        logger.exception("lore retrieval failed")
        publish_activity(lore_failed())
        return {"lore_context": ""}
    if not hits:
        publish_activity(lore_missed())
        return {"lore_context": ""}
    publish_activity(lore_found(hits))
    return {"lore_context": format_lore(hits)}


async def call_model(state, config):
    system = system_of()
    history = state["messages"][-max_messages_of(config):]
    messages = [SystemMessage(content=system)] + history
    model_with_tools = model.bind_tools(get_tools())
    response = await model_with_tools.ainvoke(messages)
    return {"messages": [response]}


def trim_messages_node(state, config):
    messages = state["messages"]
    limit = max_messages_of(config)
    if len(messages) <= limit:
        return {}
    safe_start = 0
    for idx in range(len(messages) - 1, -1, -1):
        if messages[idx].type == "human":
            safe_start = idx
            break
    cutoff = max(0, safe_start - (limit - (len(messages) - safe_start)))
    to_remove = messages[:cutoff]
    if not to_remove:
        return {}
    return {"messages": [RemoveMessage(id=m.id) for m in to_remove]}


def weather_calls(state) -> list[dict[str, Any]]:
    """Return weather tool calls on the latest model message."""
    messages = state.get("messages") or []
    if not messages:
        return []
    calls = getattr(messages[-1], "tool_calls", None) or []
    return [
        call
        for call in calls
        if isinstance(call, dict) and (call.get("name") == "get_weather" or call.get("name") == "get_weather_by_ip")
    ]


def weather_call_query(call: dict[str, Any]) -> str:
    """Return the place argument of one weather call."""
    args = call.get("args") or {}
    if not isinstance(args, dict):
        return ""
    return str(args.get("query") or "").strip()


async def run_tools(state, config, **kwargs):
    """Run tool calls and publish weather progress for the chat UI."""
    queries: dict[str, str] = {}
    for index, call in enumerate(weather_calls(state)):
        identifier = str(call.get("id") or "").strip() or f"get_weather-{index}"
        query = weather_call_query(call)
        queries[identifier] = query
        publish_activity(weather_running(identifier, query))
    try:
        result = await ToolNode(get_tools()).ainvoke(state, config, **kwargs)
    except Exception:
        for identifier, query in queries.items():
            publish_activity(weather_unfinished(identifier, query))
        raise
    finished: set[str] = set()
    messages = result.get("messages") if isinstance(result, dict) else None
    for message in messages or []:
        if getattr(message, "name", None) not in WEATHER_TOOLS:
            continue
        identifier = str(getattr(message, "tool_call_id", "") or "")
        if identifier not in queries:
            continue
        finished.add(identifier)
        publish_activity(
            weather_finished(identifier, queries[identifier], message_text(message))
        )
    for identifier, query in queries.items():
        if identifier not in finished:
            publish_activity(weather_unfinished(identifier, query))
    return result


workflow = StateGraph(AgentState)
workflow.add_node("retrieve", retrieve)
workflow.add_node("agent", call_model)
workflow.add_node("tools", run_tools)
workflow.add_node("trim", trim_messages_node)
workflow.add_edge(START, "retrieve")
workflow.add_edge("retrieve", "agent")
workflow.add_edge("agent", "trim")
workflow.add_conditional_edges("trim", should_continue, ["tools", END])
workflow.add_edge("tools", "agent")

# LangGraph API injects its own checkpointer. A custom one fails `langgraph dev`.
graph = workflow.compile()
# The HTTP API keeps its own in-process memory, keyed by session_id.
_api_graph = workflow.compile(checkpointer=InMemorySaver())


async def run_agent(input, config=None, **kwargs):
    """Non-streaming entry: guarantees thread_id."""
    return await _api_graph.ainvoke(input, config, **kwargs)


def stream_part(item: object) -> tuple[str, object] | None:
    """Return ``(mode, payload)`` for one multi-mode graph event."""
    if not isinstance(item, tuple):
        return None
    if len(item) == 2 and isinstance(item[0], str):
        return item[0], item[1]
    if len(item) == 3 and isinstance(item[1], str):
        return item[1], item[2]
    return None


async def outfit_stream_agent(input, config=None, **kwargs):
    """Stream answer tokens and activity events for one turn."""
    kwargs.pop("stream_mode", None)
    configurable = dict(config.get("configurable") or {})
    configurable.setdefault("thread_id", thread_id_of({"configurable": configurable}))
    config["configurable"] = configurable
    async for item in _api_graph.astream(
        input,
        config,
        stream_mode=["messages", "custom"],
        **kwargs,
    ):
        part = stream_part(item)
        if part is None:
            continue
        mode, payload = part
        if mode == "custom":
            if isinstance(payload, dict) and payload.get("type") == "activity":
                yield payload
            continue
        if mode == "messages":
            yield payload