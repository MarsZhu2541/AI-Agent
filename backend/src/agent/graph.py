"""Chat graph backed by OpenAI."""

from __future__ import annotations

import asyncio
import logging
from typing import Any

from langchain_core.messages import HumanMessage, RemoveMessage, SystemMessage
from langchain_core.tools import BaseTool
from langchain_openai import ChatOpenAI
from langgraph.checkpoint.memory import InMemorySaver
from langgraph.errors import NodeInterrupt
from langgraph.graph import END, StateGraph
from langgraph.prebuilt import ToolNode
from pydantic import BaseModel

from .embeddings import embedding_model
from .state import AgentState
from .tools import tools

logger = logging.getLogger(__name__)

model = ChatOpenAI(model="deepseek-v4.1-flash")

_FILE_INSTRUCTIONS = (
    "当用户想要文件时，调用 prepare_download 生成文本文件。"
    "文件要等用户在页面上确认后才会下载，回复里不要贴出整份文件。"
)

LORE_SYSTEM = (
    "你是《英雄联盟》背景故事百科全书，负责讲述符文之地的英雄、地区和传说。"
    "下面是这一轮检索到的英雄传记。回答故事时只依据这些传记；"
    "传记没有记载的情节不要编造，直接说明资料里没有。"
    "回复消息中不要明说或者暗示检索到文档/传记的存在。"
    + _FILE_INSTRUCTIONS
)

GENERAL_SYSTEM = (
    "这一轮问题与《英雄联盟》背景故事无关，按普通助手回答。"
    "不要引用英雄传记，也不要说资料里没有记载。"
    + _FILE_INSTRUCTIONS
)

LORE_LIMIT = 4

_LORE_ROUTE = (
    "判断用户这句话是否涉及《英雄联盟》或符文之地，包括英雄、背景故事、地区、阵营和世界观。"
    "只要涉及，就回答 yes，即使用户同时还要求写文件或搜索。"
    "与英雄联盟无关的问题回答 no。"
    "只回答 yes 或 no。"
)

DEFAULT_THREAD_ID = "default"

MAX_MESSAGES = 20


def configurable_of(config) -> dict:
    if not isinstance(config, dict):
        return {}
    configurable = config.get("configurable") or {}
    return configurable if isinstance(configurable, dict) else {}


def thread_id_of(config) -> str:
    session_id = configurable_of(config).get("session_id")
    if session_id is None or str(session_id).strip() == "":
        return DEFAULT_THREAD_ID
    return str(session_id)


def max_messages_of(config) -> int:
    value = configurable_of(config).get("max_messages")
    try:
        value = int(value)
    except (TypeError, ValueError):
        return MAX_MESSAGES
    return value if value > 0 else MAX_MESSAGES


def with_thread_id(config) -> dict:
    """Copy config and guarantee configurable.thread_id exists."""
    config = dict(config or {})
    configurable = dict(config.get("configurable") or {})
    if not configurable.get("thread_id"):
        configurable["thread_id"] = thread_id_of(config)
    config["configurable"] = configurable
    return config


def should_continue(state):
    messages = state["messages"]
    last_message = messages[-1]
    if not last_message.tool_calls:
        return END
    else:
        return "tools"


class AnyArgsSchema(BaseModel):
    class Config:
        extra = "allow"


class FrontendTool(BaseTool):
    def __init__(self, name: str):
        super().__init__(name=name, description="", args_schema=AnyArgsSchema)

    def _run(self, *args, **kwargs):
        raise NodeInterrupt("This is a frontend tool call")

    async def _arun(self, *args, **kwargs) -> str:
        raise NodeInterrupt("This is a frontend tool call")


def get_tool_defs(config):
    frontend_tools = configurable_of(config).get("frontend_tools") or []
    return tools + [
        {"type": "function", "function": tool} for tool in frontend_tools
    ]


def get_tools(config):
    frontend_tools = configurable_of(config).get("frontend_tools") or []
    return tools + [FrontendTool(tool.name) for tool in frontend_tools]


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
    blocks = ["检索到的英雄传记片段："]
    for index, hit in enumerate(hits, start=1):
        name = str(hit.get("name") or "")
        title = str(hit.get("title") or "")
        heading = f"{name}（{title}）" if name and title else name or title or str(hit.get("id") or "")
        blocks.append(f"{index}. {heading}\n{hit.get('text') or ''}")
    return "\n\n".join(blocks)


def system_of(config, lore: str) -> str:
    """Use the lore encyclopedia prompt only when biographies were retrieved."""
    custom = configurable_of(config).get("system")
    lore = lore.strip()
    if isinstance(custom, str) and custom.strip():
        system = custom.strip()
    elif lore:
        system = LORE_SYSTEM
    else:
        system = GENERAL_SYSTEM
    if not lore:
        return system
    return f"{system}\n\n{lore}"


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


def lore_route_answer(text: str) -> bool:
    """Treat an exact yes as a League of Legends question."""
    token = text.strip().lower().strip("。.")
    token = token.split()[0].split("，")[0].split(",")[0] if token else ""
    return token in {"yes", "y", "是"}


async def lol_related(query: str) -> bool:
    """Ask the chat model whether this question needs champion lore."""
    try:
        response = await model.ainvoke(
            [
                SystemMessage(content=_LORE_ROUTE),
                HumanMessage(content=query),
            ]
        )
    except Exception:
        logger.exception("lore route failed")
        return False
    return lore_route_answer(message_text(response))


async def route_question(state) -> str:
    """Send League of Legends questions to retrieval and skip the rest."""
    query = latest_user_text(state.get("messages") or [])
    if query and await lol_related(query):
        return "retrieve"
    return "skip_retrieve"


async def skip_retrieve(state, config):
    """Drop lore from an earlier turn when this question is unrelated."""
    del state, config
    return {"lore_context": ""}


async def retrieve(state, config):
    """Embed the latest user message and store the nearest biographies."""
    del config
    query = latest_user_text(state.get("messages") or [])
    if not query:
        return {"lore_context": ""}
    try:
        hits = await asyncio.to_thread(embedding_model.search, query, LORE_LIMIT)
    except Exception:
        logger.exception("lore retrieval failed")
        return {"lore_context": ""}
    return {"lore_context": format_lore(hits)}


async def call_model(state, config):
    system = system_of(config, state.get("lore_context") or "")
    history = state["messages"][-max_messages_of(config):]
    messages = [SystemMessage(content=system)] + history
    model_with_tools = model.bind_tools(get_tool_defs(config))
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


async def run_tools(input, config, **kwargs):
    tool_node = ToolNode(get_tools(config))
    return await tool_node.ainvoke(input, config, **kwargs)


workflow = StateGraph(AgentState)
workflow.add_node("retrieve", retrieve)
workflow.add_node("skip_retrieve", skip_retrieve)
workflow.add_node("agent", call_model)
workflow.add_node("tools", run_tools)
workflow.add_node("trim", trim_messages_node)
workflow.set_conditional_entry_point(
    route_question, ["retrieve", "skip_retrieve"]
)
workflow.add_edge("retrieve", "agent")
workflow.add_edge("skip_retrieve", "agent")
workflow.add_edge("agent", "trim")
workflow.add_conditional_edges("trim", should_continue, ["tools", END])
workflow.add_edge("tools", "agent")

graph = workflow.compile(checkpointer=InMemorySaver())


async def run_agent(input, config=None, **kwargs):
    """Non-streaming entry: guarantees thread_id."""
    return await graph.ainvoke(input, with_thread_id(config), **kwargs)


async def stream_agent(input, config=None, **kwargs):
    """Streaming entry: guarantees thread_id for checkpointer."""
    async for item in graph.astream(input, with_thread_id(config), **kwargs):
        yield item