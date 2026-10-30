"""Chat graph backed by OpenAI."""

from __future__ import annotations

from langchain_core.messages import RemoveMessage, SystemMessage
from langchain_core.tools import BaseTool
from langchain_openai import ChatOpenAI
from langgraph.checkpoint.memory import InMemorySaver
from langgraph.errors import NodeInterrupt
from langgraph.graph import END, StateGraph
from langgraph.prebuilt import ToolNode
from pydantic import BaseModel

from .state import AgentState
from .tools import tools

model = ChatOpenAI(model="deepseek-v4.1-flash")

DEFAULT_SYSTEM = "You are a helpful assistant."

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


async def call_model(state, config):
    system = configurable_of(config).get("system") or DEFAULT_SYSTEM
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
workflow.add_node("agent", call_model)
workflow.add_node("tools", run_tools)
workflow.add_node("trim", trim_messages_node)
workflow.set_entry_point("agent")
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