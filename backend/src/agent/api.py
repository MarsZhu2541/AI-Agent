"""HTTP API in front of the chat graph."""

from __future__ import annotations

from collections.abc import AsyncIterator

from fastapi import FastAPI
from fastapi.middleware.cors import CORSMiddleware
from fastapi.responses import StreamingResponse
from langchain_core.messages import (
    AIMessage,
    AIMessageChunk,
    BaseMessage,
    HumanMessage,
    SystemMessage,
)
from pydantic import BaseModel, Field

from agent.graph import graph

app = FastAPI(title="assistant-ui agent")
app.add_middleware(
    CORSMiddleware,
    allow_origins=[
        "http://localhost:5173",
        "http://127.0.0.1:5173",
        "http://localhost:5174",
        "http://127.0.0.1:5174",
    ],
    allow_methods=["POST"],
    allow_headers=["*"],
)


class ChatMessage(BaseModel):
    role: str
    content: str


class ChatRequest(BaseModel):
    messages: list[ChatMessage] = Field(default_factory=list)


def to_langchain(messages: list[ChatMessage]) -> list[BaseMessage]:
    """Convert the chat request into LangChain messages."""
    converted: list[BaseMessage] = []
    for message in messages:
        if message.role == "assistant":
            converted.append(AIMessage(content=message.content))
        elif message.role == "system":
            converted.append(SystemMessage(content=message.content))
        else:
            converted.append(HumanMessage(content=message.content))
    return converted


def chunk_text(chunk: AIMessage | AIMessageChunk) -> str:
    """Read text from a model chunk."""
    content = chunk.content
    if isinstance(content, str):
        return content
    parts: list[str] = []
    for block in content:
        if isinstance(block, str):
            parts.append(block)
        elif isinstance(block, dict) and block.get("type") == "text":
            parts.append(str(block.get("text", "")))
    return "".join(parts)


@app.post("/api/chat")
async def chat(body: ChatRequest) -> StreamingResponse:
    """Stream the graph reply as plain text."""

    async def stream() -> AsyncIterator[str]:
        async for item in graph.astream(
            {"messages": to_langchain(body.messages)},
            stream_mode="messages",
        ):
            chunk = item[0] if isinstance(item, tuple) else item
            if not isinstance(chunk, (AIMessage, AIMessageChunk)):
                continue
            text = chunk_text(chunk)
            if text:
                yield text

    return StreamingResponse(stream(), media_type="text/plain; charset=utf-8")
