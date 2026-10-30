"""HTTP API in front of the chat graph."""

from __future__ import annotations

import logging
import re
from collections.abc import AsyncIterator
from contextlib import asynccontextmanager
from typing import Annotated, Literal

from fastapi import FastAPI, HTTPException
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

from agent import history
from agent.graph import stream_agent  # 改为走带 thread_id 的流式入口

logger = logging.getLogger(__name__)

MAX_IMAGE_BYTES = 8 * 1024 * 1024
MAX_IMAGES = 8
ALLOWED_IMAGE_TYPES = {"image/jpeg", "image/png", "image/gif", "image/webp"}
DATA_URL_RE = re.compile(
    r"^data:(image/[a-zA-Z0-9.+-]+);base64,([A-Za-z0-9+/=\s]+)$"
)
IMAGE_PROMPT = "请识别这张图片。"


@asynccontextmanager
async def lifespan(_app: FastAPI) -> AsyncIterator[None]:
    try:
        await history.init_db()
    except Exception:
        logger.exception("chat history database is unavailable")
    yield


app = FastAPI(title="assistant-ui agent", lifespan=lifespan)
app.add_middleware(
    CORSMiddleware,
    allow_origins=[
        "http://localhost:5173",
        "http://127.0.0.1:5173",
        "http://localhost:5174",
        "http://127.0.0.1:5174",
    ],
    allow_methods=["GET", "POST"],
    allow_headers=["*"],
)


class TextPart(BaseModel):
    type: Literal["text"]
    text: str = ""


class ImagePart(BaseModel):
    type: Literal["image"]
    image: str


ContentPart = Annotated[TextPart | ImagePart, Field(discriminator="type")]


class ChatMessage(BaseModel):
    role: str
    content: str | list[ContentPart]


class ChatRequest(BaseModel):
    messages: list[ChatMessage] = Field(default_factory=list)


def _image_block(url: str) -> dict[str, object]:
    cleaned = url.strip()
    if cleaned.startswith(("http://", "https://")):
        if len(cleaned) > 8192:
            raise ValueError("图片链接过长")
        return {"type": "image_url", "image_url": {"url": cleaned}}

    match = DATA_URL_RE.match(cleaned)
    if not match:
        raise ValueError("图片必须是 data URL 或 http(s) 链接")
    mime = match.group(1).lower()
    if mime == "image/jpg":
        mime = "image/jpeg"
    if mime not in ALLOWED_IMAGE_TYPES:
        raise ValueError(f"不支持的图片格式：{mime}")
    payload = re.sub(r"\s+", "", match.group(2))
    if not payload:
        raise ValueError("图片数据为空")
    if len(payload) * 3 // 4 > MAX_IMAGE_BYTES:
        raise ValueError("图片不能超过 8MB")
    return {
        "type": "image_url",
        "image_url": {"url": f"data:{mime};base64,{payload}"},
    }


def _text_content(content: str | list[ContentPart]) -> str:
    if isinstance(content, str):
        return content
    return "".join(part.text for part in content if isinstance(part, TextPart))


def _user_content(content: str | list[ContentPart]) -> str | list[dict[str, object]]:
    if isinstance(content, str):
        return content

    blocks: list[dict[str, object]] = []
    images = 0
    for part in content:
        if isinstance(part, TextPart):
            if part.text:
                blocks.append({"type": "text", "text": part.text})
            continue
        images += 1
        if images > MAX_IMAGES:
            raise ValueError(f"单条消息最多 {MAX_IMAGES} 张图片")
        blocks.append(_image_block(part.image))

    if images == 0:
        return _text_content(content)
    has_text = any(
        block.get("type") == "text" and str(block.get("text", "")).strip()
        for block in blocks
    )
    if not has_text:
        blocks.insert(0, {"type": "text", "text": IMAGE_PROMPT})
    return blocks


def to_langchain(messages: list[ChatMessage]) -> list[BaseMessage]:
    converted: list[BaseMessage] = []
    for message in messages:
        if message.role == "assistant":
            converted.append(AIMessage(content=_text_content(message.content)))
        elif message.role == "system":
            converted.append(SystemMessage(content=_text_content(message.content)))
        else:
            converted.append(HumanMessage(content=_user_content(message.content)))
    return converted


def chunk_text(chunk: AIMessage | AIMessageChunk) -> str:
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


def transcript_of(body: ChatRequest, reply: str) -> list[dict[str, object]]:
    stored = [message.model_dump() for message in body.messages]
    if reply:
        stored.append({"role": "assistant", "content": reply})
    return stored


@app.get("/api/messages")
async def messages() -> dict[str, list[dict[str, object]]]:
    try:
        return {"messages": await history.list_messages()}
    except Exception as exc:
        logger.exception("failed to load chat history")
        raise HTTPException(status_code=503, detail="无法读取消息记录") from exc


@app.post("/api/chat")
async def chat(body: ChatRequest) -> StreamingResponse:
    """Stream the graph reply as plain text."""
    try:
        messages = to_langchain(body.messages)
    except ValueError as exc:
        raise HTTPException(status_code=400, detail=str(exc)) from exc

    async def stream() -> AsyncIterator[str]:
        parts: list[str] = []
        # 关键改动：走 stream_agent，它会补齐 thread_id
        async for item in stream_agent(
            {"messages": messages},
            stream_mode="messages",
        ):
            chunk = item[0] if isinstance(item, tuple) else item
            if not isinstance(chunk, (AIMessage, AIMessageChunk)):
                continue
            text = chunk_text(chunk)
            if text:
                parts.append(text)
                yield text

        if not body.messages:
            return
        try:
            await history.replace_messages(transcript_of(body, "".join(parts)))
        except Exception:
            logger.exception("failed to save chat history")

    return StreamingResponse(stream(), media_type="text/plain; charset=utf-8")