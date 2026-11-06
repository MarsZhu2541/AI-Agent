"""HTTP API in front of the chat graph."""

from __future__ import annotations

import json
import logging
from collections.abc import AsyncIterator
from contextlib import asynccontextmanager
from typing import Annotated, Any, Literal

from fastapi import FastAPI, HTTPException
from fastapi.middleware.cors import CORSMiddleware
from fastapi.responses import Response, StreamingResponse
from langchain_core.messages import (
    AIMessage,
    AIMessageChunk,
    BaseMessage,
    HumanMessage,
    SystemMessage,
)
from pydantic import BaseModel, Field

from agent import history
from agent.downloads import (
    PreparedFile,
    bind_prepared_files,
    parse_download,
    reset_prepared_files,
    take_prepared_files,
)
from agent.graph import stream_agent

logger = logging.getLogger(__name__)

MAX_IMAGES = 8
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
    allow_methods=["GET", "POST", "DELETE"],
    allow_headers=["*"],
)


class TextPart(BaseModel):
    type: Literal["text"]
    text: str = ""


class ImagePart(BaseModel):
    type: Literal["image"]
    image: str


class FilePart(BaseModel):
    type: Literal["file"]
    filename: str
    content: str
    mime: str = "text/plain"


ContentPart = Annotated[TextPart | ImagePart | FilePart, Field(discriminator="type")]


class ChatMessage(BaseModel):
    role: str
    content: str | list[ContentPart]


class ChatRequest(BaseModel):
    session_id: str
    messages: list[ChatMessage] = Field(default_factory=list)


def _image_block(url: str) -> dict[str, object]:
    cleaned = url.strip()
    if history.parse_image_id(cleaned) is not None:
        raise ValueError("图片尚未载入")
    if cleaned.startswith(("http://", "https://")):
        if len(cleaned) > 8192:
            raise ValueError("图片链接过长")
        return {"type": "image_url", "image_url": {"url": cleaned}}

    decoded = history.decode_data_image(cleaned)
    if decoded is None:
        raise ValueError("图片必须是 data URL 或 http(s) 链接")
    mime, data = decoded
    return {"type": "image_url", "image_url": {"url": history.data_url(mime, data)}}


async def _messages_for_model(
    messages: list[ChatMessage], session_id: str
) -> list[ChatMessage]:
    """Replace stored image URLs with data URLs before calling the model."""
    prepared: list[ChatMessage] = []
    for message in messages:
        content = message.content
        if not isinstance(content, list):
            prepared.append(message)
            continue
        parts: list[ContentPart] = []
        changed = False
        for part in content:
            if isinstance(part, ImagePart):
                image_id = history.parse_image_id(part.image)
                if image_id is not None:
                    url = await history.image_data_url(image_id, session_id)
                    if url is None:
                        raise ValueError("图片不存在")
                    parts.append(ImagePart(type="image", image=url))
                    changed = True
                    continue
            parts.append(part)
        prepared.append(
            message.model_copy(update={"content": parts}) if changed else message
        )
    return prepared


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
        if isinstance(part, FilePart):
            continue
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


def agent_reply_text(item: object) -> str:
    """Return streamed text from the answer node, skipping the route decision."""
    if not isinstance(item, tuple) or len(item) < 2:
        return ""
    chunk, metadata = item[0], item[1]
    if not isinstance(metadata, dict) or metadata.get("langgraph_node") != "agent":
        return ""
    if not isinstance(chunk, (AIMessage, AIMessageChunk)):
        return ""
    return chunk_text(chunk)


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


def _stored_content(content: Any) -> Any:
    """Drop file parts that no longer pass the download checks."""
    if not isinstance(content, list):
        return content
    kept: list[Any] = []
    for part in content:
        if not isinstance(part, dict) or part.get("type") != "file":
            kept.append(part)
            continue
        parsed = parse_download(str(part.get("filename", "")), str(part.get("content", "")))
        if isinstance(parsed, PreparedFile):
            kept.append(parsed.as_part())
    if not kept:
        return ""
    if len(kept) == 1 and kept[0].get("type") == "text":
        return kept[0]["text"]
    return kept


def _event(payload: dict[str, object]) -> str:
    """Encode one stream event as a single NDJSON line."""
    return json.dumps(payload, ensure_ascii=False) + "\n"


def transcript_of(
    body: ChatRequest,
    reply: str,
    files: list[PreparedFile] | None = None,
) -> list[dict[str, object]]:
    """Build the full thread to store, including the finished reply and files."""
    stored: list[dict[str, object]] = []
    for message in body.messages:
        item = message.model_dump()
        item["content"] = _stored_content(item["content"])
        stored.append(item)
    prepared = list(files or [])
    if prepared:
        parts: list[dict[str, str]] = []
        if reply:
            parts.append({"type": "text", "text": reply})
        parts.extend(file.as_part() for file in prepared)
        stored.append({"role": "assistant", "content": parts})
    elif reply:
        stored.append({"role": "assistant", "content": reply})
    return stored


@app.get("/aiapi/images/{image_id}")
async def image(image_id: int) -> Response:
    """Return one stored image."""
    try:
        found = await history.get_image(image_id)
    except Exception as exc:
        logger.exception("failed to load image")
        raise HTTPException(status_code=503, detail="无法读取图片") from exc
    if found is None:
        raise HTTPException(status_code=404, detail="图片不存在")
    mime, data = found
    return Response(
        content=data,
        media_type=mime,
        headers={"Cache-Control": "public, max-age=31536000, immutable"},
    )


@app.get("/aiapi/messages/{session_id}")
async def messages(session_id: str) -> dict[str, list[dict[str, object]]]:
    try:
        return {"messages": await history.list_messages(session_id)}
    except Exception as exc:
        logger.exception("failed to load session messages")
        raise HTTPException(status_code=503, detail="无法读取会话消息记录") from exc


@app.post("/aiapi/chat")
async def chat(body: ChatRequest) -> StreamingResponse:
    """Stream text and file proposals as NDJSON."""
    try:
        session_id = body.session_id.strip()
        if not session_id:
            raise HTTPException(status_code=400, detail="缺少 session_id")
        messages = to_langchain(await _messages_for_model(body.messages, session_id))

    except ValueError as exc:
        raise HTTPException(status_code=400, detail=str(exc)) from exc

    async def stream() -> AsyncIterator[str]:
        parts: list[str] = []
        files: list[PreparedFile] = []
        token = bind_prepared_files()
        try:
            async for item in stream_agent(
                {"messages": messages},
                {"configurable": {"session_id": session_id}},
                stream_mode="messages",
            ):
                for prepared in take_prepared_files():
                    files.append(prepared)
                    yield _event({"type": "file", **prepared.as_part()})
                text = agent_reply_text(item)
                if text:
                    parts.append(text)
                    yield _event({"type": "text", "text": text})
            for prepared in take_prepared_files():
                files.append(prepared)
                yield _event({"type": "file", **prepared.as_part()})
            reply = "".join(parts)
            if files and not reply:
                reply = "文件已生成，确认后即可下载。"
                parts.append(reply)
                yield _event({"type": "text", "text": reply})
        finally:
            reset_prepared_files(token)

        if not body.messages:
            return
        try:
            await history.replace_messages(transcript_of(body, "".join(parts), files), session_id)
        except Exception:
            logger.exception("failed to save chat history")
    
    return StreamingResponse(stream(), media_type="application/x-ndjson; charset=utf-8")



@app.get("/aiapi/sessions")
async def sessions() -> dict[str, list[dict[str, object]]]:
    try:
        return {"sessions": await history.list_sessions()}
    except Exception as exc:
        logger.exception("failed to load chat history")
        raise HTTPException(status_code=503, detail="无法读取会话记录") from exc

@app.delete("/aiapi/sessions/{session_id}")
async def delete_sessions(session_id: str):
    try:
        await history.delete_session(session_id)
    except Exception as exc:
        logger.exception("failed to delete session")
        raise HTTPException(status_code=503, detail="无法读取消息记录") from exc
