"""Persist chat threads in PostgreSQL.

Sessions live in ``chat_session`` with a title written when the thread is saved.
Image bytes live in ``chat_image``. Message JSON stores ``/aiapi/images/{id}``
instead of a data URL.
"""

from __future__ import annotations

import asyncio
import base64
import json
import re
from collections.abc import Callable
from datetime import datetime
from typing import Any

import agent.data_service as data_service
from agent import db

MAX_IMAGE_BYTES = 16 * 1024 * 1024
ALLOWED_IMAGE_TYPES = {"image/jpeg", "image/png", "image/gif", "image/webp"}
_DATA_URL_RE = re.compile(r"^data:(image/[a-zA-Z0-9.+-]+);base64,([A-Za-z0-9+/=\s]+)$")
_IMAGE_PATH_RE = re.compile(r"^/aiapi/images/(\d+)$")


def image_path(image_id: int) -> str:
    """Return the URL stored on a message and served by the image route."""
    return f"/aiapi/images/{image_id}"


def parse_image_id(url: str) -> int | None:
    """Return the id when ``url`` points at a stored image."""
    match = _IMAGE_PATH_RE.match(url.strip())
    if match is None:
        return None
    return int(match.group(1))


def decode_data_image(url: str) -> tuple[str, bytes] | None:
    """Decode a data URL into mime and bytes.

    Return None when ``url`` is not a data URL. Raise ValueError when it is a
    data URL the server will not store.
    """
    match = _DATA_URL_RE.match(url.strip())
    if match is None:
        return None
    mime = match.group(1).lower()
    if mime == "image/jpg":
        mime = "image/jpeg"
    if mime not in ALLOWED_IMAGE_TYPES:
        raise ValueError(f"不支持的图片格式：{mime}")
    payload = re.sub(r"\s+", "", match.group(2))
    if not payload:
        raise ValueError("图片数据为空")
    try:
        data = base64.b64decode(payload, validate=True)
    except Exception as exc:
        raise ValueError("图片数据无法解析") from exc
    if len(data) > MAX_IMAGE_BYTES:
        raise ValueError(f"图片不能超过 {MAX_IMAGE_BYTES // (1024 * 1024)}MB")
    return mime, data


def data_url(mime: str, data: bytes) -> str:
    """Encode image bytes as a data URL the model can read."""
    encoded = base64.b64encode(data).decode("ascii")
    return f"data:{mime};base64,{encoded}"


def message_from_row(row: dict[str, Any]) -> dict[str, Any]:
    """Turn one stored row into the message the frontend renders."""
    created = row["created_at"]
    created_at = (
        created.isoformat(timespec="seconds")
        if isinstance(created, datetime)
        else str(created)
    )
    return {
        "id": str(row["id"]),
        "role": row["role"],
        "content": json.loads(row["content"]),
        "createdAt": created_at,
    }


def _title(content: Any) -> str:
    if isinstance(content, str):
        text = content
    elif isinstance(content, list):
        text = "".join(
            part.get("text", "")
            for part in content
            if isinstance(part, dict) and part.get("type") == "text"
        )
        if not text.strip() and any(
            isinstance(part, dict) and part.get("type") == "image" for part in content
        ):
            text = "图片"
    else:
        text = ""
    text = " ".join(text.split())
    return text[:30] if text else "新对话"


def session_title(messages: list[dict[str, Any]]) -> str:
    """Return the title to store for this transcript."""
    for message in messages:
        if message.get("role") == "user":
            return _title(message.get("content"))
    return "新对话"


def apply_image_storage(
    messages: list[dict[str, Any]],
    *,
    save_image: Callable[[str, bytes], int],
    image_belongs: Callable[[int], bool],
) -> tuple[list[dict[str, Any]], set[int]]:
    """Copy messages, storing data URLs and keeping ids that still belong here."""
    kept: set[int] = set()
    stored_messages: list[dict[str, Any]] = []
    for message in messages:
        content = message.get("content")
        if not isinstance(content, list):
            stored_messages.append(dict(message))
            continue
        parts: list[Any] = []
        for part in content:
            if not isinstance(part, dict) or part.get("type") != "image":
                parts.append(part)
                continue
            url = str(part.get("image", ""))
            image_id = parse_image_id(url)
            if image_id is not None:
                if not image_belongs(image_id):
                    raise ValueError("图片不存在")
                kept.add(image_id)
                parts.append({**part, "image": image_path(image_id)})
                continue
            if url.startswith(("http://", "https://")):
                parts.append(part)
                continue
            decoded = decode_data_image(url)
            if decoded is None:
                raise ValueError("图片必须是 data URL 或图片地址")
            mime, data = decoded
            image_id = save_image(mime, data)
            kept.add(image_id)
            parts.append({**part, "image": image_path(image_id)})
        stored = dict(message)
        stored["content"] = parts
        stored_messages.append(stored)
    return stored_messages, kept


def _bytes_of(value: Any) -> bytes:
    if isinstance(value, memoryview):
        return value.tobytes()
    return bytes(value)


async def init_db() -> None:
    """Create the database, pgvector, and the application tables."""
    await asyncio.to_thread(db.init_db_sync)


async def ensure_db() -> None:
    """Connect and create the schema, including after a failed startup attempt."""
    await asyncio.to_thread(db.ensure_sync)


async def replace_messages(messages: list[dict[str, Any]], session_id: str) -> None:
    """Replace the stored thread, session title, and image bytes."""
    await ensure_db()
    await asyncio.to_thread(data_service._replace_sync, messages, session_id)


async def list_messages(session_id: str) -> list[dict[str, Any]]:
    """Return every stored message in conversation order."""
    await ensure_db()
    return await asyncio.to_thread(data_service._list_sync, session_id)


async def list_sessions() -> list[dict[str, Any]]:
    """Return stored sessions, newest first."""
    await ensure_db()
    return await asyncio.to_thread(data_service._list_sessions_sync)


async def delete_session(session_id: str) -> None:
    """Delete one session and its messages and images."""
    await ensure_db()
    await asyncio.to_thread(data_service._delete_session_sync, session_id)


async def get_image(image_id: int) -> tuple[str, bytes] | None:
    """Return the mime type and bytes for one stored image."""
    await ensure_db()
    return await asyncio.to_thread(data_service._get_image_sync, image_id)


async def image_data_url(image_id: int, session_id: str) -> str | None:
    """Return a data URL when the image belongs to this session."""
    await ensure_db()
    return await asyncio.to_thread(
        data_service._image_data_url_sync, image_id, session_id
    )
