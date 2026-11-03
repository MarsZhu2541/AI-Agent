"""Persist chat threads in PostgreSQL.

Each completed chat replaces the stored rows for that session with the full
transcript the client just sent, plus the assistant reply.
"""

from __future__ import annotations

import asyncio
import json
from datetime import datetime
from typing import Any

from agent import db


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


def _replace_sync(messages: list[dict[str, Any]], session_id: str) -> None:
    with db.connection() as conn:
        conn.execute("DELETE FROM chat_message WHERE session_id = %s", (session_id,))
        if messages:
            with conn.cursor() as cursor:
                cursor.executemany(
                    """
                    INSERT INTO chat_message (session_id, seq, role, content)
                    VALUES (%s, %s, %s, %s)
                    """,
                    [
                        (
                            session_id,
                            index,
                            message["role"],
                            json.dumps(message["content"], ensure_ascii=False),
                        )
                        for index, message in enumerate(messages)
                    ],
                )


def _list_sync(session_id: str) -> list[dict[str, Any]]:
    with db.connection() as conn:
        rows = conn.execute(
            """
            SELECT id, role, content, created_at
            FROM chat_message
            WHERE session_id = %s
            ORDER BY seq ASC, id ASC
            """,
            (session_id,),
        ).fetchall()
    return [message_from_row(row) for row in rows]


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


def _list_sessions_sync() -> list[dict[str, Any]]:
    with db.connection() as conn:
        rows = conn.execute(
            """
            SELECT g.session_id, g.updated_at, m.content
            FROM (
                SELECT session_id, MIN(id) AS first_id, MAX(created_at) AS updated_at
                FROM chat_message
                GROUP BY session_id
            ) g
            JOIN chat_message m ON m.id = g.first_id
            ORDER BY g.updated_at DESC, g.session_id DESC
            """
        ).fetchall()
    sessions: list[dict[str, Any]] = []
    for row in rows:
        updated = row["updated_at"]
        sessions.append(
            {
                "id": row["session_id"],
                "title": _title(json.loads(row["content"])),
                "updatedAt": updated.isoformat(timespec="seconds")
                if isinstance(updated, datetime)
                else str(updated),
            }
        )
    return sessions


def _delete_session_sync(session_id: str) -> None:
    with db.connection() as conn:
        conn.execute("DELETE FROM chat_message WHERE session_id = %s", (session_id,))


async def init_db() -> None:
    """Create the database, pgvector, and the application tables."""
    await asyncio.to_thread(db.init_db_sync)


async def ensure_db() -> None:
    """Connect and create the schema, including after a failed startup attempt."""
    await asyncio.to_thread(db.ensure_sync)


async def replace_messages(messages: list[dict[str, Any]], session_id: str) -> None:
    """Replace the stored thread with this complete transcript."""
    await ensure_db()
    await asyncio.to_thread(_replace_sync, messages, session_id)


async def list_messages(session_id: str) -> list[dict[str, Any]]:
    """Return every stored message in conversation order."""
    await ensure_db()
    return await asyncio.to_thread(_list_sync, session_id)


async def list_sessions() -> list[dict[str, Any]]:
    """Return stored sessions, newest first."""
    await ensure_db()
    return await asyncio.to_thread(_list_sessions_sync)


async def delete_session(session_id: str) -> None:
    """Delete one session and its messages."""
    await ensure_db()
    await asyncio.to_thread(_delete_session_sync, session_id)
