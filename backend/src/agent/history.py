"""Persist the current chat thread in MySQL.

Each completed chat replaces the stored rows with the full transcript the
client just sent, plus the assistant reply. The table is one conversation,
so loading it on startup restores that thread instead of stacking every
past snapshot.
"""

from __future__ import annotations

import asyncio
import json
import os
import re
from datetime import datetime
from typing import Any

import pymysql
from pymysql.cursors import DictCursor

_IDENTIFIER = re.compile(r"^[A-Za-z_][A-Za-z0-9_]{0,63}$")
_ready = False

_CREATE_TABLE = """
CREATE TABLE IF NOT EXISTS chat_message (
    id BIGINT NOT NULL AUTO_INCREMENT,
    seq INT NOT NULL,
    role VARCHAR(32) NOT NULL,
    content LONGTEXT NOT NULL,
    created_at TIMESTAMP NOT NULL DEFAULT CURRENT_TIMESTAMP,
    PRIMARY KEY (id),
    KEY idx_chat_message_seq (seq)
) ENGINE=InnoDB DEFAULT CHARSET=utf8mb4 COLLATE=utf8mb4_unicode_ci
"""


def _settings() -> tuple[str, int, str, str, str]:
    """Read the MySQL connection settings."""
    host = os.environ.get("MYSQL_HOST", "").strip()
    user = os.environ.get("MYSQL_USER", "").strip()
    password = os.environ.get("MYSQL_PASSWORD", "")
    database = os.environ.get("MYSQL_DATABASE", "ai_agent").strip() or "ai_agent"
    port = int(os.environ.get("MYSQL_PORT", "3306"))
    if not host or not user:
        raise RuntimeError("MYSQL_HOST and MYSQL_USER are required")
    if not _IDENTIFIER.fullmatch(database):
        raise RuntimeError("MYSQL_DATABASE must be a plain identifier")
    return host, port, user, password, database


def _connect(database: str | None) -> pymysql.Connection:
    """Open one MySQL connection."""
    host, port, user, password, default_database = _settings()
    return pymysql.connect(
        host=host,
        port=port,
        user=user,
        password=password,
        database=database or default_database,
        charset="utf8mb4",
        connect_timeout=5,
        read_timeout=30,
        write_timeout=60,
        autocommit=False,
        cursorclass=DictCursor,
    )


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


def _init_sync() -> None:
    host, port, user, password, database = _settings()
    bootstrap = pymysql.connect(
        host=host,
        port=port,
        user=user,
        password=password,
        charset="utf8mb4",
        connect_timeout=5,
        autocommit=True,
    )
    try:
        with bootstrap.cursor() as cursor:
            cursor.execute(
                f"CREATE DATABASE IF NOT EXISTS `{database}` "
                "CHARACTER SET utf8mb4 COLLATE utf8mb4_unicode_ci"
            )
    finally:
        bootstrap.close()

    connection = _connect(database)
    try:
        with connection.cursor() as cursor:
            cursor.execute(_CREATE_TABLE)
        connection.commit()
    finally:
        connection.close()


def _replace_sync(messages: list[dict[str, Any]]) -> None:
    connection = _connect(None)
    try:
        with connection.cursor() as cursor:
            cursor.execute("DELETE FROM chat_message")
            if messages:
                cursor.executemany(
                    "INSERT INTO chat_message (seq, role, content) VALUES (%s, %s, %s)",
                    [
                        (
                            index,
                            message["role"],
                            json.dumps(message["content"], ensure_ascii=False),
                        )
                        for index, message in enumerate(messages)
                    ],
                )
        connection.commit()
    except Exception:
        connection.rollback()
        raise
    finally:
        connection.close()


def _list_sync() -> list[dict[str, Any]]:
    connection = _connect(None)
    try:
        with connection.cursor() as cursor:
            cursor.execute(
                "SELECT id, role, content, created_at "
                "FROM chat_message ORDER BY seq ASC, id ASC"
            )
            rows = cursor.fetchall()
        return [message_from_row(row) for row in rows]
    finally:
        connection.close()


async def init_db() -> None:
    """Create the database and the chat message table."""
    global _ready
    await asyncio.to_thread(_init_sync)
    _ready = True


async def ensure_db() -> None:
    """Connect and create the table, including after a failed startup attempt."""
    if not _ready:
        await init_db()


async def replace_messages(messages: list[dict[str, Any]]) -> None:
    """Replace the stored thread with this complete transcript."""
    await ensure_db()
    await asyncio.to_thread(_replace_sync, messages)


async def list_messages() -> list[dict[str, Any]]:
    """Return every stored message in conversation order."""
    await ensure_db()
    return await asyncio.to_thread(_list_sync)
