"""PostgreSQL connections and schema for chat history and embeddings."""

from __future__ import annotations

import os
import re
import threading
from collections.abc import Iterator
from contextlib import contextmanager
from pathlib import Path
from typing import Any

import psycopg
from dotenv import load_dotenv
from psycopg import sql
from psycopg.rows import DictRow, dict_row
from sqlalchemy import create_engine
from sqlalchemy.engine import URL, Engine
from sqlalchemy.orm import Session

load_dotenv(Path(__file__).resolve().parents[2] / ".env")

_IDENTIFIER = re.compile(r"^[A-Za-z_][A-Za-z0-9_]{0,63}$")
_LOCK = threading.Lock()
_ready = False
_engine: Engine | None = None
EMBEDDING_DIMENSIONS = 768

_CREATE_LORE = """
CREATE TABLE IF NOT EXISTS lore_chunk (
    id TEXT PRIMARY KEY,
    slug TEXT NOT NULL DEFAULT '',
    name TEXT NOT NULL DEFAULT '',
    title TEXT NOT NULL DEFAULT '',
    chunk_index INTEGER NOT NULL DEFAULT 0,
    chunk_total INTEGER NOT NULL DEFAULT 1,
    text TEXT NOT NULL,
    faction_slug TEXT NOT NULL DEFAULT '',
    metadata JSONB NOT NULL DEFAULT '{{}}'::jsonb,
    embedding vector({dimensions}) NOT NULL,
    model TEXT NOT NULL,
    updated_at TIMESTAMPTZ NOT NULL DEFAULT CURRENT_TIMESTAMP
)
"""

_CREATE_LORE_INDEXES = (
    "CREATE INDEX IF NOT EXISTS idx_lore_chunk_slug ON lore_chunk (slug)",
    "CREATE INDEX IF NOT EXISTS idx_lore_chunk_faction ON lore_chunk (faction_slug)",
    """
    CREATE INDEX IF NOT EXISTS idx_lore_chunk_embedding
    ON lore_chunk USING hnsw (embedding vector_cosine_ops)
    """,
)


def embedding_dimensions() -> int:
    """Return the configured vector size, defaulting to the current model."""
    raw = os.environ.get("OPENAI_EMBEDDING_DIMENSIONS", "").strip()
    if not raw:
        return EMBEDDING_DIMENSIONS
    try:
        dimensions = int(raw)
    except ValueError as exc:
        raise RuntimeError("OPENAI_EMBEDDING_DIMENSIONS must be an integer") from exc
    if dimensions <= 0:
        raise RuntimeError("OPENAI_EMBEDDING_DIMENSIONS must be positive")
    return dimensions


def settings() -> tuple[str, int, str, str, str]:
    """Read the PostgreSQL connection settings."""
    host = os.environ.get("POSTGRES_HOST", "").strip()
    user = os.environ.get("POSTGRES_USER", "").strip()
    password = os.environ.get("POSTGRES_PASSWORD", "")
    database = os.environ.get("POSTGRES_DATABASE", "ai_agent").strip() or "ai_agent"
    raw_port = os.environ.get("POSTGRES_PORT", "5432").strip() or "5432"
    if not host or not user:
        raise RuntimeError("POSTGRES_HOST and POSTGRES_USER are required")
    if not _IDENTIFIER.fullmatch(database):
        raise RuntimeError("POSTGRES_DATABASE must be a plain identifier")
    try:
        port = int(raw_port)
    except ValueError as exc:
        raise RuntimeError("POSTGRES_PORT must be an integer") from exc
    return host, port, user, password, database


def connect(
    database: str | None = None,
    *,
    autocommit: bool = False,
) -> psycopg.Connection[DictRow]:
    """Open one PostgreSQL connection."""
    host, port, user, password, default_database = settings()
    return psycopg.connect(
        host=host,
        port=port,
        user=user,
        password=password,
        dbname=database or default_database,
        connect_timeout=5,
        autocommit=autocommit,
        row_factory=dict_row,
    )


def _ensure_database(database: str) -> None:
    """Create the application database when it is missing."""
    maintenance = connect("postgres", autocommit=True)
    try:
        exists = maintenance.execute(
            "SELECT 1 FROM pg_database WHERE datname = %s",
            (database,),
        ).fetchone()
        if exists is None:
            maintenance.execute(
                sql.SQL("CREATE DATABASE {}").format(sql.Identifier(database))
            )
    finally:
        maintenance.close()


def _lore_dimensions(conn: psycopg.Connection[DictRow]) -> int | None:
    """Return the existing lore_chunk vector size, or None when the table is new."""
    row: dict[str, Any] | None = conn.execute(
        """
        SELECT a.atttypmod
        FROM pg_attribute a
        JOIN pg_class c ON c.oid = a.attrelid
        JOIN pg_namespace n ON n.oid = c.relnamespace
        WHERE n.nspname = 'public'
          AND c.relname = 'lore_chunk'
          AND a.attname = 'embedding'
          AND NOT a.attisdropped
        """
    ).fetchone()
    if row is None:
        return None
    return int(row["atttypmod"])


def _drop_legacy_chat(conn: psycopg.Connection[DictRow]) -> None:
    """Drop the old message table once, before chat_session exists.

    Earlier rows stored image bytes inside the message JSON. That data is not
    migrated.
    """
    has_session = conn.execute(
        """
        SELECT 1
        FROM information_schema.tables
        WHERE table_schema = 'public' AND table_name = 'chat_session'
        """
    ).fetchone()
    if has_session is not None:
        return
    conn.execute("DROP TABLE IF EXISTS chat_message")


def _create_schema(conn: psycopg.Connection[DictRow]) -> None:
    """Create pgvector and the embedding table."""
    dimensions = embedding_dimensions()
    conn.execute("CREATE EXTENSION IF NOT EXISTS vector")
    _drop_legacy_chat(conn)
    existing = _lore_dimensions(conn)
    if existing is None:
        conn.execute(_CREATE_LORE.format(dimensions=dimensions))
        for statement in _CREATE_LORE_INDEXES:
            conn.execute(statement)
        return
    if existing != dimensions:
        raise RuntimeError(
            f"lore_chunk.embedding is vector({existing}), expected vector({dimensions})"
        )


def init_db_sync() -> None:
    """Create the database, pgvector, and the application tables."""
    global _ready
    with _LOCK:
        _ensure_database(settings()[4])
        conn = connect()
        try:
            _create_schema(conn)
            conn.commit()
        except Exception:
            conn.rollback()
            raise
        finally:
            conn.close()
        from agent.models import Base

        Base.metadata.create_all(get_engine())
        _ready = True


def ensure_sync() -> None:
    """Create the schema once, including after a failed startup attempt."""
    if not _ready:
        init_db_sync()


def get_engine() -> Engine:
    """Return the shared SQLAlchemy engine for the chat tables."""
    global _engine
    if _engine is None:
        host, port, user, password, database = settings()
        _engine = create_engine(
            URL.create(
                "postgresql+psycopg",
                username=user,
                password=password,
                host=host,
                port=port,
                database=database,
            ),
            pool_pre_ping=True,
        )
    return _engine


@contextmanager
def orm_session() -> Iterator[Session]:
    """Yield an ORM session that commits on success and rolls back on error."""
    ensure_sync()
    session = Session(get_engine())
    try:
        yield session
    except Exception:
        session.rollback()
        raise
    else:
        session.commit()
    finally:
        session.close()


@contextmanager
def connection() -> Iterator[psycopg.Connection[DictRow]]:
    """Yield a connection that commits on success and rolls back on error."""
    ensure_sync()
    conn = connect()
    try:
        yield conn
    except Exception:
        conn.rollback()
        raise
    else:
        conn.commit()
    finally:
        conn.close()
