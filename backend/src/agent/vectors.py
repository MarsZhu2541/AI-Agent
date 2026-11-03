"""Read and write embedding rows in PostgreSQL."""

from __future__ import annotations

import json
from typing import Any

from agent import db

_METADATA_KEYS = (
    "header",
    "roles",
    "races",
    "faction",
    "related",
    "quote",
    "quote_author",
    "biography_short",
    "source_url",
    "paragraph_start",
    "paragraph_end",
)


def vector_literal(values: list[float]) -> str:
    """Format a vector for a PostgreSQL cast."""
    return json.dumps(values, separators=(",", ":"))


def parse_vector(value: Any) -> list[float]:
    """Parse a vector returned by PostgreSQL."""
    if isinstance(value, list):
        return [float(item) for item in value]
    text = str(value).strip()
    parsed = json.loads(text)
    if not isinstance(parsed, list):
        raise ValueError("invalid vector")
    return [float(item) for item in parsed]


def _metadata(record: dict[str, Any]) -> dict[str, Any]:
    metadata = record.get("metadata")
    if isinstance(metadata, dict):
        return metadata
    return {key: record[key] for key in _METADATA_KEYS if key in record}


def _parameters(record: dict[str, Any]) -> tuple[Any, ...]:
    chunk_id = str(record.get("id", "")).strip()
    text = record.get("text")
    embedding = record.get("embedding")
    model = str(record.get("model", "")).strip()
    if not chunk_id:
        raise ValueError("每条记录都需要 id")
    if not isinstance(text, str) or not text.strip():
        raise ValueError("每条记录都需要 text")
    if not isinstance(embedding, list) or not embedding:
        raise ValueError("每条记录都需要 embedding")
    dimensions = db.embedding_dimensions()
    if len(embedding) != dimensions:
        raise ValueError(
            f"embedding 维度是 {len(embedding)}，数据表需要 {dimensions}"
        )
    if not model:
        raise ValueError("每条记录都需要 model")
    return (
        chunk_id,
        str(record.get("slug") or ""),
        str(record.get("name") or ""),
        str(record.get("title") or ""),
        int(record.get("chunk_index") or 0),
        int(record.get("chunk_total") or 1),
        text,
        str(record.get("faction_slug") or ""),
        json.dumps(_metadata(record), ensure_ascii=False),
        vector_literal([float(item) for item in embedding]),
        model,
    )


def upsert_embeddings(records: list[dict[str, Any]]) -> None:
    """Insert or update embedding rows."""
    if not records:
        return
    rows = [_parameters(record) for record in records]
    with db.connection() as conn:
        with conn.cursor() as cursor:
            cursor.executemany(
                """
                INSERT INTO lore_chunk (
                    id, slug, name, title, chunk_index, chunk_total, text,
                    faction_slug, metadata, embedding, model
                ) VALUES (
                    %s, %s, %s, %s, %s, %s, %s, %s, %s::jsonb, %s::vector, %s
                )
                ON CONFLICT (id) DO UPDATE SET
                    slug = EXCLUDED.slug,
                    name = EXCLUDED.name,
                    title = EXCLUDED.title,
                    chunk_index = EXCLUDED.chunk_index,
                    chunk_total = EXCLUDED.chunk_total,
                    text = EXCLUDED.text,
                    faction_slug = EXCLUDED.faction_slug,
                    metadata = EXCLUDED.metadata,
                    embedding = EXCLUDED.embedding,
                    model = EXCLUDED.model,
                    updated_at = CURRENT_TIMESTAMP
                """,
                rows,
            )


def _public_row(row: dict[str, Any], *, include_embedding: bool) -> dict[str, Any]:
    metadata = row["metadata"]
    if isinstance(metadata, str):
        metadata = json.loads(metadata)
    item = {
        "id": row["id"],
        "slug": row["slug"],
        "name": row["name"],
        "title": row["title"],
        "chunk_index": row["chunk_index"],
        "chunk_total": row["chunk_total"],
        "text": row["text"],
        "faction_slug": row["faction_slug"],
        "metadata": metadata,
        "model": row["model"],
    }
    if include_embedding:
        item["embedding"] = parse_vector(row["embedding"])
    if "distance" in row and row["distance"] is not None:
        item["distance"] = float(row["distance"])
    return item


def load_embeddings(ids: list[str]) -> list[dict[str, Any]]:
    """Load embedding rows by id, in the requested order."""
    if not ids:
        return []
    with db.connection() as conn:
        rows = conn.execute(
            """
            SELECT id, slug, name, title, chunk_index, chunk_total, text,
                   faction_slug, metadata, embedding, model
            FROM lore_chunk
            WHERE id = ANY(%s)
            """,
            (ids,),
        ).fetchall()
    by_id = {row["id"]: _public_row(row, include_embedding=True) for row in rows}
    return [by_id[chunk_id] for chunk_id in ids if chunk_id in by_id]


def search_embeddings(
    embedding: list[float],
    limit: int = 5,
    *,
    slug: str | None = None,
    faction_slug: str | None = None,
) -> list[dict[str, Any]]:
    """Return the nearest stored chunks by cosine distance."""
    if limit <= 0:
        raise ValueError("limit 必须大于 0")
    dimensions = db.embedding_dimensions()
    if len(embedding) != dimensions:
        raise ValueError(f"embedding 维度是 {len(embedding)}，数据表需要 {dimensions}")
    conditions: list[str] = []
    params: dict[str, Any] = {
        "embedding": vector_literal(embedding),
        "limit": limit,
    }
    if slug:
        conditions.append("slug = %(slug)s")
        params["slug"] = slug
    if faction_slug:
        conditions.append("faction_slug = %(faction_slug)s")
        params["faction_slug"] = faction_slug
    where = f"WHERE {' AND '.join(conditions)}" if conditions else ""
    with db.connection() as conn:
        rows = conn.execute(
            f"""
            SELECT id, slug, name, title, chunk_index, chunk_total, text,
                   faction_slug, metadata, model,
                   embedding <=> %(embedding)s::vector AS distance
            FROM lore_chunk
            {where}
            ORDER BY embedding <=> %(embedding)s::vector
            LIMIT %(limit)s
            """,
            params,
        ).fetchall()
    return [_public_row(row, include_embedding=False) for row in rows]
