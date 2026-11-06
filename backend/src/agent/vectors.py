"""Read and write embedding rows in PostgreSQL."""

from __future__ import annotations

import json
import re
from typing import Any

from agent import db

_RRF_K = 60
_CANDIDATES = 20
_KEYWORD_MIN_SCORE = 2
_CJK = re.compile(r"[\u4e00-\u9fff]+")
_LATIN = re.compile(r"[A-Za-z][A-Za-z0-9]{1,}")
_STOPWORDS = frozenset(
    {
        "为什么",
        "是什么",
        "什么",
        "怎么",
        "如何",
        "为何",
        "哪些",
        "哪个",
        "这个",
        "那个",
        "我们",
        "他们",
        "可以",
        "不是",
        "不会",
        "因为",
        "所以",
        "如果",
        "已经",
        "自己",
        "知道",
        "介绍",
        "背景",
        "故事",
        "英雄",
        "联盟",
    }
)

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


def keyword_terms(query: str) -> list[str]:
    """Extract terms that can be matched against a biography.

    A short name is kept whole. Longer Chinese text contributes 3-character
    pieces, and question words such as 为什么 are left out.
    """
    found: list[str] = []
    for match in _LATIN.finditer(query):
        found.append(match.group())
    for match in _CJK.finditer(query):
        piece = match.group()
        if _stopped(piece, query):
            continue
        if len(piece) <= 8:
            found.append(piece)
        if len(piece) < 3:
            continue
        for index in range(len(piece) - 2):
            gram = piece[index : index + 3]
            if not _stopped(gram, query):
                found.append(gram)
    unique: list[str] = []
    seen: set[str] = set()
    for term in found:
        if term in seen:
            continue
        seen.add(term)
        unique.append(term)
    return unique


def fuse_hits(
    vector_rows: list[dict[str, Any]],
    keyword_rows: list[dict[str, Any]],
    limit: int,
) -> list[dict[str, Any]]:
    """Combine vector and keyword ranks with reciprocal rank fusion."""
    if limit <= 0:
        raise ValueError("limit 必须大于 0")
    scores: dict[str, float] = {}
    rows: dict[str, dict[str, Any]] = {}
    for rank, row in enumerate(vector_rows, start=1):
        chunk_id = str(row["id"])
        rows.setdefault(chunk_id, row)
        scores[chunk_id] = scores.get(chunk_id, 0.0) + 1 / (_RRF_K + rank)
    for rank, row in enumerate(keyword_rows, start=1):
        chunk_id = str(row["id"])
        rows.setdefault(chunk_id, row)
        scores[chunk_id] = scores.get(chunk_id, 0.0) + 1 / (_RRF_K + rank)
    ranked = sorted(
        rows,
        key=lambda chunk_id: (
            -scores[chunk_id],
            float(rows[chunk_id].get("distance", 2.0)),
        ),
    )
    fused: list[dict[str, Any]] = []
    for chunk_id in ranked[:limit]:
        item = dict(rows[chunk_id])
        item["score"] = scores[chunk_id]
        fused.append(item)
    return fused


def _stopped(term: str, query: str) -> bool:
    if term in _STOPWORDS:
        return True
    return any(len(stop) > len(term) and term in stop and stop in query for stop in _STOPWORDS)


def _filters(
    slug: str | None,
    faction_slug: str | None,
    params: dict[str, Any],
) -> str:
    conditions: list[str] = []
    if slug:
        conditions.append("slug = %(slug)s")
        params["slug"] = slug
    if faction_slug:
        conditions.append("faction_slug = %(faction_slug)s")
        params["faction_slug"] = faction_slug
    if not conditions:
        return ""
    return f"WHERE {' AND '.join(conditions)}"


def _vector_hits(
    embedding: list[float],
    limit: int,
    *,
    slug: str | None,
    faction_slug: str | None,
) -> list[dict[str, Any]]:
    params: dict[str, Any] = {
        "embedding": vector_literal(embedding),
        "limit": limit,
    }
    where = _filters(slug, faction_slug, params)
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


def _keyword_hits(
    query: str,
    limit: int,
    *,
    slug: str | None,
    faction_slug: str | None,
) -> list[dict[str, Any]]:
    params: dict[str, Any] = {
        "query": query,
        "terms": keyword_terms(query),
        "limit": limit,
        "minimum": _KEYWORD_MIN_SCORE,
    }
    where = _filters(slug, faction_slug, params)
    with db.connection() as conn:
        rows = conn.execute(
            f"""
            SELECT id, slug, name, title, chunk_index, chunk_total, text,
                   faction_slug, metadata, model, keyword_score
            FROM (
                SELECT id, slug, name, title, chunk_index, chunk_total, text,
                       faction_slug, metadata, model,
                       (
                         CASE
                           WHEN char_length(name) >= 2
                            AND strpos(%(query)s, name) > 0 THEN 3
                           ELSE 0
                         END
                         + CASE
                           WHEN char_length(title) >= 2
                            AND strpos(%(query)s, title) > 0 THEN 2
                           ELSE 0
                         END
                         + COALESCE((
                           SELECT count(DISTINCT term)::int
                           FROM unnest(%(terms)s::text[]) AS keyword_term(term)
                           WHERE strpos(lore_chunk.text, term) > 0
                         ), 0)
                       ) AS keyword_score
                FROM lore_chunk
                {where}
            ) AS scored
            WHERE keyword_score >= %(minimum)s
            ORDER BY keyword_score DESC, char_length(text)
            LIMIT %(limit)s
            """,
            params,
        ).fetchall()
    return [_public_row(row, include_embedding=False) for row in rows]


def search_embeddings(
    embedding: list[float],
    limit: int = 5,
    *,
    query: str | None = None,
    slug: str | None = None,
    faction_slug: str | None = None,
) -> list[dict[str, Any]]:
    """Return chunks by blending cosine distance with keyword matches.

    Without a query, ranking uses cosine distance alone.
    """
    if limit <= 0:
        raise ValueError("limit 必须大于 0")
    dimensions = db.embedding_dimensions()
    if len(embedding) != dimensions:
        raise ValueError(f"embedding 维度是 {len(embedding)}，数据表需要 {dimensions}")
    text = (query or "").strip()
    if not text:
        return _vector_hits(embedding, limit, slug=slug, faction_slug=faction_slug)
    vector_rows = _vector_hits(
        embedding,
        _CANDIDATES,
        slug=slug,
        faction_slug=faction_slug,
    )
    keyword_rows = _keyword_hits(
        text,
        _CANDIDATES,
        slug=slug,
        faction_slug=faction_slug,
    )
    return fuse_hits(vector_rows, keyword_rows, limit)
