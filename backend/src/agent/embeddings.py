"""OpenAI-compatible text embeddings configured from the environment."""

from __future__ import annotations

import os
from collections.abc import Sequence
from pathlib import Path
from typing import Any

from dotenv import load_dotenv
from langchain_openai import OpenAIEmbeddings

from agent import vectors

load_dotenv(Path(__file__).resolve().parents[2] / ".env")

_SAVE_BATCH = 16

_API_KEY = "OPENAI_EMBEDDING_API_KEY"
_BASE_URL = "OPENAI_EMBEDDING_BASE_URL"
_MODEL = "OPENAI_EMBEDDING_MODEL"


def _embedding_base_url(value: str) -> str:
    """Return the API root. The OpenAI client appends /embeddings itself."""
    url = value.rstrip("/")
    suffix = "/embeddings"
    if url.endswith(suffix):
        url = url[: -len(suffix)]
    return url


def _settings() -> tuple[str, str, str]:
    """Read the embedding key, base URL, and model name."""
    values = {
        _API_KEY: os.getenv(_API_KEY, "").strip(),
        _BASE_URL: os.getenv(_BASE_URL, "").strip(),
        _MODEL: os.getenv(_MODEL, "").strip(),
    }
    missing = [name for name, value in values.items() if not value]
    if missing:
        raise RuntimeError("缺少 embedding 配置：" + ", ".join(missing))
    return values[_API_KEY], _embedding_base_url(values[_BASE_URL]), values[_MODEL]


class EmbeddingModel:
    """Embedding client that reads OPENAI_EMBEDDING_* on first use."""

    def __init__(self, client: OpenAIEmbeddings | None = None) -> None:
        """Store an optional client so tests can skip the network."""
        self._client = client

    def _client_or_create(self) -> OpenAIEmbeddings:
        if self._client is None:
            api_key, base_url, model = _settings()
            self._client = OpenAIEmbeddings(
                model=model,
                api_key=api_key,
                base_url=base_url,
                check_embedding_ctx_length=False,
                tiktoken_enabled=False,
            )
        return self._client

    @property
    def model(self) -> str:
        """Return the configured embedding model name."""
        return _settings()[2]

    def embed(self, texts: str | Sequence[str]) -> list[list[float]]:
        """Embed one string or a batch of strings.

        Args:
            texts: A single text, or several texts in input order.

        Returns:
            One vector for each input text.
        """
        batch = [texts] if isinstance(texts, str) else list(texts)
        if not batch:
            raise ValueError("embed 至少需要一段文本")
        return self._client_or_create().embed_documents(batch)

    def save(self, records: Sequence[dict[str, Any]]) -> int:
        """Embed each record and upsert it into PostgreSQL.

        Args:
            records: Chunk records. Each one needs `id` and `text`.

        Returns:
            The number of rows written.
        """
        rows = list(records)
        if not rows:
            return 0
        model = _settings()[2]
        saved = 0
        for start in range(0, len(rows), _SAVE_BATCH):
            batch = rows[start : start + _SAVE_BATCH]
            vectors_for_batch = self.embed([str(record["text"]) for record in batch])
            payload = [
                {**record, "embedding": vector, "model": model}
                for record, vector in zip(batch, vectors_for_batch, strict=True)
            ]
            vectors.upsert_embeddings(payload)
            saved += len(payload)
        return saved

    def load(self, ids: str | Sequence[str]) -> list[dict[str, Any]]:
        """Load stored embeddings by id.

        Args:
            ids: One chunk id, or several ids in the order to return.

        Returns:
            Stored rows, including each embedding. Missing ids are omitted.
        """
        requested = [ids] if isinstance(ids, str) else list(ids)
        return vectors.load_embeddings(requested)

    def search(
        self,
        query: str,
        limit: int = 5,
        *,
        slug: str | None = None,
        faction_slug: str | None = None,
    ) -> list[dict[str, Any]]:
        """Embed a query and blend vector similarity with keyword matches.

        Args:
            query: Natural-language query.
            limit: Maximum number of chunks to return.
            slug: Restrict the search to one champion.
            faction_slug: Restrict the search to one region.

        Returns:
            Chunks ordered by reciprocal rank fusion, closest agreement first.
        """
        if not query.strip():
            raise ValueError("查询文本不能为空")
        embedding = self.embed(query)[0]
        return vectors.search_embeddings(
            embedding,
            limit,
            query=query,
            slug=slug,
            faction_slug=faction_slug,
        )


embedding_model = EmbeddingModel()
