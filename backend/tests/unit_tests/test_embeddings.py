import pytest
from langchain_openai import OpenAIEmbeddings

from agent.embeddings import EmbeddingModel


def _set_embedding_env(monkeypatch: pytest.MonkeyPatch, base_url: str) -> None:
    monkeypatch.setenv("OPENAI_EMBEDDING_API_KEY", "test-key")
    monkeypatch.setenv("OPENAI_EMBEDDING_BASE_URL", base_url)
    monkeypatch.setenv("OPENAI_EMBEDDING_MODEL", "demo-model")


def test_embed_requires_configuration(monkeypatch: pytest.MonkeyPatch) -> None:
    for name in (
        "OPENAI_EMBEDDING_API_KEY",
        "OPENAI_EMBEDDING_BASE_URL",
        "OPENAI_EMBEDDING_MODEL",
    ):
        monkeypatch.delenv(name, raising=False)

    with pytest.raises(RuntimeError, match="OPENAI_EMBEDDING_API_KEY"):
        EmbeddingModel().embed("亚托克斯")


def test_client_uses_embedding_settings(monkeypatch: pytest.MonkeyPatch) -> None:
    _set_embedding_env(
        monkeypatch, "https://example.com/v2/embeddings/"
    )
    model = EmbeddingModel()
    model._client_or_create()

    client = model._client
    assert isinstance(client, OpenAIEmbeddings)
    assert client.model == "demo-model"
    assert client.openai_api_base == "https://example.com/v2"
    assert client.check_embedding_ctx_length is False
    assert client.openai_api_key.get_secret_value() == "test-key"


def test_embed_returns_one_vector_per_text(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    captured: dict[str, list[str]] = {}

    class FakeEmbeddings:
        def embed_documents(self, texts: list[str]) -> list[list[float]]:
            captured["texts"] = texts
            return [[float(index)] for index, _ in enumerate(texts)]

    monkeypatch.setattr(
        EmbeddingModel,
        "_client_or_create",
        lambda self: FakeEmbeddings(),
    )

    vectors = EmbeddingModel().embed(["亚托克斯", "暗裔"])

    assert captured["texts"] == ["亚托克斯", "暗裔"]
    assert vectors == [[0.0], [1.0]]
    assert EmbeddingModel().embed("亚托克斯") == [[0.0]]


def test_embed_rejects_an_empty_batch() -> None:
    with pytest.raises(ValueError, match="至少需要一段文本"):
        EmbeddingModel().embed([])


def test_save_embeds_rows_before_upsert(monkeypatch: pytest.MonkeyPatch) -> None:
    captured: list[dict] = []
    monkeypatch.setattr(
        "agent.embeddings._settings",
        lambda: ("test-key", "https://example.com/v2", "demo-model"),
    )
    monkeypatch.setattr(
        EmbeddingModel,
        "embed",
        lambda self, texts: [[float(index)] for index, _ in enumerate(texts)],
    )
    monkeypatch.setattr(
        "agent.embeddings.vectors.upsert_embeddings",
        captured.extend,
    )

    saved = EmbeddingModel().save(
        [{"id": "aatrox-0", "text": "亚托克斯"}, {"id": "kayn-0", "text": "凯隐"}]
    )

    assert saved == 2
    assert captured[0]["model"] == "demo-model"
    assert captured[0]["embedding"] == [0.0]
    assert captured[1]["id"] == "kayn-0"
    assert captured[1]["embedding"] == [1.0]
