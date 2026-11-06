import pytest

from agent.db import settings
from agent.vectors import fuse_hits, keyword_terms, parse_vector, vector_literal


def test_vector_literal_round_trip() -> None:
    values = [0.25, -1.5, 3.0]
    assert parse_vector(vector_literal(values)) == values


def test_settings_reject_unsafe_database_name(monkeypatch: pytest.MonkeyPatch) -> None:
    monkeypatch.setenv("POSTGRES_HOST", "127.0.0.1")
    monkeypatch.setenv("POSTGRES_USER", "postgres")
    monkeypatch.setenv("POSTGRES_DATABASE", "ai-agent")

    with pytest.raises(RuntimeError, match="POSTGRES_DATABASE"):
        settings()


def test_keyword_terms_keep_the_name_and_drop_question_words() -> None:
    terms = keyword_terms("亚托克斯为什么会被锁进剑里")

    assert "为什么" not in terms
    assert "什么" not in terms
    assert "亚托克" in terms
    assert "锁进剑" in terms


def test_keyword_terms_keep_a_short_name() -> None:
    assert keyword_terms("亚托克斯")[0] == "亚托克斯"


def test_fuse_hits_promotes_a_row_found_by_both_channels() -> None:
    vector_rows = [
        {"id": "other", "distance": 0.2, "text": "其他英雄"},
        {"id": "aatrox", "distance": 0.45, "text": "亚托克斯被锁进剑里"},
    ]
    keyword_rows = [{"id": "aatrox", "text": "亚托克斯被锁进剑里"}]

    fused = fuse_hits(vector_rows, keyword_rows, limit=1)

    assert fused[0]["id"] == "aatrox"


def test_fuse_hits_keeps_vector_order_without_keyword_hits() -> None:
    vector_rows = [
        {"id": "a", "distance": 0.1, "text": "甲"},
        {"id": "b", "distance": 0.2, "text": "乙"},
    ]

    fused = fuse_hits(vector_rows, [], limit=2)

    assert [row["id"] for row in fused] == ["a", "b"]


def test_settings_require_host_and_user(monkeypatch: pytest.MonkeyPatch) -> None:
    monkeypatch.delenv("POSTGRES_HOST", raising=False)
    monkeypatch.delenv("POSTGRES_USER", raising=False)

    with pytest.raises(RuntimeError, match="POSTGRES_HOST"):
        settings()
