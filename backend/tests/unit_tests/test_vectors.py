import pytest

from agent.db import settings
from agent.vectors import parse_vector, vector_literal


def test_vector_literal_round_trip() -> None:
    values = [0.25, -1.5, 3.0]
    assert parse_vector(vector_literal(values)) == values


def test_settings_reject_unsafe_database_name(monkeypatch: pytest.MonkeyPatch) -> None:
    monkeypatch.setenv("POSTGRES_HOST", "127.0.0.1")
    monkeypatch.setenv("POSTGRES_USER", "postgres")
    monkeypatch.setenv("POSTGRES_DATABASE", "ai-agent")

    with pytest.raises(RuntimeError, match="POSTGRES_DATABASE"):
        settings()


def test_settings_require_host_and_user(monkeypatch: pytest.MonkeyPatch) -> None:
    monkeypatch.delenv("POSTGRES_HOST", raising=False)
    monkeypatch.delenv("POSTGRES_USER", raising=False)

    with pytest.raises(RuntimeError, match="POSTGRES_HOST"):
        settings()
