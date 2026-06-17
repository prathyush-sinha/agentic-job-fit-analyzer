"""Tests for settings — focus on password redaction (no live DB)."""

from app.config import Settings


def test_redacted_database_url_hides_password():
    s = Settings(database_url="postgresql://user:secret@host.neon.tech/db?sslmode=require")
    red = s.redacted_database_url
    assert "secret" not in red
    assert "user:***@host.neon.tech" in red
    assert red.endswith("/db")  # query (and its params) dropped


def test_redacted_database_url_unset():
    assert Settings(database_url="").redacted_database_url == "(unset)"


def test_judge_model_differs_from_agent_by_default():
    s = Settings()
    assert s.openai_judge_model != s.openai_agent_model
