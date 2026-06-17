"""Application settings, loaded from environment / .env.

Secrets (API key, DB URL with password) live only in the environment. Nothing
here prints them; see `redacted_database_url` for safe logging.
"""

from __future__ import annotations

from urllib.parse import urlsplit, urlunsplit

from pydantic_settings import BaseSettings, SettingsConfigDict


class Settings(BaseSettings):
    model_config = SettingsConfigDict(
        env_file=".env", env_file_encoding="utf-8", extra="ignore"
    )

    # OpenAI — judge MUST differ from agent (anti self-grading bias)
    openai_api_key: str = ""
    openai_agent_model: str = "gpt-4o-mini"
    openai_judge_model: str = "gpt-4o"
    openai_embedding_model: str = "text-embedding-3-small"

    # Database (hosted Postgres + pgvector). Required for DB-backed work.
    database_url: str = ""

    # Retrieval ablation baseline flag (true = dense-only)
    dense_only: bool = False

    @property
    def redacted_database_url(self) -> str:
        """database_url with the password stripped — safe to log."""
        if not self.database_url:
            return "(unset)"
        parts = urlsplit(self.database_url)
        netloc = parts.netloc
        if "@" in netloc:
            userinfo, host = netloc.rsplit("@", 1)
            user = userinfo.split(":", 1)[0]
            netloc = f"{user}:***@{host}"
        return urlunsplit((parts.scheme, netloc, parts.path, "", ""))


def get_settings() -> Settings:
    return Settings()
