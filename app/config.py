"""Application settings, loaded from environment / .env.

Secrets (API key, DB URL with password) live only in the environment. Nothing
here prints them; see `redacted_database_url` for safe logging.
"""

from __future__ import annotations

from urllib.parse import urlsplit, urlunsplit

from pydantic_settings import BaseSettings, SettingsConfigDict

# Output dimension per OpenAI embedding model — used to size the pgvector column.
# (Gemini's dimension is configurable via gemini_output_dim, below.)
_OPENAI_EMBED_DIMS: dict[str, int] = {
    "text-embedding-3-small": 1536,
    "text-embedding-3-large": 3072,
}


class Settings(BaseSettings):
    model_config = SettingsConfigDict(
        env_file=".env", env_file_encoding="utf-8", extra="ignore"
    )

    # Active embedding provider: "gemini" (default, free tier) or "openai".
    embedding_provider: str = "gemini"

    # Gemini (default) — free key from aistudio.google.com, no billing.
    google_api_key: str = ""
    gemini_embedding_model: str = "gemini-embedding-001"
    # Native dim is 3072; we truncate (Matryoshka) to keep vectors small enough
    # for the Neon free tier and faster to index.
    gemini_output_dim: int = 768

    # OpenAI — reachable via EMBEDDING_PROVIDER=openai. Judge MUST differ from
    # agent (anti self-grading bias).
    openai_api_key: str = ""
    openai_agent_model: str = "gpt-4o-mini"
    openai_judge_model: str = "gpt-4o"
    openai_embedding_model: str = "text-embedding-3-small"

    # Database (hosted Postgres + pgvector). Required for DB-backed work.
    database_url: str = ""

    # Retrieval ablation baseline flag (true = dense-only)
    dense_only: bool = False

    @property
    def embedding_model(self) -> str:
        """The embedding model name for the active provider."""
        if self.embedding_provider == "openai":
            return self.openai_embedding_model
        return self.gemini_embedding_model

    @property
    def embedding_dim(self) -> int:
        """Vector dimension for the active embedding model."""
        if self.embedding_provider == "openai":
            return _OPENAI_EMBED_DIMS.get(self.openai_embedding_model, 1536)
        return self.gemini_output_dim

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
