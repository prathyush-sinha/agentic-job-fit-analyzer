"""Postgres + pgvector connection helpers and a health check.

Run `python -m app.db` to verify a configured DATABASE_URL: it connects,
ensures the pgvector extension exists, and prints a redacted status report.
"""

from __future__ import annotations

import sys

import psycopg
from pgvector.psycopg import register_vector

from app.config import Settings, get_settings


def get_connection(settings: Settings | None = None) -> psycopg.Connection:
    """Open a psycopg connection with pgvector types registered."""
    settings = settings or get_settings()
    if not settings.database_url:
        raise RuntimeError(
            "DATABASE_URL is not set. Add your hosted Postgres connection string "
            "to .env (see .env.example)."
        )
    conn = psycopg.connect(settings.database_url)
    ensure_pgvector(conn)
    register_vector(conn)
    return conn


def ensure_pgvector(conn: psycopg.Connection) -> None:
    """Create the vector extension if the role is permitted (hosted: usually yes)."""
    with conn.cursor() as cur:
        cur.execute("CREATE EXTENSION IF NOT EXISTS vector;")
    conn.commit()


def health_check(settings: Settings | None = None) -> dict[str, str]:
    """Connect, ensure pgvector, return a small status dict. Raises on failure."""
    settings = settings or get_settings()
    with get_connection(settings) as conn, conn.cursor() as cur:
        cur.execute("SELECT version();")
        pg_version = cur.fetchone()[0]
        cur.execute("SELECT extversion FROM pg_extension WHERE extname = 'vector';")
        row = cur.fetchone()
        vector_version = row[0] if row else None
    return {
        "database": settings.redacted_database_url,
        "postgres": pg_version.split(",")[0],
        "pgvector": vector_version or "NOT INSTALLED",
    }


def main() -> None:
    settings = get_settings()
    print(f"Connecting to: {settings.redacted_database_url}")
    try:
        status = health_check(settings)
    except Exception as exc:  # surface a clean message, no secrets
        print(f"FAILED: {type(exc).__name__}: {exc}", file=sys.stderr)
        sys.exit(1)
    print("OK")
    for k, v in status.items():
        print(f"  {k:<10} {v}")
    if status["pgvector"] == "NOT INSTALLED":
        print("WARNING: pgvector extension not available on this instance.", file=sys.stderr)
        sys.exit(2)


if __name__ == "__main__":
    main()
