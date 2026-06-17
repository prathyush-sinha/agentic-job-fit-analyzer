"""Schema management and chunk persistence for the pgvector store.

Bulk-load recipe (important for a remote DB): create the table WITHOUT the HNSW
index, stream rows in with COPY, then build the index once at the end. Inserting
into a live HNSW index row-by-row over the network is orders of magnitude slower.
"""

from __future__ import annotations

import psycopg

from app.config import Settings, get_settings
from app.db import to_vector_literal
from ingest.models import Chunk

_TABLE_DDL = """
CREATE EXTENSION IF NOT EXISTS vector;

CREATE TABLE IF NOT EXISTS chunks (
    id          TEXT PRIMARY KEY,
    posting_id  TEXT NOT NULL,
    source      TEXT NOT NULL,
    company     TEXT NOT NULL,
    title       TEXT NOT NULL,
    location    TEXT,
    url         TEXT,
    section     TEXT NOT NULL,
    chunk_index INT  NOT NULL,
    content     TEXT NOT NULL,
    embedding   vector({dim}),
    created_at  TIMESTAMPTZ NOT NULL DEFAULT now()
);

CREATE INDEX IF NOT EXISTS chunks_posting_id_idx ON chunks (posting_id);
"""

_COPY_COLS = (
    "id, posting_id, source, company, title, location, url, "
    "section, chunk_index, content, embedding"
)

_UPSERT = """
INSERT INTO chunks
    (id, posting_id, source, company, title, location, url, section, chunk_index, content, embedding)
VALUES
    (%s, %s, %s, %s, %s, %s, %s, %s, %s, %s, %s::vector)
ON CONFLICT (id) DO UPDATE SET
    content = EXCLUDED.content,
    section = EXCLUDED.section,
    embedding = EXCLUDED.embedding;
"""


def create_schema(conn: psycopg.Connection, settings: Settings | None = None) -> None:
    """Create the table + btree index. Does NOT build the vector index."""
    settings = settings or get_settings()
    with conn.cursor() as cur:
        cur.execute(_TABLE_DDL.format(dim=settings.embedding_dim))
    conn.commit()


def create_vector_index(conn: psycopg.Connection) -> None:
    """Build the HNSW cosine index. Call once, after bulk loading."""
    with conn.cursor() as cur:
        cur.execute(
            "CREATE INDEX IF NOT EXISTS chunks_embedding_hnsw "
            "ON chunks USING hnsw (embedding vector_cosine_ops);"
        )
    conn.commit()


def copy_chunks(
    conn: psycopg.Connection, chunks: list[Chunk], embeddings: list[list[float]]
) -> int:
    """Fast bulk insert via COPY (use on a fresh table, no conflict handling)."""
    if len(chunks) != len(embeddings):
        raise ValueError("chunks and embeddings length mismatch")
    with conn.cursor() as cur, cur.copy(f"COPY chunks ({_COPY_COLS}) FROM STDIN") as copy:
        for c, emb in zip(chunks, embeddings):
            copy.write_row((
                c.id, c.posting_id, c.source, c.company, c.title, c.location, c.url,
                c.section, c.chunk_index, c.content, to_vector_literal(emb),
            ))
    conn.commit()
    return len(chunks)


def upsert_chunks(
    conn: psycopg.Connection, chunks: list[Chunk], embeddings: list[list[float]]
) -> int:
    """Insert/update a batch of chunks (slower; for incremental updates)."""
    if len(chunks) != len(embeddings):
        raise ValueError("chunks and embeddings length mismatch")
    rows = [
        (
            c.id, c.posting_id, c.source, c.company, c.title, c.location, c.url,
            c.section, c.chunk_index, c.content, to_vector_literal(emb),
        )
        for c, emb in zip(chunks, embeddings)
    ]
    with conn.cursor() as cur:
        cur.executemany(_UPSERT, rows)
    conn.commit()
    return len(rows)


def count_chunks(conn: psycopg.Connection) -> int:
    with conn.cursor() as cur:
        cur.execute("SELECT count(*) FROM chunks;")
        return cur.fetchone()[0]
