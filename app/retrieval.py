"""Retrieval over the pgvector chunk store.

Phase 1 ships dense (cosine) retrieval only — this is the ablation baseline the
project keeps reachable behind the DENSE_ONLY flag. Phase 2 adds BM25 + fusion +
reranking; `search()` is the stable entry point that will dispatch on the flag.
"""

from __future__ import annotations

from dataclasses import dataclass

from app.config import Settings, get_settings
from app.db import get_connection, to_vector_literal
from ingest.embed import embed_query

_SEARCH_SQL = """
SELECT id, posting_id, company, title, section, url,
       1 - (embedding <=> %s::vector) AS score,
       content
FROM chunks
ORDER BY embedding <=> %s::vector
LIMIT %s;
"""


@dataclass
class Hit:
    id: str
    posting_id: str
    company: str
    title: str
    section: str
    url: str
    score: float
    content: str


def dense_search(query_text: str, k: int = 10, settings: Settings | None = None) -> list[Hit]:
    """Embed the query and return the top-k chunks by cosine similarity."""
    settings = settings or get_settings()
    vec = to_vector_literal(embed_query(query_text, settings))
    with get_connection(settings) as conn, conn.cursor() as cur:
        cur.execute(_SEARCH_SQL, (vec, vec, k))
        rows = cur.fetchall()
    return [
        Hit(id=r[0], posting_id=r[1], company=r[2], title=r[3], section=r[4],
            url=r[5], score=float(r[6]), content=r[7])
        for r in rows
    ]


def search(query_text: str, k: int = 10, settings: Settings | None = None) -> list[Hit]:
    """Stable retrieval entry point.

    Today this is dense-only. In Phase 2, when DENSE_ONLY is false, this will run
    hybrid retrieval + reranking; when true, it falls back to `dense_search`
    (the ablation baseline). Until then both branches are dense.
    """
    settings = settings or get_settings()
    # Phase 2 will branch on settings.dense_only here.
    return dense_search(query_text, k, settings)
