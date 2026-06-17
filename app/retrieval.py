"""Retrieval over the pgvector chunk store.

`search()` is the stable entry point. With DENSE_ONLY it runs dense (cosine)
retrieval — the ablation baseline. Otherwise it runs hybrid retrieval: dense +
BM25 fused with reciprocal rank fusion, then a cross-encoder rerank.
"""

from __future__ import annotations

from collections import defaultdict
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


def rrf_fuse(rankings: list[list[str]], rrf_k: int = 60) -> list[tuple[str, float]]:
    """Reciprocal rank fusion of several ranked id lists.

    score(id) = sum over rankings of 1 / (rrf_k + rank), rank starting at 1.
    Returns (id, score) sorted best-first.
    """
    scores: dict[str, float] = defaultdict(float)
    for ranking in rankings:
        for rank, cid in enumerate(ranking, start=1):
            scores[cid] += 1.0 / (rrf_k + rank)
    return sorted(scores.items(), key=lambda t: t[1], reverse=True)


def hybrid_search(query_text: str, k: int = 10, settings: Settings | None = None) -> list[Hit]:
    """Dense + BM25 fused via RRF, then cross-encoder reranked to top-k."""
    from app.rerank import rerank
    from app.sparse import get_sparse_index

    settings = settings or get_settings()
    n = settings.hybrid_candidates

    dense_ids = [h.id for h in dense_search(query_text, n, settings)]
    sparse = get_sparse_index(settings)
    sparse_ids = [cid for cid, _ in sparse.search(query_text, n)]

    fused = rrf_fuse([dense_ids, sparse_ids])[: settings.rerank_top]
    candidates = [
        Hit(id=row["id"], posting_id=row["posting_id"], company=row["company"],
            title=row["title"], section=row["section"], url=row["url"],
            score=score, content=row["content"])
        for cid, score in fused
        if (row := sparse.rows.get(cid)) is not None
    ]
    return rerank(query_text, candidates, k, settings)


def search(query_text: str, k: int = 10, settings: Settings | None = None) -> list[Hit]:
    """Stable retrieval entry point: dense-only (ablation) or hybrid+rerank."""
    settings = settings or get_settings()
    if settings.dense_only:
        return dense_search(query_text, k, settings)
    return hybrid_search(query_text, k, settings)
