"""BM25 sparse retrieval over the chunk store.

Loads all chunks into memory once and builds a BM25 index. For this corpus
(~hundreds-to-thousands of chunks) the in-memory index is tiny and also serves
as the canonical chunk lookup when materializing fused hybrid results.
"""

from __future__ import annotations

import re
from dataclasses import dataclass

from app.config import Settings, get_settings
from app.db import get_connection

_TOKEN = re.compile(r"[a-z0-9]+")

_LOAD_SQL = "SELECT id, posting_id, company, title, section, url, content FROM chunks;"


def tokenize(text: str) -> list[str]:
    return _TOKEN.findall(text.lower())


@dataclass
class SparseIndex:
    ids: list[str]
    rows: dict[str, dict]  # id -> chunk fields (for materializing hits)
    _bm25: object          # rank_bm25.BM25Okapi

    @classmethod
    def from_db(cls, settings: Settings | None = None) -> "SparseIndex":
        from rank_bm25 import BM25Okapi

        settings = settings or get_settings()
        rows: dict[str, dict] = {}
        ids: list[str] = []
        corpus_tokens: list[list[str]] = []
        with get_connection(settings) as conn, conn.cursor() as cur:
            cur.execute(_LOAD_SQL)
            for r in cur.fetchall():
                row = {
                    "id": r[0], "posting_id": r[1], "company": r[2], "title": r[3],
                    "section": r[4], "url": r[5], "content": r[6],
                }
                rows[row["id"]] = row
                ids.append(row["id"])
                corpus_tokens.append(tokenize(row["content"]))
        return cls(ids=ids, rows=rows, _bm25=BM25Okapi(corpus_tokens))

    def search(self, query: str, k: int) -> list[tuple[str, float]]:
        """Return up to k (chunk_id, bm25_score), best first."""
        scores = self._bm25.get_scores(tokenize(query))
        ranked = sorted(zip(self.ids, scores), key=lambda t: t[1], reverse=True)
        return [(cid, float(s)) for cid, s in ranked[:k]]


_cached: SparseIndex | None = None


def get_sparse_index(settings: Settings | None = None, refresh: bool = False) -> SparseIndex:
    """Process-lifetime cached BM25 index (rebuild with refresh=True after reindex)."""
    global _cached
    if _cached is None or refresh:
        _cached = SparseIndex.from_db(settings)
    return _cached
