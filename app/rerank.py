"""Cross-encoder reranking of retrieved candidates.

A cross-encoder scores (query, chunk) pairs jointly, which is more accurate than
the bi-encoder cosine used for first-stage retrieval — at the cost of running the
model per candidate, so it only runs over the fused top-N.
"""

from __future__ import annotations

from functools import lru_cache
from typing import TYPE_CHECKING

from app.config import Settings, get_settings

if TYPE_CHECKING:
    from app.retrieval import Hit


@lru_cache(maxsize=2)
def _model(name: str):
    from sentence_transformers import CrossEncoder

    return CrossEncoder(name)


def rerank(
    query: str, candidates: list["Hit"], top_k: int, settings: Settings | None = None
) -> list["Hit"]:
    """Reorder candidates by cross-encoder relevance; return the top_k with scores set."""
    if not candidates:
        return []
    settings = settings or get_settings()
    model = _model(settings.reranker_model)
    scores = model.predict([(query, h.content) for h in candidates])
    for hit, score in zip(candidates, scores):
        hit.score = float(score)
    candidates.sort(key=lambda h: h.score, reverse=True)
    return candidates[:top_k]
