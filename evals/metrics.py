"""Retrieval metrics: precision@k, recall@k, NDCG@k, MRR.

All operate on a ranked list of ids (best first) and a set of relevant ids,
with binary relevance. Pure functions — no I/O, fully unit-tested.
"""

from __future__ import annotations

import math
from collections.abc import Iterable, Sequence


def precision_at_k(ranked: Sequence[str], relevant: Iterable[str], k: int) -> float:
    relevant = set(relevant)
    if k <= 0:
        return 0.0
    topk = ranked[:k]
    hits = sum(1 for r in topk if r in relevant)
    return hits / k


def recall_at_k(ranked: Sequence[str], relevant: Iterable[str], k: int) -> float:
    relevant = set(relevant)
    if not relevant:
        return 0.0
    topk = ranked[:k]
    hits = sum(1 for r in topk if r in relevant)
    return hits / len(relevant)


def dcg_at_k(ranked: Sequence[str], relevant: Iterable[str], k: int) -> float:
    relevant = set(relevant)
    dcg = 0.0
    for i, r in enumerate(ranked[:k], start=1):
        if r in relevant:
            dcg += 1.0 / math.log2(i + 1)
    return dcg


def ndcg_at_k(ranked: Sequence[str], relevant: Iterable[str], k: int) -> float:
    relevant = set(relevant)
    if not relevant:
        return 0.0
    ideal_hits = min(len(relevant), k)
    idcg = sum(1.0 / math.log2(i + 1) for i in range(1, ideal_hits + 1))
    if idcg == 0:
        return 0.0
    return dcg_at_k(ranked, relevant, k) / idcg


def mrr(ranked: Sequence[str], relevant: Iterable[str]) -> float:
    """Reciprocal rank of the first relevant id (0 if none)."""
    relevant = set(relevant)
    for i, r in enumerate(ranked, start=1):
        if r in relevant:
            return 1.0 / i
    return 0.0
