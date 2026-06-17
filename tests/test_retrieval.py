"""Tests for RRF fusion and BM25 sparse search — no DB / no torch."""

from app.retrieval import rrf_fuse
from app.sparse import SparseIndex, tokenize


def test_tokenize():
    assert tokenize("Python, SQL & pgvector!") == ["python", "sql", "pgvector"]


def test_rrf_rewards_agreement_across_rankings():
    # "b" is high in both lists; should win the fusion.
    dense = ["a", "b", "c"]
    sparse = ["b", "d", "a"]
    fused = rrf_fuse([dense, sparse])
    assert fused[0][0] == "b"
    # every id that appeared is present
    assert {cid for cid, _ in fused} == {"a", "b", "c", "d"}


def test_rrf_single_ranking_preserves_order():
    fused = rrf_fuse([["x", "y", "z"]])
    assert [cid for cid, _ in fused] == ["x", "y", "z"]


def _index(docs: dict[str, str]) -> SparseIndex:
    from rank_bm25 import BM25Okapi

    ids = list(docs)
    rows = {i: {"id": i, "content": docs[i]} for i in ids}
    bm25 = BM25Okapi([tokenize(docs[i]) for i in ids])
    return SparseIndex(ids=ids, rows=rows, _bm25=bm25)


def test_bm25_ranks_lexical_match_first():
    idx = _index({
        "d1": "machine learning engineer python pytorch",
        "d2": "sales account executive enterprise",
        "d3": "data analyst sql dashboards",
    })
    top = idx.search("python machine learning", k=3)
    assert top[0][0] == "d1"
    assert top[0][1] > 0  # nonzero BM25 score for a real match
