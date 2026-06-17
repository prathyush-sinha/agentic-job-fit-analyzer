"""Unit tests for retrieval metrics."""

import math

from evals.metrics import (
    dcg_at_k,
    mrr,
    ndcg_at_k,
    precision_at_k,
    recall_at_k,
)

RANKED = ["a", "b", "c", "d", "e"]
RELEVANT = {"a", "c", "x"}  # x is not retrieved


def test_precision_at_k():
    assert precision_at_k(RANKED, RELEVANT, 1) == 1.0   # a
    assert precision_at_k(RANKED, RELEVANT, 2) == 0.5   # a, b
    assert precision_at_k(RANKED, RELEVANT, 4) == 0.5   # a,c of 4
    assert precision_at_k(RANKED, RELEVANT, 0) == 0.0


def test_recall_at_k():
    assert recall_at_k(RANKED, RELEVANT, 1) == 1 / 3
    assert recall_at_k(RANKED, RELEVANT, 3) == 2 / 3   # a, c found; x missing
    assert recall_at_k(RANKED, set(), 3) == 0.0


def test_perfect_ranking_ndcg_is_one():
    rel = {"a", "b"}
    assert ndcg_at_k(["a", "b", "c"], rel, 3) == 1.0


def test_ndcg_penalizes_lower_rank():
    # relevant item at rank 2 instead of 1 -> below 1.0 but > 0
    score = ndcg_at_k(["z", "a"], {"a"}, 2)
    assert 0 < score < 1
    assert math.isclose(score, (1 / math.log2(3)) / 1.0)


def test_mrr():
    assert mrr(["z", "y", "a"], {"a"}) == 1 / 3
    assert mrr(["a", "b"], {"a"}) == 1.0
    assert mrr(["x", "y"], {"a"}) == 0.0


def test_dcg_accumulates_multiple_hits():
    val = dcg_at_k(["a", "z", "c"], {"a", "c"}, 3)
    assert math.isclose(val, 1 / math.log2(2) + 1 / math.log2(4))
