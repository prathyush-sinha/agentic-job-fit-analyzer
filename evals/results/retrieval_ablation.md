# Retrieval Ablation

Posting-level metrics averaged over the gold queries (`evals/gold/gold.jsonl`). Relevance = title matches the target role (objective proxy; see `evals/gold.py`).

```
| system         |   ndcg@5 |  ndcg@10 |      p@5 |     r@10 |      mrr | lat p50 | lat p95 |
|----------------|----------|----------|----------|----------|----------|---------|---------|
| dense-only     |    0.756 |    0.747 |    0.700 |    0.273 |    1.000 |   4.666 |   5.957 |
| bm25-only      |    0.675 |    0.662 |    0.650 |    0.247 |    0.875 |   0.039 |   0.043 |
| hybrid+rerank  |    0.792 |    0.744 |    0.800 |    0.287 |    0.833 |   6.187 |   6.190 |
```
