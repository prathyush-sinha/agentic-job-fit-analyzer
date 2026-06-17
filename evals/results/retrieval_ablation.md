# Retrieval Ablation

Posting-level metrics averaged over the gold queries (`evals/gold/gold.jsonl`). Relevance = title matches the target role (objective proxy; see `evals/gold.py`).

```
| system         |   ndcg@5 |  ndcg@10 |      p@5 |     r@10 |      mrr | lat p50 | lat p95 |
|----------------|----------|----------|----------|----------|----------|---------|---------|
| dense-only     |    0.765 |    0.763 |    0.775 |    0.074 |    0.849 |   2.086 |   2.354 |
| bm25-only      |    0.664 |    0.680 |    0.675 |    0.069 |    0.763 |   1.762 |   1.992 |
| hybrid-rrf     |    0.766 |    0.756 |    0.775 |    0.074 |    0.818 |   3.484 |   4.096 |
| hybrid+rerank  |    0.860 |    0.835 |    0.838 |    0.077 |    0.950 |  18.892 |  19.995 |
```
