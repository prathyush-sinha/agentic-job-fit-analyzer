# Project Notes

This file tracks the current project state and remaining work. The main overview is in `README.md`; the architecture notes are in `docs/architecture.md`.

## Implemented

- Scrape postings from public ATS APIs and normalize them into one JSONL format.
- Clean posting text and split descriptions into section-aware chunks.
- Embed chunks and store them in Postgres with pgvector.
- Support dense search, BM25, reciprocal-rank fusion, and cross-encoder reranking.
- Keep dense-only retrieval available as an ablation baseline.
- Run a LangGraph workflow with planner, retriever, analyzer, critic, and synthesizer nodes.
- Expose the workflow through `POST /analyze`.
- Track retrieval metrics: precision, recall, NDCG, MRR, and latency.

## Current retrieval result

Posting-level evaluation on the current 16-role gold set:

| system | NDCG@5 | NDCG@10 | P@5 | MRR | latency p50 |
|---|---:|---:|---:|---:|---:|
| BM25-only | 0.664 | 0.680 | 0.675 | 0.763 | 1.8s |
| dense-only | 0.765 | 0.763 | 0.775 | 0.849 | 2.1s |
| hybrid RRF | 0.766 | 0.756 | 0.775 | 0.818 | 3.5s |
| hybrid + rerank | 0.860 | 0.835 | 0.838 | 0.950 | 18.9s |

The cross-encoder reranker improves the top results, but it also adds noticeable CPU latency.

## Remaining work

- Add a small `examples/` folder with sample input and output.
- Add a human-checked evaluation for final report quality.
- Add CI for tests.
- Add a simple UI if the API-only version is too bare for demos.
- Revisit reranker latency and candidate counts.

## Common commands

```bash
pytest
python -m ingest.scrape
python -m ingest.index --recreate
python -m evals.retrieval
uvicorn app.main:app --reload
```
