# Agentic Job-Fit Analyzer

FastAPI app for comparing a resume against job postings for a target role.

The system retrieves similar postings, runs a LangGraph workflow, and returns a structured report with matched skills, likely gaps, suggested next steps, and resume-bullet rewrites.

See `docs/architecture.md` for the pipeline diagram and design notes.

## What it does

- Scrapes job postings from public Greenhouse and Lever APIs.
- Cleans and chunks posting text by section.
- Indexes chunks in Postgres with pgvector.
- Compares dense retrieval, BM25, hybrid RRF, and reranked retrieval.
- Runs a planner, retriever, analyzer, critic, and synthesizer workflow.
- Returns a structured `FitReport` from the `/analyze` endpoint.

## Retrieval evaluation

Posting-level metrics over a 16-role gold set with 3,049 relevance judgments. The corpus has about 4,200 indexed postings. Relevance is based on target-role title matching, so this is a retrieval benchmark rather than a full human evaluation of job fit.

| system | NDCG@5 | NDCG@10 | P@5 | MRR | latency p50 |
|---|---:|---:|---:|---:|---:|
| BM25-only | 0.664 | 0.680 | 0.675 | 0.763 | 1.8s |
| dense-only | 0.765 | 0.763 | 0.775 | 0.849 | 2.1s |
| hybrid RRF | 0.766 | 0.756 | 0.775 | 0.818 | 3.5s |
| hybrid + cross-encoder rerank | 0.860 | 0.835 | 0.838 | 0.950 | 18.9s |

The best run uses dense retrieval + BM25, reciprocal-rank fusion, and a `BAAI/bge-reranker-base` cross-encoder. Dense-only retrieval is still available through `DENSE_ONLY=true` so the baseline can be rerun.

## Stack

- Python 3.11, FastAPI, Pydantic
- LangGraph
- Postgres + pgvector
- `rank-bm25`
- `BAAI/bge-small-en-v1.5` embeddings by default
- `BAAI/bge-reranker-base` for reranking
- Google Gemini for the graph nodes

## Quickstart

```bash
python -m venv .venv && source .venv/bin/activate    # Windows: .venv\Scripts\activate
pip install -r requirements.txt
cp .env.example .env                                  # set GOOGLE_API_KEY and DATABASE_URL
python -m ingest.scrape
python -m ingest.index --recreate
uvicorn app.main:app --reload
```

## Analyze a resume

```bash
curl -s localhost:8000/analyze -H 'content-type: application/json' \
  -d '{"resume": "Built a RAG pipeline with FastAPI, Postgres, pgvector, and retrieval evals.", "target_role": "Machine Learning Engineer"}'
```

Example response shape:

```json
{
  "report": {
    "target_role": "Machine Learning Engineer",
    "overall_fit": "The resume shows relevant retrieval and backend experience, especially around RAG, FastAPI, and vector search. The main gaps are broader model training experience and production deployment evidence.",
    "fit_score": 72,
    "matched_skills": [
      {
        "skill": "RAG and vector search",
        "resume_evidence": "Built a RAG pipeline with Postgres and pgvector.",
        "posting_evidence": "Retrieved postings mention retrieval systems, embeddings, and vector databases."
      }
    ],
    "gaps": [
      {
        "requirement": "Production ML deployment",
        "why_it_matters": "Several postings expect shipped ML systems or production monitoring.",
        "posting_evidence": "Retrieved postings mention deployment, monitoring, and reliability requirements.",
        "suggested_next_step": "Add a deployment section with latency, logging, and error-handling details."
      }
    ],
    "bullet_rewrites": [
      {
        "original": "Built a RAG pipeline.",
        "rewritten": "Built a RAG pipeline using FastAPI, Postgres/pgvector, and retrieval evaluation to compare dense, BM25, hybrid, and reranked search.",
        "rationale": "Adds stack, evaluation scope, and role-relevant detail."
      }
    ],
    "evidence_used": ["Example Company — Machine Learning Engineer"]
  },
  "meta": {
    "retries": 1,
    "llm_tokens": 4200,
    "evidence_count": 6,
    "latency_s": 84.7
  }
}
```

Actual output depends on the indexed postings and the resume text.

## Commands

| Task | Command |
| --- | --- |
| Run app | `uvicorn app.main:app --reload` |
| Run tests | `pytest` |
| Scrape corpus | `python -m ingest.scrape` |
| Build index | `python -m ingest.index --recreate` |
| Check DB + pgvector | `python -m app.db` |
| Run retrieval eval | `python -m evals.retrieval` |

## Status

- Ingestion, indexing, retrieval, reranking, graph workflow, and API endpoint are implemented.
- Current test suite: 33 passing tests.
- Retrieval evaluation is measured; report-quality evaluation still needs human checking.
- Full `/analyze` runs can take around 1-2.5 minutes on CPU because reranking and multiple graph-node calls are in the loop.
- Current index covers about 4,200 of roughly 4,500 scraped postings from the last indexing run.
