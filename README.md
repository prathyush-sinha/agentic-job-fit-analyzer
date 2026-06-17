# Agentic Job-Fit Analyzer

Given a resume and a target role, retrieves relevant job postings, analyzes fit, and
returns a structured report (matched skills, gaps, tailored bullet rewrites).

A portfolio project demonstrating three things: an **agent loop** (LangGraph
planner → retriever → analyzer → critic → synthesizer), **RAG depth** (hybrid dense +
BM25 retrieval with cross-encoder reranking), and **eval discipline** (validated metrics).

## Stack
Python 3.11+ · LangGraph · Postgres + pgvector · OpenAI (agent + judge) · FastAPI · Docker

## Quickstart
```bash
python -m venv .venv && source .venv/bin/activate   # Windows: .venv\Scripts\activate
pip install -r requirements.txt
cp .env.example .env                                 # then fill in OPENAI_API_KEY
docker compose up -d db                              # Postgres + pgvector
```

## Common commands
| Task | Command |
| --- | --- |
| Run app | `uvicorn app.main:app --reload` |
| Run tests | `pytest` |
| Retrieval evals | `python -m evals.retrieval` |
| Generation evals | `python -m evals.generation` |
| DB up | `docker compose up -d db` |

## Status
Phase 0 (scaffold) complete. See `PROJECT_PLAN.md` for the build order.

<!-- Phase 6: add architecture diagram, the retrieval ablation table (NDCG@5 baseline → hybrid),
     and the judge validation κ here — front and center. -->
