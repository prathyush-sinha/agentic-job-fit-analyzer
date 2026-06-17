# CLAUDE.md

Project context for Claude Code. Read this before doing anything.

## What this project is
An agentic job-fit analyzer. Given a resume and a target role, it retrieves relevant job postings, analyzes fit, and returns a structured report (matched skills, gaps, tailored bullet rewrites). It is a portfolio project; the goal is to demonstrate an agent loop, RAG depth, and rigorous evaluation. Correctness of the eval harness matters more than feature breadth.

## Architecture
LangGraph state machine: planner -> retriever (RAG) -> analyzer -> critic/fact-check -> synthesizer. The critic loops back to the retriever (max 2 retries) when the analysis cites a gap not grounded in retrieved evidence.

## Stack
- Python 3.11+
- Postgres + pgvector for the vector store (hosted free Neon instance; `docker compose` config also provided)
- Retrieval: dense (local `BAAI/bge-small-en-v1.5` embeddings, 384-dim) + BM25 (sparse), fused with reciprocal rank fusion, then a cross-encoder reranker
- Agent orchestration: LangGraph
- Serving: FastAPI, containerized with Docker
- LLM for agent nodes and the eval judge: Google Gemini API, free tier (`gemini-2.5-flash` for the agent, a different Gemini model for the judge — `gemini-2.0-flash` had a 0 free-tier quota on this project). Model names come from env; the API key (`GOOGLE_API_KEY`) stays in env and is never hardcoded.
- Embeddings are pluggable via `EMBEDDING_PROVIDER` (`local` default — free, no rate limits, embeds the full corpus; `gemini` and `openai` reachable as fallbacks). Gemini's free tier is too rate-limited for bulk embedding, so the full ~28K-chunk index uses the local model.

## Conventions
- Keep dense-only retrieval reachable behind a flag — it is the ablation baseline. Do not delete it.
- All agent outputs must conform to a defined Pydantic schema. No free-form text returns.
- Every retrieval and generation change must be runnable through the eval harness. If you change a prompt or a node, run the evals and report the metric delta.
- Write a test alongside new logic. Prefer small, composable functions.
- Commit at the end of each build phase with a clear message.

## Eval is first-class
- Retrieval metrics: precision@k, recall@k, NDCG@k, MRR against the labeled gold set in `evals/gold/`.
- Generation metrics: an LLM-as-judge scoring faithfulness, relevance, actionability (rubric in `evals/RUBRIC.md`).
- The judge is itself validated against human labels (Cohen's kappa). Do not treat judge scores as ground truth without that validation.
- Use a different Gemini model for the judge than for the agent (e.g. agent = `gemini-2.0-flash`, judge = `gemini-2.5-pro`). Same-model self-grading inflates scores; the human-kappa validation is what keeps the judge honest.
- Log tokens, cost-per-run, and p50/p95 latency on every run.

## Hard rules
- Never read, print, or commit `.env` or any API key. Secrets stay in the environment.
- Do not commit the raw scraped corpus if licensing is unclear — keep it gitignored, document the source.
- Do not add a feature that bypasses the eval harness.
- Ask before introducing a new heavyweight dependency; prefer the stack above.

## Common commands
- Run app: `uvicorn app.main:app --reload`
- Run tests: `pytest`
- Run retrieval evals: `python -m evals.retrieval`
- Run generation evals: `python -m evals.generation`
- DB up: `docker compose up -d db`
