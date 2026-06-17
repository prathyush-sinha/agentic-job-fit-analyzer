# Agentic Job-Fit Analyzer — Build Plan & Eval Rubric

A portfolio project that demonstrates three things hiring managers screen for in AI Engineer roles: an agent loop (not just an LLM call), RAG depth (hybrid retrieval + reranking), and eval discipline (validated metrics, not vibes).

Time estimates assume part-time solo work (evenings/weekends). Total: ~3–4 weeks. If you have to cut, cut corpus size and stretch features — never the eval harness, since that is the differentiating part.

---

## Build order

### Phase 0 — Setup & scaffold · ~half a day
- [ ] Create repo, Python venv, `requirements.txt`, `.gitignore` (exclude `.env`).
- [ ] Drop in `CLAUDE.md` (provided separately) so Claude Code has project context.
- [ ] Spin up Postgres + pgvector locally (Docker) — reuse your existing setup.
- [ ] Put API keys in `.env`; never commit them. LLM provider for the agent and judge: **Google Gemini**, free tier (set `GOOGLE_API_KEY`). Provider is pluggable via `EMBEDDING_PROVIDER` (`openai` reachable as fallback).
- [ ] Acquire a job-postings corpus: a public dataset (e.g. a Kaggle job-postings set) or scrape a few thousand. Aim for ~3–5K postings with title, company, description.

### Phase 1 — RAG baseline (dense only) · ~2–3 days
- [ ] Ingest postings → clean → chunk (by section: requirements / responsibilities / qualifications).
- [ ] Embed chunks (local `BAAI/bge-small-en-v1.5`, 384-dim — free, no rate limits), store in pgvector. Full corpus indexed (~28K chunks). Embeddings pluggable via `EMBEDDING_PROVIDER`; Gemini/OpenAI reachable but their free tiers are too rate-limited for bulk embedding.
- [ ] Dense retrieval: embed a resume, cosine top-k over chunks.
- [ ] Sanity-check retrieval manually on 5–10 resumes. **Commit.** This is your baseline.

### Phase 2 — Hybrid retrieval + reranking · ~2 days
- [ ] Add BM25 (sparse) retrieval over the same corpus.
- [ ] Fuse dense + sparse results (reciprocal rank fusion is simplest).
- [ ] Add a cross-encoder reranker over the fused candidate set (e.g. a `bge-reranker` or a hosted rerank API).
- [ ] Keep dense-only reachable behind a flag — you need it as the ablation baseline. **Commit.**

### Phase 3 — Retrieval eval harness · ~2 days
- [ ] Hand-label a gold set: 40–60 `(resume, target_role)` pairs, each with the postings you judge genuinely relevant.
- [ ] Implement precision@k, recall@k, NDCG@k, MRR.
- [ ] Run the ablation: dense-only vs BM25-only vs hybrid+rerank. Save the table.
- [ ] This table produces your resume bullet (NDCG@5 went from X baseline to Y). **Commit.**

### Phase 4 — Agent graph · ~3–4 days
- [ ] Define the LangGraph state (resume, target role, retrieved evidence, draft analysis, critique, retry count).
- [ ] Nodes: planner → retriever → analyzer → critic/fact-check → synthesizer.
- [ ] The critic loops back to the retriever when the analysis cites a gap not grounded in evidence; cap retries (e.g. 2).
- [ ] Enforce a structured output schema (matched skills, gaps, tailored bullet rewrites). **Commit per node.**

### Phase 5 — Generation eval + observability · ~2–3 days
- [ ] Define the rubric (below). Build an LLM-as-judge that scores each dimension.
- [ ] Validate the judge: hand-score ~30 outputs, compute agreement with the judge (Cohen's κ). Report it.
- [ ] Regression suite: rerun retrieval + generation evals on every prompt/graph change; store traces.
- [ ] Log tokens, cost-per-run, p50/p95 latency. **Commit.**

### Phase 6 — Serve & package · ~1–2 days
- [ ] FastAPI endpoint: POST resume + target role → structured fit report (stream if you want).
- [ ] Dockerize. Reuse your GCP Cloud SQL pattern if you want it hosted.
- [ ] README with architecture diagram, the ablation table, and the κ number front and center.

### Phase 7 — Stretch (optional)
- [ ] Web-search tool node for live company context.
- [ ] Multi-resume comparison / ranking.
- [ ] Small front end (Streamlit or Next.js).
- [ ] Embedding-model A/B.

---

## Eval rubric (Phase 5)

Score each agent output on three dimensions. Use a 1–5 scale; the judge returns a score plus a one-line justification per dimension. Treat anything ≤2 on faithfulness as a hard fail regardless of the other scores.

### 1. Faithfulness (the most important — this is the anti-hallucination dimension)
Does every claim trace to actual evidence (the resume text and the retrieved postings)?
- **5** — Every matched skill, gap, and rewritten bullet is grounded in the resume and/or a retrieved posting. No invented requirements, no fabricated experience.
- **3** — Mostly grounded, but one claim is an unsupported inference (e.g. asserts the role "requires Kubernetes" when no posting says so).
- **1** — Invents requirements or attributes experience to the candidate that isn't in the resume.
- **Judge check:** for each claim, ask "which source span supports this?" If none, dock the score.

### 2. Relevance
Are the retrieved postings and the identified gaps actually relevant to the stated target role?
- **5** — Postings match the target role/seniority; gaps are the ones that matter for *this* role.
- **3** — Generally on-target but includes some loosely related postings or a low-priority gap framed as critical.
- **1** — Retrieves off-target roles or fixates on irrelevant gaps.

### 3. Actionability
Could the candidate actually act on the output?
- **5** — Each gap pairs with a concrete next step; bullet rewrites are specific, quantified, and usable as-is.
- **3** — Advice is correct but generic ("learn more about LLMs").
- **1** — Vague, no concrete steps, or rewrites that lose the candidate's real accomplishments.

### Judge prompt skeleton
```
You are evaluating an AI-generated job-fit analysis.
Inputs you are given: the resume, the target role, the retrieved postings, and the analysis.
For each dimension (faithfulness, relevance, actionability):
  - assign a score 1-5 using the rubric provided
  - cite the specific source span that supports or contradicts each scored claim
  - give a one-sentence justification
Return strict JSON: {"faithfulness": {...}, "relevance": {...}, "actionability": {...}}.
Do not reward fluent writing that lacks grounding.
```

### Judge validation
Hand-score ~30 outputs yourself on the same scale, then compute Cohen's κ between your scores and the judge's per dimension. Report κ in the README. An unvalidated LLM judge is just another unmeasured model — saying so, and then measuring it, is the senior signal.

Since you're using Gemini for both the agent and the judge, use a *different* Gemini model for the judge than the one that generated the output (e.g. generate with `gemini-2.0-flash`, judge with `gemini-2.5-pro`). A model grading its own family's output tends to score it generously; using a separate model plus the human-κ check is your defense against that bias — and being able to explain this tradeoff is itself an interview-worthy point.

---

## What lands on the resume (target framing)
- "Built an agentic job-fit analyzer (LangGraph) with a planner–retriever–critic loop over ~5K postings; hybrid retrieval + cross-encoder reranking lifted NDCG@5 from 0.61 (dense-only baseline) to 0.88."
- "Designed an eval harness with an LLM-as-judge validated against human labels (κ = 0.79), plus a regression suite tracking faithfulness, cost-per-run, and p95 latency across prompt revisions."

Fill in your real numbers. A real 0.61→0.88 with a named baseline beats any unanchored "~50% gain."
