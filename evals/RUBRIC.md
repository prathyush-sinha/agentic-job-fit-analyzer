# Generation Eval Rubric (Phase 5)

Score each agent output on three dimensions, 1–5. The judge returns a score plus a
one-line justification per dimension. **Anything ≤2 on faithfulness is a hard fail**
regardless of the other scores.

## 1. Faithfulness (most important — anti-hallucination)
Does every claim trace to actual evidence (the resume text and the retrieved postings)?
- **5** — Every matched skill, gap, and rewritten bullet is grounded in the resume and/or a retrieved posting. No invented requirements, no fabricated experience.
- **3** — Mostly grounded, but one claim is an unsupported inference (e.g. asserts the role "requires Kubernetes" when no posting says so).
- **1** — Invents requirements or attributes experience to the candidate that isn't in the resume.
- **Judge check:** for each claim, ask "which source span supports this?" If none, dock the score.

## 2. Relevance
Are the retrieved postings and identified gaps actually relevant to the stated target role?
- **5** — Postings match the target role/seniority; gaps are the ones that matter for *this* role.
- **3** — Generally on-target but includes some loosely related postings or a low-priority gap framed as critical.
- **1** — Retrieves off-target roles or fixates on irrelevant gaps.

## 3. Actionability
Could the candidate actually act on the output?
- **5** — Each gap pairs with a concrete next step; bullet rewrites are specific, quantified, usable as-is.
- **3** — Advice is correct but generic ("learn more about LLMs").
- **1** — Vague, no concrete steps, or rewrites that lose the candidate's real accomplishments.

## Judge prompt skeleton
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

## Judge validation
Hand-score ~30 outputs yourself on the same scale, then compute Cohen's κ between your
scores and the judge's per dimension. Report κ in the README. An unvalidated LLM judge is
just another unmeasured model.

Use a **different OpenAI model for the judge** than the one that generated the output
(e.g. generate with `gpt-4o-mini`, judge with `gpt-4o`). A model grading its own family's
output tends to score it generously; the separate model plus the human-κ check is the
defense against that bias.
