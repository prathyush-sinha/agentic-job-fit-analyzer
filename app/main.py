"""FastAPI entrypoint.

POST /analyze runs the LangGraph agent (planner -> retriever -> analyzer ->
critic -> synthesizer) and returns the structured FitReport plus run metadata
(retries, LLM tokens, latency) for observability.
"""

from __future__ import annotations

import time

from fastapi import FastAPI
from pydantic import BaseModel, Field

from app.graph import analyze_fit
from app.schemas import FitReport

app = FastAPI(title="Agentic Job-Fit Analyzer")


class AnalyzeRequest(BaseModel):
    resume: str = Field(..., min_length=1)
    target_role: str = Field(..., min_length=1)


class RunMeta(BaseModel):
    retries: int
    llm_tokens: int
    evidence_count: int
    latency_s: float


class AnalyzeResponse(BaseModel):
    report: FitReport
    meta: RunMeta


@app.get("/health")
def health() -> dict[str, str]:
    return {"status": "ok"}


@app.post("/analyze", response_model=AnalyzeResponse)
def analyze(req: AnalyzeRequest) -> AnalyzeResponse:
    t0 = time.time()
    state = analyze_fit(req.resume, req.target_role)
    return AnalyzeResponse(
        report=state["report"],
        meta=RunMeta(
            retries=state.get("retries", 0),
            llm_tokens=state.get("tokens", 0),
            evidence_count=len(state.get("evidence", [])),
            latency_s=round(time.time() - t0, 1),
        ),
    )
