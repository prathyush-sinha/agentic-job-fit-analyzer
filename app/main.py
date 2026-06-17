"""FastAPI entrypoint. Phase 0 scaffold: health check only.

The job-fit report endpoint (POST resume + target role -> structured report)
arrives in Phase 6.
"""

from fastapi import FastAPI

app = FastAPI(title="Agentic Job-Fit Analyzer")


@app.get("/health")
def health() -> dict[str, str]:
    return {"status": "ok"}
