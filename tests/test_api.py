"""API tests. The agent is mocked so these stay offline and fast."""

from fastapi.testclient import TestClient

import app.main as main
from app.main import app
from app.schemas import FitReport


def _fake_state():
    report = FitReport(
        target_role="ML Engineer", overall_fit="Strong fit.", fit_score=80,
        matched_skills=[], gaps=[], bullet_rewrites=[], evidence_used=["Acme — MLE"],
    )
    return {"report": report, "retries": 1, "tokens": 1234, "evidence": [{}, {}]}


def test_analyze_endpoint_shape(monkeypatch):
    monkeypatch.setattr(main, "analyze_fit", lambda resume, target_role: _fake_state())
    client = TestClient(app)
    resp = client.post("/analyze", json={"resume": "Python ML engineer", "target_role": "ML Engineer"})
    assert resp.status_code == 200
    body = resp.json()
    assert body["report"]["fit_score"] == 80
    assert body["meta"]["retries"] == 1
    assert body["meta"]["llm_tokens"] == 1234
    assert body["meta"]["evidence_count"] == 2


def test_analyze_validates_empty_input():
    client = TestClient(app)
    resp = client.post("/analyze", json={"resume": "", "target_role": "X"})
    assert resp.status_code == 422
