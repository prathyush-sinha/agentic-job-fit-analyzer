"""Tests for agent routing + structure (no LLM/network)."""

from app.graph import MAX_RETRIES, route_after_critic
from app.schemas import Critique, FitReport


def _crit(grounded: bool) -> Critique:
    return Critique(grounded=grounded, issues=[], follow_up_queries=["x"])


def test_route_grounded_goes_to_synthesizer():
    state = {"critique": _crit(True), "retries": 1}
    assert route_after_critic(state) == "synthesizer"


def test_route_ungrounded_with_retries_left_loops_back():
    state = {"critique": _crit(False), "retries": 1}
    assert route_after_critic(state) == "retriever"


def test_route_ungrounded_at_retry_cap_synthesizes():
    state = {"critique": _crit(False), "retries": MAX_RETRIES}
    assert route_after_critic(state) == "synthesizer"


def test_fit_report_rejects_out_of_range_score():
    import pytest
    from pydantic import ValidationError

    with pytest.raises(ValidationError):
        FitReport(target_role="X", overall_fit="ok", fit_score=150,
                  matched_skills=[], gaps=[], bullet_rewrites=[])
