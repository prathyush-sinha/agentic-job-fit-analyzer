"""Structured agent outputs. Every node returns one of these — no free-form text.

Grounding is baked into the schema: matched skills and gaps must carry the
resume text and/or posting evidence that supports them, so the critic node can
verify each claim traces to retrieved evidence.
"""

from __future__ import annotations

from pydantic import BaseModel, Field


class Plan(BaseModel):
    """Planner output: how to retrieve evidence for this resume + role."""
    search_queries: list[str] = Field(
        ..., description="2-4 retrieval queries to surface relevant postings"
    )
    focus_areas: list[str] = Field(
        ..., description="Skills/requirements to pay attention to for this role"
    )


class MatchedSkill(BaseModel):
    skill: str
    resume_evidence: str = Field(..., description="Where in the resume this is shown")
    posting_evidence: str = Field(..., description="The posting requirement it matches")


class Gap(BaseModel):
    requirement: str = Field(..., description="A requirement the candidate lacks/underdemonstrates")
    why_it_matters: str
    posting_evidence: str = Field(..., description="The retrieved posting text stating this requirement")
    suggested_next_step: str = Field(..., description="Concrete, actionable step to close the gap")


class BulletRewrite(BaseModel):
    original: str = Field(..., description="An original resume bullet")
    rewritten: str = Field(..., description="A specific, quantified rewrite for the target role")
    rationale: str


class FitAnalysis(BaseModel):
    """Analyzer output (draft, pre-critique)."""
    matched_skills: list[MatchedSkill]
    gaps: list[Gap]
    bullet_rewrites: list[BulletRewrite]


class Critique(BaseModel):
    """Critic / fact-check output."""
    grounded: bool = Field(..., description="True if every claim traces to the resume or a retrieved posting")
    issues: list[str] = Field(default_factory=list, description="Specific ungrounded or unsupported claims")
    follow_up_queries: list[str] = Field(
        default_factory=list, description="Retrieval queries to fetch missing evidence (if not grounded)"
    )


class FitReport(BaseModel):
    """Synthesizer output: the final structured report returned to the user."""
    target_role: str
    overall_fit: str = Field(..., description="2-3 sentence grounded summary of fit")
    fit_score: int = Field(..., ge=0, le=100, description="Overall fit, 0-100")
    matched_skills: list[MatchedSkill]
    gaps: list[Gap]
    bullet_rewrites: list[BulletRewrite]
    evidence_used: list[str] = Field(
        default_factory=list, description="Company — title of postings used as evidence"
    )
