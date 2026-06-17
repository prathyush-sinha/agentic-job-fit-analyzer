"""LangGraph agent: planner -> retriever -> analyzer -> critic -> synthesizer.

The critic fact-checks the draft analysis against retrieved evidence. If a claim
isn't grounded, it loops back to the retriever (capped at MAX_RETRIES) with
follow-up queries; otherwise the synthesizer produces the final FitReport.
"""

from __future__ import annotations

from typing import TypedDict

from langgraph.graph import END, START, StateGraph

from app.config import Settings, get_settings
from app.llm import generate_structured
from app.retrieval import Hit, search
from app.schemas import Critique, FitAnalysis, FitReport, Plan

MAX_RETRIES = 2
EVIDENCE_PER_QUERY = 6
MAX_QUERIES = 3


class FitState(TypedDict, total=False):
    resume: str
    target_role: str
    queries: list[str]
    evidence: list[dict]      # deduped posting evidence (serialized)
    analysis: FitAnalysis
    critique: Critique
    retries: int
    report: FitReport
    tokens: int               # cumulative LLM tokens (observability)


# --- helpers --------------------------------------------------------------

def _format_evidence(evidence: list[dict]) -> str:
    lines = []
    for i, e in enumerate(evidence, 1):
        lines.append(
            f"[{i}] {e['company']} — {e['title']} ({e['section']})\n{e['content']}"
        )
    return "\n\n".join(lines) if lines else "(no evidence retrieved)"


def _hit_to_evidence(h: Hit) -> dict:
    return {
        "posting_id": h.posting_id, "company": h.company, "title": h.title,
        "section": h.section, "content": h.content,
    }


# --- nodes ----------------------------------------------------------------

def planner_node(state: FitState, settings: Settings) -> dict:
    prompt = (
        "You plan how to retrieve relevant job postings for a candidate.\n"
        f"TARGET ROLE: {state['target_role']}\n\nRESUME:\n{state['resume']}\n\n"
        "Produce 2-4 retrieval queries and the focus areas that matter for this role."
    )
    plan, usage = generate_structured(prompt, Plan, settings)
    return {
        "queries": plan.search_queries[:MAX_QUERIES],
        "tokens": state.get("tokens", 0) + usage.prompt_tokens + usage.output_tokens,
    }


def retriever_node(state: FitState, settings: Settings) -> dict:
    existing = {e["posting_id"]: e for e in state.get("evidence", [])}
    for query in state.get("queries", [])[:MAX_QUERIES]:
        for hit in search(query, k=EVIDENCE_PER_QUERY, settings=settings):
            existing.setdefault(hit.posting_id, _hit_to_evidence(hit))
    return {"evidence": list(existing.values())}


def analyzer_node(state: FitState, settings: Settings) -> dict:
    prompt = (
        "Analyze the candidate's fit for the target role. Ground every matched "
        "skill and gap in the resume and/or the retrieved postings below. Do not "
        "invent requirements.\n\n"
        f"TARGET ROLE: {state['target_role']}\n\nRESUME:\n{state['resume']}\n\n"
        f"RETRIEVED POSTINGS:\n{_format_evidence(state.get('evidence', []))}"
    )
    analysis, usage = generate_structured(prompt, FitAnalysis, settings)
    return {
        "analysis": analysis,
        "tokens": state.get("tokens", 0) + usage.prompt_tokens + usage.output_tokens,
    }


def critic_node(state: FitState, settings: Settings) -> dict:
    prompt = (
        "You are a fact-checker. For every matched skill and gap in the ANALYSIS, "
        "check it is supported by the RESUME or a RETRIEVED POSTING. If any claim "
        "is unsupported (e.g. a requirement no posting states), set grounded=false, "
        "list the issues, and propose follow_up_queries to retrieve missing evidence.\n\n"
        f"RESUME:\n{state['resume']}\n\n"
        f"RETRIEVED POSTINGS:\n{_format_evidence(state.get('evidence', []))}\n\n"
        f"ANALYSIS:\n{state['analysis'].model_dump_json(indent=2)}"
    )
    critique, usage = generate_structured(prompt, Critique, settings)
    return {
        "critique": critique,
        "queries": critique.follow_up_queries[:MAX_QUERIES],
        "retries": state.get("retries", 0) + 1,
        "tokens": state.get("tokens", 0) + usage.prompt_tokens + usage.output_tokens,
    }


def synthesizer_node(state: FitState, settings: Settings) -> dict:
    titles = [f"{e['company']} — {e['title']}" for e in state.get("evidence", [])]
    prompt = (
        "Produce the final job-fit report for the target role, incorporating the "
        "critique. Keep only grounded claims. Bullet rewrites must be specific and "
        "quantified.\n\n"
        f"TARGET ROLE: {state['target_role']}\n\n"
        f"ANALYSIS:\n{state['analysis'].model_dump_json(indent=2)}\n\n"
        f"CRITIQUE:\n{state['critique'].model_dump_json(indent=2)}\n\n"
        f"EVIDENCE POSTINGS: {titles}"
    )
    report, usage = generate_structured(prompt, FitReport, settings)
    return {
        "report": report,
        "tokens": state.get("tokens", 0) + usage.prompt_tokens + usage.output_tokens,
    }


def route_after_critic(state: FitState) -> str:
    """Loop back to retrieval if ungrounded and retries remain; else synthesize."""
    critique = state.get("critique")
    if critique is not None and not critique.grounded and state.get("retries", 0) < MAX_RETRIES:
        return "retriever"
    return "synthesizer"


def build_graph(settings: Settings | None = None):
    settings = settings or get_settings()

    def bind(fn):
        return lambda state: fn(state, settings)

    g = StateGraph(FitState)
    g.add_node("planner", bind(planner_node))
    g.add_node("retriever", bind(retriever_node))
    g.add_node("analyzer", bind(analyzer_node))
    g.add_node("critic", bind(critic_node))
    g.add_node("synthesizer", bind(synthesizer_node))

    g.add_edge(START, "planner")
    g.add_edge("planner", "retriever")
    g.add_edge("retriever", "analyzer")
    g.add_edge("analyzer", "critic")
    g.add_conditional_edges("critic", route_after_critic,
                            {"retriever": "retriever", "synthesizer": "synthesizer"})
    g.add_edge("synthesizer", END)
    return g.compile()


def analyze_fit(resume: str, target_role: str, settings: Settings | None = None) -> FitState:
    """Run the agent end-to-end. Returns the final state (report under 'report')."""
    settings = settings or get_settings()
    graph = build_graph(settings)
    return graph.invoke({"resume": resume, "target_role": target_role, "retries": 0})
