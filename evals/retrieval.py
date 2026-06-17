"""Retrieval ablation: dense-only vs BM25-only vs hybrid+rerank.

Evaluates each system on the gold set at the posting level (chunk hits are
deduped to their posting), reports precision/recall/NDCG@k + MRR averaged over
queries, plus per-query latency. Saves a table to evals/results/.

Run:  python -m evals.retrieval
"""

from __future__ import annotations

import json
import statistics
import time
from collections.abc import Callable
from pathlib import Path

from app.config import get_settings
from app.retrieval import Hit, bm25_search, dense_search, hybrid_search
from evals.gold import load_gold
from evals.metrics import mrr, ndcg_at_k, precision_at_k, recall_at_k

RESULTS_DIR = Path("evals/results")
KS = (5, 10)
RETRIEVE_DEPTH = 60  # chunks pulled before deduping to postings


def ranked_postings(hits: list[Hit]) -> list[str]:
    """Dedupe chunk hits to a ranked list of unique posting ids (order preserved)."""
    seen: set[str] = set()
    out: list[str] = []
    for h in hits:
        if h.posting_id not in seen:
            seen.add(h.posting_id)
            out.append(h.posting_id)
    return out


def _systems(settings) -> dict[str, Callable[[str], list[Hit]]]:
    return {
        "dense-only": lambda q: dense_search(q, RETRIEVE_DEPTH, settings),
        "bm25-only": lambda q: bm25_search(q, RETRIEVE_DEPTH, settings),
        "hybrid+rerank": lambda q: hybrid_search(q, settings.rerank_top, settings),
    }


def evaluate() -> dict:
    settings = get_settings()
    gold = load_gold()
    systems = _systems(settings)
    report: dict[str, dict] = {}

    for name, fn in systems.items():
        per_query: list[dict] = []
        latencies: list[float] = []
        for item in gold:
            t0 = time.time()
            ranked = ranked_postings(fn(item.resume))
            latencies.append(time.time() - t0)
            rel = item.relevant_posting_ids
            row = {"query": item.query_id, "mrr": mrr(ranked, rel)}
            for k in KS:
                row[f"p@{k}"] = precision_at_k(ranked, rel, k)
                row[f"r@{k}"] = recall_at_k(ranked, rel, k)
                row[f"ndcg@{k}"] = ndcg_at_k(ranked, rel, k)
            per_query.append(row)

        metric_keys = [c for c in per_query[0] if c != "query"]
        agg = {m: statistics.mean(q[m] for q in per_query) for m in metric_keys}
        report[name] = {
            "aggregate": agg,
            "per_query": per_query,
            "latency_p50_s": round(statistics.median(latencies), 3),
            "latency_p95_s": round(sorted(latencies)[max(0, int(0.95 * len(latencies)) - 1)], 3),
        }
    return report


def _fmt_table(report: dict) -> str:
    cols = ["ndcg@5", "ndcg@10", "p@5", "r@10", "mrr"]
    head = f"| {'system':<14} | " + " | ".join(f"{c:>8}" for c in cols) + " | lat p50 | lat p95 |"
    sep = "|" + "-" * 16 + "|" + ("|".join(["-" * 10] * len(cols))) + "|" + "-" * 9 + "|" + "-" * 9 + "|"
    lines = [head, sep]
    for name, data in report.items():
        agg = data["aggregate"]
        cells = " | ".join(f"{agg[c]:>8.3f}" for c in cols)
        lines.append(
            f"| {name:<14} | {cells} | {data['latency_p50_s']:>7.3f} | {data['latency_p95_s']:>7.3f} |"
        )
    return "\n".join(lines)


def main() -> None:
    report = evaluate()
    table = _fmt_table(report)
    print("\nRetrieval ablation (posting-level, averaged over gold queries):\n")
    print(table)

    RESULTS_DIR.mkdir(parents=True, exist_ok=True)
    (RESULTS_DIR / "retrieval_ablation.json").write_text(
        json.dumps(report, indent=2), encoding="utf-8"
    )
    md = (
        "# Retrieval Ablation\n\n"
        "Posting-level metrics averaged over the gold queries "
        "(`evals/gold/gold.jsonl`). Relevance = title matches the target role "
        "(objective proxy; see `evals/gold.py`).\n\n"
        f"```\n{table}\n```\n"
    )
    (RESULTS_DIR / "retrieval_ablation.md").write_text(md, encoding="utf-8")
    print(f"\nSaved -> {RESULTS_DIR}/retrieval_ablation.(md|json)")


if __name__ == "__main__":
    main()
