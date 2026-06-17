"""Gold set for retrieval evaluation.

Relevance is defined by an objective, retrieval-independent proxy: a posting is
relevant to a (resume, target_role) query if its title matches the role's
pattern. This is transparent and reproducible (not circular — it never consults
the embedding/BM25 retriever). A production eval would replace this with human
relevance judgments; documented as a known limitation.

Build with:  python -m evals.gold      (writes evals/gold/gold.jsonl)
"""

from __future__ import annotations

import json
import re
from dataclasses import dataclass
from pathlib import Path

from app.config import Settings, get_settings
from app.db import get_connection

SAMPLES = Path("evals/samples")
GOLD_PATH = Path("evals/gold/gold.jsonl")
MIN_RELEVANT = 5  # roles with fewer relevant postings are skipped (metrics too noisy)


@dataclass
class RoleSpec:
    role: str
    pattern: str        # case-insensitive regex matched against posting title
    resume_file: str


# Roles chosen for having enough postings in the indexed corpus. Patterns aim to
# be specific so relevant sets stay distinct across roles.
ROLE_SPECS: list[RoleSpec] = [
    RoleSpec("ML/AI Engineer",
             r"\b(machine learning|ml engineer|ai engineer|applied (scientist|ai|ml)|deep learning|nlp)\b",
             "resume_ml_engineer.txt"),
    RoleSpec("Data Analyst/Scientist",
             r"\b(data analyst|data scientist|analytics)\b",
             "resume_data_analyst.txt"),
    RoleSpec("Data Engineer",
             r"\b(data engineer|analytics engineer)\b",
             "resume_data_engineer.txt"),
    RoleSpec("Backend/Software Engineer",
             r"\b(backend|back end|back-end|software engineer|full stack|fullstack|distributed systems)\b",
             "resume_backend_engineer.txt"),
    RoleSpec("Frontend Engineer",
             r"\b(frontend|front end|front-end|web engineer|ui engineer)\b",
             "resume_frontend_engineer.txt"),
    RoleSpec("Mobile Engineer",
             r"\b(ios|android|mobile) (engineer|developer)\b",
             "resume_mobile_engineer.txt"),
    RoleSpec("DevOps/SRE/Infra",
             r"\b(devops|site reliability|sre|infrastructure|platform engineer|cloud engineer)\b",
             "resume_devops_sre.txt"),
    RoleSpec("Security Engineer",
             r"\bsecurity\b",
             "resume_security_engineer.txt"),
    RoleSpec("Engineering Manager",
             r"\b(engineering manager|eng manager|director of engineering)\b",
             "resume_engineering_manager.txt"),
    RoleSpec("Product Manager",
             r"\b(product manager|product management|group product)\b",
             "resume_product_manager.txt"),
    RoleSpec("Product Designer",
             r"\bdesign(er)?\b",
             "resume_designer.txt"),
    RoleSpec("Account Executive / Sales",
             r"\b(account executive|sales|business development)\b",
             "resume_sales_ae.txt"),
    RoleSpec("Marketing",
             r"\b(marketing|growth|brand)\b",
             "resume_marketing.txt"),
    RoleSpec("Finance/Accounting",
             r"\b(finance|financial|accounting|accountant|controller)\b",
             "resume_finance.txt"),
    RoleSpec("Customer Success/Support",
             r"\b(customer success|customer support|support engineer|solutions engineer)\b",
             "resume_customer_success.txt"),
    RoleSpec("Recruiter/People",
             r"\b(recruiter|recruiting|talent|people partner|human resources)\b",
             "resume_recruiter.txt"),
]


@dataclass
class GoldItem:
    query_id: str
    target_role: str
    resume: str
    relevant_posting_ids: list[str]


def _load_postings(settings: Settings) -> list[tuple[str, str]]:
    with get_connection(settings) as conn, conn.cursor() as cur:
        cur.execute("SELECT DISTINCT posting_id, title FROM chunks;")
        return cur.fetchall()


def build_gold(settings: Settings | None = None, min_relevant: int = MIN_RELEVANT) -> list[GoldItem]:
    settings = settings or get_settings()
    postings = _load_postings(settings)
    items: list[GoldItem] = []
    for spec in ROLE_SPECS:
        rx = re.compile(spec.pattern, re.IGNORECASE)
        relevant = sorted(pid for pid, title in postings if rx.search(title or ""))
        if len(relevant) < min_relevant:
            print(f"  skip {spec.role}: only {len(relevant)} relevant (< {min_relevant})")
            continue
        resume = (SAMPLES / spec.resume_file).read_text(encoding="utf-8")
        items.append(GoldItem(
            query_id=spec.resume_file.replace("resume_", "").replace(".txt", ""),
            target_role=spec.role,
            resume=resume,
            relevant_posting_ids=relevant,
        ))
        print(f"  {spec.role}: {len(relevant)} relevant postings")
    return items


def write_gold(items: list[GoldItem], path: Path = GOLD_PATH) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("w", encoding="utf-8") as fh:
        for it in items:
            fh.write(json.dumps({
                "query_id": it.query_id,
                "target_role": it.target_role,
                "resume": it.resume,
                "relevant_posting_ids": it.relevant_posting_ids,
            }) + "\n")


def load_gold(path: Path = GOLD_PATH) -> list[GoldItem]:
    items: list[GoldItem] = []
    with path.open(encoding="utf-8") as fh:
        for line in fh:
            line = line.strip()
            if line:
                d = json.loads(line)
                items.append(GoldItem(**d))
    return items


def main() -> None:
    print("Building gold set from indexed corpus...")
    items = build_gold()
    write_gold(items)
    total = sum(len(i.relevant_posting_ids) for i in items)
    print(f"\nWrote {len(items)} queries ({total} relevance judgments) -> {GOLD_PATH}")


if __name__ == "__main__":
    main()
