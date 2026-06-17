"""Collect job postings from public job-board APIs into data/raw/.

Sources are public ATS APIs (Greenhouse, Lever) meant for programmatic
consumption. Output is JSONL of RawPosting records plus a manifest.

Usage:
    python -m ingest.scrape                 # defaults: ~4000 postings
    python -m ingest.scrape --target 5000 --per-company-cap 400
    python -m ingest.scrape --out data/raw/postings.jsonl
"""

from __future__ import annotations

import argparse
import json
import sys
import time
from datetime import datetime, timezone
from pathlib import Path

import httpx
from pydantic import ValidationError

from ingest.html_text import strip_html
from ingest.models import RawPosting
from ingest.sources import GREENHOUSE_TOKENS, LEVER_TOKENS

USER_AGENT = "jobfit-analyzer/0.1 (portfolio project; public job-board APIs)"
GREENHOUSE_URL = "https://boards-api.greenhouse.io/v1/boards/{token}/jobs?content=true"
LEVER_URL = "https://api.lever.co/v0/postings/{token}?mode=json"

DEFAULT_TARGET = 4000
DEFAULT_PER_COMPANY_CAP = 400
DEFAULT_OUT = Path("data/raw/postings.jsonl")
REQUEST_DELAY_S = 0.4  # be polite between company requests


def _now() -> str:
    return datetime.now(timezone.utc).isoformat()


def fetch_json(client: httpx.Client, url: str, retries: int = 3) -> object | None:
    """GET JSON with simple exponential backoff. Returns None on hard failure."""
    for attempt in range(retries):
        try:
            resp = client.get(url, timeout=20.0)
            if resp.status_code == 404:
                return None  # company not on this board; skip quietly
            resp.raise_for_status()
            return resp.json()
        except (httpx.HTTPError, ValueError) as exc:
            if attempt == retries - 1:
                print(f"  ! giving up on {url}: {exc}", file=sys.stderr)
                return None
            time.sleep(1.5 * (attempt + 1))
    return None


def normalize_greenhouse(token: str, job: dict) -> RawPosting | None:
    try:
        loc = (job.get("location") or {}).get("name")
        return RawPosting(
            id=f"greenhouse:{job['id']}",
            source="greenhouse",
            company=token,
            title=job["title"],
            location=loc,
            description=strip_html(job.get("content")),
            url=job.get("absolute_url", ""),
            scraped_at=_now(),
        )
    except (KeyError, ValidationError):
        return None


def normalize_lever(token: str, job: dict) -> RawPosting | None:
    try:
        cats = job.get("categories") or {}
        # Lever provides plain text directly; fall back to stripping HTML.
        desc = job.get("descriptionPlain") or strip_html(job.get("description"))
        return RawPosting(
            id=f"lever:{job['id']}",
            source="lever",
            company=token,
            title=job["text"],
            location=cats.get("location"),
            description=desc,
            url=job.get("hostedUrl", ""),
            scraped_at=_now(),
        )
    except (KeyError, ValidationError):
        return None


def fetch_greenhouse(client: httpx.Client, token: str, cap: int) -> list[RawPosting]:
    data = fetch_json(client, GREENHOUSE_URL.format(token=token))
    if not isinstance(data, dict):
        return []
    out: list[RawPosting] = []
    for job in data.get("jobs", [])[:cap]:
        p = normalize_greenhouse(token, job)
        if p and p.description:
            out.append(p)
    return out


def fetch_lever(client: httpx.Client, token: str, cap: int) -> list[RawPosting]:
    data = fetch_json(client, LEVER_URL.format(token=token))
    if not isinstance(data, list):
        return []
    out: list[RawPosting] = []
    for job in data[:cap]:
        p = normalize_lever(token, job)
        if p and p.description:
            out.append(p)
    return out


def scrape(
    target: int = DEFAULT_TARGET,
    per_company_cap: int = DEFAULT_PER_COMPANY_CAP,
    out_path: Path = DEFAULT_OUT,
) -> dict:
    """Collect postings until `target` is reached, dedup by id, write JSONL."""
    out_path.parent.mkdir(parents=True, exist_ok=True)
    seen: set[str] = set()
    per_source: dict[str, int] = {}
    per_company: dict[str, int] = {}
    written = 0

    jobs_plan = [("greenhouse", t, fetch_greenhouse) for t in GREENHOUSE_TOKENS]
    jobs_plan += [("lever", t, fetch_lever) for t in LEVER_TOKENS]

    headers = {"User-Agent": USER_AGENT, "Accept": "application/json"}
    with httpx.Client(headers=headers, follow_redirects=True) as client, \
            out_path.open("w", encoding="utf-8") as fh:
        for source, token, fetch in jobs_plan:
            if written >= target:
                break
            postings = fetch(client, token, per_company_cap)
            kept = 0
            for p in postings:
                if p.id in seen:
                    continue
                seen.add(p.id)
                fh.write(p.model_dump_json() + "\n")
                written += 1
                kept += 1
                per_source[source] = per_source.get(source, 0) + 1
                per_company[f"{source}:{token}"] = per_company.get(f"{source}:{token}", 0) + 1
                if written >= target:
                    break
            status = f"{kept:>4} kept" if kept else "  -- (skip/empty)"
            print(f"  {source:<10} {token:<16} {status}   total={written}")
            time.sleep(REQUEST_DELAY_S)

    manifest = {
        "scraped_at": _now(),
        "total_postings": written,
        "per_source": per_source,
        "companies_with_postings": len(per_company),
        "target": target,
        "per_company_cap": per_company_cap,
        "sources": "greenhouse + lever public job-board APIs",
        "output": str(out_path),
    }
    (out_path.parent / "_manifest.json").write_text(
        json.dumps(manifest, indent=2), encoding="utf-8"
    )
    return manifest


def main() -> None:
    parser = argparse.ArgumentParser(description="Collect job postings into data/raw/")
    parser.add_argument("--target", type=int, default=DEFAULT_TARGET)
    parser.add_argument("--per-company-cap", type=int, default=DEFAULT_PER_COMPANY_CAP)
    parser.add_argument("--out", type=Path, default=DEFAULT_OUT)
    args = parser.parse_args()

    print(f"Collecting up to {args.target} postings -> {args.out}\n")
    manifest = scrape(args.target, args.per_company_cap, args.out)
    print("\n=== done ===")
    print(json.dumps(manifest, indent=2))


if __name__ == "__main__":
    main()
