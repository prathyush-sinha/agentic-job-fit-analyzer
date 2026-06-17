# Data

Job-postings corpus for retrieval. Target: ~3–5K postings with title, company, description.

## Strategy: public job-board APIs (not site scraping)
Postings are collected from **public Applicant-Tracking-System APIs** — Greenhouse and
Lever — which are explicitly designed for programmatic consumption (companies use them to
embed their own listings). We deliberately avoid scraping sites like LinkedIn/Indeed whose
terms of service prohibit it.

Collector: `python -m ingest.scrape` (see `ingest/`). Tokens live in `ingest/sources.py`.

## Layout
- `raw/postings.jsonl` — one `RawPosting` JSON object per line. **Gitignored.**
- `raw/_manifest.json` — counts per source, timestamp, run params. **Gitignored.**
- (processed/chunked artifacts land here in Phase 1 ingestion — paths TBD.)

## Record schema (`ingest/models.py: RawPosting`)
`id` (`<source>:<external_id>`), `source`, `company`, `title`, `location`, `description`
(plain text, HTML stripped), `url`, `scraped_at` (ISO-8601 UTC).

## Source & licensing
- **Sources:** Greenhouse public board API
  (`boards-api.greenhouse.io/v1/boards/<token>/jobs`) and Lever
  (`api.lever.co/v0/postings/<token>`).
- **Access:** public, unauthenticated, JSON; intended for embedding job boards.
- **Last run:** see `raw/_manifest.json` (`scraped_at`). First corpus: 4,500 postings from
  29 companies (Greenhouse).
- **Redistribution:** postings are owned by the respective companies — **do not commit or
  redistribute** the raw corpus. `raw/` is gitignored; keep it local only.
