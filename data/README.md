# Data

Job-postings corpus for retrieval. Target: ~3–5K postings with title, company, description.

## Strategy: scrape
The corpus will be **scraped** from live job postings (decided in Phase 0).

## Layout
- `raw/` — raw scraped output. **Gitignored** (licensing unclear; never commit).
- (processed/chunked artifacts land here in Phase 1 ingestion — paths TBD.)

## Source & licensing (fill in when scraper is built)
- Source site(s): _TBD_
- Date scraped: _TBD_
- Terms of use / robots.txt reviewed: _TBD_
- Notes on redistribution: do not redistribute raw postings; keep `raw/` local only.
