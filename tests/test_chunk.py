"""Unit tests for section-based chunking — no network/DB."""

from ingest.chunk import (
    MAX_CHARS,
    chunk_posting,
    classify_heading,
    interleave_by_company,
    prioritize_chunks,
    split_sections,
)
from ingest.models import Chunk, RawPosting

SAMPLE = """About the company
We build payments infrastructure.

Responsibilities:
Design and ship retrieval pipelines.
Own services end to end.

Requirements:
5+ years of backend experience.
Strong Python and SQL.

Preferred Qualifications:
Experience with pgvector or FAISS.
"""


def _posting(desc: str) -> RawPosting:
    return RawPosting(
        id="greenhouse:1", source="greenhouse", company="acme",
        title="Backend Engineer", location="Remote", description=desc,
        url="https://x/1", scraped_at="2026-06-17T00:00:00+00:00",
    )


def test_classify_heading():
    assert classify_heading("Responsibilities:") == "responsibilities"
    assert classify_heading("What you'll do") == "responsibilities"
    assert classify_heading("Requirements") == "requirements"
    assert classify_heading("Preferred Qualifications:") == "qualifications"
    assert classify_heading("Perks:") == "other"      # generic colon heading
    assert classify_heading("We build payments infrastructure.") is None
    assert classify_heading("") is None


def test_split_sections_orders_and_labels():
    sections = split_sections(SAMPLE)
    labels = [label for label, _ in sections]
    assert labels[0] == "summary"
    assert "responsibilities" in labels
    assert "requirements" in labels
    assert "qualifications" in labels
    reqs = next(text for label, text in sections if label == "requirements")
    assert "5+ years" in reqs


def test_chunk_posting_ids_and_sections():
    chunks = chunk_posting(_posting(SAMPLE))
    assert chunks, "expected at least one chunk"
    assert [c.chunk_index for c in chunks] == list(range(len(chunks)))
    assert all(c.id == f"greenhouse:1#{c.chunk_index}" for c in chunks)
    assert {"responsibilities", "requirements", "qualifications"} <= {c.section for c in chunks}


def test_long_block_is_subsplit():
    big = "Responsibilities:\n" + ("Build great systems. " * 400)
    chunks = chunk_posting(_posting(big))
    assert len(chunks) > 1
    assert all(len(c.content) <= MAX_CHARS for c in chunks)


def _chunk(section: str, idx: int) -> Chunk:
    return Chunk(
        id=f"x#{idx}", posting_id="x", source="s", company="c", title="t",
        section=section, chunk_index=idx, content="...",
    )


def test_prioritize_chunks_keeps_high_signal_sections():
    chunks = [
        _chunk("other", 0), _chunk("summary", 1),
        _chunk("requirements", 2), _chunk("responsibilities", 3),
    ]
    kept = prioritize_chunks(chunks, max_per_posting=2)
    sections = {c.section for c in kept}
    assert sections == {"requirements", "responsibilities"}


def test_interleave_by_company_alternates():
    def p(company, i):
        return RawPosting(
            id=f"{company}:{i}", source="greenhouse", company=company,
            title="t", description="d" * 30, url="u",
            scraped_at="2026-06-17T00:00:00+00:00",
        )
    postings = [p("a", 0), p("a", 1), p("b", 0), p("b", 1)]
    order = [x.company for x in interleave_by_company(postings)]
    assert order[0] != order[1]  # diversified, not all "a" first
