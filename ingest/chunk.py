"""Split postings into section-tagged chunks for retrieval.

Job descriptions are loosely structured by headings (Responsibilities,
Requirements, Qualifications, ...). We detect those headings, group content
under a canonical section label, and sub-split long blocks so each chunk is a
focused, embeddable unit.
"""

from __future__ import annotations

import json
from collections.abc import Iterator
from pathlib import Path

from ingest.models import Chunk, RawPosting

MAX_CHARS = 1500   # target upper bound per chunk
MIN_CHARS = 25     # drop trivial fragments

# Canonical section -> heading keywords (matched case-insensitively at line start).
_HEADERS: dict[str, tuple[str, ...]] = {
    "responsibilities": (
        "responsibilities", "what you'll do", "what you will do", "the role",
        "your impact", "day-to-day", "day to day", "what you'll be doing",
        "about the role", "in this role", "what you will be doing", "the opportunity",
        "role overview", "what you'll work on",
    ),
    "requirements": (
        "requirements", "what you'll need", "what we're looking for", "must have",
        "must-have", "basic qualifications", "minimum qualifications", "required",
        "skills and experience", "skills & experience", "you have", "you should have",
    ),
    "qualifications": (
        "qualifications", "preferred qualifications", "nice to have", "nice-to-have",
        "bonus", "about you", "who you are", "what makes you", "preferred",
    ),
}


def classify_heading(line: str) -> str | None:
    """Return a canonical section label if `line` looks like a section heading."""
    stripped = line.strip()
    if not stripped or len(stripped) > 80:
        return None
    norm = stripped.lower().rstrip(":").strip()
    for label, keywords in _HEADERS.items():
        if any(norm == kw or norm.startswith(kw) for kw in keywords):
            return label
    # Generic short heading ending with a colon -> bucket as "other".
    if stripped.endswith(":") and len(stripped) <= 60:
        return "other"
    return None


def split_sections(text: str) -> list[tuple[str, str]]:
    """Group lines into (section_label, block_text) in document order."""
    sections: list[tuple[str, list[str]]] = [("summary", [])]
    for line in text.splitlines():
        label = classify_heading(line)
        if label is not None:
            sections.append((label, []))
        else:
            sections[-1][1].append(line)
    out: list[tuple[str, str]] = []
    for label, lines in sections:
        block = "\n".join(lines).strip()
        if len(block) >= MIN_CHARS:
            out.append((label, block))
    return out


def _units(block: str) -> list[str]:
    """Break a block into atomic units no larger than MAX_CHARS each.

    Prefer line boundaries, then sentences, then a hard char window as a last
    resort (e.g. a single unbroken line longer than MAX_CHARS).
    """
    units: list[str] = []
    for line in block.split("\n"):
        if len(line) <= MAX_CHARS:
            units.append(line)
            continue
        for part in line.replace(". ", ".\n").split("\n"):
            if len(part) > MAX_CHARS:  # still too big -> hard window
                for i in range(0, len(part), MAX_CHARS):
                    units.append(part[i : i + MAX_CHARS])
            else:
                units.append(part)
    return units


def _split_long_block(block: str) -> list[str]:
    """Greedily pack a block's units into <= MAX_CHARS pieces."""
    if len(block) <= MAX_CHARS:
        return [block]
    pieces: list[str] = []
    current = ""
    for unit in _units(block):
        if current and len(current) + len(unit) + 1 > MAX_CHARS:
            pieces.append(current.strip())
            current = unit
        else:
            current = f"{current}\n{unit}" if current else unit
    if current.strip():
        pieces.append(current.strip())
    return [p for p in pieces if len(p) >= MIN_CHARS] or [block[:MAX_CHARS]]


def chunk_posting(posting: RawPosting) -> list[Chunk]:
    """Produce ordered, section-tagged chunks for one posting."""
    chunks: list[Chunk] = []
    idx = 0
    for label, block in split_sections(posting.description):
        for piece in _split_long_block(block):
            chunks.append(
                Chunk(
                    id=f"{posting.id}#{idx}",
                    posting_id=posting.id,
                    source=posting.source,
                    company=posting.company,
                    title=posting.title,
                    location=posting.location,
                    url=posting.url,
                    section=label,
                    chunk_index=idx,
                    content=piece,
                )
            )
            idx += 1
    return chunks


def iter_postings(path: Path) -> Iterator[RawPosting]:
    """Yield RawPosting from a JSONL file."""
    with path.open(encoding="utf-8") as fh:
        for line in fh:
            line = line.strip()
            if line:
                yield RawPosting.model_validate_json(line)


# Lower number = higher retrieval value when we must cap chunks per posting.
_SECTION_PRIORITY = {
    "requirements": 0, "qualifications": 1, "responsibilities": 2,
    "summary": 3, "other": 4,
}


def prioritize_chunks(chunks: list[Chunk], max_per_posting: int) -> list[Chunk]:
    """Keep the most retrieval-relevant chunks of one posting, capped."""
    if max_per_posting <= 0 or len(chunks) <= max_per_posting:
        return chunks
    ranked = sorted(
        chunks,
        key=lambda c: (_SECTION_PRIORITY.get(c.section, 9), c.chunk_index),
    )
    return ranked[:max_per_posting]


def interleave_by_company(postings: list[RawPosting]) -> list[RawPosting]:
    """Round-robin postings across companies for corpus diversity."""
    buckets: dict[str, list[RawPosting]] = {}
    for p in postings:
        buckets.setdefault(p.company, []).append(p)
    queues = list(buckets.values())
    out: list[RawPosting] = []
    i = 0
    while queues:
        q = queues[i % len(queues)]
        out.append(q.pop(0))
        if not q:
            queues.remove(q)
        else:
            i += 1
    return out
