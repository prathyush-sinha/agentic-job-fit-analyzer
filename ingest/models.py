"""Normalized schema for a raw job posting.

One Pydantic model so every source (Greenhouse, Lever, ...) lands in the same
shape before ingestion/chunking in Phase 1.
"""

from __future__ import annotations

from pydantic import BaseModel, Field


class RawPosting(BaseModel):
    id: str = Field(..., description="Stable id: '<source>:<external_id>'")
    source: str = Field(..., description="Origin board, e.g. 'greenhouse' or 'lever'")
    company: str
    title: str
    location: str | None = None
    description: str = Field(..., description="Plain-text job description (HTML stripped)")
    url: str
    scraped_at: str = Field(..., description="ISO-8601 UTC timestamp")


class Chunk(BaseModel):
    """A retrievable slice of a posting, tagged with its section."""

    id: str = Field(..., description="'<posting_id>#<chunk_index>'")
    posting_id: str
    source: str
    company: str
    title: str
    location: str | None = None
    url: str = ""
    section: str = Field(..., description="responsibilities|requirements|qualifications|summary|other")
    chunk_index: int
    content: str
