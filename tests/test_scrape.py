"""Unit tests for scraper pure functions — no network."""

from ingest.html_text import strip_html
from ingest.scrape import normalize_greenhouse, normalize_lever


def test_strip_html_basic_blocks():
    html = "<p>Build <b>RAG</b> pipelines.</p><ul><li>Python</li><li>SQL</li></ul>"
    text = strip_html(html)
    assert "Build RAG pipelines." in text
    assert "Python" in text and "SQL" in text
    assert "<" not in text and ">" not in text


def test_strip_html_entity_encoded():
    # Greenhouse sometimes double-encodes HTML.
    encoded = "&lt;p&gt;Senior &amp; Staff roles&lt;/p&gt;"
    assert strip_html(encoded) == "Senior & Staff roles"


def test_strip_html_skips_script_and_handles_empty():
    assert strip_html("") == ""
    assert strip_html(None) == ""
    assert "alert" not in strip_html("<p>Hi</p><script>alert(1)</script>")


def test_normalize_greenhouse_ok():
    job = {
        "id": 123,
        "title": "ML Engineer",
        "location": {"name": "Remote"},
        "content": "&lt;p&gt;Train models.&lt;/p&gt;",
        "absolute_url": "https://example.com/123",
    }
    p = normalize_greenhouse("acme", job)
    assert p is not None
    assert p.id == "greenhouse:123"
    assert p.source == "greenhouse"
    assert p.company == "acme"
    assert p.title == "ML Engineer"
    assert p.location == "Remote"
    assert p.description == "Train models."
    assert p.scraped_at  # populated


def test_normalize_greenhouse_missing_field_returns_none():
    assert normalize_greenhouse("acme", {"id": 1}) is None  # no title


def test_normalize_lever_prefers_plain_text():
    job = {
        "id": "abc",
        "text": "Data Scientist",
        "categories": {"location": "NYC"},
        "descriptionPlain": "Do analysis.",
        "description": "<p>ignored when plain present</p>",
        "hostedUrl": "https://jobs.lever.co/acme/abc",
    }
    p = normalize_lever("acme", job)
    assert p is not None
    assert p.id == "lever:abc"
    assert p.title == "Data Scientist"
    assert p.location == "NYC"
    assert p.description == "Do analysis."
