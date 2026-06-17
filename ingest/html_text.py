"""Convert posting HTML to clean plain text using only the stdlib.

Job-board APIs return descriptions as HTML (sometimes entity-encoded). We keep
the readable text and insert newlines around block-level elements so section
structure survives for later section-based chunking.
"""

from __future__ import annotations

import html
import re
from html.parser import HTMLParser

# Tags after which a line break improves readability / preserves structure.
_BLOCK_TAGS = {
    "p", "br", "li", "ul", "ol", "div", "tr", "table",
    "h1", "h2", "h3", "h4", "h5", "h6", "section", "header", "footer",
}
_SKIP_CONTENT = {"script", "style"}
_WS = re.compile(r"[ \t\f\v]+")
_BLANKS = re.compile(r"\n{3,}")


class _TextExtractor(HTMLParser):
    def __init__(self) -> None:
        super().__init__(convert_charrefs=True)
        self._parts: list[str] = []
        self._skip_depth = 0

    def handle_starttag(self, tag: str, attrs: object) -> None:
        if tag in _SKIP_CONTENT:
            self._skip_depth += 1
        elif tag in _BLOCK_TAGS:
            self._parts.append("\n")

    def handle_endtag(self, tag: str) -> None:
        if tag in _SKIP_CONTENT and self._skip_depth:
            self._skip_depth -= 1
        elif tag in _BLOCK_TAGS:
            self._parts.append("\n")

    def handle_data(self, data: str) -> None:
        if not self._skip_depth:
            self._parts.append(data)

    def text(self) -> str:
        return "".join(self._parts)


def strip_html(raw: str | None) -> str:
    """Return clean plain text from an HTML (or entity-encoded HTML) string."""
    if not raw:
        return ""
    # Some boards double-encode (e.g. "&lt;p&gt;"); unescape so tags parse.
    unescaped = html.unescape(raw)
    parser = _TextExtractor()
    parser.feed(unescaped)
    text = parser.text()
    # Normalize whitespace: collapse spaces, trim each line, cap blank runs.
    lines = [_WS.sub(" ", line).strip() for line in text.splitlines()]
    text = "\n".join(lines)
    text = _BLANKS.sub("\n\n", text)
    return text.strip()
