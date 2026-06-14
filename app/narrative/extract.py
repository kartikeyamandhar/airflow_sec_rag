"""Extract normalized text and SEC item sections from filing HTML.

Uses selectolax (lexbor), which parses HTML without DTD or external-entity
resolution, so untrusted filing HTML has no XXE surface. Section detection is
tolerant of heading variants and, critically, never drops content: if no item
headings are found it returns a single fallback section.
"""

from __future__ import annotations

import re
from dataclasses import dataclass

from selectolax.parser import HTMLParser

_ITEM_HEADING = re.compile(r"^item\s+(\d{1,2}[a-z]?)\b", re.IGNORECASE)
_DEFAULT_SECTION = "BODY"


@dataclass(frozen=True)
class Section:
    """A named span of normalized filing text."""

    name: str
    text: str


def html_to_text(html: bytes, *, max_bytes: int = 50_000_000) -> str:
    """Extract normalized plain text from HTML bytes.

    Raises ``ValueError`` if the input exceeds ``max_bytes`` (OOM guard).
    """
    if len(html) > max_bytes:
        raise ValueError(f"document exceeds max_bytes ({len(html)} > {max_bytes})")
    tree = HTMLParser(html)
    for node in tree.css("script, style"):
        node.decompose()
    root = tree.body if tree.body is not None else tree.root
    text = root.text(separator="\n", deep=True) if root is not None else ""
    return _normalize_whitespace(text)


def _normalize_whitespace(text: str) -> str:
    text = text.replace("\xa0", " ")
    lines = [re.sub(r"[ \t]+", " ", line).strip() for line in text.splitlines()]
    out: list[str] = []
    previous_blank = False
    for line in lines:
        if line:
            out.append(line)
            previous_blank = False
        elif not previous_blank:
            out.append("")
            previous_blank = True
    return "\n".join(out).strip()


def segment_sections(text: str, *, min_section_chars: int = 200) -> list[Section]:
    """Segment normalized text into SEC item sections.

    Heuristic for the table-of-contents problem: an item name often appears twice
    (once in the TOC, once as the real heading). We keep, per item name, the
    occurrence with the longest body, then drop sections below ``min_section_chars``
    as likely TOC remnants. If nothing qualifies, return one fallback section so no
    content is ever lost.
    """
    if not text.strip():
        return []
    lines = text.split("\n")
    headings: list[tuple[int, str]] = [
        (i, f"Item {match.group(1).upper()}")
        for i, line in enumerate(lines)
        if (match := _ITEM_HEADING.match(line.strip()))
    ]
    if not headings:
        return [Section(_DEFAULT_SECTION, text)]

    candidates: list[tuple[int, str, str]] = []
    for idx, (line_idx, name) in enumerate(headings):
        end_line = headings[idx + 1][0] if idx + 1 < len(headings) else len(lines)
        body = "\n".join(lines[line_idx:end_line]).strip()
        candidates.append((line_idx, name, body))

    best: dict[str, tuple[int, str]] = {}
    for line_idx, name, body in candidates:
        if name not in best or len(body) > len(best[name][1]):
            best[name] = (line_idx, body)

    kept = [
        (line_idx, name, body)
        for name, (line_idx, body) in best.items()
        if len(body) >= min_section_chars
    ]
    kept.sort(key=lambda item: item[0])
    sections = [Section(name, body) for _, name, body in kept]
    return sections if sections else [Section(_DEFAULT_SECTION, text)]
