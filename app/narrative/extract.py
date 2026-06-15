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
# A standalone page reference (1 to 3 digits on its own line). Years and large
# figures are 4+ digits, so they are not mistaken for page numbers.
_PAGE_NUMBER = re.compile(r"^\d{1,3}$")
# How many following non-empty lines to scan for a page number when classifying a
# heading as a table-of-contents entry.
_TOC_LOOKAHEAD = 2


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


def _next_nonempty(lines: list[str], start: int, count: int) -> list[str]:
    """Return up to ``count`` stripped non-empty lines at or after ``start``."""
    out: list[str] = []
    for line in lines[start:]:
        stripped = line.strip()
        if stripped:
            out.append(stripped)
            if len(out) >= count:
                break
    return out


def _is_toc_heading(lines: list[str], line_idx: int, match: re.Match[str]) -> bool:
    """True if an item heading is a table-of-contents entry, not a real section.

    Two TOC shapes are recognized: the page number trails the title on the heading
    line ("Item 1A. Risk Factors 47"), or it sits on a nearby line when the filing
    puts each token on its own line ("Item 6" / "Exhibits" / "54"). Real section
    headings are followed by prose, not a page number, so they are kept.
    """
    remainder = lines[line_idx].strip()[match.end() :].strip()
    if remainder and re.search(r"\b\d{1,3}$", remainder):
        return True
    following = _next_nonempty(lines, line_idx + 1, _TOC_LOOKAHEAD)
    return any(_PAGE_NUMBER.fullmatch(line) for line in following)


def segment_sections(text: str, *, min_section_chars: int = 200) -> list[Section]:
    """Segment normalized text into SEC item sections.

    Two table-of-contents defenses run in order. First, headings that are TOC
    entries (a page number trails the title, or sits on a nearby line) are dropped,
    so the forward-looking-statements preamble that follows the TOC can never be
    mislabeled as a real item. Second, among the remaining real headings an item
    name may still appear more than once, so we keep the occurrence with the longest
    body and drop sections below ``min_section_chars``. If nothing qualifies, return
    one fallback section so no content is ever lost.
    """
    if not text.strip():
        return []
    lines = text.split("\n")
    headings: list[tuple[int, str, bool]] = []
    for i, line in enumerate(lines):
        match = _ITEM_HEADING.match(line.strip())
        if match is None:
            continue
        name = f"Item {match.group(1).upper()}"
        headings.append((i, name, _is_toc_heading(lines, i, match)))
    if not headings:
        return [Section(_DEFAULT_SECTION, text)]

    real: list[tuple[int, str, str]] = []
    for idx, (line_idx, name, is_toc) in enumerate(headings):
        if is_toc:
            continue
        end_line = headings[idx + 1][0] if idx + 1 < len(headings) else len(lines)
        body = "\n".join(lines[line_idx:end_line]).strip()
        real.append((line_idx, name, body))

    best: dict[str, tuple[int, str]] = {}
    for line_idx, name, body in real:
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
