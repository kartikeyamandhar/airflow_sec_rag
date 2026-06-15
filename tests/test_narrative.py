"""HTML extraction, section segmentation, and chunking."""

import pytest

from app.narrative.chunk import chunk_sections, estimate_tokens
from app.narrative.extract import Section, html_to_text, segment_sections

_ACCESSION = "0000320193-23-000106"


def test_html_to_text_strips_tags_scripts_and_nbsp() -> None:
    html = (
        b"<html><head><style>.x{}</style></head><body>"
        b"<p>Hello&nbsp;world</p><script>bad()</script><p>Second line</p>"
        b"</body></html>"
    )
    text = html_to_text(html)
    assert "Hello world" in text
    assert "Second line" in text
    assert "bad()" not in text
    assert ".x{}" not in text


def test_html_to_text_size_guard() -> None:
    with pytest.raises(ValueError):
        html_to_text(b"x" * 100, max_bytes=10)


def test_segment_sections_prefers_real_body_over_toc() -> None:
    toc = "Item 1. Business\nItem 1A. Risk Factors\nItem 7. MD&A\n"
    body_1 = "Item 1. Business\n" + ("We make products. " * 30) + "\n"
    body_1a = "Item 1A. Risk Factors\n" + ("Markets are risky. " * 30) + "\n"
    body_7 = "Item 7. MD&A\n" + ("Revenue grew. " * 30) + "\n"
    sections = segment_sections(toc + body_1 + body_1a + body_7, min_section_chars=100)
    assert [s.name for s in sections] == ["Item 1", "Item 1A", "Item 7"]
    by_name = {s.name: s.text for s in sections}
    assert "We make products." in by_name["Item 1"]


def test_segment_sections_drops_toc_and_preamble() -> None:
    # Reproduces the Alphabet bug: a TOC ("Item 6" / "Exhibits" / page number)
    # precedes a forward-looking-statements preamble that name-drops the query topic,
    # then the real risk factors follow. The preamble must not be labeled "Item 6",
    # and the real Item 1A must be detected with its own content.
    text = (
        "Table of Contents\n"
        "Item 1. Business 3\n"
        "Item 1A. Risk Factors 7\n"
        "Item 6\n"
        "Exhibits\n"
        "54\n"
        "Note About Forward-Looking Statements\n"
        + ("We mention artificial intelligence repeatedly here. " * 20)
        + "\n"
        "ITEM 1A.\n"
        "RISK FACTORS\n" + ("Our AI products carry regulatory and reputational risk. " * 20) + "\n"
        "ITEM 7.\n"
        "MD&A\n" + ("Revenue grew this year. " * 20) + "\n"
    )
    sections = segment_sections(text, min_section_chars=100)
    names = [s.name for s in sections]
    assert "Item 6" not in names
    assert "Item 1A" in names
    by_name = {s.name: s.text for s in sections}
    assert "regulatory and reputational risk" in by_name["Item 1A"]
    # The preamble (with the query topic) did not leak into the real section.
    assert "artificial intelligence" not in by_name["Item 1A"]


def test_segment_sections_keeps_real_heading_with_year_nearby() -> None:
    # A 4-digit year near a heading is not a page number, so the section is kept.
    text = (
        "ITEM 7.\nManagement Discussion\n2024\n" + ("Net sales rose during the year. " * 30) + "\n"
    )
    sections = segment_sections(text, min_section_chars=100)
    assert [s.name for s in sections] == ["Item 7"]


def test_segment_sections_fallback_when_no_headings() -> None:
    sections = segment_sections("Just prose with no item headings whatsoever.")
    assert len(sections) == 1
    assert sections[0].name == "BODY"


def test_segment_sections_empty() -> None:
    assert segment_sections("   ") == []


def test_chunking_offsets_overlap_and_parentage() -> None:
    section = Section("Item 1", " ".join(f"word{i}" for i in range(1000)))
    chunks = chunk_sections(
        [section],
        accession=_ACCESSION,
        cik=320193,
        ticker="AAPL",
        form="10-K",
        child_tokens=100,
        overlap_tokens=20,
    )
    parents = [c for c in chunks if c.kind == "parent"]
    children = [c for c in chunks if c.kind == "child"]
    assert len(parents) == 1
    assert len(children) > 1

    parent = parents[0]
    for child in children:
        # Citation invariant: the parent slice equals the child text exactly.
        assert parent.text[child.char_start : child.char_end] == child.text
        assert child.parent_index == parent.chunk_index
    # Adjacent children overlap.
    assert children[1].char_start < children[0].char_end


def test_estimate_tokens() -> None:
    assert estimate_tokens("a b c") == 3
    assert estimate_tokens("") == 0
