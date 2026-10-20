"""Guard the JOSS paper: structure, JOSS word-count range, citations resolve,
library-vs-users separation, zero archive leaks, figure assets, section coverage,
valid BibTeX, and formatting integrity (12 automated checks).
"""
import re
from pathlib import Path
import yaml
import pytest

PAPER = Path(__file__).resolve().parents[1] / "paper" / "paper.md"
BIB = Path(__file__).resolve().parents[1] / "paper" / "paper.bib"


def _frontmatter_and_body():
    text = PAPER.read_text(encoding="utf-8")
    assert text.startswith("---"), "paper.md must start with YAML frontmatter"
    _, fm, body = text.split("---", 2)
    return yaml.safe_load(fm), body


# Check 1: Frontmatter validity
def test_frontmatter_valid():
    """Verify that YAML frontmatter contains required JOSS metadata."""
    fm, _ = _frontmatter_and_body()
    assert fm["title"] and fm["bibliography"] == "paper.bib"
    authors = fm["authors"]
    assert authors and any(a.get("corresponding") for a in authors)
    assert fm["affiliations"]


# JOSS asks for 750-1750 words (joss.readthedocs.io/en/latest/paper.html).
JOSS_MIN_WORDS, JOSS_MAX_WORDS = 750, 1750


# Check 2: Word count in JOSS bounds
def test_word_count_in_joss_range():
    """Verify that body word count is strictly within [750, 1750]."""
    _, body = _frontmatter_and_body()
    body_wo_refs = re.split(r"#\s*References", body)[0]
    body_wo_fig = re.sub(r"!\[.*?\]\(.*?\)\{.*?\}", "", body_wo_refs, flags=re.S)
    body_wo_notes = re.sub(r"<!--.*?-->", "", body_wo_fig, flags=re.S)
    words = len(re.findall(r"[A-Za-z0-9'-]+", body_wo_notes))
    assert JOSS_MIN_WORDS <= words <= JOSS_MAX_WORDS, (
        f"JOSS body word count {words} outside "
        f"{JOSS_MIN_WORDS}-{JOSS_MAX_WORDS}"
    )


# Check 3: Citation resolution
def test_citations_resolve():
    """Verify that all cited @keys exist in paper.bib."""
    _, body = _frontmatter_and_body()
    cited = set(re.findall(r"@([A-Za-z0-9_]+)", body))
    bibkeys = set(re.findall(r"@\w+\{([^,]+),", BIB.read_text(encoding="utf-8")))
    assert cited, "paper cites no references"
    assert cited <= bibkeys, f"cited keys missing from bib: {cited - bibkeys}"


# Check 4: Library vs downstream users separation
def test_library_users_separation():
    """Ensure downstream user research papers are not referenced as library parts."""
    _, body = _frontmatter_and_body()
    low = body.lower()
    for forbidden in ("regime uncertainty", "regime_uncertainty", "equipment paper"):
        assert forbidden not in low, f"paper references a downstream user paper: {forbidden!r}"


# Check 5: Zero archive leaks
def test_zero_archive_leaks():
    """Ensure no private research archive paths or directories leak into public paper."""
    text = PAPER.read_text(encoding="utf-8").lower()
    for forbidden in ("puremacro-archive", "techreport/", "studies/sw07/", "projects/fertility/"):
        assert forbidden not in text, f"paper contains private archive leak: {forbidden!r}"


# Check 6: Required JOSS sections
def test_required_sections_present():
    """Verify presence of all mandatory JOSS headings."""
    _, body = _frontmatter_and_body()
    required_sections = [
        "Summary",
        "Statement of need",
        "State of the field",
        "Software design",
        "Research impact statement",
        "AI usage disclosure",
        "References",
    ]
    for sec in required_sections:
        assert re.search(rf"#\s+{sec}", body, re.IGNORECASE), f"Missing section #{sec}"


# Check 7: Figure assets existence
def test_figure_files_exist():
    """Verify that all figure files referenced in markdown exist and are non-empty."""
    _, body = _frontmatter_and_body()
    figs = re.findall(r"!\[.*?\]\((.*?)\)", body)
    assert len(figs) >= 3, f"Expected at least 3 figures, found {len(figs)}"
    for fig_rel in figs:
        fig_path = PAPER.parent / fig_rel
        assert fig_path.is_file(), f"Missing figure file: {fig_path}"
        assert fig_path.stat().st_size > 1000, f"Figure file is too small: {fig_path}"


# Check 8: Figure captions and LaTeX labels
def test_figure_captions_and_labels():
    """Verify that every figure has an informative caption and a LaTeX label."""
    _, body = _frontmatter_and_body()
    matches = re.findall(r"!\[(.*?)\]\((.*?)\)", body)
    for caption, _ in matches:
        assert len(caption.strip()) > 10, f"Caption too short: {caption!r}"
        assert r"\label{fig:" in caption, f"Missing LaTeX label in caption: {caption!r}"


# Check 9: Relative markdown links resolution
def test_relative_markdown_links_resolve():
    """Verify that all relative local links in paper.md resolve to existing files."""
    _, body = _frontmatter_and_body()
    links = re.findall(r"\[.*?\]\((?!http|https|#)(.*?)\)", body)
    for link in links:
        clean_link = link.split("{")[0].strip()
        link_path = PAPER.parent / clean_link
        assert link_path.exists(), f"Broken relative link: {link_path}"


# Check 10: BibTeX syntax and entry sufficiency
def test_bibtex_syntax_and_entries_valid():
    """Verify that paper.bib is non-empty, well-formed, and contains extensive citations."""
    assert BIB.is_file(), f"Missing bibliography file: {BIB}"
    bib_text = BIB.read_text(encoding="utf-8")
    entries = re.findall(r"@\w+\{([^,]+),", bib_text)
    assert len(entries) >= 25, f"Expected >=25 bibliography entries, found {len(entries)}"
    assert bib_text.count("{") == bib_text.count("}"), "Unbalanced braces in paper.bib"


# Check 11: Balanced mathematical delimiters
def test_math_delimiters_balanced():
    """Verify that LaTeX math delimiters ($) are balanced within paragraphs."""
    _, body = _frontmatter_and_body()
    paragraphs = [p for p in body.split("\n\n") if p.strip()]
    for p in paragraphs:
        p_wo_code = re.sub(r"`.*?`", "", p)
        assert p_wo_code.count("$") % 2 == 0, f"Unbalanced $ in paragraph:\n{p[:100]}..."


# Check 12: No unresolved TODOs or FIXMEs
def test_no_unresolved_todos_or_fixmes():
    """Ensure no lingering TODO or FIXME markers remain in the public manuscript."""
    _, body = _frontmatter_and_body()
    body_wo_comments = re.sub(r"<!--.*?-->", "", body, flags=re.S)
    assert "TODO" not in body_wo_comments, "Found unresolved TODO in paper body"
    assert "FIXME" not in body_wo_comments, "Found unresolved FIXME in paper body"
