#!/usr/bin/env python3
# -*- coding: utf-8 -*-

"""
Reference Helper Module for MAS Error Detection.

This module extracts References from full-paper blocks and enriches section blocks
with locally cited references. It identifies citations in section text (e.g.,
"LeCun et al., 1998", "Rajabi & Kosecka, 2024", "[3]") and injects the corresponding
reference entries at the end of each section's blocks.

Main Function:
    enrich_section_blocks_with_local_references: Enriches section blocks with
        references cited within that section.

The module supports these citation formats:
    - Author-year format: "LeCun et al., 1998", "Rajabi & Kosecka, 2024"
    - Narrative author-year format: "Malladi et al. (2023)", "Dayi & Chen (2024)"
    - Single-author format: "(Olson, 1965)", "(Meta, 2024)"
    - Numeric format: "[1]", "[23]", etc.

It matches these against reference entries regardless of whether the entry
lists authors "Surname, Initial." (inverted) or "Firstname Surname" (natural
order, e.g. "Nicholas Carlini and David Wagner. Title...").

It matches these citations against the References section and appends matched
entries as a new text block to the section.
"""

from __future__ import annotations
from typing import List, Dict, Tuple
import re
import unicodedata


def _collect_reference_blocks(blocks: List[Dict]) -> List[Dict]:
    """
    Collect References / Reference / Bibliography related text blocks from full-paper blocks.
    Depends on normalize_blocks structure: each block may have 'section' and 'type' fields.
    """
    ref_titles = {"references", "reference", "bibliography"}
    ref_blocks: List[Dict] = []
    for b in blocks:
        sec = (b.get("section") or "").strip().lower()
        if sec in ref_titles and b.get("type") == "text":
            ref_blocks.append(b)
    return ref_blocks


def _build_references_text(ref_blocks: List[Dict]) -> str:
    """Merge reference blocks into a single long string."""
    parts = []
    for b in ref_blocks:
        if b.get("type") == "text":
            t = b.get("text") or ""
            if t.strip():
                parts.append(t.strip())
    # Keep neighbouring PDF blocks distinct: matching below treats blank lines as
    # reference-entry boundaries, and a single newline can merge two entries.
    return "\n\n".join(parts)


def _normalize_for_matching(text: str) -> str:
    normalized = unicodedata.normalize("NFKD", text or "")
    return "".join(ch for ch in normalized if not unicodedata.combining(ch)).casefold()


# Words that can get accidentally captured as a "surname" by the single-author
# pattern because they sit right before a year in constructs like "et al., 2021"
# (-> "al" would otherwise look like a single-author key on its own).
_SURNAME_STOPWORDS = {"al", "et", "pp", "vol", "no", "eq", "fig", "table"}


def _extract_citation_keys_from_section_blocks(
    section_blocks: List[Dict],
) -> Tuple[List[Tuple[str, str]], List[str]]:
    """
    Extract two types of citation keys from current section text:
      - author_year_keys: [("lecun", "1998"), ("rajabi", "2024"), ...]
      - numeric_keys: ["1", "3", "12", ...]  corresponding to [1] [3] [12] format
    """
    text_parts = []
    for b in section_blocks:
        if b.get("type") == "text":
            t = b.get("text") or ""
            if t:
                text_parts.append(t)
    full_text = unicodedata.normalize("NFC", "\n".join(text_parts))

    author_year_keys: List[Tuple[str, str]] = []
    numeric_keys: List[str] = []
    multi_author_spans = []

    # Keep an optional year suffix (e.g., 2024a) and accept Unicode surnames.
    surname_token = r"([^\W\d_][\w'’.-]*)"
    year_token = r"((?:19|20)\d{2}[a-z]?)"
    # Allow "," OR "(" between "al."/the author pair and the year, so narrative
    # citations like "Malladi et al. (2023)" / "Dayi & Chen (2024)" are matched
    # (previously only a comma was accepted here).
    sep_token = r"\s*[,\(]?\s*"

    # ---------- 1) LeCun et al., 1998  /  Malladi et al. (2023) ----------
    pattern_et_al = re.compile(
        rf"\b{surname_token}\s+et\s+al\.?{sep_token}{year_token}",
        re.IGNORECASE,
    )
    for m in pattern_et_al.finditer(full_text):
        multi_author_spans.append(m.span())
        surname = _normalize_for_matching(m.group(1).strip())
        year = m.group(2).strip().casefold()
        key = (surname, year)
        if key not in author_year_keys:
            author_year_keys.append(key)

    # ---------- 2) Rajabi & Kosecka, 2024 / Rajabi and Kosecka, 2024 / Dayi & Chen (2024) ----------
    # Note: Only use the first surname as key (e.g., Rajabi), second surname is only for pattern matching.
    pattern_and = re.compile(
        rf"\b{surname_token}\s*(?:&|and)\s*{surname_token}{sep_token}{year_token}",
        re.IGNORECASE,
    )
    for m in pattern_and.finditer(full_text):
        multi_author_spans.append(m.span())
        surname = _normalize_for_matching(m.group(1).strip())
        year = m.group(3).strip().casefold()
        key = (surname, year)
        if key not in author_year_keys:
            author_year_keys.append(key)

    # ---------- 3) Single-author: (Olson, 1965) / (Meta, 2024) ----------
    # Previously unhandled: citations with no "et al" and no "&"/"and" were
    # never extracted at all.
    pattern_single = re.compile(
        rf"\b{surname_token}\s*[,\(]\s*{year_token}",
        re.IGNORECASE,
    )
    for m in pattern_single.finditer(full_text):
        # Do not reinterpret a coauthor or "al." as a separate citation.
        if any(start <= m.start() < end for start, end in multi_author_spans):
            continue
        surname_raw = m.group(1).strip()
        if surname_raw.casefold().rstrip(".") in _SURNAME_STOPWORDS:
            continue
        surname = _normalize_for_matching(surname_raw)
        year = m.group(2).strip().casefold()
        key = (surname, year)
        if key not in author_year_keys:
            author_year_keys.append(key)

    # ---------- 4) [1] [23] numeric citations ----------
    pattern_num = re.compile(r"\[(\d+)\]")
    for m in pattern_num.finditer(full_text):
        num = m.group(1).lstrip("0") or "0"
        if num not in numeric_keys:
            numeric_keys.append(num)

    return author_year_keys, numeric_keys


def _match_reference_entries(
    ref_text: str,
    author_year_keys: List[Tuple[str, str]],
    numeric_keys: List[str],
) -> List[str]:
    """
    Match reference entries in the full references text based on author_year_keys & numeric_keys.
    Simple approach: split by blank lines, treat each paragraph as a reference entry.
    """
    if not ref_text.strip():
        return []

    # Split by blank lines into individual entries. _build_references_text keeps
    # source blocks separated by blank lines as well.
    paragraphs = re.split(r"\n\s*\n", ref_text.strip())

    # Fallback: if a "paragraph" bundles more than one entry (no blank line was
    # preserved between them in the source), split it further on entry-start
    # boundaries like "Surname, X.". Without this, only the first
    # entry in a merged block is ever matched.
    entry_start = re.compile(
        r"(?:^|\n)(?=(?:\[\d+\]\s*)?[A-Z][A-Za-z'’\-]+,\s+[A-Z]\.)"
    )
    split_paragraphs: List[str] = []
    for para in paragraphs:
        bounds = [0]
        for match in entry_start.finditer(para):
            if match.start() == 0:
                continue
            previous = para[bounds[-1]:match.start()].strip()
            # An author list may wrap at a coauthor's surname. Split only after
            # a completed entry containing a year, retaining the full prefix.
            if previous.endswith(".") and re.search(r"\b(?:19|20)\d{2}[a-z]?\b", previous):
                bounds.append(match.start())
        bounds.append(len(para))
        split_paragraphs.extend(para[a:b].strip() for a, b in zip(bounds, bounds[1:]) if para[a:b].strip())
    paragraphs = split_paragraphs

    used_entries: List[str] = []
    seen_paragraphs = set()

    # Process author-year first
    for surname, year in author_year_keys:
        esc_surname = re.escape(_normalize_for_matching(surname))
        exact_year = re.compile(
            rf"(?<!\d){re.escape(year.casefold())}(?![a-z0-9])",
            re.IGNORECASE,
        )
        # Strict: surname is specifically the FIRST author of the entry,
        # regardless of whether the entry lists authors "Surname, Initial."
        # (inverted) or "Firstname Surname" (natural order, e.g. "Nicholas
        # Carlini and David Wagner. Title..."). Only the first author's name
        # counts here -- a co-author sharing the same surname elsewhere in
        # the entry (e.g. a different "Xu" further down the author list)
        # must NOT match.
        first_author_inverted = re.compile(
            rf"^\s*(?:\[\d+\]\s*)?{esc_surname}(?:\s*[,.]|\s+)",
            re.IGNORECASE,
        )
        first_author_natural = re.compile(
            rf"^\s*(?:\[\d+\]\s*)?(?:(?!and\b|et\b|al\b)(?:[^\W\d_][\w'’\-]*|[^\W\d_]\.)\s+){{1,4}}{esc_surname}\b(?:\s*,|\s+and\b|\s*&|\s*\.)",
            re.IGNORECASE,
        )
        strict_match = None
        for para in paragraphs:
            normalized_entry = _normalize_for_matching(para.replace("\n", " "))
            if not exact_year.search(normalized_entry):
                continue
            if first_author_inverted.search(normalized_entry) or first_author_natural.search(normalized_entry):
                strict_match = para
                break
        chosen = strict_match
        if chosen is not None:
            entry_key = chosen.strip()
            if entry_key not in seen_paragraphs:
                seen_paragraphs.add(entry_key)
                used_entries.append(entry_key)

    # Then process numeric
    for num in numeric_keys:
        pattern_start = re.compile(
            rf"^\s*(?:\[{re.escape(num)}\]|{re.escape(num)}[.)])(?:\s|$)",
            re.MULTILINE,
        )
        for para in paragraphs:
            if pattern_start.search(para):
                entry_key = para.strip()
                if entry_key not in seen_paragraphs:
                    seen_paragraphs.add(entry_key)
                    used_entries.append(entry_key)
                break

    return used_entries


def _next_content_index(all_blocks: List[Dict]) -> int:
    """Assign a larger content_index for new blocks."""
    idxs: List[int] = []
    for b in all_blocks:
        ci = b.get("content_index")
        if isinstance(ci, int):
            idxs.append(ci)
    return (max(idxs) + 1) if idxs else 10_000_000


def enrich_section_blocks_with_local_references(
    section_title: str,
    section_blocks: List[Dict],
    all_blocks: List[Dict],
) -> List[Dict]:
    """
    Based on citations in section text, find corresponding entries in References section,
    and append these entries as a new text block at the end of current section's blocks.

    Args:
        section_title: Name of current section (e.g., "Related Work")
        section_blocks: All blocks of this section (usually result of slice_json_for_task_with_outline)
        all_blocks: Full-paper blocks (result of normalize_blocks)

    Returns:
        Enriched section blocks with local references appended
    """
    # 1) Find references blocks
    ref_blocks = _collect_reference_blocks(all_blocks)
    if not ref_blocks:
        return section_blocks

    ref_text = _build_references_text(ref_blocks)
    if not ref_text.strip():
        return section_blocks

    # 2) Extract citation keys from current section
    author_year_keys, numeric_keys = _extract_citation_keys_from_section_blocks(section_blocks)
    if not author_year_keys and not numeric_keys:
        return section_blocks

    # 3) Match corresponding entries in references text
    used_entries = _match_reference_entries(ref_text, author_year_keys, numeric_keys)
    if not used_entries:
        return section_blocks

    # 4) Construct new block and append to section
    ci = _next_content_index(all_blocks)
    ref_block_text = "References cited in this section:\n\n" + "\n\n".join(used_entries)

    new_block = {
        "type": "text",
        "text": ref_block_text,
        "section": section_title,
        "content_index": ci,
    }
    return section_blocks + [new_block]
