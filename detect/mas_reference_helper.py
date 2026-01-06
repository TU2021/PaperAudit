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

The module supports two types of citation formats:
    - Author-year format: "LeCun et al., 1998", "Rajabi & Kosecka, 2024"
    - Numeric format: "[1]", "[23]", etc.

It matches these citations against the References section and appends matched
entries as a new text block to the section.
"""

from __future__ import annotations
from typing import List, Dict, Tuple
import re


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
    return "\n".join(parts)


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
    full_text = "\n".join(text_parts)

    author_year_keys: List[Tuple[str, str]] = []
    numeric_keys: List[str] = []

    # ---------- 1) LeCun et al., 1998 ----------
    pattern_et_al = re.compile(r"\b([A-Z][A-Za-z\-]+)\s+et al\.,\s*(\d{4})")
    for m in pattern_et_al.finditer(full_text):
        surname = m.group(1).strip().lower()
        year = m.group(2).strip()
        key = (surname, year)
        if key not in author_year_keys:
            author_year_keys.append(key)

    # ---------- 2) Rajabi & Kosecka, 2024 / Rajabi and Kosecka, 2024 ----------
    # Note: Only use the first surname as key (e.g., Rajabi), second surname is only for pattern matching.
    pattern_and = re.compile(
        r"\b([A-Z][A-Za-z\-]+)\s*(?:&|and)\s*[A-Z][A-Za-z\-]+\s*,\s*(\d{4})"
    )
    for m in pattern_and.finditer(full_text):
        surname = m.group(1).strip().lower()
        year = m.group(2).strip()
        key = (surname, year)
        if key not in author_year_keys:
            author_year_keys.append(key)

    # ---------- 3) [1] [23] numeric citations ----------
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

    # Split by blank lines into individual entries
    paragraphs = re.split(r"\n\s*\n", ref_text.strip())
    used_entries: List[str] = []
    seen = set()

    # Process author-year first
    for surname, year in author_year_keys:
        for para in paragraphs:
            para_norm = para.replace("\n", " ")
            if surname in para_norm.lower() and year in para_norm:
                key = ("ay", surname, year, para.strip())
                if key not in seen:
                    seen.add(key)
                    used_entries.append(para.strip())
                break  # One (surname, year) needs only one match

    # Then process numeric
    for num in numeric_keys:
        pattern_start = re.compile(rf"^\s*\[{re.escape(num)}\]", re.MULTILINE)
        for para in paragraphs:
            if pattern_start.search(para):
                key = ("num", num, para.strip())
                if key not in seen:
                    seen.add(key)
                    used_entries.append(para.strip())
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
