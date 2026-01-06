#!/usr/bin/env python3
# -*- coding: utf-8 -*-

"""
Post-Clean Sections Tool

This script performs rule-based post-processing on the results from batch_label_sections.py:

Features:
    --strip-header-footer    Remove headers and footers (keeps first occurrence)
    --drop-checklist         Remove Checklist sections

Input format:
    paper_parse.json (contains content + section_labels)

Output format:
    paper_final.json

Usage:
    python filter_parse.py [OPTIONS]

Examples:
    # Basic usage with default settings
    python filter_parse.py --root-dir /path/to/papers

    # Enable header/footer stripping and checklist dropping
    python filter_parse.py --root-dir /path/to/papers --strip-header-footer --drop-checklist

    # Custom input/output names
    python filter_parse.py --root-dir /path/to/papers --input-name paper_parse_add_section.json --output-name paper_final.json

    # Show help message
    python filter_parse.py --help

Arguments:
    --root-dir               Root directory, recursively search for input JSON files (required)
    --input-name             Input JSON filename (default: paper_parse_add_section.json)
    --output-name            Output JSON filename (default: paper_final.json)
    --strip-header-footer    Remove headers and footers (disabled by default)
    --drop-checklist         Remove Checklist sections (disabled by default)
"""

from pathlib import Path
import json
import re
import argparse
from typing import List, Dict, Any
from collections import Counter
from tqdm import tqdm
from detect.utils import load_json, save_json


def find_jsons(root: Path, name: str) -> List[Path]:
    return sorted([p for p in root.rglob(name) if p.is_file()])


# ======= Rule: Header/Footer Cleaning ======= #

def _normalize_line(line: str) -> str:
    """
    Normalization rules for determining "if it's the same line":
    - Remove leading/trailing spaces
    - Remove leading Markdown heading symbols (#, *, etc.) to ensure "# VersaPRM ..." and "VersaPRM ..." are grouped together
    - Merge multiple spaces
    - Remove common leading/trailing symbols, convert to lowercase
    - Lines that are too short (e.g., single page numbers) are considered invalid
    """
    if not line:
        return ""
    s = line.strip()
    # Remove markdown heading/list symbol prefixes: #, ##, * etc.
    s = re.sub(r"^[#*]+\s*", "", s)
    # Merge multiple spaces
    s = re.sub(r"\s+", " ", s)
    # Remove leading/trailing dots/pipes/dashes etc.
    s = s.strip(" .·•|-–—~")
    s = s.lower()
    # Lines that are too short don't count (e.g., "3", "p.5")
    if len(s.replace(" ", "")) < 5:
        return ""
    return s

def detect_header_footer_patterns(
    content: List[Dict[str, Any]],
    min_occurs: int = 3,
    min_ratio: float = 0.15,
) -> List[str]:
    """
    Count repetition of first & last lines from all text blocks,
    returns a list of "normalized line text" that may belong to headers/footers.
    """
    counter = Counter()
    text_count = 0

    for block in content:
        if block.get("type") != "text":
            continue
        text = block.get("text") or ""
        lines = text.splitlines()
        if not lines:
            continue
        text_count += 1

        first = _normalize_line(lines[0])
        last = _normalize_line(lines[-1])

        if first:
            counter[first] += 1
        if last and last != first:
            counter[last] += 1

    if text_count == 0:
        return []

    patterns = []
    for norm_line, cnt in counter.items():
        if cnt >= min_occurs and (cnt / text_count) >= min_ratio:
            patterns.append(norm_line)

    return patterns

def strip_header_footer(content: List[Dict[str, Any]]) -> List[Dict[str, Any]]:
    """
    Remove header/footer lines based on patterns found by detect_header_footer_patterns.
    Requirements:
        - First occurrence of each pattern is kept (usually paper title/first header)
        - Subsequent duplicate lines are deleted
    If a text block is completely emptied, the entire block is deleted.
    """
    patterns = detect_header_footer_patterns(content)
    if not patterns:
        return content

    pattern_set = set(patterns)
    # Track how many times each pattern has appeared: used for "keep first occurrence"
    pattern_seen: Dict[str, int] = {p: 0 for p in pattern_set}

    new_content: List[Dict[str, Any]] = []

    for block in content:
        if block.get("type") != "text":
            new_content.append(block)
            continue

        text = block.get("text") or ""
        lines = text.splitlines()
        kept_lines: List[str] = []

        for line in lines:
            norm = _normalize_line(line)
            if norm and norm in pattern_set:
                # Matched header/footer pattern
                if pattern_seen[norm] == 0:
                    # First occurrence: keep and mark as "seen"
                    pattern_seen[norm] += 1
                    kept_lines.append(line)
                else:
                    # Subsequent occurrences: treat as header/footer lines, delete
                    continue
            else:
                kept_lines.append(line)

        joined = "\n".join(kept_lines).strip()
        if not joined:
            # Entire block is header/footer, delete directly
            continue

        nb = dict(block)
        nb["text"] = joined
        new_content.append(nb)

    return new_content


# ======= Drop Checklist ======= #

def drop_checklist(content: List[Dict[str, Any]]) -> List[Dict[str, Any]]:
    new = []
    for blk in content:
        sec = (blk.get("section") or "").strip().lower()
        if sec == "checklist":
            continue
        new.append(blk)
    return new


# ======= Rebuild section_labels ======= #

def rebuild_section_labels(content: List[Dict[str, Any]]) -> Dict[str, Any]:
    labels = []

    def push(start: int, end: int, section: str):
        if start is None or end is None:
            return
        if start == end:
            idx_expr = str(start)
        else:
            idx_expr = f"{start}-{end}"
        labels.append({"content_index": idx_expr, "section": section})

    current_sec = None
    rstart = None
    last_idx = None

    for blk in content:
        idx = blk["index"]
        sec = blk.get("section") or "Introduction"

        if current_sec is None:
            current_sec = sec
            rstart = idx
            last_idx = idx
            continue

        if sec == current_sec and idx == last_idx + 1:
            last_idx = idx
        else:
            push(rstart, last_idx, current_sec)
            current_sec = sec
            rstart = idx
            last_idx = idx

    push(rstart, last_idx, current_sec)
    return {"labels": labels, "model_used": "post_clean"}


# ======= Main Logic ======= #

def process_one(path_in: Path, path_out: Path, strip_hf: bool, drop_cl: bool):
    data = load_json(path_in)
    content = data.get("content", [])

    # 1) Drop Checklist
    if drop_cl:
        content = drop_checklist(content)

    # 2) Strip header/footer (only remove duplicates, keep first occurrence)
    if strip_hf:
        content = strip_header_footer(content)

    # 3) Renumber index
    new_content = []
    for i, blk in enumerate(content, start=1):
        nb = dict(blk)
        nb["index"] = i
        new_content.append(nb)

    # 4) Rebuild section_labels
    section_labels = rebuild_section_labels(new_content)

    out_obj = dict(data)
    out_obj["content"] = new_content
    out_obj["section_labels"] = section_labels

    save_json(out_obj, path_out)


def main():
    parser = argparse.ArgumentParser(
        description="Post-clean sections: remove headers/footers and checklist sections",
        formatter_class=argparse.ArgumentDefaultsHelpFormatter
    )
    parser.add_argument("--root-dir", "--root_dir", type=str, required=True,
                        help="Root directory, recursively search for input JSON files")
    parser.add_argument("--input-name", "--input_name", type=str, default="paper_parse_add_section.json",
                        help="Input JSON filename")
    parser.add_argument("--output-name", "--output_name", type=str, default="paper_final.json",
                        help="Output JSON filename")
    parser.add_argument("--strip-header-footer", "--strip_header_footer", action="store_true",
                        help="Remove headers and footers")
    parser.add_argument("--drop-checklist", "--drop_checklist", action="store_true",
                        help="Remove Checklist sections")
    args = parser.parse_args()

    root = Path(args.root_dir).expanduser().resolve()
    if not root.exists():
        raise FileNotFoundError(f"Root dir not found: {root}")

    files = find_jsons(root, args.input_name)

    print(f"[INFO] root={root}")
    print(f"[INFO] STRIP_HEADER_FOOTER={args.strip_header_footer}  DROP_CHECKLIST={args.drop_checklist}")
    print(f"[INFO] Found {len(files)} files to process.")

    for p in tqdm(files, desc="Post-cleaning"):
        out = p.parent / args.output_name
        process_one(p, out, args.strip_header_footer, args.drop_checklist)

    print("[DONE]")

if __name__ == "__main__":
    main()