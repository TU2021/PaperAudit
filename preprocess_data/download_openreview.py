#!/usr/bin/env python3
# -*- coding: utf-8 -*-

"""
OpenReview Paper Downloader

This script downloads papers, reviews, and metadata from OpenReview for a specified conference.

Usage:
    python download_openreview.py [OPTIONS]

Examples:
    # Download ICLR 2025 oral papers
    python download_openreview.py -c ICLR.cc -y 2025 -t oral -u your_email@example.com -p your_password

    # Download NeurIPS 2024 poster papers to a custom directory
    python download_openreview.py --conference NeurIPS.cc --year 2024 --type poster --username your_email@example.com --password your_password --out-dir ./neurips_downloads

Arguments:
    --conference, -c    Conference name (e.g., ICLR.cc, NeurIPS.cc, ICML.cc)
    --year, -y          Conference year (e.g., 2024, 2025)
    --type, -t          Paper type (e.g., oral, poster, spotlight)
    --username, -u      OpenReview username/email (required)
    --password, -p      OpenReview password (required)
    --out-dir, -o       Output directory for downloaded files (default: ./downloads)
    --sleep, -s         Sleep time in seconds between downloads (default: 0.35)
    --skip-existing     Skip existing files (enabled by default)
    --no-skip-existing  Force re-download even if files exist

Output Structure:
    {out_dir}/{CONFERENCE}_{YEAR}_{TYPE}/
        {paper_id}-{paper_title}/
            paper.pdf          # Paper PDF
            reviews.json       # Reviews and discussion threads
            metadata.json      # Paper metadata
        summary.json           # Summary of all downloaded papers
"""

import json, time
from pathlib import Path
from typing import List, Dict, Any
import requests
from tqdm import tqdm
import openreview
import argparse

# ==========================

BASEURL = "https://api2.openreview.net" # V2 API
PDF_URL = "https://openreview.net/pdf?id={forum}"
REVIEW_HINTS = ("/Official_Review", "/Review", "/Meta_Review", "/Decision", "/Comment")
ATTACHMENT_FIELDS = ("pdf", "submission", "paper", "file", "source")

def login(username: str, password: str):
    return openreview.api.OpenReviewClient(baseurl=BASEURL, username=username, password=password)

def extract_value(x):
    """Extract value from OpenReview field: dict({'value': ...}) -> str; str -> as-is; others -> ''"""
    if isinstance(x, dict):
        return x.get("value", "")
    if isinstance(x, str):
        return x
    return ""

def normalize_content(d):
    """Flatten nested value fields in note.content: {'decision': {'value':'Accept'}} -> {'decision':'Accept'}"""
    if not isinstance(d, dict):
        return d
    out = {}
    for k, v in d.items():
        if isinstance(v, dict) and "value" in v and len(v) <= 2:
            out[k] = extract_value(v)
        else:
            out[k] = v
    return out

def safe_title(s: str) -> str:
    s = extract_value(s)
    s = (s or "").strip()
    s = "".join(ch if ch.isalnum() or ch in "-_. " else "_" for ch in s)
    return (s.replace(" ", "_") or "no_title")[:160]

def get_all_notes(client, **kwargs):
    out, offset, limit = [], 0, 1000
    while True:
        notes = client.get_notes(offset=offset, limit=limit, **kwargs)
        if not notes: break
        out.extend(notes); offset += len(notes)
    return out

def filter_by_type(notes: List[Any], conf_type: str):
    t_norm = conf_type.lower()
    t_title = t_norm.capitalize()
    kept = []
    for n in notes:
        c = getattr(n, "content", {}) or {}
        venue = c.get("venue") or c.get("Venue") or ""
        venue = extract_value(venue)

        ok = False
        if isinstance(venue, str) and (f"({t_title})" in venue or f"({t_norm})" in venue or t_title in venue or t_norm in venue):
            ok = True
        if ok:
            kept.append(n)

    return kept

def exists_nonempty(p: Path) -> bool:
    try:
        return p.exists() and p.is_file() and p.stat().st_size > 0
    except Exception:
        return False
    
def download_pdf(client, note, pdir: Path, skip_existing: bool = True) -> bool:
    """Download PDF; return True if non-empty file already exists."""
    pdf_path = pdir / "paper.pdf"
    if skip_existing and exists_nonempty(pdf_path):
        return True

    # Try attachment first (higher success rate)
    for f in ATTACHMENT_FIELDS:
        try:
            data = client.get_attachment(note.id, f)
            if data:
                pdf_path.write_bytes(data)
                return True
        except Exception:
            pass

    # Fallback to public PDF URL
    forum = getattr(note, "forum", None) or note.id
    try:
        r = requests.get(PDF_URL.format(forum=forum), timeout=30)
        if r.status_code == 200 and r.headers.get("content-type","").lower().startswith("application/pdf"):
            pdf_path.write_bytes(r.content)
            return True
    except Exception:
        pass
    return False

def _clean_invitation_from(r):
    """
    Extract a non-system invitation name from r.invitations, returning only the last segment:
    e.g., '.../-/Official_Review' -> 'Official_Review'
    Automatically ignores Edit/Withdraw/Revision etc.
    """
    bad = ("/-Edit", "/-Withdraw", "/-Revision", "/-Desk_Rejected", "/-Withdrawn")
    inv_list = getattr(r, "invitations", []) or []
    for inv in inv_list:
        if not any(b in inv for b in bad):
            return inv.split("/")[-1]
    return ""

def collect_reviews_metadata(client, forum_id: str):
    replies = client.get_all_notes(forum=forum_id)

    # 1) Index all nodes (flatten content, clean invitation)
    nodes = {}
    for r in replies:
        node = {
            "id": getattr(r, "id", None),
            "forum": getattr(r, "forum", None),
            "replyto": getattr(r, "replyto", None),
            "invitation": _clean_invitation_from(r),  # Keep only short names like 'Official_Review'
            "cdate": getattr(r, "cdate", None),
            "mdate": getattr(r, "mdate", None),
            "content": normalize_content(getattr(r, "content", {}) or {}),
            "children": []
        }
        nodes[node["id"]] = node

    # 2) Find submission root (usually id == forum)
    submission = nodes.get(forum_id)
    if submission is None and replies:
        # Fallback: use the earliest one as submission
        root = min(replies, key=lambda x: getattr(x, "cdate", 0))
        submission = nodes.get(getattr(root, "id", None))

    # 3) Build child links
    for node in nodes.values():
        pid = node["replyto"]
        if pid and pid in nodes:
            nodes[pid]["children"].append(node)

    # 4) Classify (note: invitation is now the last segment short name)
    decisions, meta_reviews = [], []
    for node in nodes.values():
        inv = node["invitation"]
        if inv == "Decision" or inv.endswith("Decision"):
            decisions.append(node)
        elif inv == "Meta_Review" or inv.endswith("Meta_Review"):
            meta_reviews.append(node)

    # 5) Only take roots directly attached to submission as starting points for each review thread
    if submission:
        thread_roots = list(submission["children"])
    else:
        # Fallback: use nodes with replyto==forum when submission is not identified
        thread_roots = [n for n in nodes.values() if n["replyto"] == forum_id]

    # 6) Sort: each tree top-down by time from earliest to latest
    def sort_rec(n):
        n["children"].sort(key=lambda x: (x.get("cdate") or 0))
        for ch in n["children"]:
            sort_rec(ch)

    for tr in thread_roots:
        sort_rec(tr)
    thread_roots.sort(key=lambda x: (x.get("cdate") or 0))

    # 7) Add friendly type labels (based on short names)
    def label_kind(inv):
        if inv == "Official_Review" or inv.endswith("Review") and not inv.endswith("Meta_Review"):
            return "review"
        if inv.endswith("Meta_Review"):
            return "meta_review"
        if inv.endswith("Decision"):
            return "decision"
        if inv.endswith("Comment") or inv.endswith("Public_Comment"):
            return "comment"
        return "other"

    for node in nodes.values():
        node["kind"] = label_kind(node["invitation"])

    # 8) Output (exclude submission to avoid duplication; submission info goes to metadata.json)
    review_dict = {
        "forum": forum_id,
        "threads": thread_roots,
        "meta_reviews": meta_reviews,
        "decisions": decisions
    }

    # Create a simplified copy of submission: remove children field, keep other info
    submission_dict = {k: v for k, v in submission.items() if k != "children"}



    return review_dict, submission_dict

def parse_args():
    parser = argparse.ArgumentParser(
        description="Download conference papers and reviews from OpenReview",
        formatter_class=argparse.ArgumentDefaultsHelpFormatter
    )
    parser.add_argument(
        "--conference", "-c",
        type=str,
        default="ICLR.cc",
        help="Conference name, e.g., ICLR.cc, NeurIPS.cc, ICML.cc"
    )
    parser.add_argument(
        "--year", "-y",
        type=int,
        default=2025,
        help="Conference year"
    )
    parser.add_argument(
        "--type", "-t",
        type=str,
        default="oral",
        help="Paper type, e.g., oral, poster, spotlight"
    )
    parser.add_argument(
        "--username", "-u",
        type=str,
        required=True,
        help="OpenReview username/email (required)"
    )
    parser.add_argument(
        "--password", "-p",
        type=str,
        required=True,
        help="OpenReview password (required)"
    )
    parser.add_argument(
        "--out-dir", "-o",
        type=str,
        default="./downloads",
        help="Output directory"
    )
    parser.add_argument(
        "--sleep", "-s",
        type=float,
        default=0.35,
        help="Sleep time in seconds between downloads"
    )
    parser.add_argument(
        "--skip-existing",
        action="store_true",
        default=True,
        help="Skip existing files (enabled by default)"
    )
    parser.add_argument(
        "--no-skip-existing",
        dest="skip_existing",
        action="store_false",
        help="Do not skip existing files, force re-download"
    )
    return parser.parse_args()

def main():
    args = parse_args()
    
    conference = args.conference
    year = args.year
    paper_type = args.type
    username = args.username
    password = args.password
    out_dir = args.out_dir
    sleep_time = args.sleep
    skip_existing = args.skip_existing
    
    venue_id = f"{conference}/{year}/Conference"
    out_root = Path(out_dir).expanduser() / f"{conference.split('.')[0]}_{year}_{paper_type}"
    out_root.mkdir(parents=True, exist_ok=True)

    client = login(username, password)
    print(f"📥 Fetching all submissions from {venue_id}...")
    all_notes = get_all_notes(client, content={"venueid": venue_id})
    print(f"✅ Total submissions: {len(all_notes)}")

    # Filter by type
    subset = filter_by_type(all_notes, paper_type)
    print(f"🎯 Matching type ({paper_type}): {len(subset)} papers")

    summary = []
    for n in tqdm(subset, desc="Downloading"):
        c = getattr(n, "content", {}) or {}
        title = c.get("title") or c.get("Title") or "<no-title>"
        forum = getattr(n, "forum", None) or getattr(n, "id", None)
        number = getattr(n, "number", None)
        sub_id = str(number) if number else (forum[:8] if forum else "unknown")

        base = f"{sub_id}-{safe_title(title)}"
        pdir = out_root / base
        pdir.mkdir(parents=True, exist_ok=True)

        pdf_path = pdir / "paper.pdf"
        reviews_path = pdir / "reviews.json"
        metadata_path = pdir / "metadata.json"

        # 1) PDF
        pdf_ok = False
        if skip_existing and exists_nonempty(pdf_path):
            pdf_ok = True
        else:
            pdf_ok = download_pdf(client, n, pdir, skip_existing)

        # 2) Reviews / Metadata
        # Skip fetching if both files already exist and are non-empty
        if skip_existing and exists_nonempty(reviews_path) and exists_nonempty(metadata_path):
            with reviews_path.open("r", encoding="utf-8") as f:
                reviews_loaded = json.load(f)
            reviews_cnt = len(reviews_loaded.get("threads", []))
        else:
            reviews, metadata = collect_reviews_metadata(client, forum)
            reviews_path.write_text(json.dumps(reviews, indent=2, ensure_ascii=False), encoding="utf-8")
            metadata_path.write_text(json.dumps(metadata, indent=2, ensure_ascii=False), encoding="utf-8")
            reviews_cnt = len(reviews.get("threads", []))

        summary.append({"title": extract_value(title), "forum": forum, "pdf": pdf_ok, "reviews_count": reviews_cnt})
        time.sleep(sleep_time)

    (out_root / "summary.json").write_text(json.dumps(summary, indent=2, ensure_ascii=False), encoding="utf-8")
    print(f"\n✅ Download complete: {len(summary)} papers, saved to: {out_root.resolve()}")
    for s in summary[:3]:
        print(f"  - {s['title'][:80]} (pdf={s['pdf']}, reviews={s['reviews_count']})")

if __name__ == "__main__":
    main()
