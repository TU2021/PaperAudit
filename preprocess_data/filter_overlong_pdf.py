#!/usr/bin/env python3
# -*- coding: utf-8 -*-

"""
PDF Page Filter Tool

This script counts PDF pages in subdirectories and filters papers by page count.
Each subdirectory should contain a paper.pdf file.

Usage:
    python filter_overlong_pdf.py [OPTIONS]

Examples:
    # Count pages and generate statistics report
    python filter_overlong_pdf.py --root /path/to/papers

    # Filter papers with less than 30 pages to a destination directory
    python filter_overlong_pdf.py --root /path/to/papers --max-pages 30 --dest /path/to/filtered

    # Dry run to see what would be copied without actually copying
    python filter_overlong_pdf.py --root /path/to/papers --max-pages 25 --dest /path/to/filtered --dry-run

    # Show help message
    python filter_overlong_pdf.py --help

Arguments:
    --root          Root directory containing paper subdirectories (each with paper.pdf)
    --max-pages     Threshold: copy papers with pages < threshold to destination (optional)
    --dest          Destination directory for filtered papers (default: <root>__LT_<N>)
    --dry-run       Show what would be copied without actually copying

Output:
    The script generates two report files in the root directory:
    - pdf_page_report.csv: CSV file with folder, pdf_path, and pages
    - pdf_page_report.json: JSON file with summary, histogram, and detailed items
"""

import argparse, csv, json, math, shutil, sys
from pathlib import Path
from statistics import mean, median
from typing import Dict, List, Tuple


# --- Prefer pypdf; fallback to PyPDF2 ---
def _import_pdf_reader():
    try:
        from pypdf import PdfReader  # type: ignore
        return PdfReader, "pypdf"
    except Exception:
        try:
            from PyPDF2 import PdfReader  # type: ignore
            return PdfReader, "PyPDF2"
        except Exception:
            print("Please install pypdf or PyPDF2: pip install pypdf or pip install PyPDF2", file=sys.stderr)
            sys.exit(1)


PdfReader, _PDF_LIB = _import_pdf_reader()


def count_pdf_pages(pdf_path: Path) -> int:
    """Return PDF page count; return -1 if unable to read."""
    try:
        with pdf_path.open("rb") as f:
            reader = PdfReader(f)
            # Both pypdf and PyPDF2 support len(reader.pages)
            return len(reader.pages)
    except Exception:
        return -1


def find_papers(root: Path) -> List[Tuple[Path, Path]]:
    """
    Find paper.pdf in first-level subdirectories of root.
    Returns [(paper_dir, pdf_path), ...]
    """
    out = []
    for sub in sorted([p for p in root.iterdir() if p.is_dir()]):
        pdf_path = sub / "paper.pdf"
        if pdf_path.is_file():
            out.append((sub, pdf_path))
    return out


def make_bins_by5(max_pages: int) -> List[Tuple[int, int]]:
    """
    Generate [(1,5), (6,10), ...] up to max_pages (rounded up to multiple of 5).
    """
    top = int(math.ceil(max(1, max_pages) / 5.0) * 5)
    bins = []
    start = 1
    while start <= top:
        end = start + 4
        bins.append((start, end))
        start = end + 1
    return bins


def bin_label(lo: int, hi: int) -> str:
    return f"{lo}-{hi}"


def build_histogram(page_counts: List[int]) -> Dict[str, int]:
    """Build histogram by 5-page intervals (ignore counts <= 0)."""
    valid = [p for p in page_counts if p > 0]
    if not valid:
        return {}
    bins = make_bins_by5(max(valid))
    hist = {bin_label(lo, hi): 0 for (lo, hi) in bins}
    for p in valid:
        # Find corresponding bin
        idx = (int(math.ceil(p / 5.0)) - 1)
        lo, hi = bins[idx]
        hist[bin_label(lo, hi)] += 1
    return hist


def safe_copy_dir(src: Path, dst_dir: Path) -> Path:
    """
    Copy src directory to dst_dir; append number suffix if name conflicts to avoid overwriting.
    Returns destination path. Original directory is not modified.
    """
    base = src.name
    target = dst_dir / base
    if not target.exists():
        shutil.copytree(str(src), str(target))
        return target
    # Handle name conflicts: base__1, base__2, ...
    i = 1
    while True:
        cand = dst_dir / f"{base}__{i}"
        if not cand.exists():
            shutil.copytree(str(src), str(cand))
            return cand
        i += 1


def save_reports(root: Path, rows: List[Dict], hist: Dict[str, int]) -> None:
    # CSV
    csv_path = root / "pdf_page_report.csv"
    with csv_path.open("w", newline="", encoding="utf-8") as f:
        writer = csv.DictWriter(f, fieldnames=["folder", "pdf_path", "pages"])
        writer.writeheader()
        for r in rows:
            writer.writerow({
                "folder": r["folder"],
                "pdf_path": r["pdf_path"],
                "pages": r["pages"],
            })
    # JSON
    json_path = root / "pdf_page_report.json"
    report = {
        "summary": {
            "total_papers": len(rows),
            "readable_papers": sum(1 for r in rows if r["pages"] > 0),
            "unreadable_papers": sum(1 for r in rows if r["pages"] <= 0),
        },
        "histogram_by5": hist,
        "items": rows,
    }
    json_path.write_text(json.dumps(report, indent=2, ensure_ascii=False), encoding="utf-8")


def main():
    ap = argparse.ArgumentParser(
        description="Count and filter PDFs by page count (each subdirectory should contain paper.pdf)",
        formatter_class=argparse.ArgumentDefaultsHelpFormatter
    )
    ap.add_argument("--root", type=str, required=True,
                    help="Root directory path (contains paper subdirectories, each with paper.pdf)")
    ap.add_argument("--max-pages", type=int, default=None,
                    help="Threshold (e.g., 25). If provided, copy papers with pages < threshold to destination")
    ap.add_argument("--dest", type=str, default=None,
                    help="Destination directory for filtered papers (default: <root>__LT_<N>)")
    ap.add_argument("--dry-run", action="store_true", 
                    help="Show what would be copied without actually copying")
    args = ap.parse_args()

    root = Path(args.root).expanduser().resolve()
    if not root.exists() or not root.is_dir():
        print(f"[ERROR] Root does not exist or is not a directory: {root}", file=sys.stderr)
        sys.exit(2)

    # 1) Collect paper.pdf files
    pairs = find_papers(root)
    if not pairs:
        print(f"[INFO] No subdirectories containing paper.pdf found under {root}.")
        sys.exit(0)

    # 2) Count pages
    rows = []
    page_values = []
    unreadable = 0
    for folder, pdf in pairs:
        pages = count_pdf_pages(pdf)
        rows.append({"folder": str(folder.name), "pdf_path": str(pdf), "pages": pages})
        if pages > 0:
            page_values.append(pages)
        else:
            unreadable += 1

    # 3) Histogram (by 5 pages)
    hist = build_histogram(page_values)
    save_reports(root, rows, hist)

    # 4) Terminal output statistics
    total = len(rows)
    ok = len(page_values)
    print(f"\n====== Statistics (Library: {_PDF_LIB})======")
    print(f"Total papers: {total}")
    print(f"Readable: {ok} ; Unreadable: {unreadable}")
    if ok > 0:
        print(f"Min pages: {min(page_values)}  Max pages: {max(page_values)}")
        print(f"Mean: {mean(page_values):.2f}  Median: {median(page_values):.2f}")
    print("\n【Histogram by 5-page intervals】")
    if hist:
        # Ensure output is sorted by interval
        def _key(k):
            lo = int(k.split("-")[0])
            return lo
        for k in sorted(hist.keys(), key=_key):
            print(f"{k:>7}: {hist[k]}")
    else:
        print("(No valid page counts)")

    # 5) If threshold provided, filter and copy
    if args.max_pages is not None:
        threshold = int(args.max_pages)
        # Only copy papers with 0 < pages < threshold
        to_copy = [r for r in rows if 0 < r["pages"] < threshold]
        keep = [r for r in rows if r["pages"] >= threshold]

        print(f"\nThreshold: {threshold} pages")
        print(f"To copy (<{threshold} pages): {len(to_copy)} papers")
        print(f"Keep (≥{threshold} pages, or unreadable): {len(keep) + unreadable}")

        if to_copy:
            dest = Path(args.dest).expanduser().resolve() if args.dest else (
                root.parent / f"{root.name}__LT_{threshold}"
            )
            print(f"Destination directory: {dest}")
            if not args.dry_run:
                dest.mkdir(parents=True, exist_ok=True)
                # Execute copy (original directory unchanged)
                for r in to_copy:
                    src = root / r["folder"]
                    if src.exists() and src.is_dir():
                        safe_copy_dir(src, dest)
                print("\n✅ Copy completed.")
            else:
                # Only show list of what would be copied
                print("\n[Dry-Run] Will copy the following subdirectories:")
                for r in to_copy:
                    print(" -", r["folder"])

        # Report file reminder
        print(f"\nReport files:")
        print(f" - {root / 'pdf_page_report.csv'}")
        print(f" - {root / 'pdf_page_report.json'}")
    else:
        print("\n--max-pages not specified, statistics only.")
        print(f"Report files:")
        print(f" - {root / 'pdf_page_report.csv'}")
        print(f" - {root / 'pdf_page_report.json'}")


if __name__ == "__main__":
    main()
