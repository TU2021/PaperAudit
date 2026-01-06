#!/usr/bin/env python3
# -*- coding: utf-8 -*-

import argparse
from pathlib import Path


def rename_pdfs(root_dir: Path, overwrite: bool = False):
    for subdir in root_dir.rglob("*"):
        if not subdir.is_dir():
            continue

        pdfs = sorted(subdir.glob("*.pdf"))
        if not pdfs:
            continue

        target = subdir / "paper.pdf"

        if target.exists() and not overwrite:
            print(f"[SKIP] {target} already exists")
            continue

        if len(pdfs) > 1:
            print(f"[WARN] Multiple PDFs in {subdir}, using: {pdfs[0].name}")

        src = pdfs[0]

        if src == target:
            continue

        src.rename(target)
        print(f"[OK] {src} → {target}")


def main():
    parser = argparse.ArgumentParser(
        description="Rename PDFs in subfolders to paper.pdf"
    )
    parser.add_argument(
        "--root_dir",
        type=Path,
        help="Root directory to scan"
    )
    parser.add_argument(
        "--overwrite",
        action="store_true",
        help="Overwrite existing paper.pdf"
    )

    args = parser.parse_args()
    rename_pdfs(args.root_dir, args.overwrite)


if __name__ == "__main__":
    main()
