#!/usr/bin/env python3
# -*- coding: utf-8 -*-

import argparse
import shutil
from pathlib import Path
from typing import Optional


def find_dirs(root: Path, target_name: str):
    if not root.exists() or not root.is_dir():
        raise ValueError(f"Root path does not exist or is not a directory: {root}")
    # 精确匹配目录名：用 rglob("*") + name 判断最稳；你原来 rglob(target_name) 也可以
    return [p for p in root.rglob("*") if p.is_dir() and p.name == target_name]


def find_and_delete_dirs(root: Path, target_name: str, dry_run: bool = False) -> int:
    targets = find_dirs(root, target_name)

    if dry_run:
        print("[DRY-RUN] Directories to be deleted:")
        for p in targets:
            print(f"  {p}")
        print(f"[DRY-RUN] Total: {len(targets)}")
        return len(targets)

    deleted = 0
    for p in targets:
        try:
            shutil.rmtree(p)
            print(f"[DEL] {p}")
            deleted += 1
        except Exception as e:
            print(f"[ERROR] Failed to delete: {p} -> {e}")

    print(f"[SUMMARY] Deleted: {deleted} / Found: {len(targets)}")
    return deleted


def find_and_rename_dirs(root: Path, from_name: str, to_name: str, dry_run: bool = False) -> int:
    if from_name == to_name:
        print("[INFO] from_name == to_name, nothing to do.")
        return 0

    targets = find_dirs(root, from_name)

    if dry_run:
        print("[DRY-RUN] Directories to be renamed:")
        for p in targets:
            dst = p.with_name(to_name)
            print(f"  {p}  ->  {dst}")
        print(f"[DRY-RUN] Total: {len(targets)}")
        return len(targets)

    renamed = 0
    for p in targets:
        dst = p.with_name(to_name)
        try:
            if dst.exists():
                raise FileExistsError(f"destination already exists: {dst}")
            p.rename(dst)  # 同一分区内重命名最直接
            print(f"[REN] {p} -> {dst}")
            renamed += 1
        except Exception as e:
            print(f"[ERROR] Failed to rename: {p} -> {dst} -> {e}")

    print(f"[SUMMARY] Renamed: {renamed} / Found: {len(targets)}")
    return renamed


def main():
    parser = argparse.ArgumentParser(
        description="Recursively find subdirectories with a given name and delete them, or rename them."
    )
    parser.add_argument("--root", type=str, required=True, help="Root folder to search within")
    parser.add_argument("--name", type=str, required=True, help="Target subfolder name (exact match)")
    parser.add_argument(
        "--rename-to",
        type=str,
        default=None,
        help="If set, rename found subfolders from --name to this new name (instead of deleting).",
    )
    parser.add_argument(
        "--dry-run",
        action="store_true",
        help="Print planned operations without modifying anything",
    )
    args = parser.parse_args()

    root = Path(args.root).expanduser().resolve()

    if args.rename_to is None:
        find_and_delete_dirs(root, args.name, dry_run=args.dry_run)
    else:
        find_and_rename_dirs(root, args.name, args.rename_to, dry_run=args.dry_run)


if __name__ == "__main__":
    main()
