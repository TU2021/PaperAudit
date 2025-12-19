#!/usr/bin/env python3
# -*- coding: utf-8 -*-

import argparse
from pathlib import Path


def delete_inprogress(root: Path, dry_run: bool = False):
    if not root.exists() or not root.is_dir():
        print(f"[ERROR] 路径不存在或不是文件夹: {root}")
        return

    print(f"[INFO] 扫描根目录(删除 .inprogress): {root}")

    count = 0
    for file in root.rglob("*"):
        if file.is_file() and file.name.endswith(".inprogress"):
            count += 1
            if dry_run:
                print(f"[DRY-RUN] 将删除: {file}")
            else:
                try:
                    file.unlink()
                    print(f"[DEL] 已删除: {file}")
                except Exception as e:
                    print(f"[ERROR] 删除失败: {file} — {e}")

    print(f"[SUMMARY] 共找到 {count} 个 .inprogress 文件")
    if dry_run:
        print("[SUMMARY] （dry-run 模式，没有实际删除）")
    else:
        print("[SUMMARY] .inprogress 文件已全部删除完成")


def rename_json_with_substring(root: Path, old_substr: str, new_substr: str, dry_run: bool = False):
    """
    在 root 下递归查找所有 .json 文件：
    - 如果文件名中包含 old_substr，则替换为 new_substr 后重命名。
    """
    if not old_substr:
        print("[INFO] 未提供 old_substr，跳过 JSON 重命名步骤。")
        return

    print(f"[INFO] 扫描根目录(JSON 重命名): {root}")
    print(f"[INFO] 文件名中包含 '{old_substr}' 的 .json 文件，将替换为 '{new_substr}'")

    rename_count = 0
    for file in root.rglob("*.json"):
        if not file.is_file():
            continue

        if old_substr in file.name:
            new_name = file.name.replace(old_substr, new_substr)
            new_path = file.with_name(new_name)

            if new_path == file:
                continue  # 实际没变

            rename_count += 1
            if dry_run:
                print(f"[DRY-RUN] 将重命名: {file}  ->  {new_path}")
            else:
                if new_path.exists():
                    print(f"[WARN] 目标文件已存在，跳过: {new_path}")
                    continue
                try:
                    file.rename(new_path)
                    print(f"[REN] 已重命名: {file}  ->  {new_path}")
                except Exception as e:
                    print(f"[ERROR] 重命名失败: {file} — {e}")

    print(f"[SUMMARY] 需要重命名的 .json 文件数: {rename_count}")
    if dry_run:
        print("[SUMMARY] （dry-run 模式，没有实际重命名）")
    else:
        print("[SUMMARY] JSON 重命名操作完成")


def main():
    parser = argparse.ArgumentParser(
        description="删除所有子文件夹下的 .inprogress 文件，并可选重命名 .json 文件中包含指定子串的文件名"
    )
    parser.add_argument(
        "--root_dir",
        type=str,
        required=True,
        help="根目录，例如 /path/to/NeurIPS_35"
    )
    parser.add_argument(
        "--dry",
        action="store_true",
        help="仅打印将执行的操作，不实际删除/重命名"
    )
    parser.add_argument(
        "--old_substr",
        type=str,
        default="",
        help="在 .json 文件名中要查找并替换的子串（例如 'gpt-4o'）"
    )
    parser.add_argument(
        "--new_substr",
        type=str,
        default="",
        help="替换后的子串（例如 'gpt-5-2025-08-07'）"
    )

    args = parser.parse_args()
    root = Path(args.root_dir)

    # 1. 先删除 .inprogress
    delete_inprogress(root, dry_run=args.dry)

    # 2. 再进行 .json 重命名（只有提供了 old_substr 时才启用）
    if args.old_substr:
        rename_json_with_substring(
            root,
            old_substr=args.old_substr,
            new_substr=args.new_substr,
            dry_run=args.dry,
        )
    else:
        print("[INFO] 未提供 --old_substr，跳过 JSON 文件重命名。")


if __name__ == "__main__":
    main()
