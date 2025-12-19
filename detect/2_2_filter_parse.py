#!/usr/bin/env python3
# -*- coding: utf-8 -*-

"""
post_clean_sections.py (global switches version)

在 batch_label_sections.py 的解析结果基础上做规则后处理：

由两个全局变量决定是否启用特性：

    STRIP_HEADER_FOOTER = True/False   删除页眉页脚
    DROP_CHECKLIST = True/False       删除 Checklist 模块

输入格式为：
    paper_parse.json（内含 content + section_labels）

输出格式为：
    paper_clean.json
"""

from pathlib import Path
import json
import re
from typing import List, Dict, Any
from collections import Counter
from tqdm import tqdm
from utils import load_json, save_json

# ===========================================================
# 全局变量 —— 在这里改开关
# ===========================================================
STRIP_HEADER_FOOTER = True      # 删除页眉、页脚
DROP_CHECKLIST = True           # 删除 Checklist 模块
INPUT_NAME = "paper_parse_add_section.json"
OUTPUT_NAME = "paper_final.json"
ROOT_DIR = "/mnt/parallel_ssd/home/zdhs0006/mlrbench/download/data/ICLR_30"
# ===========================================================


def find_jsons(root: Path, name: str) -> List[Path]:
    return sorted([p for p in root.rglob(name) if p.is_file()])


# ======= 规则：页眉/页脚清洗 ======= #

def _normalize_line(line: str) -> str:
    """
    用于判断“是不是同一行”的归一化规则：
    - 去首尾空格
    - 去掉开头的 Markdown 标题符号 (#, *, 之类)，保证 "# VersaPRM ..." 和 "VersaPRM ..." 归到一起
    - 多空格合并
    - 去掉前后常见符号，转小写
    - 太短的行（比如单个页码）直接视为无效
    """
    if not line:
        return ""
    s = line.strip()
    # 去掉 markdown 标题、列表符号前缀：#、##、* 等
    s = re.sub(r"^[#*]+\s*", "", s)
    # 多个空格合并
    s = re.sub(r"\s+", " ", s)
    # 去掉前后点号/竖线/破折号等
    s = s.strip(" .·•|-–—~")
    s = s.lower()
    # 净长度太短的不算（如 "3"、"p.5"）
    if len(s.replace(" ", "")) < 5:
        return ""
    return s

def detect_header_footer_patterns(
    content: List[Dict[str, Any]],
    min_occurs: int = 3,
    min_ratio: float = 0.15,
) -> List[str]:
    """
    从所有 text block 中统计首行 & 末行的重复情况，
    返回可能属于页眉/页脚的“归一化行文本”列表。
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
    根据 detect_header_footer_patterns 找到的模式，删除页眉/页脚行。
    🔹 新要求：
        - 每个 pattern 的第一次出现会保留（通常是论文标题/第一次出现的 header）
        - 后续重复行会被删除
    如果一个 text block 全部被删空，则整体删除。
    """
    patterns = detect_header_footer_patterns(content)
    if not patterns:
        return content

    pattern_set = set(patterns)
    # 记录每种 pattern 已经出现过几次：用于“保留第一次”
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
                # 命中 header/footer 模式
                if pattern_seen[norm] == 0:
                    # 第一次出现：保留，并打标“已见过”
                    pattern_seen[norm] += 1
                    kept_lines.append(line)
                else:
                    # 后续出现：视为页眉/页脚行，删除
                    continue
            else:
                kept_lines.append(line)

        joined = "\n".join(kept_lines).strip()
        if not joined:
            # 整块都是 header/footer，直接删掉
            continue

        nb = dict(block)
        nb["text"] = joined
        new_content.append(nb)

    return new_content


# ======= 删除 Checklist ======= #

def drop_checklist(content: List[Dict[str, Any]]) -> List[Dict[str, Any]]:
    new = []
    for blk in content:
        sec = (blk.get("section") or "").strip().lower()
        if sec == "checklist":
            continue
        new.append(blk)
    return new


# ======= 重建 section_labels ======= #

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


# ======= 主逻辑 ======= #

def process_one(path_in: Path, path_out: Path):
    data = load_json(path_in)
    content = data.get("content", [])

    # 1) 删除 Checklist
    if DROP_CHECKLIST:
        content = drop_checklist(content)

    # 2) 删除页眉/页脚（只删重复，保留第一次）
    if STRIP_HEADER_FOOTER:
        content = strip_header_footer(content)

    # 3) 重新编号 index
    new_content = []
    for i, blk in enumerate(content, start=1):
        nb = dict(blk)
        nb["index"] = i
        new_content.append(nb)

    # 4) 重建 section_labels
    section_labels = rebuild_section_labels(new_content)

    out_obj = dict(data)
    out_obj["content"] = new_content
    out_obj["section_labels"] = section_labels

    save_json(out_obj, path_out)


def main():
    root = Path(ROOT_DIR).expanduser().resolve()
    files = find_jsons(root, INPUT_NAME)

    print(f"[INFO] root={root}")
    print(f"[INFO] STRIP_HEADER_FOOTER={STRIP_HEADER_FOOTER}  DROP_CHECKLIST={DROP_CHECKLIST}")
    print(f"[INFO] Found {len(files)} files to process.")

    for p in tqdm(files, desc="Post-cleaning"):
        out = p.parent / OUTPUT_NAME
        process_one(p, out)

    print("[DONE]")

if __name__ == "__main__":
    main()