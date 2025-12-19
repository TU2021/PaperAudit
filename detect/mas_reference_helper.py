#!/usr/bin/env python3
# -*- coding: utf-8 -*-

"""
mas_reference_helper.py

从全文 blocks 中提取 References，并根据 section 里的引用（如
  - LeCun et al., 1998
  - Rajabi & Kosecka, 2024
  - [3]
在对应 section 的 blocks 末尾，注入“本 section 用到的参考文献”。
"""

from __future__ import annotations
from typing import List, Dict, Tuple
import re


def _collect_reference_blocks(blocks: List[Dict]) -> List[Dict]:
    """
    从全文 blocks 中收集 References / Reference / Bibliography 相关的 text blocks。
    依赖 normalize_blocks 之后的结构：每个 block 可能有 'section' 和 'type' 字段。
    """
    ref_titles = {"references", "reference", "bibliography"}
    ref_blocks: List[Dict] = []
    for b in blocks:
        sec = (b.get("section") or "").strip().lower()
        if sec in ref_titles and b.get("type") == "text":
            ref_blocks.append(b)
    return ref_blocks


def _build_references_text(ref_blocks: List[Dict]) -> str:
    """把 reference blocks 合并成一个长字符串。"""
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
    从当前 section 的文本中抽取两类引用 key：
      - author_year_keys: [("lecun", "1998"), ("rajabi", "2024"), ...]
      - numeric_keys: ["1", "3", "12", ...]  对应 [1] [3] [12] 这种
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
    # 注意：只取第一个姓作为 key（比如 Rajabi），第二个姓仅用于识别模式。
    pattern_and = re.compile(
        r"\b([A-Z][A-Za-z\-]+)\s*(?:&|and)\s*[A-Z][A-Za-z\-]+\s*,\s*(\d{4})"
    )
    for m in pattern_and.finditer(full_text):
        surname = m.group(1).strip().lower()
        year = m.group(2).strip()
        key = (surname, year)
        if key not in author_year_keys:
            author_year_keys.append(key)

    # ---------- 3) [1] [23] 这种数字引用 ----------
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
    在整块 references 文本中，基于 author_year_keys & numeric_keys 找到对应的参考文献条目。
    简单做法：按空行切段，每段看成一个 reference entry。
    """
    if not ref_text.strip():
        return []

    # 用空行切成一个个 entry
    paragraphs = re.split(r"\n\s*\n", ref_text.strip())
    used_entries: List[str] = []
    seen = set()

    # 先处理 author-year
    for surname, year in author_year_keys:
        for para in paragraphs:
            para_norm = para.replace("\n", " ")
            if surname in para_norm.lower() and year in para_norm:
                key = ("ay", surname, year, para.strip())
                if key not in seen:
                    seen.add(key)
                    used_entries.append(para.strip())
                break  # 一个 (surname, year) 找到一个就够了

    # 再处理 numeric
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
    """给新 block 分配一个略大的 content_index。"""
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
    基于 section 文本中的引用，在 References section 中找对应条目，
    并把这些条目作为一个新的 text block 附加到当前 section 的 blocks 末尾。

    - section_title: 当前 section 的名字（比如 "Related Work"）
    - section_blocks: 这个 section 的所有 blocks（通常是 slice_json_for_task_with_outline 的结果）
    - all_blocks: 全文 blocks（normalize_blocks 后的结果）
    """
    # 1) 找到 references blocks
    ref_blocks = _collect_reference_blocks(all_blocks)
    if not ref_blocks:
        return section_blocks

    ref_text = _build_references_text(ref_blocks)
    if not ref_text.strip():
        return section_blocks

    # 2) 抽取当前 section 的引用 key
    author_year_keys, numeric_keys = _extract_citation_keys_from_section_blocks(section_blocks)
    if not author_year_keys and not numeric_keys:
        return section_blocks

    # 3) 在 references 文本中匹配对应条目
    used_entries = _match_reference_entries(ref_text, author_year_keys, numeric_keys)
    if not used_entries:
        return section_blocks

    # 4) 构造新的 block 附在 section 后面
    ci = _next_content_index(all_blocks)
    ref_block_text = "References cited in this section:\n\n" + "\n\n".join(used_entries)

    new_block = {
        "type": "text",
        "text": ref_block_text,
        "section": section_title,
        "content_index": ci,
    }
    return section_blocks + [new_block]
