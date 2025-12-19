#!/usr/bin/env python3
# -*- coding: utf-8 -*-

import json, time
from pathlib import Path
from typing import List, Dict, Any
import requests
from tqdm import tqdm
import openreview

# ======== 固定参数 ========

CONFERENCE = "ACL"
YEAR = 2025
TYPE = "oral"
USERNAME = "tusongjun2023@ia.ac.cn"
PASSWORD = "Tu2000112125"
OUT_DIR = "./downloads"
SLEEP = 0.35
SKIP_EXISTING = True

# ==========================

BASEURL = "https://api2.openreview.net" # V2 API
PDF_URL = "https://openreview.net/pdf?id={forum}"
REVIEW_HINTS = ("/Official_Review", "/Review", "/Meta_Review", "/Decision", "/Comment")
ATTACHMENT_FIELDS = ("pdf", "submission", "paper", "file", "source")

def login():
    return openreview.api.OpenReviewClient(baseurl=BASEURL, username=USERNAME, password=PASSWORD)

def extract_value(x):
    """OpenReview 字段统一取值：dict({'value': ...}) -> str；str -> 原样；其它 -> ''"""
    if isinstance(x, dict):
        return x.get("value", "")
    if isinstance(x, str):
        return x
    return ""

def normalize_content(d):
    """把 note.content 里常见的 value 包一层的字段拍平：{'decision': {'value':'Accept'}} -> {'decision':'Accept'}"""
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
    
def download_pdf(client, note, pdir: Path) -> bool:
    """下载 PDF；若已有非空文件则直接返回 True。"""
    pdf_path = pdir / "paper.pdf"
    if SKIP_EXISTING and exists_nonempty(pdf_path):
        return True

    # 优先尝试 attachment（成功率更高）
    for f in ATTACHMENT_FIELDS:
        try:
            data = client.get_attachment(note.id, f)
            if data:
                pdf_path.write_bytes(data)
                return True
        except Exception:
            pass

    # 回退到公开 pdf 地址
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
    从 r.invitations 中挑一个“非系统”的邀请名，并只返回最后一段标识：
    例如 '.../-/Official_Review' -> 'Official_Review'
    自动忽略 Edit/Withdraw/Revision 等。
    """
    bad = ("/-Edit", "/-Withdraw", "/-Revision", "/-Desk_Rejected", "/-Withdrawn")
    inv_list = getattr(r, "invitations", []) or []
    for inv in inv_list:
        if not any(b in inv for b in bad):
            return inv.split("/")[-1]
    return ""

def collect_reviews_metadata(client, forum_id: str):
    replies = client.get_all_notes(forum=forum_id)

    # 1) 索引所有节点（拍平 content、清洗 invitation）
    nodes = {}
    for r in replies:
        node = {
            "id": getattr(r, "id", None),
            "forum": getattr(r, "forum", None),
            "replyto": getattr(r, "replyto", None),
            "invitation": _clean_invitation_from(r),  # 只保留 'Official_Review' 这种短名
            "cdate": getattr(r, "cdate", None),
            "mdate": getattr(r, "mdate", None),
            "content": normalize_content(getattr(r, "content", {}) or {}),
            "children": []
        }
        nodes[node["id"]] = node

    # 2) 找 submission 根（通常 id == forum）
    submission = nodes.get(forum_id)
    if submission is None and replies:
        # 兜底：最早的一条当作 submission
        root = min(replies, key=lambda x: getattr(x, "cdate", 0))
        submission = nodes.get(getattr(root, "id", None))

    # 3) 建立子链
    for node in nodes.values():
        pid = node["replyto"]
        if pid and pid in nodes:
            nodes[pid]["children"].append(node)

    # 4) 分类（注意 invitation 已变成末段短名）
    decisions, meta_reviews = [], []
    for node in nodes.values():
        inv = node["invitation"]
        if inv == "Decision" or inv.endswith("Decision"):
            decisions.append(node)
        elif inv == "Meta_Review" or inv.endswith("Meta_Review"):
            meta_reviews.append(node)

    # 5) 只取“直接挂在 submission 上”的根，作为每条评审线程的起点
    if submission:
        thread_roots = list(submission["children"])
    else:
        # 没识别到 submission 时，退化为 replyto==forum 的一层
        thread_roots = [n for n in nodes.values() if n["replyto"] == forum_id]

    # 6) 排序：每棵树自顶向下按时间从早到晚
    def sort_rec(n):
        n["children"].sort(key=lambda x: (x.get("cdate") or 0))
        for ch in n["children"]:
            sort_rec(ch)

    for tr in thread_roots:
        sort_rec(tr)
    thread_roots.sort(key=lambda x: (x.get("cdate") or 0))

    # 7) 贴上友好类型标签（基于短名）
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

    # 8) 输出（不再包含 submission，以免重复；submission 的信息放 metadata.json）
    review_dict = {
        "forum": forum_id,
        "threads": thread_roots,
        "meta_reviews": meta_reviews,
        "decisions": decisions
    }

    # 构造 submission 的精简副本：移除 children 字段，保留其余信息
    submission_dict = {k: v for k, v in submission.items() if k != "children"}



    return review_dict, submission_dict

def main():
    venue_id = f"{CONFERENCE}/{YEAR}/Conference"
    out_root = Path(OUT_DIR).expanduser() / f"{CONFERENCE.split('.')[0]}_{YEAR}_{TYPE}"
    out_root.mkdir(parents=True, exist_ok=True)

    client = login()
    print(f"📥 拉取 {venue_id} 全量投稿中…")
    all_notes = get_all_notes(client, content={"venueid": venue_id})
    print(f"✅ 全量投稿数：{len(all_notes)} 篇")

    # 按类型过滤
    subset = filter_by_type(all_notes, TYPE)
    print(f"🎯 匹配类型 ({TYPE})：{len(subset)} 篇")

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
        if SKIP_EXISTING and exists_nonempty(pdf_path):
            pdf_ok = True
        else:
            pdf_ok = download_pdf(client, n, pdir)

        # 2) Reviews / Metadata
        # 如果两个文件都已存在且非空，就直接跳过抓取
        if SKIP_EXISTING and exists_nonempty(reviews_path) and exists_nonempty(metadata_path):
            with reviews_path.open("r", encoding="utf-8") as f:
                reviews_loaded = json.load(f)
            reviews_cnt = len(reviews_loaded.get("threads", []))
        else:
            reviews, metadata = collect_reviews_metadata(client, forum)
            reviews_path.write_text(json.dumps(reviews, indent=2, ensure_ascii=False), encoding="utf-8")
            metadata_path.write_text(json.dumps(metadata, indent=2, ensure_ascii=False), encoding="utf-8")
            reviews_cnt = len(reviews.get("threads", []))

        summary.append({"title": extract_value(title), "forum": forum, "pdf": pdf_ok, "reviews_count": reviews_cnt})
        time.sleep(SLEEP)

    (out_root / "summary.json").write_text(json.dumps(summary, indent=2, ensure_ascii=False), encoding="utf-8")
    print(f"\n✅ 下载完成：{len(summary)} 篇，结果保存在：{out_root.resolve()}")
    for s in summary[:3]:
        print(f"  - {s['title'][:80]} (pdf={s['pdf']}, reviews={s['reviews_count']})")

if __name__ == "__main__":
    main()
