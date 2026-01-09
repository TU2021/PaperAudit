#!/usr/bin/env python3
# -*- coding: utf-8 -*-

"""agents.py

Agent components for mas_error_detection.py (PaperAudit / MAS error detection).

This file intentionally contains the *agent-side* logic to keep the main runner short:
- Data structures: Task / Evidence / Finding
- Block normalization + context slicing
- Multimodal part builders
- Memory construction helpers
- Planner / Retriever / Specialist
- Global reviewer / Section reviewer
- Merger / Adjudicator

The main runner (mas_error_detection.py) should only orchestrate I/O, batching, and calling
these functions.
"""

from __future__ import annotations

import os
import re
import json
import base64
from dataclasses import dataclass, asdict
from pathlib import Path
from typing import Any, Dict, List, Optional, Tuple

from prompts import PromptTemplates
from utils import (
    call_llm_chat_with_empty_retries,
    call_web_search_via_tool,
    extract_json_from_text,
    load_json,
    save_json,
)

# =========================
# Global Configuration (agent-side defaults)
# =========================
DEFAULT_DETECT_MODEL = "qwen3-235b-a22b-instruct-2507"
DEFAULT_SYNTH_MODEL = "o4-mini"
OPENAI_API_KEY = os.getenv("OPENAI_API_KEY", None)

# Global constants (not from environment variables)
SEARCH_MAX_RESULTS = 5
SEARCH_TEMPERATURE = 0.0
GLOBAL_MAX_FINDINGS = 200
MEMORY_MAX_CHARS_PER_TASK = 20000
MAX_CTX_CHARS = 120000
EMPTY_RETRY_TIMES = 2

# LLM Temperature constants (global, not from environment)
TEMP_MEMORY = 0.2
TEMP_PLANNER = 0.2
TEMP_RETRIEVER = 0.4
TEMP_SPECIALIST = 0.4
TEMP_GLOBAL_REVIEW = 0.2
TEMP_SECTION_REVIEW = 0.2
TEMP_MERGER = 0.2


@dataclass
class Task:
    task_id: str
    section: str
    pages: List[int]
    risk_dimension: str
    hints: List[str]

@dataclass
class Evidence:
    span: str
    content_index: Optional[int] = None
    block_type: Optional[str] = None
    # Web search evidence metadata
    source_title: Optional[str] = None
    source_url: Optional[str] = None
    source_snippet: Optional[str] = None

@dataclass
class Finding:
    type: str
    section_location: str
    error_location: str
    explanation: str
    confidence: float
    proposed_fix: str
    id: Optional[int] = None



# =========================
# JSON Block Utilities
# =========================
def normalize_blocks(paper: dict) -> List[Dict[str, Any]]:
    raw = paper.get("content", []) or []
    blocks: List[Dict[str, Any]] = []
    for i, item in enumerate(raw):
        content_index = item.get("index", i)
        section_label = item.get("section", None)
        b: Dict[str, Any] = {
            "content_index": int(content_index) if isinstance(content_index, int) else i,
            "type": item.get("type"),
            "section": section_label if isinstance(section_label, str) and section_label.strip() else None,
        }
        if item.get("type") == "text":
            b["text"] = item.get("text", "")
        elif item.get("type") == "image_url":
            b["image_url"] = item.get("image_url")
        else:
            for k, v in item.items():
                if k not in b:
                    b[k] = v
        blocks.append(b)
    blocks.sort(key=lambda x: x.get("content_index", 0))
    return blocks

def cap_blocks_by_budget(blocks: List[Dict[str, Any]], max_chars: int) -> List[Dict[str, Any]]:
    out, cur = [], 0
    for b in blocks:
        cost = len(b.get("text") or "") if b.get("type") == "text" else 500
        if cur + cost > max_chars:
            break
        out.append(b); cur += cost
    return out

def llm_build_outline(blocks: List[Dict[str, Any]], paper_meta: str, dbgdir: Path, model: str) -> Dict[str, Any]:
    """基于 labels 的稳健 outline（无需 LLM 解析）"""
    outline_items: List[Dict[str, Any]] = []
    if not blocks:
        out = {"outline": [], "used": "no_blocks"}
        save_json(out, dbgdir / "outline.parsed.json")
        return out
    cur_title: Optional[str] = None
    cur_start: Optional[int] = None
    last_idx: Optional[int] = None
    def _flush():
        if cur_title is not None and cur_start is not None and last_idx is not None:
            outline_items.append({
                "title": cur_title,
                "start_index": int(cur_start),
                "end_index": int(last_idx),
                "aliases": [],
            })
    for b in blocks:
        sec = b.get("section")
        idx = b.get("content_index")
        if not isinstance(idx, int):
            continue
        if isinstance(sec, str) and sec.strip():
            if cur_title is None:
                cur_title = sec.strip(); cur_start = idx; last_idx = idx
            else:
                if sec.strip() == cur_title and last_idx is not None and idx == last_idx + 1:
                    last_idx = idx
                else:
                    _flush(); cur_title = sec.strip(); cur_start = idx; last_idx = idx
        else:
            if cur_title is not None and last_idx is not None and idx == last_idx + 1:
                last_idx = idx
    _flush()
    out = {"outline": outline_items, "used": "labels"}
    save_json(out, dbgdir / "outline.parsed.json")
    return out

def slice_json_for_task_with_outline(blocks: List[Dict[str, Any]], section_name: str, outline_items: List[Dict[str, Any]], max_chars: int) -> List[Dict[str, Any]]:
    """优先依据 outline 范围切片；否则退化为全文截断。"""
    name = (section_name or "").strip().lower()
    if name and outline_items:
        def match(item: Dict[str, Any]) -> bool:
            t = (item.get("title") or "").strip().lower()
            if not t: return False
            if name in t: return True
            for a in item.get("aliases") or []:
                if name in (a or "").strip().lower():
                    return True
            return False
        for item in outline_items:
            if match(item):
                start = max(0, int(item.get("start_index", 0)))
                end = int(item.get("end_index", start))
                rng_blocks = [
                    b for b in blocks
                    if isinstance(b.get("content_index"), int)
                    and start <= b["content_index"] <= end
                ]
                return cap_blocks_by_budget(rng_blocks, max_chars)
    return cap_blocks_by_budget(blocks, max_chars)

def _guess_mime_from_b64(data_b64: str) -> str:
    head = (data_b64 or "")[:20]
    if head.startswith("iVBOR"): return "image/png"
    if head.startswith("/9j/"):  return "image/jpeg"
    if head.startswith("R0lGOD"): return "image/gif"
    return "image/png"

def _block_to_parts_minimal(
    b: Dict[str, Any],
    max_text_chars_per_block: int = 8000,
    enable_mm: bool = True,
) -> List[Dict[str, Any]]:
    parts: List[Dict[str, Any]] = []
    ci = b.get("content_index", b.get("index"))
    if isinstance(ci, str):
        try: ci = int(ci)
        except: ci = None
    sec = (b.get("section") or "").strip()
    typ = b.get("type")
    header = f"[Block #{ci if ci is not None else '?'} | {typ or 'unknown'}{(' |Section: ' + sec) if sec else 'None'}]"
    parts.append({"type": "text", "text": header})

    if typ == "text":
        t = (b.get("text") or "")
        if max_text_chars_per_block and len(t) > max_text_chars_per_block:
            t = t[:max_text_chars_per_block]
        if t.strip():
            parts.append({"type": "text", "text": t})

    elif typ == "image_url":
        # If multimodal disabled, omit images and use text placeholder
        if not enable_mm:
            parts.append({"type": "text", "text": "[Image omitted: multimodal disabled]"})
            return parts

        img = b.get("image_url")
        url = None
        if isinstance(img, str):
            url = img
        elif isinstance(img, dict):
            if isinstance(img.get("url"), str):
                url = img["url"]
            elif isinstance(img.get("data_b64"), str):
                mime = img.get("mime") or _guess_mime_from_b64(img["data_b64"])
                url = f"data:{mime};base64,{img['data_b64']}"
        if isinstance(url, str) and (url.startswith("data:") or url.startswith("http")):
            parts.append({"type": "image_url", "image_url": {"url": url}})

    return parts


def build_multimodal_parts_from_blocks(
    blocks: List[Dict[str, Any]],
    max_blocks: int = 1_000,
    max_images: int = 48,
    max_text_chars_per_block: int = 10_000,
    enable_mm: bool = True,
) -> List[Dict[str, Any]]:
    parts: List[Dict[str, Any]] = []
    img_count = 0
    for b in blocks[:max_blocks]:
        sub = _block_to_parts_minimal(
            b,
            max_text_chars_per_block=max_text_chars_per_block,
            enable_mm=enable_mm
        )
        for p in sub:
            if p.get("type") == "image_url":
                if not enable_mm:
                    continue
                if img_count >= max_images:
                    continue
                img_count += 1
            parts.append(p)
    return parts


# =========================
# Memory (natural language) construction and injection
# =========================
def extract_section_titles_from_blocks(blocks: List[Dict[str, Any]]) -> List[str]:
    """从论文 blocks 顺序去重抽取非空 section 名称。"""
    seen = set()
    titles = []
    for b in blocks:
        s = (b.get("section") or "").strip()
        if not s:
            continue
        key = s.lower()
        if key in seen:
            continue
        seen.add(key)
        titles.append(s)
    return titles

def build_memory_messages(blocks: List[Dict[str, Any]], enable_mm: bool) -> List[Dict[str, Any]]:
    section_titles = extract_section_titles_from_blocks(cap_blocks_by_budget(blocks, MAX_CTX_CHARS))
    lead_text = PromptTemplates.memory_user(section_titles)
    parts = [{"type": "text", "text": lead_text}]
    parts.extend(build_multimodal_parts_from_blocks(
        cap_blocks_by_budget(blocks, MAX_CTX_CHARS),
        enable_mm=enable_mm
    ))
    return [
        {"role": "system", "content": PromptTemplates.memory_system()},
        {"role": "user", "content": parts},
    ]


def build_paper_memory(blocks: List[Dict[str, Any]], dbgdir: Path, model: str, enable_mm: bool, temperature: float = TEMP_MEMORY) -> Tuple[Optional[Dict[str, Any]], Dict[str, Any]]:
    """
    产出自然语言记忆：
    memory_obj = {
        "raw_text": "<plain text memory>",
        "section_titles": [...],
    }
    """
    def _messages(_):
        return build_memory_messages(blocks, enable_mm=enable_mm)
    raw = call_llm_chat_with_empty_retries(
        _messages,
        model=model,
        max_tokens=32768,
        tag="memory_build",
        dbgdir=dbgdir,
        expect_key=None,
        temperature=temperature,
        api_key=OPENAI_API_KEY,
    )
    meta = {"memory_parse_ok": False, "error": None, "mode": "natural_language"}
    if not raw or not raw.strip():
        meta["error"] = "memory_build_empty_raw"
        return None, meta

    out_path = dbgdir / "memory.fullpaper.txt"
    out_path.parent.mkdir(parents=True, exist_ok=True
    )
    out_path.write_text(raw, encoding="utf-8")

    section_titles = extract_section_titles_from_blocks(blocks)
    mem_obj = {"raw_text": raw, "section_titles": section_titles}
    save_json({"section_titles": section_titles, "bytes": len(raw.encode('utf-8'))}, dbgdir / "memory.meta.summary.json")
    meta["memory_parse_ok"] = True
    return mem_obj, meta

def _split_memory_by_headings(raw_text: str) -> Dict[str, str]:
    sections = {}
    if not raw_text:
        return sections
    lines = raw_text.splitlines()
    cur_title = None
    buf = []
    def _flush():
        nonlocal cur_title, buf
        if cur_title is not None:
            sections[cur_title] = "\n".join(buf).strip()
        buf = []
    for ln in lines:
        if ln.startswith("# "):
            _flush()
            cur_title = ln[2:].strip()
        else:
            buf.append(ln)
    _flush()
    return sections

def build_memory_for_task(memory_obj: Optional[Dict[str, Any]], current_section: str, max_chars: int) -> str:
    """
    Returns natural language slice: '# Global Summary' + all other sections (excluding current_section).
    若没有结构化标题，退化为截断 raw_text。
    """
    if not memory_obj or not memory_obj.get("raw_text"):
        return ""
    raw = memory_obj["raw_text"]
    parts = _split_memory_by_headings(raw)
    pieces: List[str] = []

    if "Global Summary" in parts and parts["Global Summary"]:
        pieces.append("# Global Summary")
        pieces.append(parts["Global Summary"])

    cur = (current_section or "").strip()
    for title in memory_obj.get("section_titles", []):
        if title == cur:
            continue
        if title in parts and parts[title]:
            pieces.append(f"# {title}")
            pieces.append(parts[title])

    final_text = "\n".join(pieces).strip() or raw
    if max_chars and len(final_text) > max_chars:
        final_text = final_text[:max_chars]
    return final_text

# =========================
# Planner
# =========================
def build_planner_messages_multimodal(
    abstract_blocks: List[Dict[str, Any]],
    outline_obj: Dict[str, Any],
    enable_mm: bool,
) -> List[Dict[str, Any]]:
    allowed_risks = PromptTemplates.RISK_DIMENSIONS
    lead_text = PromptTemplates.planner_user(outline_obj, allowed_risks)
    lead_parts = [{"type": "text", "text": lead_text}]
    abstract_parts = build_multimodal_parts_from_blocks(abstract_blocks, enable_mm=enable_mm)
    return [
        {"role": "system", "content": PromptTemplates.planner_system()},
        {"role": "user", "content": lead_parts + abstract_parts}
    ]


def planner_build_tasks_mm(blocks: List[Dict[str, Any]], outline_obj: Dict[str, Any], dbgdir: Path, model: str, enable_mm: bool, temperature: float = TEMP_PLANNER) -> Tuple[List[Task], Dict[str, Any]]:
    def _messages(_):
        return build_planner_messages_multimodal(blocks, outline_obj, enable_mm=enable_mm)
    raw = call_llm_chat_with_empty_retries(
        _messages,
        model=model,
        max_tokens=32768,
        tag="planner",
        dbgdir=dbgdir,
        expect_key="tasks",
        temperature=temperature,
        api_key=OPENAI_API_KEY,
    )
    tasks: List[Task] = []
    meta = {"planner_parse_ok": False, "error": None, "used": "llm_mm"}
    if raw:
        try:
            obj = extract_json_from_text(raw)
            save_json(obj, dbgdir / "planner.parsed.json")
            for i, t in enumerate(obj.get("tasks", []) or [], 1):
                risk = t.get("risk_dimension", "other")
                if risk not in PromptTemplates.RISK_DIMENSIONS:
                    risk = "other"
                hints = t.get("hints", [])
                if isinstance(hints, dict):
                    hints = [str(v) for v in hints.values()]
                tasks.append(Task(
                    task_id=t.get("task_id", f"task_{i:03d}"),
                    section=t.get("section", f"Section {i}"),
                    pages=t.get("pages", []),
                    risk_dimension=risk,
                    hints=hints if isinstance(hints, list) else []
                ))
            if tasks:
                meta["planner_parse_ok"] = True
        except Exception as e:
            meta["error"] = f"planner_parse_fail: {e}"
            try:
                save_json({"raw": raw}, dbgdir / "planner.raw.json")
            except Exception:
                pass
    if not tasks:
        meta["used"] = "llm_mm_no_tasks"
        save_json(meta, dbgdir / "planner.meta.error.json")
        raise RuntimeError("[PLANNER ERROR] No tasks produced by planner after retries.")
    save_json([asdict(t) for t in tasks], dbgdir / "planner.tasks.json")
    return tasks, meta

# =========================
# Retriever（paper-only / web-enabled）
# =========================
def _build_retriever_messages_multimodal(
    task: Task,
    doc_blocks: List[Dict[str, Any]],
    paper_only: bool,
    memory_slice: Optional[str],
    enable_mm: bool = True,
) -> List[Dict[str, Any]]:
    user_payload = {
        "task": asdict(task),
        "doc_blocks_hint": (
            "Below are the paper blocks in order. "
            + ("Extract exhaustive but non-redundant paper_evidence."
               if paper_only
               else "Extract exhaustive but non-redundant paper_evidence; emit web_queries for genuine uncertainties.")
        )
    }
    lead_parts = [
        {"type": "text", "text": "MEMORY summary of the full paper:\n" + (memory_slice or "")},
        {"type": "text", "text": "INPUT(JSON, task):\n" + json.dumps(user_payload, ensure_ascii=False)}
    ]
    block_parts = build_multimodal_parts_from_blocks(doc_blocks, enable_mm=enable_mm)
    return [
        {
            "role": "system",
            "content": (
                PromptTemplates.retriever_system_paper_only()
                if paper_only else
                PromptTemplates.retriever_system()
            ),
        },
        {"role": "user", "content": lead_parts + block_parts},
    ]

def retriever_extract_and_questions(
    task: Task,
    slice_blocks: List[Dict[str, Any]],
    dbgdir: Path,
    model: str,
    paper_only: bool,
    memory_slice: Optional[str],
    enable_mm: bool = True,
    temperature: float = TEMP_RETRIEVER,
) -> Tuple[List[Evidence], List[Dict[str, str]], Dict[str, Any]]:
    """
    执行 Retriever：
      - 返回 paper_evidence: List[Evidence]
      - 返回 web_queries: List[{'q','why'}]  (paper_only=True 时恒为空)
    """
    save_json(slice_blocks, dbgdir / f"task_{task.task_id}.doc_slice.json")

    def _messages(_):
        return _build_retriever_messages_multimodal(
            task=task, doc_blocks=slice_blocks, paper_only=paper_only, memory_slice=memory_slice, enable_mm=enable_mm
        )

    raw = call_llm_chat_with_empty_retries(
        _messages,
        model=model,
        max_tokens=32768,
        tag=f"task_{task.task_id}.retriever",
        dbgdir=dbgdir,
        expect_key="paper_evidence",
        temperature=temperature,
        api_key=OPENAI_API_KEY,
    )

    paper_e: List[Evidence] = []
    web_q: List[Dict[str, str]] = []
    meta = {"retriever_parse_ok": False, "error": None, "paper_only": paper_only}

    if raw:
        try:
            obj = extract_json_from_text(raw or "{}")
            save_json(obj, dbgdir / f"task_{task.task_id}.retriever.parsed.json")

            for e in obj.get("paper_evidence", []) or []:
                if not isinstance(e, dict):
                    continue
                span = (e.get("span") or "").strip()
                if not span:
                    continue
                paper_e.append(Evidence(
                    span=span,
                    content_index=e.get("content_index"),
                    block_type=e.get("block_type")
                ))

            if not paper_only:
                for q in obj.get("web_queries", []) or []:
                    if not isinstance(q, dict):
                        continue
                    qtext = (q.get("q") or "").strip()
                    why   = (q.get("why") or "").strip()
                    if qtext:
                        web_q.append({"q": qtext, "why": why})

            meta["retriever_parse_ok"] = True
        except Exception as e:
            meta["error"] = f"retriever_parse_fail: {e}"
            try:
                save_json({"raw": raw}, dbgdir / f"task_{task.task_id}.retriever.raw.json")
            except Exception:
                pass

    save_json([asdict(e) for e in paper_e], dbgdir / f"task_{task.task_id}.retriever.paper_evidence.json")
    if not paper_only:
        save_json(web_q, dbgdir / f"task_{task.task_id}.retriever.web_queries.json")
    else:
        save_json({"web_queries": [], "note": "web_search disabled; using paper-only retriever prompt."},
                  dbgdir / f"task_{task.task_id}.retriever.web_queries.json")

    return paper_e, web_q, meta

def perform_web_search_for_queries(
    web_queries: List[Dict[str, str]],
    dbgdir: Path,
    detect_model: str,
    max_results: int = SEARCH_MAX_RESULTS,
    temperature: float = SEARCH_TEMPERATURE,
) -> List[Evidence]:
    """
    执行联网检索（Step-B）：
      - 输入：Retriever 产出的简短可检索问句列表 [{'q','why'}...]
      - 过程：使用 Responses API 的 tools=[{'type':'web_search'}] 调用（见 call_web_search_via_tool）
      - 输出：将答案转为 Evidence（block_type='web'），不强制包含 URL（与上游兼容）
    说明：
      - max_results 只是技术上限制本次最多处理多少 query，避免目录爆炸。
      - 由于官方 web_search 返回可能不含 URL，本函数将 answer 作为 Evidence.span 保存，
        并把原 query 一并写入 span 方便审阅。
    """
    dbgdir.mkdir(parents=True, exist_ok=True)
    # 1) Normalize/deduplicate/trim queries
    qlist: List[str] = []
    for item in web_queries or []:
        if not isinstance(item, dict):
            continue
        q = (item.get("q") or "").strip()
        if q and q not in qlist:
            qlist.append(q)
        if len(qlist) >= max_results:
            break

    save_json({"input_web_queries": web_queries, "used_queries": qlist}, dbgdir / "web_queries.input.json")
    if not qlist:
        save_json({"answers": [], "note": "no queries"}, dbgdir / "web_queries.answers.json")
        return []

    # 2) Call web search tool
    answers = call_web_search_via_tool(
        queries=qlist,
        model=detect_model,
        temperature=temperature,
        api_key=OPENAI_API_KEY,
    )
    save_json({"answers": answers}, dbgdir / "web_queries.answers.json")

    # 3) Convert to Evidence (block_type='web', does not depend on URL field)
    evidences: List[Evidence] = []
    for a in answers or []:
        if not isinstance(a, dict):
            continue
        q = (a.get("query") or "").strip()
        ans = (a.get("answer") or "").strip()
        if not ans:
            continue
        span_text = f"[WEB] Q: {q}\nA: {ans}" if q else f"[WEB] {ans}"
        evidences.append(Evidence(
            span=span_text,
            content_index=None,
            block_type="web",
            source_title=None,
            source_url=None,
            source_snippet=ans
        ))

    save_json([asdict(e) for e in evidences], dbgdir / "web_queries.evidence.json")
    return evidences

# =========================
# Specialist / Global / Section / Merger
# =========================
def _build_specialist_messages_multimodal(
    task: Task,
    evidence_list: List[Evidence],
    neighbor_blocks: List[Dict[str, Any]],
    paper_only: bool,
    memory_slice: Optional[str],
    prior_findings: Optional[List[Finding]] = None,
    use_retriever: bool = True,
    enable_mm: bool = True,
) -> List[Dict[str, Any]]:
    def _finding_to_minimal_dict(f: Finding) -> Dict[str, Any]:
        return {
            "id": f.id,
            "type": f.type,
            "section_location": f.section_location,
            "error_location": f.error_location,
            "explanation": f.explanation,
            "confidence": float(f.confidence),
            "proposed_fix": f.proposed_fix,
        }

    payload = {
        "task": asdict(task),
        "evidence": [asdict(e) for e in evidence_list],
        "neighbor_hint": "Local neighborhood blocks for context.",
        "prior_findings": [_finding_to_minimal_dict(f) for f in (prior_findings or [])],
        "retriever_used": bool(use_retriever),
        "memory_text": memory_slice or "",
    }
    lead_parts = [
        {"type": "text", "text": "INPUT(JSON):\n" + json.dumps(payload, ensure_ascii=False)}
    ]
    neighbor_parts = build_multimodal_parts_from_blocks(neighbor_blocks, enable_mm=enable_mm)
    return [
        {
            "role": "system",
            "content": PromptTemplates.specialist_system_unified(task.risk_dimension),
        },
        {"role": "user", "content": lead_parts + neighbor_parts},
    ]

def specialist_review(
    task: Task,
    evid: List[Evidence],
    neighbor_blocks: List[Dict[str, Any]],
    dbgdir: Path,
    model: str,
    paper_only: bool,
    memory_slice: Optional[str],
    prior_findings: Optional[List[Finding]] = None,
    use_retriever: bool = True,
    enable_mm: bool = True,
    temperature: float = TEMP_SPECIALIST,
) -> Tuple[List[Finding], Dict[str, Any]]:
    messages = _build_specialist_messages_multimodal(
        task=task,
        evidence_list=evid,
        neighbor_blocks=neighbor_blocks,
        paper_only=paper_only,
        memory_slice=memory_slice,
        prior_findings=prior_findings,
        use_retriever=use_retriever,
        enable_mm=enable_mm
    )
    raw = call_llm_chat_with_empty_retries(
        messages,
        model=model,
        max_tokens=32768,
        tag=f"task_{task.task_id}.specialist_{task.risk_dimension}",
        dbgdir=dbgdir,
        expect_key="findings",
        temperature=temperature,
        api_key=OPENAI_API_KEY,
    )

    findings: List[Finding] = []
    meta = {"specialist_parse_ok": False, "error": None, "paper_only": paper_only, "used_retriever": bool(use_retriever)}

    if raw:
        try:
            obj = extract_json_from_text(raw)
            save_json(obj, dbgdir / f"task_{task.task_id}.specialist_{task.risk_dimension}.parsed.json")
            items = obj.get("findings", [])
            if isinstance(items, dict):
                items = [items]
            if not isinstance(items, list):
                items = []
            for f in items:
                if not isinstance(f, dict):
                    continue
                sec = (f.get("section_location") or "").strip()
                err = (f.get("error_location") or "").strip()
                exp = (f.get("explanation") or "").strip()
                if not sec or not exp or not err:
                    continue
                try:
                    conf = float(f.get("confidence", 0.5))
                except Exception:
                    conf = 0.5
                conf = max(0.0, min(conf, 1.0))
                findings.append(Finding(
                    type=f.get("type", "other"),
                    section_location=sec,
                    error_location=err,
                    explanation=exp,
                    confidence=conf,
                    proposed_fix=(f.get("proposed_fix") or "").strip()
                ))
            meta["specialist_parse_ok"] = True
        except Exception as e:
            meta["error"] = f"specialist_parse_fail: {e}"
            try:
                save_json({"raw": raw}, dbgdir / f"task_{task.task_id}.specialist_{task.risk_dimension}.raw.json")
            except Exception:
                pass
    save_json([asdict(f) for f in findings], dbgdir / f"task_{task.task_id}.specialist_{task.risk_dimension}.findings.json")
    return findings, meta

def build_global_user_parts(blocks: List[Dict[str, Any]], global_max_findings:int, enable_mm: bool) -> List[Dict[str, Any]]:
    lead = {"type": "text","text":("- Prioritize cross-section contradictions and support mismatches.\n"
                                   "- Quote exact spans in 'error_location' when possible.\n"
                                   "- No hard caps; be exhaustive yet non-redundant.\n")}
    parts = [lead]
    parts.extend(build_multimodal_parts_from_blocks(
        cap_blocks_by_budget(blocks, MAX_CTX_CHARS),
        max_blocks=2000, max_images=64, max_text_chars_per_block=10_000, enable_mm=enable_mm
    ))
    return parts

def global_cross_section_review(blocks: List[Dict[str, Any]], dbgdir: Path, model: str, global_max_findings:int, enable_mm: bool, temperature: float = TEMP_GLOBAL_REVIEW) -> Tuple[List[Finding], Dict[str, Any]]:
    def _messages(_):
        return [
            {"role": "system", "content": PromptTemplates.global_system()},
            {"role": "user", "content": build_global_user_parts(blocks, global_max_findings, enable_mm)},
        ]

    raw = call_llm_chat_with_empty_retries(
        _messages,
        model=model,
        max_tokens=32768,
        tag="global_review",
        dbgdir=dbgdir,
        expect_key="findings",
        temperature=temperature,
        api_key=OPENAI_API_KEY,
    )
    findings: List[Finding] = []
    meta = {"global_parse_ok": False, "error": None, "used": "llm_mm_global"}

    if raw:
        try:
            obj = extract_json_from_text(raw)
            save_json(obj, dbgdir / "global_review.parsed.json")
            items = obj.get("findings", [])
            if isinstance(items, dict): items = [items]
            if not isinstance(items, list): items = []
            for f in items:
                if not isinstance(f, dict): continue
                sec = (f.get("section_location") or "").strip()
                err = (f.get("error_location") or "").strip()
                exp = (f.get("explanation") or "").strip()
                if not (sec and err and exp): continue
                try: conf = float(f.get("confidence", 0.5))
                except Exception: conf = 0.5
                conf = max(0.0, min(conf, 1.0))
                findings.append(Finding(
                    type=(f.get("type") or "other").strip() or "other",
                    section_location=sec,
                    error_location=err,
                    explanation=exp,
                    confidence=conf,
                    proposed_fix=(f.get("proposed_fix") or "").strip()
                ))
            meta["global_parse_ok"] = True
        except Exception as e:
            meta["error"] = f"global_parse_fail: {e}"
            try: save_json({"raw": raw}, dbgdir / "global_review.raw.json")
            except Exception: pass
    save_json([asdict(f) for f in findings], dbgdir / "global_review.findings.json")
    return findings, meta

# ---- Section Review ----
def build_section_user_parts(
    section_blocks: List[Dict[str, Any]],
    section_title: str,
    memory_slice: Optional[str],
    enable_mm: bool = True,
) -> List[Dict[str, Any]]:
    lead_parts = []
    mem_text = memory_slice or ""
    lead_parts.append({"type": "text", "text": f"MEMORY summary of the full paper:\n{mem_text}"})
    lead_parts.append({"type": "text", "text": f"- Target section: {section_title}\n- Be exhaustive but non-redundant within this section.\n"})
    lead_parts.extend(build_multimodal_parts_from_blocks(
        cap_blocks_by_budget(section_blocks, MAX_CTX_CHARS),
        max_blocks=1500, max_images=48, max_text_chars_per_block=10_000, enable_mm=enable_mm
    ))
    return lead_parts

def section_level_review(
    section_title: str,
    section_blocks: List[Dict[str, Any]],
    dbgdir: Path,
    model: str,
    memory_slice: Optional[str],
    enable_mm: bool = True,
    temperature: float = TEMP_SECTION_REVIEW,
) -> Tuple[List[Finding], Dict[str, Any]]:
    def _messages(_):
        return [
            {"role": "system", "content": PromptTemplates.section_system()},
            {"role": "user", "content": build_section_user_parts(section_blocks, section_title, memory_slice, enable_mm)},
        ]
    raw = call_llm_chat_with_empty_retries(
        _messages,
        model=model,
        max_tokens=32768,
        tag=f"section_review.{re.sub(r'[^A-Za-z0-9._-]+','_', section_title or 'Unknown')}",
        dbgdir=dbgdir,
        expect_key="findings",
        temperature=temperature,
        api_key=OPENAI_API_KEY,
    )

    findings: List[Finding] = []
    meta = {"section": section_title, "parse_ok": False, "error": None}
    if raw:
        try:
            obj = extract_json_from_text(raw)
            save_json(obj, dbgdir / "section_review.parsed.json")
            items = obj.get("findings", [])
            if isinstance(items, dict): items = [items]
            if not isinstance(items, list): items = []
            for f in items:
                if not isinstance(f, dict): continue
                sec = (f.get("section_location") or "").strip() or section_title
                err = (f.get("error_location") or "").strip()
                exp = (f.get("explanation") or "").strip()
                if not (sec and err and exp): continue
                try: conf = float(f.get("confidence", 0.5))
                except Exception: conf = 0.5
                conf = max(0.0, min(conf, 1.0))
                findings.append(Finding(
                    type=(f.get("type") or "other").strip() or "other",
                    section_location=sec,
                    error_location=err,
                    explanation=exp,
                    confidence=conf,
                    proposed_fix=(f.get("proposed_fix") or "").strip()
                ))
            meta["parse_ok"] = True
        except Exception as e:
            meta["error"] = f"section_parse_fail: {e}"
            try: save_json({"raw": raw}, dbgdir / "section_review.raw.json")
            except Exception: pass

    save_json([asdict(x) for x in findings], dbgdir / "section_review.findings.json")
    return findings, meta

def build_merger_user(all_findings: List[Finding]) -> str:
    def finding_to_dict(f: Finding) -> Dict[str, Any]:
        return {
            "type": f.type,
            "section_location": f.section_location,
            "error_location": f.error_location,
            "explanation": f.explanation,
            "confidence": float(f.confidence),
            "proposed_fix": f.proposed_fix,
        }
    payload = {"candidates": [finding_to_dict(f) for f in all_findings]}
    return "INPUT(JSON):\n" + json.dumps(payload, ensure_ascii=False)

def merge_and_adjudicate(all_findings: List[Finding], dbgdir: Path, model: str, temperature: float = TEMP_MERGER) -> List[Finding]:
    save_json([asdict(f) for f in all_findings], dbgdir / "merge.input_findings.json")
    if not all_findings:
        save_json([], dbgdir / "merge.output_findings.json")
        print(f"[DEBUG][merge] input empty; total=0")
        return []

    sys_msg = {"role": "system", "content": PromptTemplates.merger_system()}
    user_msg = {"role": "user", "content": build_merger_user(all_findings)}

    raw = call_llm_chat_with_empty_retries(
        [sys_msg, user_msg],
        model=model,
        max_tokens=50000,
        tag="merge",
        dbgdir=dbgdir,
        temperature=temperature,
        expect_key="findings",
        retries=EMPTY_RETRY_TIMES,
        api_key=OPENAI_API_KEY,
    )

    merged: List[Finding] = []
    fallback = False
    try:
        obj = extract_json_from_text(raw or "")
        save_json(obj, dbgdir / "merge.llm.parsed.json")
        items = obj.get("findings", [])
        if isinstance(items, dict): items = [items]
        if not isinstance(items, list): items = []
        for i, f in enumerate(items, start=1):
            if not isinstance(f, dict): continue
            ftype = (f.get("type") or "").strip()
            sec = (f.get("section_location") or "").strip()
            err = (f.get("error_location") or "").strip()
            exp = (f.get("explanation") or "").strip()
            pfix = (f.get("proposed_fix") or "").strip()
            if not (ftype and sec and err and exp): continue
            try: conf = float(f.get("confidence", 0.5))
            except Exception: conf = 0.5
            conf = max(0.0, min(conf, 1.0))
            merged.append(Finding(
                id=i, type=ftype, section_location=sec, error_location=err,
                explanation=exp, confidence=conf, proposed_fix=pfix,
            ))
    except Exception as e:
        save_json({"merge_parse_error": f"{type(e).__name__}: {e}"}, dbgdir / "merge.parse_error.json")
        merged = []

    if not merged:
        save_json({"fallback": True, "reason": "merge_failed_or_empty"}, dbgdir / "merge.fallback.json")
        merged = list(all_findings)
        fallback = True

    if fallback:
        print(f"[DEBUG][merge] merge failed, fallback to original findings; total={len(merged)}")
    else:
        print(f"[DEBUG][merge] merged {len(merged)} findings (from {len(all_findings)}); total={len(merged)}")

    save_json([asdict(f) for f in merged], dbgdir / "merge.output_findings.json")
    return merged

# =========================
# Main pipeline (single file)
# =========================
