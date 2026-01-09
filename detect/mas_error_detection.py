#!/usr/bin/env python3
# -*- coding: utf-8 -*-

"""
MAS (Multi-Agent System) Error Detection Pipeline

A comprehensive error detection system that uses multiple LLM agents to identify errors in academic papers.
The system supports various detection modes (fast, standard, deep) with different feature combinations.

Pipeline Components:
1. Memory Builder: Creates natural-language memory of the full paper
2. Global Review: Cross-section error detection across the entire paper
3. Section Review: Section-level error detection before task planning
4. Planner: Generates focused tasks for specific error types and sections
5. Retriever: Extracts relevant evidence and generates web search queries
6. Web Search: Performs web searches for additional evidence
7. Specialist: Reviews each task with context, evidence, and memory
8. Merger: Merges and adjudicates findings from different sources

Usage Examples:

    # Fast detection mode
    python mas_error_detection.py \\
        --root_dir /path/to/papers \\
        --detect_mode fast \\
        --detect_model o4-mini \\
        --synth_model gpt-5-2025-08-07 \\
        --enable-global-review \\
        --enable-mm \\
        --jobs 10

    # Standard detection mode
    python mas_error_detection.py \\
        --root_dir /path/to/papers \\
        --detect_mode standard \\
        --detect_model o4-mini \\
        --synth_model gpt-5-2025-08-07 \\
        --enable-memory-build \\
        --enable-section-review \\
        --enable-mm \\
        --jobs 10

    # Deep detection mode
    # - All features enabled: global review, section review, per-task, memory, retriever, web search, merge
    python mas_error_detection.py \\
        --root_dir /path/to/papers \\
        --detect_mode deep \\
        --detect_model o4-mini \\
        --synth_model gpt-5-2025-08-07 \\
        --enable-memory-build \\
        --enable-section-review \\
        --enable-per-task \\
        --enable-retriever \\
        --enable-memory-injection \\
        --enable-merge \\
        --enable-mm \\
        --jobs 10

    # Single file detection
    python mas_error_detection.py \\
        --synth_json /path/to/paper_synth_xxx.json \\
        --detect_model o4-mini \\
        --detect_mode standard \\
        --enable-per-task \\
        --enable-global-review \\
        --enable-mm \\

Key Arguments:
    
    Input/Output:
    --root_dir: Root directory for batch processing (contains paper_synth_*.json files). Default: /mnt/parallel_ssd/home/zdhs0006/ACL/test2
    --synth_json: Single file mode - path to a single synth JSON
    --out_json: (Single mode) Explicit output json path. If omitted, auto-placed into detect/<detect_mode>_detect/...
    --debug_dir: (Single mode) Explicit debug dir. If omitted, auto-placed into detect/<detect_mode>_detect_log/...
    --synth_glob: Glob pattern to find synth JSONs under root_dir. Default: paper_synth_*.json
    --overwrite: Force overwrite existing outputs. Default: False
    --overwrite_zero: If output exists but has empty findings=[], rerun even when not overwriting. Default: False
    --max_papers: Limit the number of synth files to process. Default: None (no limit)
    
    Model and Mode:
    --detect_model: Model tag used for detection. Default: qwen3-235b-a22b-instruct-2507
        Available models: gpt-5-2025-08-07, o4-mini, gemini-2.5-pro, claude-sonnet-4-5-20250929, 
        grok-4, doubao-seed-1-6-251015, glm-4.6, deepseek-v3.1, kimi-k2-250905, 
        qwen3-235b-a22b-instruct-2507, qwen3-vl-235b-a22b-instruct
    --detect_mode: Mode name used in output paths (e.g., fast, standard, deep). Default: test
    --synth_model: Tag/name of the synthesis model whose data you are evaluating. Default: o4-mini
        Available models: gpt-5-2025-08-07, o4-mini, gemini-2.5-pro, claude-sonnet-4-5-20250929,
        grok-4, qwen3-vl-235b-a22b-instruct, doubao-seed-1-6-251015, glm-4.6
    
    Batch Processing:
    --jobs: Thread pool size for parallel processing. Default: 10
    
    Feature Flags (all default to False, must be explicitly enabled):
    --enable-global-review: Enable global cross-section review
    --global_max_findings: Soft cap for global review (for safety). Default: 200
    --enable-section-review: Enable section-level review
    --enable-per-task: Enable per-task pipeline (Planner → Retriever → Specialist)
    --enable-section-findings-as-prior: Pass section_review findings of the SAME section to each task's Specialist as prior support
    --enable-retriever: Enable Retriever (+WebSearch). Specialist will use retrieved evidence
    --enable-memory-injection: Inject memory into Specialist (requires --enable-memory-build)
    --enable-merge: Enable merging/adjudication of findings
    --enable-web-search: Enable dedicated web_search step after Retriever
    --search_max_results: Technical cap to avoid directory explosion. Default: 5
    --search_temperature: Temperature for web search LLM calls. Default: 0.0
    --enable-memory-build: Enable full-paper memory build
    --memory_max_chars: Max characters of memory injected per task (natural-language slice). Default: 20000
    --enable-mm: Enable multimodal input (images). Default: True
    
    Context and Retry Settings:
    --max_ctx_chars: Maximum context characters for LLM input. Default: 120000
    --empty_retry_times: Number of retries for empty output/parse failures. Default: 2
    
    LLM Temperature Settings:
    --temp_memory: Temperature for memory building LLM calls. Default: 0.2
    --temp_planner: Temperature for planner LLM calls. Default: 0.2
    --temp_retriever: Temperature for retriever LLM calls. Default: 0.4
    --temp_specialist: Temperature for specialist LLM calls. Default: 0.4
    --temp_global_review: Temperature for global review LLM calls. Default: 0.2
    --temp_section_review: Temperature for section review LLM calls. Default: 0.2
    --temp_merger: Temperature for merger LLM calls. Default: 0.2

Output:
    - Detection results: <paper_dir>/detect/<detect_mode>_detect/<detect_model>/<synth_stem>/<detect_mode>_detect.json
    - Debug logs: <paper_dir>/detect/<detect_mode>_detect_log/<detect_model>/<synth_stem>/
"""

from __future__ import annotations
import argparse, json, os, re, sys, time, traceback
from dataclasses import dataclass, asdict
from datetime import datetime
from pathlib import Path
from typing import Any, Callable, Dict, List, Optional, Tuple, Union
from concurrent.futures import ThreadPoolExecutor, as_completed
from mas_reference_helper import enrich_section_blocks_with_local_references

try:
    from tqdm import tqdm  # noqa
except Exception:
    tqdm = None

from utils import (
    call_llm_chat_with_empty_retries,
    call_web_search_via_tool,
    extract_json_from_text,
    load_json,
    save_json,
)
from prompts import PromptTemplates
# === Agent components are moved to agents.py to keep this runner short ===
from agents import (
    Task,
    Evidence,
    Finding,
    normalize_blocks,
    cap_blocks_by_budget,
    llm_build_outline,
    slice_json_for_task_with_outline,
    build_paper_memory,
    build_memory_for_task,
    planner_build_tasks_mm,
    retriever_extract_and_questions,
    perform_web_search_for_queries,
    specialist_review,
    global_cross_section_review,
    section_level_review,
    merge_and_adjudicate,
)


# =========================
# Global Configuration
# =========================
OPENAI_API_KEY = os.getenv("OPENAI_API_KEY", None)

def output_has_zero_findings(out_json: Path) -> bool:
    """
    Return True iff out_json exists and parses, and contains findings == [].
    """
    try:
        if not out_json.exists():
            return False
        obj = load_json(out_json)
        findings = obj.get("findings", None)
        return isinstance(findings, list) and len(findings) == 0
    except Exception:
        return False


# =========================
# Data Structures
# =========================
def _extract_ground_truth_from_synth(obj: dict) -> List[Dict[str, Any]]:
    gt: List[Dict[str, Any]] = []

    if not isinstance(obj, dict):
        return gt

    audit_log = obj.get("audit_log", {})
    if not isinstance(audit_log, dict):
        return gt

    # 1) Extract applied=True IDs from apply_results
    applied_ids = set()
    for a in audit_log.get("apply_results", []) or []:
        if not isinstance(a, dict):
            continue
        if a.get("applied") is True and isinstance(a.get("id"), int):
            applied_ids.add(a["id"])

    # 2) Filter edits to only keep those in applied_ids
    for e in audit_log.get("edits", []) or []:
        if not isinstance(e, dict):
            continue

        gid   = e.get("id")
        # If applied_ids exist, only keep those; if none, filter all (gt becomes empty)
        if applied_ids and gid not in applied_ids:
            continue

        diff  = e.get("difficulty")
        sec   = e.get("location")
        errloc = e.get("target_find") or ""
        expl  = e.get("error_explanation") or ""
        xsec  = e.get("needs_cross_section", None)

        if not isinstance(errloc, str) or not errloc.strip():
            errloc = ""

        gt.append({
            # Temporarily keep original id, will renumber later
            "id": gid if isinstance(gid, int) else None,
            "difficulty": diff if isinstance(diff, str) else None,
            "section_location": sec.strip() if isinstance(sec, str) else None,
            "error_location": errloc.strip() if isinstance(errloc, str) else None,
            "explanation": expl.strip() if isinstance(expl, str) else None,
            "needs_cross_section": bool(xsec) if isinstance(xsec, bool) else None,
        })

    # 3) Renumber IDs for remaining gt based on current order: 1,2,3,...
    for new_id, item in enumerate(gt, start=1):
        item["id"] = new_id

    return gt


def _iter_unique_sections(outline_obj: Dict[str, Any]) -> List[str]:
    """从 outline 中按顺序列出去重后的 section 标题；若为空则返回空列表。"""
    titles = []
    for it in (outline_obj.get("outline") or []):
        t = (it.get("title") or "").strip()
        if t and t not in titles:
            titles.append(t)
    return titles

def add_findings_with_log(all_findings: List[Finding], new_findings: List[Finding], phase: str) -> None:
    """Append new findings to the global list and print progress summary."""
    if not new_findings:
        print(f"[DEBUG][{phase}] no new findings; total={len(all_findings)}")
        return
    before = len(all_findings)
    all_findings.extend(new_findings)
    added = len(all_findings) - before
    print(f"[DEBUG][{phase}] added {added} findings; total={len(all_findings)}")

def run_single(
    synth_json: Path,
    out_json: Path,
    debug_dir: Path,
    detect_model: str,
    synth_model: str, 
    enable_global_review: bool,
    global_max_findings: int,
    # —— Web search —— #
    enable_web_search: bool,
    search_max_results: int = 5,
    temperature: float = 0.0,
    # —— Memory —— #
    enable_memory_build: bool = False,
    memory_max_chars: int = 20000,
    # —— Section Review —— #
    enable_section_review: bool = False,
    enable_per_task: bool = False,
    # —— Additional features —— #
    enable_section_findings_as_prior: bool = False,
    enable_retriever: bool = False,
    enable_memory_injection: bool = False,
    # —— Merge —— #
    enable_merge: bool = False,
    enable_mm: bool = False,
    # —— Context and retry settings —— #
    max_ctx_chars: int = 120000,
    empty_retry_times: int = 2,
    # —— LLM Temperature settings —— #
    temp_memory: float = 0.2,
    temp_planner: float = 0.2,
    temp_retriever: float = 0.4,
    temp_specialist: float = 0.4,
    temp_global_review: float = 0.2,
    temp_section_review: float = 0.2,
    temp_merger: float = 0.2,
) -> Dict[str, Any]:
    debug_dir.mkdir(parents=True, exist_ok=True)
    obj = load_json(synth_json)
    gt_full = _extract_ground_truth_from_synth(obj)
    save_json(gt_full, debug_dir / "ground_truth.from_synth.json")
    paper = obj.get("paper", {}) or {}
    blocks = normalize_blocks(paper)
    save_json(blocks[:200], debug_dir / "paper.blocks.head.json")

    outline_obj = llm_build_outline(blocks, "", debug_dir, model=detect_model)

    all_findings: List[Finding] = []
    step_failures = []
    global_meta = {"enabled": enable_global_review}
    section_meta_overall = {"enabled": enable_section_review, "sections": []}
    web_stats = {
        "enabled": (enable_web_search and enable_retriever),
        "retriever_web_queries": 0,
        "retriever_web_evidence": 0
    }

    # 0) Full-paper memory
    memory_obj = None
    memory_meta = {"enabled": enable_memory_build}
    if enable_memory_build:
        try:
            memory_obj, mmeta = build_paper_memory(
                blocks=cap_blocks_by_budget(blocks, max_ctx_chars),
                dbgdir=debug_dir,
                model=detect_model, 
                enable_mm=enable_mm,
                temperature=temp_memory
            )
            memory_meta.update(mmeta)
        except Exception as ex:
            memory_meta["error"] = True
            step_failures.append("memory_build_crash")
    save_json(memory_meta, debug_dir / "memory.meta.json")

    # 1) Global cross-section review
    if enable_global_review:
        try:
            global_findings, gmeta = global_cross_section_review(
                blocks=blocks, dbgdir=debug_dir, model=detect_model, global_max_findings=global_max_findings, enable_mm=enable_mm, temperature=temp_global_review
            )
            global_meta.update(gmeta)
            add_findings_with_log(all_findings, global_findings, phase="global")
        except Exception:
            step_failures.append("global_review_crash")
    save_json(global_meta, debug_dir / "global_review.meta.json")

    # 2) Section-level review (before Planner, per section)
    section_findings_map: Dict[str, List[Finding]] = {}
    if enable_section_review:
        section_titles = _iter_unique_sections(outline_obj)
        for stitle in section_titles:
            sdir = debug_dir / "section_reviews" / re.sub(r"[^A-Za-z0-9._-]+","_", stitle or "Unknown")
            sdir.mkdir(parents=True, exist_ok=True)
            try:
                slice_blocks = slice_json_for_task_with_outline(
                    blocks, stitle, outline_obj.get("outline", []), max_chars=max_ctx_chars
                )
                # Inject references from References section based on citations in current section
                slice_blocks = enrich_section_blocks_with_local_references(
                    section_title=stitle,
                    section_blocks=slice_blocks,
                    all_blocks=blocks,
                )

                save_json(slice_blocks, sdir / "section_slice.json")

                # Inject memory slice excluding current section (only for section review)
                mem_txt_sr = build_memory_for_task(
                    memory_obj=memory_obj if enable_memory_build else None,
                    current_section=stitle,
                    max_chars=memory_max_chars
                )

                (sdir / "section.memory.slice.txt").write_text(mem_txt_sr or "", encoding="utf-8")

                sec_findings, sec_meta = section_level_review(
                    section_title=stitle,
                    section_blocks=slice_blocks,
                    dbgdir=sdir,
                    model=detect_model,
                    memory_slice=mem_txt_sr, 
                    enable_mm=enable_mm,
                    temperature=temp_section_review
                )
                section_meta_overall["sections"].append({"title": stitle, **sec_meta})
                # Record findings for corresponding section
                section_findings_map.setdefault(stitle, []).extend(sec_findings)
                add_findings_with_log(all_findings, sec_findings, phase=f"section:{stitle}")
            except Exception as ex:
                section_meta_overall["sections"].append({"title": stitle, "error": f"{type(ex).__name__}: {ex}"})
                step_failures.append(f"section_review_crash:{stitle}")
        save_json(section_meta_overall, debug_dir / "section_review.meta.json")
    else:
        save_json(section_meta_overall, debug_dir / "section_review.meta.json")

    
    
    # 4) Per-task pipeline
    per_task_summary = []
    if not enable_per_task:
        tasks = []
        plan_meta = None
        save_json(
            {"enabled": False, "reason": "disabled_by_flag"},
            debug_dir / "per_task.meta.json"
        )
    else:
        # 3) Planner (open-ended planning; with empty output retry)
        try:
            tasks, plan_meta = planner_build_tasks_mm(
                blocks=blocks, outline_obj=outline_obj, dbgdir=debug_dir, model=detect_model, enable_mm=enable_mm, temperature=temp_planner
            )
        except Exception:
            step_failures.append("planner_no_tasks")
            raise

        save_json({"enabled": True}, debug_dir / "per_task.meta.json")
        for t in tasks:
            t_dir = debug_dir / f"task_{t.task_id}"
            t_dir.mkdir(parents=True, exist_ok=True)

            try:
                slice_blocks = slice_json_for_task_with_outline(
                    blocks, t.section, outline_obj.get("outline", []), max_chars=max_ctx_chars
                )
                # Inject local references for each task's section
                slice_blocks = enrich_section_blocks_with_local_references(
                    section_title=t.section,
                    section_blocks=slice_blocks,
                    all_blocks=blocks,
                )

                save_json(slice_blocks, t_dir / "doc_slice.json")

                # Memory slice injection (natural language, excluding current section) - controlled by enable_memory_injection
                mem_txt = ""
                if enable_memory_build and enable_memory_injection:
                    mem_txt = build_memory_for_task(
                        memory_obj=memory_obj,
                        current_section=t.section,
                        max_chars=memory_max_chars
                    )
                (t_dir / "task.memory.slice.txt").write_text(mem_txt or "", encoding="utf-8")

                # 4.1 Retriever (can be globally disabled)
                paper_evid: List[Evidence] = []
                web_queries: List[Dict[str, str]] = []
                web_evid:   List[Evidence] = []
                use_retriever = bool(enable_retriever)

                if use_retriever:
                    paper_evid, web_queries, rmeta = retriever_extract_and_questions(
                        task=t,
                        slice_blocks=slice_blocks,
                        dbgdir=t_dir,         # 任务级目录
                        model=detect_model,
                        paper_only=(not enable_web_search),
                        memory_slice=mem_txt, 
                        enable_mm=enable_mm,
                        temperature=temp_retriever
                    )
                    # 4.2 Web Search (only when enabled and queries exist)
                    if enable_web_search and web_queries:
                        web_evid = perform_web_search_for_queries(
                            web_queries=web_queries,
                            dbgdir=t_dir / "web_search",
                            detect_model=detect_model,
                            max_results=search_max_results,
                            temperature=temperature
                        )
                        web_stats["retriever_web_queries"] += len(web_queries)
                        web_stats["retriever_web_evidence"] += len(web_evid)

                evid_final = paper_evid + web_evid

                # 4.3 Prior findings (only same section)
                priors_for_section: List[Finding] = []
                if enable_section_findings_as_prior:
                    priors_for_section = list(section_findings_map.get(t.section, []))

                # 4.4 Specialist
                cand, smeta = specialist_review(
                    task=t,
                    evid=evid_final,
                    neighbor_blocks=slice_blocks,
                    dbgdir=t_dir,
                    model=detect_model,
                    # "paper_only" now means: no web evidence or retriever not used
                    paper_only=(not enable_web_search) or (not use_retriever),
                    memory_slice=mem_txt,
                    prior_findings=priors_for_section,
                    use_retriever=use_retriever,
                    enable_mm=enable_mm,
                    temperature=temp_specialist
                )

                per_task_summary.append({
                    "task_id": t.task_id,
                    "risk": t.risk_dimension,
                    "used_retriever": bool(use_retriever),
                    "used_prior_findings": bool(enable_section_findings_as_prior and priors_for_section),
                    "used_memory": bool(mem_txt),
                    "evidence": len(evid_final),
                    "paper_evidence": len(paper_evid),
                    "web_queries": len(web_queries),
                    "web_evidence": len(web_evid),
                    "findings": len(cand)
                })
                add_findings_with_log(all_findings, cand, phase=f"specialist:{t.task_id}")
            except Exception as ex:
                step_failures.append(f"task_crash:{t.task_id}")
                save_json({"error": f"{type(ex).__name__}: {ex}", "trace": traceback.format_exc()}, t_dir / "task.error.json")

        save_json(per_task_summary, debug_dir / "per_task.summary.json")

    # 5) Merge and adjudicate (controlled by enable_merge)
    if enable_merge:
        merged = merge_and_adjudicate(all_findings, debug_dir, model=detect_model, temperature=temp_merger)
    else:
        # Skip merge, directly keep all findings (in current order)
        merged = list(all_findings)
        save_json(
            {"note": "merge disabled; keeping all raw findings without adjudication."},
            debug_dir / "merge.disabled.json"
        )

    # 6) Summary and save
    out_obj = {
        "eval_for_detector": True,
        "created_at": datetime.utcnow().isoformat() + "Z",
        "source_synth_file": str(synth_json),
        "detect_model_used": detect_model,
        "synth_model": synth_model,
        "ground_truth": gt_full,
        "findings": [asdict(f) for f in merged],
        "stats": {
            "tasks_total": len(tasks),
            "findings_raw": len(all_findings),
            "findings_merged": len(merged),
            "step_failures": step_failures,
            "planner_meta": plan_meta,
            "global_meta": global_meta,
            "section_meta": section_meta_overall,
            "outline_used": outline_obj.get("used"),
            "outline_items": len(outline_obj.get("outline", [])),
            "max_ctx_chars": max_ctx_chars,
            "enable_global_review": enable_global_review,
            "enable_section_review": enable_section_review,
            "global_max_findings": global_max_findings,
            # Web stats
            "web_search": web_stats,
            # Memory stats
            "memory": {
                "enabled": enable_memory_build,
                "per_task_chars_limit": memory_max_chars,
                "built": bool(memory_obj)
            },
            # Additional flags
            "flags": {
                "enable_section_findings_as_prior": enable_section_findings_as_prior,
                "enable_retriever": enable_retriever,
                "enable_memory_injection": enable_memory_injection,
                "enable_merge": enable_merge,
            }
        }
    }

    # ------------- ground truth output (save next to mas_detect.json) -------------
    # try:
    #     gt = gt_full   # _extract_ground_truth_from_synth already computed this
    #     gt_path = out_json.parent / "ground_truth.json"
    #     save_json(gt, gt_path)
    # except Exception as ex:
    #     print(f"[WARN] failed to write ground_truth.json: {ex}")

    # ------------- renumber findings (1..N in output) -------------
    try:
        for idx, f in enumerate(out_obj["findings"], start=1):
            f["id"] = idx
    except Exception as ex:
        print(f"[WARN] failed renumber findings: {ex}")

    save_json(out_obj, out_json)
    return {"ok": True, "out_json": str(out_json), "debug_dir": str(debug_dir)}

# =========================
# Batch processing utilities
# =========================
def compute_output_paths(
    synth_path: Path,
    detect_mode: str,
    detect_model: str,
) -> Tuple[Path, Path]:
    paper_dir = synth_path.parent
    out_json = paper_dir / "detect" / f"{detect_mode}_detect" / detect_model / synth_path.stem / f"{detect_mode}_detect.json"
    debug_dir = paper_dir / "detect" / f"{detect_mode}_detect_log" / detect_model / synth_path.stem
    return out_json, debug_dir

def find_synth_files(root_dir: Path, synth_glob: str) -> List[Path]:
    return sorted(root_dir.rglob(synth_glob))

def process_one_job(
    synth_path: Path,
    detect_model: str,
    detect_mode: str,
    synth_model: str,
    enable_global_review: bool,
    global_max_findings: int,
    overwrite: bool = False,
    overwrite_zero: bool = False, 
    # —— Web search —— #
    enable_web_search: bool = False,
    search_max_results: int = 5,
    temperature: float = 0.0,
    # —— Memory —— #
    enable_memory_build: bool = False,
    memory_max_chars: int = 20000,
    # —— Section Review —— #
    enable_section_review: bool = False,
    enable_per_task: bool = False,
    # —— Additional features —— #
    enable_section_findings_as_prior: bool = False,
    enable_retriever: bool = False,
    enable_memory_injection: bool = False,
    # —— Merge —— #
    enable_merge: bool = False,
    enable_mm: bool = False,
    # —— Context and retry settings —— #
    max_ctx_chars: int = 120000,
    empty_retry_times: int = 2,
    # —— LLM Temperature settings —— #
    temp_memory: float = 0.2,
    temp_planner: float = 0.2,
    temp_retriever: float = 0.4,
    temp_specialist: float = 0.4,
    temp_global_review: float = 0.2,
    temp_section_review: float = 0.2,
    temp_merger: float = 0.2,
) -> Dict[str, Any]:
    try:
        out_json, debug_dir = compute_output_paths(synth_path, detect_mode, detect_model)
        inprogress = out_json.with_suffix(out_json.suffix + ".inprogress")
        if out_json.exists() and not overwrite:
            if overwrite_zero and output_has_zero_findings(out_json):
                print(f"[BATCH] overwrite_zero=1 and findings==[] -> rerun: {out_json}")
            else:
                return {"synth": str(synth_path), "skipped": True, "reason": "exists", "out_json": str(out_json)}

        inprogress.parent.mkdir(parents=True, exist_ok=True)
        try: inprogress.touch(exist_ok=True)
        except Exception: pass
        result = run_single(
            synth_json=synth_path,
            out_json=out_json,
            debug_dir=debug_dir,
            detect_model=detect_model,
            synth_model=synth_model,
            enable_global_review=enable_global_review,
            global_max_findings=global_max_findings,
            enable_web_search=(enable_web_search and enable_retriever),
            search_max_results=search_max_results,
            temperature=temperature,
            enable_memory_build=enable_memory_build,
            memory_max_chars=memory_max_chars,
            enable_section_review=enable_section_review,
            enable_per_task=enable_per_task,
            enable_section_findings_as_prior=enable_section_findings_as_prior,
            enable_retriever=enable_retriever,
            enable_memory_injection=enable_memory_injection,
            enable_merge=enable_merge,
            enable_mm=enable_mm,
            max_ctx_chars=max_ctx_chars,
            empty_retry_times=empty_retry_times,
            temp_memory=temp_memory,
            temp_planner=temp_planner,
            temp_retriever=temp_retriever,
            temp_specialist=temp_specialist,
            temp_global_review=temp_global_review,
            temp_section_review=temp_section_review,
            temp_merger=temp_merger,
        )
        return {"synth": str(synth_path), "ok": True, "out_json": str(out_json), "debug_dir": str(debug_dir)}
    except Exception as e:
        return {"synth": str(synth_path), "ok": False, "error": f"{type(e).__name__}: {e}"}
    finally:
        try:
            if inprogress.exists(): inprogress.unlink()
        except Exception: pass

def run_batch(
    root_dir: Path,
    synth_glob: str,
    detect_model: str,
    synth_model: str,
    detect_mode: str,
    enable_global_review: bool,
    global_max_findings: int,
    jobs: int = 6,
    overwrite: bool = False,
    overwrite_zero: bool = False,
    max_papers: Optional[int] = None,
    # —— Web search —— #
    enable_web_search: bool = False,
    search_max_results: int = 5,
    temperature: float = 0.0,
    # —— Memory —— #
    enable_memory_build: bool = False,
    memory_max_chars: int = 20000,
    # —— Section Review —— #
    enable_section_review: bool = False,
    enable_per_task: bool = False,
    # —— Additional features —— #
    enable_section_findings_as_prior: bool = False,
    enable_retriever: bool = False,
    enable_memory_injection: bool = False,
    # —— Merge —— #
    enable_merge: bool = False,
    enable_mm: bool = False,
    # —— Context and retry settings —— #
    max_ctx_chars: int = 120000,
    empty_retry_times: int = 2,
    # —— LLM Temperature settings —— #
    temp_memory: float = 0.2,
    temp_planner: float = 0.2,
    temp_retriever: float = 0.4,
    temp_specialist: float = 0.4,
    temp_global_review: float = 0.2,
    temp_section_review: float = 0.2,
    temp_merger: float = 0.2,
) -> Dict[str, Any]:

    root_dir = Path(root_dir)

    # 1) Find all paper_synth_*.json files matching synth_glob
    synth_files = find_synth_files(root_dir, synth_glob)

    # 2) Filter by whether filename contains synth_model
    #    e.g., if synth_model='gpt-5-2025-08-07', only keep JSONs with this substring in name
    if synth_model:
        before = len(synth_files)
        synth_files = [p for p in synth_files if synth_model in p.name]
        after = len(synth_files)
        print(f"[BATCH] synth_model filter: target='{synth_model}', matched={after}/{before}")

    # 3) Truncate by max_papers (progress bar limit is filtered count)
    if max_papers is not None:
        synth_files = synth_files[:max_papers]

    # 4) If empty after filtering, return error
    if not synth_files:
        return {
            "ok": False,
            "error": (
                f"No synth files matched '{synth_glob}' "
                f"with synth_model='{synth_model}' under {root_dir}"
            ),
        }

    results = []
    bar = tqdm(total=len(synth_files), desc="MAS-Batch") if tqdm else None
    with ThreadPoolExecutor(max_workers=max(1, jobs)) as ex:
        futs = []
        for sp in synth_files:
            fut = ex.submit(
                process_one_job,
                sp, detect_model, detect_mode, synth_model,
                enable_global_review, global_max_findings,
                overwrite,
                overwrite_zero,
                enable_web_search, search_max_results, temperature,
                enable_memory_build, memory_max_chars,
                enable_section_review,
                enable_per_task,
                enable_section_findings_as_prior,
                enable_retriever,
                enable_memory_injection,
                enable_merge,
                enable_mm,
                max_ctx_chars,
                empty_retry_times,
                temp_memory,
                temp_planner,
                temp_retriever,
                temp_specialist,
                temp_global_review,
                temp_section_review,
                temp_merger,
            )
            futs.append(fut)
        for fut in as_completed(futs):
            res = fut.result()
            results.append(res)
            if bar: bar.update(1)
    if bar: bar.close()
    ok = sum(1 for r in results if r.get("ok"))
    skipped = sum(1 for r in results if r.get("skipped"))
    failed = [r for r in results if not r.get("ok") and not r.get("skipped")]
    summary = {
        "ok": True,
        "detect_model": detect_model,
        "detect_mode": detect_mode,
        "synth_model": synth_model,
        "total": len(results),
        "done": ok,
        "skipped": skipped,
        "failed": len(failed),
        "fail_details": failed[:50],
    }
    return summary

# =========================
# CLI
# =========================
def main():
    parser = argparse.ArgumentParser(
        description="Single/Batch MAS error-review evaluator (open-ended, unified prompts, multimodal, natural-language memory, global review, section-level review, threaded batch, two-step web search, retry-on-empty for ALL stages)."
    )
    # Single file mode
    parser.add_argument("--synth_json", type=str, default=None,
                        help="Path to a single synth JSON. If set, runs single-file mode unless --root_dir is provided.")
    parser.add_argument("--out_json",   type=str, default=None,
                        help="(Single mode) Explicit output json path. If omitted, and --detect_mode is set, will auto-place into {detect_mode}_detect/...")
    parser.add_argument("--debug_dir",  type=str, default="None",
                        help="(Single mode) Explicit debug dir. If omitted, and --detect_mode is set, will auto-place into {detect_mode}_detect_log/...")

    # Batch processing mode
    parser.add_argument("--root_dir", type=str, default="/mnt/parallel_ssd/home/zdhs0006/ACL/test2", help="Batch root dir; if set, runs threaded batch over synth_glob.")
    parser.add_argument("--synth_glob", type=str, default="paper_synth_*.json",
                        help="Glob to find synth JSONs under root_dir (default: paper_synth_*.json)")
    parser.add_argument("--jobs", type=int, default=10, help="Thread pool size")
    parser.add_argument("--overwrite", action="store_true", default=False, help="Force overwrite existing outputs")
    parser.add_argument("--overwrite_zero", action="store_true", default=False,
                    help="If output exists but has empty findings=[], rerun even when not overwriting.")
    parser.add_argument("--max_papers", type=int, default=None, help="Limit the number of synth files to process")

    # Common arguments
    parser.add_argument("--detect_model", type=str, default="qwen3-235b-a22b-instruct-2507",
                        help="Model tag used for detection")
    parser.add_argument("--detect_mode", type=str, default="test",
                        help="Detect mode name used in output paths")
    parser.add_argument("--synth_model", type=str, default="o4-mini",
                        help="Tag/name of the synthesis model whose data you are evaluating; "
                             "used for output dir grouping and stats.")

    # —— Global cross-section review —— #
    parser.add_argument("--enable-global-review", action="store_true", default=False,
                        help="Enable global cross-section review (default: disabled)")
    parser.add_argument("--global_max_findings", type=int, default=200,
                        help="Soft cap for global review (for safety); reviewer is instructed to be exhaustive without hard caps.")

    # —— Section Review —— #
    parser.add_argument("--enable-section-review", action="store_true", default=False, 
                        help="Enable section-level review (default: disabled)")

    # —— Per-task Pipeline —— #
    parser.add_argument("--enable-per-task", action="store_true", default=False, 
                        help="Enable per-task pipeline (default: disabled)")
    # Whether to pass section findings to Specialist (only same section)
    parser.add_argument("--enable-section-findings-as-prior", action="store_true", default=False,
                        help="Pass section_review findings of the SAME section to each task's Specialist as prior support.")
    # Main switch: enable Retriever (including WebSearch)
    parser.add_argument("--enable-retriever", action="store_true", default=False,
                        help="Enable Retriever (+WebSearch). Specialist will use retrieved evidence.")
    # Optional: enable memory injection
    parser.add_argument("--enable-memory-injection", action="store_true", default=False,
                        help="Inject memory into Specialist (requires --enable-memory-build).")

    # —— Merge —— #
    parser.add_argument("--enable-merge", action="store_true", default=False,
                        help="Enable merging/adjudication of findings (default: disabled)")

    # —— Web search —— #
    parser.add_argument("--enable-web-search", action="store_true", default=False,
                        help="Enable dedicated web_search step after Retriever (default: disabled)")
    parser.add_argument("--search_max_results", type=int, default=5,
                        help="Technical cap to avoid directory explosion; not a quota for retrieval quality.")
    parser.add_argument("--search_temperature", type=float, default=0.0,
                        help="Temperature for web search LLM calls")

    # —— Memory —— #
    parser.add_argument("--enable-memory-build", action="store_true", default=False,
                    help="Enable full-paper memory build (default: disabled)")
    parser.add_argument("--memory_max_chars", type=int, default=20000,
                        help="Max characters of memory injected per task (natural-language slice)")

    parser.add_argument("--enable-mm", action="store_true", default=False,
                    help="Enable multimodal input (images). Default: True")

    # —— Context and retry settings —— #
    parser.add_argument("--max_ctx_chars", type=int, default=120000,
                        help="Maximum context characters for LLM input")
    parser.add_argument("--empty_retry_times", type=int, default=2,
                        help="Number of retries for empty output/parse failures")

    # —— LLM Temperature settings —— #
    parser.add_argument("--temp_memory", type=float, default=0.2,
                        help="Temperature for memory building LLM calls")
    parser.add_argument("--temp_planner", type=float, default=0.2,
                        help="Temperature for planner LLM calls")
    parser.add_argument("--temp_retriever", type=float, default=0.4,
                        help="Temperature for retriever LLM calls")
    parser.add_argument("--temp_specialist", type=float, default=0.4,
                        help="Temperature for specialist LLM calls")
    parser.add_argument("--temp_global_review", type=float, default=0.2,
                        help="Temperature for global review LLM calls")
    parser.add_argument("--temp_section_review", type=float, default=0.2,
                        help="Temperature for section review LLM calls")
    parser.add_argument("--temp_merger", type=float, default=0.2,
                        help="Temperature for merger LLM calls")
    args = parser.parse_args()

    if not OPENAI_API_KEY:
        print("[FATAL] OPENAI_API_KEY is not set.", file=sys.stderr); sys.exit(1)

    # Disable web search if retriever is disabled
    enable_web_search_final = args.enable_web_search if args.enable_retriever else False

    # Print common configuration
    print(f"[INFO] detect_mode={args.detect_mode}  detect_model={args.detect_model}")
    print(f"[INFO] synth_model={args.synth_model}")
    print(f"[INFO] enable_global_review={args.enable_global_review}  global_max_findings={args.global_max_findings}")
    print(f"[INFO] enable_section_review={args.enable_section_review}")
    print(f"[INFO] enable_web_search={enable_web_search_final}  search_k={args.search_max_results}  temp_web={args.search_temperature}")
    print(f"[INFO] enable_memory_build={args.enable_memory_build}  memory_max_chars={args.memory_max_chars}")
    print(f"[INFO] enable_per_task={args.enable_per_task}")
    print(f"[INFO] enable_section_findings_as_prior={args.enable_section_findings_as_prior}")
    print(f"[INFO] enable_retriever={args.enable_retriever}")
    print(f"[INFO] enable_memory_injection={args.enable_memory_injection}")
    print(f"[INFO] enable_merge={args.enable_merge}")
    print(f"[INFO] temps: memory={args.temp_memory} planner={args.temp_planner} retriever={args.temp_retriever} specialist={args.temp_specialist} global={args.temp_global_review} section={args.temp_section_review} merger={args.temp_merger}")
    print(f"[INFO] enable_mm={args.enable_mm}")

    # ========== Batch Processing Mode ==========
    if args.root_dir:
        root_dir = Path(args.root_dir).expanduser().resolve()
        if not root_dir.exists():
            print(f"[FATAL] root_dir not found: {root_dir}", file=sys.stderr); sys.exit(1)
        print(f"[INFO] root_dir={root_dir}")
        print(f"[INFO] synth_glob={args.synth_glob}  jobs={args.jobs}  overwrite={args.overwrite} overwrite_zero={args.overwrite_zero}")

        summary = run_batch(
            root_dir=root_dir,
            synth_glob=args.synth_glob,
            detect_model=args.detect_model,
            synth_model=args.synth_model,
            detect_mode=args.detect_mode,
            enable_global_review=args.enable_global_review,
            global_max_findings=args.global_max_findings,
            jobs=args.jobs,
            overwrite=args.overwrite,
            overwrite_zero=args.overwrite_zero,
            max_papers=args.max_papers,
            enable_web_search=enable_web_search_final,
            search_max_results=args.search_max_results,
            temperature=args.search_temperature,
            enable_memory_build=args.enable_memory_build,
            memory_max_chars=args.memory_max_chars,
            enable_section_review=args.enable_section_review,
            enable_per_task=args.enable_per_task,
            enable_section_findings_as_prior=args.enable_section_findings_as_prior,
            enable_retriever=args.enable_retriever,
            enable_memory_injection=args.enable_memory_injection,
            enable_merge=args.enable_merge,
            enable_mm=args.enable_mm,
            max_ctx_chars=args.max_ctx_chars,
            empty_retry_times=args.empty_retry_times,
            temp_memory=args.temp_memory,
            temp_planner=args.temp_planner,
            temp_retriever=args.temp_retriever,
            temp_specialist=args.temp_specialist,
            temp_global_review=args.temp_global_review,
            temp_section_review=args.temp_section_review,
            temp_merger=args.temp_merger,
        )
        print(json.dumps(summary, ensure_ascii=False, indent=2))
        return

    # ========== Single File Mode ==========
    if not args.synth_json:
        print("[FATAL] Missing --synth_json or --root_dir.", file=sys.stderr); sys.exit(1)

    synth_json = Path(args.synth_json).expanduser().resolve()
    if not synth_json.exists():
        print(f"[FATAL] synth_json not found: {synth_json}", file=sys.stderr); sys.exit(1)

    if args.out_json:
        out_json = Path(args.out_json).expanduser().resolve()
    else:
        # Note: compute_output_paths only depends on detect_mode & detect_model, not synth_model (maintains original storage structure)
        out_json, _auto_debug = compute_output_paths(synth_json, args.detect_mode, args.detect_model)

    if args.debug_dir and args.debug_dir != "None":
        debug_dir = Path(args.debug_dir).expanduser().resolve()
    else:
        _auto_out, debug_dir = compute_output_paths(synth_json, args.detect_mode, args.detect_model)


    print(f"[INFO] synth_json={synth_json}")
    print(f"[INFO] out_json={out_json}")
    print(f"[INFO] debug_dir={debug_dir}")
    print(f"[INFO] max_ctx_chars={args.max_ctx_chars}")

    summary = run_single(
        synth_json, out_json, debug_dir,
        detect_model=args.detect_model,
        synth_model=args.synth_model,
        enable_global_review=args.enable_global_review,
        global_max_findings=args.global_max_findings,
        enable_web_search=enable_web_search_final,
        search_max_results=args.search_max_results,
        temperature=args.search_temperature,
        enable_memory_build=args.enable_memory_build,
        memory_max_chars=args.memory_max_chars,
        enable_section_review=args.enable_section_review,
        enable_per_task=args.enable_per_task,
        enable_section_findings_as_prior=args.enable_section_findings_as_prior,
        enable_retriever=args.enable_retriever,
        enable_memory_injection=args.enable_memory_injection,
        enable_merge=args.enable_merge,
        enable_mm=args.enable_mm,
        max_ctx_chars=args.max_ctx_chars,
        empty_retry_times=args.empty_retry_times,
        temp_memory=args.temp_memory,
        temp_planner=args.temp_planner,
        temp_retriever=args.temp_retriever,
        temp_specialist=args.temp_specialist,
        temp_global_review=args.temp_global_review,
        temp_section_review=args.temp_section_review,
        temp_merger=args.temp_merger,
    )
    print("[DONE]", json.dumps(summary, ensure_ascii=False))

if __name__ == "__main__":
    main()
