#!/usr/bin/env python3
# -*- coding: utf-8 -*-

"""
single_eval_detector_dual_model.py

读取每个论文目录下的 paper_synth_{SYNTH_MODEL}.json：
- 提取其中的 paper.content 作为 LLM 的评审输入（逐块多模态：text / image_url）
- 使用 DETECT_MODEL 让 LLM 输出结构化的 findings（type/error_location/explanation/confidence/proposed_fix）
- 同时从 audit_log.edits 中抽取“真实” location + error_explanation（作为对照信息保存）
- 写出 single_detect/single_detect_{DETECT_MODEL}.json

特性：
- 断点续传（存在即跳过 + .inprogress）
- OpenAI 客户端懒加载、重试、并发
- 允许 0 条 findings（当模型未发现问题时）
"""

from __future__ import annotations
import argparse
import asyncio
import json, os, re, time, sys
from typing import List, Dict, Any, Tuple, Optional
from pathlib import Path
from datetime import datetime

try:
    from tqdm import tqdm
except Exception:
    tqdm = None

from openai import OpenAI
from openai import (
    APIError, RateLimitError, APITimeoutError, APIConnectionError,
    AuthenticationError, BadRequestError, PermissionDeniedError,
    UnprocessableEntityError
)
from utils import extract_json_from_text, get_openai_client, load_json, save_json

# ================= 默认配置 =================
DEFAULT_DETECT_MODEL= "o4-mini"      # 真正调用评估的模型
DEFAULT_SYNTH_MODEL = "gpt-5-2025-08-07"      # 只用于定位/命名文件

OPENAI_API_KEY  = os.getenv("OPENAI_API_KEY", None)
LLM_MAX_RETRIES = 3

MAX_DOC_CHARS   = 100_000  # 作为max_tokens等预算使用

# findings 类型枚举（用于提示与校验；允许兜底 'other'）
FINDING_TYPES = [
    "logic","data","theory","conclusion","citation","figure",
    "reproducibility","data_leakage","other"
]

# ================= 工具函数 =================

# ============== 多模态（逐块）输入构建 ==============
def build_eval_rules_prompt(title: str) -> str:
    """
    规则提示（不拼全文；后续把论文一块块追加到 user.content 中）
    """
    finding_types_hint = ", ".join(FINDING_TYPES)
    return f"""
You are an expert scientific reviewer. You will receive the manuscript as an ordered list of blocks,
each block being either a TEXT chunk or an IMAGE URL. Read blocks in order and reason over the whole paper.

Return ONLY a SINGLE JSON object with the top-level key 'findings'. Each finding object must contain exactly:
- id (int, 1-based): the running index of this finding (1, 2, 3, ...).
- type (string): one of [{finding_types_hint}] or 'other' if necessary.
- section_location (string): the section or part of the paper where the issue occurs (e.g., "Abstract", "Introduction", "Experiments", "Related Work").
- error_location (string): the exact text or claim that is incorrect or problematic (may include LaTeX or multi-line content).
- explanation (string): concise, specific, and verifiable explanation (point to sections/tables/equations/references as needed).
- confidence (float in [0,1]).
- proposed_fix (string): short, actionable correction or improvement.

If no clear issues are found, return: {{"findings": []}}.

Guidelines:
- Be specific and point to concrete anchors (e.g., "Section 4.2", "Table 3", "Equation (6)", "[44]").
- Prefer quoting exact textual spans in 'error_location'.
- Avoid minor stylistic issues and redundant duplicates.
- Do NOT include any text outside the JSON.
- When in doubt, err on the side of being thorough, output every potential scientific concern or questionable point that might merit further review.

The blocks follow below. Do NOT reconstruct the full paper; use them as-is.
""".strip()

def paper_to_mm_parts_streaming(paper: dict,
                                max_parts: int = 1_000,
                                max_text_chars_per_block: int = 10_000,
                                soft_char_budget: int = 120_000) -> List[Dict[str, Any]]:
    """
    仅两类：text / image_url；逐条按顺序放入列表，并为每块添加一行头注释，便于定位。
    - 对 text：按块截断到 max_text_chars_per_block
    - 对 image_url：透传 {"type":"image_url","image_url":{"url":...}}
    - 软预算：累计文本字符到 soft_char_budget 附近就停
    """
    parts: List[Dict[str, Any]] = []
    total_chars = 0

    for it in (paper.get("content") or []):
        typ = (it.get("type") or "").lower()
        idx = it.get("index", it.get("content_index"))
        sec = it.get("section") or ""
        header = f"[Block #{idx if idx is not None else '?'} | {typ or 'unknown'}{(' | ' + sec) if sec else ''}]"
        parts.append({"type": "text", "text": header})

        if typ == "text":
            t = (it.get("text") or "")
            if not isinstance(t, str) or not t.strip():
                continue
            if max_text_chars_per_block and len(t) > max_text_chars_per_block:
                t = t[:max_text_chars_per_block]
            parts.append({"type": "text", "text": t})
            total_chars += len(t)

        elif typ == "image_url":
            img = it.get("image_url")
            url = img.get("url") if isinstance(img, dict) else (img if isinstance(img, str) else None)
            if isinstance(url, str) and (url.startswith("http") or url.startswith("data:")):
                parts.append({"type": "image_url", "image_url": {"url": url}})

        # 预算/数量控制
        if soft_char_budget and total_chars > int(soft_char_budget * 1.05):
            break
        if max_parts and len(parts) >= max_parts:
            break

    return parts

def build_mm_user_content_streaming_eval(title: str, paper: dict) -> List[Dict[str, Any]]:
    parts = [{"type": "text", "text": build_eval_rules_prompt(title)}]
    parts.extend(paper_to_mm_parts_streaming(paper))
    return parts

def call_openai_eval_mm(user_content: List[Dict[str, Any]], detect_model: str, api_key: Optional[str]) -> Optional[dict]:
    """
    多模态 Chat Completions：messages[1].content 是一个列表，
    每个元素是 {"type":"text",...} 或 {"type":"image_url",...}
    """
    client = get_openai_client(api_key)
    last_exc = None
    for attempt in range(1, LLM_MAX_RETRIES+1):
        try:
            resp = client.chat.completions.create(
                model=detect_model,
                messages=[
                    {
                        "role":"system",
                        "content":(
                            "You are an expert scientific reviewer. "
                            "Return ONLY a SINGLE JSON object per the user's schema. "
                            "If no clear issues are found, return {\"findings\": []}. "
                            "Do not include any text outside the JSON."
                        )
                    },
                    {"role":"user","content": user_content}
                ],
                temperature=0.0,
                max_tokens=80_000,
                n=1
            )
            text = resp.choices[0].message.content
            return extract_json_from_text(text)
        except RateLimitError as e:
            last_exc = e; wait = 2 ** attempt
            print(f"[OpenAI RATE LIMIT] attempt {attempt}, waiting {wait}s...", file=sys.stderr)
            time.sleep(wait)
        except (APITimeoutError, APIConnectionError) as e:
            last_exc = e
            print(f"[OpenAI TEMP ERROR] attempt {attempt}: {e}", file=sys.stderr)
            time.sleep(1.2 * attempt)
        except (BadRequestError, AuthenticationError, PermissionDeniedError, UnprocessableEntityError, APIError) as e:
            last_exc = e
            print(f"[OpenAI FATAL] {e}", file=sys.stderr)
            break
        except Exception as e:
            last_exc = e
            print(f"[OpenAI ERROR] attempt {attempt}: {e}", file=sys.stderr)
            time.sleep(0.8 * attempt)
    print(f"[LLM FAILURE] {last_exc}", file=sys.stderr)
    return None

# ================= 结构校验 =================
def _validate_findings_list(items: Any) -> Tuple[bool, str]:
    if not isinstance(items, list):
        return False, "missing 'findings' list"
    seen_ids = set()
    for f in items:
        if not isinstance(f, dict):
            return False, "finding item not dict"
        
        required = ["id","type","error_location","section_location","explanation","confidence","proposed_fix"]
        for k in required:
            if k not in f:
                return False, f"finding missing key '{k}'"

        # id: int, >=1, unique
        try:
            fid = int(f["id"])
        except Exception:
            return False, "finding.id must be int"
        if fid < 1:
            return False, "finding.id must be >= 1"
        if fid in seen_ids:
            return False, "finding.id must be unique"
        seen_ids.add(fid)

        # confidence
        try:
            c = float(f["confidence"])
        except Exception:
            return False, "finding.confidence must be float-like"
        if not (0.0 <= c <= 1.0):
            return False, "finding.confidence must be in [0,1]"
        # strings
        for k in ["error_location","section_location","explanation","proposed_fix"]:
            if not isinstance(f[k], str) or not f[k].strip():
                return False, f"finding.{k} must be non-empty string"
    return True, "ok"

def validate_response(obj: dict) -> Tuple[bool, str]:
    """
    仅验证 findings；允许为空数组。
    """
    if not isinstance(obj, dict):
        return False, "not a dict"
    if "findings" not in obj:
        return False, "missing 'findings'"
    return _validate_findings_list(obj["findings"])

# ================= 文件命名与断点续传 =================
def _tag(name: str) -> str:
    return name.replace("/", "_")

def _synth_json_path(dir_path: Path, synth_model: str) -> Path:
    return dir_path / f"paper_synth_{_tag(synth_model)}.json"

def _out_json_path(dir_path: Path, detect_model: str) -> Path:
    return dir_path / "single_detect" / f"single_detect_{_tag(detect_model)}.json"

def _inprogress_path(dir_path: Path, detect_model: str) -> Path:
    return dir_path / "single_detect" / f"single_detect_{_tag(detect_model)}.json.inprogress"

def _already_done(dir_path: Path, detect_model: str) -> bool:
    return _out_json_path(dir_path, detect_model).exists()

# ================= 单篇处理 =================
def process_one_dir_sync(dir_path: Path, synth_model: str, detect_model: str, api_key: Optional[str]) -> Tuple[Path, bool, str]:
    out_json = _out_json_path(dir_path, detect_model)
    inprog  = _inprogress_path(dir_path, detect_model)
    synth   = _synth_json_path(dir_path, synth_model)

    if _already_done(dir_path, detect_model):
        return dir_path, True, "skip_done"
    if not synth.exists():
        return dir_path, False, f"missing synth json: {synth}"

    try:
        inprog.write_text(str(time.time()), encoding="utf-8")
    except Exception:
        pass

    try:
        obj = load_json(str(synth))

        # ground truth from audit_log.edits（作为对照信息保存）
        audit_log = obj.get("audit_log", {}) if isinstance(obj, dict) else {}
        gt_edits  = audit_log.get("edits", []) if isinstance(audit_log, dict) else []
        gt_pairs  = []
        for e in gt_edits:
            if not isinstance(e, dict): continue
            id  = e.get("id", 0)
            difficulty  = e.get("difficulty", "")
            loc = e.get("location", "")
            exp = e.get("error_explanation", "")
            needs_cross_section = e.get("needs_cross_section", "")
            replacement = e.get("replacement", "")

            if isinstance(loc, str) and loc.strip() and isinstance(exp, str) and exp.strip():
                gt_pairs.append(
                    {
                        "id": id,
                        "difficulty": difficulty.strip(), 
                        "section_location": loc.strip(), 
                        "error_location": replacement.strip(), 
                        "explanation": exp.strip(),
                        "needs_cross_section": needs_cross_section,
                     },
                     )

        # 传给检测模型的多模态 user.content（逐块）
        paper = obj.get("paper", {}) if isinstance(obj, dict) else {}
        # 提取标题（作为规则提示的一部分）
        title = ""
        meta = paper.get("metadata") if isinstance(paper, dict) else None
        if isinstance(meta, dict):
            title = meta.get("title", "") or ""
        if not title and isinstance(paper.get("content"), list) and paper["content"]:
            first = paper["content"][0]
            if isinstance(first, dict) and first.get("type") == "text":
                t0 = (first.get("text") or "").strip()
                if t0:
                    title = t0.splitlines()[0][:200]
        if not title:
            title = "Unknown Title"

        user_content = build_mm_user_content_streaming_eval(title, paper)
        pred_obj = call_openai_eval_mm(user_content, detect_model=detect_model, api_key=api_key)
        if pred_obj is None:
            # 兜底（极端情况）：仅发送规则，不附论文内容，期望输出空 findings
            print("[WARN] MM call failed; retrying with rules-only.", file=sys.stderr)
            rules_only = [{"type":"text", "text": build_eval_rules_prompt(title)}]
            pred_obj = call_openai_eval_mm(rules_only, detect_model=detect_model, api_key=api_key)

        if pred_obj is None:
            save_json({"error": "llm_none"}, str(dir_path / "eval_llm_resp_debug.json"))
            return dir_path, False, "LLM returned None"

        ok, msg = validate_response(pred_obj)
        if not ok:
            save_json({"raw_llm_obj": pred_obj, "validate_msg": msg}, str(dir_path / "eval_llm_resp_debug.json"))
            return dir_path, False, f"validate failed: {msg}"

        out_obj = {
            "eval_for_detector": True,
            "created_at": datetime.utcnow().isoformat() + "Z",
            "source_synth_file": str(synth),
            "synth_model_tag": synth_model,
            "detect_model_used": detect_model,
            "ground_truth": gt_pairs,
            "findings": pred_obj.get("findings", [])
        }
        save_json(out_obj, str(out_json))
        print(f"[DONE] Saved: {out_json}")
        return dir_path, True, "ok"

    finally:
        try:
            if inprog.exists():
                inprog.unlink()
        except Exception:
            pass

# ================= 扫描 root_dir（一级子目录） =================
def find_dirs_with_synth(root_dir: Path, synth_model: str) -> List[Path]:
    out = []
    target = f"paper_synth_{_tag(synth_model)}.json"
    for sub in sorted(root_dir.iterdir()):
        if not sub.is_dir(): continue
        if (sub / target).exists():
            out.append(sub)
    return out

# ================= 并发运行 =================
async def bounded_worker(sem: asyncio.Semaphore, loop, d: Path, synth_model: str, detect_model: str, api_key: Optional[str]):
    async with sem:
        return await loop.run_in_executor(None, process_one_dir_sync, d, synth_model, detect_model, api_key)

async def run_batch(root_dir: Path, concurrency: int, synth_model: str, detect_model: str, api_key: Optional[str]) -> Dict[str, Any]:
    dirs = find_dirs_with_synth(root_dir, synth_model)
    if not dirs:
        return {"total": 0, "ok": 0, "fail": []}

    planned, skipped_done = [], 0
    for d in dirs:
        if _already_done(d, detect_model):
            skipped_done += 1
            continue
        planned.append(d)

    sem  = asyncio.Semaphore(concurrency)
    loop = asyncio.get_running_loop()
    tasks = [bounded_worker(sem, loop, d, synth_model, detect_model, api_key) for d in planned]

    results = []
    iterator = asyncio.as_completed(tasks)
    if tqdm is not None:
        iterator = tqdm(iterator, total=len(tasks), desc="Evaluating")
    for coro in iterator:
        results.append(await coro)

    ok_cnt = sum(1 for (_, ok, msg) in results if ok)
    fail   = [(str(p), msg) for (p, ok, msg) in results if not ok and msg != "skip_done"]
    return {"total": len(results), "ok": ok_cnt + skipped_done, "fail": fail, "skipped_done": skipped_done}

# ================= CLI =================
def main():
    parser = argparse.ArgumentParser(
        description="Evaluate corrupted papers with two models: synth_model for file tag, detect_model for LLM calls."
    )
    parser.add_argument(
        "--root_dir",
        type=str, default="downloads/ICML_2025_oral_test",
        help="根目录，其子目录内含 paper_synth_{SYNTH_MODEL}.json"
    )
    parser.add_argument(
        "--concurrency", "-j",
        type=int, default=4,
        help="并发数（默认 4）"
    )
    parser.add_argument(
        "--synth_model",
        type=str, default=DEFAULT_SYNTH_MODEL,
        help=f"合成数据的模型名（用于定位/命名文件），默认 {DEFAULT_SYNTH_MODEL}"
    )
    parser.add_argument(
        "--detect_model",
        type=str, default=DEFAULT_DETECT_MODEL,
        help=f"用于 LLM 评估调用的模型名，默认 {DEFAULT_DETECT_MODEL}"
    )

    args = parser.parse_args()

    if not OPENAI_API_KEY:
        print("[FATAL] OPENAI_API_KEY is not set.", file=sys.stderr); sys.exit(1)

    root = Path(args.root_dir).expanduser().resolve()
    if not root.exists():
        print(f"[FATAL] root_dir not found: {root}", file=sys.stderr); sys.exit(1)

    print(f"[INFO] root={root}  concurrency={args.concurrency}  synth_model={args.synth_model}  detect_model={args.detect_model}")

    loop = asyncio.get_event_loop()
    summary = loop.run_until_complete(
        run_batch(root, args.concurrency, args.synth_model, args.detect_model, OPENAI_API_KEY)
    )

    print(f"[DONE] total={summary['total']}  ok={summary['ok']}  fail={len(summary['fail'])}  skipped_done={summary.get('skipped_done', 0)}")
    if summary["fail"]:
        print("[FAILED ITEMS]")
        for i, (path, msg) in enumerate(summary["fail"], 1):
            print(f"  {i}. {path} :: {msg}")

if __name__ == "__main__":
    main()
