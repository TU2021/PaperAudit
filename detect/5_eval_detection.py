#!/usr/bin/env python3
# -*- coding: utf-8 -*-

from __future__ import annotations
import argparse
import asyncio
import json, os, re, time, sys
from typing import List, Dict, Any, Tuple, Optional
from pathlib import Path
from threading import Lock
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

# ================ 默认配置 ================
DEFAULT_DETECT_MODEL = "o4-mini"
DEFAULT_SYNTH_MODEL  = "o4-mini"  # 用于按文件名过滤 paper_synth_{SYNTH_MODEL}_*.json

DEFAULT_EVAL_MODEL   = "gpt-5.1"
DEFAULT_EVAL_MODE    = "standard"  # 对应 <eval_mode>_detect


OPENAI_API_KEY  = os.getenv("OPENAI_API_KEY", None)
LLM_MAX_RETRIES = 3

# ================ 基础IO ================
def load_json(path: str) -> dict:
    with open(path, "r", encoding="utf-8") as f:
        return json.load(f)

def save_json(obj, path: str):
    Path(path).parent.mkdir(parents=True, exist_ok=True)
    with open(path, "w", encoding="utf-8") as f:
        json.dump(obj, f, indent=2, ensure_ascii=False)

def extract_json_from_text(text: str):
    text = (text or "").strip()
    try:
        return json.loads(text)
    except Exception:
        pass
    start = text.find("{"); end = text.rfind("}")
    if start != -1 and end != -1 and end > start:
        candidate = text[start:end+1]
        for _ in range(5):
            try:
                return json.loads(candidate)
            except Exception:
                end = text.rfind("}", 0, end-1)
                if end <= start: break
                candidate = text[start:end+1]
    m = re.search(r"\{(?:.|\n)*?\}", text)
    if m:
        return json.loads(m.group(0))
    raise ValueError("Failed to extract JSON from LLM output.")

# ================ OpenAI 客户端（懒加载） ================
_client_singleton: Optional[OpenAI] = None
_client_lock = Lock()

def get_openai_client(api_key: Optional[str]) -> OpenAI:
    global _client_singleton
    with _client_lock:
        if _client_singleton is None:
            if not api_key:
                raise RuntimeError("OPENAI_API_KEY not set.")
            _client_singleton = OpenAI(api_key=api_key)
        return _client_singleton

# ================ 路径与工具 ================
def _tag(name: str) -> str:
    return name.replace("/", "_")

def compute_eval_paths(
    synth_path: Path,
    detect_model: str,
    eval_model: str,
    eval_mode: str,
) -> Tuple[Path, Path, Path]:
    """
    根据 synth_json 路径 & detect/eval model 统一构造：
      - detect 输入路径：<paper_dir>/<eval_mode>_detect/<detect_model>/<synth_stem>/<eval_mode>_detect.json
      - eval 输出路径：  <paper_dir>/<eval_mode>_eval/<detect_model>/<synth_stem>/eval_{eval_model}.json
      - inprogress 路径：同 eval 输出 + .inprogress
    """
    paper_dir = synth_path.parent
    synth_stem = synth_path.stem  # 例如 "paper_synth_gpt-5-2025-08-07_xxx"

    detect_json = paper_dir / f"{eval_mode}_detect" / detect_model / synth_stem / f"{eval_mode}_detect.json"
    out_json    = paper_dir / f"{eval_mode}_eval" / detect_model / synth_stem / f"eval_{_tag(eval_model)}.json"
    inprogress  = out_json.with_suffix(out_json.suffix + ".inprogress")
    return detect_json, out_json, inprogress

def find_synth_files(root_dir: Path, synth_model: str) -> List[Path]:
    """
    在 root_dir 下递归查找 paper_synth_*.json，
    再用 synth_model 子串过滤（例如 'gpt-5-2025-08-07'）。
    """
    root_dir = Path(root_dir)
    all_synth = sorted(root_dir.rglob("paper_synth_*.json"))
    if not synth_model:
        return all_synth
    before = len(all_synth)
    synth_files = [p for p in all_synth if synth_model in p.name]
    after = len(synth_files)
    print(f"[BATCH] synth_model filter: target='{synth_model}', matched={after}/{before}")
    return synth_files

# ================ Prompt（新版结构匹配） ================
def build_match_prompt(ground_truth: List[Dict[str,Any]], findings: List[Dict[str,Any]]) -> str:
    """
    传入 GT 与 findings（新版结构），要求 LLM 对齐：
    每个 GT 是否被 findings 中任何一条命中。
    """
    # 只展示必要字段，避免噪音
    gt_min = []
    for g in ground_truth:
        gt_min.append({
            "id": g.get("id"),
            "difficulty": g.get("difficulty"),
            "section_location": g.get("section_location"),
            "error_location": g.get("error_location"),
            "explanation": g.get("explanation"),
            "needs_cross_section": g.get("needs_cross_section", False),
        })

    fd_min = []
    for f in findings:
        fd_min.append({
            "id": f.get("id"),
            "type": f.get("type"),
            "section_location": f.get("section_location"),
            "error_location": f.get("error_location"),
            "explanation": f.get("explanation"),
            "confidence": f.get("confidence"),
            "proposed_fix": f.get("proposed_fix"),
        })

    gt_json = json.dumps(gt_min, ensure_ascii=False, indent=2)
    fd_json = json.dumps(fd_min, ensure_ascii=False, indent=2)

    return f"""
You are an expert adjudicator. Compare authoritative ground-truth errors vs model-predicted findings.

DATA SCHEMAS:
- ground_truth item has:
  {{
    "id": int,
    "difficulty": "easy"|"medium"|"hard",
    "section_location": string,             // e.g., "Abstract", "Method", "Experiments", "Section VI.B"
    "error_location": string,               // exact problematic text/claim (may include LaTeX / multi-line)
    "explanation": string,                  // precise description of what's wrong and why
    "needs_cross_section": boolean          // true if verifying requires cross-section consistency checks
  }}

- findings item has:
  {{
    "id": int,
    "type": string,                         // e.g., "logic","data","theory","conclusion","citation","figure","reproducibility","data_leakage","other"
    "section_location": string,             // section(s) involved; may include multiple sections
    "error_location": string,               // the specific text/claim found problematic
    "explanation": string,                  // reason why it's problematic
    "confidence": float in [0,1],
    "proposed_fix": string
  }}

TASK:
For EACH ground_truth item, decide whether it is successfully detected by ANY finding.
A finding MATCHES a ground-truth if the combination of its "error_location" and "explanation" captures the SAME scientific issue.
- It is acceptable if the finding references multiple sections (especially when ground_truth.needs_cross_section = true).
- Section names/titles may differ yet still refer to the same part; focus on semantic equivalence.
- Minor rephrasing is fine; vague/overbroad or clearly different issues are NOT matches.
- If multiple findings together refer to the same GT issue, include all matching indices.

OUTPUT (JSON object only):
{{
  "matches": [
    {{
      "gt_index": <int, 1-based>,                  // index in 'ground_truth' array
      "matched": <true|false>,
      "matched_pred_indices": [<int, ...>],        // indices in 'findings' that best match this GT; empty if none
      "rationale": "2–4 sentences explaining the judgment with reference to error_location/explanation and section(s)"
    }},
    ...
  ]
}}

RULES:
- There must be exactly len(ground_truth) items in "matches", one per GT in order.
- Be conservative: if the meaning diverges or refers to a different scientific issue, mark not matched.
- Prefer precision over recall.
- Many-to-many is allowed: a single gt_index may match multiple pred_indices, and a single pred_index may be matched by multiple gt_indices.


GROUND_TRUTH:
{gt_json}

FINDINGS:
{fd_json}
""".strip()

# ================ LLM 调用 ================
def call_openai_match(prompt: str, eval_model: str, api_key: Optional[str]) -> Optional[dict]:
    client = get_openai_client(api_key)
    last_exc = None
    for attempt in range(1, LLM_MAX_RETRIES+1):
        try:
            resp = client.chat.completions.create(
                model=eval_model,
                messages=[
                    {
                        "role": "system",
                        "content": (
                            "You are a careful scientific adjudicator comparing ground truth vs findings. "
                            "Return a JSON object per the user's schema including 'matches'. "
                            "Do not include any text outside the JSON."
                        )
                    },
                    {"role": "user", "content": prompt}
                ],
                temperature=0.2,
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

# ================ 结构校验 ================
def validate_match_obj(obj: dict, gt_len: int) -> Tuple[bool, str]:
    if not isinstance(obj, dict):
        return False, "not a dict"
    if "matches" not in obj or not isinstance(obj["matches"], list):
        return False, "missing 'matches' list"

    matches = obj["matches"]
    if len(matches) != gt_len:
        return False, f"'matches' length {len(matches)} != ground_truth length {gt_len}"

    for i, m in enumerate(matches):
        if not isinstance(m, dict):
            return False, f"match[{i}] not dict"
        if "gt_index" not in m or "matched" not in m or "matched_pred_indices" not in m or "rationale" not in m:
            return False, f"match[{i}] missing keys"
        if not isinstance(m["gt_index"], int):
            return False, f"match[{i}].gt_index must be int"
        # 兼容 1-based（Prompt）与历史 0-based：接受 i 或 i+1
        if not (m["gt_index"] == i or m["gt_index"] == i + 1):
            return False, f"match[{i}].gt_index must equal {i} (0-based) or {i+1} (1-based)"
        if not isinstance(m["matched"], bool):
            return False, f"match[{i}].matched must be bool"
        if not isinstance(m["matched_pred_indices"], list) or not all(isinstance(x, int) for x in m["matched_pred_indices"]):
            return False, f"match[{i}].matched_pred_indices must be list[int]"
        if not isinstance(m["rationale"], str) or not m["rationale"].strip():
            return False, f"match[{i}].rationale must be non-empty string"
    return True, "ok"

# ================ 单篇处理（基于 synth_json） ================
def process_one_dir_sync(
    synth_path: Path,
    detect_model: str,
    eval_model: str,
    eval_mode: str,
    api_key: Optional[str],
    overwrite: bool
) -> Tuple[Path, bool, str]:
    """
    对单个 synth_json 对应的 detect 输出做 GT-vs-findings 匹配评估。
    - 输入： 根据 synth_path + detect_model + eval_mode 推出 detect_json
    - 输出： eval/{detect_model}/{synth_stem}/eval_{eval_model}.json
    """
    in_path, out_path, inprog = compute_eval_paths(
        synth_path=synth_path,
        detect_model=detect_model,
        eval_model=eval_model,
        eval_mode=eval_mode,
    )

    # 仅当未指定 overwrite 时才跳过
    if not overwrite and out_path.exists():
        return synth_path, True, "skip_done"
    if not in_path.exists():
        return synth_path, False, f"missing detect json: {in_path}"

    try:
        inprog.parent.mkdir(parents=True, exist_ok=True)
        inprog.write_text(str(time.time()), encoding="utf-8")
    except Exception:
        pass

    try:
        data = load_json(str(in_path))
        ground_truth = data.get("ground_truth", [])
        findings     = data.get("findings", [])

        # 规范化（保留新结构关键字段）
        def _normalize_gt(items):
            out = []
            for it in items if isinstance(items, list) else []:
                if isinstance(it, dict):
                    gid   = it.get("id")
                    diff  = it.get("difficulty")
                    sec   = it.get("section_location")
                    err   = it.get("error_location")
                    expl  = it.get("explanation")
                    xsec  = it.get("needs_cross_section", False)
                    if isinstance(sec, str) and sec.strip() and isinstance(err, str) and err.strip() and isinstance(expl, str) and expl.strip():
                        out.append({
                            "id": gid,
                            "difficulty": diff,
                            "section_location": sec.strip(),
                            "error_location": err.strip(),
                            "explanation": expl.strip(),
                            "needs_cross_section": bool(xsec),
                        })
            return out

        def _normalize_findings(items):
            out = []
            for it in items if isinstance(items, list) else []:
                if isinstance(it, dict):
                    fid   = it.get("id")
                    typ   = it.get("type")
                    sec   = it.get("section_location")
                    err   = it.get("error_location")
                    expl  = it.get("explanation")
                    conf  = it.get("confidence")
                    fix   = it.get("proposed_fix")
                    if all([
                        isinstance(typ, str) and typ.strip(),
                        isinstance(sec, str) and sec.strip(),
                        isinstance(err, str) and err.strip(),
                        isinstance(expl, str) and expl.strip(),
                        isinstance(fix, str) and fix.strip(),
                        isinstance(conf, (float, int))
                    ]):
                        out.append({
                            "id": fid,
                            "type": typ.strip(),
                            "section_location": sec.strip(),
                            "error_location": err.strip(),
                            "explanation": expl.strip(),
                            "confidence": float(conf),
                            "proposed_fix": fix.strip(),
                        })
            return out

        def _ensure_ids(items: List[Dict[str, Any]]) -> List[Dict[str, Any]]:
            """
            为缺失或非法的 id 自动分配 1-based 不重复的整数 ID。
            - 已有合法 id(>=1 的 int) 保留
            - 缺失/非法的按出现顺序补发，避开已占用的 id
            """
            used = set(int(it["id"]) for it in items if isinstance(it.get("id"), int) and it["id"] >= 1)
            next_id = 1
            for it in items:
                valid = isinstance(it.get("id"), int) and it["id"] >= 1
                if not valid:
                    while next_id in used:
                        next_id += 1
                    it["id"] = next_id
                    used.add(next_id)
                    next_id += 1
            return items

        gt  = _normalize_gt(ground_truth)
        fds = _normalize_findings(findings)

        # 若输入没有 id，就自动按顺序补上（并避免冲突）
        gt  = _ensure_ids(gt)
        fds = _ensure_ids(fds)

        # 构造 prompt
        prompt = build_match_prompt(gt, fds)
        match_obj = call_openai_match(prompt, eval_model=eval_model, api_key=api_key)
        if match_obj is None:
            debug_path = synth_path.parent / f"{eval_mode}_eval" / detect_model / synth_path.stem / "match_llm_resp_debug.json"
            save_json({"error": "llm_none"}, str(debug_path))
            return synth_path, False, "LLM returned None"

        ok, msg = validate_match_obj(match_obj, gt_len=len(gt))
        if not ok:
            debug_path = synth_path.parent / f"{eval_mode}_eval" / detect_model / synth_path.stem / "match_llm_resp_debug.json"
            save_json({"raw_llm_obj": match_obj, "validate_msg": msg}, str(debug_path))
            return synth_path, False, f"validate failed: {msg}"

        # 将原 ground_truth 的关键信息合并进每条 match 里
        matches = match_obj.get("matches", [])
        enriched_matches = []
        for i, m in enumerate(matches):
            g = gt[i] if i < len(gt) else {}
            enriched = dict(m)  # 复制原结果
            # 附加 GT 字段（直出而非加前缀，便于外部分析）
            enriched["difficulty"] = g.get("difficulty")
            enriched["section_location"] = g.get("section_location")
            enriched["needs_cross_section"] = g.get("needs_cross_section")
            enriched["error_location"] = g.get("error_location")
            enriched["explanation"] = g.get("explanation")
            enriched_matches.append(enriched)

        hit_cnt = sum(1 for m in enriched_matches if m.get("matched") is True)
        detection_rate = (hit_cnt / len(gt)) if len(gt) > 0 else 0.0

        # --- per-paper precision & f1 (from matches + finding_count) ---
        matched_pred_set = set()
        for m in enriched_matches:
            idxs = m.get("matched_pred_indices", [])
            if not isinstance(idxs, list):
                continue
            for x in idxs:
                if isinstance(x, int):
                    matched_pred_set.add(x)
        matched_pred_count = len(matched_pred_set)

        precision = (matched_pred_count / len(fds)) if len(fds) > 0 else 0.0
        f1_score = (
            2.0 * precision * detection_rate / (precision + detection_rate)
            if (precision + detection_rate) > 0 else 0.0
        )
        # -------------------------------------------------------------

        out_obj = {
            "eval_for_detector": True,
            "created_at": datetime.utcnow().isoformat() + "Z",
            "source_detect_file": str(in_path),
            "detect_model_tag": detect_model,
            "eval_model_used": eval_model,
            "ground_truth_count": len(gt),
            "finding_count": len(fds),
            "matched_pred_count": matched_pred_count,
            "matches": enriched_matches,
            "detection_rate": detection_rate,
            "precision": precision,
            "f1_score": f1_score,
        }

        save_json(out_obj, str(out_path))
        print(f"[DONE] Saved: {out_path}")
        return synth_path, True, "ok"

    finally:
        try:
            if inprog.exists():
                inprog.unlink()
        except Exception:
            pass

# ================ 并发调度 ================
async def bounded_worker(
    sem: asyncio.Semaphore,
    loop,
    synth_path: Path,
    detect_model: str,
    eval_model: str,
    eval_mode: str,
    api_key: Optional[str],
    overwrite: bool
):
    async with sem:
        return await loop.run_in_executor(
            None,
            process_one_dir_sync,
            synth_path,
            detect_model,
            eval_model,
            eval_mode,
            api_key,
            overwrite,
        )

async def run_batch(
    root_dir: Path,
    concurrency: int,
    detect_model: str,
    eval_model: str,
    eval_mode: str,
    synth_model: str,
    api_key: Optional[str],
    overwrite: bool = True
) -> Dict[str, Any]:
    """
    批量模式：
    - 在 root_dir 下查找所有 paper_synth_*.json
    - 用 synth_model 过滤（例如 gpt-5-2025-08-07）
    - 对每个匹配的 synth_json 做一次 eval
    """
    synth_files = find_synth_files(root_dir, synth_model)
    if not synth_files:
        return {"total": 0, "ok": 0, "fail": [], "skipped_done": 0}

    planned, skipped_done = [], 0
    for sp in synth_files:
        _, out_path, _ = compute_eval_paths(
            synth_path=sp,
            detect_model=detect_model,
            eval_model=eval_model,
            eval_mode=eval_mode,
        )
        if not overwrite and out_path.exists():
            skipped_done += 1
            continue
        planned.append(sp)

    if not planned:
        return {"total": 0, "ok": skipped_done, "fail": [], "skipped_done": skipped_done}

    sem  = asyncio.Semaphore(concurrency)
    loop = asyncio.get_running_loop()
    tasks = [
        bounded_worker(sem, loop, sp, detect_model, eval_model, eval_mode, api_key, overwrite)
        for sp in planned
    ]

    results = []
    iterator = asyncio.as_completed(tasks)
    if tqdm is not None:
        iterator = tqdm(iterator, total=len(tasks), desc=f"{eval_mode.capitalize()} Eval Matching")
    for coro in iterator:
        results.append(await coro)

    ok_cnt = sum(1 for (_p, ok, msg) in results if ok)
    fail   = [(str(p), msg) for (p, ok, msg) in results if not ok and msg != "skip_done"]
    return {"total": len(results), "ok": ok_cnt + skipped_done, "fail": fail, "skipped_done": skipped_done}

# ================ 数据集级统计 & eval_log.json ================
def compute_dataset_stats(
    root_dir: Path,
    detect_model: str,
    eval_model: str,
    eval_mode: str,
    synth_model: str,
) -> Dict[str, Any]:
    """
    数据集级统计（不重跑 LLM）：
    - avg_detection_rate: 平均 detection_rate（来自 eval json；若缺失则从 matches/gt_count 推导） => Recall
    - avg_finding_count: 平均 findings 数（finding_count）
    - avg_matched_pred_count: 平均“匹配成功的预测条数（去重后；若缺失则从 matches 推导）”
    - avg_precision: 平均 precision（若缺失则 matched_pred_count / finding_count）
    - avg_f1_score: 用 avg_precision 与 avg_detection_rate 计算的 F1(avgP, avgR)
    - macro_f1: 平均 per-paper F1（若缺失则用 per-paper P/R 推导）
    - avg_gt_count: 平均 ground-truth 错误数
    """
    root_dir = Path(root_dir)
    synth_files = find_synth_files(root_dir, synth_model)
    paper_total = len(synth_files)

    detection_rates: List[float] = []
    finding_counts: List[int] = []
    matched_pred_counts: List[int] = []
    precisions: List[float] = []
    f1_scores: List[float] = []
    gt_counts: List[int] = []

    def _derive_hit_cnt_from_matches(obj: Dict[str, Any]) -> Optional[int]:
        matches = obj.get("matches", None)
        if not isinstance(matches, list):
            return None
        hit = 0
        for m in matches:
            if isinstance(m, dict) and m.get("matched") is True:
                hit += 1
        return hit

    def _derive_matched_pred_count_from_matches(obj: Dict[str, Any]) -> Optional[int]:
        matches = obj.get("matches", None)
        if not isinstance(matches, list):
            return None
        s = set()
        for m in matches:
            if not isinstance(m, dict):
                continue
            idxs = m.get("matched_pred_indices", [])
            if not isinstance(idxs, list):
                continue
            for x in idxs:
                if isinstance(x, int):
                    s.add(x)
        return len(s)

    paper_with_eval = 0
    for sp in synth_files:
        _, eval_path, _ = compute_eval_paths(
            synth_path=sp,
            detect_model=detect_model,
            eval_model=eval_model,
            eval_mode=eval_mode,
        )
        if not eval_path.exists():
            continue

        try:
            obj = load_json(str(eval_path))
        except Exception as e:
            print(f"[WARN] fail to load eval json: {eval_path} — {e}", file=sys.stderr)
            continue

        paper_with_eval += 1

        # ---- gt_count / finding_count（基础）----
        gc = obj.get("ground_truth_count", None)
        if not isinstance(gc, int):
            # 没法可靠推导，按 0 处理
            gc = 0
        fc = obj.get("finding_count", None)
        if not isinstance(fc, int):
            fc = 0

        gt_counts.append(gc)
        finding_counts.append(fc)

        # ---- matched_pred_count（若缺失则从 matches 推导）----
        mpc = obj.get("matched_pred_count", None)
        if not isinstance(mpc, int):
            mpc = _derive_matched_pred_count_from_matches(obj)
        if not isinstance(mpc, int):
            mpc = 0
        matched_pred_counts.append(mpc)

        # ---- detection_rate（若缺失则从 matches/gt_count 推导）----
        dr = obj.get("detection_rate", None)
        if not isinstance(dr, (int, float)):
            hit_cnt = _derive_hit_cnt_from_matches(obj)
            if isinstance(hit_cnt, int) and gc > 0:
                dr = float(hit_cnt) / float(gc)
            else:
                dr = 0.0
        else:
            dr = float(dr)
        detection_rates.append(dr)

        # ---- precision（若缺失则 mpc/fc 推导）----
        pr = obj.get("precision", None)
        if not isinstance(pr, (int, float)):
            pr = (float(mpc) / float(fc)) if fc > 0 else 0.0
        else:
            pr = float(pr)
        precisions.append(pr)

        # ---- per-paper f1_score（若缺失则用 per-paper P/R 推导）----
        fs = obj.get("f1_score", None)
        if isinstance(fs, (int, float)):
            fs = float(fs)
        else:
            fs = (2.0 * pr * dr / (pr + dr)) if (pr + dr) > 0 else 0.0
        f1_scores.append(fs)

    avg_detection_rate = (sum(detection_rates) / len(detection_rates)) if detection_rates else 0.0
    avg_finding_count = (sum(finding_counts) / len(finding_counts)) if finding_counts else 0.0
    avg_matched_pred_count = (sum(matched_pred_counts) / len(matched_pred_counts)) if matched_pred_counts else 0.0
    avg_precision = (sum(precisions) / len(precisions)) if precisions else 0.0

    avg_f1_score = (
        2.0 * avg_precision * avg_detection_rate / (avg_precision + avg_detection_rate)
        if (avg_precision + avg_detection_rate) > 0 else 0.0
    )
    macro_f1 = (sum(f1_scores) / len(f1_scores)) if f1_scores else 0.0
    avg_gt_count = (sum(gt_counts) / len(gt_counts)) if gt_counts else 0.0

    # 兼容你原来的字段：avg_match_success_rate（= avg_matched_pred_count / avg_finding_count）
    avg_match_success_rate = (avg_matched_pred_count / avg_finding_count) if avg_finding_count > 0 else 0.0

    stats = {
        "detect_model": detect_model,
        "eval_model": eval_model,
        "eval_mode": eval_mode,
        "synth_model": synth_model,
        "root_dir": str(root_dir),
        "timestamp": datetime.utcnow().isoformat() + "Z",
        "paper_total": paper_total,
        "paper_with_eval": paper_with_eval,
        "avg_detection_rate": avg_detection_rate,  # Recall (macro-avg over papers)
        "avg_precision": avg_precision,            # Precision (macro-avg over papers)
        "avg_f1_score": avg_f1_score,              # F1(avgP, avgR)
        "macro_f1": macro_f1,                      # avg(F1_i) over papers
        "avg_finding_count": avg_finding_count,
        "avg_matched_pred_count": avg_matched_pred_count,
        "avg_match_success_rate": avg_match_success_rate,
        "avg_gt_count": avg_gt_count,
    }
    return stats


def update_eval_log(root_dir: Path, stats: Dict[str, Any]):
    """
    在 root_dir 下维护一个 eval_log.json，结构大致为：
    {
      "runs": [
        { ... 一条组合的统计 ... },
        ...
      ]
    }

    若已有相同 (detect_model, eval_model, eval_mode, synth_model) 的条目，则覆盖那一条；
    否则追加。
    """
    root_dir = Path(root_dir)
    log_path = root_dir / "eval_log.json"

    if log_path.exists():
        try:
            log_obj = load_json(str(log_path))
        except Exception as e:
            print(f"[WARN] failed to load existing eval_log.json, will overwrite: {e}", file=sys.stderr)
            log_obj = {}
    else:
        log_obj = {}

    runs = log_obj.get("runs", [])
    if not isinstance(runs, list):
        runs = []

    key_fields = ("detect_model", "eval_model", "eval_mode", "synth_model")
    def same_run(a: Dict[str, Any], b: Dict[str, Any]) -> bool:
        return all(a.get(k) == b.get(k) for k in key_fields)

    updated = False
    for i, r in enumerate(runs):
        if isinstance(r, dict) and same_run(r, stats):
            runs[i] = stats
            updated = True
            break
    if not updated:
        runs.append(stats)

    log_obj["runs"] = runs
    save_json(log_obj, str(log_path))
    print(
        f"[EVAL_LOG] updated {log_path} — "
        f"avg_detection_rate={stats.get('avg_detection_rate', 0):.4f} "
        f"avg_precision={stats.get('avg_precision', 0):.4f} "
        f"avg_f1_score={stats.get('avg_f1_score', 0):.4f} "
        f"macro_f1={stats.get('macro_f1', 0):.4f} "
        f"avg_gt_count={stats.get('avg_gt_count', 0):.4f} "
        f"avg_finding_count={stats.get('avg_finding_count', 0):.4f} "
        f"avg_matched_pred_count={stats.get('avg_matched_pred_count', 0):.4f} "
        f"avg_match_success_rate={stats.get('avg_match_success_rate', 0):.4f} "
        f"over {stats.get('paper_with_eval', 0)}/{stats.get('paper_total', 0)} papers"
    )


# ================ CLI ================
def main():
    parser = argparse.ArgumentParser(
        description="Evaluate detector outputs: adjudicate GT-vs-findings matches with EVAL_MODEL and save detection rate."
    )
    parser.add_argument("--root_dir", type=str, default="/mnt/parallel_ssd/home/zdhs0006/mlrbench/download/downloads/test2")
    parser.add_argument("--concurrency", "-j", type=int, default=8)
    parser.add_argument("--detect_model", type=str, default=DEFAULT_DETECT_MODEL)
    parser.add_argument("--eval_model", type=str, default=DEFAULT_EVAL_MODEL)
    parser.add_argument("--eval_mode", type=str, default=DEFAULT_EVAL_MODE,
                        help="决定 detect 目录前缀，例如 'section' => section_detect")
    parser.add_argument("--synth_model", type=str, default=DEFAULT_SYNTH_MODEL,
                        help="按文件名过滤 paper_synth_{SYNTH_MODEL}_*.json")
    parser.add_argument("--overwrite", action="store_true",
                        help="Force overwrite existing outputs (默认 True：总是覆盖已有 eval 结果)" )

    args = parser.parse_args()

    if not OPENAI_API_KEY:
        print("[FATAL] OPENAI_API_KEY is not set.", file=sys.stderr); sys.exit(1)

    root = Path(args.root_dir).expanduser().resolve()
    if not root.exists():
        print(f"[FATAL] root_dir not found: {root}", file=sys.stderr); sys.exit(1)

    print(
        f"[INFO] root={root}  concurrency={args.concurrency}  "
        f"detect_model={args.detect_model}  eval_model={args.eval_model}  "
        f"eval_mode={args.eval_mode}  synth_model={args.synth_model}  "
        f"overwrite={args.overwrite}"
    )

    loop = asyncio.get_event_loop()
    summary = loop.run_until_complete(
        run_batch(
            root_dir=root,
            concurrency=args.concurrency,
            detect_model=args.detect_model,
            eval_model=args.eval_model,
            eval_mode=args.eval_mode,
            synth_model=args.synth_model,
            api_key=OPENAI_API_KEY,
            overwrite=args.overwrite,
        )
    )

    print(f"[DONE] total={summary['total']}  ok={summary['ok']}  fail={len(summary['fail'])}  skipped_done={summary.get('skipped_done', 0)}")
    if summary["fail"]:
        print("[FAILED ITEMS]")
        for i, (path, msg) in enumerate(summary["fail"], 1):
            print(f"  {i}. {path} :: {msg}")

    # ---- 计算整个数据集平均 detection_rate，并写入 eval_log.json ----
    stats = compute_dataset_stats(
        root_dir=root,
        detect_model=args.detect_model,
        eval_model=args.eval_model,
        eval_mode=args.eval_mode,
        synth_model=args.synth_model,
    )
    update_eval_log(root, stats)

if __name__ == "__main__":
    main()
