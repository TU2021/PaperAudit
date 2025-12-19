#!/usr/bin/env python3
# -*- coding: utf-8 -*-

from __future__ import annotations
import argparse
import json
import sys
from pathlib import Path
from datetime import datetime
from typing import Any, Dict, List, Optional, Tuple


# ======================
# Defaults
# ======================
DEFAULT_ROOT_DIR     = "data/ICML_30"
DEFAULT_DETECT_MODEL = "gpt-5-2025-08-07"
DEFAULT_SYNTH_MODEL  = "gpt-5-2025-08-07"
DEFAULT_EVAL_MODEL   = "gpt-5-2025-08-07"
DEFAULT_EVAL_MODE    = "standard_wo_memory"  # folder: <eval_mode>_eval


# ======================
# IO utils
# ======================
def load_json(p: Path) -> Dict[str, Any]:
    with p.open("r", encoding="utf-8") as f:
        return json.load(f)

def save_json(obj: Any, p: Path) -> None:
    p.parent.mkdir(parents=True, exist_ok=True)
    with p.open("w", encoding="utf-8") as f:
        json.dump(obj, f, ensure_ascii=False, indent=2)

def eprint(*args, **kwargs):
    print(*args, file=sys.stderr, **kwargs)

def _utc_now() -> str:
    return datetime.utcnow().isoformat() + "Z"

def _tag(name: str) -> str:
    return (name or "").replace("/", "_")

def safe_str(x: Any, default: str = "") -> str:
    if isinstance(x, str) and x.strip():
        return x.strip()
    if x is None:
        return default
    s = str(x).strip()
    return s if s else default


# ======================
# Path helpers (aligned with your eval script)
# ======================
def compute_eval_path(
    paper_dir: Path,
    synth_stem: str,         # e.g. paper_synth_o4-mini
    detect_model: str,
    eval_mode: str,          # e.g. deep / deep_standard
    eval_model: str,         # e.g. gpt-5.1
) -> Path:
    # <paper_dir>/<eval_mode>_eval/<detect_model>/<synth_stem>/eval_{eval_model_tag}.json
    return paper_dir / f"{eval_mode}_eval" / detect_model / synth_stem / f"eval_{_tag(eval_model)}.json"

def find_synth_files(root_dir: Path, synth_model: str) -> List[Path]:
    """
    Prefer exact filename: paper_synth_{synth_model}.json
    Fallback: any paper_synth_*.json that contains synth_model in its name,
    BUT avoid those inside *_detect / *_eval trees when possible.
    """
    root_dir = Path(root_dir)

    exact = f"paper_synth_{synth_model}.json"
    hits = sorted(root_dir.rglob(exact))
    if hits:
        return hits

    all_synth = sorted(root_dir.rglob("paper_synth_*.json"))
    out = []
    for p in all_synth:
        if synth_model and synth_model not in p.name:
            continue
        parts = p.parts
        if any(part.endswith("_detect") or part.endswith("_eval") for part in parts):
            continue
        out.append(p)
    return out


# ======================
# GT extraction (IMPORTANT FIXED)
# ======================
def extract_gt_items(synth_obj: Dict[str, Any]) -> List[Dict[str, Any]]:
    """
    Returns list of GT dicts each with:
      corruption_type, difficulty, location, needs_cross_section

    Priority:
      1) synth_obj["apply_results"] with applied == True  (authoritative)
      2) synth_obj["ground_truth"]
      3) synth_obj["audit_log"]["edits"]

    Filtering:
      - keep only items with non-empty corruption_type, difficulty, location.
    """
    src_list: List[Dict[str, Any]] = []

    # (1) apply_results (only applied=True)
    apply_results = synth_obj.get("apply_results", None)
    if isinstance(apply_results, list):
        for g in apply_results:
            if isinstance(g, dict) and g.get("applied") is True:
                src_list.append(g)

    # fallback only if apply_results not available (or empty)
    if not src_list:
        items = synth_obj.get("ground_truth", None)
        if isinstance(items, list):
            src_list = [g for g in items if isinstance(g, dict)]
        else:
            audit = synth_obj.get("audit_log", {})
            edits = audit.get("edits", [])
            src_list = [g for g in edits if isinstance(g, dict)] if isinstance(edits, list) else []

    out: List[Dict[str, Any]] = []
    for g in src_list:
        corruption_type = safe_str(g.get("corruption_type") or g.get("type"), "")
        difficulty      = safe_str(g.get("difficulty"), "")

        location = g.get("location")
        if location is None:
            location = g.get("section_location")
        location = safe_str(location, "")

        needs_cross_section = bool(g.get("needs_cross_section", False))

        # ---- strict filtering: must be present ----
        if not corruption_type or not difficulty or not location:
            continue

        out.append({
            "corruption_type": corruption_type,
            "difficulty": difficulty,
            "location": location,
            "needs_cross_section": needs_cross_section,
        })
    return out


# ======================
# Matches extraction
# ======================
def extract_matched_flags(eval_obj: Dict[str, Any], gt_len: int) -> List[bool]:
    """
    Turn eval_obj["matches"] into a list[bool] aligned to GT index.
    Supports gt_index being 0-based or 1-based; if absent, fallback to sequential.
    """
    matches = eval_obj.get("matches", [])
    flags = [False] * gt_len
    if not isinstance(matches, list) or gt_len <= 0:
        return flags

    next_seq = 0
    for m in matches:
        if not isinstance(m, dict):
            continue
        matched = bool(m.get("matched", False))

        gi = m.get("gt_index", None)
        idx: Optional[int] = None
        if isinstance(gi, int):
            if 0 <= gi < gt_len:
                idx = gi
            elif 1 <= gi <= gt_len:
                idx = gi - 1

        if idx is None:
            if next_seq < gt_len:
                idx = next_seq
                next_seq += 1

        if idx is not None and 0 <= idx < gt_len:
            flags[idx] = bool(flags[idx] or matched)

    return flags


# ======================
# Macro+Micro breakdown aggregation
# ======================
class MacroAgg:
    """
    Store per-bucket stats across papers:
      - macro_rate = average over papers of (matched_in_bucket / total_in_bucket)
        for papers where total_in_bucket > 0.
      - micro_rate = pooled matched / pooled total (matched / total).
    """
    def __init__(self):
        self.sum_rate = 0.0
        self.paper_count = 0
        self.total = 0
        self.matched = 0

    def add_paper(self, matched: int, total: int):
        if total <= 0:
            return
        self.sum_rate += (matched / total)
        self.paper_count += 1
        self.total += total
        self.matched += matched

    def to_dict(self) -> Dict[str, Any]:
        macro_rate = (self.sum_rate / self.paper_count) if self.paper_count > 0 else 0.0
        micro_rate = (self.matched / self.total) if self.total > 0 else 0.0
        return {
            "paper_count": self.paper_count,
            "total": self.total,
            "matched": self.matched,
            "rate": micro_rate,          # IMPORTANT: default to micro (matched/total)
            "macro_rate": macro_rate,    # keep macro for reference
        }

def _sorted_dict(d: Dict[str, Any]) -> Dict[str, Any]:
    return dict(sorted(d.items(), key=lambda x: x[0]))


def compute_breakdown_over_dataset_macro(
    root_dir: Path,
    detect_model: str,
    eval_model: str,
    eval_mode: str,
    synth_model: str,
    max_papers: Optional[int] = None,
) -> Dict[str, Any]:
    synth_files = find_synth_files(root_dir, synth_model)
    if max_papers is not None:
        synth_files = synth_files[:max_papers]

    paper_total = len(synth_files)
    paper_with_eval = 0

    # macro overall across papers (kept for reference)
    overall_sum_rate = 0.0
    overall_paper_count = 0

    # pooled counts (micro support)
    gt_total = 0
    matched_total = 0

    by_corruption_type: Dict[str, MacroAgg] = {}
    by_difficulty: Dict[str, MacroAgg] = {}
    by_location: Dict[str, MacroAgg] = {}
    by_needs_xsec: Dict[str, MacroAgg] = {}

    def _get_agg(table: Dict[str, MacroAgg], key: str) -> MacroAgg:
        if key not in table:
            table[key] = MacroAgg()
        return table[key]

    for sp in synth_files:
        paper_dir = sp.parent
        synth_stem = sp.stem  # paper_synth_...

        # load GT (applied-only)
        try:
            synth_obj = load_json(sp)
        except Exception:
            continue
        gt_items = extract_gt_items(synth_obj)
        n_gt = len(gt_items)

        eval_path = compute_eval_path(
            paper_dir=paper_dir,
            synth_stem=synth_stem,
            detect_model=detect_model,
            eval_mode=eval_mode,
            eval_model=eval_model,
        )
        if not eval_path.exists():
            continue

        try:
            eval_obj = load_json(eval_path)
        except Exception:
            continue

        # sanity alignment: if mismatch, trim to common length to avoid artificial false negatives
        matches = eval_obj.get("matches", [])
        if isinstance(matches, list) and n_gt > 0 and len(matches) > 0 and len(matches) != n_gt:
            aligned = min(n_gt, len(matches))
            gt_items = gt_items[:aligned]
            n_gt = aligned

        paper_with_eval += 1

        flags = extract_matched_flags(eval_obj, gt_len=n_gt)
        paper_matched = sum(1 for x in flags if x)

        # pooled supports (micro)
        gt_total += n_gt
        matched_total += paper_matched

        # macro overall (reference)
        if n_gt > 0:
            overall_sum_rate += (paper_matched / n_gt)
            overall_paper_count += 1

        # per-paper buckets for macro aggregation
        tmp_ct: Dict[str, Tuple[int, int]] = {}
        tmp_diff: Dict[str, Tuple[int, int]] = {}
        tmp_loc: Dict[str, Tuple[int, int]] = {}
        tmp_xsec: Dict[str, Tuple[int, int]] = {}

        def _tmp_add(tmp: Dict[str, Tuple[int, int]], key: str, matched: bool):
            m, t = tmp.get(key, (0, 0))
            t += 1
            if matched:
                m += 1
            tmp[key] = (m, t)

        for g, m in zip(gt_items, flags):
            _tmp_add(tmp_ct, safe_str(g.get("corruption_type"), "unknown"), m)
            _tmp_add(tmp_diff, safe_str(g.get("difficulty"), "unknown"), m)
            _tmp_add(tmp_loc, safe_str(g.get("location"), "unknown"), m)
            _tmp_add(tmp_xsec, "true" if bool(g.get("needs_cross_section")) else "false", m)

        for k, (mm, tt) in tmp_ct.items():
            _get_agg(by_corruption_type, k).add_paper(mm, tt)
        for k, (mm, tt) in tmp_diff.items():
            _get_agg(by_difficulty, k).add_paper(mm, tt)
        for k, (mm, tt) in tmp_loc.items():
            _get_agg(by_location, k).add_paper(mm, tt)
        for k, (mm, tt) in tmp_xsec.items():
            _get_agg(by_needs_xsec, k).add_paper(mm, tt)

    # IMPORTANT: overall micro (what you asked: detected_total / total)
    overall_detection_rate = (matched_total / gt_total) if gt_total > 0 else 0.0
    # keep macro for reference/debug
    overall_detection_rate_macro = (overall_sum_rate / overall_paper_count) if overall_paper_count > 0 else 0.0

    def _finalize_table(table: Dict[str, MacroAgg]) -> Dict[str, Any]:
        return _sorted_dict({k: agg.to_dict() for k, agg in table.items()})

    return {
        "timestamp": _utc_now(),
        "root_dir": str(root_dir),

        "paper_total": paper_total,
        "paper_with_eval": paper_with_eval,

        # pooled supports
        "gt_total": gt_total,
        "matched_total": matched_total,

        # IMPORTANT: micro overall
        "overall_detection_rate": overall_detection_rate,
        # reference: macro overall
        "overall_detection_rate_macro": overall_detection_rate_macro,
        "overall_paper_count_for_macro": overall_paper_count,

        "breakdown": {
            # Each bucket dict contains:
            #   rate (micro = matched/total) + macro_rate (avg over papers)
            "corruption_type": _finalize_table(by_corruption_type),
            "difficulty": _finalize_table(by_difficulty),
            "location": _finalize_table(by_location),
            "needs_cross_section": _finalize_table(by_needs_xsec),
        },
    }


# ======================
# eval_log_detail.json update
# ======================
def update_eval_log_detail(root_dir: Path, run_obj: Dict[str, Any], remove: bool = False) -> Path:
    log_path = Path(root_dir) / "eval_log_detail.json"

    if log_path.exists():
        try:
            log_obj = load_json(log_path)
        except Exception as e:
            eprint(f"[WARN] failed to load existing eval_log_detail.json, will overwrite: {e}")
            log_obj = {}
    else:
        log_obj = {}

    runs = log_obj.get("runs", [])
    if not isinstance(runs, list):
        runs = []

    key_fields = ("detect_model", "eval_model", "eval_mode", "synth_model")

    def same_run(a: Dict[str, Any], b: Dict[str, Any]) -> bool:
        return all(a.get(k) == b.get(k) for k in key_fields)

    new_runs = []
    removed = False
    replaced = False
    for r in runs:
        if isinstance(r, dict) and same_run(r, run_obj):
            if remove:
                removed = True
                continue
            new_runs.append(run_obj)
            replaced = True
        else:
            new_runs.append(r)

    if not remove and not replaced:
        new_runs.append(run_obj)

    log_obj["runs"] = new_runs
    log_obj["updated_at"] = _utc_now()
    save_json(log_obj, log_path)

    if remove:
        eprint(f"[EVAL_LOG_DETAIL] {'removed' if removed else 'no matched run to remove'} in {log_path}")
    else:
        eprint(f"[EVAL_LOG_DETAIL] {'replaced' if replaced else 'added'} run into {log_path}")
    return log_path


# ======================
# CLI
# ======================
def main():
    ap = argparse.ArgumentParser(
        description="Compute detection breakdown (GT from apply_results applied==true) and write into eval_log_detail.json."
    )
    ap.add_argument("--root_dir", type=str, default=DEFAULT_ROOT_DIR)
    ap.add_argument("--detect_model", type=str, default=DEFAULT_DETECT_MODEL)
    ap.add_argument("--eval_model", type=str, default=DEFAULT_EVAL_MODEL)
    ap.add_argument("--eval_mode", type=str, default=DEFAULT_EVAL_MODE)
    ap.add_argument("--synth_model", type=str, default=DEFAULT_SYNTH_MODEL)
    ap.add_argument("--max_papers", type=int, default=None)
    ap.add_argument("--remove", action="store_true", help="Remove the matching run from eval_log_detail.json and exit.")

    args = ap.parse_args()
    root_dir = Path(args.root_dir).expanduser().resolve()
    if not root_dir.exists():
        raise SystemExit(f"[FATAL] root_dir not found: {root_dir}")

    base_run = {
        "detect_model": args.detect_model,
        "eval_model": args.eval_model,
        "eval_mode": args.eval_mode,
        "synth_model": args.synth_model,
        "timestamp": _utc_now(),
        "root_dir": str(root_dir),
    }

    if args.remove:
        update_eval_log_detail(root_dir, base_run, remove=True)
        print(json.dumps({"ok": True, "removed": True, "log_path": str(root_dir / "eval_log_detail.json")}, ensure_ascii=False, indent=2))
        return

    result = compute_breakdown_over_dataset_macro(
        root_dir=root_dir,
        detect_model=args.detect_model,
        eval_model=args.eval_model,
        eval_mode=args.eval_mode,
        synth_model=args.synth_model,
        max_papers=args.max_papers,
    )

    run_obj = dict(base_run)
    run_obj.update({
        "paper_total": result["paper_total"],
        "paper_with_eval": result["paper_with_eval"],
        "gt_total": result["gt_total"],
        "matched_total": result["matched_total"],
        "overall_detection_rate": result["overall_detection_rate"],                # micro
        "overall_detection_rate_macro": result["overall_detection_rate_macro"],    # reference
        "overall_paper_count_for_macro": result["overall_paper_count_for_macro"],
        "breakdown": result["breakdown"],
    })

    log_path = update_eval_log_detail(root_dir, run_obj, remove=False)

    print(json.dumps({
        "ok": True,
        "log_path": str(log_path),
        "paper_total": run_obj["paper_total"],
        "paper_with_eval": run_obj["paper_with_eval"],
        "gt_total": run_obj["gt_total"],
        "matched_total": run_obj["matched_total"],
        "overall_detection_rate": run_obj["overall_detection_rate"],
        "overall_detection_rate_macro": run_obj["overall_detection_rate_macro"],
    }, ensure_ascii=False, indent=2))


if __name__ == "__main__":
    main()
