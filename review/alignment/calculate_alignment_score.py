#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Alignment Score Aggregator

Aggregates COVERAGE alignment metrics over a folder of paper subfolders.
This script reads alignment evaluation results from multiple papers and computes
aggregated statistics (mean, std, min, max, count) for each alignment metric.

The aggregator:
- Scans all paper subfolders under input directory
- Locates alignment JSON files produced by eval_alignment.py
- Groups results by configuration key
- Computes statistics for each metric across papers
- Optionally writes summary JSON and/or JSONL rows

Usage Examples:
    # Aggregate all alignment results
    python calculate_alignment_score.py \
        --input_dir /path/to/papers \
        --judge_model gemini-2.5-pro \
        --model_tag v1

    # Aggregate with filters
    python calculate_alignment_score.py \
        --input_dir /path/to/papers \
        --judge_model gemini-2.5-pro \
        --model_tag v1 \
        --review_agent paper_audit \
        --ai_model gpt-5-2025-08-07 \
        --ai_review_file review_output_all.json


Key Arguments:
    --input_dir: Root folder containing paper subfolders. Default: /mnt/parallel_ssd/home/zdhs0006/ACL/data/ICLR_26
    --judge_model: Judge model folder under reviews/alignment_judge/. Default: gemini-2.5-pro
    --model_tag: alignment_{tag}.json tag to aggregate. Default: v1
    --review_agent: Filter: runs.config.review_agent. Default: "" (no filter)
    --ai_model: Filter: runs.config.ai_model. Default: "" (no filter)
    --ai_review_file: Filter: runs.config.ai_review_file. Default: "" (no filter)
    --out_dir: Output directory for summary and rows. Default: "" (write next to input_dir)
    --out_prefix: Prefix for output files. Default: coverage_agg
    --write_rows: Write JSONL rows (one row per paper per run). Default: False

Input:
    - Alignment files: <paper_dir>/reviews/alignment_judge/{judge_model}/paper_origin/alignment_{model_tag}.json

Output:
    - Summary JSON: {out_dir}/{out_prefix}_summary.json (if out_dir specified)
    - Rows JSONL: {out_dir}/{out_prefix}_rows.jsonl (if --write_rows and out_dir specified)

Metrics Aggregated:
    - strength_coverage_recall: Mean, std, min, max, count
    - weakness_coverage_recall: Mean, std, min, max, count
    - ai_extra_major_points_rate: Mean, std, min, max, count
    - symmetric_coverage_similarity: Mean, std, min, max, count

The aggregator groups results by config_key, allowing separate aggregation for different
experimental configurations.
"""

from __future__ import annotations

import argparse
import json
import math
from pathlib import Path
from typing import Any, Dict, List, Optional, Tuple
from datetime import datetime


METRIC_KEYS = [
    "strength_coverage_recall",
    "weakness_coverage_recall",
    "ai_extra_major_points_rate",
    "symmetric_coverage_similarity",
]


# ---------------- CLI ----------------
def parse_args() -> argparse.Namespace:
    p = argparse.ArgumentParser(description="Aggregate review coverage-alignment metrics over a folder")
    p.add_argument(
        "--input_dir",
        default="/mnt/parallel_ssd/home/zdhs0006/ACL/data/ICLR_26",
        help="Root folder containing paper subfolders",
    )
    p.add_argument(
        "--judge_model",
        default="gemini-2.5-pro",
        help="Judge model folder under reviews/alignment_judge/",
    )
    p.add_argument(
        "--model_tag",
        default="v3",
        help="alignment_<tag>.json tag to aggregate",
    )

    # Optional filters (apply to runs.config fields)
    p.add_argument("--review_agent", default="", help="Filter: runs.config.review_agent")
    p.add_argument("--ai_model", default="", help="Filter: runs.config.ai_model")
    p.add_argument("--ai_review_file", default="", help="Filter: runs.config.ai_review_file")

    # Outputs
    p.add_argument(
        "--out_dir",
        default="",
        help="If set, write summary + rows into this directory; otherwise write next to input_dir.",
    )
    p.add_argument(
        "--out_prefix",
        default="coverage_agg",
        help="Prefix for output files",
    )
    p.add_argument(
        "--write_rows",
        action="store_true",
        help="Write JSONL rows (one row per paper per run).",
    )
    return p.parse_args()


# ---------------- utils ----------------
def load_json(path: Path) -> Dict[str, Any]:
    return json.loads(path.read_text(encoding="utf-8", errors="ignore") or "{}")


def safe_float01(x: Any) -> Optional[float]:
    """
    Parse float, ensure finite, clamp to [0,1].
    NOTE: runner should already round to 2 decimals; we don't force rounding here.
    """
    try:
        v = float(x)
        if math.isnan(v) or math.isinf(v):
            return None
        if v < 0.0:
            v = 0.0
        if v > 1.0:
            v = 1.0
        return v
    except Exception:
        return None


def mean_std(vals: List[float]) -> Tuple[Optional[float], Optional[float]]:
    if not vals:
        return None, None
    m = sum(vals) / len(vals)
    if len(vals) == 1:
        return m, 0.0
    var = sum((v - m) ** 2 for v in vals) / (len(vals) - 1)
    return m, math.sqrt(var)


def min_max(vals: List[float]) -> Tuple[Optional[float], Optional[float]]:
    if not vals:
        return None, None
    return min(vals), max(vals)


def alignment_file_for(paper_dir: Path, judge_model: str, tag: str) -> Path:
    return paper_dir / "reviews" / "alignment_judge" / judge_model / "paper_origin" / f"alignment_{tag}.json"


def match_filters(config: Dict[str, Any], args: argparse.Namespace) -> bool:
    if args.review_agent and config.get("review_agent") != args.review_agent:
        return False
    if args.ai_model and config.get("ai_model") != args.ai_model:
        return False
    if args.ai_review_file and config.get("ai_review_file") != args.ai_review_file:
        return False
    return True


def normalize_config_for_group(config: Dict[str, Any]) -> Dict[str, Any]:
    # Keep only stable identifiers for grouping + metric tag if present
    keys = [
        "metric",
        "judge_model",
        "review_agent",
        "ai_model",
        "ai_review_file",
        "temperature",
        "max_tokens",
        "timeout_s",
    ]
    return {k: config.get(k) for k in keys if k in config}


# ---------------- parse runs ----------------
def iter_runs(obj: Dict[str, Any]) -> List[Dict[str, Any]]:
    """
    Preferred: obj["runs"] is a list of run objects with {"config", "config_key", "alignment", ...}
    Legacy fallback: top-level may contain "alignment" and config-like fields; wrap into one pseudo-run.
    """
    runs = obj.get("runs")
    if isinstance(runs, list) and runs:
        return [r for r in runs if isinstance(r, dict)]

    # Legacy fallback (single-run files)
    if isinstance(obj.get("alignment"), dict):
        pseudo_config = {
            "judge_model": obj.get("judge_model"),
            "review_agent": obj.get("review_agent"),
            "ai_model": obj.get("ai_model"),
            "ai_review_file": obj.get("ai_review_file"),
            "metric": obj.get("metric") or "unknown",
        }
        return [{
            "config": pseudo_config,
            "config_key": None,
            "alignment": obj.get("alignment"),
            "generated_at": obj.get("generated_at"),
            "inputs": obj.get("inputs"),
            "multi_review": obj.get("multi_review"),
        }]
    return []


# ---------------- main aggregation ----------------
def main() -> None:
    args = parse_args()
    root = Path(args.input_dir).expanduser().resolve()
    if not root.exists() or not root.is_dir():
        raise SystemExit(f"Input directory not found: {root}")

    tag = args.model_tag
    judge_model = args.judge_model

    paper_dirs = [p for p in sorted(root.iterdir()) if p.is_dir()]
    rows: List[Dict[str, Any]] = []

    missing = 0
    files_read = 0

    for pd in paper_dirs:
        fp = alignment_file_for(pd, judge_model, tag)
        if not fp.exists():
            missing += 1
            continue

        try:
            obj = load_json(fp)
            files_read += 1
        except Exception:
            continue

        paper_name = obj.get("paper") or pd.name
        for r in iter_runs(obj):
            config = r.get("config") if isinstance(r.get("config"), dict) else {}
            if not match_filters(config, args):
                continue

            align = r.get("alignment") if isinstance(r.get("alignment"), dict) else {}
            metrics = {k: safe_float01(align.get(k)) for k in METRIC_KEYS}

            # Only keep if all required metrics are present
            if any(metrics[k] is None for k in METRIC_KEYS):
                continue

            rows.append({
                "paper": paper_name,
                "paper_dir": str(pd),
                "alignment_file": str(fp),
                "config_key": r.get("config_key"),
                "config": normalize_config_for_group(config),
                **metrics,
            })

    # Group by config_key if available, else by normalized config string
    groups: Dict[str, List[Dict[str, Any]]] = {}
    for row in rows:
        ck = row.get("config_key")
        if isinstance(ck, str) and ck:
            gid = f"config_key:{ck}"
        else:
            gid = "config:" + json.dumps(row["config"], sort_keys=True, ensure_ascii=False)
        groups.setdefault(gid, []).append(row)

    def summarize_group(grows: List[Dict[str, Any]]) -> Dict[str, Any]:
        out: Dict[str, Any] = {"count": len(grows)}
        out["config"] = grows[0].get("config")
        out["config_key"] = grows[0].get("config_key")
        for k in METRIC_KEYS:
            vals = [float(r[k]) for r in grows]
            m, sd = mean_std(vals)
            mn, mx = min_max(vals)
            out[k] = {
                "mean": m,
                "std": sd,
                "min": mn,
                "max": mx,
            }
        return out

    per_config = {gid: summarize_group(grows) for gid, grows in groups.items()}

    # Global summary (across ALL rows that passed filters)
    global_summary: Dict[str, Any] = {
        "count": len(rows),
        "files_read": files_read,
        "papers_with_missing_alignment_file": missing,
    }
    for k in METRIC_KEYS:
        vals = [float(r[k]) for r in rows]
        m, sd = mean_std(vals)
        mn, mx = min_max(vals)
        global_summary[k] = {"mean": m, "std": sd, "min": mn, "max": mx}

    summary_obj: Dict[str, Any] = {
        "generated_at": datetime.utcnow().isoformat() + "Z",
        "input_dir": str(root),
        "judge_model": judge_model,
        "model_tag": tag,
        "filters": {
            "review_agent": args.review_agent or None,
            "ai_model": args.ai_model or None,
            "ai_review_file": args.ai_review_file or None,
        },
        "metrics": METRIC_KEYS,
        "global": global_summary,
        "per_config": per_config,
    }

    # Write outputs
    out_dir = Path(args.out_dir).expanduser().resolve() if args.out_dir else root
    out_dir.mkdir(parents=True, exist_ok=True)

    prefix = args.out_prefix
    summary_path = out_dir / f"{prefix}_summary_{sanitize(prefix=prefix, judge_model=judge_model, tag=tag, args=args)}.json"
    summary_path.write_text(json.dumps(summary_obj, ensure_ascii=False, indent=2), encoding="utf-8")

    if args.write_rows:
        rows_path = out_dir / f"{prefix}_rows_{sanitize(prefix=prefix, judge_model=judge_model, tag=tag, args=args)}.jsonl"
        with rows_path.open("w", encoding="utf-8") as f:
            for r in rows:
                f.write(json.dumps(r, ensure_ascii=False) + "\n")

    print(f"[DONE] aggregated_rows={len(rows)} groups={len(groups)} missing_files={missing}")
    print(f"[WRITE] {summary_path}")
    if args.write_rows:
        print(f"[WRITE] {rows_path}")


def sanitize(prefix: str, judge_model: str, tag: str, args: argparse.Namespace) -> str:
    parts = [
        f"judge={judge_model}",
        f"tag={tag}",
    ]
    if args.review_agent:
        parts.append(f"agent={args.review_agent}")
    if args.ai_model:
        parts.append(f"aimodel={args.ai_model}")
    if args.ai_review_file:
        parts.append(f"aifile={args.ai_review_file}")
    return "_".join("".join(c if c.isalnum() or c in ("-", "_", ".", "=") else "_" for c in p) for p in parts)


if __name__ == "__main__":
    main()
