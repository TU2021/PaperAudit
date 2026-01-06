#!/usr/bin/env python3
# -*- coding: utf-8 -*-

from __future__ import annotations
import argparse
import json
from pathlib import Path
from collections import Counter
from typing import Any, Dict, List, Optional


def load_json(p: Path) -> Dict[str, Any]:
    return json.loads(p.read_text(encoding="utf-8", errors="ignore") or "{}")


def safe_get(d: Dict[str, Any], path: List[str], default=None):
    cur: Any = d
    for k in path:
        if not isinstance(cur, dict) or k not in cur:
            return default
        cur = cur[k]
    return cur


def sanitize_filename(name: str) -> str:
    return "".join(c if c.isalnum() or c in ("-", "_", ".") else "_" for c in name)


# --------- NEW: support paper_variant (paper_origin or paper_synth_<model>) ---------

def paper_variant_dir(synth_model: Optional[str]) -> str:
    return "paper_origin" if not synth_model else f"paper_synth_{synth_model}"


def find_regression_files(
    root: Path,
    judge_model: Optional[str],
    tag: Optional[str],
    synth_model: Optional[str],
) -> List[Path]:
    """
    Discover regression files under each paper.

    Layout:
      <paper_dir>/reviews/regression_judge/<judge_model>/<paper_variant>/regression_<tag>.json

    paper_variant:
      - paper_origin (default)
      - paper_synth_<synth_model> (if synth_model provided)
    """
    files: List[Path] = []
    variant = paper_variant_dir(synth_model)

    for paper_dir in sorted([p for p in root.iterdir() if p.is_dir()]):
        base = paper_dir / "reviews" / "regression_judge"
        if not base.exists():
            continue

        if judge_model:
            jm = base / sanitize_filename(judge_model) / variant
            if not jm.exists():
                continue
            if tag:
                f = jm / f"regression_{sanitize_filename(tag)}.json"
                if f.exists():
                    files.append(f)
            else:
                files.extend(sorted(jm.glob("regression_*.json")))
        else:
            # any judge_model
            for jm_dir in base.iterdir():
                po = jm_dir / variant
                if not po.exists():
                    continue
                if tag:
                    f = po / f"regression_{sanitize_filename(tag)}.json"
                    if f.exists():
                        files.append(f)
                else:
                    files.extend(sorted(po.glob("regression_*.json")))

    return files


def pick_differences(differences: List[Dict[str, Any]], mode: str, k: int) -> List[Dict[str, Any]]:
    if not differences:
        return []
    if mode == "top1":
        return [differences[0]]
    if mode == "topk":
        return differences[:max(1, k)]
    if mode == "all":
        return differences
    if mode == "score_shift_only":
        return [x for x in differences if str(x.get("diff_type", "")).strip() == "score_rationale_shift"]
    raise ValueError(f"Unknown mode: {mode}")


def fmt_table(counter: Counter, topn: int = 15, title: str = "") -> str:
    lines = []
    if title:
        lines.append(title)
        lines.append("-" * len(title))
    total = sum(counter.values()) or 1
    for i, (k, v) in enumerate(counter.most_common(topn), 1):
        pct = 100.0 * v / total
        lines.append(f"{i:>2}. {k:<45} {v:>6}  ({pct:5.1f}%)")
    return "\n".join(lines)


def fmt_compare(drop: Counter, nondrop: Counter, topn: int = 20, title: str = "") -> str:
    keys = set(drop) | set(nondrop)
    drop_total = sum(drop.values()) or 1
    nd_total = sum(nondrop.values()) or 1

    rows = []
    for k in keys:
        d = drop.get(k, 0)
        n = nondrop.get(k, 0)
        total = d + n
        if total == 0:
            continue
        drop_rate = d / total
        eps = 1e-9
        drop_share = d / drop_total
        nd_share = n / nd_total
        lift = (drop_share + eps) / (nd_share + eps)
        rows.append((drop_rate, lift, total, k, d, n))

    rows.sort(key=lambda x: (x[0], x[2]), reverse=True)

    lines = []
    if title:
        lines.append(title)
        lines.append("-" * len(title))
    lines.append(f"{'type':<45} {'drop':>6} {'non':>6} {'drop_rate':>9} {'lift':>9} {'total':>6}")
    for drop_rate, lift, total, k, d, n in rows[:topn]:
        lines.append(f"{k:<45} {d:>6} {n:>6} {drop_rate:>9.2f} {lift:>9.2f} {total:>6}")
    return "\n".join(lines)


def bucket_from_delta(delta: Optional[float]) -> str:
    if delta is None:
        return "UNKNOWN"
    return "DROP" if delta < 0 else "NON_DROP"


def main():
    ap = argparse.ArgumentParser()

    ap.add_argument("--input_dir", default="/data/home/zdhs0006/ACL/ICLR_26")
    ap.add_argument("--judge_model", default=None)
    ap.add_argument("--tag", default=None)

    # NEW: paper variant selection (same as eval_review_regression.py)
    ap.add_argument(
        "--synth_model",
        default=None,
        help="If set, read regression outputs from paper_synth_<synth_model> instead of paper_origin. "
             "Example: --synth_model gpt-5-2025-08-07",
    )

    ap.add_argument("--mode", default="top1", choices=["top1", "topk", "all", "score_shift_only"])
    ap.add_argument("--k", type=int, default=3)

    ap.add_argument("--include_unknown_delta", action="store_true",
                    help="Include runs with missing/unparseable delta into UNKNOWN bucket")
    ap.add_argument("--min_abs_drop", type=float, default=0.0,
                    help="For DROP bucket only: require delta <= -min_abs_drop (e.g., 1.0 => only big drops)")

    ap.add_argument("--topn", type=int, default=15)
    ap.add_argument("--out_json", default=None)
    args = ap.parse_args()

    root = Path(args.input_dir).expanduser().resolve()
    if not root.exists():
        raise SystemExit(f"Not found: {root}")

    files = find_regression_files(root, args.judge_model, args.tag, args.synth_model)
    if not files:
        raise SystemExit("No regression_*.json found under input_dir (check --judge_model/--tag/--synth_model).")

    # Counters per bucket
    buckets = ["DROP", "NON_DROP", "UNKNOWN"]
    type_counter = {b: Counter() for b in buckets}
    diff_type_counter = {b: Counter() for b in buckets}
    joint_counter = {b: Counter() for b in buckets}
    runs_counter = Counter()
    skipped_runs = 0

    for fp in files:
        data = load_json(fp)
        runs = data.get("runs", [])
        if not isinstance(runs, list):
            continue

        for run in runs:
            if not isinstance(run, dict):
                continue

            score_delta = safe_get(run, ["regression", "score_delta"], default={})
            delta: Optional[float] = None
            if isinstance(score_delta, dict):
                try:
                    delta = float(score_delta.get("delta"))
                except Exception:
                    delta = None

            b = bucket_from_delta(delta)
            if b == "UNKNOWN" and not args.include_unknown_delta:
                skipped_runs += 1
                continue

            if b == "DROP" and delta is not None and args.min_abs_drop > 0:
                if delta > -args.min_abs_drop:
                    skipped_runs += 1
                    continue

            differences = safe_get(run, ["regression", "differences"], default=[])
            if not isinstance(differences, list) or not differences:
                skipped_runs += 1
                continue

            picked = pick_differences([x for x in differences if isinstance(x, dict)], args.mode, args.k)
            if not picked:
                skipped_runs += 1
                continue

            runs_counter[b] += 1

            for d in picked:
                dt = str(d.get("diff_type", "")).strip() or "UNKNOWN"
                diff_type_counter[b][dt] += 1

                pa_types = d.get("paperaudit_types", [])
                if not isinstance(pa_types, list) or not pa_types:
                    pa_types = ["UNKNOWN"]

                for t in pa_types:
                    ts = str(t).strip() or "UNKNOWN"
                    type_counter[b][ts] += 1
                    joint_counter[b][(dt, ts)] += 1

    variant = paper_variant_dir(args.synth_model)

    print(f"[FOUND] regression_files={len(files)}  variant={variant}  skipped_runs={skipped_runs}")
    print(f"[RUNS]  DROP={runs_counter['DROP']}  NON_DROP={runs_counter['NON_DROP']}  UNKNOWN={runs_counter['UNKNOWN']}")
    print()

    print(fmt_table(type_counter["DROP"], topn=args.topn, title=f"DROP: Top PaperAudit Types (mode={args.mode})"))
    print()
    print(fmt_table(type_counter["NON_DROP"], topn=args.topn, title=f"NON_DROP: Top PaperAudit Types (mode={args.mode})"))
    print()

    print(fmt_compare(
        type_counter["DROP"],
        type_counter["NON_DROP"],
        topn=max(20, args.topn),
        title="Type association with score drops (higher drop_rate / lift => more likely drop-related)",
    ))
    print()

    if args.out_json:
        outp = Path(args.out_json).expanduser().resolve()
        outp.parent.mkdir(parents=True, exist_ok=True)

        def counter_to_list(c: Counter):
            return c.most_common()

        summary = {
            "meta": {
                "input_dir": str(root),
                "judge_model": args.judge_model,
                "tag": args.tag,
                "mode": args.mode,
                "k": args.k,
                "min_abs_drop": args.min_abs_drop,
                "include_unknown_delta": args.include_unknown_delta,
                "synth_model": args.synth_model,
                "paper_variant": variant,
                "regression_files": len(files),
                "runs_used": dict(runs_counter),
                "skipped_runs": skipped_runs,
            },
            "drop": {
                "types": counter_to_list(type_counter["DROP"]),
                "diff_types": counter_to_list(diff_type_counter["DROP"]),
                "joint": [([k[0], k[1]], v) for k, v in joint_counter["DROP"].most_common()],
            },
            "non_drop": {
                "types": counter_to_list(type_counter["NON_DROP"]),
                "diff_types": counter_to_list(diff_type_counter["NON_DROP"]),
                "joint": [([k[0], k[1]], v) for k, v in joint_counter["NON_DROP"].most_common()],
            },
            "unknown": {
                "types": counter_to_list(type_counter["UNKNOWN"]),
                "diff_types": counter_to_list(diff_type_counter["UNKNOWN"]),
            },
        }

        outp.write_text(json.dumps(summary, ensure_ascii=False, indent=2), encoding="utf-8")
        print(f"[WROTE] {outp}")


if __name__ == "__main__":
    main()
