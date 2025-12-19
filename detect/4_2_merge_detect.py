#!/usr/bin/env python3
# -*- coding: utf-8 -*-

from __future__ import annotations
import argparse
import json
import sys
import re
from pathlib import Path
from typing import Any, Dict, List, Tuple, Optional


# =========================
# Defaults (edit as you like)
# =========================
DEFAULT_DETECT_MODEL = "gpt-5-2025-08-07"
DEFAULT_SYNTH_MODEL = "gpt-5-2025-08-07"
DEFAULT_ROOT_DIR = "data/ICML_30"
DEFAULT_MODES = ["deep", "standard"]


# =========================
# IO
# =========================
def load_json(p: Path) -> Dict[str, Any]:
    with p.open("r", encoding="utf-8") as f:
        return json.load(f)

def save_json(obj: Any, p: Path) -> None:
    p.parent.mkdir(parents=True, exist_ok=True)
    with p.open("w", encoding="utf-8") as f:
        json.dump(obj, f, ensure_ascii=False, indent=2)

def eprint(*args, **kwargs):
    print(*args, file=sys.stderr, **kwargs)


# =========================
# Core
# =========================
def mode_dirname(mode: str) -> str:
    """deep -> deep_detect ; standard -> standard_detect"""
    mode = (mode or "").strip()
    if not mode:
        raise ValueError("Empty mode")
    if mode.endswith("_detect"):
        return mode
    return f"{mode}_detect"

def detect_filename(mode: str) -> str:
    """deep -> deep_detect.json"""
    mode = (mode or "").strip()
    if not mode:
        raise ValueError("Empty mode")
    if mode.endswith("_detect"):
        mode = mode[:-7]
    return f"{mode}_detect.json"

def combined_mode_name(modes: List[str]) -> str:
    cleaned = []
    for m in modes:
        m = (m or "").strip()
        if not m:
            continue
        if m.endswith("_detect"):
            m = m[:-7]
        cleaned.append(m)
    if not cleaned:
        raise ValueError("No valid modes")
    return "_".join(cleaned)

def renumber_findings(findings: List[Dict[str, Any]]) -> List[Dict[str, Any]]:
    out = []
    for i, f in enumerate(findings, start=1):
        if not isinstance(f, dict):
            continue
        g = dict(f)
        g["id"] = i
        out.append(g)
    return out

def normalize_finding_for_dedup(f: Dict[str, Any]) -> str:
    def norm(s: Any) -> str:
        s = "" if s is None else str(s)
        s = s.lower().strip()
        s = re.sub(r"\s+", " ", s)
        return s

    return "||".join([
        norm(f.get("type")),
        norm(f.get("section_location")),
        norm(f.get("error_location")),
        norm(f.get("explanation")),
        norm(f.get("proposed_fix")),
    ])

def compute_paths_for_paper_root(
    paper_root: Path,
    detect_model: str,
    synth_model: str,
    modes: List[str],
    output_base_mode: Optional[str] = None,
) -> Tuple[List[Tuple[str, Path]], Path, Tuple[str, Path]]:
    """
    Returns:
      - srcs: [(mode, path), ...] for all modes
      - out_path: <paper_root>/<combo>_detect/<detect_model>/paper_synth_<synth_model>/<combo>_detect.json
      - base_src: (base_mode, base_path) skeleton source (usually deep)
    """
    if not modes:
        raise ValueError("modes is empty")

    # ---- base skeleton mode (default: modes[0]) ----
    base_mode = (output_base_mode or modes[0]).strip()
    base_mode_clean = base_mode[:-7] if base_mode.endswith("_detect") else base_mode

    srcs: List[Tuple[str, Path]] = []
    base_src: Optional[Tuple[str, Path]] = None

    for m in modes:
        m_clean = m[:-7] if m.endswith("_detect") else m
        in_path = (
            paper_root
            / mode_dirname(m_clean)
            / detect_model
            / f"paper_synth_{synth_model}"
            / detect_filename(m_clean)
        )
        srcs.append((m_clean, in_path))
        if m_clean == base_mode_clean:
            base_src = (m_clean, in_path)

    if base_src is None:
        m0 = modes[0]
        m0_clean = m0[:-7] if m0.endswith("_detect") else m0
        base_src = (m0_clean, (
            paper_root
            / mode_dirname(m0_clean)
            / detect_model
            / f"paper_synth_{synth_model}"
            / detect_filename(m0_clean)
        ))

    # ---- OUTPUT goes into combined-mode folder ----
    combo = combined_mode_name(modes)          # e.g. "deep_standard"
    out_path = (
        paper_root
        / mode_dirname(combo)                  # -> "deep_standard_detect"
        / detect_model
        / f"paper_synth_{synth_model}"
        / f"{combo}_detect.json"               # -> "deep_standard_detect.json"
    )

    return srcs, out_path, base_src


def find_paper_roots(root_dir: Path, detect_model: str, synth_model: str, anchor_mode: str) -> List[Path]:
    anchor_mode_clean = anchor_mode[:-7] if anchor_mode.endswith("_detect") else anchor_mode
    pattern = Path(mode_dirname(anchor_mode_clean)) / detect_model / f"paper_synth_{synth_model}" / detect_filename(anchor_mode_clean)
    hits = list(root_dir.rglob(str(pattern)))

    paper_roots = []
    for p in hits:
        try:
            paper_root = p.parents[3]
            paper_roots.append(paper_root)
        except Exception:
            continue

    # deterministic
    return sorted(set(paper_roots))


def merge_findings_from_sources(
    sources: List[Tuple[str, Path]],
    dedup: bool = False,
) -> Tuple[List[Dict[str, Any]], Dict[str, Any], List[Dict[str, str]]]:
    """
    Returns:
      - merged_findings (renumbered)
      - merge_meta (sources stats etc.)
      - missing list
    """
    merged: List[Dict[str, Any]] = []
    meta: Dict[str, Any] = {"sources": []}
    missing: List[Dict[str, str]] = []

    seen = set()
    for mode, p in sources:
        if not p.exists():
            missing.append({"mode": mode, "path": str(p)})
            meta["sources"].append({
                "mode": mode,
                "path": str(p),
                "exists": False,
                "findings_in_file": 0,
                "findings_kept": 0,
            })
            continue

        obj = load_json(p)
        raw = obj.get("findings", [])
        if not isinstance(raw, list):
            raw = []

        kept = 0
        for f in raw:
            if not isinstance(f, dict):
                continue
            if dedup:
                key = normalize_finding_for_dedup(f)
                if key in seen:
                    continue
                seen.add(key)
            merged.append(f)
            kept += 1

        meta["sources"].append({
            "mode": mode,
            "path": str(p),
            "exists": True,
            "findings_in_file": len(raw),
            "findings_kept": kept,
        })

    merged = renumber_findings(merged)
    meta["merged_total"] = len(merged)
    meta["dedup"] = bool(dedup)
    meta["missing_total"] = len(missing)
    return merged, meta, missing


def main():
    ap = argparse.ArgumentParser(
        description="Merge findings across multiple detect modes; use deep_detect.json as skeleton and only replace findings + add merge_meta."
    )
    ap.add_argument("--root_dir", type=str, default=DEFAULT_ROOT_DIR)
    ap.add_argument("--detect_model", type=str, default=DEFAULT_DETECT_MODEL)
    ap.add_argument("--synth_model", type=str, default=DEFAULT_SYNTH_MODEL)
    ap.add_argument("--modes", nargs="+", default=DEFAULT_MODES)
    ap.add_argument("--output_base_mode", type=str, default=None,
                    help="Which mode directory to write into (default: first mode). Usually 'deep'.")
    ap.add_argument("--dedup", action="store_true")
    ap.add_argument("--overwrite", action="store_true")
    ap.add_argument("--strict", action="store_true",
                    help="If set, any missing input file => fail that paper (no output).")
    ap.add_argument("--max_papers", type=int, default=None)
    args = ap.parse_args()

    root_dir = Path(args.root_dir).expanduser().resolve()
    if not root_dir.exists():
        raise SystemExit(f"[FATAL] root_dir not found: {root_dir}")

    modes = [m.strip() for m in (args.modes or []) if m and m.strip()]
    if len(modes) < 2:
        raise SystemExit("[FATAL] Need at least 2 modes to merge (e.g., deep standard).")

    anchor_mode = modes[0]
    paper_roots = find_paper_roots(root_dir, args.detect_model, args.synth_model, anchor_mode=anchor_mode)
    if not paper_roots:
        raise SystemExit(
            f"[FATAL] No papers found under {root_dir} for anchor_mode={anchor_mode}, "
            f"detect_model={args.detect_model}, synth_model={args.synth_model}"
        )

    if args.max_papers is not None:
        paper_roots = paper_roots[: args.max_papers]

    summary = {"total": 0, "ok": 0, "skipped": 0, "failed": 0, "details": []}

    for paper_root in paper_roots:
        summary["total"] += 1
        try:
            srcs, out_path, base_src = compute_paths_for_paper_root(
                paper_root=paper_root,
                detect_model=args.detect_model,
                synth_model=args.synth_model,
                modes=modes,
                output_base_mode=args.output_base_mode,
            )
            base_mode, base_path = base_src

            if out_path.exists() and not args.overwrite:
                summary["skipped"] += 1
                summary["details"].append({
                    "paper": str(paper_root),
                    "skipped": True,
                    "reason": "output_exists",
                    "out": str(out_path)
                })
                continue

            # base file must exist (skeleton)
            if not base_path.exists():
                summary["failed"] += 1
                summary["details"].append({
                    "paper": str(paper_root),
                    "ok": False,
                    "error": "base_skeleton_missing",
                    "base_mode": base_mode,
                    "base_path": str(base_path),
                })
                continue

            merged_findings, merge_meta, missing = merge_findings_from_sources(srcs, dedup=args.dedup)

            if missing and args.strict:
                summary["failed"] += 1
                summary["details"].append({
                    "paper": str(paper_root),
                    "ok": False,
                    "error": "missing_inputs",
                    "missing": missing,
                })
                continue

            # ---- skeleton preserve ----
            base_obj = load_json(base_path)

            # replace findings only
            base_obj["findings"] = merged_findings

            # add meta
            base_obj["merge_meta"] = {
                **merge_meta,
                "modes_requested": [m[:-7] if m.endswith("_detect") else m for m in modes],
                "base_skeleton": {"mode": base_mode, "path": str(base_path)},
            }
            base_obj["merged_from_modes"] = [m for m, p in srcs if p.exists()]
            base_obj["merged_created_at"] = __import__("datetime").datetime.utcnow().isoformat() + "Z"

            save_json(base_obj, out_path)

            summary["ok"] += 1
            summary["details"].append({
                "paper": str(paper_root),
                "ok": True,
                "out": str(out_path),
                "base": str(base_path),
                "merged": len(merged_findings),
                "missing": missing,
            })
        except Exception as e:
            summary["failed"] += 1
            summary["details"].append({"paper": str(paper_root), "ok": False, "error": f"{type(e).__name__}: {e}"})

    print(json.dumps(summary, ensure_ascii=False, indent=2))


if __name__ == "__main__":
    main()
