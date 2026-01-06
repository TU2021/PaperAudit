#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
DeepReviewer Agent Batch Review Runner

Batch review runner for DeepReviewerAgent that performs deep, comprehensive paper review
with multiple reviewer perspectives. The agent generates detailed reviews with structured
scores across multiple dimensions.

The runner uses async scheduler (asyncio + semaphore) with threadpool executor for
parallel processing. It supports resume (skip if output exists), overwrite, and
.inprogress file tracking.

Usage Examples:
    # Basic batch review
    python run_deepreview_agent.py \
        --input_dir /path/to/papers \
        --model gpt-5-2025-08-07 \
        --review_agent deepreviewer \
        --model_tag all \
        --jobs 10

    # Review synthetic papers
    python run_deepreview_agent.py \
        --input_dir /path/to/papers \
        --model gpt-5-2025-08-07 \
        --synth_model gpt-5-2025-08-07 \
        --review_agent deepreviewer \
        --model_tag all \
        --jobs 10


Key Arguments:
    --input_dir: Root folder containing paper subfolders. Default: /mnt/parallel_ssd/home/zdhs0006/ACL/data_test
    --model: LLM model name for review. Default: gpt-5-2025-08-07
    --model_tag: Tag used in output filenames. Default: test
    --review_agent: Folder name under reviews/. Default: deepreviewer
    --query: Instruction/query passed to the reviewer. Default: "Please provide a critical, structured peer review of the paper."
    --synth_model: Synthetic data model suffix. Default: "" (origin)
    --enable_mm: Enable multimodal markers. Default: True
    --jobs: Number of in-flight papers (parallel processing). Default: 1
    --overwrite: Force re-run and overwrite existing outputs. Default: False

Output:
    - Review text: <paper_dir>/reviews/{review_agent}/{model}/{paper_origin|paper_synth_{synth_model}}/deep_review_{model_tag}.txt
    - Review scores: <paper_dir>/reviews/{review_agent}/{model}/{paper_origin|paper_synth_{synth_model}}/review_output_{model_tag}.json
    - The JSON contains:
      - scores: Parsed scores with per-reviewer and averaged scores
        - per_reviewer: List of individual reviewer scores
        - avg: Averaged scores across reviewers
        - max: Maximum possible scores
        - counts: Number of reviewers for each field
        - num_reviewers: Total number of reviewers
      - Score fields: overall, novelty, technical_quality, clarity, confidence
"""

from __future__ import annotations

import argparse
import asyncio
import json
import re
import sys
import time
import threading
from pathlib import Path
from typing import Any, Dict, Optional, List, Tuple

try:
    from tqdm import tqdm
except Exception:
    tqdm = None

from dotenv import load_dotenv
from agents import DeepReviewerAgent  # <-- changed

load_dotenv()

_PRINT_LOCK = threading.Lock()

# ---------------- CLI ----------------
def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Batch deep review runner (async scheduled)")
    parser.add_argument(
        "--input_dir",
        default="/mnt/parallel_ssd/home/zdhs0006/ACL/data_test",
        help="Root folder containing paper subfolders",
    )
    parser.add_argument("--model", default="gpt-5-2025-08-07", help="LLM model name")

    parser.add_argument(
        "--model_tag",
        default="test",
        help="Tag used in output filenames, e.g., exp01 / runA",
    )

    parser.add_argument(
        "--review_agent",
        default="deepreviewer",
        help="Folder name under reviews/, e.g., deepreviewer",
    )

    parser.add_argument(
        "--query",
        default="Please provide a critical, structured peer review of the paper.",
        help="Instruction/query passed to the reviewer",
    )

    parser.add_argument("--synth_model", default="", help="Synthetic data model suffix")
    parser.add_argument("--enable_mm", default=True, action="store_true", help="Enable multimodal markers")

    parser.add_argument(
        "--jobs", "-j",
        type=int,
        default=1,
        help="jobs (number of in-flight papers). Uses threadpool executor under the hood.",
    )

    parser.add_argument(
        "--overwrite",
        action="store_true",
        help="Force re-run and overwrite existing outputs.",
    )

    return parser.parse_args()


# ---------------- utils ----------------
def sanitize_filename(name: str) -> str:
    return "".join(c if c.isalnum() or c in ("-", "_") else "_" for c in name)


def resolve_paper_file(paper_dir: Path, synth_model: Optional[str]) -> Optional[Path]:
    final_file = paper_dir / "paper_final.json"
    synth_model_norm = (synth_model or "").strip()
    if not synth_model_norm:
        return final_file if final_file.exists() else None

    synth_file = paper_dir / f"paper_synth_{synth_model_norm}.json"
    if synth_file.exists():
        return synth_file
    if final_file.exists():
        return final_file
    return None


def load_paper_json(path: Path) -> Dict[str, Any]:
    """
    Normalize schema so returned dict ALWAYS has top-level `content`.
    Supported:
      - paper_final.json: {"content": ...}
      - paper_synth_*.json: {"paper": {"content": ...}, ...}
    """
    with path.open("r", encoding="utf-8") as f:
        obj = json.load(f)

    if not isinstance(obj, dict):
        raise ValueError(f"Paper JSON must be a dict, got {type(obj)} from {path}")

    if "content" in obj:
        return obj

    paper = obj.get("paper")
    if isinstance(paper, dict) and "content" in paper:
        obj["content"] = paper["content"]
        return obj

    raise KeyError(f"Cannot find paper content in {path}. Expected `content` or `paper.content`.")


def load_json(path: Path) -> Dict[str, Any]:
    with path.open("r", encoding="utf-8") as f:
        return json.load(f)


def save_json(path: Path, obj: Dict[str, Any]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("w", encoding="utf-8") as f:
        json.dump(obj, f, ensure_ascii=False, indent=2)


def ensure_thread_event_loop() -> asyncio.AbstractEventLoop:
    try:
        return asyncio.get_event_loop()
    except RuntimeError:
        loop = asyncio.new_event_loop()
        asyncio.set_event_loop(loop)
        return loop


def cleanup_thread_event_loop(loop: asyncio.AbstractEventLoop) -> None:
    if loop.is_closed():
        return
    try:
        loop.run_until_complete(asyncio.sleep(0))
    except Exception:
        pass


# ---------------- score parsing ----------------

_SCORE_SPECS = [
    ("overall", "Overall", 10),
    ("novelty", "Novelty", 10),
    ("technical_quality", "Technical Quality", 10),
    ("clarity", "Clarity", 10),
    ("confidence", "Confidence", 5),
]

def extract_scores_multi(review_text: str) -> Dict[str, Any]:
    """
    Extract multiple reviewer score sets, then compute per-field averages.

    Assumes each reviewer outputs EXACTLY these markdown list lines:
      - Overall (10): <int> — ...
      - Novelty (10): <int> — ...
      - Technical Quality (10): <int> — ...
      - Clarity (10): <int> — ...
      - Confidence (5): <int> — ...

    Returns:
      {
        "per_reviewer": [
          {"overall":7,"novelty":6,"technical_quality":7,"clarity":6,"confidence":4},
          ...
        ],
        "avg": {"overall":6.5, ...},
        "max": {"overall":10, ...},
        "counts": {"overall":4, ...},
        "num_reviewers": 4
      }
    """
    text = review_text or ""

    # Light normalization: fix cases like "### Score- Overall..." into newline list items
    norm = text
    norm = re.sub(r"(###\s*Score)\s*-\s*", r"\1\n- ", norm, flags=re.IGNORECASE)
    norm = re.sub(r"(#+\s*Score)\s*:\s*", r"\1:\n", norm, flags=re.IGNORECASE)

    # Build per-field regex that finds ALL occurrences
    # Capture the integer after ":"
    field_patterns: Dict[str, re.Pattern] = {}
    for key, label, maxv in _SCORE_SPECS:
        pat = re.compile(
            rf"(?im)^\s*-\s*{re.escape(label)}\s*"
            rf"\(\s*{maxv}\s*\)\s*[:：]\s*(\d+)\b"
        )
        field_patterns[key] = pat

    # Find all occurrences for each field
    all_vals: Dict[str, List[int]] = {k: [] for k, _, _ in _SCORE_SPECS}
    for key, _, _ in _SCORE_SPECS:
        vals = []
        for m in field_patterns[key].finditer(norm):
            try:
                vals.append(int(m.group(1)))
            except Exception:
                continue
        all_vals[key] = vals

    # Decide how many reviewer sets exist:
    # Use the minimum length across the 5 fields to keep sets aligned.
    lengths = [len(all_vals[k]) for k, _, _ in _SCORE_SPECS]
    n = min(lengths) if lengths else 0

    per_reviewer: List[Dict[str, int]] = []
    for i in range(n):
        d: Dict[str, int] = {}
        for key, _, maxv in _SCORE_SPECS:
            v = all_vals[key][i]
            # range clamp safety
            if key == "confidence":
                v = max(0, min(5, v))
            else:
                v = max(0, min(10, v))
            d[key] = v
        per_reviewer.append(d)

    # Compute averages
    avg: Dict[str, Optional[float]] = {}
    counts: Dict[str, int] = {}
    max_map: Dict[str, int] = {k: maxv for k, _, maxv in _SCORE_SPECS}

    for key, _, _ in _SCORE_SPECS:
        vals = [r[key] for r in per_reviewer if key in r]
        counts[key] = len(vals)
        avg[key] = (sum(vals) / len(vals)) if vals else None

    return {
        "per_reviewer": per_reviewer,
        "avg": avg,
        "max": max_map,
        "counts": counts,
        "num_reviewers": len(per_reviewer),
    }


# ---------------- output paths & resume helpers ----------------
def review_dir_for(paper_dir: Path, model: str, review_agent: str, synth_model: Optional[str]) -> Path:
    model_dir = sanitize_filename(model)
    agent_dir = sanitize_filename(review_agent)
    synth_model_norm = (synth_model or "").strip()
    subdir = "paper_origin" if not synth_model_norm else f"paper_synth_{sanitize_filename(synth_model_norm)}"
    return paper_dir / "reviews" / agent_dir / model_dir / subdir


def out_json_path(paper_dir: Path, model: str, review_agent: str, synth_model: str, model_tag: str) -> Path:
    tag = sanitize_filename(model_tag)
    return review_dir_for(paper_dir, model, review_agent, synth_model) / f"review_output_{tag}.json"


def out_txt_path(paper_dir: Path, model: str, review_agent: str, synth_model: str, model_tag: str) -> Path:
    tag = sanitize_filename(model_tag)
    return review_dir_for(paper_dir, model, review_agent, synth_model) / f"deep_review_{tag}.txt"


def inprogress_path(paper_dir: Path, model: str, review_agent: str, synth_model: str, model_tag: str) -> Path:
    tag = sanitize_filename(model_tag)
    return review_dir_for(paper_dir, model, review_agent, synth_model) / f"review_output_{tag}.json.inprogress"


def already_done(paper_dir: Path, model: str, review_agent: str, synth_model: str, model_tag: str) -> bool:
    """
    "done" means final review output json exists and has key "reviews".
    """
    p = out_json_path(paper_dir, model, review_agent, synth_model, model_tag)
    if not p.exists():
        return False
    try:
        obj = load_json(p)
        return isinstance(obj, dict) and ("scores" in obj)
    except Exception:
        return False


def find_paper_dirs(root_dir: Path) -> List[Path]:
    return [p for p in sorted(root_dir.iterdir()) if p.is_dir()]


# ---------------- single paper worker (SYNC, run in executor) ----------------
def process_one_paper_sync(paper_dir: Path, args: argparse.Namespace) -> Tuple[Path, bool, str]:
    paper_file = resolve_paper_file(paper_dir, args.synth_model)
    if paper_file is None:
        return paper_dir, True, "skip_no_input"

    rdir = review_dir_for(paper_dir, args.model, args.review_agent, args.synth_model)
    outp = out_json_path(paper_dir, args.model, args.review_agent, args.synth_model, args.model_tag)
    out_txt = out_txt_path(paper_dir, args.model, args.review_agent, args.synth_model, args.model_tag)
    inprog = inprogress_path(paper_dir, args.model, args.review_agent, args.synth_model, args.model_tag)

    if (not args.overwrite) and already_done(paper_dir, args.model, args.review_agent, args.synth_model, args.model_tag):
        return paper_dir, True, "skip_done"

    try:
        rdir.mkdir(parents=True, exist_ok=True)
        inprog.write_text(str(time.time()), encoding="utf-8")
    except Exception:
        pass

    loop = ensure_thread_event_loop()

    try:
        paper_json = load_paper_json(paper_file)

        with _PRINT_LOCK:
            print(f"[RUN] {paper_dir.name} :: {paper_file.name}")

        agent = DeepReviewerAgent(
            model=args.model,
        )

        # ---- run deep review (async) inside this thread loop ----
        # Prefer agent.review_paper() wrapper if you added it; else parse run() SSE here.
        if hasattr(agent, "review_paper"):
            review_text = loop.run_until_complete(
                agent.review_paper(paper_json=paper_json, query=args.query, enable_mm=args.enable_mm)
            )
        else:
            async def _collect():
                chunks = []
                async for sse in agent.run(paper_json=paper_json, query=args.query, enable_mm=args.enable_mm):
                    if not sse.startswith("data:"):
                        continue
                    payload = sse[len("data:"):].strip()
                    if payload == "[DONE]":
                        break
                    try:
                        obj = json.loads(payload)
                        delta = obj.get("choices", [{}])[0].get("delta", {})
                        content = delta.get("content", "")
                        if content:
                            chunks.append(content)
                    except Exception:
                        pass
                return "".join(chunks).strip()

            review_text = loop.run_until_complete(_collect())

        # save txt
        try:
            out_txt.write_text(review_text or "", encoding="utf-8")
        except Exception:
            pass

        # parse scores
        scores_multi = extract_scores_multi(review_text)

        payload: Dict[str, Any] = {
            "scores": scores_multi
        }
        save_json(outp, payload)
        return paper_dir, True, "ok"

    except Exception as e:
        return paper_dir, False, f"error: {repr(e)}"

    finally:
        try:
            if inprog.exists():
                inprog.unlink()
        except Exception:
            pass
        try:
            cleanup_thread_event_loop(loop)
        except Exception:
            pass


# ---------------- async scheduler ----------------
async def bounded_worker(
    sem: asyncio.Semaphore,
    loop,
    paper_dir: Path,
    args: argparse.Namespace,
):
    async with sem:
        return await loop.run_in_executor(None, process_one_paper_sync, paper_dir, args)


async def run_batch(root_dir: Path, jobs: int, args: argparse.Namespace) -> Dict[str, Any]:
    paper_dirs = find_paper_dirs(root_dir)
    if not paper_dirs:
        return {"total": 0, "ok": 0, "fail": [], "skipped_done": 0}

    planned: List[Path] = []
    skipped_done = 0

    for d in paper_dirs:
        paper_file = resolve_paper_file(d, args.synth_model)
        if paper_file is None:
            continue

        if args.overwrite:
            planned.append(d)
            continue

        if already_done(d, args.model, args.review_agent, args.synth_model, args.model_tag):
            skipped_done += 1
            continue

        planned.append(d)

    if not planned:
        return {"total": 0, "ok": skipped_done, "fail": [], "skipped_done": skipped_done}

    sem = asyncio.Semaphore(jobs)
    loop = asyncio.get_running_loop()

    tasks = [bounded_worker(sem, loop, d, args) for d in planned]
    results: List[Tuple[Path, bool, str]] = []

    iterator = asyncio.as_completed(tasks)
    if tqdm is not None:
        iterator = tqdm(iterator, total=len(tasks), desc="Reviewing")

    for coro in iterator:
        d, ok, msg = await coro
        results.append((d, ok, msg))

        with _PRINT_LOCK:
            if msg == "ok":
                outp = out_json_path(d, args.model, args.review_agent, args.synth_model, args.model_tag)
                print(f"[DONE] {d.name} -> {outp.relative_to(d)}")
            elif msg == "skip_done":
                print(f"[SKIP] {d.name} (already done)")
            elif msg == "skip_no_input":
                print(f"[SKIP] {d.name} (no input json)")
            else:
                print(f"[FAIL] {d.name} :: {msg}", file=sys.stderr)

    ok_cnt = sum(1 for (_, ok, msg) in results if ok and msg == "ok")
    fail = [(str(d), msg) for (d, ok, msg) in results if not ok]

    return {
        "total": len(results),
        "ok": ok_cnt + skipped_done,
        "fail": fail,
        "skipped_done": skipped_done,
    }


def main() -> None:
    args = parse_args()
    root_dir = Path(args.input_dir).expanduser().resolve()
    if not root_dir.exists() or not root_dir.is_dir():
        raise SystemExit(f"Input directory not found: {root_dir}")

    print(
        f"[INFO] root={root_dir} jobs={args.jobs} model={args.model} "
        f"synth_model={args.synth_model or 'origin'} overwrite={args.overwrite} "
        f"model_tag={args.model_tag} review_agent={args.review_agent}"
    )

    summary = asyncio.run(run_batch(root_dir=root_dir, jobs=args.jobs, args=args))

    print(
        f"[DONE] total={summary['total']} ok={summary['ok']} "
        f"fail={len(summary['fail'])} skipped_done={summary.get('skipped_done', 0)}"
    )
    if summary["fail"]:
        print("[FAILED ITEMS]")
        for i, (path, msg) in enumerate(summary["fail"], 1):
            print(f"  {i}. {path} :: {msg}")


if __name__ == "__main__":
    main()
