#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Audit Agent Batch Review Runner

End-to-end batch review runner for AuditAgent that performs comprehensive paper review
with multiple stages including baseline review, cheating detection, motivation evaluation,
and final assessment.

The runner uses async scheduler (asyncio + semaphore) with threadpool executor for
parallel processing. It supports resume (skip if output exists), overwrite, and
.inprogress file tracking.

Pipeline Stages:
1. Baseline Review: Initial review of the paper
2. Paper Memory: Create natural-language memory of the paper
3. Cheating Detection: Detect potential data leakage or test set contamination
4. Motivation Evaluation: Assess the motivation and contribution clarity
5. Final Assessment: Refined review incorporating all previous stages

Usage Examples:
    # Basic batch review
    python run_audit_agent.py \
        --input_dir /path/to/papers \
        --model gpt-5-2025-08-07 \
        --review_agent paper_audit \
        --model_tag all \
        --jobs 10

    # Review synthetic papers
    python run_audit_agent.py \
        --input_dir /path/to/papers \
        --model gpt-5-2025-08-07 \
        --synth_model gpt-5-2025-08-07 \
        --review_agent paper_audit \
        --model_tag all \
        --jobs 10

Key Arguments:
    --input_dir: Root folder containing paper subfolders. Default: /mnt/parallel_ssd/home/zdhs0006/ACL/data_test
    --model: LLM model name for review. Default: gpt-5-2025-08-07
    --model_tag: User-specified tag used in output filenames. Default: test
    --review_agent: Reviewer agent name (used as output subdir). Default: paper_audit
    --query: Instruction/query passed to the reviewer. Default: "Please provide a critical, structured peer review of the paper."
    --synth_model: Synthetic data model suffix (empty => origin). Default: "" (origin)
    --disable_mm: Disable multimodal text markers. Default: False (multimodal enabled)
    --reuse_cache: Reuse cached artifacts if present. Default: False
    --no_cheating_detection: Disable cheating detection stage. Default: False (enabled)
    --no_motivation: Disable motivation evaluation stage. Default: False (enabled)
    --jobs: Number of in-flight papers (parallel processing). Default: 1
    --overwrite: Force re-run and overwrite existing outputs. Default: False

Output:
    - Review results: <paper_dir>/reviews/{review_agent}/{model}/{paper_origin|paper_synth_{synth_model}}/review_output_{model_tag}.json
    - The JSON contains:
      - baseline_review: Initial review text
      - final_review: Refined review text
      - scores: Parsed scores for baseline and refined reviews
        - overall, novelty, technical_quality, clarity, confidence
    - Cache artifacts (if reuse_cache enabled):
      - paper_memory.txt
      - cheat_report.txt
      - motivation_report.txt
      - baseline_review.txt
      - final_review_*.txt
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
from agents import AuditAgent

load_dotenv()

_PRINT_LOCK = threading.Lock()


# ---------------- CLI ----------------
def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Batch review runner (async scheduled)")
    parser.add_argument(
        "--input_dir",
        default="/mnt/parallel_ssd/home/zdhs0006/ACL/data_test",
        help="Root folder containing paper subfolders",
    )
    parser.add_argument("--model", default="gpt-5-2025-08-07", help="LLM model name")

    parser.add_argument(
        "--model_tag",
        default="test",
        help="User-specified tag used in output filenames, e.g., exp01 / runA",
    )

    parser.add_argument(
        "--review_agent",
        default="paper_audit",
        help="Reviewer agent name (used as output subdir), e.g., paper_audit / deepreviewer",
    )

    parser.add_argument(
        "--query",
        default="Please provide a critical, structured peer review of the paper.",
        help="Instruction/query passed to the reviewer",
    )

    parser.add_argument("--synth_model", default="", help="Synthetic data model suffix (empty => origin)")

    parser.add_argument(
        "--disable_mm",
        action="store_true",
        help="Disable multimodal text markers (enable_mm=False). Default is enabled.",
    )

    parser.add_argument(
        "--reuse_cache",
        action="store_true",
        help="Reuse cached artifacts (paper_memory / cheat_report / motivation_report / baseline) if present.",
    )

    parser.add_argument(
        "--no_cheating_detection",
        default=False,
        dest="enable_cheating_detection",
        action="store_false",
        help="Disable cheating detection stage",
    )
    parser.add_argument(
        "--no_motivation",
        default=False,
        dest="enable_motivation",
        action="store_false",
        help="Disable motivation evaluation stage",
    )

    parser.add_argument(
        "--jobs", "-j",
        type=int,
        default=10,
        help="jobs (number of in-flight papers). Uses threadpool executor under the hood.",
    )

    parser.add_argument(
        "--overwrite",
        action="store_true",
        help="Force re-run and overwrite existing outputs.",
    )

    parser.set_defaults(enable_cheating_detection=True, enable_motivation=True)
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
    Load paper json and normalize schema so that the returned dict ALWAYS has top-level `content`.

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
    """
    Ensure the current executor thread has a current event loop (required by Py3.12+ for some libs).

    IMPORTANT:
    - ThreadPoolExecutor threads are reused.
    - Do NOT close the loop per paper; keep it alive for the lifetime of the thread.
    """
    try:
        return asyncio.get_event_loop()
    except RuntimeError:
        loop = asyncio.new_event_loop()
        asyncio.set_event_loop(loop)
        return loop


def cleanup_thread_event_loop(loop: asyncio.AbstractEventLoop) -> None:
    """
    Best-effort cleanup WITHOUT closing the loop.

    Rationale:
    - Some libs (httpx/anyio) may schedule async cleanup later (e.g., via GC finalizers).
    - Closing the loop per paper can cause 'Event loop is closed' during AsyncClient.aclose().
    """
    if loop.is_closed():
        return
    try:
        loop.run_until_complete(asyncio.sleep(0))
    except Exception:
        pass
    # Do NOT cancel tasks / shutdown_asyncgens / close loop here.


# ---------------- score parsing ----------------
_SCORE_SPECS = [
    ("overall", "Overall", 10),
    ("novelty", "Novelty", 10),
    ("technical_quality", "Technical Quality", 10),
    ("clarity", "Clarity", 10),
    ("confidence", "Confidence", 5),
]


def extract_scores(review_text: str) -> Dict[str, Any]:
    """
    Extract numeric scores from review text.

    Robust to cases like:
      "...protocol if one becomes available.### Score- Overall (10): 9 — ..."
    i.e., score items may NOT start at a new line.
    """
    text = review_text or ""
    out: Dict[str, Any] = {"parsed": {}, "max": {}, "missing": []}

    # ---- light normalization to improve matchability ----
    norm = text
    norm = re.sub(r"(###\s*Score)\s*-\s*", r"\1\n- ", norm, flags=re.IGNORECASE)
    norm = re.sub(r"(#+\s*Score)\s*:\s*", r"\1:\n", norm, flags=re.IGNORECASE)

    for key, label, maxv in _SCORE_SPECS:
        out["max"][key] = maxv

        # Pass 1: strict line-anchored
        pat_strict = (
            rf"(?im)^\s*(?:[-*]\s*)?{re.escape(label)}\s*"
            rf"\(\s*{maxv}\s*\)\s*[:：]\s*(\d+)\b"
        )
        m = re.search(pat_strict, norm)

        # Pass 2: relaxed global match (handles inline / missing newline)
        if not m:
            pat_relaxed = (
                rf"(?is)\b{re.escape(label)}\s*"
                rf"\(\s*{maxv}\s*\)\s*[:：]\s*(\d+)\b"
            )
            m = re.search(pat_relaxed, norm)

        if not m:
            out["missing"].append(key)
            continue

        try:
            val = int(m.group(1))
        except Exception:
            out["missing"].append(key)
            continue

        out["parsed"][key] = val

    return out


# ---------------- output paths & resume helpers ----------------
def review_dir_for(paper_dir: Path, model: str, review_agent: str, synth_model: Optional[str]) -> Path:
    model_dir = sanitize_filename(model)
    review_agent_dir = sanitize_filename(review_agent)
    synth_model_norm = (synth_model or "").strip()

    if not synth_model_norm:
        subdir = "paper_origin"
    else:
        subdir = f"paper_synth_{sanitize_filename(synth_model_norm)}"

    return paper_dir / "reviews" / review_agent_dir / model_dir / subdir


def out_json_path(paper_dir: Path, model: str, review_agent: str, synth_model: str, model_tag: str) -> Path:
    tag = sanitize_filename(model_tag)
    return review_dir_for(paper_dir, model, review_agent, synth_model) / f"review_output_{tag}.json"


def inprogress_path(paper_dir: Path, model: str, review_agent: str, synth_model: str, model_tag: str) -> Path:
    tag = sanitize_filename(model_tag)
    return review_dir_for(paper_dir, model, review_agent, synth_model) / f"review_output_{tag}.json.inprogress"


def already_done(paper_dir: Path, model: str, review_agent: str, synth_model: str, model_tag: str) -> bool:
    """
    "done" means the final review output json exists and has key "reviews".
    IMPORTANT: this is NOT paper json, so must NOT use load_paper_json().
    """
    p = out_json_path(paper_dir, model, review_agent, synth_model, model_tag)
    if not p.exists():
        return False
    try:
        obj = load_json(p)
        return isinstance(obj, dict) and ("reviews" in obj)
    except Exception:
        return False


def find_paper_dirs(root_dir: Path) -> List[Path]:
    return [p for p in sorted(root_dir.iterdir()) if p.is_dir()]


def cache_status(rdir: Path) -> Dict[str, bool]:
    return {
        "baseline": (rdir / "baseline_review.txt").exists(),
        "final": (rdir / "final_review.txt").exists() or any(rdir.glob("final_review_*.txt")),
        "paper_memory": (rdir / "paper_memory.txt").exists(),
        "cheat": (rdir / "cheat_report.txt").exists(),
        "motivation": (rdir / "motivation_report.txt").exists(),
    }


# ---------------- single paper worker (SYNC, run in executor) ----------------
def process_one_paper_sync(paper_dir: Path, args: argparse.Namespace) -> Tuple[Path, bool, str]:
    paper_file = resolve_paper_file(paper_dir, args.synth_model)
    if paper_file is None:
        return paper_dir, True, "skip_no_input"

    rdir = review_dir_for(paper_dir, args.model, args.review_agent, args.synth_model)
    outp = out_json_path(paper_dir, args.model, args.review_agent, args.synth_model, args.model_tag)
    inprog = inprogress_path(paper_dir, args.model, args.review_agent, args.synth_model, args.model_tag)

    if (not args.overwrite) and already_done(paper_dir, args.model, args.review_agent, args.synth_model, args.model_tag):
        return paper_dir, True, "skip_done"

    # mark inprogress
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

        agent = AuditAgent(
            model=args.model,
        )

        if args.reuse_cache:
            rdir.mkdir(parents=True, exist_ok=True)
            st = cache_status(rdir)
            with _PRINT_LOCK:
                print(
                    f"[CACHE] {paper_dir.name} :: "
                    f"baseline={'Y' if st['baseline'] else 'N'} "
                    f"final={'Y' if st['final'] else 'N'} "
                    f"memory={'Y' if st['paper_memory'] else 'N'} "
                    f"cheat={'Y' if st['cheat'] else 'N'} "
                    f"motiv={'Y' if st['motivation'] else 'N'}"
                )

        enable_mm = not bool(getattr(args, "disable_mm", False))

        result = agent.review_paper(
            paper_json=paper_json,
            query=args.query,
            enable_mm=enable_mm,
            enable_cheating_detection=args.enable_cheating_detection,
            enable_motivation=args.enable_motivation,
            reuse_cache=args.reuse_cache,
            cache_dir=rdir,
            model_tag=args.model_tag,
        )

        baseline = result.get("baseline_review", "") or ""
        final_review = result.get("final_assessment", "") or ""

        baseline_scores = extract_scores(baseline)
        refined_scores = extract_scores(final_review)

        payload: Dict[str, Any] = {
            "baseline_review": baseline,
            "final_review": final_review,
            "scores": {
                "baseline": baseline_scores,
                "refined": refined_scores,
            },
            "reviews": final_review,  # backward-compat for already_done()
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
        res = await coro
        results.append(res)

        d, ok, msg = res
        with _PRINT_LOCK:
            if msg == "ok":
                outp = out_json_path(d, args.model, args.review_agent, args.synth_model, args.model_tag)
                # outp is under d/... so relative_to(d) is safe
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

    enable_mm = not bool(getattr(args, "disable_mm", False))

    print(
        f"[INFO] root={root_dir}  jobs={args.jobs}  model={args.model}  "
        f"review_agent={args.review_agent}  synth_model={args.synth_model or 'origin'}  "
        f"overwrite={args.overwrite}  reuse_cache={args.reuse_cache}  model_tag={args.model_tag}  "
        f"mm={enable_mm}  cheat={args.enable_cheating_detection}  motiv={args.enable_motivation}"
    )

    summary = asyncio.run(run_batch(root_dir=root_dir, jobs=args.jobs, args=args))

    print(
        f"[DONE] total={summary['total']}  ok={summary['ok']}  "
        f"fail={len(summary['fail'])}  skipped_done={summary.get('skipped_done', 0)}"
    )
    if summary["fail"]:
        print("[FAILED ITEMS]")
        for i, (path, msg) in enumerate(summary["fail"], 1):
            print(f"  {i}. {path} :: {msg}")


if __name__ == "__main__":
    main()
