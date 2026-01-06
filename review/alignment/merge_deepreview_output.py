#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
DeepReview Output Merger

Batch merge/consolidate DeepReviewer outputs into a single merged review per paper.
This script takes multiple review outputs (possibly from multiple reviewers) and
consolidates them into one comprehensive review.

The merger:
- Reads multiple review files from DeepReviewer output directory
- Uses LLM to consolidate reviews into one comprehensive review
- Outputs only Summary, Strengths, and Weaknesses sections
- Removes numeric scores and keeps only substantive content

Usage Examples:
    # Basic merge
    python merge_deepreview_output.py \
        --input_dir /path/to/papers \
        --review_agent deepreviewer \
        --ai_model gpt-5-2025-08-07 \
        --ai_review_file deep_review_all.txt \
        --out_file deep_review_merge.txt \
        --model gpt-5-2025-08-07 \
        --jobs 10

Key Arguments:
    --input_dir: Root folder containing paper subfolders. Default: /mnt/parallel_ssd/home/zdhs0006/ACL/data/ICLR_26
    --review_agent: Review agent name (folder under reviews/). Default: deepreviewer
    --ai_model: AI model that generated the reviews. Default: gpt-5-2025-08-07
    --ai_review_file: Input review file to merge. Default: deep_review_all.txt
    --out_file: Output merged review filename. Default: deep_review_merge.txt
    --model: LLM model for merging. Default: gpt-5-2025-08-07
    --jobs: Number of concurrent merges. Default: 1
    --overwrite: Force re-run and overwrite existing outputs. Default: False

Input:
    - Review files: <paper_dir>/reviews/{review_agent}/{ai_model}/{paper_origin}/{ai_review_file}

Output:
    - Merged review: <paper_dir>/reviews/{review_agent}/{ai_model}/{paper_origin}/{out_file}

Output Format:
    The merged review contains only three sections:
    1) Summary
    2) Strengths
    3) Weaknesses

    - No numeric scores
    - No extra sections (no Suggestions, Questions, Ratings)
    - As complete and detailed as possible
    - De-duplicates repeated points while preserving meaningful disagreements
"""

from __future__ import annotations

import argparse
import asyncio
import json
import re
import sys
import time
import threading
from dataclasses import dataclass
from pathlib import Path
from typing import Optional, List, Tuple

try:
    from tqdm import tqdm
except Exception:
    tqdm = None

from dotenv import load_dotenv

load_dotenv()
_PRINT_LOCK = threading.Lock()

# ---------------- prompts ----------------

MERGE_REVIEW_PROMPT = """You are an expert meta-reviewer.

You are given reviewer comments for the SAME paper (possibly multiple reviews concatenated).
Please produce ONE consolidated initial review for this paper.

Requirements:
- Output ONLY these three sections with headings exactly:
  1) Summary
  2) Strengths
  3) Weaknesses
- The consolidated review should be as complete and detailed as possible.
- De-duplicate repeated points, but keep meaningful disagreements if any.
- Do NOT include any numerical scores.
- Do NOT add extra sections (no Suggestions, no Questions, no Ratings).

Now merge and consolidate the following review text:
"""


# ---------------- LLM client ----------------

@dataclass
class LLMConfig:
    model: str
    api_key: Optional[str] = None
    base_url: Optional[str] = None
    temperature: float = 0.0
    max_tokens: int = 4000
    timeout_s: int = 180


class AsyncChatLLM:
    """
    Minimal async chat wrapper using OpenAI-compatible SDK.

    Env vars supported by default:
      - OPENAI_API_KEY
      - OPENAI_BASE_URL (optional)
    """
    def __init__(self, cfg: LLMConfig):
        self.cfg = cfg
        try:
            from openai import AsyncOpenAI  # type: ignore
        except Exception as e:
            raise RuntimeError(
                "Missing dependency `openai`. Please `pip install openai` (OpenAI-compatible SDK)."
            ) from e

        api_key = cfg.api_key
        if not api_key:
            import os
            api_key = os.environ.get("OPENAI_API_KEY")

        base_url = cfg.base_url
        if not base_url:
            import os
            base_url = os.environ.get("OPENAI_BASE_URL") or None

        if not api_key:
            raise RuntimeError("OPENAI_API_KEY is not set (and --api_key not provided).")

        self.client = AsyncOpenAI(api_key=api_key, base_url=base_url)

    async def complete_text(self, prompt: str) -> str:
        resp = await self.client.chat.completions.create(
            model=self.cfg.model,
            messages=[
                {"role": "system", "content": "You are a helpful assistant."},
                {"role": "user", "content": prompt},
            ],
            temperature=self.cfg.temperature,
            max_tokens=self.cfg.max_tokens,
            timeout=self.cfg.timeout_s,
        )
        return (resp.choices[0].message.content or "").strip()


# ---------------- CLI ----------------

def parse_args() -> argparse.Namespace:
    p = argparse.ArgumentParser(description="Batch merge deep reviews into deep_review_merge.txt")
    p.add_argument(
        "--input_dir",
        default="/mnt/parallel_ssd/home/zdhs0006/ACL/data/ICLR_26",
        help="Root folder containing paper subfolders (each subfolder is one paper).",
    )

    p.add_argument(
        "--judge_model",
        default="gpt-5-2025-08-07",
        help="LLM model used to merge/summarize reviews.",
    )
    p.add_argument("--api_key", default=None, help="Override OPENAI_API_KEY")
    p.add_argument("--base_url", default=None, help="Override OPENAI_BASE_URL")
    p.add_argument("--temperature", type=float, default=0.0)
    p.add_argument("--max_tokens", type=int, default=20000)
    p.add_argument("--timeout_s", type=int, default=180)

    p.add_argument(
        "--review_agent",
        default="deepreviewer",
        help="Folder name under reviews/, e.g., deepreviewer",
    )
    p.add_argument(
        "--ai_model",
        default="gpt-5-2025-08-07",
        help="Model folder under reviews/{review_agent}/",
    )
    p.add_argument(
        "--ai_review_file",
        default="deep_review_all.txt",
        help="Input review file under .../paper_origin/",
    )
    p.add_argument(
        "--out_file",
        default="deep_review_merge.txt",
        help="Output merged review filename under .../paper_origin/",
    )

    p.add_argument(
        "--jobs", "-j",
        type=int,
        default=10,
        help="Number of in-flight papers (threadpool under the hood).",
    )
    p.add_argument("--overwrite", action="store_true", help="Overwrite output if exists.")
    return p.parse_args()


# ---------------- utils ----------------

def sanitize_filename(name: str) -> str:
    return "".join(c if c.isalnum() or c in ("-", "_", ".") else "_" for c in name)

def load_text(path: Path) -> str:
    return path.read_text(encoding="utf-8", errors="ignore").strip()

def save_text(path: Path, text: str) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(text, encoding="utf-8")

def ensure_sections(text: str) -> str:
    """
    Best-effort: ensure the output contains exactly the three required headings.
    If the model returns extra sections, we keep only Summary/Strengths/Weaknesses in order.
    """
    t = (text or "").strip()
    if not t:
        return t

    # Normalize headings to exactly "Summary", "Strengths", "Weaknesses"
    def norm_head(s: str) -> str:
        s = re.sub(r"(?im)^\s*#{1,6}\s*", "", s).strip()
        s = re.sub(r"(?im)^\s*\d+\s*[\)\.\-]\s*", "", s).strip()
        s = s.replace("Weakness / Concerns", "Weaknesses").replace("Weaknesses / Concerns", "Weaknesses")
        return s.lower()

    lines = t.splitlines()
    idx = {}
    for i, ln in enumerate(lines):
        key = norm_head(ln)
        if key == "summary" and "summary" not in idx:
            idx["summary"] = i
        elif key == "strengths" and "strengths" not in idx:
            idx["strengths"] = i
        elif key in ("weaknesses", "weakness") and "weaknesses" not in idx:
            idx["weaknesses"] = i

    if not all(k in idx for k in ("summary", "strengths", "weaknesses")):
        # If headings missing, wrap into the 3-section format as-is
        return "Summary\n" + t + "\n\nStrengths\n- (not explicitly separated)\n\nWeaknesses\n- (not explicitly separated)\n"

    order = sorted([(idx["summary"], "Summary"), (idx["strengths"], "Strengths"), (idx["weaknesses"], "Weaknesses")], key=lambda x: x[0])
    if [x[1] for x in order] != ["Summary", "Strengths", "Weaknesses"]:
        return t + "\n"

    s0, _ = order[0]
    s1, _ = order[1]
    s2, _ = order[2]

    summary = "\n".join(lines[s0 + 1 : s1]).strip()
    strengths = "\n".join(lines[s1 + 1 : s2]).strip()
    weaknesses = "\n".join(lines[s2 + 1 :]).strip()

    out = []
    out.append("Summary")
    out.append(summary if summary else "- (empty)")
    out.append("")
    out.append("Strengths")
    out.append(strengths if strengths else "- (empty)")
    out.append("")
    out.append("Weaknesses")
    out.append(weaknesses if weaknesses else "- (empty)")
    return "\n".join(out).strip() + "\n"

def find_paper_dirs(root_dir: Path) -> List[Path]:
    return [p for p in sorted(root_dir.iterdir()) if p.is_dir()]

def in_path_for(paper_dir: Path, review_agent: str, ai_model: str, ai_review_file: str) -> Path:
    return paper_dir / "reviews" / sanitize_filename(review_agent) / sanitize_filename(ai_model) / "paper_origin" / ai_review_file

def out_path_for(paper_dir: Path, review_agent: str, ai_model: str, out_file: str) -> Path:
    return paper_dir / "reviews" / sanitize_filename(review_agent) / sanitize_filename(ai_model) / "paper_origin" / out_file

def already_done(out_path: Path) -> bool:
    return out_path.exists() and out_path.stat().st_size > 50


# ---------------- single paper worker (SYNC) ----------------

def process_one_paper_sync(paper_dir: Path, args: argparse.Namespace) -> Tuple[Path, bool, str]:
    in_path = in_path_for(paper_dir, args.review_agent, args.ai_model, args.ai_review_file)
    out_path = out_path_for(paper_dir, args.review_agent, args.ai_model, args.out_file)

    if not in_path.exists():
        return paper_dir, True, "skip_missing_input"

    if out_path.exists() and (not args.overwrite) and already_done(out_path):
        return paper_dir, True, "skip_done"

    try:
        raw = load_text(in_path)
        if not raw:
            return paper_dir, True, "skip_empty_input"

        prompt = MERGE_REVIEW_PROMPT + "\n\n" + raw

        loop = asyncio.new_event_loop()
        asyncio.set_event_loop(loop)

        llm = AsyncChatLLM(
            LLMConfig(
                model=args.judge_model,
                api_key=args.api_key,
                base_url=args.base_url,
                temperature=args.temperature,
                max_tokens=args.max_tokens,
                timeout_s=args.timeout_s,
            )
        )

        merged = loop.run_until_complete(llm.complete_text(prompt)).strip()
        try:
            loop.close()
        except Exception:
            pass

        merged = ensure_sections(merged)
        save_text(out_path, merged)

        return paper_dir, True, "ok"

    except Exception as e:
        return paper_dir, False, f"error: {repr(e)}"


# ---------------- async scheduler ----------------

async def bounded_worker(
    sem: asyncio.Semaphore,
    loop: asyncio.AbstractEventLoop,
    paper_dir: Path,
    args: argparse.Namespace,
):
    async with sem:
        return await loop.run_in_executor(None, process_one_paper_sync, paper_dir, args)

async def run_batch(root_dir: Path, jobs: int, args: argparse.Namespace) -> Dict[str, Any]:
    paper_dirs = find_paper_dirs(root_dir)
    if not paper_dirs:
        return {"total": 0, "ok": 0, "fail": [], "skipped": 0}

    sem = asyncio.Semaphore(jobs)
    loop = asyncio.get_running_loop()
    tasks = [bounded_worker(sem, loop, d, args) for d in paper_dirs]

    results: List[Tuple[Path, bool, str]] = []
    iterator = asyncio.as_completed(tasks)
    if tqdm is not None:
        iterator = tqdm(iterator, total=len(tasks), desc="Merging reviews")

    skipped = 0
    for coro in iterator:
        d, ok, msg = await coro
        results.append((d, ok, msg))

        with _PRINT_LOCK:
            if msg == "ok":
                outp = out_path_for(d, args.review_agent, args.ai_model, args.out_file)
                rel = outp.relative_to(d)
                print(f"[DONE] {d.name} -> {rel}")
            elif msg.startswith("skip_"):
                skipped += 1
                print(f"[SKIP] {d.name} ({msg})")
            else:
                print(f"[FAIL] {d.name} :: {msg}", file=sys.stderr)

    ok_cnt = sum(1 for (_, ok, msg) in results if ok and msg == "ok")
    fail = [(str(d), msg) for (d, ok, msg) in results if not ok]

    return {
        "total": len(results),
        "ok": ok_cnt,
        "fail": fail,
        "skipped": skipped,
    }


def main() -> None:
    args = parse_args()
    root_dir = Path(args.input_dir).expanduser().resolve()
    if not root_dir.exists() or not root_dir.is_dir():
        raise SystemExit(f"Input directory not found: {root_dir}")

    print(
        f"[INFO] root={root_dir} jobs={args.jobs} judge_model={args.judge_model} "
        f"review_agent={args.review_agent} ai_model={args.ai_model} "
        f"in={args.ai_review_file} out={args.out_file} overwrite={args.overwrite}"
    )

    summary = asyncio.run(run_batch(root_dir=root_dir, jobs=args.jobs, args=args))

    print(
        f"[DONE] total={summary['total']} ok={summary['ok']} "
        f"fail={len(summary['fail'])} skipped={summary.get('skipped', 0)}"
    )
    if summary["fail"]:
        print("[FAILED ITEMS]")
        for i, (path, msg) in enumerate(summary["fail"], 1):
            print(f"  {i}. {path} :: {msg}")


if __name__ == "__main__":
    main()
