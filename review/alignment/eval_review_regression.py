#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Review Regression Analyzer (minimal + robust)

Given TWO AI reviews for the SAME paper (baseline vs final), use an LLM to:
1) Extract explicit numeric score delta (if present)
2) Extract substantive differences (new/intensified critiques, dropped strengths, rationale shift)
   and tag each difference with 1+ PaperAudit error types.

Supports paper variants:
- paper_origin (default)
- paper_synth_<synth_model> (via --synth_model)

Batch runner:
- input_dir contains paper subfolders
- reads two review files under each paper
- outputs JSON under reviews/regression_judge/{judge_model}/{paper_variant}/regression_{tag}.json
- supports concurrency, resume via config_key, overwrite

Example:
python eval_review_regression.py \
  --input_dir /data/home/zdhs0006/ACL/data/ICML_25 \
  --judge_model gpt-5-2025-08-07 \
  --review_agent paper_audit \
  --ai_model gpt-5-2025-08-07 \
  --baseline_file baseline_review.txt \
  --final_file final_review_all.txt \
  --model_tag v1 \
  -j 10 --overwrite

Example (synth):
python eval_review_regression.py \
  --input_dir /data/home/zdhs0006/ACL/data/ICML_25 \
  --judge_model gpt-5-2025-08-07 \
  --review_agent paper_audit \
  --ai_model gpt-5-2025-08-07 \
  --synth_model gpt-5-2025-08-07 \
  --baseline_file baseline_review.txt \
  --final_file final_review_all.txt \
  --model_tag v1 \
  -j 10 --overwrite
"""

from __future__ import annotations

import argparse
import asyncio
import hashlib
import json
import re
import sys
import time
import threading
from dataclasses import dataclass
from pathlib import Path
from typing import Any, Dict, List, Optional, Tuple

try:
    from tqdm import tqdm
except Exception:
    tqdm = None

from dotenv import load_dotenv
load_dotenv()

_PRINT_LOCK = threading.Lock()

# ---------------- prompts ----------------

REVIEW_REGRESSION_JUDGE_PROMPT = """You are an expert meta-reviewer.

You are given TWO AI-generated reviews of the SAME paper:
- Review A: baseline review
- Review B: final review

Your task (ONLY do these two things):
1) Extract the numeric score difference between the two reviews (baseline vs final), if explicit.
2) Identify the substantive REVIEW DIFFERENCES between the two reviews that could explain why Review B
   assigns a LOWER score / is MORE negative than Review A.

For each difference, assign ONE OR MORE PaperAudit error types.

PaperAudit 8 error types (ENUM; you may assign multiple per difference):
- EVIDENCE_DATA_INTEGRITY
- METHOD_LOGIC_CONSISTENCY
- EXPERIMENTAL_DESIGN_PROTOCOL
- CLAIM_RESULT_DISTORTION
- REFERENCE_BACKGROUND_FABRICATION
- ETHICAL_INTEGRITY_OMISSION
- RHETORICAL_PRESENTATION_MANIPULATION
- CONTEXT_MISALIGNMENT_INCOHERENCE

Guidelines:
- Focus on content and score rationale cues, not style/tone.
- Differences can be:
  (a) new critiques in final (not in baseline),
  (b) intensified critiques (same issue but much harsher in final),
  (c) dropped strengths (baseline praised something that final no longer praises),
  (d) explicit score-rationale shift (final adds a reason for lowering score).
- Each difference must include a short evidence quote from BOTH reviews (<=25 words each).
- Keep it concise: 3–8 differences.

Score extraction rules:
- If a review has no explicit numeric score, set that score to null.
- scale_hint: "1-10" | "1-5" | "textual" | "unknown"
- If both scores are present, delta must equal (final_score - baseline_score). Otherwise delta=null.

OUTPUT FORMAT (STRICT JSON ONLY; no markdown):
{
  "score_delta": {
    "baseline_score": float|null,
    "final_score": float|null,
    "delta": float|null,
    "scale_hint": "1-10"|"1-5"|"textual"|"unknown"
  },
  "differences": [
    {
      "diff_type": "new_critique"|"intensified_critique"|"dropped_strength"|"score_rationale_shift"|"other",
      "summary": "short phrase describing the difference",
      "paperaudit_types": ["ENUM", "ENUM"],
      "why_impacts_score": "short phrase",
      "evidence": {
        "baseline_quote": "quote <=25 words",
        "final_quote": "quote <=25 words"
      }
    }
  ]
}
"""

# ---------------- normalize / schema enforcement ----------------

_ALLOWED_PAPERAUDIT_TYPES = {
    "EVIDENCE_DATA_INTEGRITY",
    "METHOD_LOGIC_CONSISTENCY",
    "EXPERIMENTAL_DESIGN_PROTOCOL",
    "CLAIM_RESULT_DISTORTION",
    "REFERENCE_BACKGROUND_FABRICATION",
    "ETHICAL_INTEGRITY_OMISSION",
    "RHETORICAL_PRESENTATION_MANIPULATION",
    "CONTEXT_MISALIGNMENT_INCOHERENCE",
}

_ALLOWED_SCALE_HINTS = {"1-10", "1-5", "textual", "unknown"}

_ALLOWED_DIFF_TYPES = {
    "new_critique",
    "intensified_critique",
    "dropped_strength",
    "score_rationale_shift",
    "other",
}


def _to_float_or_none(x: Any) -> Optional[float]:
    try:
        if x is None:
            return None
        return float(x)
    except Exception:
        return None


def _norm_str(x: Any) -> str:
    return str(x).strip() if x is not None else ""


def _norm_scale_hint(x: Any) -> str:
    s = _norm_str(x) or "unknown"
    return s if s in _ALLOWED_SCALE_HINTS else "unknown"


def _norm_diff_type(x: Any) -> str:
    t = _norm_str(x) or "other"
    return t if t in _ALLOWED_DIFF_TYPES else "other"


def _norm_types_list(x: Any) -> List[str]:
    if not isinstance(x, list):
        return []
    out: List[str] = []
    for it in x[:8]:
        s = _norm_str(it)
        if s in _ALLOWED_PAPERAUDIT_TYPES and s not in out:
            out.append(s)
    return out


def normalize_obj(obj: Dict[str, Any]) -> Dict[str, Any]:
    """
    Minimal output:
    - score_delta
    - differences (each diff tagged with 0..N PaperAudit types)
    """
    out: Dict[str, Any] = {}

    # --- score_delta ---
    sd = obj.get("score_delta") if isinstance(obj.get("score_delta"), dict) else {}
    baseline = _to_float_or_none(sd.get("baseline_score"))
    final = _to_float_or_none(sd.get("final_score"))
    scale_hint = _norm_scale_hint(sd.get("scale_hint", "unknown"))

    delta: Optional[float]
    if baseline is not None and final is not None:
        delta = round(final - baseline, 2)
    else:
        delta = None

    out["score_delta"] = {
        "baseline_score": baseline,
        "final_score": final,
        "delta": delta,
        "scale_hint": scale_hint,
    }

    # --- differences ---
    diffs = obj.get("differences", [])
    norm_diffs: List[Dict[str, Any]] = []

    if isinstance(diffs, list):
        for d in diffs[:8]:
            if not isinstance(d, dict):
                continue
            ev = d.get("evidence") if isinstance(d.get("evidence"), dict) else {}

            summary = _norm_str(d.get("summary"))
            if not summary:
                continue

            norm_diffs.append(
                {
                    "diff_type": _norm_diff_type(d.get("diff_type")),
                    "summary": summary,
                    "paperaudit_types": _norm_types_list(d.get("paperaudit_types")),
                    "why_impacts_score": _norm_str(d.get("why_impacts_score")),
                    "evidence": {
                        "baseline_quote": _norm_str(ev.get("baseline_quote")),
                        "final_quote": _norm_str(ev.get("final_quote")),
                    },
                }
            )

    out["differences"] = norm_diffs
    return out


# ---------------- LLM client ----------------

@dataclass
class LLMConfig:
    model: str
    api_key: Optional[str] = None
    base_url: Optional[str] = None
    temperature: float = 0.0
    max_tokens: int = 1600
    timeout_s: int = 180


class AsyncChatLLM:
    """
    Minimal async chat wrapper using OpenAI-compatible SDK.
    Env vars:
      - OPENAI_API_KEY
      - OPENAI_BASE_URL (optional)
    """
    def __init__(self, cfg: LLMConfig):
        self.cfg = cfg
        try:
            from openai import AsyncOpenAI  # type: ignore
        except Exception as e:
            raise RuntimeError("Missing dependency `openai`. Please `pip install openai`.") from e

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
    p = argparse.ArgumentParser(description="Batch review regression analyzer (baseline vs final)")
    p.add_argument(
        "--input_dir",
        default="/data/home/zdhs0006/ACL/data/ICML_25",
        help="Root folder containing paper subfolders.",
    )

    p.add_argument("--judge_model", default="gpt-5-2025-08-07")
    p.add_argument("--api_key", default=None)
    p.add_argument("--base_url", default=None)
    p.add_argument("--temperature", type=float, default=0.0)
    p.add_argument("--max_tokens", type=int, default=20000)
    p.add_argument("--timeout_s", type=int, default=600)

    p.add_argument("--review_agent", default="paper_audit")
    p.add_argument("--ai_model", default="gpt-5-2025-08-07")
    p.add_argument("--baseline_file", default="baseline_review.txt")
    p.add_argument("--final_file", default="final_review_all.txt")

    # NEW: paper variant selection
    p.add_argument(
        "--synth_model",
        default="gpt-5-2025-08-07",
        help="If set, read reviews from paper_synth_<synth_model> instead of paper_origin. "
             "Example: --synth_model gpt-5-2025-08-07",
    )

    p.add_argument("--model_tag", default="v1")
    p.add_argument("--jobs", "-j", type=int, default=10)
    p.add_argument("--overwrite", action="store_true", default=False)

    # Useful for debugging a couple of papers quickly
    p.add_argument("--debug_llm", action="store_true", default=False,
                   help="Print LLM output head + parsed keys.")
    return p.parse_args()


# ---------------- utils ----------------

def sanitize_filename(name: str) -> str:
    return "".join(c if c.isalnum() or c in ("-", "_", ".") else "_" for c in name)

def load_text(path: Path) -> str:
    return path.read_text(encoding="utf-8", errors="ignore").strip()

def save_json(path: Path, obj: Dict[str, Any]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps(obj, ensure_ascii=False, indent=2), encoding="utf-8")

def load_json(path: Path) -> Dict[str, Any]:
    return json.loads(path.read_text(encoding="utf-8", errors="ignore") or "{}")

def paper_variant_dir(args: argparse.Namespace) -> str:
    return "paper_origin" if not args.synth_model else f"paper_synth_{args.synth_model}"

def regression_dir(paper_dir: Path, judge_model: str, variant: str) -> Path:
    return paper_dir / "reviews" / "regression_judge" / sanitize_filename(judge_model) / variant

def regression_out(paper_dir: Path, judge_model: str, tag: str, variant: str) -> Path:
    return regression_dir(paper_dir, judge_model, variant) / f"regression_{sanitize_filename(tag)}.json"

def find_paper_dirs(root_dir: Path) -> List[Path]:
    return [p for p in sorted(root_dir.iterdir()) if p.is_dir()]

def stable_json_dumps(obj: Any) -> str:
    return json.dumps(obj, ensure_ascii=False, sort_keys=True, separators=(",", ":"))

def make_config(args: argparse.Namespace) -> Dict[str, Any]:
    variant = paper_variant_dir(args)
    return {
        "judge_model": args.judge_model,
        "review_agent": args.review_agent,
        "ai_model": args.ai_model,
        "baseline_file": args.baseline_file,
        "final_file": args.final_file,
        "paper_variant": variant,           # NEW
        "synth_model": args.synth_model,    # NEW (None for origin)
        "temperature": float(args.temperature),
        "max_tokens": int(args.max_tokens),
        "timeout_s": int(args.timeout_s),
        "metric": "regression_v2_minimal",
        "prompt_hash": hashlib.sha1(REVIEW_REGRESSION_JUDGE_PROMPT.encode("utf-8")).hexdigest()[:10],
    }

def make_config_key(config: Dict[str, Any]) -> str:
    s = stable_json_dumps(config).encode("utf-8")
    return hashlib.sha1(s).hexdigest()

def get_runs_container(existing: Dict[str, Any]) -> List[Dict[str, Any]]:
    runs = existing.get("runs")
    return runs if isinstance(runs, list) else []

def has_run_with_key(existing: Dict[str, Any], config_key: str) -> bool:
    for r in get_runs_container(existing):
        if isinstance(r, dict) and r.get("config_key") == config_key:
            return True
    return False


# ---------------- robust JSON extraction ----------------

def extract_first_json_object(s: str) -> Dict[str, Any]:
    raw = (s or "").strip()
    if not raw:
        raise ValueError("Empty LLM response.")

    try:
        obj = json.loads(raw)
        if isinstance(obj, dict):
            return obj
    except Exception:
        pass

    starts = [m.start() for m in re.finditer(r"\{", raw)]
    for st in starts:
        for ed in range(len(raw), st + 2, -1):
            if raw[ed - 1] != "}":
                continue
            cand = raw[st:ed]
            try:
                obj = json.loads(cand)
                if isinstance(obj, dict):
                    return obj
            except Exception:
                continue
    raise ValueError("Could not extract a valid JSON object from LLM response.")


# ---------------- per-paper worker ----------------

def process_one_paper_sync(paper_dir: Path, args: argparse.Namespace) -> Tuple[Path, bool, str]:
    variant = paper_variant_dir(args)

    base_path = paper_dir / "reviews" / args.review_agent / args.ai_model / variant / args.baseline_file
    final_path = paper_dir / "reviews" / args.review_agent / args.ai_model / variant / args.final_file

    if not base_path.exists():
        return paper_dir, True, "skip_missing_baseline"
    if not final_path.exists():
        return paper_dir, True, "skip_missing_final"

    outp = regression_out(paper_dir, args.judge_model, args.model_tag, variant)

    config = make_config(args)
    config_key = make_config_key(config)

    existing: Dict[str, Any] = {}
    if outp.exists():
        try:
            existing = load_json(outp)
        except Exception:
            existing = {}

    if (not args.overwrite) and outp.exists() and has_run_with_key(existing, config_key):
        return paper_dir, True, "skip_done_same_config"

    try:
        outp.parent.mkdir(parents=True, exist_ok=True)

        baseline_review = load_text(base_path)
        final_review = load_text(final_path)

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

        prompt = (
            REVIEW_REGRESSION_JUDGE_PROMPT
            + "\n\n[Review A — Baseline]\n"
            + baseline_review
            + "\n\n[Review B — Final]\n"
            + final_review
        )

        resp = loop.run_until_complete(llm.complete_text(prompt)).strip()
        loop.close()

        raw_obj = extract_first_json_object(resp)

        if args.debug_llm:
            with _PRINT_LOCK:
                print(f"[DEBUG] {paper_dir.name} LLM head:", resp[:300].replace("\n", " "))
                print(f"[DEBUG] {paper_dir.name} parsed keys:", list(raw_obj.keys()))

        reg_obj = normalize_obj(raw_obj)

        run_obj: Dict[str, Any] = {
            "config": config,
            "config_key": config_key,
            "inputs": {
                "baseline_review": str(base_path.relative_to(paper_dir)),
                "final_review": str(final_path.relative_to(paper_dir)),
            },
            "regression": reg_obj,
            "generated_at": time.strftime("%Y-%m-%dT%H:%M:%S", time.localtime()),
        }

        # if overwrite, discard the entire existing file
        if args.overwrite:
            container = {"paper": paper_dir.name, "runs": []}
        else:
            if isinstance(existing, dict) and existing:
                container = existing
            else:
                container = {"paper": paper_dir.name, "runs": []}

        if "paper" not in container:
            container["paper"] = paper_dir.name

        runs = container.get("runs")
        if not isinstance(runs, list):
            runs = []

        runs.append(run_obj)
        container["runs"] = runs

        save_json(outp, container)
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
        return {"total": 0, "ok": 0, "fail": [], "skipped_done": 0}

    sem = asyncio.Semaphore(jobs)
    loop = asyncio.get_running_loop()
    tasks = [bounded_worker(sem, loop, d, args) for d in paper_dirs]

    results: List[Tuple[Path, bool, str]] = []
    iterator = asyncio.as_completed(tasks)
    if tqdm is not None:
        iterator = tqdm(iterator, total=len(tasks), desc="Review regression judging")

    skipped_done = 0
    for coro in iterator:
        d, ok, msg = await coro
        results.append((d, ok, msg))
        with _PRINT_LOCK:
            if msg == "ok":
                outp = regression_out(d, args.judge_model, args.model_tag, paper_variant_dir(args))
                print(f"[DONE] {d.name} -> {outp.relative_to(d)}")
            elif msg.startswith("skip_"):
                skipped_done += 1
                print(f"[SKIP] {d.name} ({msg})")
            else:
                print(f"[FAIL] {d.name} :: {msg}", file=sys.stderr)

    ok_cnt = sum(1 for (_, ok, msg) in results if ok and msg == "ok")
    fail = [(str(d), msg) for (d, ok, msg) in results if not ok]
    return {"total": len(results), "ok": ok_cnt, "fail": fail, "skipped_done": skipped_done}

def main() -> None:
    args = parse_args()
    root_dir = Path(args.input_dir).expanduser().resolve()
    if not root_dir.exists() or not root_dir.is_dir():
        raise SystemExit(f"Input directory not found: {root_dir}")

    print(
        f"[INFO] root={root_dir} jobs={args.jobs} judge_model={args.judge_model} "
        f"review_agent={args.review_agent} ai_model={args.ai_model} "
        f"variant={paper_variant_dir(args)} "
        f"baseline_file={args.baseline_file} final_file={args.final_file} "
        f"overwrite={args.overwrite} tag={args.model_tag} debug_llm={args.debug_llm}"
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
