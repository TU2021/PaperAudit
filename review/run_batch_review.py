#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
End-to-end batch review runner aligned with detect/4_1_mas_error_detection.py style.

Usage examples:
    python review/run_batch_review.py \
        --input-dir /path/to/papers \
        --model qwen3-235b-a22b-instruct-2507 \
        --synth-model o4-mini \
        --use-synth \
        --no-cheating-detection
"""
from __future__ import annotations

import argparse
import json
from datetime import datetime
from pathlib import Path
from typing import Any, Dict, Optional

from dotenv import load_dotenv

from agents import PaperReviewAgent

load_dotenv()


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Batch review runner")
    parser.add_argument("--input-dir", required=True, help="Root folder containing paper subfolders")
    parser.add_argument("--model", required=True, help="LLM model name (required)")
    parser.add_argument("--reasoning-model", help="Optional reasoning model override")
    parser.add_argument("--embedding-model", help="Optional embedding model override")
    parser.add_argument(
        "--query",
        default="Please provide a critical, structured peer review of the paper.",
        help="Instruction/query passed to the reviewer",
    )
    parser.add_argument("--synth-model", default="o4-mini", help="Synthetic data model suffix")
    parser.add_argument(
        "--use-synth",
        action="store_true",
        help="Prefer paper_synth_{synth_model}.json over paper_final.json",
    )
    parser.add_argument("--enable-mm", action="store_true", help="Enable multimodal text markers")
    parser.add_argument(
        "--no-cheating-detection",
        dest="enable_cheating_detection",
        action="store_false",
        help="Disable cheating detection stage",
    )
    parser.add_argument(
        "--no-motivation",
        dest="enable_motivation",
        action="store_false",
        help="Disable motivation evaluation stage",
    )
    parser.set_defaults(enable_cheating_detection=True, enable_motivation=True)
    return parser.parse_args()


def sanitize_filename(name: str) -> str:
    return "".join(c if c.isalnum() or c in ("-", "_") else "_" for c in name)


def resolve_paper_file(paper_dir: Path, use_synth: bool, synth_model: str) -> Optional[Path]:
    synth_file = paper_dir / f"paper_synth_{synth_model}.json"
    final_file = paper_dir / "paper_final.json"

    if use_synth and synth_file.exists():
        return synth_file
    if not use_synth and final_file.exists():
        return final_file

    # Fallback: try the other option if preferred one is missing
    if final_file.exists():
        return final_file
    if synth_file.exists():
        return synth_file
    return None


def load_json(path: Path) -> Dict[str, Any]:
    with path.open("r", encoding="utf-8") as f:
        return json.load(f)


def save_json(path: Path, obj: Dict[str, Any]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("w", encoding="utf-8") as f:
        json.dump(obj, f, ensure_ascii=False, indent=2)


def main() -> None:
    args = parse_args()
    input_dir = Path(args.input_dir).expanduser().resolve()
    if not input_dir.exists() or not input_dir.is_dir():
        raise SystemExit(f"Input directory not found: {input_dir}")

    agent = PaperReviewAgent(
        model=args.model,
        reasoning_model=args.reasoning_model,
        embedding_model=args.embedding_model,
    )

    paper_dirs = [p for p in sorted(input_dir.iterdir()) if p.is_dir()]
    if not paper_dirs:
        raise SystemExit(f"No paper subdirectories found under {input_dir}")

    print(f"Found {len(paper_dirs)} papers under {input_dir}\n")

    for idx, paper_dir in enumerate(paper_dirs, start=1):
        paper_file = resolve_paper_file(paper_dir, args.use_synth, args.synth_model)
        if paper_file is None:
            print(f"[{idx}/{len(paper_dirs)}] Skipping {paper_dir.name}: no paper_final.json or synth file found")
            continue

        print(f"[{idx}/{len(paper_dirs)}] Reviewing {paper_dir.name} using {paper_file.name} ...")
        paper_json = load_json(paper_file)

        result = agent.review_paper(
            paper_json=paper_json,
            query=args.query,
            enable_mm=args.enable_mm,
            enable_cheating_detection=args.enable_cheating_detection,
            enable_motivation=args.enable_motivation,
        )

        timestamp = datetime.utcnow().isoformat()
        review_dir = paper_dir / "reviews"
        review_dir.mkdir(parents=True, exist_ok=True)

        model_tag = sanitize_filename(args.model)
        summary_path = review_dir / f"final_summary_{model_tag}.txt"
        summary_path.write_text(result.get("final_assessment", ""), encoding="utf-8")

        output_path = review_dir / f"review_output_{model_tag}.json"
        save_json(
            output_path,
            {
                "model": args.model,
                "reasoning_model": args.reasoning_model or args.model,
                "embedding_model": args.embedding_model or args.model,
                "enable_mm": args.enable_mm,
                "enable_cheating_detection": args.enable_cheating_detection,
                "enable_motivation": args.enable_motivation,
                "paper_source": str(paper_file.name),
                "timestamp": timestamp,
                "results": result,
            },
        )

        print(f"    ✓ Saved summary to {summary_path.relative_to(paper_dir)}")
        print(f"    ✓ Saved detailed output to {output_path.relative_to(paper_dir)}\n")

    print("Batch review completed.")


if __name__ == "__main__":
    main()
