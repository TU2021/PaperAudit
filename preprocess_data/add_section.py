#!/usr/bin/env python3
# -*- coding: utf-8 -*-

"""
Batch Section Labeling Tool

This script processes paper JSON files in batch:
1) For each text in content, calls LLM to split into clearer paragraphs based on section boundaries (separated by \\n\\n\\n\\n)
2) Merges split text paragraphs with original images/other blocks in original order to form new content, optionally saves as preseg JSON
3) Performs section labeling on new content, writes index(1-based)/section to each content, and attaches section_labels at root

Features:
- Concurrent execution (asyncio + thread pool)
- Resume capability (skips if final output exists or .inprogress exists)
- Exponential backoff retry
- Both splitting and labeling use OpenAI Chat Completions, prompts are configurable

Usage:
    python add_section.py [OPTIONS]

Examples:
    # Basic usage with default settings
    python add_section.py --root-dir /path/to/papers --model gpt-5-2025-08-07

    # With custom input/output names and save preseg
    python add_section.py --root-dir /path/to/papers --model gpt-5-2025-08-07 --input-name paper_parse_origin.json --output-name section_paper_parse.json --save-preseg

    # With custom concurrency and max blocks
    python add_section.py --root-dir /path/to/papers --model gpt-5-2025-08-07 -j 6 --max-blocks 1000

    # Show help message
    python add_section.py --help

Arguments:
    --root-dir          Root directory, recursively search for input JSON files (required)
    --model             LLM model name for text splitting and section labeling (required)
    --input-name        Input JSON filename (default: paper_parse_origin.json)
    --output-name       Final output JSON filename (default: paper_parse_add_section.json)
    --concurrency, -j   Number of concurrent tasks (default: 20)
    --max-blocks        Maximum number of blocks to send to LLM in labeling stage (default: 1200)
    --max-tokens        Maximum tokens for LLM calls (default: 40000)
    --save-preseg       Save intermediate presegmented JSON (disabled by default)
    --preseg-name       Intermediate JSON filename (default: preseg_<output_name>)

Environment Variables:
    OPENAI_API_KEY      OpenAI API key (required)
"""

import os, json, re, sys, time, asyncio
from pathlib import Path
from typing import List, Dict, Any, Optional, Tuple, Set
from dataclasses import dataclass

from tqdm import tqdm
from openai import OpenAI
from detect.utils import extract_json_from_text, load_json, save_json

# ================= Global Configuration =================
LLM_MAX_RETRIES = 3

ALLOWED_SECTIONS = [
    "Abstract", "Introduction", "Motivation", "Preliminaries",
    "Related Work", "Method", "Experiments", "Conclusion",
    "References", "Appendix", "Checklist"
]

# Step A: Text splitting prompt (based on OCR_SYS_PROMPT semantics, adapted for "raw text" scenario)
SEG_SYS_PROMPT = (
    "You will be given raw text paragraphs from a research paper. "
    "Please correct obvious OCR-like issues (e.g., hyphenated line-breaks) and normalize whitespace, "
    "while preserving math/LaTeX. Then split the text into clear blocks based on sections: "
    "blocks should follow section boundaries (e.g., a heading and its following paragraphs belong to one block) "
    "and MUST NOT arbitrarily split text that belongs to the same section. "
    "Each block MUST be separated by four newlines (\\n\\n\\n\\n). "
    "Return ONLY the corrected text with block breaks — no explanations and no JSON."
)

# Step B: Section labeling prompt
LABEL_SYS_PROMPT = (
    "You are an expert in academic document structure analysis. "
    "Given a research paper represented as a list of ordered content blocks (text paragraphs, figures, tables), "
    "assign ONE section label to EVERY block using ONLY the following set: "
    "Abstract, Introduction, Motivation, Preliminaries, Related Work, Method, Experiments, Discussion, References, Appendix. "
    "Use 1-based indices exactly as provided. Group contiguous indices with ranges. "
    "Cover ALL indices exactly once in total (no gaps). "
    "For figures/tables, infer their section based on captions or semantic clues — note that their section may differ from the surrounding text. "
    "Output STRICT JSON, no extra text. The format is:\n"
    "{"
    "  \"labels\": ["
    "    {\"content_index\": \"1-3,5\", \"section\": \"Abstract\"},"
    "    {\"content_index\": \"4,6-9\", \"section\": \"Introduction\"},"
    "    {\"content_index\": \"10-14\", \"section\": \"Method\"},"
    "  ]"
    "}"
    "Tips: merge adjacent blocks sharing the same section into concise ranges; keep sections in document order."
)

# ================= Utility Functions =================
def normalize_to_1_based_indices(blocks: List[Dict[str, Any]]) -> List[int]:
    return list(range(1, len(blocks) + 1))

def format_block_preview(i1: int, b: Dict[str, Any]) -> str:
    btype = b.get("type") or "other"
    if btype == "text":
        snippet = (b.get("text") or "").replace("\n", " ").strip()[:250]
        return f"[{i1}] TEXT: {snippet}"
    elif btype in ("image_url", "image", "figure", "table"):
        url = ""
        if isinstance(b.get("image_url"), dict):
            url = (b["image_url"].get("url") or "")[:120]
        elif isinstance(b.get("image_url"), str):
            url = b["image_url"][:120]
        cap = (b.get("caption") or b.get("text") or "").replace("\n", " ").strip()[:160]
        tail = f" | CAPTION: {cap}" if cap else ""
        return f"[{i1}] FIG/TBL: {url}{tail}"
    else:
        snippet = (b.get("text") or "").replace("\n", " ").strip()[:200]
        return f"[{i1}] OTHER: {snippet}"

def parse_index_expr(expr: str) -> List[int]:
    expr = (expr or "").strip()
    if not expr:
        return []
    out: List[int] = []
    for part in expr.split(","):
        part = part.strip()
        if not part:
            continue
        if "-" in part:
            a, b = part.split("-", 1)
            try:
                a1, b1 = int(a.strip()), int(b.strip())
                if a1 <= b1:
                    out.extend(range(a1, b1 + 1))
                else:
                    out.extend(range(b1, a1 + 1))
            except:
                continue
        else:
            try:
                out.append(int(part))
            except:
                continue
    seen: Set[int] = set()
    uniq = []
    for x in out:
        if x not in seen:
            uniq.append(x)
            seen.add(x)
    return uniq

def build_index_to_section(labels: List[Dict[str, Any]], total: int) -> Dict[int, str]:
    i2s: Dict[int, str] = {}
    for item in labels:
        section = (item.get("section") or "").strip()
        if not section:
            continue
        normalized = None
        for s in ALLOWED_SECTIONS:
            if s.lower() == section.lower():
                normalized = s
                break
        if normalized is None:
            normalized = section
        idx_list = parse_index_expr(item.get("content_index", ""))
        for i in idx_list:
            if 1 <= i <= total:
                i2s[i] = normalized

    last = None
    for i in range(1, total + 1):
        if i in i2s:
            last = i2s[i]
        else:
            if last is not None:
                i2s[i] = last
            else:
                i2s[i] = "Introduction"
    return i2s

# ================= LLM Helper =================
_client = None
_openai_api_key = None

def get_client(openai_api_key: str):
    global _client, _openai_api_key
    if _client is None or _openai_api_key != openai_api_key:
        _client = OpenAI(api_key=openai_api_key)
        _openai_api_key = openai_api_key
    return _client

def call_llm(messages, model: str, max_tokens: int, openai_api_key: str):
    client = get_client(openai_api_key)
    last_exc = None
    for attempt in range(1, LLM_MAX_RETRIES + 1):
        try:
            resp = client.chat.completions.create(
                model=model,
                messages=messages,
                temperature=0.0,
                max_tokens=max_tokens,
            )
            return resp.choices[0].message.content
        except Exception as e:
            last_exc = e
            print(f"[Retry {attempt}] LLM call failed: {e}")
            time.sleep(2 ** attempt)
    print(f"[LLM failure] {last_exc}")
    return None

# ================= Step A: Text Splitting =================
def split_text_block_via_llm(text: str, model: str, max_tokens: int, openai_api_key: str) -> List[str]:
    """
    Send a single text content to LLM, split by paragraph/section boundaries; returns list of paragraphs.
    """
    if not text or not text.strip():
        return []
    messages = [
        {"role": "system", "content": SEG_SYS_PROMPT},
        {"role": "user", "content": text.strip()},
    ]
    raw = call_llm(messages, model=model, max_tokens=max_tokens, openai_api_key=openai_api_key)
    if not raw:
        # Fallback: return as-is if splitting fails (to avoid interrupting workflow)
        return [text.strip()]
    # Use four newlines as boundary (strictly as per prompt)
    parts = [p.strip() for p in raw.split("\n\n\n\n") if p.strip()]
    return parts if parts else [text.strip()]

def build_presegmented_content(
    original_content: List[Dict[str, Any]], 
    model: str, 
    max_tokens: int, 
    openai_api_key: str
) -> List[Dict[str, Any]]:
    """
    Iterate through original content:
    - For blocks with type=='text', call LLM to split and expand into multiple text blocks
    - Other types (image_url/figure/table/other) are inserted as-is (maintain original order)
    Returns new content list
    """
    new_content: List[Dict[str, Any]] = []
    for item in original_content:
        btype = item.get("type") or "other"
        if btype == "text":
            text = item.get("text") or ""
            pieces = split_text_block_via_llm(text, model=model, max_tokens=max_tokens, openai_api_key=openai_api_key)
            for para in pieces:
                new_content.append({"type": "text", "text": para})
        else:
            new_content.append(dict(item))  # Keep images and other blocks
    return new_content

# ================= Step B: Section Labeling =================
def label_one_document_core_from_content(
    content_blocks: List[Dict[str, Any]], 
    model: str, 
    max_tokens: int, 
    max_blocks: int,
    openai_api_key: str
) -> Dict[str, Any]:
    blocks = content_blocks
    if len(blocks) == 0:
        return {
            "content": [],
            "section_labels": {"labels": [], "model_used": model}
        }

    use_blocks = blocks
    if len(use_blocks) > max_blocks:
        print(f"[WARN] too many blocks ({len(use_blocks)}), truncating to {max_blocks}")
        use_blocks = use_blocks[:max_blocks]

    idxs = normalize_to_1_based_indices(use_blocks)

    previews = []
    for i1, b in zip(idxs, use_blocks):
        previews.append(format_block_preview(i1, b))

    allowed_str = ", ".join(ALLOWED_SECTIONS)
    user_prompt = (
        "Below are sequential content blocks from a research paper. "
        "Assign one label from the allowed set to EVERY block. "
        f"Allowed sections: {allowed_str}.\n\n"
        + "\n\n".join(previews)
        + "\n\nReturn STRICT JSON only, with the exact schema shown before. "
          "Use 1-based indices as shown in square brackets []. Merge contiguous indices with ranges."
    )

    messages = [{"role": "system", "content": LABEL_SYS_PROMPT},
                {"role": "user", "content": user_prompt}]

    raw = call_llm(messages, model=model, max_tokens=max_tokens, openai_api_key=openai_api_key)
    if not raw:
        raise RuntimeError("No LLM output for section labeling")

    try:
        obj = extract_json_from_text(raw)
    except Exception:
        Path("debug_raw_llm.txt").write_text(raw, encoding="utf-8")
        raise

    labels = obj.get("labels", [])
    if not isinstance(labels, list):
        raise ValueError("LLM output missing 'labels' list")

    i2s = build_index_to_section(labels, total=len(use_blocks))
    default_section = i2s.get(len(use_blocks), "Introduction")

    new_content = []
    for global_i, b in enumerate(blocks, start=1):
        nb = dict(b)
        nb["index"] = global_i
        if global_i <= len(use_blocks):
            nb["section"] = i2s.get(global_i, "Introduction")
        else:
            nb["section"] = default_section
        new_content.append(nb)

    return {
        "content": new_content,
        "section_labels": {"labels": labels, "model_used": model}
    }

# ================= Batch Processing Wrapper =================
@dataclass
class TaskResult:
    input_path: Path
    output_path: Path
    ok: bool
    err: Optional[str] = None

async def process_one_file(
    input_path: Path,
    output_path: Path,
    save_preseg: bool,
    preseg_name: Optional[str],
    model: str,
    max_tokens: int,
    max_blocks: int,
    openai_api_key: str,
    max_retries: int = 4,
    backoff_base: float = 1.6,
) -> TaskResult:
    """
    Single file processing: text splitting → merge preseg content (optionally save) → section labeling → final output
    Resume capability: skip if final output exists or .inprogress exists
    """
    try:
        if output_path.exists():
            return TaskResult(input_path, output_path, ok=True)

        inprog = output_path.with_suffix(output_path.suffix + ".inprogress")
        if inprog.exists():
            return TaskResult(input_path, output_path, ok=True)

        inprog.parent.mkdir(parents=True, exist_ok=True)
        inprog.write_text(str(time.time()), encoding="utf-8")

        last_err = None
        for attempt in range(1, max_retries + 1):
            try:
                data = load_json(input_path)
                original_content: List[Dict[str, Any]] = data.get("content", [])
                # A. Text splitting (put in thread pool to avoid blocking)
                preseg_content = await asyncio.to_thread(
                    build_presegmented_content, original_content, model, max_tokens, openai_api_key
                )

                # Optionally save preseg JSON (doesn't affect final skip logic)
                if save_preseg:
                    preseg_path = output_path.parent / (
                        (preseg_name or f"preseg_{output_path.name}")
                    )
                    preseg_obj = dict(data)
                    preseg_obj["content"] = preseg_content
                    save_json(preseg_obj, preseg_path)

                # B. Section labeling
                labeled = await asyncio.to_thread(
                    label_one_document_core_from_content, preseg_content, model, max_tokens, max_blocks, openai_api_key
                )

                # Merge into final output (keep original root fields, replace content + attach section_labels)
                out_obj = dict(data)
                out_obj["content"] = labeled["content"]
                out_obj["section_labels"] = labeled["section_labels"]

                save_json(out_obj, output_path)
                try:
                    if inprog.exists():
                        inprog.unlink()
                except Exception:
                    pass
                return TaskResult(input_path, output_path, ok=True)

            except Exception as e:
                last_err = str(e)
                sleep_s = (backoff_base ** (attempt - 1)) + 0.25 * attempt
                print(f"[WARN] Failed on {input_path} (attempt {attempt}/{max_retries}): {e}")
                await asyncio.sleep(sleep_s)

        try:
            if inprog.exists():
                inprog.unlink()
        except Exception:
            pass
        return TaskResult(input_path, output_path, ok=False, err=last_err)

    except Exception as e:
        return TaskResult(input_path, output_path, ok=False, err=str(e))

def find_input_jsons(root_dir: Path, input_name: str) -> List[Path]:
    results: List[Path] = []
    for p in root_dir.rglob(input_name):
        if p.is_file():
            results.append(p)
    return sorted(results)

async def bounded_worker(sem: asyncio.Semaphore, coro):
    async with sem:
        return await coro

async def run_batch(
    root_dir: Path,
    input_name: str,
    output_name: str,
    concurrency: int,
    save_preseg: bool,
    preseg_name: Optional[str],
    model: str,
    max_tokens: int,
    max_blocks: int,
    openai_api_key: str,
) -> Dict[str, Any]:
    inputs = find_input_jsons(root_dir, input_name=input_name)
    if not inputs:
        return {"total": 0, "done": 0, "failed": []}

    sem = asyncio.Semaphore(concurrency)
    tasks = []
    for ip in inputs:
        op = ip.parent / output_name
        tasks.append(bounded_worker(sem, process_one_file(
            input_path=ip,
            output_path=op,
            save_preseg=save_preseg,
            preseg_name=preseg_name,
            model=model,
            max_tokens=max_tokens,
            max_blocks=max_blocks,
            openai_api_key=openai_api_key,
        )))

    results: List[TaskResult] = []
    for fut in tqdm(asyncio.as_completed(tasks), total=len(tasks), desc="Split+Section Labeling"):
        res = await fut
        results.append(res)

    done = sum(1 for r in results if r.ok)
    failed = [f"{r.input_path}: {r.err}" for r in results if not r.ok]
    return {"total": len(results), "done": done, "failed": failed}

# ================= CLI =================
def main():
    import argparse
    parser = argparse.ArgumentParser(
        description="Batch split text blocks and section labeling for paper JSONs.",
        formatter_class=argparse.ArgumentDefaultsHelpFormatter
    )
    parser.add_argument("--root-dir", "--root_dir", type=str, required=True,
                        help="Root directory, recursively search for input_name JSON")
    parser.add_argument("--input-name", "--input_name", type=str, default="paper_parse_origin.json",
                        help="Input JSON filename")
    parser.add_argument("--output-name", "--output_name", type=str, default="paper_parse_add_section.json",
                        help="Final output JSON filename")
    parser.add_argument("--concurrency", "-j", type=int, default=20,
                        help="Number of concurrent tasks")
    parser.add_argument("--model", type=str, required=True,
                        help="LLM model name for text splitting and section labeling")
    parser.add_argument("--max-blocks", "--max_blocks", type=int, default=1200,
                        help="Maximum number of blocks to send to LLM in labeling stage")
    parser.add_argument("--max-tokens", "--max_tokens", type=int, default=40000,
                        help="Maximum tokens for LLM calls")
    parser.add_argument("--save-preseg", "--save_preseg", action="store_true",
                        help="Save intermediate presegmented JSON")
    parser.add_argument("--preseg-name", "--preseg_name", type=str, default=None,
                        help="Intermediate JSON filename (default: preseg_<output_name>)")
    args = parser.parse_args()

    openai_api_key = os.getenv("OPENAI_API_KEY")
    if not openai_api_key:
        print("[FATAL] OPENAI_API_KEY not set.")
        sys.exit(1)

    root = Path(args.root_dir).expanduser().resolve()
    if not root.exists():
        raise FileNotFoundError(f"Root dir not found: {root}")

    print(f"[INFO] root={root}")
    print(f"[INFO] input_name={args.input_name}  output_name={args.output_name}")
    print(f"[INFO] concurrency={args.concurrency}  model={args.model}  max_blocks={args.max_blocks}  max_tokens={args.max_tokens}")
    if args.save_preseg:
        print(f"[INFO] save_preseg=True  preseg_name={args.preseg_name or f'preseg_{args.output_name}'}")

    loop = asyncio.get_event_loop()
    summary = loop.run_until_complete(
        run_batch(
            root_dir=root,
            input_name=args.input_name,
            output_name=args.output_name,
            concurrency=args.concurrency,
            save_preseg=args.save_preseg,
            preseg_name=args.preseg_name,
            model=args.model,
            max_tokens=args.max_tokens,
            max_blocks=args.max_blocks,
            openai_api_key=openai_api_key,
        )
    )
    print(f"[DONE] total={summary['total']}  done={summary['done']}")
    if summary["failed"]:
        print("[FAILED] some tasks failed:")
        for i, err in enumerate(summary["failed"], 1):
            print(f"  {i}. {err}")

if __name__ == "__main__":
    main()
