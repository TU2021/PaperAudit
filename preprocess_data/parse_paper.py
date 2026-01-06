#!/usr/bin/env python3
# -*- coding: utf-8 -*-

"""
Batch Paper Parser

This script batch parses paper folders:
- Takes a root directory (e.g., downloads/ICLR_2025_oral)
- Automatically finds paper.pdf in subdirectories
- Calls async parsing pipeline to generate paper_parse_origin.json in each subdirectory
- Parallel execution (configurable concurrency) with resume capability (skips existing paper_parse_origin.json)
- Exponential backoff retry with detailed logging

Usage:
    python parse_paper.py [OPTIONS]

Examples:
    # Parse papers with default settings
    python parse_paper.py --root-dir /path/to/papers --model gpt-5-2025-08-07

    # Parse with custom concurrency
    python parse_paper.py --root-dir /path/to/papers --model gpt-5-2025-08-07 --concurrency 10

    # Show help message
    python parse_paper.py --help

Arguments:
    --root-dir          Root directory containing paper subdirectories (each with paper.pdf)
    --model             Model name for OCR correction and image classification (required)
    --concurrency, -j   Number of concurrent parsing tasks (default: 10)

Environment Variables:
    LLAMA_API_KEY       LlamaParse API key (required)
    OPENAI_API_KEY      OpenAI API key (required)
"""

import argparse
import asyncio
import base64
import json
import mimetypes
import os
import shutil
import time
from dataclasses import dataclass
from pathlib import Path
from typing import Any, Dict, List, Tuple, Optional

from tqdm import tqdm
import litellm
from openai import OpenAI
from litellm import acompletion
from PIL import Image

try:
    import cv2  # Optional: used for template matching fallback when no bbox
except Exception:
    cv2 = None

# ========== Global Settings ==========
# litellm._turn_on_debug()
OCR_SYS_PROMPT = (
    "You will be given an OCR output and associated image crops. "
    "Review and regenerate the OCR text with better accuracy. "
    "Return only the corrected text."
)

CLS_SYS_PROMPT = (
    "You will be given a full-page screenshot and a cropped image. "
    "Decide whether the cropped image is a labeled figure belonging to "
    "the manuscript (e.g., Figure 1, Table 2) or an extraneous artifact "
    "(e.g., banner, logo, advertisement). "
    "If it is a manuscript figure, return [TRUE]; otherwise return [FALSE]. "
    "Respond with exactly [TRUE] or [FALSE]."
)

# ========== Utility Functions (compatible with single script) ==========

def _area_of_bbox(b):
    x0, y0, x1, y1 = b
    return max(0, x1 - x0) * max(0, y1 - y0)

def _bbox_contains(b_big, b_small, tol: int = 2, over_ratio: float = 0.10) -> bool:
    """
    Check if b_small is contained in b_big; allows pixel tolerance,
    and considers contained if overflow ratio in all four directions <= over_ratio
    """
    x0, y0, x1, y1 = b_big
    x2, y2, x3, y3 = b_small
    fully_inside = (x0 - tol <= x2) and (y0 - tol <= y2) and (x1 + tol >= x3) and (y1 + tol >= y3)
    if fully_inside:
        return True
    bw_big = max(1e-6, x1 - x0)
    bh_big = max(1e-6, y1 - y0)
    left_over   = max(0, x0 - x2) / bw_big
    top_over    = max(0, y0 - y2) / bh_big
    right_over  = max(0, x3 - x1) / bw_big
    bottom_over = max(0, y3 - y1) / bh_big
    return all(r <= over_ratio for r in (left_over, top_over, right_over, bottom_over))

def _img_size(path: Path) -> Tuple[int, int]:
    with Image.open(path) as im:
        return im.size  # (w, h)

def _is_subimage_via_template(small_path: Path, large_path: Path,
                              scales=(0.75, 0.9, 1.0, 1.1, 1.25),
                              thr: float = 0.92) -> bool:
    if cv2 is None:
        return False
    lg = cv2.imread(str(large_path))
    sm = cv2.imread(str(small_path))
    if lg is None or sm is None:
        return False
    lg_g = cv2.cvtColor(lg, cv2.COLOR_BGR2GRAY)
    sm_g0 = cv2.cvtColor(sm, cv2.COLOR_BGR2GRAY)
    for s in scales:
        nw = max(1, int(sm_g0.shape[1] * s))
        nh = max(1, int(sm_g0.shape[0] * s))
        sm_g = cv2.resize(sm_g0, (nw, nh), interpolation=cv2.INTER_AREA)
        if sm_g.shape[0] >= lg_g.shape[0] or sm_g.shape[1] >= lg_g.shape[1]:
            continue
        res = cv2.matchTemplate(lg_g, sm_g, cv2.TM_CCOEFF_NORMED)
        _, max_val, _, _ = cv2.minMaxLoc(res)
        if max_val >= thr:
            return True
    return False

def _construct_bbox_from_xywh(obj) -> Optional[Tuple[float, float, float, float]]:
    try:
        x = float(getattr(obj, "x"))
        y = float(getattr(obj, "y"))
        w = float(getattr(obj, "width"))
        h = float(getattr(obj, "height"))
        if w <= 0 or h <= 0:
            return None
        return (x, y, x + w, y + h)
    except Exception:
        return None

def filter_subimages(
    items: List[Dict[str, Any]],
    keep_full_page: bool = False,
    bbox_key: str = "bbox",
    type_key: str = "type",
    path_key: str = "path",
    page_key: str = "page_index",
    min_area_ratio: float = 1.0,
) -> Tuple[List[Dict[str, Any]], List[Dict[str, Any]]]:
    """
    Deduplicate subimages by containment: keep by area from large to small;
    drop those that are contained and significantly smaller
    """
    enriched = []
    for it in items:
        p = Path(it[path_key])
        try:
            w, h = _img_size(p)
        except Exception:
            w, h = (1, 1)
        bbox = it.get(bbox_key)
        area_guess = _area_of_bbox(bbox) if bbox else (w * h)
        enriched.append({**it, "_w": w, "_h": h, "_area": area_guess})

    enriched.sort(key=lambda d: d["_area"], reverse=True)

    kept, dropped = [], []
    for cand in enriched:
        if keep_full_page:  # keep full_page_screenshot
            if cand.get(type_key) == "full_page_screenshot":
                kept.append(cand)
                continue
        else:
            if cand.get(type_key) == "full_page_screenshot":
                dropped.append(cand)
                continue
        is_sub = False
        for big in kept:
            if (page_key in cand) and (page_key in big) and (cand[page_key] != big[page_key]):
                continue
            if big["_area"] <= 0:
                continue
            area_ratio = cand["_area"] / float(big["_area"])
            if area_ratio >= min_area_ratio:
                continue
            if cand.get(bbox_key) and big.get(bbox_key):
                if _bbox_contains(big[bbox_key], cand[bbox_key]):
                    is_sub = True
            else:
                if _is_subimage_via_template(Path(cand[path_key]), Path(big[path_key])):
                    is_sub = True
            if is_sub:
                break
        (dropped if is_sub else kept).append(cand)

    for d in kept + dropped:
        for k in ["_w", "_h", "_area"]:
            d.pop(k, None)
    return kept, dropped

def encode_data_uri(path: Path) -> str:
    raw = path.read_bytes()
    b64 = base64.b64encode(raw).decode()
    mime = mimetypes.guess_type(path.name)[0] or "application/octet-stream"
    return f"data:{mime};base64,{b64}"

# ========== Parsing Core (extracted from single script as functions) ==========

@dataclass
class ParseResult:
    ok: bool
    error: Optional[str] = None
    out_path: Optional[Path] = None

async def parse_one_pdf(
    pdf_path: Path,
    out_json_path: Path,
    llama_api_key: str,
    openai_api_key: str,
    model: str,
    max_retries: int = 4,
    backoff_base: float = 1.5,
) -> ParseResult:
    """
    Parse a single PDF and write to out_json_path (paper_parse_origin.json).
    If out_json_path already exists, return ok=True directly (resume capability).
    """
    if out_json_path.exists():
        return ParseResult(ok=True, out_path=out_json_path)

    # Write .inprogress marker to avoid duplicates
    inprog = out_json_path.with_suffix(out_json_path.suffix + ".inprogress")
    try:
        inprog.write_text(str(time.time()), encoding="utf-8")
    except Exception:
        pass

    # Lazy import to avoid errors when environment not set
    from llama_cloud_services import LlamaParse

    # Set environment variables (litellm/openai read from environment variables)
    os.environ["LLAMA_API_KEY"] = llama_api_key
    os.environ["OPENAI_API_KEY"] = openai_api_key

    # Parsing retry
    last_err = None
    for attempt in range(1, max_retries + 1):
        try:
            parser = LlamaParse(
                api_key=llama_api_key,
                verbose=True,
                language="en",
                extract_layout=True,
                parse_mode="parse_page_with_agent",
            )
            result = await parser.aparse(str(pdf_path))
            img_dir = pdf_path.parent / "images"
            img_dir.mkdir(exist_ok=True)
            all_img_paths = await result.asave_all_images(img_dir)

            content: List[Dict[str, Any]] = []
            used_imgs: set[str] = set()

            # Iterate pages
            for page in result.pages:
                # 1) OCR correction
                review_payload = [{"type": "text", "text": page.md}]
                for img in page.images:
                    if img.type in ("layout_text", "layout_formula"):
                        p = img_dir / img.name
                        if not p.exists():
                            continue
                        data_uri = encode_data_uri(p)
                        review_payload.append({"type": "image_url", "image_url": {"url": data_uri}})
                llm_query = [
                    {"role": "system", "content": OCR_SYS_PROMPT},
                    {"role": "user", "content": review_payload},
                ]
                resp = await acompletion(model=model, messages=llm_query)
                corrected = (resp.choices[0].message.content or "").strip()
                content.append({"type": "text", "text": corrected})

                # 2) Only embed true manuscript images (layout_picture)
                full_page = next((i for i in page.images if i.type == "full_page_screenshot"), None)

                # Subimage deduplication allowlist
                __items = []
                for __img in page.images:
                    if __img.type == "layout_picture":
                        __p = img_dir / __img.name
                        __bbox = _construct_bbox_from_xywh(__img)
                        __items.append({
                            "type": "layout_picture",
                            "path": str(__p),
                            "bbox": __bbox,
                            "page_index": getattr(page, "page", None),
                            "name": __img.name,
                        })
                if full_page:
                    __fp_path = img_dir / full_page.name
                    __items.append({
                        "type": "full_page_screenshot",
                        "path": str(__fp_path),
                        "bbox": _construct_bbox_from_xywh(full_page),
                        "page_index": getattr(page, "page", None),
                        "name": full_page.name,
                    })
                __kept, __dropped = filter_subimages(__items, keep_full_page=False)
                __allow_names = {it["name"] for it in __kept if it["type"] == "layout_picture"}

                if full_page:
                    fp_path = img_dir / full_page.name
                    if fp_path.exists():
                        fp_uri = encode_data_uri(fp_path)
                        for img in page.images:
                            if img.type != "layout_picture":
                                continue
                            if img.name not in __allow_names:
                                continue
                            pic_path = img_dir / img.name
                            if not pic_path.exists():
                                continue
                            pic_uri = encode_data_uri(pic_path)

                            messages = [
                                {"role": "system", "content": CLS_SYS_PROMPT},
                                {"role": "user", "content": [
                                    {"type": "image_url", "image_url": {"url": fp_uri}},
                                    {"type": "image_url", "image_url": {"url": pic_uri}},
                                ]},
                            ]
                            cls_resp = await acompletion(model=model, messages=messages)
                            decision = (cls_resp.choices[0].message.content or "").strip()
                            if "TRUE" in decision.upper():
                                content.append({"type": "image_url", "image_url": {"url": pic_uri}})
                                used_imgs.add(img.name)

            # 3) Clean up unused images
            for path in all_img_paths or []:
                try:
                    if Path(path).name not in used_imgs:
                        os.remove(path)
                except OSError:
                    pass

            # 4) Write final JSON
            out = {
                "pdf": pdf_path.name,
                "content": content,
                "images": [f"images/{n}" for n in sorted(used_imgs)],
            }
            out_json_path.write_text(json.dumps(out, ensure_ascii=False, indent=2), encoding="utf-8")
            try:
                if inprog.exists():
                    inprog.unlink()
            except Exception:
                pass
            return ParseResult(ok=True, out_path=out_json_path)

        except Exception as e:
            last_err = str(e)
            # Exponential backoff
            sleep_s = (backoff_base ** (attempt - 1)) + 0.25 * attempt
            print(f"[WARN] parse failed for {pdf_path} (attempt {attempt}/{max_retries}): {e}")
            await asyncio.sleep(sleep_s)

    try:
        if inprog.exists():
            inprog.unlink()
    except Exception:
        pass
    return ParseResult(ok=False, error=last_err)

# ========== Directory Scanning and Parallel Scheduling ==========

def find_paper_pdfs(root_dir: Path) -> List[Path]:
    """
    Find paper.pdf in first-level subdirectories of root_dir.
    Format: root_dir/<paper_folder>/paper.pdf
    """
    pdfs = []
    for sub in sorted(root_dir.iterdir()):
        if not sub.is_dir():
            continue
        pdf = sub / "paper.pdf"
        if pdf.exists():
            pdfs.append(pdf)
    return pdfs

async def bounded_worker(
    sem: asyncio.Semaphore,
    task_coro,
):
    async with sem:
        return await task_coro

async def run_batch(
    root_dir: Path,
    concurrency: int,
    llama_api_key: str,
    openai_api_key: str,
    model: str,
) -> Dict[str, Any]:
    pdf_paths = find_paper_pdfs(root_dir)
    if not pdf_paths:
        return {"total": 0, "done": 0, "failed": []}

    sem = asyncio.Semaphore(concurrency)
    tasks = []
    for pdf in pdf_paths:
        out_json = pdf.parent / "paper_parse_origin.json"
        tasks.append(
            bounded_worker(
                sem,
                parse_one_pdf(
                    pdf_path=pdf,
                    out_json_path=out_json,
                    llama_api_key=llama_api_key,
                    openai_api_key=openai_api_key,
                    model=model,
                )
            )
        )

    results: List[ParseResult] = []
    # Progress bar wrapper for gather
    for f in tqdm(asyncio.as_completed(tasks), total=len(tasks), desc="Parsing"):
        res = await f
        results.append(res)

    done = sum(1 for r in results if r.ok)
    failed = [r.error for r in results if not r.ok]
    return {"total": len(results), "done": done, "failed": failed}

# ========== CLI ==========

def main():
    parser = argparse.ArgumentParser(
        description="Batch parse folders that contain paper.pdf and write paper_parse_origin.json",
        formatter_class=argparse.ArgumentDefaultsHelpFormatter
    )
    parser.add_argument(
        "--root-dir",
        type=str,
        required=True,
        help="Root directory, e.g., downloads/ICLR_2025_oral (subdirectories contain paper.pdf)",
    )
    parser.add_argument(
        "--concurrency", "-j",
        type=int,
        default=10,
        help="Number of concurrent parsing tasks",
    )
    parser.add_argument(
        "--model",
        type=str,
        required=True,
        help="Model name for OCR correction and image classification",
    )
    args = parser.parse_args()

    # Environment variable validation
    for name in ("LLAMA_API_KEY", "OPENAI_API_KEY"):
        if not os.getenv(name):
            raise EnvironmentError(f"{name} environment variable is required")
    llama_api_key = os.getenv("LLAMA_API_KEY")
    openai_api_key = os.getenv("OPENAI_API_KEY")

    root = Path(args.root_dir).expanduser().resolve()
    if not root.exists():
        raise FileNotFoundError(f"Root dir not found: {root}")

    print(f"[INFO] root={root}  concurrency={args.concurrency}  model={args.model}")

    loop = asyncio.get_event_loop()
    summary = loop.run_until_complete(
        run_batch(
            root_dir=root,
            concurrency=args.concurrency,
            llama_api_key=llama_api_key,
            openai_api_key=openai_api_key,
            model=args.model,
        )
    )
    print(f"[DONE] total={summary['total']}  done={summary['done']}")
    if summary["failed"]:
        print("[FAILED] some tasks failed:")
        for i, err in enumerate(summary["failed"], 1):
            print(f"  {i}. {err}")

if __name__ == "__main__":
    main()
