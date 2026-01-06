#!/usr/bin/env python3
# -*- coding: utf-8 -*-

"""
Synthetic Corruptions Generator for Detector

This script batch synthesizes corrupted papers by applying JSON patches to paper_final.json files.
It generates realistic corruptions across 8 categories for training/evaluating detection models.

Features:
- Concurrent processing with asyncio
- Resume capability (skips existing synthetic files)
- Multimodal support (text + images)
- Configurable corruption types and parameters

Usage:
    python synth_corruptions_for_detector.py [OPTIONS]

Examples:
    # Basic usage with default settings
    python synth_corruptions_for_detector.py --root-dir /path/to/papers --model gpt-5-2025-08-07

    # With custom concurrency and overwrite existing files
    python synth_corruptions_for_detector.py --root-dir /path/to/papers --model gpt-5-2025-08-07 -j 10 --overwrite

    # Regenerate files with low applied count
    python synth_corruptions_for_detector.py --root-dir /path/to/papers --model gpt-5-2025-08-07 --overwrite-apply 10

    # Show help message
    python synth_corruptions_for_detector.py --help

Arguments:
    --root-dir              Root directory containing paper subdirectories (each with paper_final.json) (required)
    --model                  LLM model name for generating corruptions (required)
    --concurrency, -j        Number of concurrent tasks (default: 10)
    --overwrite              Force regenerate and overwrite existing synthetic files
    --overwrite-apply        Regenerate outputs whose applied=True count is less than this value
    --max-doc-chars          Maximum document characters for LLM (default: 32000)
    --max-snippet-chars      Maximum snippet characters for target_find (default: 2000)
    --max-replace-chars      Maximum replacement characters (default: 2000)
    --max-edits              Maximum number of edits per paper (default: 20)
    --min-edits              Minimum number of edits per paper (default: 10)
    --temperature            LLM temperature (default: 0.0)
    --llm-max-retries        Maximum retries for LLM calls (default: 3)

Environment Variables:
    OPENAI_API_KEY           OpenAI API key (required)
"""

from __future__ import annotations
import argparse
import asyncio
import json, os, re, time, copy, uuid, sys
from datetime import datetime
from typing import List, Dict, Any, Tuple, Optional
from pathlib import Path

try:
    from tqdm import tqdm
except Exception:
    tqdm = None

from openai import OpenAI
from openai import (
    APIError, RateLimitError, APITimeoutError, APIConnectionError,
    AuthenticationError, BadRequestError, PermissionDeniedError,
    UnprocessableEntityError
)
from detect.utils import extract_json_from_text, get_openai_client, load_json, save_json

# 8 merged corruption types (with fine-grained descriptions, encouraging model diversity)
CORRUPTION_TYPES = {
    "evidence_data_integrity": (
        "Corrupt or manipulate any form of experimental evidence or data. This includes fabricating, deleting, "
        "or altering table rows, figure samples, performance metrics, dataset statistics, or summary numbers; "
        "cherry-picking only favorable results while hiding failure cases; misreporting variance or statistical "
        "significance; modifying plotting ranges or visual encodings to exaggerate differences; or providing "
        "incomplete or misleading descriptions of dataset construction and preprocessing. These examples are "
        "not exhaustive—the model is encouraged to propose additional realistic evidence- or data-related corruption."
    ),

    "method_logic_consistency": (
        "Introduce flaws, contradictions, or inconsistencies in method descriptions, definitions, theory, or logic. "
        "This includes invalid derivation steps, mistaken formulas, undefined symbols, conflicting notation, mismatched "
        "objective functions across sections, hidden or incorrect assumptions, and discrepancies between the method, "
        "the theory, and the experiments. Any corruption that breaks the logical or conceptual consistency of the "
        "paper is allowed. The model may extend to any other plausible form of methodological or logical corruption."
    ),

    "experimental_design_protocol": (
        "Manipulate experimental setups, baselines, hyperparameters, or evaluation protocols. This includes using "
        "non-comparable baselines, asymmetric data or compute budgets, unreported training tricks, incomplete or "
        "biased ablations, obscuring critical hyperparameters, misreporting compute usage or hardware settings, "
        "or designing experiments that unfairly favor the proposed method. Other realistic experiment-level corruption "
        "is also encouraged."
    ),

    "claim_interpretation_distortion": (
        "Distort the interpretation of results, figures, or evidence. Examples include overstating conclusions, "
        "making universal claims from narrow evidence, misreading or misrepresenting trends in charts, drawing "
        "unsupported causal explanations, or exaggerating robustness, safety, or generalization beyond what the "
        "results justify. Additional forms of claim-level distortion are encouraged."
    ),

    "reference_background_fabrication": (
        "Fabricate or misuse citations, datasets, or factual background. This includes citing non-existent papers, "
        "misattributing key ideas, inventing datasets or tasks, introducing false domain knowledge, or incorrectly "
        "describing prior work to give an impression of novelty or support. The model may introduce any other "
        "plausible reference- or background-related fabrication."
    ),

    "ethical_integrity_omission": (
        "Remove, weaken, or obscure ethical, safety, or transparency information. Examples include omitting IRB or "
        "consent statements, hiding risks or limitations, deleting bias or fairness analyses, removing conflict-of-interest "
        "or funding disclosures, or failing to acknowledge sensitive data issues. The model may propose any similar "
        "omission that compromises ethical integrity."
    ),

    "rhetorical_presentation_manipulation": (
        "Modify the tone, structure, or rhetorical style in ways that artificially strengthen the perceived contribution. "
        "Examples include overly absolute language such as completely solves or eliminates all limitations, promotional "
        "or marketing-style phrasing, inflated adjectives, or rewriting neutral statements into exaggerated claims. "
        "Additional stylistic manipulations beyond these examples are encouraged."
    ),

    "context_misalignment_incoherence": (
        "Introduce inconsistencies or mismatches across the paper’s global narrative. This includes contradictions "
        "between the abstract, introduction, methods, experiments, and conclusions; claims made in one section but "
        "not supported elsewhere; definitions that change across the paper; or missing connections between problem "
        "statement, method, and results. The model is encouraged to generate any other realistic cross-section or "
        "narrative-level corruption."
    )
}


# ================== NEW: Multimodal related (add block by block, don't concatenate full text) ==================
def build_rules_prompt(title: str, min_edits: int, max_edits: int, max_snippet_chars: int, max_replace_chars: int) -> str:
    corruption_keys = list(CORRUPTION_TYPES.keys())
    # Dynamically construct type definitions from CORRUPTION_TYPES to ensure consistency with dict
    type_def_lines = []
    for key, desc in CORRUPTION_TYPES.items():
        type_def_lines.append(f"- {key}:\n    {desc}")
    type_defs_text = "\n\n".join(type_def_lines)

    return f"""
You are a synthetic corruption generator for academic papers.

You will receive the paper as an ordered list of blocks (each block is either a TEXT chunk or an IMAGE URL).
Read blocks in order. Do NOT reconstruct the full paper; propose realistic edits based on the provided blocks.

-------------------------
CORRUPTION TYPE DEFINITIONS
(Choose exactly one per edit; each type includes multiple fine-grained patterns, and the model is encouraged to **invent
additional plausible corruption modes** beyond the examples.)

{type_defs_text}

-------------------------

EDIT COVERAGE REQUIREMENTS:
- Produce {min_edits}–{max_edits} edits.
- Cover at least four sections among: Abstract, Introduction, Related Work, Method, Experiments,
  Discussion, Conclusion, References.
- Keep coherence; edits must feel like genuine manuscript content.

HUMAN-LIKENESS & STYLE:
- Match local scholarly tone/notation/citation style.
- Vary sentence structure; no meta-commentary or disclaimers.

ERROR_EXPLANATION (Reviewer-style, 3–6 sentences):
- Diagnose issues in the modified paper; be specific with anchors (for example, Section 4.2, Table 3, [44]).
- Never mention that text is synthetic.

OUTPUT RULES:
- Output ONE SINGLE JSON ONLY.
- The output MUST be valid standard JSON:
  * Use double quotes "..." for all keys and string values.
  * NEVER use trailing commas.
  * No comments, markdown, or extra text outside the JSON object.
- Top-level keys: "global_explanation" (string) and "edits" (list of {min_edits}–{max_edits}).
- Each edit object EXACTLY has:
  - id (int, 1-based)
  - corruption_type (one of {corruption_keys})
  - difficulty ("easy"|"medium"|"hard")
  - location (for example, "Method", "Abstract", "Experiments")
  - rationale (short reason)
  - error_explanation (reviewer-style)
  - needs_cross_section (boolean):
     * true —
           Mark TRUE **only when detecting the corruption requires comparing content
           across two DIFFERENT SECTIONS** of the paper.
           This means:
             - If Section A states X and Section B states ¬X, and the error is only
               discoverable through cross-section reasoning → TRUE.
             - If the Abstract or Conclusion contradicts Methods or Experiments → TRUE.
           IMPORTANT:
             → TRUE is allowed **ONLY IF** multi-section reasoning is *strictly necessary*.
     * false —
           The corruption can be fully detected **within the same section**, even if:
             - The inconsistent content is far apart in the same section;
             - It spans multiple paragraphs within the section;
             - It appears in different blocks (text, table, figure) of the same section.
           Examples (ALL must be FALSE):
             - A table contradicts another table in the SAME section, even if they are far apart.
             - A figure and a paragraph contradict each other in the SAME section.
             - Two equations within Methods disagree with each other.
           Cross-block or cross-paragraph inconsistencies do NOT count as cross-section.

     KEY RULE:
        → Use TRUE **only when the inconsistency crosses section boundaries**.
        → If the error can be detected without leaving the current section,
          it MUST be FALSE.

  - target_find (<= {max_snippet_chars} chars)
      → "target_find" **MUST be copied verbatim from the original TEXT blocks**, with characters identical
        to the source (no paraphrasing, no normalization, no added/removed whitespace).

  - replacement (<= {max_replace_chars} chars)
      → "replacement" is new content but MUST NOT duplicate large parts of the original text; keep it
        concise and self-contained.


- Prefer selecting target_find from TEXT blocks. If image-driven, reference the nearest textual anchor
  (for example, "Figure 3 caption", "Table 2").

- Do NOT output the full paper or full tables. Keep edits paragraph-level and self-contained.

→ IMPORTANT AGAIN: "target_find" MUST be an exact verbatim substring copied directly from the original TEXT blocks
    (character-for-character identical; no paraphrasing, no reformatting, no normalization, no added/removed whitespace).

Return ONLY JSON. No extra text.
""".strip()


def paper_to_mm_parts_streaming(paper: dict,
                                max_parts: int = 1_000,
                                max_text_chars_per_block: int = 10_000,
                                soft_char_budget: int = 120_000) -> List[Dict[str, Any]]:
    """
    Only two types: text / image_url; add to list sequentially.
    - For text: truncate to max_text_chars_per_block per block
    - For image_url: pass through {"type":"image_url","image_url":{"url":...}}
    - Soft budget: stop when cumulative text characters reach near soft_char_budget
    """
    parts: List[Dict[str, Any]] = []
    total_chars = 0

    for it in (paper.get("content") or []):
        typ = (it.get("type") or "").lower()
        idx = it.get("index")
        sec = it.get("section") or ""
        header = f"[Block #{idx} | {typ or 'unknown'}{(' |Section: ' + sec) if sec else 'None'}]"
        parts.append({"type": "text", "text": header})
        if typ == "text":
            t = (it.get("text") or "")
            if not isinstance(t, str) or not t.strip():
                continue
            if max_text_chars_per_block and len(t) > max_text_chars_per_block:
                t = t[:max_text_chars_per_block]
            parts.append({"type": "text", "text": t})
            total_chars += len(t)

        elif typ == "image_url":
            img = it.get("image_url")
            url = img.get("url") if isinstance(img, dict) else (img if isinstance(img, str) else None)
            if isinstance(url, str) and (url.startswith("http") or url.startswith("data:")):
                parts.append({"type": "image_url", "image_url": {"url": url}})

        # Budget/quantity control
        if soft_char_budget and total_chars > int(soft_char_budget * 1.05):
            break
        if max_parts and len(parts) >= max_parts:
            break

    return parts

def build_mm_user_content_streaming(title: str, paper: dict, min_edits: int, max_edits: int, max_snippet_chars: int, max_replace_chars: int) -> List[Dict[str, Any]]:
    """
    First part: rules prompt; then: add paper content block by block in order (text/image_url)
    """
    parts = [{"type": "text", "text": build_rules_prompt(title, min_edits, max_edits, max_snippet_chars, max_replace_chars)}]
    parts.extend(paper_to_mm_parts_streaming(paper))
    return parts

def call_openai_for_patches_mm(user_content: List[Dict[str, Any]], model: str, api_key: Optional[str], 
                                temperature: float, max_doc_chars: int, llm_max_retries: int) -> Optional[dict]:
    """
    Multimodal Chat Completions: messages[1].content is a list, each element is {"type":"text",...} or {"type":"image_url",...}
    """
    client = get_openai_client(api_key)
    last_exc = None
    for attempt in range(1, llm_max_retries+1):
        try:
            resp = client.chat.completions.create(
                model=model,
                messages=[
                    {"role":"system","content":"You are a strict JSON-only generator that outputs a single JSON object and nothing else."},
                    {"role":"user","content": user_content}
                ],
                temperature=temperature,
                max_tokens=max_doc_chars,
                n=1
            )
            text = resp.choices[0].message.content
            return extract_json_from_text(text)
        except RateLimitError as e:
            last_exc = e; wait = 2 ** attempt
            print(f"[OpenAI RATE LIMIT] attempt {attempt}, waiting {wait}s...", file=sys.stderr)
            time.sleep(wait)
        except (APITimeoutError, APIConnectionError) as e:
            last_exc = e
            print(f"[OpenAI TEMP ERROR] attempt {attempt}: {e}", file=sys.stderr)
            time.sleep(1.2 * attempt)
        except (BadRequestError, AuthenticationError, PermissionDeniedError, UnprocessableEntityError, APIError) as e:
            last_exc = e
            print(f"[OpenAI FATAL] {e}", file=sys.stderr)
            break
        except Exception as e:
            last_exc = e
            print(f"[OpenAI ERROR] attempt {attempt}: {e}", file=sys.stderr)
            time.sleep(0.8 * attempt)
    print(f"[LLM FAILURE] {last_exc}", file=sys.stderr)
    return None

# ================== Patch Structure Validation and Application ==================
def validate_patches(obj: dict, min_edits: int, max_edits: int, max_snippet_chars: int, max_replace_chars: int) -> Tuple[bool, str]:
    if not isinstance(obj, dict):
        return False, "Not a dict"
    if "edits" not in obj or not isinstance(obj["edits"], list):
        return False, "Missing 'edits' list"

    edits = obj["edits"]
    if not (min_edits <= len(edits) <= max_edits):
        return False, f"edits count must be {min_edits}..{max_edits}"

    seen_ids = set()
    MAX_LOCATION_LEN = 100  # Maximum length for location string (section name/short path is sufficient)
    for e in edits:
        if not isinstance(e, dict):
            return False, "edit not dict"

        # New needs_cross_section boolean field
        required_keys = {
            "id", "corruption_type", "difficulty",
            "location", "rationale", "error_explanation",
            "target_find", "replacement", "needs_cross_section"
        }
        if not required_keys.issubset(e.keys()):
            return False, f"edit missing keys, need {required_keys}"

        # corruption_type
        if e["corruption_type"] not in CORRUPTION_TYPES.keys():
            return False, "invalid corruption_type"

        # difficulty
        if e["difficulty"] not in ("easy", "medium", "hard"):
            return False, "invalid difficulty"
        
        # needs_cross_section must be boolean
        if not isinstance(e["needs_cross_section"], bool):
            return False, "needs_cross_section must be boolean"

        # location
        if not isinstance(e["location"], str) or not e["location"].strip():
            return False, "location must be non-empty string"
        if len(e["location"]) > MAX_LOCATION_LEN:
            return False, "location too long"

        # rationale
        if not isinstance(e["rationale"], str) or not e["rationale"].strip():
            return False, "rationale must be non-empty string"

        # error_explanation
        if not isinstance(e["error_explanation"], str) or not e["error_explanation"].strip():
            return False, "error_explanation must be non-empty string"
        if len(e["error_explanation"]) > 2000:
            return False, "error_explanation too long"

        # target_find
        if not isinstance(e["target_find"], str) or not e["target_find"].strip():
            return False, "target_find must be non-empty string"
        if len(e["target_find"]) > max_snippet_chars:
            return False, "target_find too long"

        # replacement
        if not isinstance(e["replacement"], str):
            return False, "replacement must be string"
        if len(e["replacement"]) > max_replace_chars:
            return False, "replacement too long"

        # id uniqueness
        if e["id"] in seen_ids:
            return False, "duplicate id"
        seen_ids.add(e["id"])

    return True, "ok"


def apply_edits_to_paper(paper: dict, edits: List[Dict[str, Any]]) -> Tuple[dict, List[Dict[str, Any]]]:
    modified = copy.deepcopy(paper)
    content  = modified.get("content", [])
    applied_records = []

    for e in edits:
        find     = e["target_find"]
        repl     = e["replacement"]
        applied  = False
        ctx_info = None

        for idx, item in enumerate(content):
            if item.get("type") == "text":
                text = item.get("text", "")
                pos  = text.find(find)
                if pos != -1:
                    new_text = text[:pos] + repl + text[pos+len(find):]
                    modified["content"][idx]["text"] = new_text
                    applied = True
                    ctx_info = {
                        "content_index": idx,
                        "offset": pos,
                        "before": text[max(0, pos-120):pos],
                        "after": text[pos+len(find):pos+len(find)+120]
                    }
                    break

        applied_records.append({
            "id": e["id"],
            "applied": applied,
            "corruption_type": e["corruption_type"],
            "difficulty": e["difficulty"],
            "location": e.get("location", ""),
            "needs_cross_section": e.get("needs_cross_section", None),
            "rationale": e["rationale"],
            "error_explanation": e.get("error_explanation", ""),
            "context": ctx_info,
        })

    return modified, applied_records


# ============== NEW: Consistent with batch_parse_and_review "skip if exists + .inprogress" ==============
def _model_tag(model: str) -> str:
    return model.replace("/", "_")

def _out_json_path(paper_json_path: Path, model: str) -> Path:
    return paper_json_path.parent / f"paper_synth_{_model_tag(model)}.json"

def _inprogress_path(paper_json_path: Path, model: str) -> Path:
    return paper_json_path.parent / f"paper_synth_{_model_tag(model)}.json.inprogress"

def _already_done(paper_json_path: Path, model: str) -> bool:
    outp = _out_json_path(paper_json_path, model)
    if not outp.exists():
        return False
    try:
        obj = load_json(str(outp))
        return bool(obj.get("synthetic_for_detector") is True and "paper" in obj)
    except Exception:
        return False

def _get_applied_true_count(paper_json_path: Path, model: str) -> Optional[int]:
    """
    If corresponding synth file doesn't exist, return None;
    If exists, return count of applied=True in audit_log.apply_results.
    """
    outp = _out_json_path(paper_json_path, model)
    if not outp.exists():
        return None
    try:
        obj = load_json(str(outp))
        results = obj.get("audit_log", {}).get("apply_results", [])
        return sum(1 for r in results if r.get("applied") is True)
    except Exception:
        return None

# ================== Single Paper Processing (sync function, changed to block-by-block multimodal) ==================
def process_one_paper_sync(paper_json_path: Path, model: str, api_key: Optional[str], overwrite: bool,
                          min_edits: int, max_edits: int, max_snippet_chars: int, max_replace_chars: int,
                          max_doc_chars: int, temperature: float, llm_max_retries: int) -> Tuple[Path, bool, str]:
    """
    Synchronous processing of single paper (with resume capability):
    - If target paper_synth_{model}.json exists and overwrite not specified => skip directly (resume)
    - Create .inprogress before processing, try to delete on completion/exception
    """
    parent = paper_json_path.parent
    out_json = _out_json_path(paper_json_path, model)
    inprog  = _inprogress_path(paper_json_path, model)

    # ===== Resume: skip if done (only when overwrite not enabled) =====
    # if (not overwrite) and _already_done(paper_json_path, model):
    #     return parent, True, "skip_done"

    # Write inprogress (consistent with batch_parse_and_review approach)
    try:
        inprog.write_text(str(time.time()), encoding="utf-8")
    except Exception:
        pass

    try:
        # Read paper_final.json
        try:
            paper = load_json(str(paper_json_path))
        except Exception as e:
            return parent, False, f"load_json failed: {e}"

        # Extract title
        paper_title = ""
        if isinstance(paper, dict):
            if "metadata" in paper and isinstance(paper["metadata"], dict):
                paper_title = paper["metadata"].get("title", "")
            if not paper_title and "content" in paper and paper["content"]:
                first_text = ""
                if isinstance(paper["content"][0], dict) and paper["content"][0].get("type") == "text":
                    first_text = paper["content"][0].get("text", "")
                paper_title = (first_text.splitlines()[0].strip()[:200]) if first_text else ""
        if not paper_title:
            paper_title = "Unknown Title"

        # NEW —— Block-by-block multimodal: no longer concatenate full text, directly build multimodal content list
        user_content = build_mm_user_content_streaming(paper_title, paper, min_edits, max_edits, max_snippet_chars, max_replace_chars)

        # Call LLM (multimodal)
        llm_obj = call_openai_for_patches_mm(user_content, model=model, api_key=api_key, 
                                             temperature=temperature, max_doc_chars=max_doc_chars, llm_max_retries=llm_max_retries)
        if llm_obj is None:
            # Fallback: if model routing doesn't support multimodal, fall back to "rules-only text" (still requires JSON output)
            print("[WARN] MM call failed; falling back to text-only rules.", file=sys.stderr)
            rules_only = [{"type":"text","text": build_rules_prompt(paper_title, min_edits, max_edits, max_snippet_chars, max_replace_chars)}]
            llm_obj = call_openai_for_patches_mm(rules_only, model=model, api_key=api_key, 
                                                temperature=temperature, max_doc_chars=max_doc_chars, llm_max_retries=llm_max_retries)

        # if llm_obj is None:
        #     debug_path = parent / "llm_resp_debug.json"
        #     save_json({"error": "llm_none"}, str(debug_path))
        #     return parent, False, "LLM returned None"

        ok, msg = validate_patches(llm_obj, min_edits, max_edits, max_snippet_chars, max_replace_chars)
        if not ok:
            debug_path = parent / "llm_resp_debug.json"
            save_json({"raw_llm_obj": llm_obj, "validate_msg": msg}, str(debug_path))
            return parent, False, f"validate failed: {msg}"

        # Apply corruptions
        edits = llm_obj["edits"]
        modified_paper, applied_records = apply_edits_to_paper(paper, edits)

        # Audit log
        audit_log = {
            "source_file": str(paper_json_path),
            "paper_title": paper_title,
            "created_at": datetime.utcnow().isoformat() + "Z",
            "global_explanation": llm_obj.get("global_explanation", ""),
            "edits": edits,
            "apply_results": applied_records
        }

        # Write final result (consistent with previous approach)
        out_obj = {
            "synthetic_for_detector": True,  # Explicitly labeled as "synthetic sample for detection/training"
            "audit_log": audit_log,
            "paper": modified_paper
        }
        save_json(out_obj, str(out_json))
        print(f"[DONE] Saved: {out_json}")

        return parent, True, "ok"

    finally:
        # Clean up inprogress
        try:
            if inprog.exists():
                inprog.unlink()
        except Exception:
            pass


# ================== Scan root_dir (first-level subdirectories) ==================
def find_paper_jsons(root_dir: Path) -> List[Path]:
    """
    Find paper_final.json in first-level subdirectories of root_dir.
    Format: root_dir/<paper_folder>/paper_final.json
    """
    out = []
    for sub in sorted(root_dir.iterdir()):
        if not sub.is_dir():
            continue
        cand = sub / "paper_final.json"
        if cand.exists():
            out.append(cand)
    return out

# ================== Concurrent Execution (async + thread pool) ==================
async def bounded_worker(sem: asyncio.Semaphore, loop, paper_json_path: Path, model: str, api_key: Optional[str], overwrite: bool,
                         min_edits: int, max_edits: int, max_snippet_chars: int, max_replace_chars: int,
                         max_doc_chars: int, temperature: float, llm_max_retries: int):
    async with sem:
        # Put synchronous processing into default thread pool
        return await loop.run_in_executor(None, process_one_paper_sync, paper_json_path, model, api_key, overwrite,
                                         min_edits, max_edits, max_snippet_chars, max_replace_chars,
                                         max_doc_chars, temperature, llm_max_retries)

async def run_batch(root_dir: Path, concurrency: int, model: str,
                    api_key: Optional[str], overwrite: bool,
                    overwrite_apply: Optional[int],
                    min_edits: int, max_edits: int, max_snippet_chars: int, max_replace_chars: int,
                    max_doc_chars: int, temperature: float, llm_max_retries: int) -> Dict[str, Any]:
    paper_paths = find_paper_jsons(root_dir)
    if not paper_paths:
        return {"total": 0, "ok": 0, "fail": [], "skipped_done": 0}

    # Consistent with "previous script" style: skip completed items directly from planned queue (but allow overwrite/overwrite_apply to control rerun)
    planned: List[Path] = []
    skipped_done = 0  # Count skipped items (optional)

    for p in paper_paths:

        # 1) overwrite=True: force rerun
        if overwrite:
            planned.append(p)
            continue

        # 2) If overwrite_apply is set: decide whether to rerun based on applied=True count
        if overwrite_apply is not None:
            cnt = _get_applied_true_count(p, model)
            if cnt is None:
                # No synth file → must run
                planned.append(p)
                continue
            if cnt < overwrite_apply:
                # Too few applied → need rerun
                planned.append(p)
                continue
            else:
                # Coverage sufficient → skip
                skipped_done += 1
                continue

        # 3) Original logic: skip if done
        if _already_done(p, model):
            skipped_done += 1
            continue

        planned.append(p)

    if not planned:
        return {
            "total": 0,
            "ok": skipped_done,
            "fail": [],
            "skipped_done": skipped_done,
        }

    sem  = asyncio.Semaphore(concurrency)
    loop = asyncio.get_running_loop()
    tasks = [bounded_worker(sem, loop, p, model, api_key, overwrite,
                           min_edits, max_edits, max_snippet_chars, max_replace_chars,
                           max_doc_chars, temperature, llm_max_retries) for p in planned]

    results = []
    iterator = asyncio.as_completed(tasks)
    if tqdm is not None:
        iterator = tqdm(iterator, total=len(tasks), desc="Synthesizing")
    for coro in iterator:
        res = await coro
        results.append(res)

    ok_cnt = sum(1 for (_, ok, msg) in results if ok)
    fail   = [(str(p), msg) for (p, ok, msg) in results if not ok and msg != "skip_done"]
    return {
        "total": len(results),
        "ok": ok_cnt + skipped_done,   # Include pre-skipped items in "processed" count
        "fail": fail,
        "skipped_done": skipped_done,
    }

def compute_applied_distribution(root_dir: Path, model: str) -> None:
    """
    Scan all paper_final.json under root_dir, count distribution of applied=True for existing synth files.
    Only for printing statistics, doesn't affect workflow.
    """
    paper_paths = find_paper_jsons(root_dir)
    hist: Dict[int, int] = {}
    total = 0
    total_papers_with_synth = 0
    min_cnt: Optional[int] = None
    max_cnt: Optional[int] = None

    for p in paper_paths:
        cnt = _get_applied_true_count(p, model)
        if cnt is None:
            continue
        total_papers_with_synth += 1
        total += cnt
        hist[cnt] = hist.get(cnt, 0) + 1
        if min_cnt is None or cnt < min_cnt:
            min_cnt = cnt
        if max_cnt is None or cnt > max_cnt:
            max_cnt = cnt

    if total_papers_with_synth == 0:
        print("[APPLIED DIST] No existing synth files found for distribution analysis.")
        return

    mean_cnt = total / total_papers_with_synth

    print("\n[APPLIED DIST] applied=True count distribution across existing synth files:")
    print(f"  total_files_with_synth = {total_papers_with_synth}")
    print(f"  min_applied_true       = {min_cnt}")
    print(f"  max_applied_true       = {max_cnt}")
    print(f"  mean_applied_true      = {mean_cnt:.2f}")
    print("  histogram (applied_true -> num_files):")
    for k in sorted(hist.keys()):
        print(f"    {k:3d} -> {hist[k]} files")
    print()

# ================== CLI ==================
def main():
    parser = argparse.ArgumentParser(
        description="Batch synthesize corrupted papers (JSON patches applied) under folders that contain paper_final.json",
        formatter_class=argparse.ArgumentDefaultsHelpFormatter
    )
    parser.add_argument(
        "--root-dir", "--root_dir",
        type=str,
        required=True,
        help="Root directory, e.g., downloads/ICML_2025_oral_test (subdirectories contain paper_final.json)",
    )
    parser.add_argument(
        "--concurrency", "-j",
        type=int,
        default=10,
        help="Number of concurrent tasks",
    )
    parser.add_argument(
        "--model",
        type=str,
        required=True,
        help="LLM model name for generating corruptions",
    )
    parser.add_argument(
        "--overwrite",
        action="store_true",
        help="Force regenerate and overwrite existing synthetic files even if they already exist.",
    )
    parser.add_argument(
        "--overwrite-apply", "--overwrite_apply",
        type=int,
        default=None,
        help="If set, re-generate outputs whose applied=True count is less than this value."
    )
    parser.add_argument(
        "--max-doc-chars", "--max_doc_chars",
        type=int,
        default=32000,
        help="Maximum document characters for LLM",
    )
    parser.add_argument(
        "--max-snippet-chars", "--max_snippet_chars",
        type=int,
        default=2000,
        help="Maximum snippet characters for target_find",
    )
    parser.add_argument(
        "--max-replace-chars", "--max_replace_chars",
        type=int,
        default=2000,
        help="Maximum replacement characters",
    )
    parser.add_argument(
        "--max-edits", "--max_edits",
        type=int,
        default=20,
        help="Maximum number of edits per paper",
    )
    parser.add_argument(
        "--min-edits", "--min_edits",
        type=int,
        default=10,
        help="Minimum number of edits per paper",
    )
    parser.add_argument(
        "--temperature",
        type=float,
        default=0.0,
        help="LLM temperature",
    )
    parser.add_argument(
        "--llm-max-retries", "--llm_max_retries",
        type=int,
        default=3,
        help="Maximum retries for LLM calls",
    )
    args = parser.parse_args()

    openai_api_key = os.getenv("OPENAI_API_KEY")
    if not openai_api_key:
        print("[FATAL] OPENAI_API_KEY is not set.", file=sys.stderr)
        sys.exit(1)

    root = Path(args.root_dir).expanduser().resolve()
    if not root.exists():
        print(f"[FATAL] root_dir not found: {root}", file=sys.stderr)
        sys.exit(1)

    print(f"[INFO] root={root}  concurrency={args.concurrency}  model={args.model}  overwrite={args.overwrite}  overwrite_apply={args.overwrite_apply}")

    loop = asyncio.get_event_loop()
    summary = loop.run_until_complete(
        run_batch(
            root_dir=root,
            concurrency=args.concurrency,
            model=args.model,
            api_key=openai_api_key,
            overwrite=args.overwrite,
            overwrite_apply=args.overwrite_apply,
            min_edits=args.min_edits,
            max_edits=args.max_edits,
            max_snippet_chars=args.max_snippet_chars,
            max_replace_chars=args.max_replace_chars,
            max_doc_chars=args.max_doc_chars,
            temperature=args.temperature,
            llm_max_retries=args.llm_max_retries,
        )
    )

    print(f"[DONE] total={summary['total']}  ok={summary['ok']}  fail={len(summary['fail'])}  skipped_done={summary.get('skipped_done', 0)}")
    if summary["fail"]:
        print("[FAILED ITEMS]")
        for i, (path, msg) in enumerate(summary["fail"], 1):
            print(f"  {i}. {path} :: {msg}")

    # If --overwrite_apply is set, also compute distribution of applied counts under current root_dir
    if args.overwrite_apply is not None:
        compute_applied_distribution(root, args.model)

if __name__ == "__main__":
    main()
