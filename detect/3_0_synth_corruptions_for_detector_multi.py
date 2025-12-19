#!/usr/bin/env python3
# -*- coding: utf-8 -*-

from __future__ import annotations
import argparse
import asyncio
import json, os, re, time, copy, uuid, sys
from datetime import datetime
from typing import List, Dict, Any, Tuple, Optional
from pathlib import Path
from threading import Lock

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

# ================= 默认配置（与单文档版保持一致） =================
DEFAULT_MODEL = "gpt-5-2025-08-07"   # 你的模型名
OPENAI_API_KEY       = os.getenv("OPENAI_API_KEY", None)
LLM_MAX_RETRIES      = 3

MAX_DOC_CHARS        = 32_000
MAX_SNIPPET_CHARS    = 2_000
MAX_REPLACE_CHARS    = 2_000
MAX_EDITS            = 20
MIN_EDITS            = 10

TEMPERATURE          = 0.0

# 8 类合并后的篡改类型（带更细粒度说明，鼓励模型发散）
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


# ================== 基础工具函数（未改动核心逻辑） ==================
def load_json(path: str) -> dict:
    with open(path, "r", encoding="utf-8") as f:
        return json.load(f)

def save_json(obj, path: str):
    Path(path).parent.mkdir(parents=True, exist_ok=True)
    with open(path, "w", encoding="utf-8") as f:
        json.dump(obj, f, indent=2, ensure_ascii=False)

def extract_json_from_text(text: str):
    text = text.strip()
    try:
        return json.loads(text)
    except Exception:
        pass
    start = text.find("{")
    end   = text.rfind("}")
    if start != -1 and end != -1 and end > start:
        candidate = text[start:end+1]
        for _ in range(5):
            try:
                return json.loads(candidate)
            except Exception:
                end = text.rfind("}", 0, end-1)
                if end <= start:
                    break
                candidate = text[start:end+1]
    m = re.search(r"\{(?:.|\n)*?\}", text)
    if m:
        return json.loads(m.group(0))
    raise ValueError("Failed to extract JSON from LLM output.")

# ================== OpenAI 客户端（线程安全懒加载） ==================
_client_singleton: Optional[OpenAI] = None
_client_lock = Lock()

def get_openai_client(api_key: Optional[str]) -> OpenAI:
    global _client_singleton
    with _client_lock:
        if _client_singleton is None:
            if not api_key:
                raise RuntimeError("OPENAI_API_KEY not set.")
            _client_singleton = OpenAI(api_key=api_key)
        return _client_singleton

# ================== NEW: 多模态相关（逐块加入，不拼全文） ==================
def build_rules_prompt(title: str) -> str:
    corruption_keys = list(CORRUPTION_TYPES.keys())
    # 动态从 CORRUPTION_TYPES 构造类型定义，保证与 dict 完全一致
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
- Produce {MIN_EDITS}–{MAX_EDITS} edits.
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
- Top-level keys: "global_explanation" (string) and "edits" (list of {MIN_EDITS}–{MAX_EDITS}).
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

  - target_find (<= {MAX_SNIPPET_CHARS} chars)
      → "target_find" **MUST be copied verbatim from the original TEXT blocks**, with characters identical
        to the source (no paraphrasing, no normalization, no added/removed whitespace).

  - replacement (<= {MAX_REPLACE_CHARS} chars)
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
    仅两类：text / image_url；逐条按顺序放入列表。
    - 对 text：按块截断到 max_text_chars_per_block
    - 对 image_url：透传 {"type":"image_url","image_url":{"url":...}}
    - 软预算：累计文本字符到 soft_char_budget 附近就停
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

        # 预算/数量控制
        if soft_char_budget and total_chars > int(soft_char_budget * 1.05):
            break
        if max_parts and len(parts) >= max_parts:
            break

    return parts

def build_mm_user_content_streaming(title: str, paper: dict) -> List[Dict[str, Any]]:
    """
    第一段：规则提示；随后：按顺序逐块加入论文内容（text/image_url）
    """
    parts = [{"type": "text", "text": build_rules_prompt(title)}]
    parts.extend(paper_to_mm_parts_streaming(paper))
    return parts

def call_openai_for_patches_mm(user_content: List[Dict[str, Any]], model: str, api_key: Optional[str]) -> Optional[dict]:
    """
    多模态 Chat Completions：messages[1].content 是一个列表，每个元素是 {"type":"text",...} 或 {"type":"image_url",...}
    """
    client = get_openai_client(api_key)
    last_exc = None
    for attempt in range(1, LLM_MAX_RETRIES+1):
        try:
            resp = client.chat.completions.create(
                model=model,
                messages=[
                    {"role":"system","content":"You are a strict JSON-only generator that outputs a single JSON object and nothing else."},
                    {"role":"user","content": user_content}
                ],
                temperature=TEMPERATURE,
                max_tokens=MAX_DOC_CHARS,
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

# ================== Patch 结构校验与应用（未改） ==================
def validate_patches(obj: dict) -> Tuple[bool, str]:
    if not isinstance(obj, dict):
        return False, "Not a dict"
    if "edits" not in obj or not isinstance(obj["edits"], list):
        return False, "Missing 'edits' list"

    edits = obj["edits"]
    if not (MIN_EDITS <= len(edits) <= MAX_EDITS):
        return False, f"edits count must be {MIN_EDITS}..{MAX_EDITS}"

    seen_ids = set()
    MAX_LOCATION_LEN = 100  # 位置字符串的长度上限（章节名/简短路径足够）
    for e in edits:
        if not isinstance(e, dict):
            return False, "edit not dict"

        # 新增 needs_cross_section 布尔字段
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
        
        # needs_cross_section 必须是布尔
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
        if len(e["target_find"]) > MAX_SNIPPET_CHARS:
            return False, "target_find too long"

        # replacement
        if not isinstance(e["replacement"], str):
            return False, "replacement must be string"
        if len(e["replacement"]) > MAX_REPLACE_CHARS:
            return False, "replacement too long"

        # id 唯一性
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


# ============== NEW: 与 batch_parse_and_review 一致的“存在即跳过 + .inprogress” ==============
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
    如果对应的 synth 文件不存在，返回 None；
    如果存在，返回 audit_log.apply_results 里 applied=True 的数量。
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

# ================== 单篇处理（同步函数，调用改为逐块多模态） ==================
def process_one_paper_sync(paper_json_path: Path, model: str, api_key: Optional[str], overwrite: bool) -> Tuple[Path, bool, str]:
    """
    同步处理单篇（具备断点续传）：
    - 若目标 paper_synth_{model}.json 已存在且未指定 overwrite => 直接跳过（resume）
    - 处理前创建 .inprogress，结束/异常均尝试删除
    """
    parent = paper_json_path.parent
    out_json = _out_json_path(paper_json_path, model)
    inprog  = _inprogress_path(paper_json_path, model)

    # ===== 断点续传：已完成则跳过（仅在未开启 overwrite 时） =====
    # if (not overwrite) and _already_done(paper_json_path, model):
    #     return parent, True, "skip_done"

    # 写入 inprogress（与 batch_parse_and_review 一致的思路）
    try:
        inprog.write_text(str(time.time()), encoding="utf-8")
    except Exception:
        pass

    try:
        # 读取 paper_final.json
        try:
            paper = load_json(str(paper_json_path))
        except Exception as e:
            return parent, False, f"load_json failed: {e}"

        # 提取标题
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

        # NEW —— 逐块多模态：不再拼接全文，直接构建多模态 content 列表
        user_content = build_mm_user_content_streaming(paper_title, paper)

        # 调 LLM（多模态）
        llm_obj = call_openai_for_patches_mm(user_content, model=model, api_key=api_key)
        if llm_obj is None:
            # 兜底：若模型路由不支持多模态，则退回“仅规则文本”（仍然要求输出 JSON）
            print("[WARN] MM call failed; falling back to text-only rules.", file=sys.stderr)
            rules_only = [{"type":"text","text": build_rules_prompt(paper_title)}]
            llm_obj = call_openai_for_patches_mm(rules_only, model=model, api_key=api_key)

        # if llm_obj is None:
        #     debug_path = parent / "llm_resp_debug.json"
        #     save_json({"error": "llm_none"}, str(debug_path))
        #     return parent, False, "LLM returned None"

        ok, msg = validate_patches(llm_obj)
        if not ok:
            debug_path = parent / "llm_resp_debug.json"
            save_json({"raw_llm_obj": llm_obj, "validate_msg": msg}, str(debug_path))
            return parent, False, f"validate failed: {msg}"

        # 应用篡改
        edits = llm_obj["edits"]
        modified_paper, applied_records = apply_edits_to_paper(paper, edits)

        # 审计日志
        audit_log = {
            "source_file": str(paper_json_path),
            "paper_title": paper_title,
            "created_at": datetime.utcnow().isoformat() + "Z",
            "global_explanation": llm_obj.get("global_explanation", ""),
            "edits": edits,
            "apply_results": applied_records
        }

        # 写出最终结果（与以往一致）
        out_obj = {
            "synthetic_for_detector": True,  # 明确标注为“检测/训练用的合成样本”
            "audit_log": audit_log,
            "paper": modified_paper
        }
        save_json(out_obj, str(out_json))
        print(f"[DONE] Saved: {out_json}")

        return parent, True, "ok"

    finally:
        # 清理 inprogress
        try:
            if inprog.exists():
                inprog.unlink()
        except Exception:
            pass


# ================== 扫描 root_dir（一级子目录） ==================
def find_paper_jsons(root_dir: Path) -> List[Path]:
    """
    在 root_dir 的一级子目录中查找 paper_final.json
    形如：root_dir/<paper_folder>/paper_final.json
    """
    out = []
    for sub in sorted(root_dir.iterdir()):
        if not sub.is_dir():
            continue
        cand = sub / "paper_final.json"
        if cand.exists():
            out.append(cand)
    return out

# ================== 并发运行（async + 线程池） ==================
async def bounded_worker(sem: asyncio.Semaphore, loop, paper_json_path: Path, model: str, api_key: Optional[str], overwrite: bool):
    async with sem:
        # 将同步处理放入默认线程池
        return await loop.run_in_executor(None, process_one_paper_sync, paper_json_path, model, api_key, overwrite)

async def run_batch(root_dir: Path, concurrency: int, model: str,
                    api_key: Optional[str], overwrite: bool,
                    overwrite_apply: Optional[int]) -> Dict[str, Any]:
    paper_paths = find_paper_jsons(root_dir)
    if not paper_paths:
        return {"total": 0, "ok": 0, "fail": [], "skipped_done": 0}

    # 与“以前脚本”的风格保持一致：已完成的直接跳过计划队列（但允许 overwrite/overwrite_apply 控制重跑）
    planned: List[Path] = []
    skipped_done = 0  # 统计跳过数量（可选）

    for p in paper_paths:

        # 1) overwrite=True: 强制重跑
        if overwrite:
            planned.append(p)
            continue

        # 2) 如果设置了 overwrite_apply：根据 applied=True 的数量决定是否重跑
        if overwrite_apply is not None:
            cnt = _get_applied_true_count(p, model)
            if cnt is None:
                # 没有 synth 文件 → 必须跑
                planned.append(p)
                continue
            if cnt < overwrite_apply:
                # applied 太少 → 需要重跑
                planned.append(p)
                continue
            else:
                # 覆盖率足够 → 跳过
                skipped_done += 1
                continue

        # 3) 原始逻辑：已完成就跳过
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
    tasks = [bounded_worker(sem, loop, p, model, api_key, overwrite) for p in planned]

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
        "ok": ok_cnt + skipped_done,   # 把预先跳过的也计入“已处理”
        "fail": fail,
        "skipped_done": skipped_done,
    }

def compute_applied_distribution(root_dir: Path, model: str) -> None:
    """
    扫描 root_dir 下所有 paper_final.json，对已有 synth 文件统计 applied=True 的分布。
    仅用于打印统计信息，不影响流程。
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
        print(f"    {k:3d} -> {hist[k]} 文件")
    print()

# ================== CLI ==================
def main():
    parser = argparse.ArgumentParser(
        description="Batch synthesize corrupted papers (JSON patches applied) under folders that contain paper_final.json"
    )
    parser.add_argument(
        "--root_dir",
        type=str, default="/mnt/parallel_ssd/home/zdhs0006/mlrbench/download/downloads/test2",
        help="根目录，例如 downloads/ICML_2025_oral_test（其子目录内含 paper_final.json）",
    )
    parser.add_argument(
        "--concurrency", "-j",
        type=int, default=10,
        help="并发数（默认 4）",
    )
    parser.add_argument(
        "--model",
        type=str, default=DEFAULT_MODEL,
        help=f"用于 LLM 生成的模型名（默认 {DEFAULT_MODEL}）",
    )
    parser.add_argument(
        "--overwrite",
        # default=True,
        action="store_true",
        help="Force regenerate and overwrite existing synthetic files even if they already exist.",
    )
    parser.add_argument(
        "--overwrite_apply",
        type=int,
        default=None,
        help="If set, re-generate outputs whose applied=True count is less than this value."
    )
    args = parser.parse_args()

    if not OPENAI_API_KEY:
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
            api_key=OPENAI_API_KEY,
            overwrite=args.overwrite,
            overwrite_apply=args.overwrite_apply,
        )
    )

    print(f"[DONE] total={summary['total']}  ok={summary['ok']}  fail={len(summary['fail'])}  skipped_done={summary.get('skipped_done', 0)}")
    if summary["fail"]:
        print("[FAILED ITEMS]")
        for i, (path, msg) in enumerate(summary["fail"], 1):
            print(f"  {i}. {path} :: {msg}")

    # 如果设置了 --overwrite_apply，则顺便统计一下当前 root_dir 下的 applied 分布
    if args.overwrite_apply is not None:
        compute_applied_distribution(root, args.model)

if __name__ == "__main__":
    main()
