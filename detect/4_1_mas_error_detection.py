#!/usr/bin/env python3
# -*- coding: utf-8 -*-

from __future__ import annotations
import argparse, json, os, re, sys, time, traceback
from dataclasses import dataclass, asdict
from datetime import datetime
from pathlib import Path
from typing import Any, Dict, List, Optional, Tuple
from threading import Lock
from concurrent.futures import ThreadPoolExecutor, as_completed
from typing import Callable, Union
from mas_reference_helper import enrich_section_blocks_with_local_references

try:
    from tqdm import tqdm  # noqa
except Exception:
    tqdm = None

# ---- OpenAI SDK ----
from openai import OpenAI
from openai import (
    APIError, RateLimitError, APITimeoutError, APIConnectionError,
    AuthenticationError, BadRequestError, PermissionDeniedError,
    UnprocessableEntityError
)

# =========================
# 全局配置
# =========================
DEFAULT_DETECT_MODEL = "qwen3-235b-a22b-instruct-2507"
DEFAULT_SYNTH_MODEL  = "o4-mini"  # ★ 新增
OPENAI_API_KEY  = os.getenv("OPENAI_API_KEY", None)

# ---------- DEFAULT FLAGS ----------
DEFAULT_ENABLE_MEMORY_BUILD   = True    # 全文记忆构建
DEFAULT_ENABLE_GLOBAL_REVIEW  = True   # 全局找错
DEFAULT_ENABLE_SECTION_REVIEW = True    # Section 找错
DEFAULT_ENABLE_PER_TASK       = True   # (Planner→(Retriever)→(Web)→Specialist)


# —— Retriever / Specialist 控制 —— #
DEFAULT_ENABLE_RETRIEVER = True                # True → 启用 Retriever(+WebSearch)
DEFAULT_ENABLE_MEMORY_INJECTION = True         # True → Specialist 可注入 memory
DEFAULT_ENABLE_SECTION_FINDINGS_AS_PRIOR = False  # True → 将同名 section findings 作为先验传入 Specialist
DEFAULT_ENABLE_WEB_SEARCH     = False

# —— Merge 控制 —— #
DEFAULT_ENABLE_MERGE = False   # 默认进行合并裁决（与原行为一致）

# ---------- ★ NEW: 是否允许多模态输入（image_url） ----------
DEFAULT_ENABLE_MM = False     


DEFAULT_SEARCH_MAX_RESULTS    = int(os.getenv("SEARCH_MAX_RESULTS", "5"))
DEFAULT_SEARCH_TEMPERATURE    = float(os.getenv("SEARCH_TEMPERATURE", "0.0"))  # Web Search 阶段温度
MEMORY_MAX_CHARS_PER_TASK     = int(os.getenv("MEMORY_MAX_CHARS_PER_TASK", "20000"))

# 统一上下文预算（仅限 *输入*）
MAX_CTX_CHARS = int(os.getenv("MAX_CTX_CHARS", "120000"))

LLM_MAX_RETRIES   = 3   # SDK 层网络/速率重试
EMPTY_RETRY_TIMES = 2   # 逻辑层空输出/解析失败重试（本文件所有阶段统一使用）

DEFAULT_GLOBAL_MAX_FINDINGS = int(os.getenv("GLOBAL_MAX_FINDINGS", "200"))

# ---------- 每阶段 LLM Temperature（可通过环境变量注入） ----------
TEMP_MEMORY          = float(os.getenv("TEMP_MEMORY",          "0.2"))
TEMP_PLANNER         = float(os.getenv("TEMP_PLANNER",         "0.2"))
TEMP_RETRIEVER       = float(os.getenv("TEMP_RETRIEVER",       "0.4"))
TEMP_SPECIALIST      = float(os.getenv("TEMP_SPECIALIST",      "0.4"))
TEMP_GLOBAL_REVIEW   = float(os.getenv("TEMP_GLOBAL_REVIEW",   "0.2"))
TEMP_SECTION_REVIEW  = float(os.getenv("TEMP_SECTION_REVIEW",  "0.2"))
TEMP_MERGER          = float(os.getenv("TEMP_MERGER",          "0.2"))
# Web Search 使用 DEFAULT_SEARCH_TEMPERATURE

# =========================
# I/O & util
# =========================
def load_json(path: Path) -> dict:
    with path.open("r", encoding="utf-8") as f:
        return json.load(f)

def save_json(obj, path: Path):
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("w", encoding="utf-8") as f:
        json.dump(obj, f, indent=2, ensure_ascii=False)

def extract_json_from_text(text: str):
    """尽力从 LLM 自由文本中抽取 JSON 对象（容错）。"""
    text = (text or "").strip()
    if not text:
        raise ValueError("Empty text.")
    # 1) 直接解析
    try:
        return json.loads(text)
    except Exception:
        pass
    # 2) 最外层大括号范围
    start = text.find("{"); end = text.rfind("}")
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
    # 3) 第一个 JSON-like
    m = re.search(r"\{(?:.|\n)*?\}", text)
    if m:
        return json.loads(m.group(0))
    raise ValueError("Failed to extract JSON from LLM output.")

def output_has_zero_findings(out_json: Path) -> bool:
    """
    Return True iff out_json exists and parses, and contains findings == [].
    """
    try:
        if not out_json.exists():
            return False
        obj = load_json(out_json)
        findings = obj.get("findings", None)
        return isinstance(findings, list) and len(findings) == 0
    except Exception:
        return False

# =========================
# OpenAI client & LLM 调用
# =========================
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

def call_llm_chat(messages: List[Dict[str, Any]], model: str, max_tokens: int, temperature: float) -> Optional[str]:
    """底层 chat.completions 调用，已含异常与速率重试。"""
    client = get_openai_client(OPENAI_API_KEY)
    last_exc = None
    for attempt in range(1, LLM_MAX_RETRIES + 1):
        try:
            resp = client.chat.completions.create(
                model=model, messages=messages,
                temperature=temperature, max_tokens=max_tokens, n=1
            )
            return resp.choices[0].message.content
        except RateLimitError as e:
            last_exc = e; wait = 2 ** attempt
            print(f"[RATE LIMIT] attempt {attempt}, waiting {wait}s...", file=sys.stderr)
            time.sleep(wait)
        except (APITimeoutError, APIConnectionError) as e:
            last_exc = e
            print(f"[TEMP ERROR] attempt {attempt}: {e}", file=sys.stderr)
            time.sleep(1.5 * attempt)
        except (BadRequestError, AuthenticationError, PermissionDeniedError, UnprocessableEntityError, APIError) as e:
            last_exc = e
            print(f"[FATAL] {e}", file=sys.stderr); break
        except Exception as e:
            last_exc = e
            print(f"[ERROR] attempt {attempt}: {e}", file=sys.stderr)
            time.sleep(1.0 * attempt)
    print(f"[LLM FAILURE] {last_exc}", file=sys.stderr)
    return None

def call_llm_chat_with_empty_retries(
    msg_builder: Union[Callable[[float], Any], List[Dict[str, Any]], Dict[str, Any]],
    model: str,
    max_tokens: int,
    tag: str,
    dbgdir: Path,
    temperature: float = 0.0,
    expect_key: Optional[str] = None,
    retries: int = EMPTY_RETRY_TIMES
) -> Optional[str]:
    """
    空输出/解析失败重试统一入口：
      - msg_builder: callable(temperature)->messages 或 直接 messages(list/dict)
      - expect_key: 若不为空，则强制解析 JSON 并检查该 key 非空
    """
    last_raw = None

    def _coerce_messages():
        if callable(msg_builder):
            return msg_builder(temperature)
        if isinstance(msg_builder, (list, dict)):
            return msg_builder
        raise TypeError(f"msg_builder must be a callable or list/dict, got: {type(msg_builder)}")

    for k in range(retries + 1):
        try:
            messages = _coerce_messages()
        except Exception as e:
            print(f"[FATAL][{tag}] building messages failed: {e}", file=sys.stderr)
            break

        raw = call_llm_chat(messages, model=model, max_tokens=max_tokens, temperature=temperature)
        last_raw = raw

        if raw is None or not raw.strip():
            print(f"[WARN][{tag}] empty raw (try {k+1}/{retries+1})", file=sys.stderr)
            continue

        if expect_key:
            try:
                obj = extract_json_from_text(raw)
                # expect_key 存在且非空即视为成功
                if obj.get(expect_key) is not None:
                    return raw
                print(f"[WARN][{tag}] parsed but '{expect_key}' missing/empty (try {k+1}/{retries+1})", file=sys.stderr)
                continue
            except Exception as e:
                print(f"[WARN][{tag}] parse failed: {e} (try {k+1}/{retries+1})", file=sys.stderr)
                continue
        else:
            return raw

    if last_raw is not None:
        try:
            save_json({"raw": last_raw}, dbgdir / f"{tag}.last_raw.json")
        except Exception:
            pass
    return last_raw

# --- Responses API + tools（专供 Web Search：Step-B） ---
def call_web_search_via_tool(queries: List[str], model: str, temperature: float) -> List[Dict[str, str]]:
    """
    使用 Responses API + tools=[{"type":"web_search"}]。
    输入 queries（短问句），让模型执行联网检索并直接返回答案。
    返回 [{'query': '...', 'answer': '...'}...]
    """
    if not queries:
        return []

    client = get_openai_client(OPENAI_API_KEY)
    sys_prompt = (
        "You are a web-search assistant. Given a list of short self-contained factual questions, "
        "use the web_search tool to find relevant information on the internet, "
        "and then return concise, direct answers for each question.\n"
        "Return ONLY valid JSON in the form:\n"
        "{'answers':[{'query':'...','answer':'...'}]}\n"
        "Each answer should be brief, factual, and self-contained."
    )
    user_payload = {"queries": queries}
    messages = [
        {"role": "system", "content": sys_prompt},
        {"role": "user", "content": "INPUT(JSON):\n" + json.dumps(user_payload, ensure_ascii=False)}
    ]

    last_exc = None
    for attempt in range(1, LLM_MAX_RETRIES + 1):
        try:
            resp = client.responses.create(
                model=model,
                input=messages,
                tools=[{"type": "web_search"}],
                temperature=temperature,
                max_output_tokens=4000
            )
            # 抓取输出文本
            text_fragments = []
            if hasattr(resp, "output") and resp.output:
                for block in getattr(resp, "output"):
                    t = getattr(block, "type", None)
                    if t == "message":
                        for part in getattr(block, "content", []) or []:
                            if getattr(part, "type", None) in ("output_text", "text"):
                                txt = getattr(part, "text", "")
                                if txt: text_fragments.append(txt)
                    elif t in ("output_text", "text"):
                        txt = getattr(block, "text", "")
                        if txt: text_fragments.append(txt)
            if not text_fragments and hasattr(resp, "output_text"):
                text_fragments.append(resp.output_text)

            raw = "\n".join(text_fragments).strip() if text_fragments else ""
            obj = extract_json_from_text(raw or "{}")

            items = []
            for it in obj.get("answers", []) or []:
                if not isinstance(it, dict):
                    continue
                q = (it.get("query") or "").strip()
                ans = (it.get("answer") or "").strip()
                if q and ans:
                    items.append({"query": q, "answer": ans})
            return items
        except RateLimitError as e:
            last_exc = e; wait = 2 ** attempt
            print(f"[RATE LIMIT] web-search attempt {attempt}, waiting {wait}s...", file=sys.stderr)
            time.sleep(wait)
        except (APITimeoutError, APIConnectionError) as e:
            last_exc = e
            print(f"[TEMP ERROR] web-search attempt {attempt}: {e}", file=sys.stderr)
            time.sleep(1.5 * attempt)
        except (BadRequestError, AuthenticationError, PermissionDeniedError, UnprocessableEntityError, APIError) as e:
            last_exc = e
            print(f"[FATAL] web-search {e}", file=sys.stderr); break
        except Exception as e:
            last_exc = e
            print(f"[ERROR] web-search attempt {attempt}: {e}", file=sys.stderr)
            time.sleep(1.0 * attempt)
    print(f"[WEB SEARCH FAILURE] {last_exc}", file=sys.stderr)
    return []

# =========================
# 数据结构
# =========================
@dataclass
class Task:
    task_id: str
    section: str
    pages: List[int]
    risk_dimension: str
    hints: List[str]

@dataclass
class Evidence:
    span: str
    content_index: Optional[int] = None
    block_type: Optional[str] = None
    # —— web 搜证元数据 —— #
    source_title: Optional[str] = None
    source_url: Optional[str] = None
    source_snippet: Optional[str] = None

@dataclass
class Finding:
    type: str
    section_location: str
    error_location: str
    explanation: str
    confidence: float
    proposed_fix: str
    id: Optional[int] = None

# =========================
# Prompt 模板（统一）
# =========================
class PromptTemplates:
    RISK_DIMENSIONS = [
        "logic","data","theory","conclusion","figure",
        "reproducibility","citation","data_leakage",
        "ethics","rhetoric",
        "other"
    ]


    # —— 新增：统一的 specialist 后缀映射 —— #
    SPECIALIST_SUFFIX = {
        "data": (
            " Focus on dataset and result reliability issues, such as inconsistent dataset names or splits, "
            "missing ablations, unclear data preprocessing, or text–table mismatches."
        ),
        "logic": (
            " Focus on logical consistency and reasoning validity, such as contradictions between statements, "
            "undefined terms, or conclusions not supported by prior steps or evidence."
        ),
        "theory": (
            " Focus on theoretical or mathematical correctness, including incorrect derivations, "
            "invalid assumptions, inconsistent notation or formula usage, or formulas whose results contradict experimental outcomes."
        ),
        "conclusion": (
            " Focus on whether conclusions are properly supported, avoiding overclaiming or "
            "misinterpretation of the reported results."
        ),
        "citation": (
            " Focus on citation accuracy and relevance, such as misused references, missing key prior work, "
            "or citations that do not actually support the claims."
        ),
        "figure": (
            " Focus on inconsistencies in figures or tables, including mislabeled axes, incorrect units, "
            "or mismatches between figure content and textual descriptions."
        ),
        "reproducibility": (
            " Focus on missing experimental details that hinder reproducibility, "
            "such as absent hyperparameters, seeds, number of runs, or dataset preprocessing steps."
        ),
        "data_leakage": (
            " Focus on potential data leakage or contamination, including overlap between training and evaluation data, "
            "or use of privileged information from test sets."
        ),
        "ethical_integrity": (
            " Focus on ethical, safety, or transparency issues, such as missing statements about human data, "
            "unclear dataset licenses, lack of risk discussion, or omitted limitations that should be acknowledged."
        ),
        "rhetorical_style": (
            " Focus on misleading or exaggerated rhetorical choices, such as promotional language, "
            "unsupported superlatives, ambiguous phrasing, or wording that overstates the contribution."
        ),
    }
    DEFAULT_SPECIALIST_SUFFIX = (
        " Focus on general scientific soundness, ensuring claims are evidence-based and internally consistent."
    )

    @staticmethod
    def _suffix_for_risk(risk: str) -> str:
        """Return a concise suffix line tailored to the risk dimension."""
        return PromptTemplates.SPECIALIST_SUFFIX.get(risk, PromptTemplates.DEFAULT_SPECIALIST_SUFFIX)

    # ---- Global Review (FULL PAPER, SECTION + CROSS-SECTION) ----
    @staticmethod
    def global_system() -> str:
        FINDING_TYPES_HINT = ", ".join(PromptTemplates.RISK_DIMENSIONS)
        return (
            "You are a scientific reviewer responsible for checking the **entire manuscript**, including ALL sections, "
            "formulas, tables, figures, and appendices. You must identify **every substantive scientific, logical, "
            "evidential, methodological, theoretical, or empirical issue**, including:\n"
            "- issues **within individual sections**, and\n"
            "- issues that require **cross-section comparison** (Abstract ↔ Experiments, Method ↔ Theory, Claims ↔ Tables, etc.).\n\n"

            "Your goal is to produce the **most complete list of real errors** that impact the validity, correctness, coherence, "
            "or reliability of the paper.\n\n"

            "========================\n"
            "CHECKLIST OF POSSIBLE ISSUES\n"
            "========================\n"
            "When examining the full manuscript, check exhaustively for issues such as:\n"
            "1. *Evidence/Data Integrity* — inconsistent numbers, mismatched metrics between text/table, incorrect dataset stats, "
            "    suspicious numerical claims, missing hyperparameters or run details, errors in reported baselines.\n"
            "2. *Method/Logic Consistency* — undefined symbols, broken derivations, incorrect equations, contradictions across sections, "
            "    steps that do not logically follow, algorithmic steps that cannot be executed.\n"
            "3. *Experimental Design/Protocol* — unclear baselines, unfair comparisons, missing ablations, misleading experiment settings, "
            "    dataset leakage or improper splits.\n"
            "4. *Claim Interpretation Distortion* — exaggerated claims inconsistent with results; contradictions between Abstract, Method, "
            "    and Experiments; claims not supported by evidence.\n"
            "5. *Reference/Background Issues* — factual inaccuracies about prior work, incorrect citations, invented datasets or results.\n"
            "6. *Ethical/Integrity Problems* — missing disclosures, unsafe data use, missing limitation statements.\n"
            "7. *Figure/Table Errors* — mislabeled axes, wrong units, incorrect figure–text alignment, missing legends, visual inconsistencies.\n"
            "8. *Rhetorical/Presentation Manipulation* — misleading phrasing, inflated certainty, vague definitions.\n"
            "9. *Cross-section contradictions* — Abstract claims inconsistent with experiments; Method assumptions contradicted in results; "
            "    stated dataset sizes mismatching tables.\n"
            "10. *Any other issue* that affects scientific soundness.\n\n"

            "========================\n"
            "GUIDELINES\n"
            "========================\n"
            "- Be **exhaustive**: include every valid issue you can find.\n"
            "- Be **specific and grounded**: quote exact spans in 'error_location'.\n"
            "- Avoid stylistic or trivial comments; focus on substantive scientific issues.\n"
            "- If no issues are found, return {\"findings\": []}.\n"
            "- Do NOT output text outside JSON."

            "Return ONLY a valid JSON object with top-level key 'findings'. Each finding must contain EXACTLY:\n"
            f"- type (string): one of [{FINDING_TYPES_HINT}] or 'other'.\n"
            "- section_location (string): section where the problematic claim appears.\n"
            "- error_location (string): the **exact quoted** claim/text/formula/number that is wrong or suspicious.\n"
            "- explanation (string): a concise, factual, and verifiable description of *what is wrong and why*.\n"
            "- confidence (float in [0,1]).\n"
            "- proposed_fix (string): short, concrete correction or improvement.\n\n"
        )


    # ---- Section Review ----
    @staticmethod
    def section_system() -> str:
        FINDING_TYPES_HINT = ", ".join(PromptTemplates.RISK_DIMENSIONS)
        return (
        "You are a scientific reviewer focusing on a SINGLE section of the manuscript. "
        "Given the section’s blocks (text/tables/figures) and an optional MEMORY FOR TOTAL PAPER, "
        "identify scientific, logical, evidential, or methodological problems in this section, "
        "including inconsistencies with the MEMORY summary when relevant.\n\n"

        "Return ONLY a valid JSON object with the top-level key 'findings'. "
        "Each finding object must contain exactly the following fields:\n"
        f"- type (string): one of [{FINDING_TYPES_HINT}] or 'other'.\n"
        "- section_location (string): the section title.\n"
        "- error_location (string): the exact text/claim/passage that is problematic.\n"
        "- explanation (string): concise, verifiable reason.\n"
        "- confidence (float in [0,1]).\n"
        "- proposed_fix (string): short actionable suggestion.\n\n"

        "Possible categories of issues include:\n"
        "1. *Evidence/Data Integrity* — questionable or inconsistent numbers, mismatched statistics, missing details, "
        "or discrepancies between text, tables, or figures.\n"
        "2. *Method/Logic Consistency* — errors in formulas, undefined symbols, conflicting notation, or steps that "
        "do not logically follow.\n"
        "3. *Experimental Design/Protocol* — unclear or asymmetric baselines, missing hyperparameters, or experiment "
        "descriptions that appear incomplete or biased.\n"
        "4. *Claim Interpretation Distortion* — exaggerated or unsupported statements that are not justified by the "
        "evidence presented here or by the MEMORY summary.\n"
        "5. *Reference/Background Fabrication* — suspicious citations, incorrect factual statements, or misdescribed "
        "datasets or prior work.\n"
        "6. *Ethical/Integrity Omission* — missing disclosures, absent notes on sensitive data, or other ethical "
        "information that should reasonably appear.\n"
        "7. *Rhetorical/Presentation Manipulation* — overly strong or promotional language that inflates the contribution.\n"
        "8. *Context Misalignment/Incoherence* — contradictions, shifting definitions, or missing conceptual connections "
        "within this section or relative to the MEMORY summary.\n\n"

        "Guidelines:\n"
        "- Quote exact spans in 'error_location' when possible.\n"
        "- Be exhaustive but avoid redundant findings.\n"
        "- If no issues are found, return {\"findings\": []}.\n"
        "Do NOT include any text outside JSON."
    )



    # ---- Memory（自然语言） ----
    @staticmethod
    def memory_system() -> str:
        return (
            "You are a meticulous scientific summarizer. Read the entire manuscript (all sections, tables, figures)\n"
            "and produce a NATURAL-LANGUAGE MEMORY document (no JSON). The memory should be concise but information-dense,\n"
            "suitable for later reviewers to quickly recall what the paper does and claims, and to check consistency\n"
            "between individual sections and the overall narrative.\n"
            "STRICT RULES:\n"
            "1) Output plain text only (no JSON, no markdown code blocks). Use simple ATX-style headings.\n"
            "2) Begin with a top-level heading: '# Global Summary'.\n"
            "3) Then write one top-level heading per section USING EXACTLY the section titles provided.\n"
            "4) Under each section, use short paragraphs or bullets (bullets with '-' are OK in plain text).\n"
            "5) Capture key points: problems, methods, datasets, metrics, baselines, important claims.\n"
            "6) **Always record core quantitative information** such as key performance numbers, SOTA margins, sample sizes,\n"
            "   error bars, training budgets, or any numeric evidence explicitly reported.\n"
            "7) Prefer dense, verifiable facts. When exact numbers appear, quote them directly.\n"
            "8) Do NOT invent datasets, metrics, numbers, or claims not present in the manuscript.\n"
            "   If an important detail is missing, explicitly write 'Not specified'.\n"
            "9) Do NOT evaluate or criticize the work; describe only what the paper states.\n"
        )


    @staticmethod
    def memory_user(section_titles: List[str]) -> str:
        titles_block = "\n".join([f"- {t}" for t in section_titles]) if section_titles else "- (no sections detected)"
        return (
            "Write a NATURAL-LANGUAGE MEMORY for the paper with the following structure:\n\n"
            "REQUIRED HEADINGS (in this order):\n"
            "1) # Global Summary\n"
            "2) Then one '# <Section Title>' for EACH of the following section titles (use EXACTLY these spellings):\n"
            f"{titles_block}\n\n"
            "CONTENT GUIDELINES PER HEADING:\n"
            "- Global Summary: a compact overview of the problem, core approach, evaluation scope, key findings, and\n"
            "  explicitly stated caveats. Include any major quantitative results if highlighted by the authors.\n"
            "- For each section: extract key ideas, datasets, metrics, baselines, and specific claims.\n"
            "  Always record important quantitative details such as accuracy, scores, improvement margins, runtime,\n"
            "  sample sizes, or other numbers that the authors emphasize. Quote numbers exactly when available.\n"
            "- Use short paragraphs or '-' bullets. Avoid long verbatim quotes.\n"
            "- If important contextual details (e.g., number of runs, dataset license, ablation structure) are missing,\n"
            "  you may note 'Not specified in this section.'\n\n"
            "FORMAT:\n"
            "- Plain text only with ATX-style headings starting with '# '. No JSON. No markdown tables. No code fences.\n"
            "- Do NOT add your own judgments or opinions; summarize only what the manuscript states.\n"
        )



    # ---- Planner ----
    @staticmethod
    def planner_system() -> str:
        return (
            "You are a planning agent for error review.\n"
            "Produce tasks that cover the paper across sections and risk dimensions,\n"
            "but enforce STRICT cardinality: for each (section, risk_dimension) pair, output AT MOST ONE task.\n"
            "- Pack ALL checks for that pair into the task's 'hints' list (concise, surgical, non-redundant bullets).\n"
            "- Prefer breadth (more distinct (section,risk) pairs) over verbosity (many tasks for the same pair).\n"
            "- Deduplicate near-identical hints; merge overlapping checks.\n"
            "- Include figure/table labeling/units, dataset protocol details, metrics/units, and cross-section support checks when relevant.\n"
            "Return ONLY a JSON object with 'tasks': [...]."
        )

    @staticmethod
    def planner_user(outline_obj: Dict[str, Any], risks: List[str]) -> str:
        section_outline_json = json.dumps(outline_obj.get("outline", []), ensure_ascii=False, indent=2)
        allowed = ",".join(risks)
        output_contract = (
            "You MUST return ONLY a valid JSON object with top-level key 'tasks'.\n"
            f"- Allowed values for 'risk_dimension': [{allowed}].\n"
            "- For each (section, risk_dimension) pair, output AT MOST ONE task.\n"
            "- Put ALL the detailed checks for that pair into 'hints' (concise bullets, no redundancy).\n"
            "- NO markdown, NO comments, NO extra text outside JSON.\n\n"
            "OUTPUT(JSON) EXAMPLE (schema only):\n"
            '{"tasks":[{"task_id":"Method.logic","section":"Method","pages":[5,6],'
            '"risk_dimension":"logic","hints":["check loss definition vs text","verify symbol reuse","contrast claim vs ablations"]}]}'
        )
        guidance = (
            "Guidelines:\n"
            "- Aim for as many DISTINCT (section,risk) pairs as reasonably supported by the outline.\n"
            "- For each pair, provide a SINGLE task whose 'hints' exhaustively list checks for that pair.\n"
            "- Merge/cluster near-duplicate hints; avoid trivial boilerplate.\n"
        )
        return (
            "Paper Section Outline (JSON):\n" + section_outline_json + "\n\n" +
            guidance + "\n" + output_contract
        )

    # ---- Retriever ----
    @staticmethod
    def retriever_system() -> str:
        """含 web_queries 的版本（web-enabled）"""
        return (
            "You are an evidence retrieval agent.\n"
            "Given ordered multimodal paper blocks and a TASK, you MUST:\n"
            "1) Extract **as many high-signal** verbatim excerpts as needed as 'paper_evidence'. "
            "   Prefer dense spans that directly support/undermine the claim; avoid trivial/boilerplate text; "
            "   aggressively deduplicate near-duplicates; keep each span focused.\n"
            "2) Propose 'web_queries' ONLY for claims that cannot be confidently verified from the paper alone. "
            "   Each query must be short, self-contained, and factual (no URLs, no citations, no long context).\n\n"
            "Return ONLY a valid JSON object with EXACTLY TWO top-level keys: 'paper_evidence' and 'web_queries'.\n"
            "Schema:\n"
            "{\n"
            '  "paper_evidence": [\n'
            '    {\n'
            '      "span": "<verbatim excerpt from the manuscript>",\n'
            '      "content_index": <int | null>,\n'
            '      "block_type": "text" | "table" | "figure"\n'
            '    }, ...\n'
            '  ],\n'
            '  "web_queries": [\n'
            '    {\n'
            '      "q": "<short self-contained factual question suitable for general web search>",\n'
            '      "why": "<what this query aims to verify or cross-check>"\n'
            '    }, ...\n'
            '  ]\n'
            "}\n\n"
            "Requirements:\n"
            "- 'paper_evidence' MUST contain focused, non-trivial spans (avoid boilerplate). Each item MUST include 'span', "
            "'content_index' (if known; else null), and 'block_type' (one of 'text'/'table'/'figure').\n"
            "- 'web_queries' MAY be empty. Use them only for external/background facts (e.g., canonical dataset stats/splits/years, "
            "baseline definitions, standard formulae/metrics/constants). DO NOT ask about this paper’s own new claims or internal results.\n"
            "- Keep queries neutral and fact-oriented; do NOT include URLs or references.\n"
            "- No hard caps. Prefer **coverage + precision** with deduplication and information density."
        )

    @staticmethod
    def retriever_system_paper_only() -> str:
        """不含 web_search 的版本（paper-only）"""
        return (
            "You are an evidence retrieval agent.\n"
            "Given ordered multimodal paper blocks and a TASK, you MUST:\n"
            "Extract **as many high-signal** verbatim excerpts as needed as 'paper_evidence'. "
            "Prefer dense spans that directly support/undermine the claim; avoid trivial/boilerplate text; "
            "aggressively deduplicate near-duplicates; keep each span focused.\n\n"
            "Return ONLY a valid JSON object with a top-level key 'paper_evidence'. "
            "Each item in 'paper_evidence' must include at least:\n"
            "- 'span': the exact quoted text (verbatim excerpt),\n"
            "- 'content_index': the integer index of the corresponding block (if known),\n"
            "- 'block_type': e.g., 'text', 'table', or 'figure'.\n"
            "Avoid listing trivial or generic sentences. "
            "- No hard caps. Prefer **coverage + precision** with deduplication and information density."
        )

    # ---- Specialist（统一自适应 prompt）----
    @staticmethod
    def specialist_system_unified(risk: str) -> str:
        suffix = PromptTemplates._suffix_for_risk(risk)
        return (
            "You are a specialized scientific reviewer.\n"
            "Inputs may include:\n"
            "- 'paper' evidence: excerpts from the manuscript (may be empty).\n"
            "- 'web' evidence: optional external answers (may be absent).\n"
            "- 'prior_findings': optional prior section-level findings for THIS section (may be absent).\n"
            "- 'memory_text': optional plain-text memory slice (may be empty).\n"
            "Analyze the task and ALL available inputs and return ONLY JSON with top-level key 'findings'.\n"
            "Each element in 'findings' must be an object containing exactly the following fields:\n\n"
            "  - type (string): category of the issue, e.g. 'logic', 'data', 'theory', etc.\n"
            "  - section_location (string): the section or part of the paper where the issue occurs (e.g., \"Abstract\", \"Introduction\", \"Experiments\", \"Related Work\").\n"
            "  - error_location (string): the exact text or claim that is incorrect or problematic (may include LaTeX or multi-line content).\n"
            "  - explanation (string): concise, specific, and verifiable explanation (point to sections/tables/equations/references as needed).\n"
            "  - confidence (float): model’s confidence in [0,1].\n"
            "  - proposed_fix (string): a short suggestion or correction for improvement.\n\n"
            "Be concise, non-redundant, and evidence-grounded. If inputs are minimal, reason from the provided section slice.\n"
            "Critically, DO NOT repeat or rephrase any prior_findings — only output genuinely new or complementary issues.\n"
        ) + suffix

    # ---- Merger ----
    @staticmethod
    def merger_system() -> str:
        return (
            "You are an adjudication reviewer. You are given a list of candidate findings from prior agents.\n"
            "Your task is to merge only truly duplicate or semantically identical findings, and consolidate overlapping ones that clearly describe the same issue.\n"
            "- Do NOT create new findings.\n"
            "- Do NOT delete findings unless they are exact duplicates or completely meaningless.\n"
            "- Merge findings when their underlying error or issue has the same meaning, even if the wording differs.\n"
            "- When merging, keep the clearest and most informative version of each field (e.g., the explanation that best summarizes the issue, or the most specific error_location).\n"
            "- If multiple errors are reported for the exact same location/claim, merge them into a single finding and expand the 'explanation' to cover all distinct issues concisely.\n"
            "- Preserve all non-duplicate findings and maintain roughly the original order.\n\n"
            "Return ONLY a valid JSON object with top-level key 'findings'. Each finding must have exactly:\n"
            "- type (string)\n"
            "- section_location (string)\n"
            "- error_location (string)\n"
            "- explanation (string)\n"
            "- confidence (float in [0,1])\n"
            "- proposed_fix (string)\n\n"
            "Be conservative: prefer preserving content over over-merging. Do not include markdown or any text outside JSON."
        )


# =========================
# JSON block 工具
# =========================
def normalize_blocks(paper: dict) -> List[Dict[str, Any]]:
    raw = paper.get("content", []) or []
    blocks: List[Dict[str, Any]] = []
    for i, item in enumerate(raw):
        content_index = item.get("index", i)
        section_label = item.get("section", None)
        b: Dict[str, Any] = {
            "content_index": int(content_index) if isinstance(content_index, int) else i,
            "type": item.get("type"),
            "section": section_label if isinstance(section_label, str) and section_label.strip() else None,
        }
        if item.get("type") == "text":
            b["text"] = item.get("text", "")
        elif item.get("type") == "image_url":
            b["image_url"] = item.get("image_url")
        else:
            for k, v in item.items():
                if k not in b:
                    b[k] = v
        blocks.append(b)
    blocks.sort(key=lambda x: x.get("content_index", 0))
    return blocks

def cap_blocks_by_budget(blocks: List[Dict[str, Any]], max_chars: int) -> List[Dict[str, Any]]:
    out, cur = [], 0
    for b in blocks:
        cost = len(b.get("text") or "") if b.get("type") == "text" else 500
        if cur + cost > max_chars:
            break
        out.append(b); cur += cost
    return out

def llm_build_outline(blocks: List[Dict[str, Any]], paper_meta: str, dbgdir: Path, model: str) -> Dict[str, Any]:
    """基于 labels 的稳健 outline（无需 LLM 解析）"""
    outline_items: List[Dict[str, Any]] = []
    if not blocks:
        out = {"outline": [], "used": "no_blocks"}
        save_json(out, dbgdir / "outline.parsed.json")
        return out
    cur_title: Optional[str] = None
    cur_start: Optional[int] = None
    last_idx: Optional[int] = None
    def _flush():
        if cur_title is not None and cur_start is not None and last_idx is not None:
            outline_items.append({
                "title": cur_title,
                "start_index": int(cur_start),
                "end_index": int(last_idx),
                "aliases": [],
            })
    for b in blocks:
        sec = b.get("section")
        idx = b.get("content_index")
        if not isinstance(idx, int):
            continue
        if isinstance(sec, str) and sec.strip():
            if cur_title is None:
                cur_title = sec.strip(); cur_start = idx; last_idx = idx
            else:
                if sec.strip() == cur_title and last_idx is not None and idx == last_idx + 1:
                    last_idx = idx
                else:
                    _flush(); cur_title = sec.strip(); cur_start = idx; last_idx = idx
        else:
            if cur_title is not None and last_idx is not None and idx == last_idx + 1:
                last_idx = idx
    _flush()
    out = {"outline": outline_items, "used": "labels"}
    save_json(out, dbgdir / "outline.parsed.json")
    return out

def slice_json_for_task_with_outline(blocks: List[Dict[str, Any]], section_name: str, outline_items: List[Dict[str, Any]], max_chars: int) -> List[Dict[str, Any]]:
    """优先依据 outline 范围切片；否则退化为全文截断。"""
    name = (section_name or "").strip().lower()
    if name and outline_items:
        def match(item: Dict[str, Any]) -> bool:
            t = (item.get("title") or "").strip().lower()
            if not t: return False
            if name in t: return True
            for a in item.get("aliases") or []:
                if name in (a or "").strip().lower():
                    return True
            return False
        for item in outline_items:
            if match(item):
                start = max(0, int(item.get("start_index", 0)))
                end = int(item.get("end_index", start))
                rng_blocks = [
                    b for b in blocks
                    if isinstance(b.get("content_index"), int)
                    and start <= b["content_index"] <= end
                ]
                return cap_blocks_by_budget(rng_blocks, max_chars)
    return cap_blocks_by_budget(blocks, max_chars)

def _guess_mime_from_b64(data_b64: str) -> str:
    head = (data_b64 or "")[:20]
    if head.startswith("iVBOR"): return "image/png"
    if head.startswith("/9j/"):  return "image/jpeg"
    if head.startswith("R0lGOD"): return "image/gif"
    return "image/png"

def _block_to_parts_minimal(
    b: Dict[str, Any],
    max_text_chars_per_block: int = 8000,
    enable_mm: bool = True,   # ★ NEW
) -> List[Dict[str, Any]]:
    parts: List[Dict[str, Any]] = []
    ci = b.get("content_index", b.get("index"))
    if isinstance(ci, str):
        try: ci = int(ci)
        except: ci = None
    sec = (b.get("section") or "").strip()
    typ = b.get("type")
    header = f"[Block #{ci if ci is not None else '?'} | {typ or 'unknown'}{(' |Section: ' + sec) if sec else 'None'}]"
    parts.append({"type": "text", "text": header})

    if typ == "text":
        t = (b.get("text") or "")
        if max_text_chars_per_block and len(t) > max_text_chars_per_block:
            t = t[:max_text_chars_per_block]
        if t.strip():
            parts.append({"type": "text", "text": t})

    elif typ == "image_url":
        # ★ NEW: disable_mm -> 不输入 image，只给一个 text 占位（仍是纯文本输入）
        if not enable_mm:
            parts.append({"type": "text", "text": "[Image omitted: multimodal disabled]"})
            return parts

        img = b.get("image_url")
        url = None
        if isinstance(img, str):
            url = img
        elif isinstance(img, dict):
            if isinstance(img.get("url"), str):
                url = img["url"]
            elif isinstance(img.get("data_b64"), str):
                mime = img.get("mime") or _guess_mime_from_b64(img["data_b64"])
                url = f"data:{mime};base64,{img['data_b64']}"
        if isinstance(url, str) and (url.startswith("data:") or url.startswith("http")):
            parts.append({"type": "image_url", "image_url": {"url": url}})

    return parts


def build_multimodal_parts_from_blocks(
    blocks: List[Dict[str, Any]],
    max_blocks: int = 1_000,
    max_images: int = 48,
    max_text_chars_per_block: int = 10_000,
    enable_mm: bool = True,
) -> List[Dict[str, Any]]:
    parts: List[Dict[str, Any]] = []
    img_count = 0
    for b in blocks[:max_blocks]:
        sub = _block_to_parts_minimal(
            b,
            max_text_chars_per_block=max_text_chars_per_block,
            enable_mm=enable_mm
        )
        for p in sub:
            if p.get("type") == "image_url":
                if not enable_mm:
                    continue
                if img_count >= max_images:
                    continue
                img_count += 1
            parts.append(p)
    return parts


# =========================
# Memory（自然语言）构建与注入
# =========================
def extract_section_titles_from_blocks(blocks: List[Dict[str, Any]]) -> List[str]:
    """从论文 blocks 顺序去重抽取非空 section 名称。"""
    seen = set()
    titles = []
    for b in blocks:
        s = (b.get("section") or "").strip()
        if not s:
            continue
        key = s.lower()
        if key in seen:
            continue
        seen.add(key)
        titles.append(s)
    return titles

def build_memory_messages(blocks: List[Dict[str, Any]], enable_mm: bool) -> List[Dict[str, Any]]:
    section_titles = extract_section_titles_from_blocks(cap_blocks_by_budget(blocks, MAX_CTX_CHARS))
    lead_text = PromptTemplates.memory_user(section_titles)
    parts = [{"type": "text", "text": lead_text}]
    parts.extend(build_multimodal_parts_from_blocks(
        cap_blocks_by_budget(blocks, MAX_CTX_CHARS),
        enable_mm=enable_mm
    ))
    return [
        {"role": "system", "content": PromptTemplates.memory_system()},
        {"role": "user", "content": parts},
    ]


def build_paper_memory(blocks: List[Dict[str, Any]], dbgdir: Path, model: str, enable_mm: bool) -> Tuple[Optional[Dict[str, Any]], Dict[str, Any]]:
    """
    产出自然语言记忆：
    memory_obj = {
        "raw_text": "<plain text memory>",
        "section_titles": [...],
    }
    """
    def _messages(_):
        return build_memory_messages(blocks, enable_mm=enable_mm)
    raw = call_llm_chat_with_empty_retries(
        _messages, model=model, max_tokens=32768, tag="memory_build", dbgdir=dbgdir,
        expect_key=None, temperature=TEMP_MEMORY
    )
    meta = {"memory_parse_ok": False, "error": None, "mode": "natural_language"}
    if not raw or not raw.strip():
        meta["error"] = "memory_build_empty_raw"
        return None, meta

    out_path = dbgdir / "memory.fullpaper.txt"
    out_path.parent.mkdir(parents=True, exist_ok=True
    )
    out_path.write_text(raw, encoding="utf-8")

    section_titles = extract_section_titles_from_blocks(blocks)
    mem_obj = {"raw_text": raw, "section_titles": section_titles}
    save_json({"section_titles": section_titles, "bytes": len(raw.encode('utf-8'))}, dbgdir / "memory.meta.summary.json")
    meta["memory_parse_ok"] = True
    return mem_obj, meta

def _split_memory_by_headings(raw_text: str) -> Dict[str, str]:
    sections = {}
    if not raw_text:
        return sections
    lines = raw_text.splitlines()
    cur_title = None
    buf = []
    def _flush():
        nonlocal cur_title, buf
        if cur_title is not None:
            sections[cur_title] = "\n".join(buf).strip()
        buf = []
    for ln in lines:
        if ln.startswith("# "):
            _flush()
            cur_title = ln[2:].strip()
        else:
            buf.append(ln)
    _flush()
    return sections

def build_memory_for_task(memory_obj: Optional[Dict[str, Any]], current_section: str, max_chars: int) -> str:
    """
    返回**自然语言**片段：'# Global Summary' + 其它所有章节（排除 current_section）。
    若没有结构化标题，退化为截断 raw_text。
    """
    if not memory_obj or not memory_obj.get("raw_text"):
        return ""
    raw = memory_obj["raw_text"]
    parts = _split_memory_by_headings(raw)
    pieces: List[str] = []

    if "Global Summary" in parts and parts["Global Summary"]:
        pieces.append("# Global Summary")
        pieces.append(parts["Global Summary"])

    cur = (current_section or "").strip()
    for title in memory_obj.get("section_titles", []):
        if title == cur:
            continue
        if title in parts and parts[title]:
            pieces.append(f"# {title}")
            pieces.append(parts[title])

    final_text = "\n".join(pieces).strip() or raw
    if max_chars and len(final_text) > max_chars:
        final_text = final_text[:max_chars]
    return final_text

# =========================
# Planner
# =========================
def build_planner_messages_multimodal(
    abstract_blocks: List[Dict[str, Any]],
    outline_obj: Dict[str, Any],
    enable_mm: bool,
) -> List[Dict[str, Any]]:
    allowed_risks = PromptTemplates.RISK_DIMENSIONS
    lead_text = PromptTemplates.planner_user(outline_obj, allowed_risks)
    lead_parts = [{"type": "text", "text": lead_text}]
    abstract_parts = build_multimodal_parts_from_blocks(abstract_blocks, enable_mm=enable_mm)
    return [
        {"role": "system", "content": PromptTemplates.planner_system()},
        {"role": "user", "content": lead_parts + abstract_parts}
    ]


def planner_build_tasks_mm(blocks: List[Dict[str, Any]], outline_obj: Dict[str, Any], dbgdir: Path, model: str, enable_mm: bool) -> Tuple[List[Task], Dict[str, Any]]:
    def _messages(_):
        return build_planner_messages_multimodal(blocks, outline_obj, enable_mm=enable_mm)
    raw = call_llm_chat_with_empty_retries(
        _messages, model=model, max_tokens=32768, tag="planner", dbgdir=dbgdir, expect_key="tasks",
        temperature=TEMP_PLANNER
    )
    tasks: List[Task] = []
    meta = {"planner_parse_ok": False, "error": None, "used": "llm_mm"}
    if raw:
        try:
            obj = extract_json_from_text(raw)
            save_json(obj, dbgdir / "planner.parsed.json")
            for i, t in enumerate(obj.get("tasks", []) or [], 1):
                risk = t.get("risk_dimension", "other")
                if risk not in PromptTemplates.RISK_DIMENSIONS:
                    risk = "other"
                hints = t.get("hints", [])
                if isinstance(hints, dict):
                    hints = [str(v) for v in hints.values()]
                tasks.append(Task(
                    task_id=t.get("task_id", f"task_{i:03d}"),
                    section=t.get("section", f"Section {i}"),
                    pages=t.get("pages", []),
                    risk_dimension=risk,
                    hints=hints if isinstance(hints, list) else []
                ))
            if tasks:
                meta["planner_parse_ok"] = True
        except Exception as e:
            meta["error"] = f"planner_parse_fail: {e}"
            try:
                save_json({"raw": raw}, dbgdir / "planner.raw.json")
            except Exception:
                pass
    if not tasks:
        meta["used"] = "llm_mm_no_tasks"
        save_json(meta, dbgdir / "planner.meta.error.json")
        raise RuntimeError("[PLANNER ERROR] No tasks produced by planner after retries.")
    save_json([asdict(t) for t in tasks], dbgdir / "planner.tasks.json")
    return tasks, meta

# =========================
# Retriever（paper-only / web-enabled）
# =========================
def _build_retriever_messages_multimodal(
    task: Task,
    doc_blocks: List[Dict[str, Any]],
    paper_only: bool,
    memory_slice: Optional[str],
    enable_mm: bool = True,
) -> List[Dict[str, Any]]:
    user_payload = {
        "task": asdict(task),
        "doc_blocks_hint": (
            "Below are the paper blocks in order. "
            + ("Extract exhaustive but non-redundant paper_evidence."
               if paper_only
               else "Extract exhaustive but non-redundant paper_evidence; emit web_queries for genuine uncertainties.")
        )
    }
    lead_parts = [
        {"type": "text", "text": "MEMORY summary of the full paper:\n" + (memory_slice or "")},
        {"type": "text", "text": "INPUT(JSON, task):\n" + json.dumps(user_payload, ensure_ascii=False)}
    ]
    block_parts = build_multimodal_parts_from_blocks(doc_blocks, enable_mm=enable_mm)
    return [
        {
            "role": "system",
            "content": (
                PromptTemplates.retriever_system_paper_only()
                if paper_only else
                PromptTemplates.retriever_system()
            ),
        },
        {"role": "user", "content": lead_parts + block_parts},
    ]

def retriever_extract_and_questions(
    task: Task,
    slice_blocks: List[Dict[str, Any]],
    dbgdir: Path,
    model: str,
    paper_only: bool,
    memory_slice: Optional[str],
    enable_mm: bool = True,
) -> Tuple[List[Evidence], List[Dict[str, str]], Dict[str, Any]]:
    """
    执行 Retriever：
      - 返回 paper_evidence: List[Evidence]
      - 返回 web_queries: List[{'q','why'}]  (paper_only=True 时恒为空)
    """
    save_json(slice_blocks, dbgdir / f"task_{task.task_id}.doc_slice.json")

    def _messages(_):
        return _build_retriever_messages_multimodal(
            task=task, doc_blocks=slice_blocks, paper_only=paper_only, memory_slice=memory_slice, enable_mm=enable_mm
        )

    raw = call_llm_chat_with_empty_retries(
        _messages, model=model, max_tokens=32768,
        tag=f"task_{task.task_id}.retriever", dbgdir=dbgdir,
        expect_key="paper_evidence", temperature=TEMP_RETRIEVER
    )

    paper_e: List[Evidence] = []
    web_q: List[Dict[str, str]] = []
    meta = {"retriever_parse_ok": False, "error": None, "paper_only": paper_only}

    if raw:
        try:
            obj = extract_json_from_text(raw or "{}")
            save_json(obj, dbgdir / f"task_{task.task_id}.retriever.parsed.json")

            for e in obj.get("paper_evidence", []) or []:
                if not isinstance(e, dict):
                    continue
                span = (e.get("span") or "").strip()
                if not span:
                    continue
                paper_e.append(Evidence(
                    span=span,
                    content_index=e.get("content_index"),
                    block_type=e.get("block_type")
                ))

            if not paper_only:
                for q in obj.get("web_queries", []) or []:
                    if not isinstance(q, dict):
                        continue
                    qtext = (q.get("q") or "").strip()
                    why   = (q.get("why") or "").strip()
                    if qtext:
                        web_q.append({"q": qtext, "why": why})

            meta["retriever_parse_ok"] = True
        except Exception as e:
            meta["error"] = f"retriever_parse_fail: {e}"
            try:
                save_json({"raw": raw}, dbgdir / f"task_{task.task_id}.retriever.raw.json")
            except Exception:
                pass

    save_json([asdict(e) for e in paper_e], dbgdir / f"task_{task.task_id}.retriever.paper_evidence.json")
    if not paper_only:
        save_json(web_q, dbgdir / f"task_{task.task_id}.retriever.web_queries.json")
    else:
        save_json({"web_queries": [], "note": "web_search disabled; using paper-only retriever prompt."},
                  dbgdir / f"task_{task.task_id}.retriever.web_queries.json")

    return paper_e, web_q, meta

def perform_web_search_for_queries(
    web_queries: List[Dict[str, str]],
    dbgdir: Path,
    detect_model: str,
    max_results: int = DEFAULT_SEARCH_MAX_RESULTS,
    temperature: float = DEFAULT_SEARCH_TEMPERATURE,
) -> List[Evidence]:
    """
    执行联网检索（Step-B）：
      - 输入：Retriever 产出的简短可检索问句列表 [{'q','why'}...]
      - 过程：使用 Responses API 的 tools=[{'type':'web_search'}] 调用（见 call_web_search_via_tool）
      - 输出：将答案转为 Evidence（block_type='web'），不强制包含 URL（与上游兼容）
    说明：
      - max_results 只是技术上限制本次最多处理多少 query，避免目录爆炸。
      - 由于官方 web_search 返回可能不含 URL，本函数将 answer 作为 Evidence.span 保存，
        并把原 query 一并写入 span 方便审阅。
    """
    dbgdir.mkdir(parents=True, exist_ok=True)
    # 1) 规范/去重/裁剪 queries
    qlist: List[str] = []
    for item in web_queries or []:
        if not isinstance(item, dict):
            continue
        q = (item.get("q") or "").strip()
        if q and q not in qlist:
            qlist.append(q)
        if len(qlist) >= max_results:
            break

    save_json({"input_web_queries": web_queries, "used_queries": qlist}, dbgdir / "web_queries.input.json")
    if not qlist:
        save_json({"answers": [], "note": "no queries"}, dbgdir / "web_queries.answers.json")
        return []

    # 2) 调用联网搜索工具
    answers = call_web_search_via_tool(queries=qlist, model=detect_model, temperature=temperature)
    save_json({"answers": answers}, dbgdir / "web_queries.answers.json")

    # 3) 转为 Evidence（block_type='web'，不依赖 URL 字段）
    evidences: List[Evidence] = []
    for a in answers or []:
        if not isinstance(a, dict):
            continue
        q = (a.get("query") or "").strip()
        ans = (a.get("answer") or "").strip()
        if not ans:
            continue
        span_text = f"[WEB] Q: {q}\nA: {ans}" if q else f"[WEB] {ans}"
        evidences.append(Evidence(
            span=span_text,
            content_index=None,
            block_type="web",
            source_title=None,
            source_url=None,
            source_snippet=ans
        ))

    save_json([asdict(e) for e in evidences], dbgdir / "web_queries.evidence.json")
    return evidences

# =========================
# Specialist / Global / Section / Merger
# =========================
def _build_specialist_messages_multimodal(
    task: Task,
    evidence_list: List[Evidence],
    neighbor_blocks: List[Dict[str, Any]],
    paper_only: bool,
    memory_slice: Optional[str],
    prior_findings: Optional[List[Finding]] = None,
    use_retriever: bool = True,
    enable_mm: bool = True,
) -> List[Dict[str, Any]]:
    def _finding_to_minimal_dict(f: Finding) -> Dict[str, Any]:
        return {
            "id": f.id,
            "type": f.type,
            "section_location": f.section_location,
            "error_location": f.error_location,
            "explanation": f.explanation,
            "confidence": float(f.confidence),
            "proposed_fix": f.proposed_fix,
        }

    payload = {
        "task": asdict(task),
        "evidence": [asdict(e) for e in evidence_list],
        "neighbor_hint": "Local neighborhood blocks for context.",
        "prior_findings": [_finding_to_minimal_dict(f) for f in (prior_findings or [])],
        "retriever_used": bool(use_retriever),
        "memory_text": memory_slice or "",
    }
    lead_parts = [
        {"type": "text", "text": "INPUT(JSON):\n" + json.dumps(payload, ensure_ascii=False)}
    ]
    neighbor_parts = build_multimodal_parts_from_blocks(neighbor_blocks, enable_mm=enable_mm)
    return [
        {
            "role": "system",
            "content": PromptTemplates.specialist_system_unified(task.risk_dimension),
        },
        {"role": "user", "content": lead_parts + neighbor_parts},
    ]

def specialist_review(
    task: Task,
    evid: List[Evidence],
    neighbor_blocks: List[Dict[str, Any]],
    dbgdir: Path,
    model: str,
    paper_only: bool,
    memory_slice: Optional[str],
    prior_findings: Optional[List[Finding]] = None,
    use_retriever: bool = True,
    enable_mm: bool = True,
) -> Tuple[List[Finding], Dict[str, Any]]:
    messages = _build_specialist_messages_multimodal(
        task=task,
        evidence_list=evid,
        neighbor_blocks=neighbor_blocks,
        paper_only=paper_only,
        memory_slice=memory_slice,
        prior_findings=prior_findings,
        use_retriever=use_retriever,
        enable_mm=enable_mm
    )
    raw = call_llm_chat_with_empty_retries(
        messages, model=model, max_tokens=32768,
        tag=f"task_{task.task_id}.specialist_{task.risk_dimension}",
        dbgdir=dbgdir, expect_key="findings",
        temperature=TEMP_SPECIALIST
    )

    findings: List[Finding] = []
    meta = {"specialist_parse_ok": False, "error": None, "paper_only": paper_only, "used_retriever": bool(use_retriever)}

    if raw:
        try:
            obj = extract_json_from_text(raw)
            save_json(obj, dbgdir / f"task_{task.task_id}.specialist_{task.risk_dimension}.parsed.json")
            items = obj.get("findings", [])
            if isinstance(items, dict):
                items = [items]
            if not isinstance(items, list):
                items = []
            for f in items:
                if not isinstance(f, dict):
                    continue
                sec = (f.get("section_location") or "").strip()
                err = (f.get("error_location") or "").strip()
                exp = (f.get("explanation") or "").strip()
                if not sec or not exp or not err:
                    continue
                try:
                    conf = float(f.get("confidence", 0.5))
                except Exception:
                    conf = 0.5
                conf = max(0.0, min(conf, 1.0))
                findings.append(Finding(
                    type=f.get("type", "other"),
                    section_location=sec,
                    error_location=err,
                    explanation=exp,
                    confidence=conf,
                    proposed_fix=(f.get("proposed_fix") or "").strip()
                ))
            meta["specialist_parse_ok"] = True
        except Exception as e:
            meta["error"] = f"specialist_parse_fail: {e}"
            try:
                save_json({"raw": raw}, dbgdir / f"task_{task.task_id}.specialist_{task.risk_dimension}.raw.json")
            except Exception:
                pass
    save_json([asdict(f) for f in findings], dbgdir / f"task_{task.task_id}.specialist_{task.risk_dimension}.findings.json")
    return findings, meta

def build_global_user_parts(blocks: List[Dict[str, Any]], global_max_findings:int, enable_mm: bool) -> List[Dict[str, Any]]:
    lead = {"type": "text","text":("- Prioritize cross-section contradictions and support mismatches.\n"
                                   "- Quote exact spans in 'error_location' when possible.\n"
                                   "- No hard caps; be exhaustive yet non-redundant.\n")}
    parts = [lead]
    parts.extend(build_multimodal_parts_from_blocks(
        cap_blocks_by_budget(blocks, MAX_CTX_CHARS),
        max_blocks=2000, max_images=64, max_text_chars_per_block=10_000, enable_mm=enable_mm
    ))
    return parts

def global_cross_section_review(blocks: List[Dict[str, Any]], dbgdir: Path, model: str, global_max_findings:int, enable_mm: bool) -> Tuple[List[Finding], Dict[str, Any]]:
    def _messages(_):
        return [
            {"role": "system", "content": PromptTemplates.global_system()},
            {"role": "user", "content": build_global_user_parts(blocks, global_max_findings, enable_mm)},
        ]

    raw = call_llm_chat_with_empty_retries(
        _messages, model=model, max_tokens=32768, tag="global_review", dbgdir=dbgdir, expect_key="findings",
        temperature=TEMP_GLOBAL_REVIEW
    )
    findings: List[Finding] = []
    meta = {"global_parse_ok": False, "error": None, "used": "llm_mm_global"}

    if raw:
        try:
            obj = extract_json_from_text(raw)
            save_json(obj, dbgdir / "global_review.parsed.json")
            items = obj.get("findings", [])
            if isinstance(items, dict): items = [items]
            if not isinstance(items, list): items = []
            for f in items:
                if not isinstance(f, dict): continue
                sec = (f.get("section_location") or "").strip()
                err = (f.get("error_location") or "").strip()
                exp = (f.get("explanation") or "").strip()
                if not (sec and err and exp): continue
                try: conf = float(f.get("confidence", 0.5))
                except Exception: conf = 0.5
                conf = max(0.0, min(conf, 1.0))
                findings.append(Finding(
                    type=(f.get("type") or "other").strip() or "other",
                    section_location=sec,
                    error_location=err,
                    explanation=exp,
                    confidence=conf,
                    proposed_fix=(f.get("proposed_fix") or "").strip()
                ))
            meta["global_parse_ok"] = True
        except Exception as e:
            meta["error"] = f"global_parse_fail: {e}"
            try: save_json({"raw": raw}, dbgdir / "global_review.raw.json")
            except Exception: pass
    save_json([asdict(f) for f in findings], dbgdir / "global_review.findings.json")
    return findings, meta

# ---- Section Review ----
def build_section_user_parts(
    section_blocks: List[Dict[str, Any]],
    section_title: str,
    memory_slice: Optional[str],
    enable_mm: bool = True,
) -> List[Dict[str, Any]]:
    lead_parts = []
    mem_text = memory_slice or ""
    lead_parts.append({"type": "text", "text": f"MEMORY summary of the full paper:\n{mem_text}"})
    lead_parts.append({"type": "text", "text": f"- Target section: {section_title}\n- Be exhaustive but non-redundant within this section.\n"})
    lead_parts.extend(build_multimodal_parts_from_blocks(
        cap_blocks_by_budget(section_blocks, MAX_CTX_CHARS),
        max_blocks=1500, max_images=48, max_text_chars_per_block=10_000, enable_mm=enable_mm
    ))
    return lead_parts

def section_level_review(
    section_title: str,
    section_blocks: List[Dict[str, Any]],
    dbgdir: Path,
    model: str,
    memory_slice: Optional[str],
    enable_mm: bool = True
) -> Tuple[List[Finding], Dict[str, Any]]:
    def _messages(_):
        return [
            {"role": "system", "content": PromptTemplates.section_system()},
            {"role": "user", "content": build_section_user_parts(section_blocks, section_title, memory_slice, build_section_user_parts)},
        ]
    raw = call_llm_chat_with_empty_retries(
        _messages, model=model, max_tokens=32768,
        tag=f"section_review.{re.sub(r'[^A-Za-z0-9._-]+','_', section_title or 'Unknown')}",
        dbgdir=dbgdir, expect_key="findings",
        temperature=TEMP_SECTION_REVIEW
    )

    findings: List[Finding] = []
    meta = {"section": section_title, "parse_ok": False, "error": None}
    if raw:
        try:
            obj = extract_json_from_text(raw)
            save_json(obj, dbgdir / "section_review.parsed.json")
            items = obj.get("findings", [])
            if isinstance(items, dict): items = [items]
            if not isinstance(items, list): items = []
            for f in items:
                if not isinstance(f, dict): continue
                sec = (f.get("section_location") or "").strip() or section_title
                err = (f.get("error_location") or "").strip()
                exp = (f.get("explanation") or "").strip()
                if not (sec and err and exp): continue
                try: conf = float(f.get("confidence", 0.5))
                except Exception: conf = 0.5
                conf = max(0.0, min(conf, 1.0))
                findings.append(Finding(
                    type=(f.get("type") or "other").strip() or "other",
                    section_location=sec,
                    error_location=err,
                    explanation=exp,
                    confidence=conf,
                    proposed_fix=(f.get("proposed_fix") or "").strip()
                ))
            meta["parse_ok"] = True
        except Exception as e:
            meta["error"] = f"section_parse_fail: {e}"
            try: save_json({"raw": raw}, dbgdir / "section_review.raw.json")
            except Exception: pass

    save_json([asdict(x) for x in findings], dbgdir / "section_review.findings.json")
    return findings, meta

def build_merger_user(all_findings: List[Finding]) -> str:
    def finding_to_dict(f: Finding) -> Dict[str, Any]:
        return {
            "type": f.type,
            "section_location": f.section_location,
            "error_location": f.error_location,
            "explanation": f.explanation,
            "confidence": float(f.confidence),
            "proposed_fix": f.proposed_fix,
        }
    payload = {"candidates": [finding_to_dict(f) for f in all_findings]}
    return "INPUT(JSON):\n" + json.dumps(payload, ensure_ascii=False)

def merge_and_adjudicate(all_findings: List[Finding], dbgdir: Path, model: str) -> List[Finding]:
    save_json([asdict(f) for f in all_findings], dbgdir / "merge.input_findings.json")
    if not all_findings:
        save_json([], dbgdir / "merge.output_findings.json")
        print(f"[DEBUG][merge] input empty; total=0")
        return []

    sys_msg = {"role": "system", "content": PromptTemplates.merger_system()}
    user_msg = {"role": "user", "content": build_merger_user(all_findings)}

    raw = call_llm_chat_with_empty_retries(
        [sys_msg, user_msg],
        model=model,
        max_tokens=50000,
        tag="merge",
        dbgdir=dbgdir,
        temperature=TEMP_MERGER,
        expect_key="findings",
        retries=EMPTY_RETRY_TIMES
    )

    merged: List[Finding] = []
    fallback = False
    try:
        obj = extract_json_from_text(raw or "")
        save_json(obj, dbgdir / "merge.llm.parsed.json")
        items = obj.get("findings", [])
        if isinstance(items, dict): items = [items]
        if not isinstance(items, list): items = []
        for i, f in enumerate(items, start=1):
            if not isinstance(f, dict): continue
            ftype = (f.get("type") or "").strip()
            sec = (f.get("section_location") or "").strip()
            err = (f.get("error_location") or "").strip()
            exp = (f.get("explanation") or "").strip()
            pfix = (f.get("proposed_fix") or "").strip()
            if not (ftype and sec and err and exp): continue
            try: conf = float(f.get("confidence", 0.5))
            except Exception: conf = 0.5
            conf = max(0.0, min(conf, 1.0))
            merged.append(Finding(
                id=i, type=ftype, section_location=sec, error_location=err,
                explanation=exp, confidence=conf, proposed_fix=pfix,
            ))
    except Exception as e:
        save_json({"merge_parse_error": f"{type(e).__name__}: {e}"}, dbgdir / "merge.parse_error.json")
        merged = []

    if not merged:
        save_json({"fallback": True, "reason": "merge_failed_or_empty"}, dbgdir / "merge.fallback.json")
        merged = list(all_findings)
        fallback = True

    if fallback:
        print(f"[DEBUG][merge] merge failed, fallback to original findings; total={len(merged)}")
    else:
        print(f"[DEBUG][merge] merged {len(merged)} findings (from {len(all_findings)}); total={len(merged)}")

    save_json([asdict(f) for f in merged], dbgdir / "merge.output_findings.json")
    return merged

# =========================
# 主流程（单文件）
# =========================
def _extract_ground_truth_from_synth(obj: dict) -> List[Dict[str, Any]]:
    gt: List[Dict[str, Any]] = []

    if not isinstance(obj, dict):
        return gt

    audit_log = obj.get("audit_log", {})
    if not isinstance(audit_log, dict):
        return gt

    # 1) 先从 apply_results 里找出 applied=True 的 id
    applied_ids = set()
    for a in audit_log.get("apply_results", []) or []:
        if not isinstance(a, dict):
            continue
        if a.get("applied") is True and isinstance(a.get("id"), int):
            applied_ids.add(a["id"])

    # 2) 遍历 edits，只保留 applied_ids 里面的
    for e in audit_log.get("edits", []) or []:
        if not isinstance(e, dict):
            continue

        gid   = e.get("id")
        # 如果有 applied_ids，则只保留其中的；如果一个都没有，默认全过滤掉（gt 为空）
        if applied_ids and gid not in applied_ids:
            continue

        diff  = e.get("difficulty")
        sec   = e.get("location")
        errloc = e.get("target_find") or ""
        expl  = e.get("error_explanation") or ""
        xsec  = e.get("needs_cross_section", None)

        if not isinstance(errloc, str) or not errloc.strip():
            errloc = ""

        gt.append({
            # 先临时保留原始 id，后面统一重排
            "id": gid if isinstance(gid, int) else None,
            "difficulty": diff if isinstance(diff, str) else None,
            "section_location": sec.strip() if isinstance(sec, str) else None,
            "error_location": errloc.strip() if isinstance(errloc, str) else None,
            "explanation": expl.strip() if isinstance(expl, str) else None,
            "needs_cross_section": bool(xsec) if isinstance(xsec, bool) else None,
        })

    # 3) 对保留下来的 gt 根据当前顺序重新编号 id：1,2,3,...
    for new_id, item in enumerate(gt, start=1):
        item["id"] = new_id

    return gt


def _iter_unique_sections(outline_obj: Dict[str, Any]) -> List[str]:
    """从 outline 中按顺序列出去重后的 section 标题；若为空则返回空列表。"""
    titles = []
    for it in (outline_obj.get("outline") or []):
        t = (it.get("title") or "").strip()
        if t and t not in titles:
            titles.append(t)
    return titles

def add_findings_with_log(all_findings: List[Finding], new_findings: List[Finding], phase: str) -> None:
    """Append new findings to the global list and print progress summary."""
    if not new_findings:
        print(f"[DEBUG][{phase}] no new findings; total={len(all_findings)}")
        return
    before = len(all_findings)
    all_findings.extend(new_findings)
    added = len(all_findings) - before
    print(f"[DEBUG][{phase}] added {added} findings; total={len(all_findings)}")

def run_single(
    synth_json: Path,
    out_json: Path,
    debug_dir: Path,
    detect_model: str,
    synth_model: str, 
    enable_global_review: bool,
    global_max_findings: int,
    # —— 联网 —— #
    enable_web_search: bool,
    search_max_results: int = DEFAULT_SEARCH_MAX_RESULTS,
    temperature: float = DEFAULT_SEARCH_TEMPERATURE,
    # —— 记忆 —— #
    enable_memory_build: bool = DEFAULT_ENABLE_MEMORY_BUILD,
    memory_max_chars: int = MEMORY_MAX_CHARS_PER_TASK,
    # —— Section Review —— #
    enable_section_review: bool = DEFAULT_ENABLE_SECTION_REVIEW,
    enable_per_task: bool = DEFAULT_ENABLE_PER_TASK,
    # —— 新增 —— #
    enable_section_findings_as_prior: bool = DEFAULT_ENABLE_SECTION_FINDINGS_AS_PRIOR,
    enable_retriever: bool = DEFAULT_ENABLE_RETRIEVER,
    enable_memory_injection: bool = DEFAULT_ENABLE_MEMORY_INJECTION,
    # —— Merge 开关 —— #
    enable_merge: bool = DEFAULT_ENABLE_MERGE,
    enable_mm: bool = DEFAULT_ENABLE_MM
) -> Dict[str, Any]:
    debug_dir.mkdir(parents=True, exist_ok=True)
    obj = load_json(synth_json)
    gt_full = _extract_ground_truth_from_synth(obj)
    save_json(gt_full, debug_dir / "ground_truth.from_synth.json")
    paper = obj.get("paper", {}) or {}
    blocks = normalize_blocks(paper)
    save_json(blocks[:200], debug_dir / "paper.blocks.head.json")

    outline_obj = llm_build_outline(blocks, "", debug_dir, model=detect_model)

    all_findings: List[Finding] = []
    step_failures = []
    global_meta = {"enabled": enable_global_review}
    section_meta_overall = {"enabled": enable_section_review, "sections": []}
    web_stats = {
        "enabled": (enable_web_search and enable_retriever),
        "retriever_web_queries": 0,
        "retriever_web_evidence": 0
    }

    # 0) 全文记忆
    memory_obj = None
    memory_meta = {"enabled": enable_memory_build}
    if enable_memory_build:
        try:
            memory_obj, mmeta = build_paper_memory(
                blocks=cap_blocks_by_budget(blocks, MAX_CTX_CHARS),
                dbgdir=debug_dir,
                model=detect_model, 
                enable_mm=enable_mm
            )
            memory_meta.update(mmeta)
        except Exception as ex:
            memory_meta["error"] = f"{type(ex).__name__}: {ex}"
            step_failures.append("memory_build_crash")
    save_json(memory_meta, debug_dir / "memory.meta.json")

    # 1) 全局交叉检错
    if enable_global_review:
        try:
            global_findings, gmeta = global_cross_section_review(
                blocks=blocks, dbgdir=debug_dir, model=detect_model, global_max_findings=global_max_findings, enable_mm=enable_mm
            )
            global_meta.update(gmeta)
            add_findings_with_log(all_findings, global_findings, phase="global")
        except Exception:
            step_failures.append("global_review_crash")
    save_json(global_meta, debug_dir / "global_review.meta.json")

    # 2) Section 找错（Planner 之前，逐个 section）
    section_findings_map: Dict[str, List[Finding]] = {}
    if enable_section_review:
        section_titles = _iter_unique_sections(outline_obj)
        for stitle in section_titles:
            sdir = debug_dir / "section_reviews" / re.sub(r"[^A-Za-z0-9._-]+","_", stitle or "Unknown")
            sdir.mkdir(parents=True, exist_ok=True)
            try:
                slice_blocks = slice_json_for_task_with_outline(
                    blocks, stitle, outline_obj.get("outline", []), max_chars=MAX_CTX_CHARS
                )
                # ★ NEW: 基于当前 section 中的引用，从 References 注入对应参考文献
                slice_blocks = enrich_section_blocks_with_local_references(
                    section_title=stitle,
                    section_blocks=slice_blocks,
                    all_blocks=blocks,
                )

                save_json(slice_blocks, sdir / "section_slice.json")

                # 注入剔除当前段的记忆切片（仅用于 section review）
                mem_txt_sr = build_memory_for_task(
                    memory_obj=memory_obj if enable_memory_build else None,
                    current_section=stitle,
                    max_chars=memory_max_chars
                )

                (sdir / "section.memory.slice.txt").write_text(mem_txt_sr or "", encoding="utf-8")

                sec_findings, sec_meta = section_level_review(
                    section_title=stitle,
                    section_blocks=slice_blocks,
                    dbgdir=sdir,
                    model=detect_model,
                    memory_slice=mem_txt_sr, 
                    enable_mm=enable_mm
                )
                section_meta_overall["sections"].append({"title": stitle, **sec_meta})
                # 记录对应 section 的发现
                section_findings_map.setdefault(stitle, []).extend(sec_findings)
                add_findings_with_log(all_findings, sec_findings, phase=f"section:{stitle}")
            except Exception as ex:
                section_meta_overall["sections"].append({"title": stitle, "error": f"{type(ex).__name__}: {ex}"})
                step_failures.append(f"section_review_crash:{stitle}")
        save_json(section_meta_overall, debug_dir / "section_review.meta.json")
    else:
        save_json(section_meta_overall, debug_dir / "section_review.meta.json")

    
    
    # 4) Per-task pipeline
    per_task_summary = []
    if not enable_per_task:
        tasks = []
        plan_meta = None
        save_json(
            {"enabled": False, "reason": "disabled_by_flag"},
            debug_dir / "per_task.meta.json"
        )
    else:
        # 3) Planner（开放式规划；带空输出重试）
        try:
            tasks, plan_meta = planner_build_tasks_mm(
                blocks=blocks, outline_obj=outline_obj, dbgdir=debug_dir, model=detect_model, enable_mm=enable_mm
            )
        except Exception:
            step_failures.append("planner_no_tasks")
            raise

        save_json({"enabled": True}, debug_dir / "per_task.meta.json")
        for t in tasks:
            t_dir = debug_dir / f"task_{t.task_id}"
            t_dir.mkdir(parents=True, exist_ok=True)

            try:
                slice_blocks = slice_json_for_task_with_outline(
                    blocks, t.section, outline_obj.get("outline", []), max_chars=MAX_CTX_CHARS
                )
                # ★ NEW: 给每个 task 的 section 也注入本地引用
                slice_blocks = enrich_section_blocks_with_local_references(
                    section_title=t.section,
                    section_blocks=slice_blocks,
                    all_blocks=blocks,
                )

                save_json(slice_blocks, t_dir / "doc_slice.json")

                # 记忆切片注入（自然语言，剔除当前 section）— 受 enable_memory_injection 控制
                mem_txt = ""
                if enable_memory_build and enable_memory_injection:
                    mem_txt = build_memory_for_task(
                        memory_obj=memory_obj,
                        current_section=t.section,
                        max_chars=memory_max_chars
                    )
                (t_dir / "task.memory.slice.txt").write_text(mem_txt or "", encoding="utf-8")

                # 4.1 Retriever（可整体禁用）
                paper_evid: List[Evidence] = []
                web_queries: List[Dict[str, str]] = []
                web_evid:   List[Evidence] = []
                use_retriever = bool(enable_retriever)

                if use_retriever:
                    paper_evid, web_queries, rmeta = retriever_extract_and_questions(
                        task=t,
                        slice_blocks=slice_blocks,
                        dbgdir=t_dir,         # 任务级目录
                        model=detect_model,
                        paper_only=(not enable_web_search),
                        memory_slice=mem_txt, 
                        enable_mm=enable_mm
                    )
                    # 4.2 Web Search（仅当启用且有 queries）
                    if enable_web_search and web_queries:
                        web_evid = perform_web_search_for_queries(
                            web_queries=web_queries,
                            dbgdir=t_dir / "web_search",
                            detect_model=detect_model,
                            max_results=search_max_results,
                            temperature=temperature
                        )
                        web_stats["retriever_web_queries"] += len(web_queries)
                        web_stats["retriever_web_evidence"] += len(web_evid)

                evid_final = paper_evid + web_evid

                # 4.3 prior findings（仅同名 section）
                priors_for_section: List[Finding] = []
                if enable_section_findings_as_prior:
                    priors_for_section = list(section_findings_map.get(t.section, []))

                # 4.4 Specialist
                cand, smeta = specialist_review(
                    task=t,
                    evid=evid_final,
                    neighbor_blocks=slice_blocks,
                    dbgdir=t_dir,
                    model=detect_model,
                    # “paper_only” 现在语义：无 web 证据或未使用 retriever
                    paper_only=(not enable_web_search) or (not use_retriever),
                    memory_slice=mem_txt,
                    prior_findings=priors_for_section,
                    use_retriever=use_retriever,
                    enable_mm=enable_mm
                )

                per_task_summary.append({
                    "task_id": t.task_id,
                    "risk": t.risk_dimension,
                    "used_retriever": bool(use_retriever),
                    "used_prior_findings": bool(enable_section_findings_as_prior and priors_for_section),
                    "used_memory": bool(mem_txt),
                    "evidence": len(evid_final),
                    "paper_evidence": len(paper_evid),
                    "web_queries": len(web_queries),
                    "web_evidence": len(web_evid),
                    "findings": len(cand)
                })
                add_findings_with_log(all_findings, cand, phase=f"specialist:{t.task_id}")
            except Exception as ex:
                step_failures.append(f"task_crash:{t.task_id}")
                save_json({"error": f"{type(ex).__name__}: {ex}", "trace": traceback.format_exc()}, t_dir / "task.error.json")

        save_json(per_task_summary, debug_dir / "per_task.summary.json")

    # 5) 合并与裁决（受 enable_merge 控制）
    if enable_merge:
        merged = merge_and_adjudicate(all_findings, debug_dir, model=detect_model)
    else:
        # 跳过合并，直接保留所有 findings（按当前顺序）
        merged = list(all_findings)
        save_json(
            {"note": "merge disabled; keeping all raw findings without adjudication."},
            debug_dir / "merge.disabled.json"
        )

    # 6) 汇总落盘
    out_obj = {
        "eval_for_detector": True,
        "created_at": datetime.utcnow().isoformat() + "Z",
        "source_synth_file": str(synth_json),
        "detect_model_used": detect_model,
        "synth_model": synth_model,
        "ground_truth": gt_full,
        "findings": [asdict(f) for f in merged],
        "stats": {
            "tasks_total": len(tasks),
            "findings_raw": len(all_findings),
            "findings_merged": len(merged),
            "step_failures": step_failures,
            "planner_meta": plan_meta,
            "global_meta": global_meta,
            "section_meta": section_meta_overall,
            "outline_used": outline_obj.get("used"),
            "outline_items": len(outline_obj.get("outline", [])),
            "max_ctx_chars": MAX_CTX_CHARS,
            "enable_global_review": enable_global_review,
            "enable_section_review": enable_section_review,
            "global_max_findings": global_max_findings,
            # —— web stats —— #
            "web_search": web_stats,
            # —— memory stats —— #
            "memory": {
                "enabled": enable_memory_build,
                "per_task_chars_limit": memory_max_chars,
                "built": bool(memory_obj)
            },
            # —— 新增 flags —— #
            "flags": {
                "enable_section_findings_as_prior": enable_section_findings_as_prior,
                "enable_retriever": enable_retriever,
                "enable_memory_injection": enable_memory_injection,
                "enable_merge": enable_merge,
            }
        }
    }

    # ------------- ground truth output (save next to mas_detect.json) -------------
    # try:
    #     gt = gt_full   # _extract_ground_truth_from_synth already computed this
    #     gt_path = out_json.parent / "ground_truth.json"
    #     save_json(gt, gt_path)
    # except Exception as ex:
    #     print(f"[WARN] failed to write ground_truth.json: {ex}")

    # ------------- renumber findings (1..N in output) -------------
    try:
        for idx, f in enumerate(out_obj["findings"], start=1):
            f["id"] = idx
    except Exception as ex:
        print(f"[WARN] failed renumber findings: {ex}")

    save_json(out_obj, out_json)
    return {"ok": True, "out_json": str(out_json), "debug_dir": str(debug_dir)}

# =========================
# 批处理工具
# =========================
def compute_output_paths(
    synth_path: Path,
    detect_mode: str,
    detect_model: str,
) -> Tuple[Path, Path]:
    paper_dir = synth_path.parent
    out_json = paper_dir / f"{detect_mode}_detect" / detect_model / synth_path.stem / f"{detect_mode}_detect.json"
    debug_dir = paper_dir / f"{detect_mode}_detect_log" / detect_model / synth_path.stem
    return out_json, debug_dir

def find_synth_files(root_dir: Path, synth_glob: str) -> List[Path]:
    return sorted(root_dir.rglob(synth_glob))

def process_one_job(
    synth_path: Path,
    detect_model: str,
    detect_mode: str,
    synth_model: str,
    enable_global_review: bool,
    global_max_findings: int,
    overwrite: bool = False,
    overwrite_zero: bool = False, 
    # —— 联网 —— #
    enable_web_search: bool = DEFAULT_ENABLE_WEB_SEARCH,
    search_max_results: int = DEFAULT_SEARCH_MAX_RESULTS,
    temperature: float = DEFAULT_SEARCH_TEMPERATURE,
    # —— 记忆 —— #
    enable_memory_build: bool = DEFAULT_ENABLE_MEMORY_BUILD,
    memory_max_chars: int = MEMORY_MAX_CHARS_PER_TASK,
    # —— Section Review —— #
    enable_section_review: bool = DEFAULT_ENABLE_SECTION_REVIEW,
    enable_per_task: bool = DEFAULT_ENABLE_PER_TASK,
    # —— 新增 —— #
    enable_section_findings_as_prior: bool = DEFAULT_ENABLE_SECTION_FINDINGS_AS_PRIOR,
    enable_retriever: bool = DEFAULT_ENABLE_RETRIEVER,
    enable_memory_injection: bool = DEFAULT_ENABLE_MEMORY_INJECTION,
    # —— Merge —— #
    enable_merge: bool = DEFAULT_ENABLE_MERGE,
    enable_mm: bool = DEFAULT_ENABLE_MM
) -> Dict[str, Any]:
    try:
        out_json, debug_dir = compute_output_paths(synth_path, detect_mode, detect_model)
        inprogress = out_json.with_suffix(out_json.suffix + ".inprogress")
        if out_json.exists() and not overwrite:
            if overwrite_zero and output_has_zero_findings(out_json):
                print(f"[BATCH] overwrite_zero=1 and findings==[] -> rerun: {out_json}")
            else:
                return {"synth": str(synth_path), "skipped": True, "reason": "exists", "out_json": str(out_json)}

        inprogress.parent.mkdir(parents=True, exist_ok=True)
        try: inprogress.touch(exist_ok=True)
        except Exception: pass
        result = run_single(
            synth_json=synth_path,
            out_json=out_json,
            debug_dir=debug_dir,
            detect_model=detect_model,
            synth_model=synth_model,
            enable_global_review=enable_global_review,
            global_max_findings=global_max_findings,
            enable_web_search=(enable_web_search and enable_retriever),
            search_max_results=search_max_results,
            temperature=temperature,
            enable_memory_build=enable_memory_build,
            memory_max_chars=memory_max_chars,
            enable_section_review=enable_section_review,
            enable_per_task=enable_per_task,
            enable_section_findings_as_prior=enable_section_findings_as_prior,
            enable_retriever=enable_retriever,
            enable_memory_injection=enable_memory_injection,
            enable_merge=enable_merge,
            enable_mm=enable_mm
        )
        return {"synth": str(synth_path), "ok": True, "out_json": str(out_json), "debug_dir": str(debug_dir)}
    except Exception as e:
        return {"synth": str(synth_path), "ok": False, "error": f"{type(e).__name__}: {e}"}
    finally:
        try:
            if inprogress.exists(): inprogress.unlink()
        except Exception: pass

def run_batch(
    root_dir: Path,
    synth_glob: str,
    detect_model: str,
    synth_model: str,
    detect_mode: str,
    enable_global_review: bool,
    global_max_findings: int,
    jobs: int = 6,
    overwrite: bool = False,
    overwrite_zero: bool = False,
    max_papers: Optional[int] = None,
    # —— 联网 —— #
    enable_web_search: bool = DEFAULT_ENABLE_WEB_SEARCH,
    search_max_results: int = DEFAULT_SEARCH_MAX_RESULTS,
    temperature: float = DEFAULT_SEARCH_TEMPERATURE,
    # —— 记忆 —— #
    enable_memory_build: bool = DEFAULT_ENABLE_MEMORY_BUILD,
    memory_max_chars: int = MEMORY_MAX_CHARS_PER_TASK,
    # —— Section Review —— #
    enable_section_review: bool = DEFAULT_ENABLE_SECTION_REVIEW,
    enable_per_task: bool = DEFAULT_ENABLE_PER_TASK,
    # —— 新增 —— #
    enable_section_findings_as_prior: bool = DEFAULT_ENABLE_SECTION_FINDINGS_AS_PRIOR,
    enable_retriever: bool = DEFAULT_ENABLE_RETRIEVER,
    enable_memory_injection: bool = DEFAULT_ENABLE_MEMORY_INJECTION,
    # —— Merge —— #
    enable_merge: bool = DEFAULT_ENABLE_MERGE,
    enable_mm: bool = DEFAULT_ENABLE_MM
) -> Dict[str, Any]:

    root_dir = Path(root_dir)

    # 1) 先按 synth_glob 找到所有 paper_synth_*.json
    synth_files = find_synth_files(root_dir, synth_glob)

    # 2) 再用「文件名是否包含 synth_model」来做过滤
    #    比如 synth_model='gpt-5-2025-08-07'，
    #    就只保留名字里带这个子串的 json
    if synth_model:
        before = len(synth_files)
        synth_files = [p for p in synth_files if synth_model in p.name]
        after = len(synth_files)
        print(f"[BATCH] synth_model filter: target='{synth_model}', matched={after}/{before}")

    # 3) 再按 max_papers 截断（进度条上限就是过滤后的数量）
    if max_papers is not None:
        synth_files = synth_files[:max_papers]

    # 4) 若最终为空，直接返回错误
    if not synth_files:
        return {
            "ok": False,
            "error": (
                f"No synth files matched '{synth_glob}' "
                f"with synth_model='{synth_model}' under {root_dir}"
            ),
        }

    results = []
    bar = tqdm(total=len(synth_files), desc="MAS-Batch") if tqdm else None
    with ThreadPoolExecutor(max_workers=max(1, jobs)) as ex:
        futs = []
        for sp in synth_files:
            fut = ex.submit(
                process_one_job,
                sp, detect_model, detect_mode, synth_model,
                enable_global_review, global_max_findings,
                overwrite,
                overwrite_zero,
                enable_web_search, search_max_results, temperature,
                enable_memory_build, memory_max_chars,
                enable_section_review,
                enable_per_task,
                enable_section_findings_as_prior,
                enable_retriever,
                enable_memory_injection,
                enable_merge,
                enable_mm
            )
            futs.append(fut)
        for fut in as_completed(futs):
            res = fut.result()
            results.append(res)
            if bar: bar.update(1)
    if bar: bar.close()
    ok = sum(1 for r in results if r.get("ok"))
    skipped = sum(1 for r in results if r.get("skipped"))
    failed = [r for r in results if not r.get("ok") and not r.get("skipped")]
    summary = {
        "ok": True,
        "detect_model": detect_model,
        "detect_mode": detect_mode,
        "synth_model": synth_model,
        "total": len(results),
        "done": ok,
        "skipped": skipped,
        "failed": len(failed),
        "fail_details": failed[:50],
    }
    return summary

# =========================
# CLI
# =========================
def main():
    parser = argparse.ArgumentParser(
        description="Single/Batch MAS error-review evaluator (open-ended, unified prompts, multimodal, natural-language memory, global review, section-level review, threaded batch, two-step web search, retry-on-empty for ALL stages)."
    )
    # --- 单文件模式 ---
    parser.add_argument("--synth_json", type=str, default=None,
                        help="Path to a single synth JSON. If set, runs single-file mode unless --root_dir is provided.")
    parser.add_argument("--out_json",   type=str, default=None,
                        help="(Single mode) Explicit output json path. If omitted, and --detect_mode is set, will auto-place into {detect_mode}_detect/...")
    parser.add_argument("--debug_dir",  type=str, default="None",
                        help="(Single mode) Explicit debug dir. If omitted, and --detect_mode is set, will auto-place into {detect_mode}_detect_log/...")

    # --- 批处理模式 ---
    parser.add_argument("--root_dir", type=str, default="/mnt/parallel_ssd/home/zdhs0006/ACL/test2", help="Batch root dir; if set, runs threaded batch over synth_glob.")
    parser.add_argument("--synth_glob", type=str, default="paper_synth_*.json",
                        help="Glob to find synth JSONs under root_dir (default: paper_synth_*.json)")
    parser.add_argument("--jobs", type=int, default=8, help="Thread pool size (default 8)")
    parser.add_argument("--overwrite", action="store_true", default=False, help="Force overwrite existing outputs")
    parser.add_argument("--overwrite_zero", action="store_true", default=False,
                    help="If output exists but has empty findings=[], rerun even when not overwriting.")
    parser.add_argument("--max_papers", type=int, default=None, help="Limit the number of synth files to process")

    # --- 通用 ---
    parser.add_argument("--detect_model", type=str, default=DEFAULT_DETECT_MODEL)
    parser.add_argument("--detect_mode", type=str, default="section",
                        help="Detect mode name used in output paths")
    parser.add_argument("--synth_model", type=str, default=DEFAULT_SYNTH_MODEL,
                        help="Tag/name of the synthesis model whose data you are evaluating; "
                             "used for output dir grouping and stats.")

    # —— 整体交叉检错 —— #
    parser.add_argument("--disable_global_review", action="store_true", default=False,
                        help=f"Disable global cross-section review (default enabled={DEFAULT_ENABLE_GLOBAL_REVIEW})")
    parser.add_argument("--global_max_findings", type=int, default=DEFAULT_GLOBAL_MAX_FINDINGS,
                        help="Soft cap for global review (for safety); reviewer is instructed to be exhaustive without hard caps.")

    # —— Section Review —— #
    parser.add_argument("--disable_section_review", action="store_true", default=False, 
                        help=f"Disable section-level review (default enabled={DEFAULT_ENABLE_SECTION_REVIEW})")

    # —— Per-task Pipeline —— #
    parser.add_argument("--disable_per_task", action="store_true", default=False, 
                        help=f"Disable per-task pipeline (default enabled={DEFAULT_ENABLE_PER_TASK})")
    # 是否把 section findings 传给 Specialist（只传同名 section）
    parser.add_argument("--use_section_findings_as_prior", action="store_true",
                        default=DEFAULT_ENABLE_SECTION_FINDINGS_AS_PRIOR,
                        help="Pass section_review findings of the SAME section to each task's Specialist as prior support.")
    # 总开关：禁用 Retriever（含 WebSearch）
    parser.add_argument("--disable_retriever", action="store_true", default=False,
                        help="Disable Retriever (+WebSearch). Specialist will rely on doc slice (+ optional memory) + prior section findings.")
    # 可选：禁用记忆注入（即使已构建 memory 也不往 Specialist 传）
    parser.add_argument("--disable_memory_injection", action="store_true", default=False,
                        help="Do not inject memory into Specialist (even if memory is built).")

    # —— Merge —— #
    parser.add_argument("--disable_merge", action="store_true", default=False,
                        help=f"Disable merging/adjudication of findings (default enabled={DEFAULT_ENABLE_MERGE})")

    # —— 联网搜索 —— #
    parser.add_argument("--enable_web_search", action="store_true", default=DEFAULT_ENABLE_WEB_SEARCH,
                        help=f"Enable dedicated web_search step after Retriever (default {DEFAULT_ENABLE_WEB_SEARCH})")
    parser.add_argument("--search_max_results", type=int, default=DEFAULT_SEARCH_MAX_RESULTS,
                        help="Technical cap to avoid directory explosion; not a quota for retrieval quality.")
    parser.add_argument("--search_temperature", type=float, default=DEFAULT_SEARCH_TEMPERATURE)

    # —— Memory —— #
    parser.add_argument("--disable_memory_build", action="store_true", default=False,
                    help=f"Disable full-paper memory build (default enabled={DEFAULT_ENABLE_MEMORY_BUILD})")
    parser.add_argument("--memory_max_chars", type=int, default=MEMORY_MAX_CHARS_PER_TASK,
                        help="Max characters of memory injected per task (natural-language slice)")

    parser.add_argument("--disable_mm", action="store_true", default=False,
                    help=f"Disable multimodal input (images). Default enabled={DEFAULT_ENABLE_MM}")

    # （温度通过全局变量/环境变量注入；如需 CLI，也可加对应 --temp_* 参数）
    args = parser.parse_args()

    if not OPENAI_API_KEY:
        print("[FATAL] OPENAI_API_KEY is not set.", file=sys.stderr); sys.exit(1)

    # 统一计算 enable flags（只算一次）
    enable_web      = args.enable_web_search
    enable_memory   = (not args.disable_memory_build)  and DEFAULT_ENABLE_MEMORY_BUILD
    enable_global   = (not args.disable_global_review) and DEFAULT_ENABLE_GLOBAL_REVIEW
    enable_section  = (not args.disable_section_review) and DEFAULT_ENABLE_SECTION_REVIEW
    enable_per_task = (not args.disable_per_task)      and DEFAULT_ENABLE_PER_TASK
    enable_merge    = (not args.disable_merge)         and DEFAULT_ENABLE_MERGE
    # enable input
    enable_mm    = (not args.disable_mm)         and DEFAULT_ENABLE_MM

    # —— 三个你指定也按一致风格计算 —— #
    enable_section_findings_as_prior = args.use_section_findings_as_prior
    enable_retriever                 = (not args.disable_retriever)        and DEFAULT_ENABLE_RETRIEVER
    enable_memory_injection          = (not args.disable_memory_injection) and DEFAULT_ENABLE_MEMORY_INJECTION

    # 禁用 Retriever 时，强制关闭 WebSearch
    if not enable_retriever:
        enable_web = False

    # ========== 批处理模式 ==========
    if args.root_dir:
        root_dir = Path(args.root_dir).expanduser().resolve()
        if not root_dir.exists():
            print(f"[FATAL] root_dir not found: {root_dir}", file=sys.stderr); sys.exit(1)
        print(f"[BATCH] root_dir={root_dir}")
        print(f"[BATCH] synth_glob={args.synth_glob}  jobs={args.jobs}  overwrite={args.overwrite} overwrite_zero={args.overwrite_zero}")
        print(f"[BATCH] detect_mode={args.detect_mode}  detect_model={args.detect_model}")
        print(f"[BATCH] synth_model={args.synth_model}")
        print(f"[BATCH] enable_global_review={enable_global}  global_max_findings={args.global_max_findings}")
        print(f"[BATCH] enable_section_review={enable_section}")
        print(f"[BATCH] enable_web_search={enable_web}  search_k={args.search_max_results}  temp_web={args.search_temperature}")
        print(f"[BATCH] enable_memory_build={enable_memory}  memory_max_chars={args.memory_max_chars}")
        print(f"[INFO]  enable_per_task={enable_per_task}")
        print(f"[INFO]  enable_section_findings_as_prior={enable_section_findings_as_prior}")
        print(f"[INFO]  enable_retriever={enable_retriever}")
        print(f"[INFO]  enable_memory_injection={enable_memory_injection}")
        print(f"[INFO]  enable_merge={enable_merge}")
        print(f"[INFO]  temps: memory={TEMP_MEMORY} planner={TEMP_PLANNER} retriever={TEMP_RETRIEVER} specialist={TEMP_SPECIALIST} global={TEMP_GLOBAL_REVIEW} section={TEMP_SECTION_REVIEW} merger={TEMP_MERGER}")
        print(f"[INFO]  enable_mm={enable_mm}")

        summary = run_batch(
            root_dir=root_dir,
            synth_glob=args.synth_glob,
            detect_model=args.detect_model,
            synth_model=args.synth_model,
            detect_mode=args.detect_mode,
            enable_global_review=enable_global,
            global_max_findings=args.global_max_findings,
            jobs=args.jobs,
            overwrite=args.overwrite,
            overwrite_zero=args.overwrite_zero,
            max_papers=args.max_papers,
            enable_web_search=enable_web,
            search_max_results=args.search_max_results,
            temperature=args.search_temperature,
            enable_memory_build=enable_memory,
            memory_max_chars=args.memory_max_chars,
            enable_section_review=enable_section,
            enable_per_task=enable_per_task,
            enable_section_findings_as_prior=enable_section_findings_as_prior,
            enable_retriever=enable_retriever,
            enable_memory_injection=enable_memory_injection,
            enable_merge=enable_merge,
        )
        print(json.dumps(summary, ensure_ascii=False, indent=2))
        return

    # ========== 单文件模式 ==========
    if not args.synth_json:
        print("[FATAL] Missing --synth_json or --root_dir.", file=sys.stderr); sys.exit(1)

    synth_json = Path(args.synth_json).expanduser().resolve()
    if not synth_json.exists():
        print(f"[FATAL] synth_json not found: {synth_json}", file=sys.stderr); sys.exit(1)

    if args.out_json:
        out_json = Path(args.out_json).expanduser().resolve()
    else:
        # 注意：compute_output_paths 只依赖 detect_mode & detect_model，不依赖 synth_model（保持原有存储结构）
        out_json, _auto_debug = compute_output_paths(synth_json, args.detect_mode, args.detect_model)

    if args.debug_dir and args.debug_dir != "None":
        debug_dir = Path(args.debug_dir).expanduser().resolve()
    else:
        _auto_out, debug_dir = compute_output_paths(synth_json, args.detect_mode, args.detect_model)


    print(f"[INFO] synth_json={synth_json}")
    print(f"[INFO] out_json={out_json}")
    print(f"[INFO] debug_dir={debug_dir}")
    print(f"[INFO] detect_model={args.detect_model}  detect_mode={args.detect_mode}")
    print(f"[INFO] synth_model={args.synth_model}")
    print(f"[INFO] MAX_CTX_CHARS={MAX_CTX_CHARS}")
    print(f"[INFO] enable_global_review={enable_global}  global_max_findings={args.global_max_findings}")
    print(f"[INFO] enable_section_review={enable_section}")
    print(f"[INFO] enable_web_search={enable_web}  search_k={args.search_max_results}  temp_web={args.search_temperature}")
    print(f"[INFO] enable_memory_build={enable_memory}  memory_max_chars={args.memory_max_chars}")
    print(f"[INFO] enable_per_task={enable_per_task}")
    print(f"[INFO] enable_section_findings_as_prior={enable_section_findings_as_prior}")
    print(f"[INFO] enable_retriever={enable_retriever}")
    print(f"[INFO] enable_memory_injection={enable_memory_injection}")
    print(f"[INFO] enable_merge={enable_merge}")
    print(f"[INFO] temps: memory={TEMP_MEMORY} planner={TEMP_PLANNER} retriever={TEMP_RETRIEVER} specialist={TEMP_SPECIALIST} global={TEMP_GLOBAL_REVIEW} section={TEMP_SECTION_REVIEW} merger={TEMP_MERGER}")

    summary = run_single(
        synth_json, out_json, debug_dir,
        detect_model=args.detect_model,
        synth_model=args.synth_model,
        enable_global_review=enable_global,
        global_max_findings=args.global_max_findings,
        enable_web_search=enable_web,
        search_max_results=args.search_max_results,
        temperature=args.search_temperature,
        enable_memory_build=enable_memory,
        memory_max_chars=args.memory_max_chars,
        enable_section_review=enable_section,
        enable_per_task=enable_per_task,
        enable_section_findings_as_prior=enable_section_findings_as_prior,
        enable_retriever=enable_retriever,
        enable_memory_injection=enable_memory_injection,
        enable_merge=enable_merge,
    )
    print("[DONE]", json.dumps(summary, ensure_ascii=False))

if __name__ == "__main__":
    main()
