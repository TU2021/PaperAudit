#!/usr/bin/env python3
# -*- coding: utf-8 -*-

"""
batch_label_sections.py

批量处理流程（每个输入 JSON）：
1) 对每个 content 中的 text，调用 LLM 按“基于章节边界”的规则切分为更清晰的段落（以 \\n\\n\\n\\n 分隔）
2) 将切分后的 text 段落与原始的图片/其它块按原来顺序融合为新的 content，保存为 preseg JSON（可选）
3) 在新的 content 上执行章节标注，为每个 content 写入 index(1-based)/section，并在根部附上 section_labels

特性：
- 并发执行（asyncio + 线程池）
- 断点续传（最终输出已存在或 .inprogress 则跳过）
- 指数回退重试
- 分割与标注均使用 OpenAI Chat Completions，提示词可调

用法示例：
python batch_label_sections.py \
  --root_dir downloads/test2/appendix_root \
  --input_name paper_parse.json \
  --output_name section_paper_parse.json \
  --save_preseg \
  -j 6
"""

import os, json, re, sys, time, asyncio
from pathlib import Path
from typing import List, Dict, Any, Optional, Tuple, Set
from dataclasses import dataclass

from tqdm import tqdm
from openai import OpenAI
from utils import extract_json_from_text, load_json, save_json

# ================= 全局配置 =================
MODEL = "gpt-5-2025-08-07"
OPENAI_API_KEY = os.getenv("OPENAI_API_KEY")
LLM_MAX_RETRIES = 3
MAX_BLOCKS = 1200  # 标注阶段送入 LLM 的最大块数
MAX_TOKENS = 40000

ALLOWED_SECTIONS = [
    "Abstract", "Introduction", "Motivation", "Preliminaries",
    "Related Work", "Method", "Experiments", "Conclusion",
    "References", "Appendix", "Checklist"
]

# —— 步骤 A：文本切分提示（基于你给的 OCR_SYS_PROMPT 语义，改成“原始 text”场景）——
SEG_SYS_PROMPT = (
    "You will be given raw text paragraphs from a research paper. "
    "Please correct obvious OCR-like issues (e.g., hyphenated line-breaks) and normalize whitespace, "
    "while preserving math/LaTeX. Then split the text into clear blocks based on sections: "
    "blocks should follow section boundaries (e.g., a heading and its following paragraphs belong to one block) "
    "and MUST NOT arbitrarily split text that belongs to the same section. "
    "Each block MUST be separated by four newlines (\\n\\n\\n\\n). "
    "Return ONLY the corrected text with block breaks — no explanations and no JSON."
)

# —— 步骤 B：章节标注提示（与你的原版一致）——
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

# ================= 工具函数 =================
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
def get_client():
    global _client
    if _client is None:
        _client = OpenAI(api_key=OPENAI_API_KEY)
    return _client

def call_llm(messages, model=MODEL, max_tokens=MAX_TOKENS):
    client = get_client()
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

# ================= 步骤 A：文本切分 =================
def split_text_block_via_llm(text: str) -> List[str]:
    """
    将单个 text 内容送入 LLM，按段落/章节边界切分；返回段落列表。
    """
    if not text or not text.strip():
        return []
    messages = [
        {"role": "system", "content": SEG_SYS_PROMPT},
        {"role": "user", "content": text.strip()},
    ]
    raw = call_llm(messages)
    if not raw:
        # 回退：切不动就原样返回（避免中断流程）
        return [text.strip()]
    # 以四个换行作为边界（严格按提示）
    parts = [p.strip() for p in raw.split("\n\n\n\n") if p.strip()]
    return parts if parts else [text.strip()]

def build_presegmented_content(original_content: List[Dict[str, Any]]) -> List[Dict[str, Any]]:
    """
    遍历原 content：
    - 对 type=='text' 的块调用 LLM 切分并展开为多个 text 块
    - 其它类型（image_url/figure/table/other）原样穿插回去（保持原顺序）
    返回新的 content 列表
    """
    new_content: List[Dict[str, Any]] = []
    for item in original_content:
        btype = item.get("type") or "other"
        if btype == "text":
            text = item.get("text") or ""
            pieces = split_text_block_via_llm(text)
            for para in pieces:
                new_content.append({"type": "text", "text": para})
        else:
            new_content.append(dict(item))  # 保留图片与其它块
    return new_content

# ================= 步骤 B：章节标注（与你原来的核心一致） =================
def label_one_document_core_from_content(content_blocks: List[Dict[str, Any]]) -> Dict[str, Any]:
    blocks = content_blocks
    if len(blocks) == 0:
        return {
            "content": [],
            "section_labels": {"labels": [], "model_used": MODEL}
        }

    use_blocks = blocks
    if len(use_blocks) > MAX_BLOCKS:
        print(f"[WARN] too many blocks ({len(use_blocks)}), truncating to {MAX_BLOCKS}")
        use_blocks = use_blocks[:MAX_BLOCKS]

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

    raw = call_llm(messages)
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
        "section_labels": {"labels": labels, "model_used": MODEL}
    }

# ================= 批处理封装 =================
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
    max_retries: int = 4,
    backoff_base: float = 1.6,
) -> TaskResult:
    """
    单文件处理：text 切分 → 合成 preseg content（可保存）→ 章节标注 → 最终输出
    断点续传：若最终输出已存在或 .inprogress 存在则跳过
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
                # A. 文本切分（放到线程池，避免阻塞）
                preseg_content = await asyncio.to_thread(
                    build_presegmented_content, original_content
                )

                # 可选保存 preseg JSON（不影响最终跳过逻辑）
                if save_preseg:
                    preseg_path = output_path.parent / (
                        (preseg_name or f"preseg_{output_path.name}")
                    )
                    preseg_obj = dict(data)
                    preseg_obj["content"] = preseg_content
                    save_json(preseg_obj, preseg_path)

                # B. 章节标注
                labeled = await asyncio.to_thread(
                    label_one_document_core_from_content, preseg_content
                )

                # 合并到最终输出（保留原根部字段，替换 content + 附加 section_labels）
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
    parser = argparse.ArgumentParser(description="Batch split text blocks and section labeling for paper JSONs.")
    parser.add_argument("--root_dir", type=str, default="downloads/ICLR_2025_TEST",
                        help="根目录，递归查找 input_name JSON")
    parser.add_argument("--input_name", type=str, default="paper_parse_origin.json",
                        help="要作为输入的 JSON 文件名（默认 paper_parse_origin.json）")
    parser.add_argument("--output_name", type=str, default="paper_parse_add_section.json",
                        help="最终输出 JSON 文件名（默认 paper_parse.json）")
    parser.add_argument("--concurrency", "-j", type=int, default=20,
                        help="并发数（默认 4）")
    parser.add_argument("--model", type=str, default=MODEL,
                        help=f"LLM 模型名（默认 {MODEL}）")
    parser.add_argument("--max_blocks", type=int, default=MAX_BLOCKS,
                        help=f"章节标注阶段最多送入的块数（默认 {MAX_BLOCKS}）")
    parser.add_argument("--save_preseg", action="store_true",
                        help="保存切分后的中间 JSON（默认不保存）")
    parser.add_argument("--preseg_name", type=str, default=None,
                        help="中间 JSON 文件名（默认 preseg_<output_name>）")
    args = parser.parse_args()

    if not OPENAI_API_KEY:
        print("[FATAL] OPENAI_API_KEY not set.")
        sys.exit(1)

    root = Path(args.root_dir).expanduser().resolve()
    if not root.exists():
        raise FileNotFoundError(f"Root dir not found: {root}")

    print(f"[INFO] root={root}")
    print(f"[INFO] input_name={args.input_name}  output_name={args.output_name}")
    print(f"[INFO] concurrency={args.concurrency}  model={MODEL}  max_blocks={MAX_BLOCKS}")
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
        )
    )
    print(f"[DONE] total={summary['total']}  done={summary['done']}")
    if summary["failed"]:
        print("[FAILED] some tasks failed:")
        for i, err in enumerate(summary["failed"], 1):
            print(f"  {i}. {err}")

if __name__ == "__main__":
    main()
