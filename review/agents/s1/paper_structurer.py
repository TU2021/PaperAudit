from typing import List, Dict, Any
import re
from ..base_agent import BaseAgent
from ..logger import get_logger

logger = get_logger(__name__)

class PaperStructurer(BaseAgent):
    def __init__(self, **kwargs):
        super().__init__(**kwargs)

    """
    用 LLM 对原始 PDF 文本做“结构化切片”，但不让 LLM 重写正文：
    - 只让 LLM 找出若干个大的章节块（Part 1, Part 2, ...）的“起点位置”
    - 每个块包含：
        * 一个简短标题（title）
        * 该块在原文中的第一句话（anchor_sentence），必须与原文逐字一致
    - 然后在本地根据 anchor_sentence 在原始 pdf_text 中做字符串匹配，
      用这些 anchor 把全文切成连续的若干大块。

    最终返回一个 sections 列表，每个元素：
    - title: 章节标题（如 "Part 1: Introduction and Motivation"）
    - content: 该章节在原文中的原始片段（未被 LLM 改写）
    """

    # 🔢 最多几大块（可按需调整）
    MAX_SECTIONS = 6

    # 现在的 SYSTEM_PROMPT：让模型只输出“切片索引 + 锚点句子”，不重写正文
    SYSTEM_PROMPT = (
        "You are a meticulous segmenter for scientific papers. "
        "Your job is NOT to rewrite the paper, but ONLY to identify a small number "
        "of major contiguous segments (high-level parts) in the manuscript.\n\n"
        "You will see the full (possibly noisy) plain-text manuscript. "
        "Your task is to propose AT MOST {max_parts} major parts that together cover "
        "the whole paper from beginning to end.\n\n"
        "CRITICAL: You MUST NOT rewrite, summarize, or modify the manuscript text. "
        "You ONLY choose segmentation points.\n\n"
        "For each part, you must output:\n"
        "- a short human-readable title describing that part, and\n"
        "- the EXACT FIRST SENTENCE of that part, copied VERBATIM from the manuscript.\n\n"
        "The first sentence (anchor) must:\n"
        "- **be copied exactly as it appears in the RAW MANUSCRIPT** (same wording, same order),\n"
        "- be uniquely identifiable (avoid very short or generic phrases),\n"
        "- correspond to the first sentence of that segment.\n\n"
        "You **MUST NOT paraphrase or clean the anchor sentence.** "
        "Do NOT fix grammar, do NOT delete words, do NOT change line breaks inside it. "
        "We will use this anchor sentence to locate the segment in the original text "
        "via exact string matching.\n\n"
        "================ OUTPUT FORMAT (STRICT) ================\n"
        "Output ONE line per part, in order from the beginning of the paper to the end.\n"
        "Each line MUST follow this exact pattern:\n"
        "    Part <k> | <short title> | <EXACT first sentence of this part>\n"
        "Where:\n"
        "    - <k> is 1, 2, 3, ... (no gaps, strictly increasing)\n"
        "    - <short title> is a brief descriptor (e.g., 'Introduction and Motivation')\n"
        "    - <EXACT first sentence> is copied verbatim from the manuscript.\n\n"
        "Do NOT output anything else: no explanations, no JSON, no bullet points.\n"
        "If you think fewer parts are sufficient, use fewer. "
        "NEVER exceed {max_parts} parts.\n"
    )

    USER_PROMPT_TEMPLATE = (
        "Here is the full manuscript extracted from a PDF (with possible noise such as page headers, "
        "footers, and line breaks).\n"
        "Your job is ONLY to propose at most {max_parts} major segments.\n\n"
        "Important:\n"
        "- Do NOT rewrite or summarize the text.\n"
        "- Do NOT clean or modify the anchor sentences.\n"
        "- The anchor sentences MUST be copied verbatim from the RAW MANUSCRIPT below.\n\n"
        "Again, for each part, output exactly one line in this format:\n"
        "    Part <k> | <short title> | <EXACT first sentence of this part>\n\n"
        "=== BEGIN RAW MANUSCRIPT ===\n"
        "{text}\n"
        "=== END RAW MANUSCRIPT ===\n"
    )

    @staticmethod
    def _extract_text_from_message_content(raw_content: Any) -> str:
        """
        兼容 OpenAI 风格的 message.content：
        - 可能是 str
        - 也可能是 list[TextPart] 或 list[dict(text=...)]
        """
        if isinstance(raw_content, str):
            return raw_content

        if isinstance(raw_content, list):
            parts: List[str] = []
            for part in raw_content:
                if isinstance(part, dict):
                    t = part.get("text") or part.get("content")
                    if t:
                        parts.append(str(t))
                else:
                    t = getattr(part, "text", None) or getattr(part, "content", None)
                    if t:
                        parts.append(str(t))
            return "".join(parts)

        return str(raw_content)

    async def run(self, pdf_text: str) -> List[Dict[str, Any]]:
        """
        新逻辑：
        1. 调用 LLM，请它输出若干行：
              Part k | <title> | <EXACT first sentence>
        2. 解析这些行，得到 (k, title, anchor_sentence) 列表
        3. 在原始 pdf_text 中用 anchor_sentence 做字符串查找，得到每个块的起始位置
        4. 按起始位置排序，把 pdf_text 切成连续的若干大段
        5. 返回 sections 列表，每段完全来自原始 pdf_text（不经 LLM 改写）
        """
        user_prompt = self.USER_PROMPT_TEMPLATE.format(
            text=pdf_text,
            max_parts=self.MAX_SECTIONS,
        )

        system_prompt = self.SYSTEM_PROMPT.format(max_parts=self.MAX_SECTIONS)

        messages = [
            {"role": "system", "content": system_prompt},
            {"role": "user", "content": user_prompt},
        ]

        logger.info("Calling LLM to propose segment anchors (stream=False)...")

        try:
            resp = await self._call_llm_with_retry(
                model=self.model,
                messages=messages,
                stream=False,
                temperature=self.config.get("agents.paper_structurer.temperature", None),
            )
        except Exception as e:
            logger.error(f"Failed to get response from LLM: {e}")
            return [{"title": "Full Paper (fallback)", "content": pdf_text}]

        # -------- 解析非流式返回，得到纯文本 --------
        try:
            choices = getattr(resp, "choices", None)
            if not choices:
                logger.warning("No choices in response, fallback.")
                return [{"title": "Full Paper (fallback)", "content": pdf_text}]

            first = choices[0]
            message = getattr(first, "message", None)
            if message is None:
                logger.warning("No message in first choice, fallback.")
                return [{"title": "Full Paper (fallback)", "content": pdf_text}]

            raw_content = getattr(message, "content", "")
            llm_text = self._extract_text_from_message_content(raw_content).strip()
        except Exception as e:
            logger.error(f"Error parsing LLM response: {e}")
            return [{"title": "Full Paper (fallback)", "content": pdf_text}]

        if not llm_text:
            logger.warning("Empty content from LLM, fallback.")
            return [{"title": "Full Paper (fallback)", "content": pdf_text}]

        logger.info("LLM segment proposal raw text:")
        logger.info(llm_text[:500] + ("\n..." if len(llm_text) > 500 else ""))

        # ---------- 1) 解析 LLM 输出成 parts_info ----------

        parts_info = self._parse_parts_from_llm(llm_text)
        if not parts_info:
            logger.warning("Failed to parse any 'Part k | title | anchor' lines, fallback.")
            return [{"title": "Full Paper (fallback)", "content": pdf_text}]

        logger.info(f"Parsed {len(parts_info)} parts from LLM output.")

        # ---------- 2) 根据 anchor 在原文中切片 ----------

        sections = self._segment_by_anchors(pdf_text, parts_info)

        if not sections:
            logger.warning("Segmentation by anchors failed, fallback to full paper.")
            return [{"title": "Full Paper (fallback)", "content": pdf_text}]

        logger.info(f"Successfully segmented into {len(sections)} sections.")
        return sections

    @staticmethod
    def _parse_parts_from_llm(llm_text: str) -> List[Dict[str, Any]]:
        """
        解析 LLM 输出的若干行：
            Part k | <title> | <anchor_sentence>
        返回：
            [
                {"index": k, "title": <title>, "anchor": <anchor_sentence>},
                ...
            ]
        """
        parts_info: List[Dict[str, Any]] = []

        line_pattern = re.compile(
            r"^Part\s+(\d+)\s*\|\s*(.*?)\s*\|\s*(.+)$", re.IGNORECASE
        )

        for line in llm_text.splitlines():
            line = line.strip()
            if not line:
                continue
            m = line_pattern.match(line)
            if not m:
                continue
            idx_str, title, anchor = m.group(1), m.group(2), m.group(3)
            try:
                idx = int(idx_str)
            except ValueError:
                continue

            title = title.strip() or f"Part {idx}"
            anchor = anchor.strip()
            if not anchor:
                continue

            parts_info.append(
                {
                    "index": idx,
                    "title": title,
                    "anchor": anchor,
                }
            )

        # 按 index 排序，避免乱序
        parts_info.sort(key=lambda x: x["index"])
        return parts_info

    @staticmethod
    def _build_normalized_index(text: str) -> tuple[str, List[int]]:
        """
        将原文 text 中所有空白字符折叠为单个空格，用于“模糊匹配”（忽略多空格 / 换行）。
        返回:
            norm_text: 折叠后的字符串
            norm_to_orig: 长度与 norm_text 相同的列表，norm_to_orig[i] = 原文对应的字符下标
        """
        norm_chars: List[str] = []
        norm_to_orig: List[int] = []
        prev_wspace = False

        for i, ch in enumerate(text):
            if ch.isspace():
                # 连续空白只保留一个空格
                if prev_wspace:
                    continue
                norm_chars.append(" ")
                norm_to_orig.append(i)
                prev_wspace = True
            else:
                norm_chars.append(ch)
                norm_to_orig.append(i)
                prev_wspace = False

        return "".join(norm_chars), norm_to_orig

    @staticmethod
    def _normalize_for_matching(s: str) -> str:
        """
        对 anchor 句子做同样的空白折叠处理：
        - 所有空白（空格/换行/tab）折叠为单空格
        - 去掉首尾空白
        """
        out_chars: List[str] = []
        prev_wspace = False
        for ch in s:
            if ch.isspace():
                if prev_wspace:
                    continue
                out_chars.append(" ")
                prev_wspace = True
            else:
                out_chars.append(ch)
                prev_wspace = False
        return "".join(out_chars).strip()

    @staticmethod
    def _fuzzy_find_norm_pos(norm_text: str, norm_anchor: str, min_ratio: float = 0.5) -> int:
        """
        在归一化后的全文 norm_text 中，模糊查找 norm_anchor：
        - 使用 difflib.SequenceMatcher.find_longest_match
        - 如果最长公共子串长度 / norm_anchor 长度 >= min_ratio，则返回该位置
        - 否则返回 -1
        """
        from difflib import SequenceMatcher

        if not norm_anchor:
            return -1

        # anchor 过短时，匹配本身就不稳定，这里直接返回 -1，交给上层处理
        if len(norm_anchor) < 10:
            return -1

        sm = SequenceMatcher(None, norm_anchor, norm_text, autojunk=True)
        match = sm.find_longest_match(0, len(norm_anchor), 0, len(norm_text))

        if match.size == 0:
            return -1

        ratio = match.size / len(norm_anchor)
        if ratio >= min_ratio:
            # match.b 是 norm_text 中的起始位置
            return match.b
        return -1

    @staticmethod
    def _segment_by_anchors(
        text: str, parts_info: List[Dict[str, Any]]
    ) -> List[Dict[str, Any]]:
        """
        根据 anchor_sentence 在原始 text 中的位置，把全文切成若干连续大块：
        - 先用“空白折叠 + 精确匹配”
        - 匹配失败时再用“空白折叠 + 相似度 ≥ 0.7 的模糊匹配”
        - anchors 按在 text 中的起始位置排序
        - 每个块从该 anchor 开始，到下一个 anchor 之前为止（最后一块到文末）
        """
        # 先构建归一化后的整篇文本及其映射
        norm_text, norm_to_orig = PaperStructurer._build_normalized_index(text)

        positions: List[Dict[str, Any]] = []

        for p in parts_info:
            raw_anchor = p["anchor"]
            norm_anchor = PaperStructurer._normalize_for_matching(raw_anchor)

            if not norm_anchor:
                logger.warning(f"WARNING: empty normalized anchor for part '{p['title']}'")
                continue

            # 1) 先尝试精确匹配（在归一化文本上）
            pos_norm = norm_text.find(norm_anchor)

            # 2) 如果精确匹配失败，再尝试模糊匹配
            if pos_norm == -1:
                fuzzy_pos = PaperStructurer._fuzzy_find_norm_pos(norm_text, norm_anchor, min_ratio=0.7)
                if fuzzy_pos != -1:
                    logger.info(
                        f"INFO: fuzzy matched anchor for part "
                        f"'{p['title']}' at norm_pos={fuzzy_pos}"
                    )
                    pos_norm = fuzzy_pos

            if pos_norm == -1:
                # 找不到：提示 warning，然后跳过这个 part
                logger.warning(
                    f"WARNING: anchor not found (even with fuzzy match) for part "
                    f"'{p['title']}'. Raw anchor preview: {raw_anchor[:120]!r}"
                )
                continue

            # 映射回原文中的起始位置
            orig_start = norm_to_orig[pos_norm]

            positions.append(
                {
                    "start": orig_start,
                    "index": p["index"],
                    "title": p["title"],
                    "anchor": raw_anchor,
                }
            )

        if not positions:
            return []

        # 按在原文中的位置排序，保证从前到后
        positions.sort(key=lambda x: x["start"])

        sections: List[Dict[str, Any]] = []
        n = len(positions)

        for i, info in enumerate(positions):
            start = info["start"]
            end = positions[i + 1]["start"] if i + 1 < n else len(text)
            content = text[start:end].strip()
            title = f"Part {info['index']}: {info['title']}"
            sections.append(
                {
                    "title": title,
                    "content": content,
                }
            )

        return sections


# ===================== 本地测试入口 =====================

if __name__ == "__main__":
    import asyncio
    import base64
    from pathlib import Path

    async def main():
        PDF_FILE = "attention_is_all_you_need.pdf"

        pdf_path = Path(PDF_FILE)
        if not pdf_path.exists():
            print(f"[PaperStructurer __main__] PDF file not found: {pdf_path.resolve()}")
            return

        try:
            with open(pdf_path, "rb") as f:
                pdf_base64 = base64.b64encode(f.read()).decode("utf-8")
        except Exception as e:
            print(f"[PaperStructurer __main__] 读取 PDF 失败: {e}")
            return

        structurer = PaperStructurer()
        raw_text = structurer.extract_pdf_text_from_base64(pdf_base64)
        print(f"[PaperStructurer __main__] Raw PDF text length: {len(raw_text)} characters")

        sections = await structurer.run(raw_text)

        print("\n================ STRUCTURED SECTIONS ================\n")
        print(f"Total sections: {len(sections)}\n")
        for i, sec in enumerate(sections, start=1):
            title = sec.get("title", "")
            content = sec.get("content", "") or ""
            print(f"[Section {i}] {title}")
            print(f"  Content length: {len(content)} characters")
            print("  Preview (first 300 chars):")
            print("  " + content[:300].replace("\n", " ") + ("..." if len(content) > 300 else ""))
            print("------------------------------------------------\n")

    asyncio.run(main())
