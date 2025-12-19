from typing import List, Dict, Any
from ..base_agent import BaseAgent


class PaperStructurer(BaseAgent):
    def __init__(self, **kwargs):
        super().__init__(**kwargs)

    """
    用 LLM 对原始 PDF 文本做“结构化规整”：
    - 识别出若干个大的章节块（Part 1, Part 2, ...）
    - 归并该章节下的原文内容（只整理换行/分页/页眉页脚）
    - 不做语义改写，不增删 claim，只做轻度清洗

    最终返回一个 sections 列表，每个元素：
    - title: 章节标题（如 "Part 1: Introduction and Motivation"）
    - content: 该章节的规整后原文内容
    """

    # 🔢 最多几大块（可按需调整）
    MAX_SECTIONS = 5

    SYSTEM_PROMPT = (
        "You are a meticulous document structurer for scientific papers. "
        "Your job is to reorganize a noisy plain-text manuscript (with page breaks, "
        "headers, footers, OCR breaks, and inconsistent formatting) into clean, "
        "coherent, section-organized text.\n\n"

        "================ GLOBAL GOAL ================\n"
        f"Split the ENTIRE manuscript into AT MOST {MAX_SECTIONS} major parts.\n"
        "Each part must be a contiguous block of text from the original paper,\n"
        "and together these parts MUST cover ALL meaningful content of the paper.\n\n"

        "================ CRITICAL NON-NEGOTIABLE RULE =================\n"
        "ABSOLUTELY NO ALTERATION, LOSS, OR MODIFICATION OF SCIENTIFIC CONTENT.\n"
        "The model is STRICTLY PROHIBITED from:\n"
        "  - changing or paraphrasing any sentence beyond minimal OCR cleanup,\n"
        "  - modifying formulas, equations, mathematical symbols, theorems, or definitions,\n"
        "  - altering numbers, experimental results, table values, or figure-related text,\n"
        "  - rewriting scientific claims, contributions, or method descriptions,\n"
        "  - merging or deleting any scientific details.\n"
        "EVERY piece of scientific content MUST appear exactly as in the manuscript.\n"
        "If the original contains errors or inconsistencies, preserve them faithfully.\n\n"

        "================ BEHAVIOR RULES ================\n"
        "1) ABSOLUTELY NO INVENTION:\n"
        "   Do NOT invent any claims, formulas, results, baselines, or citations.\n"
        "   Only the original manuscript content is allowed.\n\n"

        "2) DO NOT REMOVE, MODIFY, OR SUMMARIZE MEANING:\n"
        "   You may clean OCR artifacts (broken lines, hyphenation), but you MUST NOT:\n"
        "     - paraphrase,\n"
        "     - shorten,\n"
        "     - summarize,\n"
        "     - compress paragraphs,\n"
        "     - rewrite sentences.\n"
        "   All scientific concepts, equations, algorithm steps, and explanatory text\n"
        "   MUST remain exactly as originally written.\n\n"

        "3) FULL COVERAGE REQUIRED:\n"
        "   You must include EVERY part of the manuscript.\n"
        "   Do NOT skip any paragraph, sentence, formula, table, figure caption,\n"
        "   or experimental detail.\n\n"

        "4) FORMULAS / EQUATIONS — STRICT PRESERVATION:\n"
        "   ALL equations must be preserved EXACTLY:\n"
        "     - identical symbols,\n"
        "     - identical formatting (LaTeX code, unicode math, or ASCII math),\n"
        "     - identical structure.\n"
        "   DO NOT rewrite formulas into prose.\n"
        "   DO NOT simplify or alter any mathematical expression.\n\n"

        "5) FIGURES AND TABLES — MAXIMUM PRESERVATION:\n"
        "   - If the manuscript contains text associated with figures (captions, labels,\n"
        "     axis descriptions, annotations), you MUST preserve all of it.\n"
        "   - For tables:\n"
        "       • Preserve the entire table if possible.\n"
        "       • If the table cannot be reproduced in grid form, you MUST convert it\n"
        "         into detailed textual form while preserving ALL rows, columns, and numbers.\n"
        "         Example textual fallback:\n"
        "             \"Table X: <title>\"\n"
        "             \"Row 1: metric1 = ..., metric2 = ...\"\n"
        "             \"Row 2: ...\"\n"
        "       • You MUST NOT drop any entry, statistic, or ablation value.\n"
        "   - If figure content cannot be reproduced as a figure, convert every visible\n"
        "     part into text (e.g., labels, axes, measurement values). DO NOT omit anything.\n\n"

        "6) ALLOWED TRANSFORMATION (FORMATTING ONLY — NO CONTENT CHANGE):\n"
        "   - Fix OCR line breaks.\n"
        "   - Join hyphenated words.\n"
        "   - Remove repeated page headers/footers.\n"
        "   - Remove standalone page numbers.\n"
        "   - Normalize whitespace.\n"
        "   Absolutely NO modification of scientific meaning or content.\n\n"

        "================ SECTION FORMAT REQUIREMENT ================\n"
        f"You MUST create NO MORE THAN {MAX_SECTIONS} MAJOR PARTS.\n"
        "Each major part MUST start with a Markdown heading of the form:\n"
        "    # Part 1: <Very short title>\n"
        "    # Part 2: <Very short title>\n"
        "    ...\n"
        "    # Part K: <Very short title>\n"
        "where K <= {MAX_SECTIONS} and part numbers increase monotonically.\n\n"
        "Within each part, you may use '##' or '###' for subsections when helpful,\n"
        "but NEVER begin them with '# Part'.\n\n"

        "================ OUTPUT FORMAT ================\n"
        "Produce ONE plain-text document in the following pattern:\n\n"
        "# Part 1: <short title>\n"
        "<cleaned but complete text>\n\n"
        "# Part 2: <short title>\n"
        "<cleaned but complete text>\n\n"
        "(... up to at most {MAX_SECTIONS} parts ...)\n\n"
        "DO NOT output JSON.\n"
        "DO NOT summarize.\n"
        "DO NOT rewrite.\n"
        "DO NOT remove formulas.\n"
        "DO NOT remove citations.\n"
        "DO NOT remove any figure/table content.\n"
        "DO NOT alter ANY scientific wording.\n"
        "Your output MUST be a faithful, complete, structurally reorganized reproduction\n"
        "of the original manuscript — with ZERO loss or alteration of scientific information.\n"
    )

    USER_PROMPT_TEMPLATE = (
        "Here is the full manuscript extracted from a PDF (with possible noise such as page headers, "
        "footers, and line breaks).\n"
        "Please lightly clean it and reorganize it into AT MOST {max_parts} major parts, using "
        "Markdown-style headings of the form '# Part k: <short title>'.\n\n"
        "You MUST keep all important content; do not summarize or shorten aggressively.\n\n"
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
        对原始 PDF 文本做规整（内部流式，外部仍然一次性返回文本）：
        - 调 LLM 输出一个带 `# Part k: ...` 标题的“规整全文”
        - 在本地按 `# Part k: ...` 标题切分成 sections 列表
        """
        user_prompt = self.USER_PROMPT_TEMPLATE.format(
            text=pdf_text,
            max_parts=self.MAX_SECTIONS,
        )

        messages = [
            {"role": "system", "content": self.SYSTEM_PROMPT},
            {"role": "user", "content": user_prompt},
        ]

        print("[PaperStructurer] Calling LLM to produce cleaned, headed text (stream=True)...")

        try:
            stream = await self._call_llm_with_retry(
                model=self.model,
                messages=messages,
                stream=True,       # ⭐ 改回 True
                temperature=self.config.get("agents.paper_structurer_old.temperature", None),
            )
        except Exception as e:
            print(f"[PaperStructurer] Failed to get response from LLM: {e}")
            return [{"title": "Full Paper (fallback)", "content": pdf_text}]

        # -------- 累积流式内容成一个 structured_text --------
        content_chunks: List[str] = []

        try:
            async for chunk in stream:
                if not getattr(chunk, "choices", None):
                    continue
                delta = chunk.choices[0].delta
                delta_content = getattr(delta, "content", None)
                if not delta_content:
                    continue

                if isinstance(delta_content, str):
                    content_chunks.append(delta_content)
                elif isinstance(delta_content, list):
                    parts: List[str] = []
                    for part in delta_content:
                        if isinstance(part, dict):
                            t = part.get("text") or part.get("content")
                            if t:
                                parts.append(str(t))
                        else:
                            t = getattr(part, "text", None) or getattr(part, "content", None)
                            if t:
                                parts.append(str(t))
                    if parts:
                        content_chunks.append("".join(parts))
        except Exception as e:
            print(f"[PaperStructurer] Error while streaming LLM response: {e}")
            return [{"title": "Full Paper (fallback)", "content": pdf_text}]

        structured_text = "".join(content_chunks).strip()
        print(f"[PaperStructurer] Received structured text length: {len(structured_text)} characters")

        if not structured_text:
            print("[PaperStructurer] Empty content from LLM, fallback to full paper.")
            return [{"title": "Full Paper (fallback)", "content": pdf_text}]

        # ---------- 2) 本地按 `# Part k: ...` 标题切分 section ----------
        sections = self._split_by_markdown_headings(structured_text)

        if not sections:
            print("[PaperStructurer] No headings found, fallback to single-section structured text.")
            return [{"title": "Full Paper (structured)", "content": structured_text}]

        print(f"[PaperStructurer] Successfully parsed {len(sections)} sections from '# Part k: ...' headings.")
        return sections


    @staticmethod
    def _split_by_markdown_headings(structured_text: str) -> List[Dict[str, Any]]:
        """
        按 '# Part k: <title>' 切分章节：
        - 只有形如 '# Part 1: ...' 的行作为新的 section。
        - 其他 '#XXX'、'##'、'###' 等全部当作当前 section 的内容。
        """
        import re

        lines = structured_text.splitlines()

        sections: List[Dict[str, Any]] = []
        current_title: str | None = None
        current_lines: List[str] = []

        # 只匹配 "# Part 1: xxx" 这种
        part_pattern = re.compile(r"^#\s*Part\s+(\d+)\s*:(.*)$", re.IGNORECASE)

        def flush():
            nonlocal current_title, current_lines
            if current_title is not None:
                content = "\n".join(current_lines).strip()
                sections.append(
                    {
                        "title": current_title,
                        "content": content,
                    }
                )
            current_title = None
            current_lines = []

        for line in lines:
            stripped = line.lstrip()
            m = part_pattern.match(stripped)

            if m:
                # 命中 '# Part k: ...' -> 新 section
                flush()
                idx = m.group(1).strip()
                raw_title = m.group(2).strip()
                if not raw_title:
                    raw_title = f"Part {idx}"
                current_title = f"Part {idx}: {raw_title}"
                current_lines = []
            else:
                # 其他所有行都归入当前 section
                if current_title is None:
                    current_title = "Part 0: Other / Unassigned"
                    current_lines = []
                current_lines.append(line)

        flush()
        return sections


# ===================== 本地测试入口 =====================

if __name__ == "__main__":
    import asyncio
    import base64
    from pathlib import Path

    async def main():
        # 换成你想测试的 PDF 文件路径
        PDF_FILE = "attention_is_all_you_need.pdf"

        pdf_path = Path(PDF_FILE)
        if not pdf_path.exists():
            print(f"[PaperStructurer __main__] PDF file not found: {pdf_path.resolve()}")
            return

        # 先读 PDF -> base64，再用 BaseAgent 的 extract_pdf_text_from_base64
        try:
            with open(pdf_path, "rb") as f:
                pdf_base64 = base64.b64encode(f.read()).decode("utf-8")
        except Exception as e:
            print(f"[PaperStructurer __main__] 读取 PDF 失败: {e}")
            return

        structurer = PaperStructurer()

        raw_text = structurer.extract_pdf_text_from_base64(pdf_base64)
        print(f"[PaperStructurer __main__] Raw PDF text length: {len(raw_text)} characters")
        print("---- Raw text preview (first 500 chars) ----")
        print(raw_text[:500])
        print("------------------------------------------------\n")

        # 调用 run 做规整
        sections = await structurer.run(raw_text)

        print("\n================ STRUCTURED SECTIONS ================\n")
        print(f"Total sections: {len(sections)}\n")
        for i, sec in enumerate(sections, start=1):
            title = sec.get("title", "")
            content = sec.get("content", "") or ""
            print(f"[Section {i}] {title}")
            print(f"  Content length: {len(content)} characters")
            print("  Preview (first 400 chars):")
            print("  " + content[:400].replace("\n", " ") + ("..." if len(content) > 400 else ""))
            print("------------------------------------------------\n")

    asyncio.run(main())
