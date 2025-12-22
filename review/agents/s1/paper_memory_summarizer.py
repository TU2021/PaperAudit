# agents/s1/paper_memory_summarizer.py
import json
from typing import AsyncGenerator, Any
from ..base_agent import BaseAgent
from ..logger import get_logger

logger = get_logger(__name__)

class PaperMemorySummarizer(BaseAgent):
    """
    Agent for building a natural-language MEMORY of the paper.
    输出：纯文本，高信息密度，用于下游 reviewer 快速 recall 论文内容。
    """

    def __init__(self, **kwargs: Any):
        super().__init__(**kwargs)

    SYSTEM_PROMPT = (
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

    USER_PROMPT_TEMPLATE = """Write a NATURAL-LANGUAGE MEMORY for the paper with the following structure:

REQUIRED HEADINGS (in this order):
1) # Global Summary
2) Then one '# <Section Title>' for EACH of the following section titles (use EXACTLY these spellings):
{section_titles}

CONTENT GUIDELINES PER HEADING:
- Global Summary: a compact overview of the problem, core approach, evaluation scope, key findings, and explicitly stated caveats. Include major quantitative results if highlighted by the authors.
- For each section: extract key ideas, datasets, metrics, baselines, and specific claims. Always record important quantitative details such as accuracy, scores, improvement margins, runtime, sample sizes, or other numbers that the authors emphasize. Quote numbers exactly when available.
- Use short paragraphs or '-' bullets. Avoid long verbatim quotes.
- If important contextual details (e.g., number of runs, dataset license, ablation structure) are missing, you may note 'Not specified in this section.'

FORMAT:
- Plain text only with ATX-style headings starting with '# '. No JSON. No markdown tables. No code fences.
- Do NOT add your own judgments or opinions; summarize only what the manuscript states.

Full Manuscript:
{text}
"""

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
            parts = []
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

    async def run(self, pdf_text: str, *, section_titles: list[str] | None = None) -> str:
        """
        Build a natural-language memory from the full paper text.

        Args:
            pdf_text: 已经从 PDF 提取好的整篇论文文本（不再是 base64）

        Returns:
            纯文本 memory（单个字符串，不再是 SSE / AsyncGenerator）
        """
        titles_block = "\n".join([f"- {t}" for t in section_titles]) if section_titles else "- (no sections detected)"
        prompt = self.USER_PROMPT_TEMPLATE.format(text=pdf_text, section_titles=titles_block)

        messages = [
            {"role": "system", "content": self.SYSTEM_PROMPT},
            {"role": "user", "content": prompt},
        ]

        logger.info("Calling LLM to build memory (stream=False)...")

        try:
            temp = self.config.get("agents.paper_memory_summarizer.temperature", None)
            resp = await self._call_llm_with_retry(
                model=self.model,
                messages=messages,
                stream=False,
                temperature=temp,
            )
        except Exception as e:
            logger.error(f"Failed to get response from LLM: {e}")
            return f"[MEMORY_ERROR] {e}"

        try:
            choices = getattr(resp, "choices", None)
            if not choices:
                logger.error("No choices in response.")
                return "[MEMORY_ERROR] Empty response"

            first = choices[0]
            message = getattr(first, "message", None)
            if message is None:
                logger.error("No message in first choice.")
                return "[MEMORY_ERROR] No message in response"

            raw_content = getattr(message, "content", "")
            full_text = self._extract_text_from_message_content(raw_content).strip()

            if not full_text:
                logger.error("Empty content from LLM.")
                return "[MEMORY_ERROR] Empty content"
        except Exception as e:
            logger.error(f"Error while parsing non-stream response: {e}")
            return f"[MEMORY_ERROR] {e}"

        logger.info(f"Memory text length: {len(full_text)} characters")
        return full_text


if __name__ == "__main__":
    # Example usage (for testing purposes)
    import asyncio
    import base64

    async def main():
        PDF_FILE = "IEEE_TMM_main.pdf"

        try:
            with open(PDF_FILE, "rb") as f:
                pdf_base64 = base64.b64encode(f.read()).decode("utf-8")
        except Exception as e:
            print(f"读取 PDF 失败: {e}")
            return

        paper_memory_summarizer = PaperMemorySummarizer()
        sample_pdf_content = paper_memory_summarizer.extract_pdf_text_from_base64(pdf_base64)
        print(f"PDF text length: {len(sample_pdf_content)} characters")
        print(sample_pdf_content[:500])

        memory = await paper_memory_summarizer.run(sample_pdf_content)
        print("\n==============================")
        print(f"[FINAL] Paper memory length: {len(memory)} characters")
        print("==============================")
        print("Preview (first 1000 chars):")
        print(memory[:1000])

    asyncio.run(main())
