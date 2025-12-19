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

    SYSTEM_PROMPT = (
        "You are a meticulous scientific summarizer. Read the entire manuscript "
        "(all sections, tables, figures) and produce a NATURAL-LANGUAGE MEMORY document "
        "(no JSON). The memory should be concise but information-dense, suitable for "
        "later reviewers or agents to quickly recall what the paper does and claims.\n\n"
        "PRIMARY GOALS (ESPECIALLY IMPORTANT):\n"
        "1) Clearly capture the PROBLEM and MOTIVATION:\n"
        "   - What concrete problem does the paper try to solve?\n"
        "   - Why is this problem important or non-trivial?\n"
        "   - What limitations or gaps in prior work motivate this paper?\n"
        "2) Clearly capture the CORE IDEA and NOVELTY:\n"
        "   - What is the main method / algorithm / system proposed?\n"
        "   - In what sense is it new or different from existing approaches?\n"
        "   - If the claimed novelty is weak or incremental, state that explicitly.\n"
        "3) Clearly capture KEY NUMBERS and CLAIMS:\n"
        "   - Datasets, benchmarks, and tasks used.\n"
        "   - Main metrics and the most important numbers (e.g., accuracy, BLEU, F1, AUROC).\n"
        "   - How much the method improves / degrades vs. key baselines (report concrete deltas when available).\n"
        "   - Any ablations or sensitivity results that strongly support or weaken the claims.\n\n"
        "STRICT FORMAT RULES:\n"
        "1) Output plain text only (no JSON, no markdown code blocks).\n"
        "2) Use simple ATX-style headings.\n"
        "3) Begin with a top-level heading: '# Global Summary'. Under this heading, briefly summarize:\n"
        "   - The problem and motivation.\n"
        "   - The core idea and novelty.\n"
        "   - The main experimental findings and key numbers.\n"
        "4) Then write one top-level heading per section USING EXACTLY the section titles provided in the manuscript\n"
        "   (e.g., '# Introduction', '# Method', '# Experiments', '# Related Work', etc.).\n"
        "5) Under each section heading, use short paragraphs or plain-text bullets (bullets with '-' are allowed).\n"
        "6) In each section, prioritize:\n"
        "   - Motivation-related content (why this section matters for the overall problem).\n"
        "   - Novel mechanisms, design choices, or assumptions.\n"
        "   - Concrete experimental setups, key baselines, and the most important reported numbers.\n"
        "7) Prefer dense, verifiable facts (quote numbers/units when available). Avoid boilerplate or vague praise.\n"
        "8) Do NOT speculate beyond what is stated in the manuscript. If something is unclear or missing, state that it is unclear or not specified.\n"
    )

    USER_PROMPT_TEMPLATE = """Full Manuscript:
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

    async def run(self, pdf_text: str) -> str:
        """
        Build a natural-language memory from the full paper text.

        Args:
            pdf_text: 已经从 PDF 提取好的整篇论文文本（不再是 base64）

        Returns:
            纯文本 memory（单个字符串，不再是 SSE / AsyncGenerator）
        """
        prompt = self.USER_PROMPT_TEMPLATE.format(text=pdf_text)

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
