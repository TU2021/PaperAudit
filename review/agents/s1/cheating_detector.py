from typing import AsyncGenerator, Any, Dict, List, Union
import json
from ..base_agent import BaseAgent
from ..logger import get_logger

logger = get_logger(__name__)


class CheatingDetector(BaseAgent):
    """Agent for detecting cheating / critical issues in research papers"""
    
    # 原来的全局审稿 prompt（整篇论文级别）
    SYSTEM_PROMPT = """[System Role] You are an expert research paper reviewer specializing in detecting potential cheating or unethical practices in academic papers.
Your task is to critically assess the content of the paper for any signs of plagiarism, data fabrication, or other forms of academic dishonesty.

[Critical Constraints]
1) Evidence-first: Every claim MUST be supported by anchors to the manuscript
   (figure/table/equation/section/page). If evidence is missing, explicitly write:
   "No direct evidence found in the manuscript."
2) Maintain anonymity; do not guess author identities/affiliations; keep a constructive tone.
3) Avoid speculative claims; do not cite external sources unless they appear in the paper's reference list.

[Input]
- Full anonymous manuscript (plain text or OCR output).

[Output Template]
Write an academic dishonesty report.
"""

    # ✨ 针对「单章节 + 全局记忆」的批判性审稿 prompt
    SECTION_SYSTEM_PROMPT = """[System Role] You are an experienced reviewer for top-tier ML/AI conferences.
    You will receive two inputs:
    - A GLOBAL MEMORY summarizing the key ideas of the whole paper
    - A SINGLE SECTION of the paper that you should examine carefully

    Note: the text you see is a cleaned and possibly slightly condensed version of the original paper.
    Some low-level formatting or minor details may be missing. Do NOT over-focus on tiny issues
    caused by this preprocessing. Instead, prioritize higher-level, substantive problems that
    would matter for an actual conference review.

    Your job is to read this section in depth, relate it to the global memory when useful,
    and provide thoughtful, critical reviewer-style comments. Focus on the following:

    1) Identify anything that feels scientifically weak, incomplete, or potentially suspicious.
    This includes unusually strong or clean results, missing details that might hide flaws,
    data inconsistencies, or patterns that could hint at unethical behavior. Be careful:
    any suspicion must be based on what you can observe in the text.

    2) Point out missing or insufficient comparisons, baselines, ablations, or explanations
    that make the claims in this section less convincing.

    3) Ask concrete, critical questions that a reviewer would reasonably raise for this section.

    [Critical Constraints]
    - Focus on substantive, high-impact issues rather than minor wording or cosmetic problems.
    - Base every criticism on textual evidence. Quote or paraphrase relevant phrases
    and mention whether they come from the section or from the global memory.
    - If something only feels suspicious and you cannot provide direct evidence,
    clearly state that this is just a suspicion and explain why.
    - Do NOT speculate about author identities, affiliations, or motives.
    - Do NOT introduce external papers or knowledge unless they already appear in the manuscript/memory.
    - DO NOT consider content unrelated to the main text and appendices, such as checklists.
    
    [Output]
    Write your comments in a natural reviewer voice. You do not need to follow a fixed structure.
    Group points naturally, as a human reviewer would when providing critical feedback on a section.
    """

    USER_PROMPT_TEMPLATE = """Research Paper Content:
{text}
"""

    # ---------- 小工具：统一解析 message.content ----------
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

        # 兜底
        return str(raw_content)

    # ===== 整篇论文级别接口：对外仍然是 SSE，但内部改为非流式 =====
    async def run(self, pdf_content: str) -> AsyncGenerator[str, None]:
        """
        Detect cheating in the provided research paper content (full-manuscript level).

        Args:
            pdf_content: paper content in well-formed structured text
        Yields:
            SSE-style content chunks from the API response:
            'data: {...}\\n\\n'
        """
        prompt = self.USER_PROMPT_TEMPLATE.format(text=pdf_content)

        messages = [
            {"role": "system", "content": self.SYSTEM_PROMPT},
            {"role": "user", "content": prompt}
        ]

        logger.info(f"Calling LLM with retry mechanism (full-paper, stream=False)...")

        try:
            # ✅ 非流式：一次性拿完整结果
            resp = await self._call_llm_with_retry(
                model=self.reasoning_model,
                messages=messages,
                stream=False,
                temperature=self.config.get("agents.cheating_detector.temperature", None)
            )
        except Exception as e:
            logger.error(f"Failed to get response from LLM: {e}")
            yield "data: [ERROR]\n\n"
            return

        # 解析非流式返回
        try:
            choices = getattr(resp, "choices", None)
            if not choices:
                logger.error("No choices in response.")
                yield "data: [ERROR]\n\n"
                return

            first = choices[0]
            message = getattr(first, "message", None)
            if message is None:
                logger.error("No message in first choice.")
                yield "data: [ERROR]\n\n"
                return

            raw_content = getattr(message, "content", "")
            full_text = self._extract_text_from_message_content(raw_content).strip()

        except Exception as e:
            logger.error(f"Error while parsing non-stream response: {e}")
            yield "data: [ERROR]\n\n"
            return

        if not full_text:
            logger.error("Empty content from LLM.")
            yield "data: [ERROR]\n\n"
            return

        # ✅ 手动切成小块，以 SSE 形式吐给前端（保持接口兼容）
        CHUNK_SIZE = 512
        for i in range(0, len(full_text), CHUNK_SIZE):
            piece = full_text[i:i + CHUNK_SIZE]
            response_data = {
                "object": "chat.completion.chunk",
                "choices": [{
                    "delta": {
                        "content": piece
                    }
                }]
            }
            yield f"data: {json.dumps(response_data)}\n\n"

        yield "data: [DONE]\n\n"

    # ===== Section 级别批判性审稿：改成并行调用 =====
    async def run_sectionwise(
        self,
        structured_paper: Union[Dict[str, Any], List[Dict[str, Any]]],
        paper_memory: str,
    ) -> str:
        """
        对“提取完的文章 JSON”进行 section 级别批判性审稿（并行调用 LLM）。

        参数:
            structured_paper:
                - 可以是形如 { "sections": [ { "title": ..., "content": ... }, ... ] } 的 dict
                - 也可以直接是 [ { "title": ..., "content": ... }, ... ] 的 list
            paper_memory:
                - 全局 GLOBAL MEMORY 文本（由 PaperMemorySummarizer 生成）

        返回:
            一个大的字符串，按 section 拼接好的批判性分析结果，形如：

            [SECTION 1] Introduction
            ...该章节的批判性分析...

            [SECTION 2] Method
            ...该章节的批判性分析...
        """
        import asyncio  # ✅ NEW: 并行需要

        # 1) 解析 sections 列表
        if isinstance(structured_paper, dict):
            sections = structured_paper.get("sections", [])
        else:
            sections = structured_paper

        # 做个轻量清洗，防止坏数据
        clean_sections: List[Dict[str, str]] = []
        for sec in sections:
            title = str(sec.get("title", "")).strip() or "Untitled"
            content = str(sec.get("content", "")).strip()
            if not content:
                continue
            clean_sections.append({"title": title, "content": content})

        if not clean_sections:
            logger.warning("No valid sections found in structured_paper; fallback to empty result.")
            return "[CheatingDetector] No sections to review.\n"

        num_sections = len(clean_sections)
        logger.info(f"Starting section-wise critical review on {num_sections} sections...")

        # ✅ NEW: 控制最大并发，避免一下子开太多并行请求
        max_concurrency = min(self.config.get("concurrency.cheating_detector", 6), num_sections)  # 例如最多 6 个同时跑，你可以按需要调节
        sem = asyncio.Semaphore(max_concurrency)

        async def _review_single_section(idx: int, sec: Dict[str, str]) -> str:
            """
            并行跑单个 section 的审稿，最终返回带有 [SECTION k] 头的文本。
            """
            sec_title = sec["title"]
            sec_content = sec["content"]
            logger.info(f" Scheduling Section {idx+1}/{num_sections}: {sec_title}")

            section_header = f"\n\n[SECTION {idx+1}] {sec_title}\n"

            # 构造单节的输入：带上 GLOBAL MEMORY + FOCUSED SECTION
            user_prompt = (
                "# GLOBAL MEMORY (for context)\n"
                f"{paper_memory}\n\n"
                "# FOCUSED SECTION (to review)\n"
                f"# {sec_title}\n"
                f"{sec_content}\n"
            )

            messages = [
                {"role": "system", "content": self.SECTION_SYSTEM_PROMPT},
                {"role": "user", "content": user_prompt},
            ]

            async with sem:  # ✅ NEW: 并发控制
                logger.info(f" Reviewing Section {idx+1}: {sec_title}")
                try:
                    resp = await self._call_llm_with_retry(
                        model=self.reasoning_model,
                        messages=messages,
                        stream=False,      # 非流式
                        temperature=self.config.get("agents.cheating_detector.temperature", None),
                    )
                except Exception as e:
                    err_msg = f"[CheatingDetector ERROR in section '{sec_title}']: {e}\n"
                    logger.error(err_msg)
                    return section_header + err_msg

            # 解析非流式结果
            try:
                choices = getattr(resp, "choices", None)
                if not choices:
                    msg = "[CheatingDetector] Empty response for this section.\n"
                    return section_header + msg

                first = choices[0]
                message = getattr(first, "message", None)
                if message is None:
                    msg = "[CheatingDetector] No message in response for this section.\n"
                    return section_header + msg

                raw_content = getattr(message, "content", "")
                section_text = self._extract_text_from_message_content(raw_content).strip()
            except Exception as e:
                err_msg = f"[CheatingDetector] Error parsing section '{sec_title}' response: {e}\n"
                logger.error(err_msg)
                return section_header + err_msg

            return section_header + section_text

        # ✅ NEW: 为每个 section 创建一个任务，并行跑
        tasks = [
            _review_single_section(idx, sec)
            for idx, sec in enumerate(clean_sections)
        ]

        results: List[str] = await asyncio.gather(*tasks)

        full_result = "".join(results)
        logger.info("Section-level critical review completed (parallel).")
        return full_result


if __name__ == "__main__":
    # Example usage (for testing purposes)
    import asyncio
    import base64
    
    async def main():
        PDF_FILE = "attention_is_all_you_need.pdf"
        
        try:
            with open(PDF_FILE, "rb") as f:
                pdf_base64 = base64.b64encode(f.read()).decode('utf-8')
        except Exception as e:
            print(f"读取 PDF 失败: {e}")
            return

        detector = CheatingDetector()
        sample_pdf_content = detector.extract_pdf_text_from_base64(pdf_base64)
        print(f"PDF text length: {len(sample_pdf_content)} characters")
        print(sample_pdf_content[:500])  # Print first 500 characters of the PDF text

        print("\n===== 测试原来的 full-paper run() （现在内部非流式）=====")
        async for chunk in detector.run(sample_pdf_content):
            if chunk == "data: [DONE]\n\n":
                print("\n[Stream completed]")
                break
            elif chunk.startswith("data: "):
                content = chunk[6:].strip()
                if content != "[ERROR]":
                    data = json.loads(content)
                    if "choices" in data and data["choices"]:
                        delta = data["choices"][0].get("delta", {})
                        if "content" in delta:
                            print(delta["content"], end="", flush=True)
                else:
                    print("\n[Error occurred during processing]")

        print("\n\n===== 测试新的 run_sectionwise()（用 fake sections，并行） =====")
        fake_structured = {
            "sections": [
                {"title": "Introduction", "content": sample_pdf_content[:3000]},
                {"title": "Method", "content": sample_pdf_content[3000:6000]},
            ]
        }
        fake_memory = "# Global Summary\nThis is a fake memory of the paper.\n"

        sectionwise_result = await detector.run_sectionwise(fake_structured, fake_memory)
        print(sectionwise_result)

    asyncio.run(main())
