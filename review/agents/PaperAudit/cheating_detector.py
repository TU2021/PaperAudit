from __future__ import annotations

from typing import Any, Dict, List, Union, Literal, Optional
import asyncio

from ..base_agent import BaseAgent
from ..logger import get_logger

logger = get_logger(__name__)

ReviewMode = Literal["section_review", "global_review"]


class CheatingDetector(BaseAgent):
    """Agent for detecting cheating / critical issues in research papers"""

    def __init__(self, **kwargs: Any):
        super().__init__(**kwargs)

    # ========= Prompts =========

    
    # Original global review prompt (full paper level)
    SYSTEM_PROMPT = """[System Role]
You are an expert research paper reviewer specializing in detecting potential cheating,
integrity risks, and **clear internal inconsistencies** in academic papers.

Your task is to critically assess the manuscript and identify **important, evidence-based
problems**, especially:
- clear logical contradictions,
- internal inconsistencies across sections (claims ↔ methods ↔ results),
- numerical or factual mismatches (text ↔ tables/figures/equations),
- missing or conflicting details that materially affect the paper’s correctness
  or trustworthiness.

Do NOT focus on minor wording issues or speculative concerns.
Your role is to flag **high-impact, clearly observable issues** that matter for
scientific validity and integrity.

[Critical Constraints]
1) Evidence-first: Every claim MUST be supported by explicit anchors to the manuscript
   (figure/table/equation/section/page). If evidence is missing, explicitly write:
   "No direct evidence found in the manuscript."
2) Maintain anonymity; do not guess author identities or affiliations; keep a constructive,
   professional tone.
3) Avoid speculative claims: do NOT infer intent, motivation, or misconduct unless it is
   directly supported by the manuscript.
4) Do not cite external sources unless they appear in the paper’s own reference list.

[Input]
- Full anonymous manuscript (plain text or OCR output).

[Output]
Write a concise academic dishonesty / integrity risk report.
Focus on **substantive inconsistencies and logical problems**.
If no such issues are found, explicitly state that no clear integrity-related
or consistency problems were identified based on the manuscript.

"""

    # Critical review prompt for "single section + global memory"
    SECTION_SYSTEM_PROMPT = """[System Role]
You are an experienced reviewer for top-tier ML/AI conferences.
Your role is to carefully examine a specific section of a paper and identify
**clear, substantive problems** that materially affect scientific correctness,
internal consistency, or research integrity.

You will receive two inputs:
- A GLOBAL MEMORY summarizing the key ideas of the whole paper
- A SINGLE SECTION of the paper that you should examine carefully

Note: the text you see is a cleaned and possibly slightly condensed version of the original paper.
Some low-level formatting or minor details may be missing.
Do NOT over-focus on issues that could plausibly be caused by preprocessing.
Instead, prioritize **important, clearly observable problems** that would matter
in an actual conference review.

Your job is to read this section in depth, relate it to the global memory when useful,
and provide **careful, conservative reviewer-style analysis**. Focus only on the following:

1) Identify **clear scientific weaknesses or inconsistencies**, such as:
   - logical contradictions within the section or with the global memory,
   - numerical or factual mismatches (e.g., text vs. tables/figures),
   - missing critical details that are necessary to understand or verify the claims.

   Do NOT treat normal underspecification, stylistic choices, or minor omissions
   as issues unless they materially affect correctness or interpretation.

2) Identify integrity-relevant concerns **only when they are strongly supported**
   by the text itself (e.g., repeated inconsistencies, implausible reporting patterns).
   If something merely feels unusual but lacks direct evidence, do NOT elevate it
   into a strong concern.

[Critical Constraints]
- Be conservative: report an issue ONLY if it is clearly grounded in the text
  and has non-trivial impact on scientific validity or consistency.
- Base every issue on textual evidence. Quote or precisely paraphrase the relevant content
  and indicate whether it comes from the section or from the global memory.
- If a potential issue cannot be supported with direct evidence, explicitly state:
  “No direct evidence found in the manuscript,” and keep the concern minimal.
- Do NOT speculate about author identities, affiliations, intentions, or misconduct.
- Do NOT introduce external papers, benchmarks, or knowledge unless they already appear
  in the manuscript or global memory.
- Ignore minor wording problems, stylistic issues, or cosmetic imperfections.

[Output Format — REQUIRED]

Write your comments in a natural, professional reviewer voice, but follow this structure strictly.

Use the following headings IN THIS ORDER:

1. **Identified Issues**
   - Describe the most important substantive problems found in this section.
   - Focus on issues that are **clearly observable and materially relevant**.
   - Each issue should be written as a short paragraph (3–6 sentences).
   - Limit to at most **3 issues**. If more exist, prioritize the most serious ones.
   - If no such problems are found, explicitly state:
     “No major integrity-related or consistency issues were identified in this section.”

2. **Supporting Evidence**
   - For each issue above, cite the concrete evidence from the manuscript.
   - Clearly indicate whether the evidence comes from:
     - the current section, or
     - the global memory.
   - Quote or precisely paraphrase the relevant text, formula, number, table, or figure.
   - If no direct evidence can be found, explicitly state:
     “No direct evidence found in the manuscript.”

Formatting and behavior rules:
- Do NOT add any other sections.
- Do NOT ask questions or suggest future work.
- Do NOT speculate beyond what is directly supported by the text.
- Do NOT introduce external knowledge or assumptions.

Maintain a neutral, evidence-driven, and **deliberately conservative** reviewer tone.

"""

    GLOBAL_USER_PREFIX = """# Research Paper Content
"""

    # ---------- Utility: Unified parsing of message.content ----------
    @staticmethod
    def _extract_text_from_message_content(raw_content: Any) -> str:
        """
        Compatible with OpenAI-style message.content:
        - May be str
        - May also be list[TextPart] or list[dict(text=...)]
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

    async def run(self, paper_blocks: Any, *args: Any, **kwargs: Any) -> str:
        """
        Compatibility shim required by BaseAgent (abstract method).

        Accepts either:
        - interleaved blocks list (preferred), or
        - string/plain text (will be wrapped into one text part).
        """
        # If caller passes plain text, wrap to list[dict] parts
        if isinstance(paper_blocks, str):
            blocks_payload: List[Dict[str, Any]] = [{"type": "text", "text": paper_blocks}]
        else:
            blocks_payload = paper_blocks  # assume list[dict] parts already

        # default to global_review unless explicitly provided
        mode = kwargs.pop("mode", "global_review")
        if mode == "section_review":
            structured_paper = kwargs.get("structured_paper")
            paper_memory = kwargs.get("paper_memory")
            if structured_paper is None or paper_memory is None:
                raise ValueError("section_review requires structured_paper and paper_memory")
            return await self.run_review(
                "section_review",
                structured_paper=structured_paper,
                paper_memory=paper_memory,
            )

        return await self.run_review("global_review", paper_blocks=blocks_payload)

    # ========= Unified entry: choose section_review / global_review (non-streaming, returns str) =========
    async def run_review(
        self,
        mode: ReviewMode,
        *,
        # global_review input: must be "full paper interleaved blocks"
        paper_blocks: Optional[List[Dict[str, Any]]] = None,
        # section_review input
        structured_paper: Optional[Union[Dict[str, Any], List[Dict[str, Any]]]] = None,
        paper_memory: Optional[str] = None,
    ) -> str:
        """
        mode = "global_review":
            - Input paper_blocks: Full paper multimodal interleaved blocks (list)
            - Output: str (non-streaming)

        mode = "section_review":
            - Input structured_paper + paper_memory
            - Output: str (non-streaming)
        """
        if mode == "global_review":
            if not paper_blocks:
                raise ValueError("global_review requires `paper_blocks` (interleaved blocks list).")
            return await self._global_review_text(paper_blocks)

        if mode == "section_review":
            if structured_paper is None or paper_memory is None:
                raise ValueError("section_review requires `structured_paper` and `paper_memory`.")
            return await self.run_sectionwise(structured_paper, paper_memory)

        raise ValueError(f"Unknown mode: {mode}")

    # ========= global_review: input is blocks (non-streaming) =========
    async def _global_review_text(self, paper_blocks: List[Dict[str, Any]]) -> str:
        """
        Non-streaming global review: input must be interleaved blocks.
        Returns complete text.
        """
        user_content: List[Dict[str, Any]] = [{"type": "text", "text": self.GLOBAL_USER_PREFIX}]
        user_content.extend(paper_blocks)

        messages = [
            {"role": "system", "content": self.SYSTEM_PROMPT},
            {"role": "user", "content": user_content},
        ]

        logger.info("Calling LLM with retry mechanism (global_review, non-stream)...")

        resp = await self._call_llm_with_retry(
            model=self.model,
            messages=messages,
            temperature=self.config.get("agents.cheating_detector.temperature", None),
        )
        full_text = self._get_text_from_response(resp)
        if not full_text:
            raise RuntimeError("Empty content from LLM (global_review).")
        return full_text

    # ========= section_review: preserve parallel logic (non-streaming) =========
    async def run_sectionwise(
        self,
        structured_paper: Union[Dict[str, Any], List[Dict[str, Any]]],
        paper_memory: str,
    ) -> str:
        """
        Perform section-level critical review on "extracted paper JSON" (parallel LLM calls).
        Returns concatenated large string (non-streaming).
        """
        # 1) Parse sections list
        if isinstance(structured_paper, dict):
            sections = structured_paper.get("sections", [])
        else:
            sections = structured_paper

        # 2) Clean sections
        clean_sections: List[Dict[str, Any]] = []
        for sec in sections:
            title = str(sec.get("title", "")).strip() or "Untitled"
            raw_content = sec.get("content")

            if isinstance(raw_content, list):
                content = raw_content if raw_content else None
            elif isinstance(raw_content, str):
                content = raw_content.strip()
            else:
                content = str(raw_content or "").strip()

            if not content:
                continue
            clean_sections.append({"title": title, "content": content})

        if not clean_sections:
            logger.warning("No valid sections found in structured_paper; fallback to empty result.")
            return "[CheatingDetector] No sections to review.\n"

        num_sections = len(clean_sections)
        logger.info(f"Starting section-wise critical review on {num_sections} sections...")

        # Concurrency control
        max_concurrency = min(self.config.get("concurrency.cheating_detector", 3), num_sections)
        sem = asyncio.Semaphore(max_concurrency)

        async def _review_single_section(idx: int, sec: Dict[str, Any]) -> str:
            sec_title = sec["title"]
            sec_content = sec["content"]
            logger.info(f"Scheduling Section {idx+1}/{num_sections}: {sec_title}")

            section_header = f"\n\n[SECTION {idx+1}] {sec_title}\n"

            base_prompt = (
                "# GLOBAL MEMORY (for context)\n"
                f"{paper_memory}\n\n"
                "# FOCUSED SECTION (to review)\n"
                f"# {sec_title}\n"
            )

            # section content supports interleaved blocks
            if isinstance(sec_content, list):
                section_user_content: Any = [{"type": "text", "text": base_prompt}]
                section_user_content.extend(sec_content)
            else:
                section_user_content = base_prompt + f"{sec_content}\n"

            messages = [
                {"role": "system", "content": self.SECTION_SYSTEM_PROMPT},
                {"role": "user", "content": section_user_content},
            ]

            async with sem:
                logger.info(f"Reviewing Section {idx+1}: {sec_title}")
                try:
                    resp = await self._call_llm_with_retry(
                        model=self.model,
                        messages=messages,
                        temperature=self.config.get("agents.cheating_detector.temperature", None),
                    )
                except Exception as e:
                    err_msg = f"[CheatingDetector ERROR in section '{sec_title}']: {e}\n"
                    logger.error(err_msg)
                    return section_header + err_msg

            try:
                section_text = self._get_text_from_response(resp)
            except Exception as e:
                err_msg = f"[CheatingDetector] Error parsing section '{sec_title}' response: {e}\n"
                logger.error(err_msg)
                return section_header + err_msg

            return section_header + (section_text or "")

        tasks = [_review_single_section(idx, sec) for idx, sec in enumerate(clean_sections)]
        results: List[str] = await asyncio.gather(*tasks)

        full_result = "".join(results)
        logger.info("Section-level critical review completed (parallel).")
        return full_result
