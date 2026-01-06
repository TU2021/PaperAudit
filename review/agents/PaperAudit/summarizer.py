# agents/PaperAudit/summarizer.py
import json
from typing import Any, AsyncGenerator, Optional

from ..base_agent import BaseAgent
from ..logger import get_logger
from typing import List


logger = get_logger(__name__)


class Summarizer(BaseAgent):
    def __init__(self, **kwargs):
        super().__init__(**kwargs)

    SYSTEM_PROMPT = """
[System Role]
You are an experienced reviewer for top-tier ML/AI venues (AAAI/NeurIPS/ICLR style).
Produce a text-only, structured review with NO accept/reject decision.

[LANGUAGE]
Use the same language as the USER PROMPT.

[Critical Constraints]
1) Use EXACTLY these section headings in this order (no extras, no omissions):
   - Summary
   - Strengths
   - Weaknesses
   - Suggestions for Improvement
   - Score
2) Do NOT output any scores, ratings, or accept/reject verdict.
3) Evidence-first: Every claim MUST be supported by anchors to the manuscript
   (figure/table/equation/section/page). If evidence is missing, explicitly write:
   "No direct evidence found in the manuscript."
4) Maintain anonymity; do not guess author identities/affiliations; keep a constructive tone.
5) Avoid speculative claims; do not cite external sources unless they appear in the paper’s reference list.

[Input]
- Full anonymous manuscript (plain text or OCR output).

[OUTPUT TEMPLATE]

1) Summary
   - Concisely and neutrally restate the problem, method, core contributions, and main results (≤150 words).
   - Avoid subjective judgments or decision-like language.

2) Strengths
   - Generate AS MANY items as the manuscript supports (≥3 encouraged; more is better).
   - Use UNNUMBERED bullet items with concise BOLDED titles (no numbering).
   - For each item, include sub-point examples (≥3 encouraged; more is better) that belong to the item.
   - Each sub-point example should include evidence (Figure/Table/Section/Page references supporting this strength) and why it matters (novelty/technical soundness/experimental rigor/clarity/impact).

3) Weaknesses
   - Generate AS MANY items as the manuscript supports (≥3 encouraged; more is better).
   - Use UNNUMBERED bullet items with concise BOLDED titles (no numbering).
   - For each item, include sub-point examples (≥3 encouraged; more is better) that belong to the item.
   - Each sub-point example should include evidence (Figure/Table/Section/Page references supporting this strength) and why it matters (novelty/technical soundness/experimental rigor/clarity/impact).

4) Suggestions for Improvement
   - Provide concrete, actionable, and verifiable recommendations; the number of recommendations should be the same as the number of Weaknesses, and they should correspond one to one.
   - Use UNNUMBERED bullet items with concise BOLDED titles (no numbering).
   - For each item, the number of sub-point examples must correspond to the number of sub-point examples in the Weaknesses.

5) Score
   - Provide numeric scores and a one-line justification with manuscript anchors for each category.
   - Use EXACTLY the following Markdown list format:
     - Overall (10): <integer 0–10> — <one-sentence justification with anchors>
     - Novelty (10): <integer 0–10> — <one-sentence justification with anchors>
     - Technical Quality (10): <integer 0–10> — <one-sentence justification with anchors>
     - Clarity (10): <integer 0–10> — <one-sentence justification with anchors>
     - Confidence (5): <integer 0–5> — <one-sentence note about reviewer confidence and basis>

[Style & Length]
- Tone: objective, polite, and constructive.
- Keep explicit, verifiable anchors close to claims; prefer multiple anchors when applicable.
- Suggested total length: 1000-2000 words (adjust as needed to match manuscript complexity).

"""

    USER_PROMPT_BASELINE = """Review the following paper.

Paper:
{text}

Instruction: {query}
"""

    USER_PROMPT_REFINE = """
You will REFINE an existing baseline review with MINIMAL edits.

Your task:
- Re-check the manuscript and apply small, targeted corrections to the baseline review.
- Use the two auxiliary reports ONLY as silent attention cues for WHERE to re-check.
- The baseline review remains the primary and authoritative review.

────────────────────────────────────────
[NON-NEGOTIABLE RULES]

1) Auxiliary reports = navigation only
- Do NOT mention, cite, quote, paraphrase, or allude to the auxiliary reports in any form.
- Do NOT treat auxiliary reports as evidence. They only hint where to re-check.
- Auxiliary reports may guide where to re-check, but must NOT influence the severity, framing, or confidence of any critique unless the manuscript itself independently justifies such changes.

2) Minimal edits only
- Do NOT rewrite or restructure the review.
- Preserve the original structure, bullet organization, wording style, and overall tone.
- Prefer small clarifications, corrections, or tightening wording. you MAY add a new point if it is clearly warranted by the manuscript, directly addresses a material issue, and was genuinely missed in the baseline review.
- Avoid overly aggressive or accusatory language; keep a constructive reviewer tone.

3) Reviewer-style output only
- The final output MUST read like a standard academic peer review, not an audit report or forensic analysis.
- Do NOT use block IDs, chunk numbers, line numbers, or any other implementation-specific localization markers
  (e.g., "Block #", "Chunk #", "Span #").
- Use only conventional manuscript anchors (Section, Figure, Table, Equation, Appendix, Page) when grounding claims.
- Avoid checklist-like or enumerative auditing language; maintain a natural reviewer tone throughout.

────────────────────────────────────────
[SCORING POLICY — VERY IMPORTANT]

Score Stability Priority
- The baseline review remains the authoritative reference.
- Score re-evaluation is required as a verification step, not a default trigger for change.
- When uncertainty exists, prioritize score stability over modification.

Score Re-evaluation Requirement

- You MUST re-evaluate the review scores after re-checking the manuscript, using the two auxiliary reports ONLY as silent navigation cues.

- If closer inspection uncovers previously missed, high-confidence issues that materially affect the core contribution or main conclusions, you MAY modestly lower the scores.
  Examples include (but are not limited to):
  (i) a core methodological assumption that is violated or unjustified in the manuscript,
  (ii) a key experimental result that is incorrectly reported or not supported by the presented evidence,
  or (iii) a central claim that is contradicted by the paper’s own analysis, figures, or ablations.

- Conversely, if closer inspection shows that the baseline review overstates issues, mischaracterizes their severity, or underestimates the paper’s contribution or soundness, you SHOULD modestly adjust the scores upward.

- Any score change must be conservative and explicitly grounded in the manuscript.

────────────────────────────────────────
[EDITING PRINCIPLES]

- Keep the same section headers and bullet layout as the baseline review.
- Only adjust lines that are clearly incorrect, overstated, or missing a needed anchor.
- Re-calibrating the severity, scope, or phrasing of an existing critique based on closer manuscript inspection
  counts as a minimal edit, as long as no new substantive claims are introduced.
- If you add a new Weakness item, also add the corresponding Suggestion item with matched subpoints.
- If you soften language, do so without removing substantive critique.

────────────────────────────────────────
[INPUTS]

Baseline Review (to be minimally refined):
{baseline_review}

Paper Memory (navigation only; do NOT cite):
{paper_memory}

Silent Attention Cues (DO NOT cite or mention):
[Cheating-related cues]
{cheating_report}

[Motivation-related cues]
{motivation_report}

────────────────────────────────────────
[OUTPUT]

- Return the refined review text only.
- Preserve the baseline review’s structure and formatting as much as possible.
- All modified or added claims MUST include conventional manuscript anchors.
- The refined review should be indistinguishable in style from a careful human peer review.
- Do NOT expose internal checking processes, re-verification steps, or analysis artifacts.
- The auxiliary reports must remain completely invisible in the final output.
"""



    async def run(
        self,
        pdf_content: Any,
        query: str,
        cheating_report: str = "",
        motivation_report: str = "",
        *,
        baseline_review: Optional[str] = None,
        paper_memory: Optional[str] = None,
        mode: str = "auto",  # "baseline" | "refine" | "auto"
    ) -> AsyncGenerator[str, None]:
        """
        Backward-compatible entrypoint required by BaseAgent (abstract method).

        - mode="baseline": run baseline only (no helper reports)
        - mode="refine":   run refine/update; requires baseline_review (or will compute it)
        - mode="auto":     if baseline_review provided -> refine; else baseline then refine
        """
        mode_norm = (mode or "auto").strip().lower()
        if mode_norm not in ("baseline", "refine", "auto"):
            mode_norm = "auto"

        if mode_norm == "baseline":
            async for chunk in self.run_baseline(pdf_content, query):
                yield chunk
            return

        # refine / auto
        if baseline_review is None:
            # compute baseline first, then refine
            baseline_chunks: List[str] = []
            async for ch in self.run_baseline(pdf_content, query):
                # keep same SSE format passthrough
                yield ch
                try:
                    # parse SSE to accumulate baseline text for refine
                    if isinstance(ch, str) and ch.startswith("data: "):
                        payload = ch[6:].strip()
                        if payload != "[DONE]":
                            data = json.loads(payload)
                            delta = data.get("choices", [{}])[0].get("delta", {})
                            txt = delta.get("content")
                            if txt:
                                baseline_chunks.append(txt)
                except Exception:
                    pass
            baseline_review = "".join(baseline_chunks)

        # now refine
        async for chunk in self.run_refine(
            pdf_content,
            query,
            baseline_review=baseline_review or "",
            paper_memory=paper_memory or "",
            cheating_report=cheating_report or "",
            motivation_report=motivation_report or "",
        ):
            yield chunk

    async def _run_once_emit_sse(self, messages: list) -> AsyncGenerator[str, None]:
        logger.info("Calling LLM (non-stream) with retry mechanism...")
        logger.info(f"Model: {self.model}")
        logger.info(f"Base URL: {self.client.base_url}")

        try:
            resp = await self._call_llm_with_retry(
                model=self.model,
                messages=messages,
                temperature=self.config.get("agents.summarizer.temperature", None),
            )
        except Exception as e:
            logger.error(f"Error during LLM call: {type(e).__name__}: {str(e)}")
            import traceback

            logger.error(traceback.format_exc())
            error_message = f"Error: {type(e).__name__}: {str(e)}"
            yield f"data: {json.dumps({'object': 'chat.completion.chunk', 'choices': [{'delta': {'content': error_message}}]})}\n\n"
            yield "data: [DONE]\n\n"
            return

        try:
            full_text = self._get_text_from_response(resp)
        except Exception as parse_err:
            logger.error(f"Error while parsing completion response: {parse_err}")
            error_message = f"Error: {type(parse_err).__name__}: {parse_err}"
            yield f"data: {json.dumps({'object': 'chat.completion.chunk', 'choices': [{'delta': {'content': error_message}}]})}\n\n"
            yield "data: [DONE]\n\n"
            return

        # Emit SSE-style chunks from the full text for compatibility with upstream consumers
        for paragraph in full_text.split("\n\n"):
            para = paragraph.strip()
            if not para:
                continue
            response_data = {
                "object": "chat.completion.chunk",
                "choices": [{"delta": {"content": para}}],
            }
            yield f"data: {json.dumps(response_data)}\n\n"

        yield "data: [DONE]\n\n"

    def _build_user_content(self, prompt_text: str, pdf_content: Any) -> Any:
        if isinstance(pdf_content, list):
            # multimodal parts: prefix the instruction as a text part, then append paper parts
            user_content: Any = [{"type": "text", "text": prompt_text}]
            user_content.extend(pdf_content)
            return user_content
        return prompt_text

    async def run_baseline(self, pdf_content: Any, query: str) -> AsyncGenerator[str, None]:
        """
        Stage A: baseline review using ONLY the paper (no auxiliary reports, no memory, no baseline).
        Emits SSE-style chunks.
        """
        logger.info("Starting BASELINE paper review...")
        logger.info(f"Query: {query[:100] if query else '(empty)'}...")

        prompt_text = self.USER_PROMPT_BASELINE.format(text="" if isinstance(pdf_content, list) else pdf_content, query=query)
        user_content = self._build_user_content(prompt_text, pdf_content)

        messages = [
            {"role": "system", "content": self.SYSTEM_PROMPT},
            {"role": "user", "content": user_content},
        ]

        async for chunk in self._run_once_emit_sse(messages):
            yield chunk

    async def run_refine(
        self,
        pdf_content: Any,
        query: str,
        *,
        baseline_review: str,
        paper_memory: Optional[str],
        cheating_report: str,
        motivation_report: str,
    ) -> AsyncGenerator[str, None]:
        """
        Stage B: refine/update the baseline review using auxiliary reports as silent attention cues.
        Emits SSE-style chunks.
        """
        logger.info("Starting REFINED paper review (update mode)...")
        logger.info(f"Query: {query[:100] if query else '(empty)'}...")

        prompt_text = self.USER_PROMPT_REFINE.format(
            query=query,
            baseline_review=baseline_review,
            paper_memory=paper_memory or "",
            cheating_report=cheating_report,
            motivation_report=motivation_report,
            # text="" if isinstance(pdf_content, list) else pdf_content,
        )
        user_content = self._build_user_content(prompt_text, pdf_content)

        messages = [
            {"role": "system", "content": self.SYSTEM_PROMPT},
            {"role": "user", "content": user_content},
        ]

        async for chunk in self._run_once_emit_sse(messages):
            yield chunk
