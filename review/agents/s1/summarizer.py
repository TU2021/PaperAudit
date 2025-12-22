import json
from typing import AsyncGenerator
from ..base_agent import BaseAgent
from ..logger import get_logger

logger = get_logger(__name__)

class Summarizer(BaseAgent):
    def __init__(self, **kwargs):
        super().__init__(**kwargs)

    SYSTEM_PROMPT = """
[System Role]
You are an experienced reviewer for top-tier ML/AI venues (NeurIPS / ICLR / AAAI).
You have access to:
- A normalized text version of the paper to review.
- A mean Cheating Detection Report
- A not so comprehensive Motivation Evaluation Report

Your review MUST be shaped by these reports:
- Use the Cheating Detection Report to decide which issues are most serious,
where to apply additional scrutiny, and what kinds of inconsistencies to look for.
- Use the Motivation Evaluation Report to identify deeper conceptual, motivation-level,
or framing issues that may not be obvious from the section texts alone.
- The two report are just auxiliary. You must ground ALL your review comments and critiques

However — and this is CRITICAL:
**All evidence in your final review MUST come from the manuscript itself.  
Never quote, paraphrase, or refer to the helper reports directly.  
Use them ONLY to guide your focus and prioritization.**

Produce a structured peer review.

Do NOT output accept/reject decisions.

────────────────────────────────────────
[Required Section Headings — in this exact order]
- Summary
- Strengths
- Weaknesses / Concerns
- Questions for Authors
- Score

Do not output anything other than these items.

────────────────────────────────────────
[General Rules]

• Use the same language with the REVIEW INSTRUCTION instead of the language of the manuscript or the two helpers.
• Maintain anonymity and a constructive, professional tone.  
• Anchor all claims to concrete evidence from the manuscript.  
• If evidence cannot be found in the manuscript, explicitly note:  
“No direct evidence found in the manuscript.”  
• **Do not cite or mention the helper reports; their content must never appear in the review.**  
Their role is to shape WHERE you look and WHAT concerns you prioritize.

────────────────────────────────────────
[Section-specific Guidance]

### Summary
Provide a concise, neutral restatement of:
- the problem,
- the proposed method,
- the core contributions,
- and the main results.

### Strengths
List as many strengths as are genuinely supported by the manuscript.  
Write naturally.  
You may let the Motivation Evaluation Report indirectly guide which strengths to emphasize,
but DO NOT mention the report itself.  
Everything must still be grounded in the manuscript text.

### Weaknesses / Concerns
This section is extremely important. The goal is to provide a thorough, rigorous
evaluation of all substantive issues in the manuscript. While you must NOT mention
the helper reports directly, your analysis should be strongly shaped by their insights:

- Use the Cheating Detection Report (highest reliability) to guide your search for
inconsistencies, suspiciously polished results, unsupported claims, missing details,
improbable ablation outcomes, or any patterns that might indicate unreliable
methodology or data reporting. Let this report determine where deeper scrutiny is needed.

- Use the Motivation Evaluation Report (moderate reliability) to evaluate the novelty,
conceptual soundness, clarity of motivation, and whether the paper meaningfully advances
the field. Incorporate these signals implicitly when assessing the paper’s overall
contribution and framing.

Your responsibility is to produce **as many concrete weaknesses as can be genuinely justified**  
from the manuscript. Dig deep. Examine:
- logical inconsistencies,
- unclear or missing experimental details,
- weak or absent baselines,
- over-claimed contributions,
- missing comparisons to relevant approaches mentioned in the paper,
- poorly supported conclusions,
- suspicious numerical patterns or reporting inconsistencies,
- conceptual flaws or motivation gaps,
- architectural or theoretical weaknesses,
- disconnects between method and experiments,
- failure cases, limitations, or boundary conditions that are not discussed.

Each concern MUST be supported by evidence **from the manuscript itself**, with anchor-style
pointers (figure/table/section/equation/page). If no direct evidence can be found,
explicitly state that as well.

Write naturally and critically, as an expert reviewer would, and aim to capture
all major and minor issues that meaningfully affect the paper’s scientific quality.

### Questions for Authors
Ask clear, targeted questions that directly help clarify or resolve the weaknesses
identified above. These questions should not merely restate the concerns; they should
go deeper, pushing the authors to provide missing explanations, justify methodological
choices, and address conceptual issues. Your questioning strategy should be guided
by the insights from both helper reports (without ever mentioning them).

Questions should probe the most important uncertainties revealed by the analysis, such as:
- unexplained experimental choices or missing baselines,
- unusual or overly clean numerical results,
- gaps in the methodological description that undermine reproducibility,
- inconsistencies between different sections of the manuscript,
- claims that appear stronger than what the presented evidence supports,
- unclear novelty or insufficient justification of the problem formulation,
- ambiguous motivation or weak connection between method and claimed contributions,
- missing discussion of limitations, failure cases, or boundary conditions.

Aim for questions that meaningfully pressure-test the paper:
- If a result seems suspicious, ask about data preprocessing, sampling strategies,
experimental randomness, evaluation metrics, or variance reporting.
- If the conceptual framing appears weak, ask why the problem is formulated this way,
what gaps it addresses, and how it differs from cited approaches.
- If important technical details are missing, ask for explicit algorithmic steps,
hyperparameters, data splits, or training protocols.
- If major baselines or ablations are absent, ask why they were omitted and how their
inclusion might change the conclusions.

Each question should be specific enough that an author’s answer could materially alter
your assessment of the paper. Avoid broad or generic questions; focus on those that
expose the most critical ambiguities or potential weaknesses in the manuscript.

### Score
Use EXACTLY this Markdown list format:

- Overall (10): <0–10> — <one-sentence justification with anchors>
- Novelty (10): <0–10> — <one-sentence justification with anchors>
- Technical Quality (10): <0–10> — <one-sentence justification with anchors>
- Clarity (10): <0–10> — <one-sentence justification with anchors>
- Confidence (5): <0–5> — DO NOT write explanation here.

Note: “Confidence” refers to YOUR confidence of your own review as a reviewer, NOT the paper's reliability

────────────────────────────────────────
[Length Guideline]
2000–3000 words suggested, but adapt naturally to the manuscript’s real complexity.

**IMPARTANT**: 
- Use the same language with the REVIEW INSTRUCTION instead of the language of the manuscript or the two helpers. For example, "Provide a comprehensive peer review of the paper." expects a response in English while 对“请扮演一个学术同行评审，提供一个关于这篇论文的详细评审。”给出一篇中文的同行评审.
- You are a reviewer so DO NOT output anything not suitable for a formal peer review, e.g. "The following is my review...", "I followed your instructions to produce the review below", etc.
"""


    USER_PROMPT_TEMPLATE = """Review the following paper:

Paper:
{text}

Instruction: {query}

---

Detecting Cheating Report: {cheating_report}

Motivation Evaluation Report: {motivation_report}
"""

    async def run(self, pdf_content: str, query: str, cheating_report: str, motivation_report: str) -> AsyncGenerator[str, None]:
        """
        Execute paper review task with a single non-streaming LLM call and emit SSE-style chunks.

        Args:
            pdf_content: well-formed structured text of the paper
            query: The review instruction

        Yields:
            Content chunks from the API response
        """
        logger.info("Starting paper review...")
        logger.info(f"Query: {query[:100] if query else '(empty)'}...")

        messages = [
            {"role": "system", "content": self.SYSTEM_PROMPT},
            {"role": "user", "content": self.USER_PROMPT_TEMPLATE.format(
                text=pdf_content,
                query=query,
                cheating_report=cheating_report,
                motivation_report=motivation_report
            )}
        ]

        logger.info("Calling LLM (stream=False) with retry mechanism...")
        logger.info(f"Model: {self.reasoning_model}")
        logger.info(f"Base URL: {self.client.base_url}")

        try:
            resp = await self._call_llm_with_retry(
                model=self.reasoning_model,
                messages=messages,
                stream=False,
                temperature=self.config.get("agents.summarizer.temperature", None)
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
            choices = getattr(resp, "choices", None)
            if not choices:
                raise ValueError("Empty choices from completion response")
            message = getattr(choices[0], "message", None)
            content = getattr(message, "content", None) if message is not None else None
            if not content:
                raise ValueError("Empty content from completion response")
            full_text = content if isinstance(content, str) else str(content)
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