"""
Modified from: https://github.com/NeuroDong/Ai-Review
"""
import json
from typing import Any, AsyncGenerator, Optional

from .base_agent import BaseAgent
from .logger import get_logger

logger = get_logger(__name__)

class NeuroDongAgent(BaseAgent):
    """Agent for testing the /paper_review endpoint"""

    def __init__(
        self,
        *,
        model: str,
        reasoning_model: Optional[str] = None,
        embedding_model: Optional[str] = None,
        base_url: Optional[str] = None,
        api_key: Optional[str] = None,
        embedding_base_url: Optional[str] = None,
        embedding_api_key: Optional[str] = None,
    ):
        super().__init__(
            model=model,
            reasoning_model=reasoning_model,
            embedding_model=embedding_model,
            base_url=base_url,
            api_key=api_key,
            embedding_base_url=embedding_base_url,
            embedding_api_key=embedding_api_key,
        )
    
    SYSTEM_PROMPT = """
[System Role] You are an experienced reviewer for top-tier ML/AI venues (AAAI/NeurIPS/ICLR style).
Produce a text-only, structured review with scores in the Score section; do NOT output any accept/reject decision.

[Critical Constraints]
1) Use EXACTLY these section headings in this order (no extras, no omissions):
   - Summary
   - Strengths
   - Weaknesses / Concerns
   - Questions for Authors
   - Suggestions for Improvement
   - References
   - Score
2) Evidence-first: Every claim MUST be supported by anchors to the manuscript
   (figure/table/equation/section/page). If evidence is missing, explicitly write:
   "No direct evidence found in the manuscript."
3) Maintain anonymity; do not guess author identities/affiliations; keep a constructive tone.
4) Avoid speculative claims; do not cite external sources unless they appear in the paper's reference list.
5) Use the same language with the USER PROMPT, instead of the language of the manuscript. For example, "Provide a comprehensive peer review of the paper." expects a response in English while 对“请扮演一个学术同行评审，提供一个关于这篇论文的详细评审。”给出一篇中文的同行评审.

[Input]
- User prompt to guide the review, which decides your language preference.
- Full anonymous manuscript (plain text or OCR output).

[Output Template]
Write the review using the seven headings—exactly these and only these:

1) Summary
   - Concisely and neutrally restate the problem, method, core contributions, and main results (≤150 words).
   - Avoid subjective judgments or decision-like language.

2) Strengths
   - Generate AS MANY items as the manuscript supports (≥3 encouraged; more is better).
   - Use UNNUMBERED bullet items with concise BOLDED titles (no numbering).
   - For each item, include sub-point examples (≥3 encouraged; more is better) that belong to the item.
   - Each sub-point example should include evidence (Figure/Table/Section/Page references supporting this strength) and why it matters (novelty/technical soundness/experimental rigor/clarity/impact).
   - Coverage suggestions (if information allows): problem setting/assumptions; relation to prior work; method limitations; experimental design/statistical significance; generalization/fairness/robustness; reproducibility/resource consumption; ethics/social impact; writing clarity; etc.

3) Weaknesses / Concerns
   - Generate AS MANY items as the manuscript supports (≥3 encouraged; more is better).
   - Include one item that evaluates the correctness, clarity, or consistency of mathematical formulations (e.g., equations, notation, derivations).
   - Use UNNUMBERED bullet items with concise BOLDED titles (no numbering).
   - For each item, include sub-point examples (≥3 encouraged; more is better) that belong to the item.
   - Each sub-point example should include evidence (Figure/Table/Section/Page references supporting this weakness) and why it matters (novelty/technical soundness/experimental rigor/clarity/impact).
   - Coverage suggestions (if information allows): problem setting/assumptions; relation to prior work; method limitations; experimental design/statistical significance; generalization/fairness/robustness; reproducibility/resource consumption; ethics/social impact; writing clarity; etc.

4) Questions for Authors
   - List succinct, targeted questions that, if answered, would help clarify or address the concerns above.
   - Use bullet points.

5) Suggestions for Improvement
    - Provide concrete, actionable, and verifiable recommendations; the number of recommendations should be the same as the number of Weaknesses, and they should correspond one to one.
    - Use UNNUMBERED bullet items with concise BOLDED titles (no numbering).
    - For each item, the number of sub-point examples must correspond to the number of sub-point examples in the Weaknesses.

6) References
   - List ONLY items that you explicitly cite within this review AND that appear in the manuscript's reference list.

7) Score
   - Provide numeric scores and a one-line justification (with evidence anchors) for each category.
   - Use this exact format (Markdown list):
     - Overall (10): <integer 0–10> — <one-sentence justification with anchors>
     - Novelty (10): <integer 0–10> — <one-sentence justification with anchors>
     - Technical Quality (10): <integer 0–10> — <one-sentence justification with anchors>
     - Clarity (10): <integer 0–10> — <one-sentence justification with anchors>
     - Confidence (5): <integer 0–5> — <one-sentence note about reviewer confidence and basis>

[Style & Length]
- Tone: objective, polite, and constructive.
- Keep explicit, verifiable anchors close to claims; prefer multiple anchors when applicable.
- Suggested total length: 800–1200 words (adjust as needed to match manuscript complexity)."""

    USER_PROMPT_TEMPLATE = """Review the following paper:

Paper:
{text}

Instruction: {query}"""

    async def run(
        self,
        paper_json: Any,
        query: str,
        *,
        enable_mm: bool = False,
    ) -> AsyncGenerator[str, None]:
        """
        Execute paper review task with streaming response

        Args:
            paper_json: Pre-parsed paper content JSON (dict/string/path)
            query: The review instruction

        Yields:
            Content chunks from the API response
        """
        logger.info(f"Starting paper review...")
        logger.info(f"Query: {query[:100] if query else '(empty)'}...")
        
        text = self.blocks_to_text(self.prepare_paper_blocks(paper_json), enable_mm=enable_mm)
        logger.info(f"Extracted paper text length: {len(text)} characters")
        
        if not text:
            logger.warning("WARNING: No text extracted from PDF!")
            yield f"data: {json.dumps({'object': 'chat.completion.chunk', 'choices': [{'delta': {'content': 'Error: Failed to extract text from PDF'}}]})}\n\n"
            yield "data: [DONE]\n\n"
            return

        messages = [
            {"role": "system", "content": self.SYSTEM_PROMPT},
            {"role": "user", "content": self.USER_PROMPT_TEMPLATE.format(text=text, query=query)}
        ]

        logger.info(f"Calling LLM with retry mechanism...")
        logger.info(f"Model: {self.reasoning_model}")
        logger.info(f"Base URL: {self.client.base_url}")
        
        try:
            # Call LLM model with streaming and retry mechanism
            stream = await self._call_llm_with_retry(
                model=self.reasoning_model,
                messages=messages,
                stream=True,
                temperature=self.config.get("agents.neurodong.temperature", None)
            )
            
            logger.info(f"LLM stream object type: {type(stream)}")
            logger.info("LLM call successful, streaming response...")
            chunk_count = 0
            content_count = 0
            
            # Stream back results
            async for chunk in stream:
                chunk_count += 1
                if chunk_count <= 3:  # 只打印前3个chunk的详细信息
                    logger.debug(f"Chunk #{chunk_count}: {chunk}")
                
                if chunk.choices and len(chunk.choices) > 0:
                    delta_content = chunk.choices[0].delta.content
                    if delta_content:
                        content_count += 1
                        if content_count <= 3:
                            logger.debug(f"Content #{content_count}: {delta_content[:100]}...")
                        response_data = {
                            "object": "chat.completion.chunk",
                            "choices": [{
                                "delta": {
                                    "content": delta_content
                                }
                            }]
                        }
                        yield f"data: {json.dumps(response_data)}\n\n"
                    else:
                        if chunk_count <= 3:
                            logger.debug(f"Chunk #{chunk_count} has no content in delta")
                else:
                    if chunk_count <= 3:
                        logger.debug(f"Chunk #{chunk_count} has no choices")
            
            logger.info(f"Streaming completed. Total chunks: {chunk_count}, Content chunks: {content_count}")
            yield "data: [DONE]\n\n"
            
        except Exception as e:
            logger.error(f"Error during LLM call: {type(e).__name__}: {str(e)}")
            import traceback
            logger.error(traceback.format_exc())
            error_message = f"Error: {type(e).__name__}: {str(e)}"
            yield f"data: {json.dumps({'object': 'chat.completion.chunk', 'choices': [{'delta': {'content': error_message}}]})}\n\n"
            yield "data: [DONE]\n\n"