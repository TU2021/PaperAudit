"""
Modified from: https://github.com/NeuroDong/Ai-Review
"""
import json
from typing import Any, AsyncGenerator, Optional, List

from .base_agent import BaseAgent
from .logger import get_logger

logger = get_logger(__name__)


class DeepReviewerAgent(BaseAgent):
    """Agent for testing the /paper_review endpoint"""

    def __init__(
        self,
        *,
        model: str,
        base_url: Optional[str] = None,
        api_key: Optional[str] = None,
    ):
        super().__init__(
            model=model,
            base_url=base_url,
            api_key=api_key,
        )

    SYSTEM_PROMPT = """
You are an expert academic reviewer tasked with providing a thorough and balanced evaluation of research papers. 
Your thinking mode is Best Mode. In this mode, you should aim to provide the most reliable review results by conducting a thorough analysis of the paper. 
For each paper submitted, conduct a comprehensive review addressing the following aspects:

Use EXACTLY these section headings in this order (no extras, no omissions):
   - Summary
   - Soundness
   - Presentation
   - Contribution
   - Strengths
   - Weaknesses
   - Questions
   - Rating

1. Summary: Briefly outline main points and objectives.
2. Soundness: Assess methodology and logical consistency.
3. Presentation: Evaluate clarity, organization, and visual aids.
4. Contribution: Analyze significance and novelty in the field.
5. Strengths: Identify the paper's strongest aspects.
6. Weaknesses: Point out areas for improvement.
7. Questions: Pose questions for the authors.
8. Rating: 
    - Provide numeric scores and a one-line justification with manuscript anchors for each category.
    - Use EXACTLY the following Markdown list format:
        - Overall (10): <integer 0–10> — <one-sentence justification with anchors>
        - Novelty (10): <integer 0–10> — <one-sentence justification with anchors>
        - Technical Quality (10): <integer 0–10> — <one-sentence justification with anchors>
        - Clarity (10): <integer 0–10> — <one-sentence justification with anchors>
        - Confidence (5): <integer 0–5> — <one-sentence note about reviewer confidence and basis>

Maintain objectivity and provide specific examples from the paper to support your evaluation.

You MUST produce EXACTLY 4 complete reviews by simulating *4* different reviewers. Each review MUST be self-contained and complete, and use self-verification to double-check any paper deficiencies identified. Finally, provide complete review results.
"""

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
        Execute paper review task with streaming response (SSE-style strings).
        """
        logger.info("Starting paper review...")
        logger.info(f"Query: {query[:100] if query else '(empty)'}...")

        paper_blocks = self.prepare_paper_blocks(paper_json)
        paper_content = self.blocks_to_prompt_content(paper_blocks, enable_mm=enable_mm)
        log_text = self.blocks_to_text(paper_blocks, enable_mm=True)
        logger.info(f"Extracted paper text length: {len(log_text)} characters (with image markers if any)")

        if not log_text:
            logger.warning("WARNING: No text extracted from PDF!")
            yield (
                "data: "
                + json.dumps(
                    {
                        "object": "chat.completion.chunk",
                        "choices": [{"delta": {"content": "Error: Failed to extract text from PDF"}}],
                    }
                )
                + "\n\n"
            )
            yield "data: [DONE]\n\n"
            return

        if enable_mm:
            user_content: List[dict] = [{"type": "text", "text": "Review the following paper.\n\nPaper:"}]
            if isinstance(paper_content, list):
                user_content.extend(paper_content)
            else:
                user_content.append({"type": "text", "text": str(paper_content)})
            user_content.append({"type": "text", "text": f"\nInstruction: {query}"})
        else:
            user_content = self.USER_PROMPT_TEMPLATE.format(text=paper_content, query=query)

        messages = [
            {"role": "system", "content": self.SYSTEM_PROMPT},
            {"role": "user", "content": user_content},
        ]

        logger.info("Calling LLM with retry mechanism (non-stream)...")
        logger.info(f"Model: {self.model}")
        logger.info(f"Base URL: {self.client.base_url}")

        try:
            response = await self._call_llm_with_retry(
                model=self.model,
                messages=messages,
                temperature=self.config.get("agents.summarizer.temperature", None),
            )
            full_text = self._get_text_from_response(response)
        except Exception as e:
            logger.error(f"Error during LLM call: {type(e).__name__}: {str(e)}")
            import traceback

            logger.error(traceback.format_exc())
            error_message = f"Error: {type(e).__name__}: {str(e)}"
            yield (
                "data: "
                + json.dumps(
                    {
                        "object": "chat.completion.chunk",
                        "choices": [{"delta": {"content": error_message}}],
                    }
                )
                + "\n\n"
            )
            yield "data: [DONE]\n\n"
            return

        response_data = {
            "object": "chat.completion.chunk",
            "choices": [{"delta": {"content": full_text}}],
        }
        yield f"data: {json.dumps(response_data)}\n\n"
        yield "data: [DONE]\n\n"

    async def review_paper(
        self,
        *,
        paper_json: Any,
        query: str,
        enable_mm: bool = False,
    ) -> str:
        """
        Convenience wrapper: collect run() SSE yields and return final review text (plain string).

        This is designed for batch runners (threadpool + per-thread event loop).
        """
        chunks: List[str] = []
        async for sse in self.run(paper_json=paper_json, query=query, enable_mm=enable_mm):
            if not isinstance(sse, str) or not sse.startswith("data:"):
                continue
            payload = sse[len("data:") :].strip()
            if payload == "[DONE]":
                break
            try:
                obj = json.loads(payload)
                delta = obj.get("choices", [{}])[0].get("delta", {})
                content = delta.get("content", "")
                if content:
                    chunks.append(content)
            except Exception:
                # ignore malformed chunks
                pass

        return "".join(chunks).strip()
