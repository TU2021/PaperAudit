"""
Paper Review Agent
"""
import json
from typing import Any, AsyncGenerator, Optional

from .base_agent import BaseAgent
from .logger import get_logger

logger = get_logger(__name__)

class BaseLineAgent(BaseAgent):
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

    async def run(
        self,
        paper_json: Any,
        query: str,
        *,
        enable_mm: bool = False,
    ) -> AsyncGenerator[str, None]:
        """Execute paper review task with streaming response."""

        paper_blocks = self.prepare_paper_blocks(paper_json)
        paper_content = self.blocks_to_prompt_content(paper_blocks, enable_mm=enable_mm)

        log_text = self.blocks_to_text(paper_blocks, enable_mm=True)
        logger.info(f"Prepared paper content: {len(log_text)} characters (including image markers if any)")

        # Build prompt with PDF content
        if enable_mm:
            prompt_content = [
                {
                    "type": "text",
                    "text": "Review the following paper."
                            "\n\nPaper:",
                },
            ]
            if isinstance(paper_content, list):
                prompt_content.extend(paper_content)
            else:
                prompt_content.append({"type": "text", "text": str(paper_content)})
            prompt_content.append({"type": "text", "text": f"\nInstruction: {query}"})
        else:
            prompt_content = (
                f"""Review the following paper:

Paper:
{paper_content}

Instruction: {query}"""
            )

        temp = self.config.get("agents.baseline.temperature", None)
        try:
            response = await self._call_llm_with_retry(
                model=self.model,
                messages=[{"role": "user", "content": prompt_content}],
                max_tokens=2048,
                temperature=temp,
            )
            full_text = self._get_text_from_response(response)
        except Exception as e:
            error_message = f"Error: {type(e).__name__}: {e}"
            response_data = {
                "object": "chat.completion.chunk",
                "choices": [{
                    "delta": {
                        "content": error_message
                    }
                }]
            }
            yield f"data: {json.dumps(response_data)}\n\n"
            yield "data: [DONE]\n\n"
            return

        # Stream back results as single completion chunk
        response_data = {
            "object": "chat.completion.chunk",
            "choices": [{
                "delta": {
                    "content": full_text
                }
            }]
        }
        yield f"data: {json.dumps(response_data)}\n\n"
        yield "data: [DONE]\n\n"