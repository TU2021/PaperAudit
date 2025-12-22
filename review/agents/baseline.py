"""
Paper Review Agent
"""
import json
from typing import Any, AsyncGenerator, Optional

from .base_agent import BaseAgent

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

        text = self.blocks_to_text(self.prepare_paper_blocks(paper_json), enable_mm=enable_mm)

        # Build prompt with PDF content
        prompt = f"""Review the following paper:

Paper:
{text}

Instruction: {query}"""

        temp = self.config.get("agents.baseline.temperature", None)
        try:
            response = await self._call_llm_with_retry(
                model=self.model,
                messages=[{"role": "user", "content": prompt}],
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