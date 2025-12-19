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

        # Call LLM model with streaming
        temp = self.config.get("agents.baseline.temperature", None)
        stream = await self.client.chat.completions.create(
            model=self.model,
            messages=[{"role": "user", "content": prompt}],
            max_tokens=2048,
            temperature=temp,
            stream=True
        )

        # Stream back results
        async for chunk in stream:
            if chunk.choices and len(chunk.choices) > 0:
                delta_content = chunk.choices[0].delta.content
                if delta_content:
                    response_data = {
                        "object": "chat.completion.chunk",
                        "choices": [{
                            "delta": {
                                "content": delta_content
                            }
                        }]
                    }
                    yield f"data: {json.dumps(response_data)}\n\n"

        yield "data: [DONE]\n\n"