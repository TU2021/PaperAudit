"""
Paper Review Agent
"""
import json
from typing import AsyncGenerator
from .base_agent import BaseAgent

class BaseLineAgent(BaseAgent):
    """Agent for testing the /paper_review endpoint"""

    async def run(self, pdf_content: str, query: str) -> AsyncGenerator[str, None]:
        """
        Execute paper review task with streaming response

        Args:
            pdf_content: Base64-encoded PDF content
            query: The review instruction

        Yields:
            Content chunks from the API response
        """
        text = self.extract_pdf_text_from_base64(pdf_content)

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