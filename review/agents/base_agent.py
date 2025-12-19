"""
Base Agent class for all API testing agents
"""
from abc import ABC, abstractmethod
from io import BytesIO
from typing import Any, AsyncGenerator, Dict, List, Optional

import PyPDF2
import asyncio
import base64
import httpx
import json
import os
import ssl

from openai import APIConnectionError, APIError, APITimeoutError, AsyncOpenAI, RateLimitError

from .logger import get_logger

# ---- JSON helpers reused from detect/utils.py style ---- #
def _normalize_blocks(paper: Dict[str, Any]) -> List[Dict[str, Any]]:
    raw = paper.get("content", []) or []
    blocks: List[Dict[str, Any]] = []
    for i, item in enumerate(raw):
        content_index = item.get("index", i)
        section_label = item.get("section", None)
        block: Dict[str, Any] = {
            "content_index": int(content_index) if isinstance(content_index, int) else i,
            "type": item.get("type"),
            "section": section_label if isinstance(section_label, str) and section_label.strip() else None,
        }
        if item.get("type") == "text":
            block["text"] = item.get("text", "")
        elif item.get("type") == "image_url":
            block["image_url"] = item.get("image_url")
        else:
            for k, v in item.items():
                if k not in block:
                    block[k] = v
        blocks.append(block)
    blocks.sort(key=lambda x: x.get("content_index", 0))
    return blocks


def _blocks_to_text(blocks: List[Dict[str, Any]], enable_mm: bool = False) -> str:
    lines: List[str] = []
    for b in blocks:
        if b.get("type") == "text":
            lines.append(b.get("text", ""))
        elif enable_mm and b.get("type") == "image_url":
            url = b.get("image_url")
            lines.append(f"[IMAGE]{f' {url}' if url else ''}")
    return "\n".join(lines)

logger = get_logger(__name__)

class BaseAgent(ABC):
    """Base class for all API testing agents"""
    
    # Retry configuration
    from .config import config
    MAX_RETRIES = config.get("llm.max_retries", 5)
    INITIAL_RETRY_DELAY = config.get("llm.initial_retry_delay", 1.0)  # seconds
    MAX_RETRY_DELAY = config.get("llm.max_retry_delay", 60.0)  # seconds
    BACKOFF_MULTIPLIER = config.get("llm.backoff_multiplier", 2.0)

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
        """Initialize the agent with user-specified models and env-based endpoints."""

        if not model:
            raise ValueError("model must be provided by the caller")

        self.model = model
        self.reasoning_model = reasoning_model or model
        self.emb_model = embedding_model or model

        resolved_base_url = base_url or os.getenv("OPENAI_BASE_URL")
        resolved_api_key = api_key or os.getenv("OPENAI_API_KEY")
        if not resolved_base_url:
            raise ValueError("OPENAI_BASE_URL is not set")
        if not resolved_api_key:
            raise ValueError("OPENAI_API_KEY is not set")

        self.client = AsyncOpenAI(base_url=resolved_base_url, api_key=resolved_api_key)

        resolved_emb_base_url = embedding_base_url or resolved_base_url
        resolved_emb_api_key = embedding_api_key or resolved_api_key
        self.emb_client = AsyncOpenAI(base_url=resolved_emb_base_url, api_key=resolved_emb_api_key)

    def extract_pdf_text_from_base64(self, pdf_b64: str) -> str:
        """Extract text from base64-encoded PDF (legacy support)."""
        try:
            pdf_bytes = base64.b64decode(pdf_b64)
            reader = PyPDF2.PdfReader(BytesIO(pdf_bytes))

            pages = []

            for i, page in enumerate(reader.pages, start=1):
                try:
                    text = page.extract_text() or ""
                except Exception as e:
                    logger.warning(f"Page {i} extract_text() failed: {e}")
                    text = ""
                pages.append(text)

            joined = "\n".join(pages)
            logger.info(
                f"\nPDF Parsing Completed. Total pages: {len(pages)}, total length: {len(joined)} characters\n"
            )
            return joined

        except Exception as e:
            logger.error(f"PDF parsing error: {str(e)}")
            return ""

    def prepare_paper_blocks(self, paper_json: Any) -> List[Dict[str, Any]]:
        """Normalize user-provided paper JSON (dict/string/path) into ordered blocks."""
        data: Dict[str, Any] = {}
        if isinstance(paper_json, str):
            # try parsing JSON string first, then fallback to filesystem path
            try:
                data = json.loads(paper_json)
            except Exception:
                from pathlib import Path

                path = Path(paper_json).expanduser()
                if not path.exists():
                    raise ValueError(f"Paper JSON not found: {path}")
                with path.open("r", encoding="utf-8") as f:
                    data = json.load(f)
        elif isinstance(paper_json, dict):
            data = paper_json
        else:
            raise TypeError("paper_json must be dict or JSON/string path")

        paper = data.get("paper", data) if isinstance(data, dict) else {}
        return _normalize_blocks(paper)

    def blocks_to_text(self, blocks: List[Dict[str, Any]], enable_mm: bool = False) -> str:
        """Convert normalized blocks to plain text, optionally including multimodal markers."""
        return _blocks_to_text(blocks, enable_mm=enable_mm)
        
    async def get_embedding(self, text: str) -> list[float]:
        """
        Get embedding vector for text using embedding model
        """
        try:
            response = await self.emb_client.embeddings.create(
                model=self.emb_model,
                input=text
            )
            embedding = response.data[0].embedding
            # Log embedding results (truncated)
            logger.info(f"Text: {text[:100]}{'...' if len(text) > 100 else ''}")
            logger.info(f"Embedding dimension: {len(embedding)}")
            logger.info(f"Embedding (first 5 values): {embedding[:5]}")
            return embedding
        except Exception as e:
            logger.error(f"Embedding error: {str(e)}")
            return []
        
    async def _call_llm_once(
        self,
        *,
        model: str,
        messages: list[dict[str, Any]],
        stream: bool = True,
        temperature: float = 0.2,
        max_tokens: int = 65536,
    ):
        """
        单次 API 调用（可流式/非流式）

        stream=True  -> 返回 AsyncStream[ChatCompletionChunk]
        stream=False -> 返回 ChatCompletion
        """
        return await self.client.chat.completions.create(
            model=model,
            messages=messages,  # type: ignore
            max_tokens=max_tokens,
            temperature=temperature,
            stream=stream,
        )

    def _should_retry_error(self, e: Exception) -> bool:
        # if isinstance(e, (APIConnectionError, APITimeoutError, RateLimitError, ValueError, ssl.SSLError, httpx.RemoteProtocolError, httpx.ReadTimeout)):
        #     return True
        # elif isinstance(e, APIError):
        #     status_code = getattr(e, "status_code", None)
        #     if status_code and status_code >= 500:
        #         return True
        return True

    def _get_caller_info(self) -> str:
        """Safely get caller info for logging"""
        try:
            import inspect
            # frame 0 is this function
            # frame 1 is _call_llm_with_retry (or _stream_with_retry)
            # frame 2 is the actual caller
            stack = inspect.stack()
            if len(stack) > 2:
                caller_frame = stack[2]
                info = f"{caller_frame.function} (line {caller_frame.lineno})"
                if 'self' in caller_frame.frame.f_locals:
                    caller_class = caller_frame.frame.f_locals['self'].__class__.__name__
                    info = f"{caller_class}.{info}"
                return info
        except Exception:
            pass
        return "Unknown caller"

    async def _stream_with_retry(
        self,
        *,
        model: str,
        messages: list[dict[str, Any]],
        temperature: float,
        max_tokens: int,
        max_retries: int,
        initial_delay: float,
        caller_info: Optional[str] = None,  # Deprecated, calculated internally on error
    ) -> AsyncGenerator[Any, None]:
        """
        Internal helper to handle streaming retries.
        Retries as long as no content has been yielded.
        """
        delay = initial_delay
        last_exception = None

        for attempt in range(max_retries + 1):
            has_yielded_content = False
            try:
                response = await self._call_llm_once(
                    model=model,
                    messages=messages,
                    stream=True,
                    temperature=temperature,
                    max_tokens=max_tokens,
                )
                
                async for chunk in response:
                    content = None
                    if chunk.choices and chunk.choices[0].delta:
                        content = chunk.choices[0].delta.content
                    
                    if content:
                        has_yielded_content = True
                    
                    yield chunk
                
                return

            except Exception as e:
                # Calculate caller info only when error occurs
                current_caller = self._get_caller_info()
                
                if has_yielded_content:
                    logger.error(f"Caller: {current_caller} | Stream interrupted after content yielded: {type(e).__name__}: {e}")
                    raise e
                
                if not self._should_retry_error(e):
                    logger.error(f"Caller: {current_caller} | Non-retryable error: {type(e).__name__}: {e}")
                    raise e

                last_exception = e
                if attempt < max_retries:
                    logger.warning(
                        f"Retry {attempt + 1}/{max_retries} | "
                        f"Caller: {current_caller} | Error: {type(e).__name__}: {e}"
                    )
                    logger.info(
                        f"Retry {attempt + 1}/{max_retries} | "
                        f"Caller: {current_caller} | Retrying in {delay:.2f} seconds..."
                    )
                    await asyncio.sleep(delay)
                    delay = min(delay * self.BACKOFF_MULTIPLIER, self.MAX_RETRY_DELAY)
                else:
                    logger.error(f"Caller: {current_caller} | All {max_retries} retry attempts exhausted")
        
        if last_exception:
            raise last_exception

    async def _call_llm_with_retry(
        self,
        *,
        model: str,
        messages: list[dict[str, Any]],
        stream: bool = True,
        temperature: Optional[float] = None,
        max_tokens: Optional[int] = None,
        max_retries: Optional[int] = None,
        initial_delay: Optional[float] = None,
    ):
        """
        带重试的 API 调用

        - stream=True  时，返回 AsyncStream[ChatCompletionChunk]
        - stream=False 时，返回 ChatCompletion
        """
        if temperature is None:
            temperature = self.config.get("llm.default_temperature", 0.2)
        if max_tokens is None:
            max_tokens = self.config.get("llm.default_max_tokens", 65536)
        if max_retries is None:
            max_retries = self.MAX_RETRIES
        if initial_delay is None:
            initial_delay = self.INITIAL_RETRY_DELAY

        if stream:
            return self._stream_with_retry(
                model=model,
                messages=messages,
                temperature=temperature,
                max_tokens=max_tokens,
                max_retries=max_retries,
                initial_delay=initial_delay,
            )

        # Non-streaming logic
        
        # 1. For reasoning models, use pseudo-streaming to avoid timeouts
        if model == self.reasoning_model:
            full_content = []
            try:
                generator = self._stream_with_retry(
                    model=model,
                    messages=messages,
                    temperature=temperature,
                    max_tokens=max_tokens,
                    max_retries=max_retries,
                    initial_delay=initial_delay,
                )
                
                async for chunk in generator:
                    if chunk.choices and chunk.choices[0].delta.content:
                        full_content.append(chunk.choices[0].delta.content)
                
                content_str = "".join(full_content)
                
                # Construct a mock response object that mimics ChatCompletion
                from types import SimpleNamespace
                message = SimpleNamespace(content=content_str)
                choice = SimpleNamespace(message=message)
                response = SimpleNamespace(choices=[choice])
                return response

            except Exception as e:
                 # _stream_with_retry already handles retries and logging.
                 raise e

        # 2. For standard models, use normal non-streaming call with retry
        last_exception: Optional[Exception] = None
        delay = initial_delay

        for attempt in range(max_retries + 1):
            try:
                response = await self._call_llm_once(
                    model=model,
                    messages=messages,
                    stream=False,
                    temperature=temperature,
                    max_tokens=max_tokens,
                )
                
                if not response.choices:
                    raise ValueError("LLM response has no choices")
                if not response.choices[0].message.content:
                    raise ValueError("LLM response content is empty")
                return response

            except Exception as e:
                current_caller = self._get_caller_info()

                if not self._should_retry_error(e):
                    logger.error(f"Caller: {current_caller} | [Non-Streaming] Non-retryable error: {type(e).__name__}: {e}")
                    raise e

                last_exception = e
                if attempt < max_retries:
                    logger.warning(
                        f"Retry {attempt + 1}/{max_retries} | "
                        f"Caller: {current_caller} | [Non-Streaming] Error: {type(e).__name__}: {e}"
                    )
                    logger.info(
                        f"Retry {attempt + 1}/{max_retries} | "
                        f"Caller: {current_caller} | [Non-Streaming] Retrying in {delay:.2f} seconds..."
                    )
                    await asyncio.sleep(delay)
                    delay = min(delay * self.BACKOFF_MULTIPLIER, self.MAX_RETRY_DELAY)
                else:
                    logger.error(f"Caller: {current_caller} | [Non-Streaming] All {max_retries} retry attempts exhausted")

        if last_exception:
            raise last_exception
        raise Exception("API call failed after all retries")


    @abstractmethod
    async def run(self, **kwargs) -> AsyncGenerator[str, None]:
        """
        Execute the agent's main task and return streaming response
        
        This method must be implemented by all subclasses
        
        Yields:
            Content chunks from the API response
        """
        pass
