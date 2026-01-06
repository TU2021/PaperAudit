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


def _guess_mime_from_b64(data_b64: str) -> str:
    head = (data_b64 or "")[:20]
    if head.startswith("iVBOR"):  # PNG
        return "image/png"
    if head.startswith("/9j/"):  # JPEG
        return "image/jpeg"
    if head.startswith("R0lGOD"):  # GIF
        return "image/gif"
    return "image/png"


def _block_to_multimodal_parts(
    b: Dict[str, Any],
    *,
    max_text_chars_per_block: int = 10_000,
    enable_mm: bool = True,
) -> List[Dict[str, Any]]:
    parts: List[Dict[str, Any]] = []
    ci = b.get("content_index", b.get("index"))
    if isinstance(ci, str):
        try:
            ci = int(ci)
        except Exception:
            ci = None

    sec = (b.get("section") or "").strip()
    typ = b.get("type")
    header = f"[Block #{ci if ci is not None else '?'} | {typ or 'unknown'}{(' |Section: ' + sec) if sec else 'None'}]"
    parts.append({"type": "text", "text": header})

    if typ == "text":
        t = (b.get("text") or "")
        if max_text_chars_per_block and len(t) > max_text_chars_per_block:
            t = t[:max_text_chars_per_block]
        if t.strip():
            parts.append({"type": "text", "text": t})
    elif typ == "image_url":
        if not enable_mm:
            parts.append({"type": "text", "text": "[Image omitted: multimodal disabled]"})
            return parts

        img = b.get("image_url")
        url = None
        if isinstance(img, str):
            url = img
        elif isinstance(img, dict):
            if isinstance(img.get("url"), str):
                url = img["url"]
            elif isinstance(img.get("data_b64"), str):
                mime = img.get("mime") or _guess_mime_from_b64(img["data_b64"])
                url = f"data:{mime};base64,{img['data_b64']}"
        if isinstance(url, str) and (url.startswith("data:") or url.startswith("http")):
            parts.append({"type": "image_url", "image_url": {"url": url}})

    return parts


def _blocks_to_multimodal_document(
    blocks: List[Dict[str, Any]],
    *,
    max_blocks: int = 1_000,
    max_images: int = 48,
    max_text_chars_per_block: int = 10_000,
    enable_mm: bool = True,
) -> List[Dict[str, Any]]:
    parts: List[Dict[str, Any]] = []
    img_count = 0

    for b in blocks[:max_blocks]:
        for p in _block_to_multimodal_parts(
            b,
            max_text_chars_per_block=max_text_chars_per_block,
            enable_mm=enable_mm,
        ):
            if p.get("type") == "image_url":
                if not enable_mm:
                    continue
                if img_count >= max_images:
                    continue
                img_count += 1
            parts.append(p)

    return parts

logger = get_logger(__name__)


def _extract_text_from_message_content(raw_content: Any) -> str:
    """
    Normalize OpenAI-style message content into a string.

    The content field may be:
    - a plain string
    - a list of content parts (objects or dicts containing ``text``/``content``)
    - other objects (fallback to ``str``)
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
        base_url: Optional[str] = None,
        api_key: Optional[str] = None,
    ):
        """Initialize the agent with user-specified models and env-based endpoints."""

        if not model:
            raise ValueError("model must be provided by the caller")

        self.model = model

        resolved_base_url = base_url or os.getenv("OPENAI_BASE_URL")
        resolved_api_key = api_key or os.getenv("OPENAI_API_KEY")
        if not resolved_base_url:
            raise ValueError("OPENAI_BASE_URL is not set")
        if not resolved_api_key:
            raise ValueError("OPENAI_API_KEY is not set")

        self.client = AsyncOpenAI(base_url=resolved_base_url, api_key=resolved_api_key)

    def _get_text_from_response(self, resp: Any) -> str:
        """Safely extract string content from a ChatCompletion response."""

        choices = getattr(resp, "choices", None)
        if not choices:
            raise ValueError("LLM response has no choices")

        first = choices[0]
        message = getattr(first, "message", None)
        if message is None:
            raise ValueError("LLM response has no message")

        raw_content = getattr(message, "content", None)
        if raw_content is None:
            raise ValueError("LLM response content is empty")

        text = _extract_text_from_message_content(raw_content).strip()
        if not text:
            raise ValueError("LLM response content is empty")

        return text

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

    def blocks_to_prompt_content(
        self,
        blocks: List[Dict[str, Any]],
        *,
        enable_mm: bool = False,
        max_blocks: int = 1_000,
        max_images: int = 48,
        max_text_chars_per_block: int = 10_000,
    ) -> str | List[Dict[str, Any]]:
        """
        Build prompt-ready content from normalized blocks.

        - When ``enable_mm`` is False, returns plain text (no image URLs).
        - When True, returns a multimodal content list mixing text and ``image_url``
          parts similar to detect/4_1_mas_error_detection.py.
        """

        if not enable_mm:
            return _blocks_to_text(blocks, enable_mm=False)

        return _blocks_to_multimodal_document(
            blocks,
            max_blocks=max_blocks,
            max_images=max_images,
            max_text_chars_per_block=max_text_chars_per_block,
            enable_mm=True,
        )
    async def _call_llm_once(
        self,
        *,
        model: str,
        messages: list[dict[str, Any]],
        temperature: float = 0.2,
        max_tokens: int = 65536,
    ):
        """Single API call (non-streaming)."""
        return await self.client.chat.completions.create(
            model=model,
            messages=messages,  # type: ignore
            max_tokens=max_tokens,
            temperature=temperature,
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
            # frame 1 is _call_llm_with_retry
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

    async def _call_llm_with_retry(
        self,
        *,
        model: str,
        messages: list[dict[str, Any]],
        temperature: Optional[float] = None,
        max_tokens: Optional[int] = None,
        max_retries: Optional[int] = None,
        initial_delay: Optional[float] = None,
    ):
        """
        API call with retry (non-streaming, returns ChatCompletion)
        """
        if temperature is None:
            temperature = self.config.get("llm.default_temperature", 0.2)
        if max_tokens is None:
            max_tokens = self.config.get("llm.default_max_tokens", 65536)
        if max_retries is None:
            max_retries = self.MAX_RETRIES
        if initial_delay is None:
            initial_delay = self.INITIAL_RETRY_DELAY

        last_exception: Optional[Exception] = None
        delay = initial_delay

        for attempt in range(max_retries + 1):
            try:
                response = await self._call_llm_once(
                    model=model,
                    messages=messages,
                    temperature=temperature,
                    max_tokens=max_tokens,
                )

                # Validate that the response contains usable content; raises for retries otherwise
                self._get_text_from_response(response)
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
