"""
Base Agent class for all API testing agents
"""
from abc import ABC, abstractmethod
from openai import AsyncOpenAI
from openai import APIConnectionError, APITimeoutError, RateLimitError, APIError
from typing import AsyncGenerator
import os
import asyncio
import base64
import PyPDF2
import ssl
import httpx
from io import BytesIO
from typing import Optional, Any
from .logger import get_logger

logger = get_logger(__name__)

class BaseAgent(ABC):
    """Base class for all API testing agents"""
    
    # Retry configuration
    from .config import config
    MAX_RETRIES = config.get("llm.max_retries", 5)
    INITIAL_RETRY_DELAY = config.get("llm.initial_retry_delay", 1.0)  # seconds
    MAX_RETRY_DELAY = config.get("llm.max_retry_delay", 60.0)  # seconds
    BACKOFF_MULTIPLIER = config.get("llm.backoff_multiplier", 2.0)

    def __init__(self):
        """
        Initialize the agent
        """
        model = os.getenv("SCI_LLM_MODEL")
        if model is None:
            raise ValueError("SCI_LLM_MODEL environment variable is not set")
        self.model = model
        
        reasoning_model = os.getenv("SCI_LLM_REASONING_MODEL")
        if reasoning_model is None:
            raise ValueError("SCI_LLM_REASONING_MODEL environment variable is not set")
        self.reasoning_model = reasoning_model
        
        emb_model = os.getenv("SCI_EMBEDDING_MODEL")
        if emb_model is None:
            raise ValueError("SCI_EMBEDDING_MODEL environment variable is not set")
        self.emb_model = emb_model

        # Initialize AsyncOpenAI client for chat model
        base_url = os.getenv("SCI_MODEL_BASE_URL")
        if base_url is None:
            raise ValueError("SCI_MODEL_BASE_URL environment variable is not set")
        api_key = os.getenv("SCI_MODEL_API_KEY")
        if api_key is None:
            raise ValueError("SCI_MODEL_API_KEY environment variable is not set")
        self.client = AsyncOpenAI(
            base_url=base_url,
            api_key=api_key
        )
        
        # Initialize AsyncOpenAI client for embedding model
        emb_base_url = os.getenv("SCI_EMBEDDING_BASE_URL")
        if emb_base_url is None:
            raise ValueError("SCI_EMBEDDING_BASE_URL environment variable is not set")
        emb_api_key = os.getenv("SCI_EMBEDDING_API_KEY")
        if emb_api_key is None:
            raise ValueError("SCI_EMBEDDING_API_KEY environment variable is not set")
        self.emb_client = AsyncOpenAI(
            base_url=emb_base_url,
            api_key=emb_api_key
        )
        
    def extract_pdf_text_from_base64(self, pdf_b64: str) -> str:
        """
        Extract text from base64-encoded PDF using PyPDF2,
        and print page index and extracted text for debugging.
        """
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
                # 打印页号和文字前几百个字符避免控制台爆掉
                # logger.debug(f"\n=== Page {i} ({len(text)} chars) ===")
                # logger.debug(text)
                pages.append(text)

            joined = "\n".join(pages)
            logger.info(f"\nPDF Parsing Completed. Total pages: {len(pages)}, total length: {len(joined)} characters\n")
            return joined

        except Exception as e:
            logger.error(f"PDF parsing error: {str(e)}")
            return ""
        
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
