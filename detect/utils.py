"""Common utilities for detection scripts."""
from __future__ import annotations

import json
import os
import re
import sys
import time
from pathlib import Path
from threading import Lock
from typing import Any, Callable, Dict, List, Optional, Union

from openai import OpenAI
from openai import (
    APIConnectionError,
    APIError,
    APITimeoutError,
    AuthenticationError,
    BadRequestError,
    PermissionDeniedError,
    RateLimitError,
    UnprocessableEntityError,
)

JsonPath = Union[str, Path]

DEFAULT_LLM_MAX_RETRIES = 3


# ================ JSON helpers ================

def load_json(path: JsonPath) -> Any:
    """Read a JSON file with UTF-8 encoding."""
    with Path(path).open("r", encoding="utf-8") as f:
        return json.load(f)


def save_json(obj: Any, path: JsonPath) -> None:
    """Write JSON with UTF-8 encoding and ensure parent directories exist."""
    p = Path(path)
    p.parent.mkdir(parents=True, exist_ok=True)
    with p.open("w", encoding="utf-8") as f:
        json.dump(obj, f, indent=2, ensure_ascii=False)


# ================ LLM helpers ================

_client_singleton: Optional[OpenAI] = None
_client_lock = Lock()


def get_openai_client(api_key: Optional[str] = None) -> OpenAI:
    """Thread-safe lazy initialization of a shared OpenAI client.

    Falls back to the ``OPENAI_API_KEY`` environment variable if ``api_key``
    is not provided.
    """
    global _client_singleton
    key = api_key or os.getenv("OPENAI_API_KEY")
    with _client_lock:
        if _client_singleton is None:
            if not key:
                raise RuntimeError("OPENAI_API_KEY is not set.")
            _client_singleton = OpenAI(api_key=key)
        return _client_singleton


# ================ Text parsing helpers ================

def extract_json_from_text(text: str) -> Any:
    """Best-effort JSON extraction from possibly noisy LLM output."""
    text = (text or "").strip()
    try:
        return json.loads(text)
    except Exception:
        pass

    start, end = text.find("{"), text.rfind("}")
    if start != -1 and end != -1 and end > start:
        candidate = text[start : end + 1]
        for _ in range(5):
            try:
                return json.loads(candidate)
            except Exception:
                end = text.rfind("}", 0, end - 1)
                if end <= start:
                    break
                candidate = text[start : end + 1]

    m = re.search(r"\{(?:.|\n)*?\}", text)
    if m:
        return json.loads(m.group(0))

    raise ValueError("Failed to extract JSON from LLM output.")


# ================ Chat helpers ================

def call_llm_chat(
    messages: List[Dict[str, Any]],
    model: str,
    max_tokens: int,
    temperature: float,
    api_key: Optional[str] = None,
    retries: int = DEFAULT_LLM_MAX_RETRIES,
) -> Optional[str]:
    """Call ``chat.completions.create`` with basic retry handling."""

    client = get_openai_client(api_key)
    last_exc: Optional[BaseException] = None

    for attempt in range(1, retries + 1):
        try:
            resp = client.chat.completions.create(
                model=model,
                messages=messages,
                temperature=temperature,
                max_tokens=max_tokens,
                n=1,
            )
            return resp.choices[0].message.content
        except RateLimitError as e:
            last_exc = e
            wait = 2 ** attempt
            print(f"[RATE LIMIT] attempt {attempt}, waiting {wait}s...", file=sys.stderr)
            time.sleep(wait)
        except (APITimeoutError, APIConnectionError) as e:
            last_exc = e
            print(f"[TEMP ERROR] attempt {attempt}: {e}", file=sys.stderr)
            time.sleep(1.5 * attempt)
        except (
            BadRequestError,
            AuthenticationError,
            PermissionDeniedError,
            UnprocessableEntityError,
            APIError,
        ) as e:
            last_exc = e
            print(f"[FATAL] {e}", file=sys.stderr)
            break
        except Exception as e:  # pragma: no cover - defensive
            last_exc = e
            print(f"[ERROR] attempt {attempt}: {e}", file=sys.stderr)
            time.sleep(1.0 * attempt)

    print(f"[LLM FAILURE] {last_exc}", file=sys.stderr)
    return None


def call_llm_chat_with_empty_retries(
    msg_builder: Union[Callable[[float], Any], List[Dict[str, Any]], Dict[str, Any]],
    model: str,
    max_tokens: int,
    tag: str,
    dbgdir: Path,
    temperature: float = 0.0,
    expect_key: Optional[str] = None,
    retries: int = 2,
    api_key: Optional[str] = None,
) -> Optional[str]:
    """
    Retry chat calls that return empty output or missing keys.

    * ``msg_builder`` can be a callable(temperature)->messages, or a messages
      list/dict directly.
    * ``expect_key`` (optional) enforces JSON parsing and non-empty key.
    """

    last_raw: Optional[str] = None

    def _coerce_messages():
        if callable(msg_builder):
            return msg_builder(temperature)
        if isinstance(msg_builder, (list, dict)):
            return msg_builder
        raise TypeError(
            f"msg_builder must be a callable or list/dict, got: {type(msg_builder)}"
        )

    for k in range(retries + 1):
        try:
            messages = _coerce_messages()
        except Exception as e:  # pragma: no cover - defensive
            print(f"[FATAL][{tag}] building messages failed: {e}", file=sys.stderr)
            break

        raw = call_llm_chat(
            messages,
            model=model,
            max_tokens=max_tokens,
            temperature=temperature,
            api_key=api_key,
            retries=retries,
        )
        last_raw = raw

        if raw is None or not raw.strip():
            print(f"[WARN][{tag}] empty raw (try {k+1}/{retries+1})", file=sys.stderr)
            continue

        if expect_key:
            try:
                obj = extract_json_from_text(raw)
                if obj.get(expect_key) is not None:
                    return raw
                print(
                    f"[WARN][{tag}] parsed but '{expect_key}' missing/empty (try {k+1}/{retries+1})",
                    file=sys.stderr,
                )
                continue
            except Exception as e:  # pragma: no cover - defensive
                print(
                    f"[WARN][{tag}] parse failed: {e} (try {k+1}/{retries+1})",
                    file=sys.stderr,
                )
                continue
        else:
            return raw

    if last_raw is not None:
        try:
            save_json({"raw": last_raw}, dbgdir / f"{tag}.last_raw.json")
        except Exception:
            pass
    return last_raw


def call_web_search_via_tool(
    queries: List[str],
    model: str,
    temperature: float,
    api_key: Optional[str] = None,
    retries: int = DEFAULT_LLM_MAX_RETRIES,
) -> List[Dict[str, str]]:
    """
    Use the Responses API + ``web_search`` tool for a batch of short queries.

    Returns a list of ``{"query": ..., "answer": ...}`` items.
    """

    if not queries:
        return []

    client = get_openai_client(api_key)
    sys_prompt = (
        "You are a web-search assistant. Given a list of short self-contained factual questions, "
        "use the web_search tool to find relevant information on the internet, "
        "and then return concise, direct answers for each question.\n"
        "Return ONLY valid JSON in the form:\n"
        "{'answers':[{'query':'...','answer':'...'}]}\n"
        "Each answer should be brief, factual, and self-contained."
    )
    user_payload = {"queries": queries}
    messages = [
        {"role": "system", "content": sys_prompt},
        {"role": "user", "content": "INPUT(JSON):\n" + json.dumps(user_payload, ensure_ascii=False)},
    ]

    for attempt in range(1, retries + 1):
        try:
            resp = client.responses.create(
                model=model,
                input=messages,
                tools=[{"type": "web_search"}],
                max_output_tokens=2_000,
                temperature=temperature,
            )
            text = resp.output_parsed
            if not text:
                text = resp.output[0].content[0].text

            obj = extract_json_from_text(text)
            answers = obj.get("answers", []) if isinstance(obj, dict) else []
            if isinstance(answers, list):
                return [a for a in answers if isinstance(a, dict)]
        except RateLimitError as e:
            wait = 2 ** attempt
            print(f"[RATE LIMIT][web] attempt {attempt}, waiting {wait}s...", file=sys.stderr)
            time.sleep(wait)
        except (APITimeoutError, APIConnectionError) as e:
            print(f"[TEMP ERROR][web] attempt {attempt}: {e}", file=sys.stderr)
            time.sleep(1.5 * attempt)
        except (
            BadRequestError,
            AuthenticationError,
            PermissionDeniedError,
            UnprocessableEntityError,
            APIError,
        ) as e:
            print(f"[FATAL][web] {e}", file=sys.stderr)
            break
        except Exception as e:  # pragma: no cover - defensive
            print(f"[ERROR][web] attempt {attempt}: {e}", file=sys.stderr)
            time.sleep(1.0 * attempt)

    return []
