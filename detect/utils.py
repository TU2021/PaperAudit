"""Common utilities for detection scripts."""
from __future__ import annotations

import json
import re
from pathlib import Path
from threading import Lock
from typing import Any, Optional, Union

from openai import OpenAI

JsonPath = Union[str, Path]


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


def get_openai_client(api_key: Optional[str]) -> OpenAI:
    """Thread-safe lazy initialization of a shared OpenAI client."""
    global _client_singleton
    with _client_lock:
        if _client_singleton is None:
            if not api_key:
                raise RuntimeError("OPENAI_API_KEY is not set.")
            _client_singleton = OpenAI(api_key=api_key)
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
