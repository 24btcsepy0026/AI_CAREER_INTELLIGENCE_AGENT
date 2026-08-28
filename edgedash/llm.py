"""
edgedash/llm.py

THE ONLY MODULE ALLOWED TO IMPORT AN LLM SDK (steering rule 15).

Public API
----------
    complete_json(prompt, schema, *, max_retries=1) -> dict

Providers
---------
    "gemini"  — google-generativeai, key from env var GEMINI_API_KEY
    "ollama"  — local HTTP server, no key required

Adding a third provider: add a class that implements _Provider, register it
in _PROVIDER_REGISTRY. complete_json never needs to change.

Rate limiting (rule 15)
-----------------------
    • Minimum 1 second between any two calls.
    • Rolling cap of 15 calls per 60 seconds.
    • 429 / quota errors → exponential back-off, up to 3 attempts, then raise.
"""

from __future__ import annotations

import argparse
import json
import logging
import os
import re
import time
from abc import ABC, abstractmethod
from collections import deque
from typing import Any

import jsonschema

from edgedash.config import Config, load_config

logger = logging.getLogger(__name__)

# ---------------------------------------------------------------------------
# Custom exception
# ---------------------------------------------------------------------------

class LLMError(Exception):
    """Raised when the LLM call fails unrecoverably after all retries."""


# ---------------------------------------------------------------------------
# Rate limiter
# ---------------------------------------------------------------------------

class _RateLimiter:
    """Enforces min 1 s between calls and max 15 calls per 60 s."""

    MIN_INTERVAL = 1.0        # seconds between any two calls
    WINDOW = 60.0             # rolling window in seconds
    MAX_IN_WINDOW = 15        # max calls within the window

    def __init__(self) -> None:
        self._last_call: float = 0.0
        self._timestamps: deque[float] = deque()

    def wait(self) -> None:
        now = time.monotonic()

        # Enforce minimum interval
        gap = now - self._last_call
        if gap < self.MIN_INTERVAL:
            time.sleep(self.MIN_INTERVAL - gap)
            now = time.monotonic()

        # Enforce rolling window cap
        cutoff = now - self.WINDOW
        while self._timestamps and self._timestamps[0] < cutoff:
            self._timestamps.popleft()

        if len(self._timestamps) >= self.MAX_IN_WINDOW:
            oldest = self._timestamps[0]
            sleep_for = (oldest + self.WINDOW) - now
            if sleep_for > 0:
                logger.debug("Rate limit: sleeping %.1f s", sleep_for)
                time.sleep(sleep_for)
            now = time.monotonic()

        self._last_call = now
        self._timestamps.append(now)


_rate_limiter = _RateLimiter()


# ---------------------------------------------------------------------------
# JSON extraction helpers
# ---------------------------------------------------------------------------

_FENCE_RE = re.compile(r"```(?:json)?\s*(.*?)\s*```", re.DOTALL)


def _extract_json(text: str) -> dict[str, Any]:
    """Strip markdown fences and any surrounding prose, then parse JSON."""
    # Try fenced block first
    match = _FENCE_RE.search(text)
    candidate = match.group(1) if match else text

    # Find the first { ... } span as a fallback
    start = candidate.find("{")
    end = candidate.rfind("}")
    if start != -1 and end != -1 and end > start:
        candidate = candidate[start : end + 1]

    return json.loads(candidate)


def _validate(data: dict[str, Any], schema: dict) -> None:
    """Raise jsonschema.ValidationError if data doesn't match schema."""
    jsonschema.validate(instance=data, schema=schema)


# ---------------------------------------------------------------------------
# Provider abstraction
# ---------------------------------------------------------------------------

class _Provider(ABC):
    """Each provider implements exactly one method."""

    @abstractmethod
    def call(self, prompt: str) -> str:
        """Send prompt, return raw text response."""


class _GeminiProvider(_Provider):
    def __init__(self, model: str, api_key: str) -> None:
        import google.generativeai as genai  # local import — only this class touches the SDK
        genai.configure(api_key=api_key)
        self._model = genai.GenerativeModel(model)
        self._model_name = model

    def call(self, prompt: str) -> str:
        response = self._model.generate_content(prompt)
        return response.text


class _OllamaProvider(_Provider):
    """Calls a locally running Ollama server (default http://localhost:11434)."""

    def __init__(self, model: str, base_url: str = "http://localhost:11434") -> None:
        import requests as _req  # stdlib-adjacent; already a project dependency
        self._requests = _req
        self._model = model
        self._url = f"{base_url}/api/generate"

    def call(self, prompt: str) -> str:
        resp = self._requests.post(
            self._url,
            json={"model": self._model, "prompt": prompt, "stream": False},
            timeout=60,
        )
        resp.raise_for_status()
        return resp.json()["response"]


# Registry: name -> (factory that receives model_name and config)
def _make_gemini(model: str, cfg: Config) -> _GeminiProvider:
    api_key = os.environ.get("GEMINI_API_KEY", "").strip()
    if not api_key:
        raise LLMError(
            "GEMINI_API_KEY is not set.\n"
            "Add it to your .env file (see .env.example) and make sure it is loaded.\n"
            "The .env file must NOT be committed to git."
        )
    return _GeminiProvider(model, api_key)


def _make_ollama(model: str, cfg: Config) -> _OllamaProvider:
    base_url = os.environ.get("OLLAMA_BASE_URL", "http://localhost:11434")
    return _OllamaProvider(model, base_url)


_PROVIDER_REGISTRY: dict[str, Any] = {
    "gemini": _make_gemini,
    "ollama": _make_ollama,
}

# Module-level provider cache — instantiated once per process
_provider_cache: dict[str, _Provider] = {}


def _get_provider(cfg: Config) -> _Provider:
    provider_name = cfg.llm_provider.lower()
    cache_key = f"{provider_name}::{cfg.llm_model}"

    if cache_key not in _provider_cache:
        if provider_name not in _PROVIDER_REGISTRY:
            raise LLMError(
                f"Unknown llm_provider '{provider_name}'. "
                f"Valid choices: {list(_PROVIDER_REGISTRY)}"
            )
        factory = _PROVIDER_REGISTRY[provider_name]
        _provider_cache[cache_key] = factory(cfg.llm_model, cfg)

    return _provider_cache[cache_key]


# ---------------------------------------------------------------------------
# Back-off helper for 429 / quota errors
# ---------------------------------------------------------------------------

_QUOTA_SIGNALS = ("429", "quota", "resource_exhausted", "rate", "ratelimit")
_FATAL_SIGNALS = ("404", "not found", "invalid", "not supported")


def _is_quota_error(exc: Exception) -> bool:
    msg = str(exc).lower()
    # Never retry 404 / model-not-found — that's a config error, not transient
    if any(sig in msg for sig in _FATAL_SIGNALS):
        return False
    return any(sig in msg for sig in _QUOTA_SIGNALS)


def _parse_retry_delay(exc: Exception) -> float | None:
    """Extract the suggested retry delay (seconds) from a Gemini 429 error, if present."""
    import re
    msg = str(exc)
    # Gemini errors include: retry_delay { seconds: 21 }
    match = re.search(r"retry_delay\s*\{[^}]*seconds:\s*(\d+)", msg)
    if match:
        return float(match.group(1)) + 2.0   # add 2s buffer
    return None


def _call_with_backoff(provider: _Provider, prompt: str) -> str:
    """Call provider.call() with up to 3 attempts on quota/429 errors.

    Respects the retry_delay the API sends back; falls back to exponential
    backoff (2 s, 4 s) when no delay is specified.
    """
    for attempt in range(1, 4):
        _rate_limiter.wait()
        try:
            return provider.call(prompt)
        except Exception as exc:
            if _is_quota_error(exc) and attempt < 3:
                suggested = _parse_retry_delay(exc)
                wait = suggested if suggested is not None else 2 ** attempt
                logger.warning(
                    "Quota/429 error (attempt %d/3): %s — retrying in %.0fs",
                    attempt, exc, wait,
                )
                time.sleep(wait)
            else:
                raise
    raise LLMError("Unreachable")  # pragma: no cover


# ---------------------------------------------------------------------------
# Public API
# ---------------------------------------------------------------------------

def complete_json(
    prompt: str,
    schema: dict,
    *,
    max_retries: int = 1,
    _cfg: Config | None = None,
) -> dict[str, Any]:
    """Send *prompt* to the configured LLM, validate against *schema*, return dict.

    Retries once on parse/validation failure with an appended correction instruction.
    Raises LLMError if still failing after retries, or on quota exhaustion.
    """
    cfg = _cfg or load_config()
    provider = _get_provider(cfg)

    last_error: str = ""
    current_prompt = prompt

    for attempt in range(max_retries + 1):
        if attempt > 0:
            # Append a correction instruction with the exact previous error
            current_prompt = (
                f"{prompt}\n\n"
                f"CORRECTION REQUIRED: Your previous response failed validation "
                f"with this error: {last_error}\n"
                "Reply with valid JSON only. No markdown fences. No prose. "
                "Start your reply with {{ and end with }}."
            )

        try:
            raw = _call_with_backoff(provider, current_prompt)
        except LLMError:
            raise
        except Exception as exc:
            raise LLMError(f"LLM call failed: {exc}") from exc

        # Parse
        try:
            data = _extract_json(raw)
        except (json.JSONDecodeError, ValueError) as exc:
            last_error = f"JSON parse error: {exc} | raw snippet: {raw[:200]!r}"
            logger.warning("Attempt %d: %s", attempt + 1, last_error)
            if attempt < max_retries:
                continue
            raise LLMError(
                f"LLM returned invalid JSON after {max_retries + 1} attempt(s).\n"
                f"Last error: {last_error}"
            )

        # Validate
        try:
            _validate(data, schema)
            return data
        except jsonschema.ValidationError as exc:
            last_error = f"Schema validation error: {exc.message}"
            logger.warning("Attempt %d: %s", attempt + 1, last_error)
            if attempt < max_retries:
                continue
            raise LLMError(
                f"LLM response failed schema validation after {max_retries + 1} attempt(s).\n"
                f"Last error: {last_error}"
            )

    raise LLMError("Unreachable")  # pragma: no cover


# ---------------------------------------------------------------------------
# CLI check: python -m edgedash.llm --check
# ---------------------------------------------------------------------------

_CHECK_SCHEMA = {
    "type": "object",
    "properties": {"ok": {"type": "boolean"}, "msg": {"type": "string"}},
    "required": ["ok", "msg"],
}

_CHECK_PROMPT = (
    'Reply with this exact JSON and nothing else: {"ok": true, "msg": "hello from edgedash"}'
)


def _cli_check() -> None:
    cfg = load_config()
    print(f"Provider : {cfg.llm_provider}")
    print(f"Model    : {cfg.llm_model}")
    print("Sending test prompt…")
    try:
        result = complete_json(_CHECK_PROMPT, _CHECK_SCHEMA, _cfg=cfg)
        print(f"Response : {result}")
        print("✓  LLM connection OK")
    except LLMError as exc:
        print(f"✗  LLM check FAILED: {exc}")
        raise SystemExit(1)


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="EdgeDash LLM utility")
    parser.add_argument("--check", action="store_true", help="Send a test prompt and verify the connection")
    args = parser.parse_args()
    if args.check:
        _cli_check()
    else:
        parser.print_help()
