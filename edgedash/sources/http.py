"""
edgedash/sources/http.py

The ONLY place in the project that performs HTTP requests (steering rule 11).

Provides get_json() with:
  - 10 s timeout
  - 2 retries with exponential back-off (1 s, 2 s)
  - A real User-Agent header
  - Raises SourceError on any unrecoverable failure
"""

from __future__ import annotations

import logging
import time
from typing import Any

import requests

logger = logging.getLogger(__name__)

_USER_AGENT = (
    "EdgeDash/0.1 (AI career intelligence agent; "
    "https://github.com/your-org/edgedash; contact@example.com)"
)
_TIMEOUT_SECONDS = 10
_MAX_RETRIES = 2
_BACKOFF_BASE = 1.0  # seconds; attempt n waits _BACKOFF_BASE * 2^(n-1)


class SourceError(Exception):
    """Raised when a source HTTP call fails after all retries."""


def get_json(
    url: str,
    params: dict[str, Any] | None = None,
    headers: dict[str, str] | None = None,
) -> Any:
    """GET *url*, parse JSON, and return the parsed object.

    Retries up to _MAX_RETRIES times with exponential back-off.
    Raises SourceError if all attempts fail.
    """
    merged_headers = {"User-Agent": _USER_AGENT}
    if headers:
        merged_headers.update(headers)

    last_exc: Exception | None = None

    for attempt in range(1, _MAX_RETRIES + 2):  # attempts: 1, 2, 3 (initial + retries)
        try:
            response = requests.get(
                url,
                params=params,
                headers=merged_headers,
                timeout=_TIMEOUT_SECONDS,
            )
            response.raise_for_status()
            return response.json()

        except requests.RequestException as exc:
            last_exc = exc
            if attempt <= _MAX_RETRIES:
                wait = _BACKOFF_BASE * (2 ** (attempt - 1))
                logger.warning(
                    "HTTP attempt %d/%d failed for %s: %s — retrying in %.1fs",
                    attempt,
                    _MAX_RETRIES + 1,
                    url,
                    exc,
                    wait,
                )
                time.sleep(wait)
            else:
                logger.error(
                    "All %d HTTP attempts failed for %s: %s",
                    _MAX_RETRIES + 1,
                    url,
                    exc,
                )

    raise SourceError(f"Failed to fetch {url} after {_MAX_RETRIES + 1} attempts: {last_exc}") from last_exc
