"""
edgedash/agents/extractor.py

Extraction step for the Scorer pipeline (steering rules 16, 17, 18).

Public API
----------
    extract(listing: dict, db_path: str) -> dict

What it does
------------
  1. Hashes the job description text (SHA-256, 32 hex chars).
  2. Returns the cached result immediately on a cache hit — zero model calls.
  3. On a cache miss: calls llm.complete_json with EXTRACTION_SCHEMA.
  4. Normalises all skill strings to lowercase before storing.
  5. Writes the result to the extraction cache via storage (rule 2).

What it never does
------------------
  - Never mentions the candidate, the user's profile, or scoring weights.
  - Never adds a score field to the result (rule 16).
  - Never calls sqlite3 directly (rule 2).
"""

from __future__ import annotations

import hashlib
import logging
from typing import Any

from edgedash import storage
from edgedash.llm import LLMError, complete_json

logger = logging.getLogger(__name__)


# ---------------------------------------------------------------------------
# Schema (rule 17)
# No score field — must never be added here (rule 16).
# ---------------------------------------------------------------------------

EXTRACTION_SCHEMA: dict[str, Any] = {
    "type": "object",
    "required": ["required_skills", "nice_to_have", "seniority", "years_required", "remote_ok"],
    "additionalProperties": False,
    "properties": {
        "required_skills": {
            "type": "array",
            "items": {"type": "string"},
        },
        "nice_to_have": {
            "type": "array",
            "items": {"type": "string"},
        },
        "seniority": {
            "type": "string",
            "enum": ["junior", "mid", "senior", "lead", "unknown"],
        },
        "years_required": {
            "type": ["integer", "null"],
            "minimum": 0,
        },
        "remote_ok": {
            "type": ["boolean", "null"],
        },
    },
}


# ---------------------------------------------------------------------------
# Prompt (rule 16 — no candidate, no profile, no scoring context)
# ---------------------------------------------------------------------------

_PROMPT_TEMPLATE = """\
You are reading a job listing. Extract only facts that are explicitly stated in the text below.

Return a JSON object with exactly these five fields:
  required_skills  - list of skills the role explicitly requires (strings)
  nice_to_have     - list of skills described as preferred, a plus, or beneficial (strings)
  seniority        - one of: "junior", "mid", "senior", "lead", "unknown"
  years_required   - integer years of experience explicitly stated, or null if not stated
  remote_ok        - true if the listing explicitly says remote is allowed, false if explicitly
                     office-only, null if the listing does not say either way

Rules you must follow:
  - Do not infer, guess, or assume anything not written in the text.
  - Do not evaluate any candidate. You do not know a candidate exists.
  - If the listing does not state something, the value is null or an empty list.
  - Skill names must be lowercase.
  - Reply with the JSON object only. No markdown fences. No prose.

JOB LISTING:
{description}"""


# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------

def _hash_description(text: str) -> str:
    """Return a stable 32-char SHA-256 hex digest of the description text."""
    return hashlib.sha256(text.encode("utf-8")).hexdigest()[:32]


def _normalise_skills(extraction: dict[str, Any]) -> dict[str, Any]:
    """Lowercase all skill strings in-place and return the dict."""
    extraction["required_skills"] = [
        s.lower().strip() for s in extraction.get("required_skills") or []
    ]
    extraction["nice_to_have"] = [
        s.lower().strip() for s in extraction.get("nice_to_have") or []
    ]
    return extraction


# ---------------------------------------------------------------------------
# Public API
# ---------------------------------------------------------------------------

def extract(
    listing: dict[str, Any],
    db_path: str,
    max_retries: int = 1,
) -> dict[str, Any] | None:
    """Extract structured facts from *listing*'s description.

    Returns the extraction dict on success, or None if the LLM call failed
    (callers log the failure and skip this listing — rule 17).

    max_retries is passed through to complete_json.  The default is 1 (one
    retry on validation failure).  The Scorer passes 2 during a widen-
    distribution retry so the LLM gets more attempts to produce richer facts.
    """
    description: str = (listing.get("description") or "").strip()
    if not description:
        logger.warning(
            "Listing %s has no description — skipping extraction.",
            listing.get("id", "unknown"),
        )
        return None

    desc_hash = _hash_description(description)

    # --- Cache hit (rule 18) ---
    # Note: the Scorer clears the cache for middle-band listings before calling
    # extract in widen_distribution mode, so a stale shallow result is never
    # returned on the retry pass.
    cached = storage.get_cached_extraction(db_path, desc_hash)
    if cached is not None:
        logger.debug("Cache hit for listing %s", listing.get("id", "unknown"))
        return cached

    # --- Cache miss: call the model ---
    prompt = _PROMPT_TEMPLATE.format(description=description)

    try:
        result = complete_json(prompt, EXTRACTION_SCHEMA, max_retries=max_retries)
    except LLMError as exc:
        # Rule 17: log the failure for THIS listing only; caller must continue
        logger.error(
            "Extraction failed for listing %s: %s",
            listing.get("id", "unknown"),
            exc,
        )
        return None

    result = _normalise_skills(result)

    # --- Persist to cache (rule 18) ---
    storage.set_cached_extraction(db_path, desc_hash, result)
    logger.debug("Extracted and cached listing %s", listing.get("id", "unknown"))

    return result
