"""
edgedash/sources/arbeitnow.py

Arbeitnow job board source — free public API, no key required.
API docs: https://www.arbeitnow.com/api/job-board-api

Behaviour:
  - Fetches page 1 first; keeps paging while results match config keywords,
    up to a hard cap of MAX_PAGES pages.
  - Filters by config.keywords (any keyword present in title/description).
  - Filters by config.target_city in location.
  - If city filtering would leave fewer than MIN_RESULTS rows, the location
    filter is relaxed and a warning is logged (steering rule 12).
  - Rate-limited to 1 request/s (steering rule 14).
  - External ID is the stable slug from the API, never a hash or row number.
"""

from __future__ import annotations

import logging
import time
from datetime import datetime, timezone
from typing import Any

from edgedash.config import Config
from edgedash.sources.base import Source, normalise, register
from edgedash.sources.http import SourceError, get_json

logger = logging.getLogger(__name__)

_API_URL = "https://www.arbeitnow.com/api/job-board-api"
_MAX_PAGES = 5
_MIN_RESULTS = 5
_RATE_LIMIT_SECONDS = 1.0  # max 1 req/s (steering rule 14)


def _parse_posted_at(raw_val: Any) -> str | None:
    """Convert epoch int or ISO string to ISO-8601 string, or None."""
    if raw_val is None:
        return None
    try:
        if isinstance(raw_val, (int, float)):
            return datetime.fromtimestamp(raw_val, tz=timezone.utc).isoformat()
        return str(raw_val)
    except (OSError, ValueError, TypeError):
        return None


def _matches_keywords(job: dict[str, Any], keywords: list[str]) -> bool:
    """True if any keyword appears (case-insensitive) in title or description."""
    searchable = (
        (job.get("title") or "") + " " + (job.get("description") or "")
    ).lower()
    return any(kw.lower() in searchable for kw in keywords)


def _matches_city(job: dict[str, Any], city: str) -> bool:
    """True if city appears (case-insensitive) in the location field."""
    location = (job.get("location") or "").lower()
    return city.lower() in location


def _to_normalised(job: dict[str, Any]) -> dict[str, Any]:
    """Map an Arbeitnow API job object onto our canonical keys."""
    return normalise(
        {
            "external_id": job.get("slug") or None,
            "title": job.get("title") or None,
            "company": job.get("company_name") or None,
            "location": job.get("location") or None,
            "url": job.get("url") or None,
            "description": job.get("description") or None,
            "posted_at": _parse_posted_at(job.get("created_at")),
            "raw": job,
        },
        source_name="arbeitnow",
    )


@register
class ArbeitnowSource(Source):
    name = "arbeitnow"

    def fetch(self, config: Config) -> list[dict[str, Any]]:
        """Fetch and filter Arbeitnow listings; return normalised rows."""
        all_raw: list[dict[str, Any]] = []
        page = 1

        while page <= _MAX_PAGES:
            logger.info("Arbeitnow: fetching page %d", page)
            try:
                data = get_json(_API_URL, params={"page": page})
            except SourceError:
                logger.error("Arbeitnow: page %d fetch failed — stopping pagination", page)
                break

            jobs: list[dict[str, Any]] = data.get("data", [])
            if not jobs:
                logger.info("Arbeitnow: no more results at page %d", page)
                break

            # Filter page results by keywords before deciding to continue
            keyword_matched = [j for j in jobs if _matches_keywords(j, config.keywords)]
            all_raw.extend(keyword_matched)

            logger.debug(
                "Arbeitnow page %d: %d total, %d keyword-matched",
                page,
                len(jobs),
                len(keyword_matched),
            )

            # Stop early if this page had zero keyword matches — results are drifting
            if not keyword_matched:
                logger.info(
                    "Arbeitnow: page %d had 0 keyword matches — stopping early", page
                )
                break

            page += 1
            if page <= _MAX_PAGES:
                time.sleep(_RATE_LIMIT_SECONDS)

        total_raw = len(all_raw)
        print(f"[arbeitnow] raw (keyword-matched) results: {total_raw}")

        # --- Location filtering with graceful relaxation ---
        city_filtered = [j for j in all_raw if _matches_city(j, config.target_city)]

        if len(city_filtered) < _MIN_RESULTS:
            logger.warning(
                "Arbeitnow: city filter '%s' left only %d result(s) (< %d minimum). "
                "Relaxing location filter — returning all keyword-matched results.",
                config.target_city,
                len(city_filtered),
                _MIN_RESULTS,
            )
            print(
                f"[arbeitnow] ⚠  city filter '{config.target_city}' matched only "
                f"{len(city_filtered)} result(s); relaxing to show all keyword matches."
            )
            filtered = all_raw
        else:
            filtered = city_filtered

        print(f"[arbeitnow] results after filtering: {len(filtered)}")

        return [_to_normalised(j) for j in filtered]
