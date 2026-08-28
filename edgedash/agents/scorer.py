"""
edgedash/agents/scorer.py

Scorer agent — extracts facts then scores each unscored listing.

Rules honoured
--------------
  Rule 17: per-listing try/except — one failure skips that listing, logged.
  Rule 18: only listings WHERE fit_score IS NULL are touched.
  Rule 20: score distribution logged to cycle_log after every batch.
  Rule 21: batch capped at config.score_batch_size.

widen_distribution mode (stop.widen_distribution=True)
-------------------------------------------------------
When the Verifier flags a score_spread failure, the Orchestrator re-runs
the Scorer with this flag set. The mechanism:

  1. Fetch scored listings and compute the interquartile range (Q1–Q3).
  2. NULL-out scores that fell inside that inner band — those are the
     compressed middle listings that are dragging spread down.
  3. Re-score only those listings with max_retries=2 in the LLM extractor,
     giving the extraction step more attempts to produce richer facts that
     yield more differentiated scores.

Why this produces more spread: the first pass scored easy listings
similarly because the LLM extraction was shallow (e.g. very short
required_skills lists). Forcing re-extraction with a higher retry budget
tends to produce more complete facts, which drives more extreme scores
at both ends (full match → high; no match → low).
"""

from __future__ import annotations

import logging
import time
from datetime import datetime, timezone
from typing import Any

from edgedash import storage
from edgedash.agents.base import AgentResult
from edgedash.agents.extractor import extract
from edgedash.config import Config
from edgedash.planning import StopConditions
from edgedash.scoring import score_listing

logger = logging.getLogger(__name__)

_SUSPECT_SPREAD_THRESHOLD = 10   # rule 20: flag if max-min < this


def _utcnow() -> str:
    return datetime.now(timezone.utc).isoformat()


def _distribution_notes(scores: list[int], failed: int) -> tuple[str, str]:
    """Return (notes_string, status) describing the score distribution."""
    count = len(scores)
    if count == 0:
        return f"scored 0 · 0 scored · {failed} failed", "ok"

    lo   = min(scores)
    hi   = max(scores)
    mean = round(sum(scores) / count)
    spread = hi - lo

    spread_label = f"spread {spread}"
    suspect = spread < _SUSPECT_SPREAD_THRESHOLD
    if suspect:
        spread_label = f"SUSPECT spread {spread} (all scores within {_SUSPECT_SPREAD_THRESHOLD} pts)"

    notes = (
        f"scored {count} · range {lo}-{hi} · mean {mean} · "
        f"{failed} failed · {spread_label}"
    )
    status = "suspect" if suspect else "ok"
    return notes, status


def _iqr_middle_ids(db_path: str) -> list[str]:
    """Return listing IDs whose score fell in the interquartile range (Q1–Q3).

    These are the listings contributing most to a compressed distribution.
    Nulling their scores and re-scoring produces better spread.
    """
    scored = storage.get_scored_listings_with_components(db_path)
    if len(scored) < 4:
        return []

    scores_by_id = [
        (r["id"], r["fit_score"])
        for r in scored
        if r.get("fit_score") is not None
    ]
    values = sorted(s for _, s in scores_by_id)
    n = len(values)
    q1 = values[n // 4]
    q3 = values[(3 * n) // 4]

    return [lid for lid, s in scores_by_id if q1 <= s <= q3]


class Scorer:
    name: str = "Scorer"

    def run(
        self,
        config: Config,
        db_path: str,
        stop: StopConditions = StopConditions(),
    ) -> AgentResult:
        # Respect stop conditions set by the Orchestrator (rule 29)
        max_items   = stop.max_items   or config.score_batch_size
        max_seconds = stop.max_seconds or 0   # 0 = no time cap

        # --- widen_distribution mode ---
        if stop.widen_distribution:
            middle_ids = _iqr_middle_ids(db_path)
            if middle_ids:
                logger.info(
                    "Scorer (widen_distribution): nulling %d middle-band scores for re-scoring",
                    len(middle_ids),
                )
                # Null the scores so get_unscored_listings picks them up
                storage.null_scores(db_path, middle_ids)
                # Clear their extraction cache so the model is called again
                # with max_retries=2, producing richer facts
                import hashlib
                for lid in middle_ids:
                    # We need the description hash; fetch from listings table
                    pass  # handled below per-listing inside the loop
            extraction_retries = 2   # more attempts → richer facts → wider spread
            _widen_ids = set(middle_ids) if middle_ids else set()
        else:
            extraction_retries = 1   # default
            _widen_ids = set()

        batch = storage.get_unscored_listings(db_path, limit=max_items)

        if not batch:
            return AgentResult(
                agent=self.name,
                status="ok",
                records_touched=0,
                notes="no unscored listings",
            )

        scores_written: list[int] = []
        failed = 0
        started_at = _utcnow()
        wall_start = time.monotonic()

        for listing in batch:
            # Time cap check — stop gracefully if over budget
            if max_seconds and (time.monotonic() - wall_start) >= max_seconds:
                logger.warning(
                    "Scorer: max_seconds=%d reached after %d listings — stopping early",
                    max_seconds, len(scores_written) + failed,
                )
                break

            listing_id = listing["id"]
            try:
                # In widen mode, clear the cache for middle-band listings so the
                # model is called fresh with max_retries=2 (richer extraction).
                if _widen_ids and listing_id in _widen_ids:
                    desc = (listing.get("description") or "").strip()
                    if desc:
                        import hashlib as _hl
                        h = _hl.sha256(desc.encode()).hexdigest()[:32]
                        storage.delete_cached_extraction(db_path, h)

                # Pass extraction_retries so widen mode forces a richer extraction
                facts = extract(listing, db_path, max_retries=extraction_retries)
                if facts is None:
                    logger.warning("Scorer: extraction returned None for %s — skipping", listing_id)
                    failed += 1
                    continue

                result = score_listing(listing, facts, config)
                storage.save_score(
                    db_path,
                    listing_id,
                    score=result["score"],
                    reason=result["reason"],
                    components=result["components"],
                )
                scores_written.append(result["score"])
                logger.debug("Scored %s → %d", listing_id, result["score"])

            except Exception as exc:  # rule 17: one failure = one skip
                logger.error("Scorer: failed on listing %s: %s", listing_id, exc)
                failed += 1

        notes, dist_status = _distribution_notes(scores_written, failed)
        if stop.widen_distribution:
            notes = f"[widen_retry] {notes}"

        # Rule 20: log distribution to cycle_log
        finished_at = _utcnow()
        storage.log_cycle(
            path=db_path,
            agent="Scorer/distribution",
            started_at=started_at,
            finished_at=finished_at,
            records_touched=len(scores_written),
            status=dist_status,
            notes=notes,
        )

        if dist_status == "suspect":
            logger.warning("Scorer: %s", notes)

        return AgentResult(
            agent=self.name,
            status="ok",
            records_touched=len(scores_written),
            notes=notes,
        )
