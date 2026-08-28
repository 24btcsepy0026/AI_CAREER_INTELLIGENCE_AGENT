"""
edgedash/agents/gap_analyzer.py

GapAnalyzer — identifies and ranks skill gaps from scored listings.
Deterministic SQL + Python. Zero LLM calls anywhere. (Rules 22-27)

opportunity_cost arithmetic
---------------------------
For each skill gap S, for every scored listing L that requires S:

    opportunity_cost(S) = sum( L.fit_score / 100  for each L )

This means a gap in a listing scored 80 contributes 0.80 to the cost;
a gap in a listing scored 25 contributes 0.25.  Same raw count of listings
can produce very different opportunity costs — that is the whole point.
(Rule 24: weighted by score, never raw frequency alone.)

also_nice_to_have is tracked SEPARATELY and never mixed into required counts.
"required in 5 listings" and "preferred in 5 listings" are different facts.
"""

from __future__ import annotations

import hashlib
import json
import logging
import time
from collections import defaultdict
from datetime import datetime, timezone
from typing import Any

from edgedash import storage
from edgedash.agents.base import AgentResult
from edgedash.config import Config
from edgedash.planning import StopConditions
from edgedash.skills import canonical

logger = logging.getLogger(__name__)

_LOW_CONFIDENCE_THRESHOLD = 3   # rule 27: flag gaps from fewer than this many listings
_TOP_N = 10                      # report top N gaps in AgentResult notes


def _utcnow() -> str:
    return datetime.now(timezone.utc).isoformat()


def _desc_hash(description: str) -> str:
    return hashlib.sha256(description.encode("utf-8")).hexdigest()[:32]


def _load_facts(listing: dict[str, Any], db_path: str) -> dict[str, Any] | None:
    """Fetch extraction facts for a listing from the cache."""
    description = (listing.get("description") or "").strip()
    if not description:
        return None
    h = _desc_hash(description)
    return storage.get_cached_extraction(db_path, h)


# ---------------------------------------------------------------------------
# Core arithmetic
# ---------------------------------------------------------------------------

def _compute_gaps(
    listings: list[dict[str, Any]],
    facts_map: dict[str, dict[str, Any]],
    my_skills: set[str],
    aliases: dict[str, list[str]],
) -> list[dict[str, Any]]:
    """Compute opportunity-cost-weighted gap report.

    opportunity_cost = sum(fit_score / 100) for each listing that requires the skill.

    Returns list of gap dicts sorted by opportunity_cost descending.
    """
    # canonical_skill -> accumulators
    required_acc: dict[str, dict[str, Any]] = defaultdict(lambda: {
        "opportunity_cost": 0.0,
        "listings_blocked": 0,
        "score_sum": 0.0,
        "top_score": 0,
        "example_ids": [],       # up to 5, highest score first (rule 26)
    })
    nice_acc: dict[str, int] = defaultdict(int)   # also_nice_to_have count

    for listing in listings:
        lid = listing["id"]
        score = listing.get("fit_score") or 0
        facts = facts_map.get(lid)
        if facts is None:
            continue

        # --- required skills gaps ---
        for raw in (facts.get("required_skills") or []):
            canon = canonical(raw, aliases)
            if canon in my_skills:
                continue
            acc = required_acc[canon]
            acc["opportunity_cost"] += score / 100.0   # ← the key arithmetic
            acc["listings_blocked"] += 1
            acc["score_sum"] += score
            if score > acc["top_score"]:
                acc["top_score"] = score
            acc["example_ids"].append((score, lid))

        # --- nice_to_have (tracked separately, never mixed in) ---
        for raw in (facts.get("nice_to_have") or []):
            canon = canonical(raw, aliases)
            if canon in my_skills:
                continue
            nice_acc[canon] += 1

    # Build result rows
    result = []
    for skill, acc in required_acc.items():
        n = acc["listings_blocked"]
        mean_score = round(acc["score_sum"] / n, 1) if n > 0 else 0.0
        # Sort examples by score desc, keep top 5 (rule 26)
        top_examples = [lid for _, lid in sorted(acc["example_ids"], reverse=True)[:5]]
        result.append({
            "skill": skill,
            "opportunity_cost": round(acc["opportunity_cost"], 3),
            "listings_blocked": n,
            "mean_score": mean_score,
            "top_score": acc["top_score"],
            "sample_size": n,               # rule 27
            "low_confidence": n < _LOW_CONFIDENCE_THRESHOLD,
            "also_nice_to_have": nice_acc.get(skill, 0),
            "example_ids": top_examples,    # rule 26
        })

    result.sort(key=lambda x: x["opportunity_cost"], reverse=True)
    return result


# ---------------------------------------------------------------------------
# Agent
# ---------------------------------------------------------------------------

class GapAnalyzer:
    name: str = "GapAnalyzer"

    def run(
        self,
        config: Config,
        db_path: str,
        stop: StopConditions = StopConditions(),
    ) -> AgentResult:
        max_seconds = stop.max_seconds or 0
        wall_start  = time.monotonic()

        # Load all scored listings that also have a description (need cache lookup)
        scored = storage.get_scored_listings_with_descriptions(db_path)

        if not scored:
            return AgentResult(
                agent=self.name, status="ok", records_touched=0,
                notes="no scored listings yet — run Scorer first",
            )

        # Fetch facts from extraction cache for each listing
        facts_map: dict[str, dict[str, Any]] = {}
        for listing in scored:
            if max_seconds and (time.monotonic() - wall_start) >= max_seconds:
                logger.warning("GapAnalyzer: max_seconds=%d reached during fact loading", max_seconds)
                break
            facts = _load_facts(listing, db_path)
            if facts is not None:
                facts_map[listing["id"]] = facts

        if not facts_map:
            return AgentResult(
                agent=self.name, status="ok", records_touched=0,
                notes="no cached extractions found — run Scorer first",
            )

        my_skills = {canonical(s, config.skill_aliases) for s in config.my_skills}
        gaps = _compute_gaps(scored, facts_map, my_skills, config.skill_aliases)

        if not gaps:
            return AgentResult(
                agent=self.name, status="ok", records_touched=0,
                notes=f"no gaps found across {len(facts_map)} analysed listings",
            )

        # Rule 25: write NEW timestamped snapshot — never overwrite
        run_at = _utcnow()
        storage.save_gap_snapshot_v2(db_path, run_at, gaps)

        # Build notes for orchestrator display
        top = gaps[:_TOP_N]
        top_str = ", ".join(
            f"{g['skill']} ({g['listings_blocked']} listings, cost {g['opportunity_cost']:.1f})"
            for g in top[:3]
        )
        lc_count = sum(1 for g in gaps if g["low_confidence"])
        notes = (
            f"{len(gaps)} gaps · top: {top_str} · "
            f"{len(facts_map)} listings analysed"
            + (f" · {lc_count} low-confidence" if lc_count else "")
        )
        logger.info("GapAnalyzer: %s", notes)

        return AgentResult(
            agent=self.name,
            status="ok",
            records_touched=len(gaps),
            notes=notes,
        )
