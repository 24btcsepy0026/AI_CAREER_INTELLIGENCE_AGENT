"""
edgedash/scoring.py

Deterministic scoring arithmetic. Pure functions only.
No network. No model calls. No imports from llm.py (steering rule 16).

Public API
----------
    score_listing(listing, facts, config) -> {"score": int, "reason": str, "components": dict}
    build_reason(components, facts, config) -> str
"""

from __future__ import annotations

import math
from datetime import datetime, timezone
from typing import Any

from edgedash.config import Config

# ---------------------------------------------------------------------------
# Seniority band scale
# Ordered from junior (0) to lead (3). Distance between bands drives the penalty.
# ---------------------------------------------------------------------------

_SENIORITY_BANDS: dict[str, int] = {
    "junior":  0,
    "mid":     1,
    "senior":  2,
    "lead":    3,
    "unknown": -1,   # sentinel — handled separately
}

_SENIORITY_SCORE: dict[int, float] = {
    0: 1.0,   # exact match
    1: 0.6,   # one band away
    2: 0.25,  # two bands away
}
# three or more bands away → 0.0


def _component_skill_match(facts: dict[str, Any], config: Config) -> tuple[float, dict]:
    """Fraction of required skills present in config.my_skills + nice_to_have at 1/3 weight.

    Returns (normalised_score 0.0-1.0, detail_dict).
    Handles empty required_skills without dividing by zero.
    """
    my_skills = {s.lower().strip() for s in config.my_skills}

    required: list[str] = facts.get("required_skills") or []
    nice: list[str] = facts.get("nice_to_have") or []

    if not required and not nice:
        # No skill data at all → neutral 0.5
        return 0.5, {"matched_required": 0, "total_required": 0,
                     "matched_nice": 0, "total_nice": 0, "score": 0.5}

    if not required:
        # Only nice-to-have present — score purely on those
        matched_nice = sum(1 for s in nice if s in my_skills)
        raw = matched_nice / len(nice)
        return raw, {"matched_required": 0, "total_required": 0,
                     "matched_nice": matched_nice, "total_nice": len(nice), "score": raw}

    matched_required = sum(1 for s in required if s in my_skills)
    matched_nice = sum(1 for s in nice if s in my_skills)

    # Required counts at full weight; nice-to-have at 1/3
    numerator = matched_required + (matched_nice / 3.0)
    denominator = len(required) + (len(nice) / 3.0)
    raw = numerator / denominator  # denominator always > 0 because required is non-empty

    return raw, {
        "matched_required": matched_required,
        "total_required": len(required),
        "matched_nice": matched_nice,
        "total_nice": len(nice),
        "score": raw,
    }


def _component_seniority_fit(facts: dict[str, Any], config: Config) -> tuple[float, dict]:
    """Compare facts.seniority to config.target_seniority on the band scale."""
    target = config.target_seniority.lower().strip()
    actual = (facts.get("seniority") or "unknown").lower().strip()

    target_band = _SENIORITY_BANDS.get(target, 1)   # default to mid if unknown target
    actual_band = _SENIORITY_BANDS.get(actual, -1)

    if actual_band == -1:
        # Listing seniority is unknown — neutral 0.5
        raw = 0.5
    else:
        distance = abs(target_band - actual_band)
        raw = _SENIORITY_SCORE.get(distance, 0.0)

    return raw, {"target": target, "actual": actual, "score": raw}


def _component_location_fit(facts: dict[str, Any], listing: dict[str, Any], config: Config) -> tuple[float, dict]:
    """Score location: remote > city match > unknown > clearly elsewhere."""
    remote_ok = facts.get("remote_ok")            # True | False | None
    location: str = (listing.get("location") or "").lower()
    city: str = config.target_city.lower().strip()

    if remote_ok is True:
        raw = 1.0
        label = "remote"
    elif city and city in location:
        raw = 1.0
        label = "city_match"
    elif remote_ok is None:
        raw = 0.5
        label = "unknown"
    else:
        # remote_ok is False and city didn't match → clearly elsewhere
        raw = 0.1
        label = "elsewhere"

    return raw, {"remote_ok": remote_ok, "location": listing.get("location"), "label": label, "score": raw}


def _component_recency(listing: dict[str, Any]) -> tuple[float, dict]:
    """Decay from 1.0 today to 0.0 at 30 days. Null posted_at → 0.5."""
    posted_at: str | None = listing.get("posted_at")

    if not posted_at:
        return 0.5, {"posted_at": None, "days_ago": None, "score": 0.5}

    try:
        # Parse ISO-8601; strip timezone if needed for comparison
        dt = datetime.fromisoformat(posted_at)
        if dt.tzinfo is None:
            dt = dt.replace(tzinfo=timezone.utc)
        now = datetime.now(timezone.utc)
        days_ago = max(0.0, (now - dt).total_seconds() / 86400.0)
    except (ValueError, TypeError):
        return 0.5, {"posted_at": posted_at, "days_ago": None, "score": 0.5}

    # Linear decay: 1.0 at day 0, 0.0 at day 30
    raw = max(0.0, 1.0 - days_ago / 30.0)
    return raw, {"posted_at": posted_at, "days_ago": round(days_ago, 1), "score": raw}


# ---------------------------------------------------------------------------
# Public API
# ---------------------------------------------------------------------------

def score_listing(
    listing: dict[str, Any],
    facts: dict[str, Any],
    config: Config,
) -> dict[str, Any]:
    """Compute fit score for *listing* given extracted *facts* and *config*.

    Returns:
        {
            "score":      int  0-100,
            "reason":     str  (human-readable, generated from components),
            "components": {
                "skill_match":   {"score": float, ...},
                "seniority_fit": {"score": float, ...},
                "location_fit":  {"score": float, ...},
                "recency":       {"score": float, ...},
                "weights":       {...},
            }
        }
    """
    w_skill    = config.weight_skill_match
    w_seniority = config.weight_seniority_fit
    w_location  = config.weight_location_fit
    w_recency   = config.weight_recency

    skill_score,    skill_detail    = _component_skill_match(facts, config)
    seniority_score, seniority_detail = _component_seniority_fit(facts, config)
    location_score, location_detail = _component_location_fit(facts, listing, config)
    recency_score,  recency_detail  = _component_recency(listing)

    weighted = (
        skill_score    * w_skill
        + seniority_score * w_seniority
        + location_score  * w_location
        + recency_score   * w_recency
    )

    score = round(weighted * 100)
    score = max(0, min(100, score))   # clamp to [0, 100]

    components = {
        "skill_match":   skill_detail,
        "seniority_fit": seniority_detail,
        "location_fit":  location_detail,
        "recency":       recency_detail,
        "weights": {
            "skill_match":   w_skill,
            "seniority_fit": w_seniority,
            "location_fit":  w_location,
            "recency":       w_recency,
        },
    }

    reason = build_reason(components, facts, config)

    return {"score": score, "reason": reason, "components": components}


def build_reason(
    components: dict[str, Any],
    facts: dict[str, Any],
    config: Config,
) -> str:
    """Build a compact human-readable reason string FROM the score components.

    Format: "4/6 required skills · seniority fits · remote · posted 2d ago · gap: kubernetes, spark"
    Rule 19: assembled deterministically from numbers — never free text from the model.
    """
    parts: list[str] = []

    # --- Skill match ---
    sk = components["skill_match"]
    m_req = sk["matched_required"]
    t_req = sk["total_required"]
    if t_req > 0:
        parts.append(f"{m_req}/{t_req} required skills")
    elif sk["total_nice"] > 0:
        parts.append(f"{sk['matched_nice']}/{sk['total_nice']} preferred skills")
    else:
        parts.append("no skill data")

    # --- Seniority ---
    sen = components["seniority_fit"]
    sen_score = sen["score"]
    if sen_score == 1.0:
        parts.append("seniority fits")
    elif sen_score >= 0.6:
        parts.append(f"seniority close ({sen['actual']} vs {sen['target']})")
    elif sen["actual"] == "unknown":
        parts.append("seniority unknown")
    else:
        parts.append(f"seniority mismatch ({sen['actual']} vs {sen['target']})")

    # --- Location ---
    loc = components["location_fit"]
    label = loc.get("label", "unknown")
    if label == "remote":
        parts.append("remote")
    elif label == "city_match":
        parts.append(f"in {config.target_city}")
    elif label == "unknown":
        parts.append("location unknown")
    else:
        parts.append(f"not remote/local ({loc.get('location') or 'unknown location'})")

    # --- Recency ---
    rec = components["recency"]
    days = rec.get("days_ago")
    if days is None:
        parts.append("date unknown")
    elif days < 1:
        parts.append("posted today")
    else:
        parts.append(f"posted {math.floor(days)}d ago")

    # --- Skill gaps (the most actionable part) ---
    my_skills = {s.lower().strip() for s in config.my_skills}
    required: list[str] = facts.get("required_skills") or []
    gaps = [s for s in required if s not in my_skills]
    if gaps:
        gap_str = ", ".join(gaps[:5])   # cap at 5 to keep it readable
        if len(gaps) > 5:
            gap_str += f" (+{len(gaps) - 5} more)"
        parts.append(f"gap: {gap_str}")

    return " · ".join(parts)
