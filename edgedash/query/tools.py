"""
edgedash/query/tools.py

Query tool registry. Deterministic. No LLM. No direct sqlite3.

Every tool is a read-only parameterised query over the last passing cycle.
All parameters are validated and clamped before use — treat every parameter
as untrusted input from a model, because it is.

Public API
----------
    TOOLS: dict[str, ToolSpec]   - registry
    @tool decorator              - registers a function
    call(name, **kwargs)         - validated dispatch

Each tool returns:
    {"rows": list[dict], "summary": str}
"""

from __future__ import annotations

import json
import re
from dataclasses import dataclass, field
from datetime import datetime, timezone, timedelta
from typing import Any, Callable, Optional

from edgedash import storage
from edgedash.config import load_config
from edgedash.skills import canonical as _canonical

# ---------------------------------------------------------------------------
# @tool decorator and registry
# ---------------------------------------------------------------------------

@dataclass
class ParamSpec:
    type: str                    # "int" | "str"
    description: str
    default: Any
    # for int params
    min: Optional[int] = None
    max: Optional[int] = None


@dataclass
class ToolSpec:
    name: str
    description: str             # what the router model sees — be specific
    params: dict[str, ParamSpec]
    fn: Callable[..., dict]


# Global registry — populated by @tool
TOOLS: dict[str, ToolSpec] = {}


def tool(
    description: str,
    params: dict[str, ParamSpec] | None = None,
) -> Callable:
    """Class decorator that registers a function in TOOLS.

    Usage::

        @tool(
            description="...",
            params={"n": ParamSpec(type="int", description="...", default=10, min=1, max=25)},
        )
        def best_matches(n: int = 10) -> dict: ...
    """
    def decorator(fn: Callable) -> Callable:
        TOOLS[fn.__name__] = ToolSpec(
            name=fn.__name__,
            description=description,
            params=params or {},
            fn=fn,
        )
        return fn
    return decorator


# ---------------------------------------------------------------------------
# Parameter validation and clamping (rule 41)
# ---------------------------------------------------------------------------

def _clamp(value: Any, spec: ParamSpec) -> Any:
    """Validate and clamp a single parameter value."""
    if spec.type == "int":
        try:
            v = int(value)
        except (TypeError, ValueError):
            v = spec.default
        if spec.min is not None:
            v = max(spec.min, v)
        if spec.max is not None:
            v = min(spec.max, v)
        return v
    # str
    if value is None:
        return spec.default
    return str(value).strip()


def _validate_params(spec: ToolSpec, raw: dict[str, Any]) -> dict[str, Any]:
    """Return a dict of validated, clamped parameters for *spec*."""
    out: dict[str, Any] = {}
    for name, pspec in spec.params.items():
        raw_val = raw.get(name, pspec.default)
        out[name] = _clamp(raw_val, pspec)
    return out


def call(name: str, **kwargs: Any) -> dict:
    """Look up and call a tool by name with validated parameters.

    Returns {"rows": [...], "summary": "..."}.
    Raises KeyError for unknown tool names.
    """
    spec = TOOLS[name]
    params = _validate_params(spec, kwargs)
    return spec.fn(**params)


# ---------------------------------------------------------------------------
# Shared helpers
# ---------------------------------------------------------------------------

def _cfg():
    return load_config()


def _db() -> str:
    return _cfg().db_path


def _utc_days_ago(n: int) -> str:
    """Return ISO timestamp for N days ago (UTC)."""
    dt = datetime.now(timezone.utc) - timedelta(days=n)
    return dt.isoformat()


def _latest_gap_run_at(db: str) -> Optional[str]:
    with storage._tx(db) as cx:
        row = storage._fetchone(
            cx, "SELECT MAX(run_at) AS v FROM gap_snapshots_v2", path=db
        )
        return row["v"] if row else None


def _skills_in_db(db: str) -> set[str]:
    """All distinct canonical skill names currently in gap_snapshots_v2."""
    run_at = _latest_gap_run_at(db)
    if not run_at:
        return set()
    p = storage._get_backend(db).ph()
    with storage._tx(db) as cx:
        rows = storage._fetchall(
            cx,
            f"SELECT DISTINCT skill FROM gap_snapshots_v2 WHERE run_at={p}",
            (run_at,),
            path=db,
        )
    return {r["skill"] for r in rows}


def _canonicalise_skill(raw: str, db: str) -> Optional[str]:
    """Return the canonical form of *raw* if it is present in the DB, else None."""
    aliases = _cfg().skill_aliases
    canon = _canonical(raw, aliases)
    known = _skills_in_db(db)
    # Accept exact match or prefix match (e.g. "typescript" matches "typescript")
    if canon in known:
        return canon
    # Fuzzy fallback: case-insensitive search through known skills
    canon_lower = canon.lower()
    for k in known:
        if k.lower() == canon_lower:
            return k
    return None


# ---------------------------------------------------------------------------
# Tool 1 — companies_hiring
# ---------------------------------------------------------------------------

@tool(
    description=(
        "Returns companies that have posted job listings in the last N days, "
        "with the count of their listings. Use this when the user asks who is "
        "hiring, which companies are active, or wants a hiring landscape overview. "
        "Not for questions about specific jobs or scores."
    ),
    params={
        "days": ParamSpec(
            type="int",
            description="How many days back to look (1-90). Default 7.",
            default=7,
            min=1,
            max=90,
        )
    },
)
def companies_hiring(days: int = 7) -> dict:
    days = max(1, min(90, int(days)))   # belt-and-suspenders clamp
    db = _db()
    cutoff = _utc_days_ago(days)
    p = storage._get_backend(db).ph()

    with storage._tx(db) as cx:
        rows = storage._fetchall(
            cx,
            f"""
            SELECT company,
                   COUNT(*) AS listing_count,
                   MAX(posted_at) AS latest_posting
            FROM listings
            WHERE posted_at >= {p}
              AND company IS NOT NULL
              AND company != ''
            GROUP BY company
            ORDER BY listing_count DESC, latest_posting DESC
            """,
            (cutoff,),
            path=db,
        )

    result = [
        {
            "company": r["company"],
            "listing_count": r["listing_count"],
            "latest_posting": str(r["latest_posting"] or "")[:10],
        }
        for r in rows
    ]
    summary = (
        f"{len(result)} companies with listings in the last {days} day(s)"
        if result else
        f"No listings found in the last {days} day(s)."
    )
    return {"rows": result, "summary": summary}


# ---------------------------------------------------------------------------
# Tool 2 — best_matches
# ---------------------------------------------------------------------------

@tool(
    description=(
        "Returns the N highest-scoring job listings, with score, title, company, "
        "location, URL, and the human-readable reason string explaining the score. "
        "Use this when the user asks for their best matches, top jobs, or "
        "which listings suit them most. Not for gap or company analysis."
    ),
    params={
        "n": ParamSpec(
            type="int",
            description="How many listings to return (1-25). Default 10.",
            default=10,
            min=1,
            max=25,
        )
    },
)
def best_matches(n: int = 10) -> dict:
    n = max(1, min(25, int(n)))
    db = _db()
    p = storage._get_backend(db).ph()

    with storage._tx(db) as cx:
        rows = storage._fetchall(
            cx,
            f"""
            SELECT title, company, location, url,
                   fit_score, fit_reason, posted_at
            FROM listings
            WHERE fit_score IS NOT NULL
            ORDER BY fit_score DESC
            LIMIT {p}
            """,
            (n,),
            path=db,
        )

    result = [
        {
            "score":      r["fit_score"],
            "title":      r["title"],
            "company":    r["company"],
            "location":   r["location"],
            "url":        r["url"],
            "reason":     r["fit_reason"] or "",
            "posted_at":  str(r["posted_at"] or "")[:10],
        }
        for r in rows
    ]
    summary = (
        f"Top {len(result)} listings by fit score"
        + (f" (scores {result[-1]['score']}\u2013{result[0]['score']})" if result else "")
    )
    return {"rows": result, "summary": summary}


# ---------------------------------------------------------------------------
# Tool 3 — top_gaps
# ---------------------------------------------------------------------------

@tool(
    description=(
        "Returns the top N skill gaps ranked by opportunity cost — the weighted "
        "sum of fit scores of listings blocked by each missing skill. "
        "Use this for gap analysis, learning priority questions, or 'what should "
        "I learn next'. opportunity_cost is the key metric, not raw frequency."
    ),
    params={
        "n": ParamSpec(
            type="int",
            description="How many gaps to return (1-25). Default 5.",
            default=5,
            min=1,
            max=25,
        )
    },
)
def top_gaps(n: int = 5) -> dict:
    n = max(1, min(25, int(n)))
    db = _db()
    run_at = _latest_gap_run_at(db)
    if not run_at:
        return {"rows": [], "summary": "No gap data yet."}
    p = storage._get_backend(db).ph()

    with storage._tx(db) as cx:
        rows = storage._fetchall(
            cx,
            f"""
            SELECT skill, opportunity_cost, listings_blocked,
                   mean_score, top_score, sample_size, low_confidence,
                   also_nice_to_have
            FROM gap_snapshots_v2
            WHERE run_at = {p}
            ORDER BY opportunity_cost DESC
            LIMIT {p}
            """,
            (run_at, n),
            path=db,
        )

    result = [
        {
            "skill":            r["skill"],
            "opportunity_cost": round(r["opportunity_cost"], 2),
            "listings_blocked": r["listings_blocked"],
            "mean_score":       round(r["mean_score"], 1),
            "top_score":        r["top_score"],
            "sample_size":      r["sample_size"],
            "low_confidence":   bool(r["low_confidence"]),
            "also_nice_to_have": r["also_nice_to_have"],
        }
        for r in rows
    ]
    summary = f"Top {len(result)} gaps from snapshot at {str(run_at)[:10]}"
    return {"rows": result, "summary": summary}


# ---------------------------------------------------------------------------
# Tool 4 — gap_detail  (rule 26 drill-down)
# ---------------------------------------------------------------------------

@tool(
    description=(
        "Returns the specific listings blocked by a single named skill gap, "
        "ranked by fit score. Use this when the user asks why a skill matters, "
        "which jobs require it, or wants to see the evidence behind a gap. "
        "Requires an exact skill name from top_gaps. Returns empty if the skill "
        "is not in the gap snapshot."
    ),
    params={
        "skill": ParamSpec(
            type="str",
            description="Canonical skill name to drill into (e.g. 'typescript').",
            default="",
        )
    },
)
def gap_detail(skill: str = "") -> dict:
    if not skill:
        return {"rows": [], "summary": "No skill specified."}

    db = _db()
    canon = _canonicalise_skill(skill, db)
    if canon is None:
        return {
            "rows": [],
            "summary": f"Skill '{skill}' not found in the gap snapshot.",
        }

    run_at = _latest_gap_run_at(db)
    if not run_at:
        return {"rows": [], "summary": "No gap data yet."}

    p = storage._get_backend(db).ph()

    # Get example_ids from the snapshot
    with storage._tx(db) as cx:
        row = storage._fetchone(
            cx,
            f"SELECT opportunity_cost, listings_blocked, example_ids "
            f"FROM gap_snapshots_v2 WHERE run_at={p} AND skill={p}",
            (run_at, canon),
            path=db,
        )

    if not row:
        return {
            "rows": [],
            "summary": f"'{canon}' not in current gap snapshot.",
        }

    opp_cost = row["opportunity_cost"]
    blocked  = row["listings_blocked"]
    try:
        example_ids: list[str] = json.loads(row["example_ids"])
    except (json.JSONDecodeError, TypeError):
        example_ids = []

    if not example_ids:
        return {
            "rows": [],
            "summary": (
                f"Gap '{canon}' blocks {blocked} listings "
                f"(cost {opp_cost:.2f}) but no example IDs were stored."
            ),
        }

    # Fetch the actual listing rows — use placeholders, never interpolation
    placeholders = ",".join([p] * len(example_ids))
    with storage._tx(db) as cx:
        listings = storage._fetchall(
            cx,
            f"SELECT title, company, location, url, fit_score, fit_reason "
            f"FROM listings WHERE id IN ({placeholders}) "
            f"ORDER BY fit_score DESC",
            tuple(example_ids),
            path=db,
        )

    result = [
        {
            "score":    l["fit_score"],
            "title":    l["title"],
            "company":  l["company"],
            "location": l["location"],
            "url":      l["url"],
            "reason":   l["fit_reason"] or "",
        }
        for l in listings
    ]
    summary = (
        f"'{canon}' blocks {blocked} listings (opportunity cost {opp_cost:.2f}); "
        f"showing {len(result)} example(s)"
    )
    return {"rows": result, "summary": summary}


# ---------------------------------------------------------------------------
# Tool 5 — trend
# ---------------------------------------------------------------------------

@tool(
    description=(
        "Compares gap opportunity_cost across the last N weekly snapshots to "
        "show which skill gaps are growing, shrinking, or new. Use this for "
        "trend questions: 'is X getting worse?', 'what gaps appeared recently?', "
        "'has my priority list changed?'"
    ),
    params={
        "weeks": ParamSpec(
            type="int",
            description="How many recent weekly snapshots to compare (1-12). Default 3.",
            default=3,
            min=1,
            max=12,
        )
    },
)
def trend(weeks: int = 3) -> dict:
    weeks = max(1, min(12, int(weeks)))
    db = _db()
    p = storage._get_backend(db).ph()

    with storage._tx(db) as cx:
        # Get the N most-recent distinct run_ats
        run_ats_rows = storage._fetchall(
            cx,
            f"SELECT DISTINCT run_at FROM gap_snapshots_v2 "
            f"ORDER BY run_at DESC LIMIT {p}",
            (weeks,),
            path=db,
        )

        if not run_ats_rows:
            return {"rows": [], "summary": "No gap snapshot data yet."}

        run_ats = [str(r["run_at"]) for r in run_ats_rows]
        # Load skill → cost for each snapshot
        snapshots: dict[str, dict[str, float]] = {}
        for run_at in run_ats:
            rows = storage._fetchall(
                cx,
                f"SELECT skill, opportunity_cost FROM gap_snapshots_v2 "
                f"WHERE run_at = {p}",
                (run_at,),
                path=db,
            )
            snapshots[run_at] = {r["skill"]: r["opportunity_cost"] for r in rows}

    if len(run_ats) < 2:
        # Only one snapshot — report it as-is, no delta
        latest_at = run_ats[0]
        rows_out = [
            {
                "skill": skill,
                "latest_cost": round(cost, 2),
                "earliest_cost": None,
                "delta": None,
                "pct_change": None,
                "status": "only_one_snapshot",
            }
            for skill, cost in sorted(
                snapshots[latest_at].items(),
                key=lambda x: x[1], reverse=True
            )[:10]
        ]
        return {
            "rows": rows_out,
            "summary": (
                f"Only 1 snapshot found ({latest_at[:10]}); "
                "need at least 2 to show a trend."
            ),
        }

    latest_at   = run_ats[0]
    earliest_at = run_ats[-1]
    latest_map   = snapshots[latest_at]
    earliest_map = snapshots[earliest_at]

    all_skills = set(latest_map) | set(earliest_map)
    rows_out = []
    for skill in all_skills:
        now_cost  = latest_map.get(skill)
        then_cost = earliest_map.get(skill)

        if now_cost is None:
            status = "dropped_out"
            delta = None
            pct   = None
        elif then_cost is None:
            status = "new"
            delta = None
            pct   = None
        else:
            delta = round(now_cost - then_cost, 3)
            pct   = round(delta / then_cost * 100, 1) if then_cost else 0.0
            status = "rising" if delta > 0 else ("falling" if delta < 0 else "stable")

        rows_out.append({
            "skill":          skill,
            "latest_cost":    round(now_cost, 2) if now_cost is not None else None,
            "earliest_cost":  round(then_cost, 2) if then_cost is not None else None,
            "delta":          delta,
            "pct_change":     pct,
            "status":         status,
        })

    # Sort: rising first, then new, then falling, then dropped
    order = {"rising": 0, "new": 1, "stable": 2, "falling": 3, "dropped_out": 4}
    rows_out.sort(key=lambda x: (order.get(x["status"], 9),
                                  -(x["latest_cost"] or 0)))

    summary = (
        f"Trend over {len(run_ats)} snapshots "
        f"({earliest_at[:10]} → {latest_at[:10]}); "
        f"{sum(1 for r in rows_out if r['status']=='rising')} rising, "
        f"{sum(1 for r in rows_out if r['status']=='new')} new, "
        f"{sum(1 for r in rows_out if r['status']=='falling')} falling"
    )
    return {"rows": rows_out, "summary": summary}


# ---------------------------------------------------------------------------
# Tool 6 — listing_count
# ---------------------------------------------------------------------------

@tool(
    description=(
        "Returns high-level counts: total listings, scored, unscored, and the "
        "date of the newest listing. Use this for pipeline status questions: "
        "'how many jobs have been analysed?', 'is the database up to date?', "
        "'how complete is the scoring?'"
    ),
    params={},
)
def listing_count() -> dict:
    db = _db()
    with storage._tx(db) as cx:
        r_total = storage._fetchone(cx, "SELECT COUNT(*) AS n FROM listings", path=db)
        total = r_total["n"] if r_total else 0
    with storage._tx(db) as cx:
        r_scored = storage._fetchone(
            cx,
            "SELECT COUNT(*) AS n FROM listings WHERE fit_score IS NOT NULL",
            path=db,
        )
        scored = r_scored["n"] if r_scored else 0
    with storage._tx(db) as cx:
        r_newest = storage._fetchone(
            cx, "SELECT MAX(posted_at) AS v FROM listings", path=db
        )
        newest = r_newest["v"] if r_newest else None

    unscored = total - scored
    pct = round(100 * scored / total, 1) if total else 0.0
    row = {
        "total_listings": total,
        "scored":         scored,
        "unscored":       unscored,
        "pct_scored":     pct,
        "newest_listing": str(newest or "")[:10],
    }
    summary = (
        f"{total} listings total; {scored} scored ({pct}%), "
        f"{unscored} unscored; newest posted {str(newest or '?')[:10]}"
    )
    return {"rows": [row], "summary": summary}


# ---------------------------------------------------------------------------
# Tool 7 — skill_demand
# ---------------------------------------------------------------------------

@tool(
    description=(
        "Returns how often a named skill appears as required vs nice-to-have "
        "across all extracted job listings. Use this when the user asks how "
        "important a skill is, whether a skill is commonly required or just "
        "preferred, or to compare demand for two skills. Requires a skill name."
    ),
    params={
        "skill": ParamSpec(
            type="str",
            description="Skill name to look up (e.g. 'kubernetes', 'python').",
            default="",
        )
    },
)
def skill_demand(skill: str = "") -> dict:
    if not skill:
        return {"rows": [], "summary": "No skill specified."}

    aliases = _cfg().skill_aliases
    # Canonicalise but do NOT require it to be in the gap snapshot —
    # it might be a skill I already have, so it would have zero gap cost.
    canon = _canonical(skill, aliases)

    db = _db()
    with storage._tx(db) as cx:
        cache_rows = storage._fetchall(
            cx, "SELECT extraction_json FROM extraction_cache", path=db
        )

    required_count  = 0
    nice_count      = 0
    total_listings  = 0

    for r in cache_rows:
        raw_json = r.get("extraction_json", "")
        try:
            data = json.loads(raw_json)
        except (json.JSONDecodeError, TypeError):
            continue

        total_listings += 1
        req_skills = [
            _canonical(s, aliases)
            for s in (data.get("required_skills") or [])
        ]
        nice_skills = [
            _canonical(s, aliases)
            for s in (data.get("nice_to_have") or [])
        ]

        if canon in req_skills:
            required_count += 1
        if canon in nice_skills:
            nice_count += 1

    if required_count == 0 and nice_count == 0:
        return {
            "rows": [],
            "summary": (
                f"'{canon}' was not found in any extracted skill list "
                f"({total_listings} listings searched)."
            ),
        }

    req_pct  = round(100 * required_count / total_listings, 1) if total_listings else 0
    nice_pct = round(100 * nice_count    / total_listings, 1) if total_listings else 0

    row = {
        "skill":           canon,
        "required_in":     required_count,
        "required_pct":    req_pct,
        "nice_to_have_in": nice_count,
        "nice_pct":        nice_pct,
        "total_listings":  total_listings,
    }
    summary = (
        f"'{canon}': required in {required_count}/{total_listings} listings "
        f"({req_pct}%), nice-to-have in {nice_count} ({nice_pct}%)"
    )
    return {"rows": [row], "summary": summary}
