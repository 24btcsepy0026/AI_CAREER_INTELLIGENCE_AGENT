"""
edgedash/verification.py

Deterministic verification checks. No LLM anywhere in this file.
A model cannot be the judge of a model's own output.

Public API
----------
    check_score_spread(scores, config)         -> CheckResult
    check_extraction_sanity(facts_list, config) -> CheckResult
    check_gap_sample_size(gaps, config)        -> CheckResult
    check_freshness(latest_fetch_at, config, now) -> CheckResult
    run_all_checks(...) -> Verdict

Design rules
------------
  - Every check is a pure function of its arguments.
  - No clock, no network, no database reads inside any check.
  - `now` is always a parameter, never datetime.now() inside a function.
  - Thresholds come exclusively from config — never hardcoded here.
"""

from __future__ import annotations

import math
from dataclasses import dataclass, field
from datetime import datetime, timezone
from typing import Any, Optional

from edgedash.config import Config

# ---------------------------------------------------------------------------
# Result types
# ---------------------------------------------------------------------------

@dataclass
class CheckResult:
    name: str
    passed: bool
    observed: str      # what we actually measured, as a human-readable string
    threshold: str     # what the threshold is, for side-by-side comparison
    message: str       # full sentence verdict — copy-pasteable into a report


@dataclass
class Verdict:
    passed: bool
    failed_checks: list[CheckResult]
    all_checks: list[CheckResult]

    @property
    def summary(self) -> str:
        total = len(self.all_checks)
        failed = len(self.failed_checks)
        if failed == 0:
            return f"All {total} checks passed."
        names = ", ".join(c.name for c in self.failed_checks)
        return f"{failed}/{total} checks FAILED: {names}"


# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------

_MIN_SCORES_FOR_SPREAD = 5   # fewer than this → trivially pass with explanation


def _stdev(values: list[float]) -> float:
    """Population standard deviation. Returns 0.0 for fewer than 2 values."""
    n = len(values)
    if n < 2:
        return 0.0
    mean = sum(values) / n
    variance = sum((v - mean) ** 2 for v in values) / n
    return math.sqrt(variance)


def _parse_iso(ts: str) -> Optional[datetime]:
    try:
        dt = datetime.fromisoformat(ts)
        if dt.tzinfo is None:
            dt = dt.replace(tzinfo=timezone.utc)
        return dt
    except (ValueError, TypeError):
        return None


# ---------------------------------------------------------------------------
# check_score_spread
# Catches score compression / inflation.
# ---------------------------------------------------------------------------

def check_score_spread(scores: list[int], config: Config) -> CheckResult:
    """Fail if score spread (max-min) or stdev is below thresholds.

    Passes trivially when fewer than _MIN_SCORES_FOR_SPREAD scores are present
    — not enough data to judge, so we don't flag a false failure.
    """
    name = "score_spread"

    if len(scores) < _MIN_SCORES_FOR_SPREAD:
        return CheckResult(
            name=name,
            passed=True,
            observed=f"n={len(scores)} scores",
            threshold=f"need >= {_MIN_SCORES_FOR_SPREAD} to evaluate",
            message=(
                f"PASS (trivial): only {len(scores)} score(s) present — "
                f"need at least {_MIN_SCORES_FOR_SPREAD} to evaluate spread."
            ),
        )

    lo = min(scores)
    hi = max(scores)
    spread = hi - lo
    stdev = _stdev([float(s) for s in scores])

    spread_ok = spread >= config.min_score_spread
    stdev_ok  = stdev  >= config.min_score_stdev
    passed    = spread_ok and stdev_ok

    observed  = f"spread={spread}, stdev={stdev:.1f} (n={len(scores)})"
    threshold = (
        f"min_spread={config.min_score_spread}, "
        f"min_stdev={config.min_score_stdev}"
    )

    if passed:
        message = (
            f"PASS: score spread={spread} >= {config.min_score_spread} "
            f"and stdev={stdev:.1f} >= {config.min_score_stdev}."
        )
    else:
        parts: list[str] = []
        if not spread_ok:
            parts.append(
                f"spread={spread} < min_score_spread={config.min_score_spread}"
            )
        if not stdev_ok:
            parts.append(
                f"stdev={stdev:.1f} < min_score_stdev={config.min_score_stdev}"
            )
        message = f"FAIL: scores appear compressed — {'; '.join(parts)}."

    return CheckResult(name=name, passed=passed,
                       observed=observed, threshold=threshold, message=message)


# ---------------------------------------------------------------------------
# check_extraction_sanity
# Catches a broken extractor and a model returning sentences as skills.
# ---------------------------------------------------------------------------

def check_extraction_sanity(
    facts_list: list[dict[str, Any]],
    config: Config,
) -> CheckResult:
    """Fail if too many listings have empty required_skills, or any listing
    has an implausibly large skill list.
    """
    name = "extraction_sanity"

    if not facts_list:
        return CheckResult(
            name=name,
            passed=True,
            observed="n=0 extractions",
            threshold="n/a",
            message="PASS (trivial): no extractions to check.",
        )

    total = len(facts_list)
    empty_count = sum(
        1 for f in facts_list
        if not f.get("required_skills")
    )
    empty_pct = 100.0 * empty_count / total

    max_skills = max(
        len(f.get("required_skills") or []) for f in facts_list
    )

    empty_ok = empty_pct <= config.max_empty_extraction_pct
    skills_ok = max_skills <= config.max_skills_per_listing
    passed    = empty_ok and skills_ok

    observed  = (
        f"empty={empty_pct:.1f}% ({empty_count}/{total}), "
        f"max_skills_in_one_listing={max_skills}"
    )
    threshold = (
        f"max_empty_pct={config.max_empty_extraction_pct}%, "
        f"max_skills_per_listing={config.max_skills_per_listing}"
    )

    if passed:
        message = (
            f"PASS: {empty_pct:.1f}% empty (limit {config.max_empty_extraction_pct}%) "
            f"and max skills/listing={max_skills} (limit {config.max_skills_per_listing})."
        )
    else:
        parts: list[str] = []
        if not empty_ok:
            parts.append(
                f"{empty_pct:.1f}% empty extractions "
                f"> max_empty_extraction_pct={config.max_empty_extraction_pct}%"
            )
        if not skills_ok:
            parts.append(
                f"a listing has {max_skills} skills "
                f"> max_skills_per_listing={config.max_skills_per_listing} "
                f"(extractor may have returned full sentences)"
            )
        message = f"FAIL: {'; '.join(parts)}."

    return CheckResult(name=name, passed=passed,
                       observed=observed, threshold=threshold, message=message)


# ---------------------------------------------------------------------------
# check_gap_sample_size
# Catches ranking a gap seen in too few listings (a rumour, not a signal).
# ---------------------------------------------------------------------------

def check_gap_sample_size(
    gaps: list[dict[str, Any]],
    config: Config,
) -> CheckResult:
    """Fail if the top-ranked gap was computed from fewer than min_gap_sample listings."""
    name = "gap_sample_size"

    if not gaps:
        return CheckResult(
            name=name,
            passed=True,
            observed="n=0 gaps",
            threshold="n/a",
            message="PASS (trivial): no gaps to check.",
        )

    # Gaps are expected to be sorted by opportunity_cost descending already,
    # but we check the worst-case (minimum sample) among the top entry only.
    top_gap  = gaps[0]
    top_skill  = top_gap.get("skill", "?")
    top_sample = top_gap.get("sample_size", top_gap.get("listings_blocked", 0))

    passed    = top_sample >= config.min_gap_sample
    observed  = f"top_gap='{top_skill}', sample_size={top_sample}"
    threshold = f"min_gap_sample={config.min_gap_sample}"

    if passed:
        message = (
            f"PASS: top gap '{top_skill}' has sample_size={top_sample} "
            f">= min_gap_sample={config.min_gap_sample}."
        )
    else:
        message = (
            f"FAIL: top gap '{top_skill}' computed from only {top_sample} listing(s) "
            f"< min_gap_sample={config.min_gap_sample} — not enough signal to rank."
        )

    return CheckResult(name=name, passed=passed,
                       observed=observed, threshold=threshold, message=message)


# ---------------------------------------------------------------------------
# check_freshness
# Catches stale data that makes recency scores meaningless.
# ---------------------------------------------------------------------------

def check_freshness(
    latest_fetch_at: Optional[str],
    config: Config,
    now: datetime,
) -> CheckResult:
    """Fail if the newest listing is older than max_data_age_days.

    `now` is a parameter — datetime.now() is never called inside this function.
    """
    name = "freshness"

    if not latest_fetch_at:
        return CheckResult(
            name=name,
            passed=False,
            observed="latest_fetch_at=None",
            threshold=f"max_data_age_days={config.max_data_age_days}",
            message=(
                "FAIL: no listings in the database — "
                "run the Fetcher before verifying freshness."
            ),
        )

    fetch_dt = _parse_iso(latest_fetch_at)
    if fetch_dt is None:
        return CheckResult(
            name=name,
            passed=False,
            observed=f"latest_fetch_at={latest_fetch_at!r} (unparseable)",
            threshold=f"max_data_age_days={config.max_data_age_days}",
            message=f"FAIL: could not parse timestamp {latest_fetch_at!r}.",
        )

    # Ensure now is timezone-aware for the comparison
    if now.tzinfo is None:
        now = now.replace(tzinfo=timezone.utc)

    age_days = (now - fetch_dt).total_seconds() / 86400.0
    passed   = age_days <= config.max_data_age_days

    observed  = f"data_age={age_days:.2f} days (newest fetch: {latest_fetch_at[:19]})"
    threshold = f"max_data_age_days={config.max_data_age_days}"

    if passed:
        message = (
            f"PASS: newest data is {age_days:.2f} days old "
            f"<= max_data_age_days={config.max_data_age_days}."
        )
    else:
        message = (
            f"FAIL: newest data is {age_days:.2f} days old "
            f"> max_data_age_days={config.max_data_age_days} — "
            "run the Fetcher to refresh listings."
        )

    return CheckResult(name=name, passed=passed,
                       observed=observed, threshold=threshold, message=message)


# ---------------------------------------------------------------------------
# run_all_checks
# ---------------------------------------------------------------------------

def run_all_checks(
    scores: list[int],
    facts_list: list[dict[str, Any]],
    gaps: list[dict[str, Any]],
    latest_fetch_at: Optional[str],
    config: Config,
    now: datetime,
) -> Verdict:
    """Run every check and return a Verdict.

    Passes only if ALL individual checks pass.
    """
    results = [
        check_score_spread(scores, config),
        check_extraction_sanity(facts_list, config),
        check_gap_sample_size(gaps, config),
        check_freshness(latest_fetch_at, config, now),
    ]

    failed = [r for r in results if not r.passed]
    return Verdict(
        passed=len(failed) == 0,
        failed_checks=failed,
        all_checks=results,
    )
