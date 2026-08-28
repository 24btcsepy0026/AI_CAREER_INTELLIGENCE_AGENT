"""
tests/test_verification.py

Unit tests for edgedash/verification.py.
Every function under test is a pure function — no DB, no network, no clock.
"""

from __future__ import annotations

from dataclasses import replace
from datetime import datetime, timedelta, timezone

import pytest

from edgedash.config import Config
from edgedash.verification import (
    CheckResult,
    Verdict,
    check_extraction_sanity,
    check_freshness,
    check_gap_sample_size,
    check_score_spread,
    run_all_checks,
)

# ---------------------------------------------------------------------------
# Minimal Config fixture — only verification threshold fields matter here.
# All other fields are set to innocuous valid values.
# ---------------------------------------------------------------------------

_BASE_CFG = Config(
    target_role="Data Analyst",
    target_city="Bengaluru",
    keywords=[],
    my_skills=[],
    experience_years=2,
    db_path="edgedash.db",
    min_fit_score=60,
    sources=["arbeitnow"],
    use_mock_fetcher=False,
    llm_provider="gemini",
    llm_model="gemini-flash-lite-latest",
    target_seniority="mid",
    score_batch_size=25,
    weight_skill_match=0.45,
    weight_seniority_fit=0.25,
    weight_location_fit=0.15,
    weight_recency=0.15,
    skill_aliases={},
    fetch_interval_hours=6,
    score_max_seconds=300,
    analyse_max_seconds=120,
    fetch_max_pages=5,
    fetch_max_listings=500,
    # verification thresholds
    min_score_spread=10,
    min_score_stdev=5.0,
    max_empty_extraction_pct=20.0,
    max_skills_per_listing=20,
    min_gap_sample=3,
    max_data_age_days=3,
)

_NOW = datetime(2026, 8, 20, 12, 0, 0, tzinfo=timezone.utc)


def _cfg(**overrides) -> Config:
    return replace(_BASE_CFG, **overrides)


# ---------------------------------------------------------------------------
# check_score_spread
# ---------------------------------------------------------------------------

class TestCheckScoreSpread:

    # --- passing case ---
    def test_pass_healthy_spread(self):
        scores = [20, 35, 50, 65, 80]       # spread=60, stdev≈21
        result = check_score_spread(scores, _BASE_CFG)
        assert result.passed is True
        assert result.name == "score_spread"
        assert "PASS" in result.message

    # --- failing case: spread too small ---
    def test_fail_spread_too_small(self):
        scores = [50, 52, 54, 55, 56]       # spread=6 < 10, stdev≈2.1 < 5
        result = check_score_spread(scores, _BASE_CFG)
        assert result.passed is False
        assert "spread" in result.message.lower()
        assert "FAIL" in result.message

    # --- failing case: stdev too small even if spread is exactly threshold ---
    def test_fail_stdev_too_small(self):
        # spread=10 (exactly at threshold) but all values cluster near ends
        # → stdev can still be below 5 with the right distribution
        cfg = _cfg(min_score_spread=5, min_score_stdev=8.0)
        scores = [50, 51, 51, 52, 55]       # spread=5, stdev≈1.7
        result = check_score_spread(scores, cfg)
        assert result.passed is False
        assert "stdev" in result.message.lower()

    # --- trivial-pass: fewer than 5 scores ---
    def test_trivial_pass_fewer_than_5(self):
        result = check_score_spread([30, 80], _BASE_CFG)
        assert result.passed is True
        assert "trivial" in result.message.lower()
        assert "2" in result.message   # mentions the actual count

    def test_trivial_pass_empty_list(self):
        result = check_score_spread([], _BASE_CFG)
        assert result.passed is True
        assert "trivial" in result.message.lower()

    def test_trivial_pass_exactly_4_scores(self):
        result = check_score_spread([10, 20, 30, 40], _BASE_CFG)
        assert result.passed is True
        assert "trivial" in result.message.lower()

    # --- observed and threshold are populated ---
    def test_result_fields_populated(self):
        scores = [10, 20, 30, 40, 50]
        result = check_score_spread(scores, _BASE_CFG)
        assert result.observed != ""
        assert result.threshold != ""


# ---------------------------------------------------------------------------
# check_extraction_sanity
# ---------------------------------------------------------------------------

class TestCheckExtractionSanity:

    def _facts(self, required_skills_lists: list[list[str]]) -> list[dict]:
        return [{"required_skills": s} for s in required_skills_lists]

    # --- passing case ---
    def test_pass_healthy_extractions(self):
        facts = self._facts([
            ["python", "sql"],
            ["java", "spring", "docker"],
            ["typescript", "react"],
            ["aws", "terraform"],
            ["python", "pandas"],
        ])
        result = check_extraction_sanity(facts, _BASE_CFG)
        assert result.passed is True
        assert "PASS" in result.message

    # --- failing case: too many empty extractions ---
    def test_fail_too_many_empty(self):
        # 3 out of 5 empty = 60% > 20%
        facts = self._facts([[], [], [], ["python"], ["sql"]])
        result = check_extraction_sanity(facts, _BASE_CFG)
        assert result.passed is False
        assert "empty" in result.message.lower()
        assert "FAIL" in result.message

    # --- failing case: a listing with too many skills ---
    def test_fail_too_many_skills_in_one_listing(self):
        # 21 skills in one listing > max 20
        long_skills = [f"skill_{i}" for i in range(21)]
        facts = self._facts([long_skills, ["python"], ["sql"]])
        result = check_extraction_sanity(facts, _BASE_CFG)
        assert result.passed is False
        assert "21" in result.message
        assert "FAIL" in result.message

    # --- exact boundary: 20 skills is allowed ---
    def test_pass_exactly_at_max_skills(self):
        skills_20 = [f"skill_{i}" for i in range(20)]
        facts = self._facts([skills_20, ["python"]])
        result = check_extraction_sanity(facts, _BASE_CFG)
        assert result.passed is True

    # --- empty extraction list: trivial pass ---
    def test_trivial_pass_no_extractions(self):
        result = check_extraction_sanity([], _BASE_CFG)
        assert result.passed is True
        assert "trivial" in result.message.lower()

    # --- both failures at once ---
    def test_fail_both_conditions(self):
        long_skills = [f"skill_{i}" for i in range(25)]
        facts = self._facts([[], [], [], long_skills])  # 75% empty AND 25 skills
        result = check_extraction_sanity(facts, _BASE_CFG)
        assert result.passed is False

    # --- exactly at empty pct threshold (should pass) ---
    def test_pass_exactly_at_empty_pct_boundary(self):
        # 1 out of 5 empty = 20.0% == threshold → passes (<=)
        facts = self._facts([[], ["python"], ["sql"], ["java"], ["go"]])
        result = check_extraction_sanity(facts, _BASE_CFG)
        assert result.passed is True

    # --- just over empty pct threshold (should fail) ---
    def test_fail_just_over_empty_pct_boundary(self):
        # 2 out of 5 empty = 40% > 20%
        facts = self._facts([[], [], ["python"], ["sql"], ["java"]])
        result = check_extraction_sanity(facts, _BASE_CFG)
        assert result.passed is False


# ---------------------------------------------------------------------------
# check_gap_sample_size
# ---------------------------------------------------------------------------

class TestCheckGapSampleSize:

    def _gap(self, skill: str, sample_size: int) -> dict:
        return {
            "skill": skill,
            "sample_size": sample_size,
            "opportunity_cost": sample_size * 0.5,
            "listings_blocked": sample_size,
        }

    # --- passing case ---
    def test_pass_sufficient_sample(self):
        gaps = [self._gap("typescript", 10), self._gap("aws", 5)]
        result = check_gap_sample_size(gaps, _BASE_CFG)
        assert result.passed is True
        assert "PASS" in result.message

    # --- failing case: top gap has too few listings ---
    def test_fail_top_gap_too_few(self):
        gaps = [self._gap("kubernetes", 2), self._gap("terraform", 8)]
        result = check_gap_sample_size(gaps, _BASE_CFG)
        assert result.passed is False
        assert "kubernetes" in result.message
        assert "2" in result.message
        assert "FAIL" in result.message

    # --- exact boundary: exactly min_gap_sample (should pass) ---
    def test_pass_exactly_at_min_sample(self):
        gaps = [self._gap("docker", 3)]
        result = check_gap_sample_size(gaps, _BASE_CFG)
        assert result.passed is True

    # --- one below boundary: should fail ---
    def test_fail_one_below_min_sample(self):
        gaps = [self._gap("docker", 2)]
        result = check_gap_sample_size(gaps, _BASE_CFG)
        assert result.passed is False

    # --- trivial pass: no gaps ---
    def test_trivial_pass_no_gaps(self):
        result = check_gap_sample_size([], _BASE_CFG)
        assert result.passed is True
        assert "trivial" in result.message.lower()

    # --- uses listings_blocked as fallback when sample_size absent ---
    def test_fallback_to_listings_blocked(self):
        gap_no_sample = {"skill": "go", "listings_blocked": 2, "opportunity_cost": 1.0}
        result = check_gap_sample_size([gap_no_sample], _BASE_CFG)
        assert result.passed is False   # 2 < 3


# ---------------------------------------------------------------------------
# check_freshness
# ---------------------------------------------------------------------------

class TestCheckFreshness:

    def _ts(self, days_ago: float) -> str:
        dt = _NOW - timedelta(days=days_ago)
        return dt.isoformat()

    # --- passing case ---
    def test_pass_fresh_data(self):
        result = check_freshness(self._ts(1.0), _BASE_CFG, _NOW)
        assert result.passed is True
        assert "PASS" in result.message

    # --- passing case: exactly at boundary ---
    def test_pass_exactly_at_max_age(self):
        result = check_freshness(self._ts(3.0), _BASE_CFG, _NOW)
        assert result.passed is True

    # --- failing case: data too old ---
    def test_fail_stale_data(self):
        result = check_freshness(self._ts(5.0), _BASE_CFG, _NOW)
        assert result.passed is False
        assert "FAIL" in result.message
        assert "5.0" in result.message or "5.00" in result.message

    # --- failing case: no fetch at all ---
    def test_fail_none_timestamp(self):
        result = check_freshness(None, _BASE_CFG, _NOW)
        assert result.passed is False
        assert "FAIL" in result.message

    # --- failing case: unparseable timestamp ---
    def test_fail_unparseable_timestamp(self):
        result = check_freshness("not-a-date", _BASE_CFG, _NOW)
        assert result.passed is False
        assert "FAIL" in result.message

    # --- now is a parameter, not datetime.now() ---
    def test_now_is_a_parameter(self):
        """Passing a different 'now' changes the result — proves no internal clock."""
        ts = self._ts(2.0)   # 2 days before _NOW
        future_now = _NOW + timedelta(days=10)   # 12 days after the timestamp
        result_fresh  = check_freshness(ts, _BASE_CFG, _NOW)
        result_stale  = check_freshness(ts, _BASE_CFG, future_now)
        assert result_fresh.passed is True
        assert result_stale.passed is False

    # --- config threshold is respected ---
    def test_custom_threshold(self):
        cfg = _cfg(max_data_age_days=1)
        result = check_freshness(self._ts(2.0), cfg, _NOW)
        assert result.passed is False


# ---------------------------------------------------------------------------
# run_all_checks
# ---------------------------------------------------------------------------

class TestRunAllChecks:

    def _good_inputs(self):
        scores = [20, 40, 60, 70, 85]
        facts = [
            {"required_skills": ["python", "sql"]},
            {"required_skills": ["java", "spring"]},
            {"required_skills": ["aws", "terraform"]},
        ]
        gaps = [
            {"skill": "typescript", "sample_size": 10, "opportunity_cost": 5.0, "listings_blocked": 10},
            {"skill": "kubernetes", "sample_size": 7,  "opportunity_cost": 3.5, "listings_blocked": 7},
        ]
        latest_fetch_at = (_NOW - timedelta(hours=6)).isoformat()
        return scores, facts, gaps, latest_fetch_at

    def test_all_pass_healthy_data(self):
        scores, facts, gaps, ts = self._good_inputs()
        verdict = run_all_checks(scores, facts, gaps, ts, _BASE_CFG, _NOW)
        assert verdict.passed is True
        assert verdict.failed_checks == []
        assert len(verdict.all_checks) == 4
        assert "All 4 checks passed" in verdict.summary

    def test_one_failure_marks_verdict_failed(self):
        scores, facts, gaps, ts = self._good_inputs()
        stale_ts = (_NOW - timedelta(days=10)).isoformat()
        verdict = run_all_checks(scores, facts, gaps, stale_ts, _BASE_CFG, _NOW)
        assert verdict.passed is False
        assert any(c.name == "freshness" for c in verdict.failed_checks)

    def test_multiple_failures_all_reported(self):
        scores = [50, 51, 52, 53, 54]      # compressed → score_spread fails
        facts = [{"required_skills": []}] * 5  # all empty → extraction_sanity fails
        gaps = [{"skill": "go", "sample_size": 1, "opportunity_cost": 0.5, "listings_blocked": 1}]
        stale_ts = (_NOW - timedelta(days=10)).isoformat()
        verdict = run_all_checks(scores, facts, gaps, stale_ts, _BASE_CFG, _NOW)
        assert verdict.passed is False
        failed_names = {c.name for c in verdict.failed_checks}
        assert "score_spread" in failed_names
        assert "extraction_sanity" in failed_names
        assert "freshness" in failed_names

    def test_summary_names_failed_checks(self):
        scores, facts, gaps, _ = self._good_inputs()
        stale_ts = (_NOW - timedelta(days=10)).isoformat()
        verdict = run_all_checks(scores, facts, gaps, stale_ts, _BASE_CFG, _NOW)
        assert "freshness" in verdict.summary

    def test_all_checks_always_present_even_when_passed(self):
        scores, facts, gaps, ts = self._good_inputs()
        verdict = run_all_checks(scores, facts, gaps, ts, _BASE_CFG, _NOW)
        names = {c.name for c in verdict.all_checks}
        assert names == {"score_spread", "extraction_sanity", "gap_sample_size", "freshness"}
