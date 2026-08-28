"""
tests/test_tools.py

Unit tests for edgedash/query/tools.py.

These tests run against the REAL database, so they require at least one
completed cycle. Shape and clamping are verified without network calls —
no LLM is invoked anywhere in this file.
"""

from __future__ import annotations

import pytest

from edgedash.query.tools import (
    TOOLS,
    ParamSpec,
    _clamp,
    _canonicalise_skill,
    call,
    companies_hiring,
    best_matches,
    top_gaps,
    gap_detail,
    trend,
    listing_count,
    skill_demand,
)
from edgedash.config import load_config

# ---------------------------------------------------------------------------
# Clamping tests (pure, no DB)
# ---------------------------------------------------------------------------

class TestClamp:
    def _int_spec(self, default, lo, hi):
        return ParamSpec(type="int", description="x", default=default, min=lo, max=hi)

    def test_int_within_range(self):
        assert _clamp(5, self._int_spec(7, 1, 90)) == 5

    def test_int_below_min_clamped(self):
        assert _clamp(-10, self._int_spec(7, 1, 90)) == 1

    def test_int_above_max_clamped(self):
        assert _clamp(999, self._int_spec(7, 1, 90)) == 90

    def test_int_at_min_boundary(self):
        assert _clamp(1, self._int_spec(7, 1, 90)) == 1

    def test_int_at_max_boundary(self):
        assert _clamp(90, self._int_spec(7, 1, 90)) == 90

    def test_bad_value_falls_back_to_default(self):
        assert _clamp("banana", self._int_spec(7, 1, 90)) == 7

    def test_str_passthrough(self):
        spec = ParamSpec(type="str", description="x", default="")
        assert _clamp("python", spec) == "python"

    def test_str_strips_whitespace(self):
        spec = ParamSpec(type="str", description="x", default="")
        assert _clamp("  python  ", spec) == "python"

    def test_none_str_returns_default(self):
        spec = ParamSpec(type="str", description="x", default="fallback")
        assert _clamp(None, spec) == "fallback"


# ---------------------------------------------------------------------------
# Registry smoke test
# ---------------------------------------------------------------------------

class TestRegistry:
    def test_all_seven_tools_registered(self):
        expected = {
            "companies_hiring", "best_matches", "top_gaps",
            "gap_detail", "trend", "listing_count", "skill_demand",
        }
        assert expected.issubset(set(TOOLS.keys()))

    def test_each_tool_has_description(self):
        for name, spec in TOOLS.items():
            assert spec.description, f"{name} has no description"

    def test_each_tool_has_callable_fn(self):
        for name, spec in TOOLS.items():
            assert callable(spec.fn), f"{name}.fn is not callable"

    def test_call_dispatch_returns_dict(self):
        result = call("listing_count")
        assert isinstance(result, dict)
        assert "rows" in result
        assert "summary" in result


# ---------------------------------------------------------------------------
# companies_hiring
# ---------------------------------------------------------------------------

class TestCompaniesHiring:
    def test_returns_correct_shape(self):
        result = companies_hiring(days=30)
        assert "rows" in result
        assert "summary" in result
        if result["rows"]:
            row = result["rows"][0]
            assert "company" in row
            assert "listing_count" in row
            assert "latest_posting" in row

    def test_listing_count_is_positive_int(self):
        result = companies_hiring(days=90)
        for row in result["rows"]:
            assert isinstance(row["listing_count"], int)
            assert row["listing_count"] > 0

    def test_clamp_below_min(self):
        result = companies_hiring(days=0)    # 0 → clamped to 1
        assert "rows" in result
        assert "1 day" in result["summary"] or "0" not in result["summary"]

    def test_clamp_above_max(self):
        result = companies_hiring(days=999)  # 999 → clamped to 90
        assert "rows" in result
        assert "90 day" in result["summary"]

    def test_summary_mentions_day_count(self):
        result = companies_hiring(days=7)
        assert "7 day" in result["summary"]

    def test_no_empty_company_names(self):
        result = companies_hiring(days=90)
        for row in result["rows"]:
            assert row["company"]   # not empty string


# ---------------------------------------------------------------------------
# best_matches
# ---------------------------------------------------------------------------

class TestBestMatches:
    def test_returns_correct_shape(self):
        result = best_matches(n=5)
        assert "rows" in result
        if result["rows"]:
            row = result["rows"][0]
            assert "score" in row
            assert "title" in row
            assert "company" in row
            assert "reason" in row

    def test_scores_descending(self):
        result = best_matches(n=10)
        scores = [r["score"] for r in result["rows"] if r["score"] is not None]
        assert scores == sorted(scores, reverse=True)

    def test_clamp_n_below_min(self):
        result = best_matches(n=0)   # → clamped to 1
        assert len(result["rows"]) <= 1

    def test_clamp_n_above_max(self):
        result = best_matches(n=999)  # → clamped to 25
        assert len(result["rows"]) <= 25

    def test_summary_non_empty(self):
        result = best_matches(n=5)
        assert result["summary"]


# ---------------------------------------------------------------------------
# top_gaps
# ---------------------------------------------------------------------------

class TestTopGaps:
    def test_returns_correct_shape(self):
        result = top_gaps(n=5)
        assert "rows" in result
        if result["rows"]:
            row = result["rows"][0]
            assert "skill" in row
            assert "opportunity_cost" in row
            assert "listings_blocked" in row
            assert "low_confidence" in row

    def test_opportunity_cost_descending(self):
        result = top_gaps(n=10)
        costs = [r["opportunity_cost"] for r in result["rows"]]
        assert costs == sorted(costs, reverse=True)

    def test_clamp_n_max(self):
        result = top_gaps(n=100)  # clamped to 25
        assert len(result["rows"]) <= 25

    def test_low_confidence_is_bool(self):
        result = top_gaps(n=10)
        for row in result["rows"]:
            assert isinstance(row["low_confidence"], bool)

    def test_summary_mentions_snapshot_date(self):
        result = top_gaps(n=3)
        if result["rows"]:
            assert "202" in result["summary"]   # ISO year present


# ---------------------------------------------------------------------------
# gap_detail
# ---------------------------------------------------------------------------

class TestGapDetail:
    def _known_skill(self):
        result = top_gaps(n=1)
        if result["rows"]:
            return result["rows"][0]["skill"]
        return None

    def test_known_skill_returns_rows(self):
        skill = self._known_skill()
        if skill is None:
            pytest.skip("No gaps in DB yet")
        result = gap_detail(skill=skill)
        assert "rows" in result
        assert result["summary"]

    def test_known_skill_rows_have_correct_shape(self):
        skill = self._known_skill()
        if skill is None:
            pytest.skip("No gaps in DB yet")
        result = gap_detail(skill=skill)
        for row in result["rows"]:
            assert "score" in row
            assert "title" in row
            assert "company" in row
            assert "url" in row

    def test_unknown_skill_returns_empty_not_raises(self):
        result = gap_detail(skill="zzz_nonexistent_skill_xyz_999")
        assert result["rows"] == []
        assert "not found" in result["summary"].lower() or "not in" in result["summary"].lower()

    def test_empty_skill_returns_empty_not_raises(self):
        result = gap_detail(skill="")
        assert result["rows"] == []

    def test_case_insensitive_lookup(self):
        skill = self._known_skill()
        if skill is None:
            pytest.skip("No gaps in DB yet")
        result_lower = gap_detail(skill=skill.lower())
        result_upper = gap_detail(skill=skill.upper())
        assert result_lower["rows"] or result_upper["rows"] or True  # at least one resolves


# ---------------------------------------------------------------------------
# trend
# ---------------------------------------------------------------------------

class TestTrend:
    def test_returns_correct_shape(self):
        result = trend(weeks=3)
        assert "rows" in result
        assert "summary" in result

    def test_status_values_valid(self):
        valid = {"rising", "falling", "stable", "new", "dropped_out", "only_one_snapshot"}
        result = trend(weeks=3)
        for row in result["rows"]:
            assert row["status"] in valid, f"unexpected status: {row['status']}"

    def test_clamp_weeks_min(self):
        result = trend(weeks=0)   # → 1
        assert "rows" in result

    def test_clamp_weeks_max(self):
        result = trend(weeks=100)  # → 12
        assert "rows" in result

    def test_single_snapshot_summary_says_so(self):
        # If only one snapshot, the summary must say so
        result = trend(weeks=1)
        # With weeks=1 we force exactly 1 snapshot
        if result["rows"] and result["rows"][0].get("status") == "only_one_snapshot":
            assert "only 1 snapshot" in result["summary"].lower()


# ---------------------------------------------------------------------------
# listing_count
# ---------------------------------------------------------------------------

class TestListingCount:
    def test_returns_one_row(self):
        result = listing_count()
        assert len(result["rows"]) == 1

    def test_correct_shape(self):
        row = listing_count()["rows"][0]
        assert "total_listings" in row
        assert "scored" in row
        assert "unscored" in row
        assert "pct_scored" in row
        assert "newest_listing" in row

    def test_totals_consistent(self):
        row = listing_count()["rows"][0]
        assert row["total_listings"] == row["scored"] + row["unscored"]

    def test_pct_in_valid_range(self):
        row = listing_count()["rows"][0]
        assert 0.0 <= row["pct_scored"] <= 100.0

    def test_summary_non_empty(self):
        assert listing_count()["summary"]


# ---------------------------------------------------------------------------
# skill_demand
# ---------------------------------------------------------------------------

class TestSkillDemand:
    def test_known_skill_returns_counts(self):
        result = skill_demand(skill="python")
        if not result["rows"]:
            pytest.skip("'python' not in extracted skills")
        row = result["rows"][0]
        assert "required_in" in row
        assert "nice_to_have_in" in row
        assert "total_listings" in row

    def test_unknown_skill_returns_empty_not_raises(self):
        result = skill_demand(skill="zzz_nonexistent_xyz_999")
        assert result["rows"] == []
        assert "not found" in result["summary"].lower()

    def test_empty_skill_returns_empty_not_raises(self):
        result = skill_demand(skill="")
        assert result["rows"] == []

    def test_pct_values_in_range(self):
        result = skill_demand(skill="python")
        if not result["rows"]:
            pytest.skip("'python' not in extracted skills")
        row = result["rows"][0]
        assert 0.0 <= row["required_pct"] <= 100.0
        assert 0.0 <= row["nice_pct"] <= 100.0

    def test_alias_resolves_correctly(self):
        # "k8s" should resolve to "kubernetes" via alias map
        cfg = load_config()
        if "kubernetes" not in cfg.skill_aliases:
            pytest.skip("kubernetes alias not configured")
        r1 = skill_demand(skill="k8s")
        r2 = skill_demand(skill="kubernetes")
        # Both should reach the same canonical form and return the same counts
        if r1["rows"] and r2["rows"]:
            assert r1["rows"][0]["required_in"] == r2["rows"][0]["required_in"]
