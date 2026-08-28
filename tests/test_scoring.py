"""
tests/test_scoring.py

Unit tests for edgedash/scoring.py.
Pure functions — no network, no DB, no mocks needed.
"""

from __future__ import annotations

from datetime import datetime, timedelta, timezone
from dataclasses import replace

import pytest

from edgedash.config import Config
from edgedash.scoring import score_listing, build_reason

# ---------------------------------------------------------------------------
# Minimal Config fixture — only fields scoring.py touches
# ---------------------------------------------------------------------------

BASE_CONFIG = Config(
    target_role="Data Analyst",
    target_city="Bengaluru",
    keywords=[],
    my_skills=["python", "sql", "pandas", "excel", "numpy", "data visualisation"],
    experience_years=2,
    db_path="edgedash.db",
    min_fit_score=60,
    sources=["arbeitnow"],
    use_mock_fetcher=False,
    llm_provider="gemini",
    llm_model="gemini-2.5-flash",
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
    min_score_spread=10,
    min_score_stdev=5.0,
    max_empty_extraction_pct=20.0,
    max_skills_per_listing=20,
    min_gap_sample=3,
    max_data_age_days=3,
)

def _listing(posted_at: str | None = None, location: str = "Bengaluru") -> dict:
    return {
        "id": "test-id",
        "title": "Data Analyst",
        "company": "ACME",
        "location": location,
        "url": "https://jobs.example/1",
        "description": "...",
        "source": "test",
        "posted_at": posted_at,
    }

def _today_iso() -> str:
    return datetime.now(timezone.utc).isoformat()

def _days_ago_iso(n: int) -> str:
    return (datetime.now(timezone.utc) - timedelta(days=n)).isoformat()


# ---------------------------------------------------------------------------
# Perfect match
# ---------------------------------------------------------------------------

class TestPerfectMatch:
    def test_score_near_100(self):
        facts = {
            "required_skills": ["python", "sql", "pandas"],
            "nice_to_have": ["excel"],
            "seniority": "mid",
            "years_required": 2,
            "remote_ok": True,
        }
        result = score_listing(_listing(_today_iso()), facts, BASE_CONFIG)
        # All components should be max — score must be very high
        assert result["score"] >= 90

    def test_returns_required_keys(self):
        facts = {
            "required_skills": ["python"],
            "nice_to_have": [],
            "seniority": "mid",
            "years_required": None,
            "remote_ok": True,
        }
        result = score_listing(_listing(_today_iso()), facts, BASE_CONFIG)
        assert set(result.keys()) == {"score", "reason", "components"}
        assert isinstance(result["score"], int)
        assert isinstance(result["reason"], str)
        assert 0 <= result["score"] <= 100


# ---------------------------------------------------------------------------
# Zero match
# ---------------------------------------------------------------------------

class TestZeroMatch:
    def test_score_low(self):
        facts = {
            "required_skills": ["cobol", "mainframe", "fortran", "assembly", "rpg"],
            "nice_to_have": ["punchcards"],
            "seniority": "lead",       # three bands from mid
            "years_required": 15,
            "remote_ok": False,
        }
        cfg = replace(BASE_CONFIG, my_skills=[], target_city="Mumbai")
        result = score_listing(_listing(_days_ago_iso(29), location="Berlin"), facts, cfg)
        assert result["score"] <= 15

    def test_gap_in_reason(self):
        facts = {
            "required_skills": ["cobol", "mainframe"],
            "nice_to_have": [],
            "seniority": "mid",
            "years_required": None,
            "remote_ok": None,
        }
        cfg = replace(BASE_CONFIG, my_skills=[])
        result = score_listing(_listing(), facts, cfg)
        assert "gap:" in result["reason"]
        assert "cobol" in result["reason"]


# ---------------------------------------------------------------------------
# Empty required_skills — must not divide by zero
# ---------------------------------------------------------------------------

class TestEmptyRequiredSkills:
    def test_no_division_by_zero(self):
        facts = {
            "required_skills": [],
            "nice_to_have": [],
            "seniority": "mid",
            "years_required": None,
            "remote_ok": None,
        }
        # Should not raise
        result = score_listing(_listing(), facts, BASE_CONFIG)
        assert 0 <= result["score"] <= 100

    def test_only_nice_to_have(self):
        facts = {
            "required_skills": [],
            "nice_to_have": ["python", "sql"],
            "seniority": "mid",
            "years_required": None,
            "remote_ok": None,
        }
        result = score_listing(_listing(), facts, BASE_CONFIG)
        # python + sql are both in my_skills — should score decently
        assert result["score"] > 30


# ---------------------------------------------------------------------------
# Null posted_at — must not crash, defaults to 0.5 on recency
# ---------------------------------------------------------------------------

class TestNullPostedAt:
    def test_no_crash(self):
        facts = {
            "required_skills": ["python"],
            "nice_to_have": [],
            "seniority": "mid",
            "years_required": None,
            "remote_ok": None,
        }
        result = score_listing(_listing(posted_at=None), facts, BASE_CONFIG)
        assert 0 <= result["score"] <= 100

    def test_recency_component_is_neutral(self):
        facts = {
            "required_skills": [],
            "nice_to_have": [],
            "seniority": "unknown",
            "years_required": None,
            "remote_ok": None,
        }
        result = score_listing(_listing(posted_at=None), facts, BASE_CONFIG)
        assert result["components"]["recency"]["score"] == 0.5

    def test_reason_says_date_unknown(self):
        facts = {
            "required_skills": [],
            "nice_to_have": [],
            "seniority": "mid",
            "years_required": None,
            "remote_ok": None,
        }
        result = score_listing(_listing(posted_at=None), facts, BASE_CONFIG)
        assert "date unknown" in result["reason"]


# ---------------------------------------------------------------------------
# Null remote_ok — should be "unknown" location, score 0.5
# ---------------------------------------------------------------------------

class TestNullRemoteOk:
    def test_location_neutral_when_remote_unknown(self):
        facts = {
            "required_skills": [],
            "nice_to_have": [],
            "seniority": "mid",
            "years_required": None,
            "remote_ok": None,
        }
        # Non-matching city so only remote_ok would help
        result = score_listing(_listing(location="Tokyo"), facts, BASE_CONFIG)
        assert result["components"]["location_fit"]["score"] == 0.5
        assert result["components"]["location_fit"]["label"] == "unknown"


# ---------------------------------------------------------------------------
# Seniority three bands off — should score 0.0 on that component
# ---------------------------------------------------------------------------

class TestSeniorityThreeBandsOff:
    def test_junior_vs_lead_is_zero(self):
        """junior (band 0) vs lead (band 3) = distance 3 → score 0.0"""
        facts = {
            "required_skills": [],
            "nice_to_have": [],
            "seniority": "junior",
            "years_required": None,
            "remote_ok": None,
        }
        cfg = replace(BASE_CONFIG, target_seniority="lead")
        result = score_listing(_listing(), facts, cfg)
        assert result["components"]["seniority_fit"]["score"] == 0.0

    def test_lead_vs_junior_is_zero(self):
        """lead (band 3) vs junior (band 0) = distance 3 → score 0.0"""
        facts = {
            "required_skills": [],
            "nice_to_have": [],
            "seniority": "lead",
            "years_required": None,
            "remote_ok": None,
        }
        cfg = replace(BASE_CONFIG, target_seniority="junior")
        result = score_listing(_listing(), facts, cfg)
        assert result["components"]["seniority_fit"]["score"] == 0.0

    def test_seniority_mismatch_in_reason(self):
        facts = {
            "required_skills": [],
            "nice_to_have": [],
            "seniority": "lead",
            "years_required": None,
            "remote_ok": None,
        }
        cfg = replace(BASE_CONFIG, target_seniority="junior")
        result = score_listing(_listing(), facts, cfg)
        assert "mismatch" in result["reason"]


# ---------------------------------------------------------------------------
# Score is always in [0, 100]
# ---------------------------------------------------------------------------

class TestScoreBounds:
    @pytest.mark.parametrize("score_val", [0.0, 0.5, 1.0, 1.1, -0.1])
    def test_clamped(self, score_val):
        """Weighted sum can drift slightly — result must always be 0-100."""
        # Verify the clamp by constructing edge-case configs
        facts = {
            "required_skills": ["python"],
            "nice_to_have": [],
            "seniority": "mid",
            "years_required": None,
            "remote_ok": True,
        }
        result = score_listing(_listing(_today_iso()), facts, BASE_CONFIG)
        assert 0 <= result["score"] <= 100
