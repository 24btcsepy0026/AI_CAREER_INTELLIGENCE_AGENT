"""
tests/test_planning.py

Unit tests for build_plan().
Pure function — no I/O, no DB, no mocks needed.
"""

from __future__ import annotations

from dataclasses import replace

import pytest

from edgedash.config import Config
from edgedash.planning import build_plan, Plan, Task
from edgedash.state import SystemState

# ---------------------------------------------------------------------------
# Minimal Config fixture — only fields planning.py touches
# ---------------------------------------------------------------------------

BASE_CONFIG = Config(
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
    min_score_spread=10,
    min_score_stdev=5.0,
    max_empty_extraction_pct=20.0,
    max_skills_per_listing=20,
    min_gap_sample=3,
    max_data_age_days=3,
)

# ---------------------------------------------------------------------------
# SystemState factories
# ---------------------------------------------------------------------------

def _state(**overrides) -> SystemState:
    base = SystemState(
        last_fetch_at="2026-08-15T00:00:00+00:00",
        hours_since_fetch=3.0,        # < 6h → fetch skip by default
        unscored_count=0,
        last_scored_at="2026-08-15T01:00:00+00:00",
        gaps_computed_at="2026-08-15T01:30:00+00:00",
        gaps_stale=False,
        last_cycle_verdict="ok",
        last_cycle_at="2026-08-15T01:30:00+00:00",
    )
    return replace(base, **overrides)


def _task(plan: Plan, name: str) -> Task:
    for t in plan.tasks:
        if t.agent_name == name:
            return t
    raise KeyError(f"{name} not in plan")


# ---------------------------------------------------------------------------
# Scenario 1: everything stale — all three agents run
# ---------------------------------------------------------------------------

class TestAllStale:
    def setup_method(self):
        state = _state(
            hours_since_fetch=8.0,    # >= 6 → fetch due
            unscored_count=42,        # > 0  → score due
            gaps_stale=True,          # → analyse due
        )
        self.plan = build_plan(state, BASE_CONFIG)

    def test_three_tasks_total(self):
        assert len(self.plan.tasks) == 3

    def test_all_run(self):
        assert len(self.plan.agents_to_run()) == 3

    def test_none_skipped(self):
        assert len(self.plan.agents_skipped()) == 0

    def test_fetcher_runs(self):
        assert not _task(self.plan, "Fetcher").skipped

    def test_scorer_runs(self):
        assert not _task(self.plan, "Scorer").skipped

    def test_gap_analyzer_runs(self):
        assert not _task(self.plan, "GapAnalyzer").skipped

    def test_fetcher_reason_names_state(self):
        reason = _task(self.plan, "Fetcher").reason
        assert "hours_since_fetch" in reason

    def test_scorer_reason_names_count(self):
        reason = _task(self.plan, "Scorer").reason
        assert "42" in reason

    def test_fetcher_stop_conditions_set(self):
        sc = _task(self.plan, "Fetcher").stop_conditions
        assert sc.max_pages == BASE_CONFIG.fetch_max_pages
        assert sc.max_listings == BASE_CONFIG.fetch_max_listings

    def test_scorer_stop_conditions_set(self):
        sc = _task(self.plan, "Scorer").stop_conditions
        assert sc.max_items == BASE_CONFIG.score_batch_size
        assert sc.max_seconds == BASE_CONFIG.score_max_seconds

    def test_analyser_stop_conditions_set(self):
        sc = _task(self.plan, "GapAnalyzer").stop_conditions
        assert sc.max_seconds == BASE_CONFIG.analyse_max_seconds


# ---------------------------------------------------------------------------
# Scenario 2: nothing to do — all three skip
# ---------------------------------------------------------------------------

class TestNothingToDo:
    def setup_method(self):
        state = _state(
            hours_since_fetch=1.0,   # < 6 → fetch skip
            unscored_count=0,        # → score skip
            gaps_stale=False,        # → analyse skip
            gaps_computed_at="2026-08-15T01:30:00+00:00",
        )
        self.plan = build_plan(state, BASE_CONFIG)

    def test_three_tasks_total(self):
        assert len(self.plan.tasks) == 3

    def test_all_skipped(self):
        assert len(self.plan.agents_skipped()) == 3

    def test_none_run(self):
        assert len(self.plan.agents_to_run()) == 0

    def test_all_reasons_contain_skipped(self):
        for t in self.plan.tasks:
            assert "skipped" in t.reason.lower()

    def test_render_contains_skip(self):
        rendered = self.plan.render()
        assert rendered.count("SKIP") == 3
        assert "RUN" not in rendered

    def test_fetcher_reason_names_threshold(self):
        reason = _task(self.plan, "Fetcher").reason
        assert "hours_since_fetch" in reason
        assert "threshold" in reason


# ---------------------------------------------------------------------------
# Scenario 3: only unscored listings — only Scorer runs
# ---------------------------------------------------------------------------

class TestOnlyUnscored:
    def setup_method(self):
        state = _state(
            hours_since_fetch=1.0,   # fetch skip
            unscored_count=10,       # score runs
            gaps_stale=False,        # analyse skip
        )
        self.plan = build_plan(state, BASE_CONFIG)

    def test_only_scorer_runs(self):
        running = [t.agent_name for t in self.plan.agents_to_run()]
        assert running == ["Scorer"]

    def test_fetcher_skipped(self):
        assert _task(self.plan, "Fetcher").skipped

    def test_gap_skipped(self):
        assert _task(self.plan, "GapAnalyzer").skipped

    def test_scorer_goal_mentions_batch_size(self):
        goal = _task(self.plan, "Scorer").goal
        assert str(BASE_CONFIG.score_batch_size) in goal


# ---------------------------------------------------------------------------
# Scenario 4: gaps stale, nothing unscored — only GapAnalyzer runs
# ---------------------------------------------------------------------------

class TestGapsStaleNoUnscored:
    def setup_method(self):
        state = _state(
            hours_since_fetch=1.0,   # fetch skip
            unscored_count=0,        # score skip
            gaps_stale=True,         # analyse runs
        )
        self.plan = build_plan(state, BASE_CONFIG)

    def test_only_gap_analyzer_runs(self):
        running = [t.agent_name for t in self.plan.agents_to_run()]
        assert running == ["GapAnalyzer"]

    def test_fetcher_skipped(self):
        assert _task(self.plan, "Fetcher").skipped

    def test_scorer_skipped(self):
        assert _task(self.plan, "Scorer").skipped

    def test_gap_reason_mentions_stale(self):
        reason = _task(self.plan, "GapAnalyzer").reason
        assert "stale" in reason.lower()


# ---------------------------------------------------------------------------
# Scenario 5: never fetched before — Fetcher runs regardless of hours
# ---------------------------------------------------------------------------

class TestNeverFetched:
    def setup_method(self):
        state = _state(
            last_fetch_at=None,
            hours_since_fetch=None,
            unscored_count=0,
            gaps_stale=False,
        )
        self.plan = build_plan(state, BASE_CONFIG)

    def test_fetcher_runs(self):
        assert not _task(self.plan, "Fetcher").skipped

    def test_reason_mentions_never(self):
        reason = _task(self.plan, "Fetcher").reason
        assert "never" in reason.lower()


# ---------------------------------------------------------------------------
# Scenario 6: gaps never computed — GapAnalyzer runs
# ---------------------------------------------------------------------------

class TestGapsNeverComputed:
    def setup_method(self):
        state = _state(
            hours_since_fetch=1.0,
            unscored_count=0,
            gaps_computed_at=None,
            gaps_stale=False,        # state.py sets this to False when no scores
        )
        self.plan = build_plan(state, BASE_CONFIG)

    def test_gap_analyzer_runs(self):
        assert not _task(self.plan, "GapAnalyzer").skipped

    def test_reason_mentions_none(self):
        reason = _task(self.plan, "GapAnalyzer").reason
        assert "none" in reason.lower()


# ---------------------------------------------------------------------------
# render() smoke test
# ---------------------------------------------------------------------------

class TestRender:
    def test_render_nothing_to_do(self):
        state = _state(hours_since_fetch=1.0, unscored_count=0,
                       gaps_stale=False, gaps_computed_at="2026-08-15T01:30:00+00:00")
        plan = build_plan(state, BASE_CONFIG)
        rendered = plan.render()
        lines = rendered.strip().splitlines()
        assert len(lines) == 3
        assert all("SKIP" in l for l in lines)

    def test_render_all_run_contains_stop_conditions(self):
        state = _state(hours_since_fetch=8.0, unscored_count=5, gaps_stale=True)
        plan = build_plan(state, BASE_CONFIG)
        rendered = plan.render()
        assert "max_pages" in rendered
        assert "max_items" in rendered
        assert "max_seconds" in rendered
