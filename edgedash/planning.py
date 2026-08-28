"""
edgedash/planning.py

Pure planning logic.  No I/O.  No LLM.  No datetime.now().
Takes (SystemState, Config), returns a Plan.

`build_plan` is the function to read first.

Decision rules
--------------
  fetch    run if hours_since_fetch >= config.fetch_interval_hours,
                or never fetched before
  score    run if unscored_count > 0
  analyse  run if gaps_stale, or gaps_computed_at is None

  Each Task carries explicit stop_conditions set by the Orchestrator
  from config — agents never decide their own limits (rule 29).

  Skipped agents appear as Tasks with skipped=True and a reason
  so the plan is always complete (rule 31).
"""

from __future__ import annotations

from dataclasses import dataclass, field
from typing import Any

from edgedash.config import Config
from edgedash.state import SystemState


# ---------------------------------------------------------------------------
# Data model
# ---------------------------------------------------------------------------

@dataclass
class StopConditions:
    max_items: int   = 0      # 0 = not applicable for this agent
    max_seconds: int = 0
    max_pages: int   = 0
    max_listings: int = 0
    widen_distribution: bool = False  # set True on retry after score_spread failure

    def describe(self) -> str:
        parts: list[str] = []
        if self.max_pages:
            parts.append(f"max_pages={self.max_pages}")
        if self.max_listings:
            parts.append(f"max_listings={self.max_listings}")
        if self.max_items:
            parts.append(f"max_items={self.max_items}")
        if self.max_seconds:
            parts.append(f"max_seconds={self.max_seconds}")
        if self.widen_distribution:
            parts.append("widen_distribution=True")
        return ", ".join(parts) if parts else "none"


@dataclass
class Task:
    agent_name: str
    goal: str
    stop_conditions: StopConditions
    reason: str           # the state value that caused the decision
    skipped: bool = False


@dataclass
class Plan:
    tasks: list[Task]

    # ------------------------------------------------------------------
    # render() — one line per agent, compact, printable before execution
    # ------------------------------------------------------------------
    def render(self) -> str:
        lines: list[str] = []
        for t in self.tasks:
            if t.skipped:
                lines.append(
                    f"  SKIP  {t.agent_name:<14}  {t.reason}"
                )
            else:
                lines.append(
                    f"  RUN   {t.agent_name:<14}  goal={t.goal!r}"
                    f"  stop=[{t.stop_conditions.describe()}]"
                    f"  because: {t.reason}"
                )
        return "\n".join(lines)

    def agents_to_run(self) -> list[Task]:
        return [t for t in self.tasks if not t.skipped]

    def agents_skipped(self) -> list[Task]:
        return [t for t in self.tasks if t.skipped]


# ---------------------------------------------------------------------------
# build_plan — read this first
# ---------------------------------------------------------------------------

def build_plan(state: SystemState, config: Config) -> Plan:
    """Decide which agents to run based on current state.

    Pure function of (state, config).  No I/O, no side effects.

    Returns a Plan containing ALL agents — both to-run and skipped —
    each with an explicit reason naming the state value (rule 31).
    """
    tasks: list[Task] = []

    # ── Fetcher ─────────────────────────────────────────────────────────
    fetch_due = (
        state.hours_since_fetch is None                       # never fetched
        or state.hours_since_fetch >= config.fetch_interval_hours
    )

    if fetch_due:
        if state.hours_since_fetch is None:
            fetch_reason = "never_fetched"
        else:
            fetch_reason = (
                f"hours_since_fetch={state.hours_since_fetch:.1f}"
                f" >= threshold={config.fetch_interval_hours}"
            )
        tasks.append(Task(
            agent_name = "Fetcher",
            goal       = "fetch new job listings from all enabled sources",
            stop_conditions = StopConditions(
                max_pages    = config.fetch_max_pages,
                max_listings = config.fetch_max_listings,
                max_seconds  = 0,   # Fetcher uses per-source rate limiting instead
            ),
            reason  = fetch_reason,
            skipped = False,
        ))
    else:
        tasks.append(Task(
            agent_name = "Fetcher",
            goal       = "fetch new job listings",
            stop_conditions = StopConditions(),
            reason  = (
                f"skipped: hours_since_fetch={state.hours_since_fetch:.1f}"
                f" < threshold={config.fetch_interval_hours}"
            ),
            skipped = True,
        ))

    # ── Scorer ───────────────────────────────────────────────────────────
    if state.unscored_count > 0:
        tasks.append(Task(
            agent_name = "Scorer",
            goal       = f"score up to {config.score_batch_size} unscored listings",
            stop_conditions = StopConditions(
                max_items   = config.score_batch_size,
                max_seconds = config.score_max_seconds,
            ),
            reason  = f"unscored_count={state.unscored_count}",
            skipped = False,
        ))
    else:
        tasks.append(Task(
            agent_name = "Scorer",
            goal       = "score listings",
            stop_conditions = StopConditions(),
            reason  = "skipped: unscored_count=0",
            skipped = True,
        ))

    # ── GapAnalyzer ──────────────────────────────────────────────────────
    analyse_due = state.gaps_computed_at is None or state.gaps_stale

    if analyse_due:
        if state.gaps_computed_at is None:
            analyse_reason = "gaps_computed_at=None (never run)"
        else:
            analyse_reason = (
                f"gaps_stale=True (last_scored_at={state.last_scored_at} "
                f"> gaps_computed_at={state.gaps_computed_at})"
            )
        tasks.append(Task(
            agent_name = "GapAnalyzer",
            goal       = "recompute skill gap report from all scored listings",
            stop_conditions = StopConditions(
                max_seconds = config.analyse_max_seconds,
            ),
            reason  = analyse_reason,
            skipped = False,
        ))
    else:
        tasks.append(Task(
            agent_name = "GapAnalyzer",
            goal       = "analyse skill gaps",
            stop_conditions = StopConditions(),
            reason  = (
                f"skipped: gaps_stale=False "
                f"(gaps_computed_at={state.gaps_computed_at} is current)"
            ),
            skipped = True,
        ))

    return Plan(tasks=tasks)
