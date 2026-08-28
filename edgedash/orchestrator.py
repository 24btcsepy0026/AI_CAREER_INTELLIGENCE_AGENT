"""
edgedash/orchestrator.py

Orchestrator — reads system state, builds a plan, executes it, verifies,
optionally retries once, and logs a single summary row.

Steering rules honoured
-----------------------
  Rule 28: plan is derived from state, not a fixed sequence.
           Skipping an agent because there is no work is a successful outcome.
  Rule 29: every Task carries explicit stop_conditions set here, not by the agent.
  Rule 30: no fetching, scoring, or analysis logic here — delegation only.
  Rule 31: Plan is printed BEFORE execution; every agent shows goal + reason.
  Rule 32: one agent failing does not stop the cycle — failure is caught,
           logged, and the cycle continues as "partial".
  Rule 33: one cycle_summary row written at the end with per-agent timings
           and the overall outcome.
  Rule 36: Verifier runs after Scorer + GapAnalyzer. On fail, the specific
           failing agent is re-run once with adjusted stop_conditions.
           A second verification failure marks the cycle "degraded" — no
           further retries, no raise.
"""

from __future__ import annotations

import logging
import traceback
from datetime import datetime, timezone
from typing import Any

from edgedash import storage
from edgedash.agents.base import Agent, AgentResult
from edgedash.agents.fetcher import Fetcher
from edgedash.agents.gap_analyzer import GapAnalyzer
from edgedash.agents.mock_fetcher import MockFetcher
from edgedash.agents.scorer import Scorer
from edgedash.agents.verifier import Verifier
from edgedash.config import Config
from edgedash.planning import Plan, StopConditions, Task, build_plan
from edgedash.state import read_state
from edgedash.verification import Verdict

logger = logging.getLogger(__name__)

_SEP  = "─" * 64
_WIDE = "═" * 64


# ---------------------------------------------------------------------------
# Agent registry — maps agent_name → Agent instance.
# Only place to swap implementations (rule 7 / registry rule).
# ---------------------------------------------------------------------------

def _agent_registry(config: Config) -> dict[str, Agent]:
    fetcher: Agent = MockFetcher() if config.use_mock_fetcher else Fetcher()
    return {
        "Fetcher":     fetcher,
        "Scorer":      Scorer(),
        "GapAnalyzer": GapAnalyzer(),
        "Verifier":    Verifier(),
    }


# ---------------------------------------------------------------------------
# Display helpers
# ---------------------------------------------------------------------------

def _fmt(iso: str | None) -> str:
    if iso is None:
        return "never"
    return iso[:19].replace("T", " ") + " UTC"


def _print_state_banner(state: Any, config: Config) -> None:
    print(_WIDE)
    print("  EdgeDash — Cycle Starting")
    print(_WIDE)
    print(f"  Last fetch    : {_fmt(state.last_fetch_at)}")
    print(f"  Hours ago     : {state.hours_since_fetch if state.hours_since_fetch is not None else 'n/a'}")
    print(f"  Unscored      : {state.unscored_count}")
    print(f"  Gaps stale    : {state.gaps_stale}  (computed: {_fmt(state.gaps_computed_at)})")
    print(f"  Last cycle    : {state.last_cycle_verdict or 'none'}  ({_fmt(state.last_cycle_at)})")
    print(_SEP)


def _print_plan(plan: Plan) -> None:
    print("  PLAN  (printed before execution — rule 31)")
    print(_SEP)
    print(plan.render())
    print(_SEP)


def _print_agent_result(result: AgentResult, elapsed_ms: int, prefix: str = "") -> None:
    icon = "✓" if result.status == "ok" else "✗"
    label = f"{prefix}{result.agent}" if prefix else result.agent
    print(
        f"  {icon} {label:<24}  status={result.status:<8}"
        f"  touched={result.records_touched:<5}  ({elapsed_ms} ms)"
    )
    if result.notes:
        print(f"      → {result.notes}")


def _print_cycle_summary(
    ran: list[tuple[str, AgentResult, int]],
    skipped: list[Task],
    cycle_ms: int,
    verdict: str,
    retry_count: int,
) -> None:
    print(_SEP)
    print("  CYCLE SUMMARY")
    print(_SEP)
    for agent_name, result, elapsed in ran:
        icon = "✓" if result.status == "ok" else "✗"
        note_snip = (result.notes or "")[:60]
        print(f"  {icon} {agent_name:<20}  {elapsed:>6} ms  {note_snip}")
    for task in skipped:
        print(f"  — {task.agent_name:<20}  SKIPPED  {task.reason}")
    print(_SEP)
    print(f"  Verdict      : {verdict}")
    if retry_count:
        print(f"  Retries      : {retry_count}")
    print(f"  Wall time    : {cycle_ms} ms")
    print(_WIDE)


# ---------------------------------------------------------------------------
# Single-agent runner — shared by the main loop and the retry pass
# ---------------------------------------------------------------------------

def _run_one(
    agent_name: str,
    registry: dict[str, Agent],
    config: Config,
    stop: StopConditions,
    prefix: str = "",
) -> tuple[AgentResult, int]:
    """Run one agent; return (AgentResult, elapsed_ms).

    Catches unhandled exceptions (rule 32) and returns a failed AgentResult
    rather than propagating. Never raises.
    """
    agent = registry.get(agent_name)
    if agent is None:
        msg = f"No agent registered for '{agent_name}'"
        logger.error(msg)
        return AgentResult(agent=agent_name, status="failed",
                           records_touched=0, notes=msg), 0

    t0 = _now_ms()
    try:
        result = agent.run(config, config.db_path, stop)
    except Exception as exc:
        elapsed = _now_ms() - t0
        msg = f"unhandled exception: {exc}"
        logger.error("Agent %s crashed: %s\n%s",
                     agent_name, exc, traceback.format_exc())
        result = AgentResult(agent=agent_name, status="failed",
                             records_touched=0, notes=msg)

    elapsed = _now_ms() - t0
    _print_agent_result(result, elapsed, prefix=prefix)
    return result, elapsed


# ---------------------------------------------------------------------------
# Verification + retry logic (rule 36)
# ---------------------------------------------------------------------------

def _retry_stop_for_failure(check_name: str, config: Config) -> tuple[str, StopConditions]:
    """Return (agent_to_retry, adjusted StopConditions) for a named failed check.

    score_spread  → re-run Scorer with widen_distribution=True
    extraction_sanity → re-run Scorer (triggers fresh extraction pass)
    gap_sample_size   → re-run GapAnalyzer with more time
    freshness         → no mechanical retry (data age can't be fixed mid-cycle)
    """
    if check_name in ("score_spread", "extraction_sanity"):
        return "Scorer", StopConditions(
            max_items=config.score_batch_size,
            max_seconds=config.score_max_seconds,
            widen_distribution=(check_name == "score_spread"),
        )
    if check_name == "gap_sample_size":
        return "GapAnalyzer", StopConditions(
            max_seconds=config.analyse_max_seconds * 2,
        )
    # freshness and unknown checks: retry the Verifier itself is pointless —
    # log it as unresolvable in this cycle.
    return "", StopConditions()


def _run_verifier(
    registry: dict[str, Agent],
    config: Config,
    label: str = "",
) -> tuple[AgentResult, int, Verdict | None]:
    """Run the Verifier; return (result, elapsed_ms, Verdict|None)."""
    result, elapsed = _run_one("Verifier", registry, config,
                                StopConditions(), prefix=label)
    verdict: Verdict | None = getattr(result, "_verdict", None)
    return result, elapsed, verdict


# ---------------------------------------------------------------------------
# Main entry point
# ---------------------------------------------------------------------------

def run_cycle(config: Config) -> None:
    """Initialise DB, read state, build plan, execute, verify, log summary."""
    cycle_started = _utcnow()
    cycle_start_ms = _now_ms()

    # (a) Init DB
    storage.init_db(config.db_path)

    # (b) Read state — cheap queries only
    now = datetime.now(timezone.utc)
    state = read_state(config, now)
    _print_state_banner(state, config)

    # (c) Build plan from state — pure function, no I/O (rules 28, 31)
    plan = build_plan(state, config)
    _print_plan(plan)

    # (d) Execute plan — per-agent isolation (rules 29, 30, 32)
    registry = _agent_registry(config)
    # ran tracks (agent_name_label, AgentResult, elapsed_ms) for summary
    ran: list[tuple[str, AgentResult, int]] = []
    skipped: list[Task] = plan.agents_skipped()
    any_failed = False
    retry_count = 0

    print("  RUNNING")
    print(_SEP)

    for task in plan.agents_to_run():
        result, elapsed = _run_one(
            task.agent_name, registry, config, task.stop_conditions
        )
        storage.log_cycle(
            path=config.db_path,
            agent=result.agent,
            started_at=_utcnow(),
            finished_at=_utcnow(),
            records_touched=result.records_touched,
            status=result.status,
            notes=result.notes,
        )
        ran.append((task.agent_name, result, elapsed))
        if result.status == "failed":
            any_failed = True

    # (e) Verification pass (rule 36) — only run if something actually ran
    verdict_label = "ok" if not any_failed else "partial"
    first_verdict: Verdict | None = None

    if ran:
        print(_SEP)
        print("  VERIFYING")
        print(_SEP)
        v_result, v_elapsed, first_verdict = _run_verifier(registry, config)
        storage.log_cycle(
            path=config.db_path,
            agent=v_result.agent,
            started_at=_utcnow(),
            finished_at=_utcnow(),
            records_touched=v_result.records_touched,
            status=v_result.status,
            notes=v_result.notes,
        )
        ran.append(("Verifier", v_result, v_elapsed))

        if first_verdict is not None and not first_verdict.passed:
            # --- Retry pass (rule 36): one retry per cycle maximum ---
            failed_names = [c.name for c in first_verdict.failed_checks]
            # Take only the first actionable failure to drive the retry
            retry_agent, retry_stop = _retry_stop_for_failure(
                failed_names[0], config
            )

            if retry_agent:
                retry_count = 1
                print(_SEP)
                print(f"  RETRY  (verification failed: {', '.join(failed_names)})")
                print(_SEP)

                r2, e2 = _run_one(
                    retry_agent, registry, config, retry_stop, prefix="retry:"
                )
                storage.log_cycle(
                    path=config.db_path,
                    agent=f"retry:{r2.agent}",
                    started_at=_utcnow(),
                    finished_at=_utcnow(),
                    records_touched=r2.records_touched,
                    status=r2.status,
                    notes=r2.notes,
                )
                ran.append((f"retry:{retry_agent}", r2, e2))

                # Re-verify once
                print(_SEP)
                print("  RE-VERIFYING")
                print(_SEP)
                v2_result, v2_elapsed, second_verdict = _run_verifier(
                    registry, config, label="re-verify:"
                )
                storage.log_cycle(
                    path=config.db_path,
                    agent="re-verify:Verifier",
                    started_at=_utcnow(),
                    finished_at=_utcnow(),
                    records_touched=v2_result.records_touched,
                    status=v2_result.status,
                    notes=v2_result.notes,
                )
                ran.append(("re-verify:Verifier", v2_result, v2_elapsed))

                if second_verdict is not None and not second_verdict.passed:
                    # Rule 36: second failure → degraded, no more retries
                    verdict_label = "degraded"
                    logger.warning(
                        "Orchestrator: cycle marked DEGRADED — "
                        "verification failed after retry. Checks: %s",
                        [c.name for c in second_verdict.failed_checks],
                    )
                else:
                    verdict_label = "ok" if not any_failed else "partial"
            else:
                # Unresolvable check (e.g. freshness) — degrade immediately
                verdict_label = "degraded"
                logger.warning(
                    "Orchestrator: check '%s' cannot be resolved mid-cycle — "
                    "marking degraded.", failed_names[0]
                )

    # Override to nothing_to_do if nothing ran at all
    if not ran:
        verdict_label = "nothing_to_do"
    elif verdict_label == "ok" and any_failed:
        verdict_label = "partial"

    cycle_ms = _now_ms() - cycle_start_ms

    # (f) Print + log the ONE cycle summary row (rule 33)
    _print_cycle_summary(ran, skipped, cycle_ms, verdict_label, retry_count)

    ran_names     = [name for name, _, _ in ran]
    skipped_names = [t.agent_name for t in skipped]
    timing_parts  = [f"{name}={ms}ms" for name, _, ms in ran]
    failed_checks = (
        [c.name for c in first_verdict.failed_checks]
        if first_verdict and not first_verdict.passed
        else []
    )

    summary_notes = (
        f"ran={','.join(ran_names) or 'none'}  "
        f"skipped={','.join(skipped_names) or 'none'}  "
        f"timings=[{' '.join(timing_parts)}]  "
        f"retries={retry_count}  "
        f"failed_checks={','.join(failed_checks) or 'none'}  "
        f"verdict={verdict_label}"
    )

    storage.log_cycle(
        path=config.db_path,
        agent="cycle_summary",
        started_at=cycle_started,
        finished_at=_utcnow(),
        records_touched=sum(r.records_touched for _, r, _ in ran),
        status=verdict_label,
        notes=summary_notes,
    )


# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------

def _utcnow() -> str:
    return datetime.now(timezone.utc).isoformat()


def _now_ms() -> int:
    return int(datetime.now(timezone.utc).timestamp() * 1000)
