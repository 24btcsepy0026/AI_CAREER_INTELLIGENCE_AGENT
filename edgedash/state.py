"""
edgedash/state.py

Read system state from the database.  Cheap queries only — counts and
MAX(timestamp), never full table loads.  All I/O goes through storage.

Public API
----------
    read_state(config, now) -> SystemState

`now` is a parameter, never datetime.now() inside this module — making
every function here fully testable without time mocking.
"""

from __future__ import annotations

from dataclasses import dataclass
from datetime import datetime, timezone
from typing import Optional

from edgedash import storage
from edgedash.config import Config


# ---------------------------------------------------------------------------
# Dataclass
# ---------------------------------------------------------------------------

@dataclass
class SystemState:
    # Fetch
    last_fetch_at: Optional[str]       # ISO timestamp or None
    hours_since_fetch: Optional[float] # None if never fetched

    # Scoring
    unscored_count: int
    last_scored_at: Optional[str]      # ISO timestamp of most recent score write

    # Gap analysis
    gaps_computed_at: Optional[str]    # ISO timestamp of most recent gap snapshot
    gaps_stale: bool                   # True if any score is newer than gap snapshot

    # Last cycle
    last_cycle_verdict: Optional[str]  # status field from last cycle_summary row
    last_cycle_at: Optional[str]       # started_at of last cycle_summary row


# ---------------------------------------------------------------------------
# Helper: parse ISO timestamp safely
# ---------------------------------------------------------------------------

def _parse_iso(ts: Optional[str]) -> Optional[datetime]:
    if not ts:
        return None
    try:
        dt = datetime.fromisoformat(ts)
        if dt.tzinfo is None:
            dt = dt.replace(tzinfo=timezone.utc)
        return dt
    except (ValueError, TypeError):
        return None


def _hours_between(earlier: Optional[str], now: datetime) -> Optional[float]:
    dt = _parse_iso(earlier)
    if dt is None:
        return None
    delta = now - dt
    return delta.total_seconds() / 3600.0


def _gaps_stale(last_scored_at: Optional[str], gaps_computed_at: Optional[str]) -> bool:
    """True when there are scores newer than the gap snapshot — gaps need recomputing."""
    scored_dt = _parse_iso(last_scored_at)
    gaps_dt = _parse_iso(gaps_computed_at)

    if scored_dt is None:
        return False   # nothing scored yet → gaps can't be stale
    if gaps_dt is None:
        return True    # never computed → definitely stale

    return scored_dt > gaps_dt


# ---------------------------------------------------------------------------
# Public API
# ---------------------------------------------------------------------------

def read_state(config: Config, now: datetime) -> SystemState:
    """Read current system state from the database.

    `now` must be a timezone-aware datetime.  Passing it as a parameter
    (rather than calling datetime.now() here) makes this fully testable.
    """
    db = config.db_path

    last_fetch_at    = storage.last_fetch_time(db)
    hours_since      = _hours_between(last_fetch_at, now)
    unscored         = storage.count_unscored(db)
    last_scored      = storage.last_scored_at(db)
    gaps_at          = storage.last_gap_run_at(db)
    stale            = _gaps_stale(last_scored, gaps_at)
    cycle_summary    = storage.last_cycle_summary(db)

    return SystemState(
        last_fetch_at       = last_fetch_at,
        hours_since_fetch   = round(hours_since, 2) if hours_since is not None else None,
        unscored_count      = unscored,
        last_scored_at      = last_scored,
        gaps_computed_at    = gaps_at,
        gaps_stale          = stale,
        last_cycle_verdict  = cycle_summary["status"] if cycle_summary else None,
        last_cycle_at       = cycle_summary["started_at"] if cycle_summary else None,
    )
