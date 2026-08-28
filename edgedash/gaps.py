"""
edgedash/gaps.py

Morning terminal view of the latest gap snapshot, plus trend reporting.

Usage
-----
    python -m edgedash.gaps              # show top 10
    python -m edgedash.gaps --top 20     # show top N
    python -m edgedash.gaps --all        # show everything
    python -m edgedash.gaps --trend      # trend vs earliest snapshot

Trend rules
-----------
  - Read-only. No new writes. Deterministic.
  - Compares current top-10 against the EARLIEST snapshot on record.
  - If only one snapshot exists: says so exactly, no fabrication.
  - Marks NEW skills (not present at earliest) and DROPPED skills
    (were in earliest top-10 but not in current top-10).
  - Prints the snapshot dates so the window is always visible.
"""

from __future__ import annotations

import argparse

from edgedash.config import load_config
from edgedash import storage


_BAR_WIDTH = 20
_TREND_TOP_N = 10          # skills tracked in trend view
_MIN_SNAPSHOTS_FOR_TREND = 2


# ---------------------------------------------------------------------------
# Shared helpers
# ---------------------------------------------------------------------------

def _fmt_dt(iso: str) -> str:
    """Trim ISO timestamp to readable date+time, drop microseconds."""
    return iso[:19].replace("T", " ") + " UTC"


def _bar(value: float, max_value: float, width: int = _BAR_WIDTH) -> str:
    if max_value <= 0:
        return ""
    filled = round(width * value / max_value)
    return "█" * filled + "░" * (width - filled)


# ---------------------------------------------------------------------------
# Latest snapshot table
# ---------------------------------------------------------------------------

def _print_table(gaps: list[dict], top_n: int | None) -> None:
    if not gaps:
        print("No gap data found. Run `python run_cycle.py` first.")
        return

    rows = gaps if top_n is None else gaps[:top_n]
    max_cost = rows[0]["opportunity_cost"] if rows else 1.0

    print()
    print(
        f"{'#':>3}  {'SKILL':<28} {'BLOCKED':>7}  {'COST':>6}  "
        f"{'MEAN':>5}  {'TOP':>4}  {'BAR':<{_BAR_WIDTH}}  NOTE"
    )
    print("─" * 90)

    for rank, g in enumerate(rows, 1):
        skill = g["skill"][:27]
        blocked = g["listings_blocked"]
        cost = g["opportunity_cost"]
        mean = g["mean_score"]
        top = g["top_score"]
        bar = _bar(cost, max_cost)
        note_parts = []
        if g.get("low_confidence"):
            note_parts.append("low-confidence")
        nice = g.get("also_nice_to_have", 0)
        if nice:
            note_parts.append(f"+{nice} nice-to-have")
        note = ", ".join(note_parts)

        print(
            f"{rank:>3}  {skill:<28} {blocked:>7}  {cost:>6.2f}  "
            f"{mean:>5.1f}  {top:>4}  {bar:<{_BAR_WIDTH}}  {note}"
        )

    print()
    total = len(gaps)
    lc = sum(1 for g in gaps if g.get("low_confidence"))
    shown = len(rows)
    print(f"Showing {shown} of {total} gaps  ·  {lc} low-confidence (n < 3)")
    print()


# ---------------------------------------------------------------------------
# Trend view
# ---------------------------------------------------------------------------

def _print_trend(trend: dict) -> None:
    n_snapshots = trend["snapshot_count"]

    if n_snapshots == 0:
        print("\nNo gap snapshots found. Run `python run_cycle.py` first.\n")
        return

    if n_snapshots < _MIN_SNAPSHOTS_FOR_TREND:
        needed = _MIN_SNAPSHOTS_FOR_TREND - n_snapshots
        print()
        print("─" * 60)
        print("  TREND REPORT")
        print("─" * 60)
        print(f"  Only 1 snapshot exists (taken {_fmt_dt(trend['latest_at'])}).")
        print(f"  {needed} more day(s) of `python run_cycle.py` runs needed")
        print("  before a trend can be shown.")
        print()
        print("  No data has been fabricated, interpolated, or extrapolated.")
        print("─" * 60)
        print()
        return

    earliest_at = trend["earliest_at"]
    latest_at = trend["latest_at"]
    earliest_map: dict[str, float] = trend["earliest"]
    latest_top10: list[dict] = trend["latest_top10"]

    # Earliest top-10 skills by opportunity_cost
    earliest_top10_skills: set[str] = set(
        sorted(earliest_map, key=lambda s: earliest_map[s], reverse=True)[:_TREND_TOP_N]
    )
    current_skills: set[str] = {g["skill"] for g in latest_top10}

    dropped = earliest_top10_skills - current_skills   # were top-10 then, not now
    max_cost = latest_top10[0]["opportunity_cost"] if latest_top10 else 1.0

    print()
    print("═" * 90)
    print("  TREND REPORT — opportunity cost change")
    print(f"  From : {_fmt_dt(earliest_at)}  (snapshot 1 of {n_snapshots})")
    print(f"  To   : {_fmt_dt(latest_at)}   (latest)")
    print("═" * 90)
    print(
        f"{'#':>3}  {'SKILL':<28}  {'THEN':>7}  {'NOW':>7}  "
        f"{'CHG':>7}  {'CHG%':>6}  {'BAR':<14}  FLAG"
    )
    print("─" * 90)

    for rank, g in enumerate(latest_top10, 1):
        skill = g["skill"]
        now_cost = g["opportunity_cost"]
        then_cost = earliest_map.get(skill)   # None if skill is NEW
        bar = _bar(now_cost, max_cost, width=14)

        if then_cost is None:
            chg_str = "    n/a"
            pct_str = "   n/a"
            flag = "NEW"
        else:
            delta = now_cost - then_cost
            chg_str = f"{delta:+7.2f}"
            pct = (delta / then_cost * 100) if then_cost > 0 else 0.0
            pct_str = f"{pct:+6.1f}%"
            flag = "▲" if delta > 0 else ("▼" if delta < 0 else "—")

        skill_disp = g["skill"][:27]
        then_disp = f"{then_cost:7.2f}" if then_cost is not None else "    new"
        print(
            f"{rank:>3}  {skill_disp:<28}  {then_disp}  {now_cost:7.2f}  "
            f"{chg_str}  {pct_str}  {bar:<14}  {flag}"
        )

    # Dropped skills section
    if dropped:
        print()
        print("  DROPPED from top-10 since earliest snapshot:")
        for skill in sorted(dropped, key=lambda s: earliest_map[s], reverse=True):
            old = earliest_map[skill]
            now = trend["latest"].get(skill, 0.0)
            still = f"still tracked at {now:.2f}" if now else "no longer tracked"
            print(f"    • {skill:<30}  was {old:.2f}  ({still})")

    print()
    print(f"  {n_snapshots} snapshots on record · window covers "
          f"{_window_days(earliest_at, latest_at)}")
    print("═" * 90)
    print()


def _window_days(earliest: str, latest: str) -> str:
    """Return a human-readable description of the time window."""
    try:
        from datetime import datetime, timezone
        fmt = "%Y-%m-%dT%H:%M:%S"
        t0 = datetime.fromisoformat(earliest.split(".")[0])
        t1 = datetime.fromisoformat(latest.split(".")[0])
        days = (t1 - t0).days
        hours = int(((t1 - t0).seconds) / 3600)
        if days == 0:
            return f"< 1 day ({hours}h)"
        return f"{days} day(s)"
    except Exception:
        return "unknown duration"


# ---------------------------------------------------------------------------
# CLI entry point
# ---------------------------------------------------------------------------

def main() -> None:
    parser = argparse.ArgumentParser(description="EdgeDash gap report")
    parser.add_argument("--top", type=int, default=10,
                        help="Number of gaps to show (default 10)")
    parser.add_argument("--all", action="store_true",
                        help="Show all gaps, not just top N")
    parser.add_argument("--trend", action="store_true",
                        help="Show trend vs earliest snapshot")
    parser.add_argument("--db", default=None,
                        help="Path to DB (default: from config.yaml)")
    args = parser.parse_args()

    cfg = load_config()
    db_path = args.db or cfg.db_path
    storage.init_db(db_path)

    if args.trend:
        trend = storage.get_gap_trend_data(db_path)
        _print_trend(trend)
    else:
        gaps = storage.get_latest_gap_snapshot(db_path)
        top_n = None if args.all else args.top
        _print_table(gaps, top_n)


if __name__ == "__main__":
    main()
