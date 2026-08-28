"""
app.py — EdgeDash Streamlit dashboard. Read-only. Never runs a cycle.

Rule 38: every data panel reads from the LAST PASSING CYCLE only.
         The activity log is the exception: it shows ALL cycles so
         failures are visible.
"""

from __future__ import annotations

import re
import logging
from datetime import datetime, timezone
from typing import Any, Optional

logger = logging.getLogger(__name__)

import streamlit as st

from edgedash import storage
from edgedash.config import load_config

# ---------------------------------------------------------------------------
# Page config
# ---------------------------------------------------------------------------

st.set_page_config(
    page_title="EdgeDash",
    page_icon="⚡",
    layout="wide",
    initial_sidebar_state="collapsed",
)

# ---------------------------------------------------------------------------
# Config (cached forever — config.yaml doesn't change at runtime)
# ---------------------------------------------------------------------------

@st.cache_resource
def _cfg():
    return load_config()


cfg = _cfg()
DB = cfg.db_path

# ---------------------------------------------------------------------------
# Cached storage readers
# ---------------------------------------------------------------------------

_TTL = 30   # seconds — short enough to feel live, cheap enough for SQLite


@st.cache_data(ttl=_TTL)
def _last_passing_cycle() -> Optional[dict]:
    return storage.get_last_passing_cycle(DB)


@st.cache_data(ttl=_TTL)
def _latest_cycle() -> Optional[dict]:
    """Most recent cycle_summary row regardless of status."""
    conn = storage._connect(DB)
    try:
        row = conn.execute(
            "SELECT agent, started_at, finished_at, status, notes "
            "FROM cycle_log WHERE agent='cycle_summary' ORDER BY id DESC LIMIT 1"
        ).fetchone()
        return dict(row) if row else None
    finally:
        conn.close()


@st.cache_data(ttl=_TTL)
def _activity_log(limit: int = 30) -> list[dict]:
    """All cycle_summary rows, newest first, for the activity log panel."""
    conn = storage._connect(DB)
    try:
        rows = conn.execute(
            "SELECT id, started_at, finished_at, status, notes "
            "FROM cycle_log WHERE agent='cycle_summary' "
            "ORDER BY id DESC LIMIT ?",
            (limit,),
        ).fetchall()
        return [dict(r) for r in rows]
    finally:
        conn.close()


@st.cache_data(ttl=_TTL)
def _top_listings(limit: int = 10) -> list[dict]:
    return storage.get_listings(DB, limit=limit, min_score=1)


@st.cache_data(ttl=_TTL)
def _top_gaps(limit: int = 10) -> list[dict]:
    gaps = storage.get_latest_gap_snapshot(DB)
    return gaps[:limit]


@st.cache_data(ttl=_TTL)
def _counts() -> tuple[int, int]:
    conn = storage._connect(DB)
    try:
        total  = conn.execute("SELECT COUNT(*) FROM listings").fetchone()[0]
        scored = conn.execute(
            "SELECT COUNT(*) FROM listings WHERE fit_score IS NOT NULL"
        ).fetchone()[0]
        return total, scored
    finally:
        conn.close()


# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------

def _fmt_ts(iso: Optional[str], fallback: str = "—") -> str:
    """ISO → human-readable UTC string."""
    if not iso:
        return fallback
    try:
        dt = datetime.fromisoformat(iso)
        if dt.tzinfo is None:
            dt = dt.replace(tzinfo=timezone.utc)
        return dt.strftime("%Y-%m-%d %H:%M UTC")
    except (ValueError, TypeError):
        return iso[:19] if iso else fallback


def _ms_to_s(ms_str: str) -> str:
    """'76637ms' → '76.6 s'"""
    try:
        return f"{int(ms_str.rstrip('ms')) / 1000:.1f} s"
    except (ValueError, AttributeError):
        return ms_str


def _parse_notes(notes: Optional[str]) -> dict[str, str]:
    """Parse 'key=value  key2=value2 ...' notes string into a dict.

    Handles brackets in timings=[...] and commas in lists.
    """
    if not notes:
        return {}
    result: dict[str, str] = {}
    # Extract bracketed timings first
    bracket = re.search(r"timings=\[([^\]]*)\]", notes)
    if bracket:
        result["timings"] = bracket.group(1)
        notes = notes.replace(bracket.group(0), "")

    for part in notes.split("  "):
        if "=" in part:
            k, _, v = part.strip().partition("=")
            result[k.strip()] = v.strip()
    return result


def _wall_time(notes: Optional[str]) -> str:
    """Extract total wall time from timings field in notes."""
    parsed = _parse_notes(notes)
    timings_str = parsed.get("timings", "")
    if not timings_str:
        return "—"
    # Sum all Xms values
    total_ms = sum(
        int(m) for m in re.findall(r"(\d+)ms", timings_str)
    )
    if total_ms == 0:
        return "—"
    return f"{total_ms / 1000:.1f} s"


def _verdict_badge(status: str) -> str:
    mapping = {
        "ok":            "🟢 ok",
        "partial":       "🟡 partial",
        "nothing_to_do": "⚪ nothing to do",
        "degraded":      "🔴 degraded",
        "failed":        "🔴 failed",
    }
    return mapping.get(status, f"⚫ {status}")


# ---------------------------------------------------------------------------
# ══ Section 1: Header strip ══
# ---------------------------------------------------------------------------

def _render_header() -> None:
    latest     = _latest_cycle()
    passing    = _last_passing_cycle()
    total, scored = _counts()

    current_verdict = (latest or {}).get("status", "no data")
    is_stale = (
        latest is not None
        and current_verdict not in ("ok", "nothing_to_do")
        and passing is not None
    )

    st.title("⚡ EdgeDash — Agent Activity Dashboard")

    # Stale-data warning banner
    if latest is None:
        st.info("No cycles yet — first run is scheduled for shortly.")
        return

    if is_stale:
        st.warning(
            f"⚠️  **Newest cycle is '{current_verdict}'.** "
            f"Data panels below show the last **verified** cycle "
            f"({_fmt_ts(passing['started_at'])}).  "
            "The activity log shows all cycles including failures."
        )

    # Metric strip
    col1, col2, col3, col4 = st.columns(4)
    with col1:
        label = "Last verified cycle"
        val   = _fmt_ts((passing or {}).get("started_at"), "none yet")
        st.metric(label, val)
    with col2:
        st.metric("Total listings", f"{total:,}")
    with col3:
        pct = f"{100*scored//total}%" if total else "—"
        st.metric("Scored", f"{scored:,}  ({pct})")
    with col4:
        st.metric("Current verdict", _verdict_badge(current_verdict))

    st.divider()


# ---------------------------------------------------------------------------
# ══ Section 2: Agent activity log ══
# ---------------------------------------------------------------------------

def _render_activity_log() -> None:
    st.subheader("🔄 Agent Activity Log  *(all cycles, most recent first)*")

    rows = _activity_log(30)
    if not rows:
        st.info("No cycles recorded yet.")
        return

    # Legend
    st.caption("🟢 ok  |  🟡 partial / nothing_to_do  |  🔴 degraded / failed")

    # Column header
    hcols = st.columns([2, 1, 3, 2, 2, 1, 1])
    headers = ["Timestamp", "Verdict", "Agents run", "Skipped", "Failed check", "Retries", "Wall time"]
    for col, label in zip(hcols, headers):
        col.markdown(f"**{label}**")
    st.markdown("---")

    for row in rows:
        parsed     = _parse_notes(row.get("notes"))
        verdict    = row["status"]
        ts         = _fmt_ts(row.get("started_at"))
        ran        = parsed.get("ran", "—")
        skipped    = parsed.get("skipped", "—")
        failed_chk = parsed.get("failed_checks", "—")
        retries    = parsed.get("retries", "0")
        wall       = _wall_time(row.get("notes"))

        # Tidy up long agent lists — one per line using bullet points
        def _agent_pills(s: str) -> str:
            if not s or s in ("—", "none"):
                return "—"
            parts = [p.strip() for p in s.split(",") if p.strip()]
            return "  \n".join(f"• {p}" for p in parts)

        # Background colour via markdown container trick
        if verdict == "ok":
            badge = "🟢 ok"
            bg_style = "background:#1a3a2a; border-left:4px solid #28a745; padding:6px 10px; border-radius:4px; margin-bottom:4px;"
        elif verdict in ("degraded", "failed"):
            badge = "🔴 " + verdict
            bg_style = "background:#3a1a1a; border-left:4px solid #dc3545; padding:6px 10px; border-radius:4px; margin-bottom:4px;"
        elif verdict in ("partial",):
            badge = "🟡 partial"
            bg_style = "background:#3a3010; border-left:4px solid #ffc107; padding:6px 10px; border-radius:4px; margin-bottom:4px;"
        elif verdict == "nothing_to_do":
            badge = "⚪ idle"
            bg_style = "background:#1e1e1e; border-left:4px solid #6c757d; padding:6px 10px; border-radius:4px; margin-bottom:4px;"
        else:
            badge = verdict
            bg_style = "background:#1e1e1e; border-left:4px solid #aaa; padding:6px 10px; border-radius:4px; margin-bottom:4px;"

        with st.container():
            st.markdown(f'<div style="{bg_style}">', unsafe_allow_html=True)
            rcols = st.columns([2, 1, 3, 2, 2, 1, 1])
            rcols[0].markdown(f"`{ts}`")
            rcols[1].markdown(badge)
            rcols[2].markdown(_agent_pills(ran))
            rcols[3].markdown(_agent_pills(skipped))
            if failed_chk and failed_chk not in ("—", "none"):
                rcols[4].markdown(f"⚠ `{failed_chk}`")
            else:
                rcols[4].markdown("—")
            rcols[5].markdown(retries)
            rcols[6].markdown(wall)
            st.markdown("</div>", unsafe_allow_html=True)

    st.divider()


# ---------------------------------------------------------------------------
# ══ Section 3a: Top scored listings ══
# ---------------------------------------------------------------------------

def _render_top_listings() -> None:
    st.subheader("🏆 Top 10 Scored Listings")

    listings = _top_listings(10)
    if not listings:
        st.info("No scored listings yet.")
        return

    import pandas as pd

    rows = []
    for l in sorted(listings, key=lambda x: x.get("fit_score") or 0, reverse=True):
        score  = l.get("fit_score", "—")
        title  = l.get("title", "—")
        company = l.get("company", "—")
        reason = (l.get("fit_reason") or "—")
        url    = l.get("url", "")
        rows.append({
            "Score":   score,
            "Title":   f"[{title}]({url})" if url else title,
            "Company": company,
            "Why":     reason[:120] + ("…" if len(reason) > 120 else ""),
        })

    df = pd.DataFrame(rows)
    st.dataframe(
        df,
        use_container_width=True,
        height=min(60 + 35 * len(rows), 420),
        hide_index=True,
        column_config={
            "Title": st.column_config.LinkColumn("Title"),
            "Score": st.column_config.NumberColumn("Score", format="%d"),
        },
    )


# ---------------------------------------------------------------------------
# ══ Section 3b: Top skill gaps ══
# ---------------------------------------------------------------------------

def _render_top_gaps() -> None:
    st.subheader("🎯 Top 10 Skill Gaps")

    gaps = _top_gaps(10)
    if not gaps:
        st.info("No gap data yet — run at least one full scoring cycle.")
        return

    import pandas as pd

    rows = []
    max_cost = gaps[0].get("opportunity_cost", 1) if gaps else 1
    for rank, g in enumerate(gaps, 1):
        skill    = g.get("skill", "—")
        cost     = g.get("opportunity_cost", 0)
        blocked  = g.get("listings_blocked", g.get("raw_frequency", "—"))
        mean_s   = g.get("mean_score", "—")
        n        = g.get("sample_size", "—")
        lc       = g.get("low_confidence", False)
        rows.append({
            "#":          rank,
            "Skill":      skill,
            "Cost":       round(cost, 2),
            "Blocked":    blocked,
            "Mean score": round(mean_s, 1) if isinstance(mean_s, float) else mean_s,
            "n":          n,
            "Flag":       "⚠ low n" if lc else "",
        })

    df = pd.DataFrame(rows)
    st.dataframe(
        df,
        use_container_width=True,
        height=min(60 + 38 * len(rows), 440),
        hide_index=True,
        column_config={
            "Cost":    st.column_config.NumberColumn("Opp. cost", format="%.2f",
                           help="Sum of fit_score/100 across blocked listings"),
            "Blocked": st.column_config.NumberColumn("Blocked", help="Number of listings requiring this skill"),
            "n":       st.column_config.NumberColumn("n", help="Sample size"),
        },
    )

    st.caption(
        "**Opp. cost** = sum(fit_score / 100) for each listing that requires the skill.  "
        "**⚠ low n** = computed from fewer than 3 listings."
    )


# ---------------------------------------------------------------------------
# Layout
# ---------------------------------------------------------------------------

try:
    _render_header()
except Exception as exc:
    logger.error("Database connection or header render failed", exc_info=True)
    st.error("Database not configured or unreachable.")
    st.stop()

try:
    _render_activity_log()
except Exception as exc:
    logger.error("Activity log panel failed", exc_info=True)
    st.error("Could not load activity log.")

col_left, col_right = st.columns(2)
with col_left:
    try:
        _render_top_listings()
    except Exception as exc:
        logger.error("Top listings panel failed", exc_info=True)
        st.error("Could not load top listings.")
with col_right:
    try:
        _render_top_gaps()
    except Exception as exc:
        logger.error("Top gaps panel failed", exc_info=True)
        st.error("Could not load top gaps.")

# ---------------------------------------------------------------------------
# Section 4: Ask your data
# ---------------------------------------------------------------------------

_EXAMPLE_QUESTIONS = [
    "Which companies are hiring right now?",
    "What should I learn next to improve my chances?",
    "Show me my best job matches.",
]


def _render_ask() -> None:
    st.divider()
    st.subheader("Ask your data")
    st.caption(
        "Two LLM calls: one to route your question to the right query, "
        "one to phrase the result in plain English. Read-only."
    )

    # Example question buttons
    ex_cols = st.columns(len(_EXAMPLE_QUESTIONS))
    clicked_example = None
    for col, q in zip(ex_cols, _EXAMPLE_QUESTIONS):
        with col:
            if st.button(q, use_container_width=True):
                clicked_example = q

    if "ask_question" not in st.session_state:
        st.session_state["ask_question"] = ""
    if clicked_example:
        st.session_state["ask_question"] = clicked_example

    question = st.text_input(
        "Your question",
        value=st.session_state["ask_question"],
        placeholder="e.g. What are my top 5 skill gaps?",
        label_visibility="collapsed",
    )
    st.session_state["ask_question"] = question

    if not question.strip():
        return

    @st.cache_data(ttl=60, show_spinner=False)
    def _run_ask(q: str):
        from edgedash.query.ask import ask
        return ask(q)

    with st.spinner("Thinking..."):
        try:
            answer = _run_ask(question.strip())
        except Exception as exc:
            st.error(f"Pipeline error: {exc}")
            return

    if answer.confidence == "no_match":
        st.warning(answer.text)
    else:
        st.success(answer.text)
        if answer.tool_used:
            param_str = (
                ", ".join(f"{k}={v}" for k, v in answer.params.items())
                or "defaults"
            )
            st.caption(
                f"**Tool:** `{answer.tool_used}`  |  "
                f"**Params:** {param_str}  |  "
                f"**Confidence:** {answer.confidence}  |  "
                f"{answer.summary}"
            )

    # Raw rows table per rule 44
    if answer.rows:
        import pandas as pd
        with st.expander(f"Raw data ({len(answer.rows)} rows)", expanded=False):
            st.dataframe(
                pd.DataFrame(answer.rows),
                use_container_width=True,
                hide_index=True,
            )


try:
    _render_ask()
except Exception as exc:
    logger.error("Ask panel failed", exc_info=True)
    st.error("Could not load ask panel.")

# ---------------------------------------------------------------------------
# Footer
# ---------------------------------------------------------------------------
st.divider()
passing = None
try:
    passing = _last_passing_cycle()
except Exception:
    pass
ts_str = _fmt_ts(passing["started_at"]) if passing else "never"

st.markdown(
    f"""
    <div style="text-align: center; color: gray; font-size: small;">
        Last successful cycle: {ts_str} | <a href="https://github.com" target="_blank" style="color: gray;">GitHub Repo</a>
    </div>
    """,
    unsafe_allow_html=True
)
