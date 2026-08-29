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
    page_title="EdgeDash — AI Career Intelligence",
    page_icon="⚡",
    layout="wide",
    initial_sidebar_state="collapsed",
)

# ---------------------------------------------------------------------------
# Premium custom CSS
# ---------------------------------------------------------------------------

st.markdown("""
<style>
  /* ── Typography ── */
  @import url('https://fonts.googleapis.com/css2?family=Inter:wght@300;400;500;600;700&display=swap');
  html, body, [class*="css"] { font-family: 'Inter', sans-serif; }

  /* ── Gradient header banner ── */
  .dash-header {
    background: linear-gradient(135deg, #0d1117 0%, #0f2027 40%, #203a43 70%, #2c5364 100%);
    border-radius: 16px;
    padding: 32px 36px 24px 36px;
    margin-bottom: 24px;
    border: 1px solid rgba(0, 188, 212, 0.2);
    box-shadow: 0 4px 32px rgba(0, 188, 212, 0.08);
  }
  .dash-header h1 {
    font-size: 2.2rem;
    font-weight: 700;
    color: #ffffff;
    margin: 0 0 4px 0;
    letter-spacing: -0.5px;
  }
  .dash-header .subtitle {
    color: rgba(0, 188, 212, 0.9);
    font-size: 0.95rem;
    font-weight: 400;
    margin: 0;
  }

  /* ── Metric cards ── */
  [data-testid="metric-container"] {
    background: #0d1117;
    border: 1px solid rgba(0, 188, 212, 0.18);
    border-radius: 12px;
    padding: 16px 20px;
    box-shadow: 0 2px 12px rgba(0,0,0,0.3);
    transition: border-color 0.2s ease;
  }
  [data-testid="metric-container"]:hover { border-color: rgba(0, 188, 212, 0.45); }
  [data-testid="stMetricLabel"] { color: rgba(255,255,255,0.55) !important; font-size: 0.8rem !important; }
  [data-testid="stMetricValue"] { color: #ffffff !important; font-size: 1.5rem !important; font-weight: 600 !important; }

  /* ── Section headings ── */
  h2, h3 { color: #e0f7fa !important; }

  /* ── Activity log rows ── */
  .log-row-ok      { background: rgba(0,200,83,0.07);    border-left: 3px solid #00c853; padding: 6px 12px; border-radius: 6px; margin-bottom: 4px; }
  .log-row-fail    { background: rgba(229,57,53,0.08);   border-left: 3px solid #e53935; padding: 6px 12px; border-radius: 6px; margin-bottom: 4px; }
  .log-row-partial { background: rgba(255,179,0,0.07);   border-left: 3px solid #ffb300; padding: 6px 12px; border-radius: 6px; margin-bottom: 4px; }
  .log-row-idle    { background: rgba(97,97,97,0.08);    border-left: 3px solid #616161; padding: 6px 12px; border-radius: 6px; margin-bottom: 4px; }

  /* ── Dataframe overrides ── */
  [data-testid="stDataFrame"] { border-radius: 10px; overflow: hidden; }

  /* ── Divider ── */
  hr { border-color: rgba(0, 188, 212, 0.12) !important; }

  /* ── Footer ── */
  .dash-footer { text-align: center; color: rgba(255,255,255,0.35); font-size: 0.78rem; padding-top: 8px; }
  .dash-footer a { color: rgba(0, 188, 212, 0.7); text-decoration: none; }
  .dash-footer a:hover { color: #00bcd4; }
</style>
""", unsafe_allow_html=True)

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
    with storage._tx(DB) as cur:
        sql = storage._adapt_sql(
            "SELECT agent, started_at, finished_at, status, notes "
            "FROM cycle_log WHERE agent='cycle_summary' ORDER BY id DESC LIMIT 1", DB)
        return storage._fetchone(cur, sql)


@st.cache_data(ttl=_TTL)
def _activity_log(limit: int = 30) -> list[dict]:
    """All cycle_summary rows, newest first, for the activity log panel."""
    with storage._tx(DB) as cur:
        sql = storage._adapt_sql(
            "SELECT id, started_at, finished_at, status, notes "
            "FROM cycle_log WHERE agent='cycle_summary' "
            "ORDER BY id DESC LIMIT ?", DB)
        return storage._fetchall(cur, sql, (limit,))


@st.cache_data(ttl=_TTL)
def _top_listings(limit: int = 10) -> list[dict]:
    return storage.get_listings(DB, limit=limit, min_score=1)


@st.cache_data(ttl=_TTL)
def _top_gaps(limit: int = 10) -> list[dict]:
    gaps = storage.get_latest_gap_snapshot(DB)
    return gaps[:limit]


@st.cache_data(ttl=_TTL)
def _counts() -> tuple[int, int]:
    # Use two separate transactions to avoid cursor reuse issues on Postgres
    with storage._tx(DB) as cur:
        row1 = storage._fetchone(cur, "SELECT COUNT(*) AS c FROM listings", path=DB)
        total = row1["c"] if row1 else 0
    with storage._tx(DB) as cur:
        row2 = storage._fetchone(
            cur,
            "SELECT COUNT(*) AS c FROM listings WHERE fit_score IS NOT NULL",
            path=DB,
        )
        scored = row2["c"] if row2 else 0
    return total, scored


# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------

def _fmt_ts(iso: Optional[str], fallback: str = "—") -> str:
    """ISO → human-readable UTC string."""
    if not iso:
        return fallback
    try:
        # Handle datetime objects that may come from Postgres backend
        if isinstance(iso, datetime):
            dt = iso
        else:
            dt = datetime.fromisoformat(str(iso))
        if dt.tzinfo is None:
            dt = dt.replace(tzinfo=timezone.utc)
        return dt.strftime("%Y-%m-%d %H:%M UTC")
    except (ValueError, TypeError):
        s = str(iso)
        return s[:19] if s else fallback


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

    st.markdown("""
    <div class="dash-header">
      <h1>\u26a1 EdgeDash</h1>
      <p class="subtitle">AI Career Intelligence Agent \u2014 Real-time job market analysis powered by Gemini</p>
    </div>
    """, unsafe_allow_html=True)

    # Stale-data warning banner
    if latest is None:
        st.info("\U0001f680 No cycles yet \u2014 trigger the first run via GitHub Actions or run `python run_cycle.py` locally.")
        return

    if is_stale:
        st.warning(
            f"\u26a0\ufe0f  **Newest cycle is '{current_verdict}'.** "
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
        pct = f"{100*scored//total}%" if total else "\u2014"
        st.metric("Scored", f"{scored:,}  ({pct})")
    with col4:
        st.metric("Current verdict", _verdict_badge(current_verdict))

    st.divider()


# ---------------------------------------------------------------------------
# ══ Section 2: Agent activity log ══
# ---------------------------------------------------------------------------

def _render_activity_log() -> None:
    st.subheader("\U0001f504 Agent Activity Log")
    st.caption("\U0001f7e2 ok  |  \U0001f7e1 partial / nothing_to_do  |  \U0001f534 degraded / failed  *(most recent first)*")

    rows = _activity_log(30)
    if not rows:
        st.info("No cycles recorded yet.")
        return

    # Table header
    hcols = st.columns([2, 1, 3, 2, 2, 1, 1])
    headers = ["Timestamp", "Verdict", "Agents run", "Skipped", "Failed check", "Retries", "Wall time"]
    for col, label in zip(hcols, headers):
        col.markdown(
            f"<small style='color:rgba(255,255,255,0.5);font-weight:600;text-transform:uppercase;letter-spacing:0.05em'>{label}</small>",
            unsafe_allow_html=True,
        )

    for row in rows:
        parsed     = _parse_notes(row.get("notes"))
        verdict    = row["status"]
        ts         = _fmt_ts(row.get("started_at"))
        ran        = parsed.get("ran", "\u2014")
        skipped    = parsed.get("skipped", "\u2014")
        failed_chk = parsed.get("failed_checks", "\u2014")
        retries    = parsed.get("retries", "0")
        wall       = _wall_time(row.get("notes"))

        # Tidy up long agent lists — one per line using bullet points
        def _agent_pills(s: str) -> str:
            if not s or s in ("\u2014", "none"):
                return "\u2014"
            parts = [p.strip() for p in s.split(",") if p.strip()]
            return "  \n".join(f"\u2022 {p}" for p in parts)

        if verdict == "ok":
            badge = "\U0001f7e2 ok"
            row_class = "log-row-ok"
        elif verdict in ("degraded", "failed"):
            badge = "\U0001f534 " + verdict
            row_class = "log-row-fail"
        elif verdict == "partial":
            badge = "\U0001f7e1 partial"
            row_class = "log-row-partial"
        elif verdict == "nothing_to_do":
            badge = "\u26aa idle"
            row_class = "log-row-idle"
        else:
            badge = verdict
            row_class = "log-row-idle"

        with st.container():
            rcols = st.columns([2, 1, 3, 2, 2, 1, 1])
            rcols[0].markdown(
                f'<div class="{row_class}"><code style="font-size:0.8rem">{ts}</code></div>',
                unsafe_allow_html=True,
            )
            rcols[1].markdown(badge)
            rcols[2].markdown(_agent_pills(ran))
            rcols[3].markdown(_agent_pills(skipped))
            if failed_chk and failed_chk not in ("\u2014", "none"):
                rcols[4].markdown(f"\u26a0 `{failed_chk}`")
            else:
                rcols[4].markdown("\u2014")
            rcols[5].markdown(retries)
            rcols[6].markdown(wall)

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
    <div class="dash-footer">
        Last successful cycle: {ts_str} &nbsp;|&nbsp;
        <a href="https://github.com/24btcsepy0026/AI_CAREER_INTELLIGENCE_AGENT" target="_blank">GitHub Repo</a>
        &nbsp;|&nbsp; EdgeDash v1.0 &middot; Powered by Gemini AI
    </div>
    """,
    unsafe_allow_html=True
)
