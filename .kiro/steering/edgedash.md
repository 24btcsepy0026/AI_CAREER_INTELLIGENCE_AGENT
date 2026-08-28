# EdgeDash — Project Steering

## Project Overview

EdgeDash is an autonomous AI career intelligence agent. It runs as a scheduled loop that fetches live job listings, scores them for fit against a user profile, surfaces skill gaps, verifies its own output, and publishes results to a Streamlit dashboard.

---

## Architecture

Do not deviate from this architecture without informing the user first.

```
Trigger (scheduled)
  -> Orchestrator
       -> Fetcher        (sub-agent: fetches raw job listings)
       -> Scorer         (sub-agent: scores jobs against profile)
       -> GapAnalyzer    (sub-agent: identifies skill gaps)
  -> Verifier            (validates agent output)
  -> Storage             (persistence layer)
  -> Dashboard           (Streamlit, read-only)
```

**Orchestrator rules:**
- Reads state and delegates to sub-agents.
- Never fetches job data or scores directly — that is a sub-agent's job.

**Sub-agent rules:**
- Each sub-agent has exactly one goal and one stop condition.
- Sub-agents do not share state with each other directly; all state flows through Storage.

---

## Hard Rules

1. **Python 3.11+. Standard library first.**
   Add a third-party dependency only when it genuinely saves real work. State the dependency name and the reason before adding it to any requirements file.

2. **All storage access goes through a single storage module.**
   No other module may `import sqlite3` directly. The storage module exposes a thin interface. The goal is that swapping SQLite for hosted Postgres in week 4 is a one-file change.

3. **Never hardcode user-specific values.**
   Role, city, keywords, skills profile, and any other user-specific data must live in config (e.g., `config.yaml` or similar). Code reads from config; it never embeds these values.

4. **No secrets in code.**
   API keys, tokens, and credentials are loaded from environment variables only, in one dedicated place (e.g., `config/settings.py` or equivalent). No `.env` values are scattered across modules.

5. **Every agent run writes a cycle log row.**
   The `cycle_log` table must record: what ran, when it ran, how many records were touched, pass/fail status, and any retry reason.

6. **Fail loudly.**
   No bare `except: pass`. No silently swallowed exceptions. If something is wrong, raise or log at ERROR level and let it surface.

7. **Type hints on every function signature.**
   Docstrings only where the intent is not obvious from the name alone.

8. **Keep files under ~150 lines.**
   Split a module before it becomes a problem, not after.

---

## Style Guidelines

- Small, testable functions over large procedural blocks.
- Plain, readable Python over clever Python.
- When the user asks for one module, build that one module. Do not scaffold the entire app unprompted.
- Prefer explicit over implicit.
- No unnecessary abstractions — solve the problem at hand.

---

## Network & Sources

9. **Every external source lives behind a `Source` class with a uniform interface.**
   The Fetcher never contains source-specific parsing. Adding a source must never require editing the Fetcher.

10. **Every `Source` returns a list of normalised dicts with EXACTLY these keys:**
    `source`, `external_id`, `title`, `company`, `location`, `url`, `description`, `posted_at`, `raw`.
    Missing values are `None` — never empty string, never `"N/A"`.

11. **All network calls go through one shared helper.**
    The helper enforces a 10 s timeout, 2 retry attempts with exponential backoff, and a real `User-Agent` header.
    No bare `requests.get` (or equivalent) anywhere else in the codebase.

12. **A source failing must NEVER kill the cycle.**
    Catch per-source, log the failure to `cycle_log` with `status = "failed"`, then continue to the next source.
    One dead job board must not stop the other sources from running.

13. **Secrets come from environment variables via a `.env` file that is gitignored.**
    Never a literal key in code, never a key in `config.yaml`.
    If a required key is missing, that source skips itself and emits a clear `WARNING` log line — it does not raise or crash the cycle.

14. **Respect the source.**
    Rate-limit to at most 1 request per second per source, set a descriptive `User-Agent`, and honour any documented page limits.

---

## Intelligence & Scoring

15. **All LLM calls go through one module: `edgedash/llm.py`, exposing one function.**
    The provider and model name come from config, never hardcoded.
    Rate-limit to stay inside a free tier: default 1 request per second, max 15 per minute.
    No other file imports an LLM SDK directly.

16. **Never ask the model for a score, ranking, or numeric rating.**
    The model extracts structured facts only (skills mentioned, requirements, seniority signals, etc.).
    All scoring arithmetic is deterministic Python in one function.
    The model never sees the scoring weights.

17. **Every model response is validated against an explicit schema before use.**
    A response that fails validation is retried once, then logged as a failure for that listing only —
    it must not crash the cycle or stop the remaining listings from being processed.
    Never `json.loads` raw model text without a validation and repair path.

18. **Scoring is idempotent.**
    Never re-score a listing that already has a score — select only `WHERE fit_score IS NULL`.
    Cache extraction results keyed on a hash of the job description so the same text is never sent
    to the model twice within or across cycles.

19. **Every score carries a human-readable reason generated from the score components by our code —
    never free text written by the model.**
    The reason string is assembled deterministically from the numeric components (e.g. skill matches,
    experience delta, keyword hits) after scoring arithmetic completes.

20. **Log the score distribution to `cycle_log` on every scoring run.**
    Record: count, min, max, mean, and spread (max − min).
    A run where all scores fall within a 10-point window is a suspect run and must be logged
    as such with a clear warning note.

21. **Cap listings scored per cycle at a configurable batch size (default 25).**
    This makes a cost or rate-limit blowup structurally impossible regardless of how many
    unscored listings accumulate.

---

## Aggregate Analysis

22. **Aggregate analysis is deterministic SQL and Python. No LLM call may produce, adjust, or rank
    an aggregate number.**
    A model may only suggest canonical groupings for a human to approve.
    All reported numbers must be reproducible by re-running the same query against the same data.

23. **Skill names are canonicalised through an explicit alias map in `config.yaml` that I own and
    can read.**
    Never auto-merge skill names by model judgement or string similarity alone.
    If a skill has no alias entry it is kept as-is; it is never silently merged with another.

24. **Gap ranking is weighted by the fit score of the listing the gap came from.**
    A gap in a listing scored 20 is worth far less than a gap in a listing scored 85.
    Never rank gaps by raw frequency alone.

25. **Every gap report run writes a timestamped snapshot. Never overwrite the previous report.**
    Trend over time is a first-class output, not an afterthought.
    The snapshot table must store the run timestamp alongside every reported gap and its weight.

26. **Every aggregate number must be traceable to the rows that produced it.**
    Any reported gap must be able to list the specific listing IDs it was computed from.
    No number appears in the dashboard that cannot be drilled into.

27. **Report the sample size alongside every aggregate.**
    A gap computed from 3 listings and a gap computed from 90 listings must never be presented
    as equally reliable. Sample size is a required column, not a footnote.

---

## Orchestration

28. **The Orchestrator reads system state and decides which agents to run.**
    It never runs a fixed sequence. Skipping an agent because there is no work for it is a
    successful outcome, not a failure.

29. **Every delegation carries an explicit goal and an explicit stop condition.**
    Max items and max duration are set by the Orchestrator before the agent runs.
    A sub-agent never decides its own limits.

30. **The Orchestrator never does an agent's work.**
    It reads state, delegates, collects results, and logs. No fetching, scoring, or analysis
    logic belongs in the Orchestrator file.

31. **The Orchestrator prints and logs its plan before executing it.**
    The plan must state: which agents will run, which are skipped, and the state value that
    caused each decision (e.g. "149 unscored listings → run Scorer").

32. **One sub-agent failing does not stop the cycle.**
    Log the failure, continue with the remaining plan, and mark the cycle partial.

33. **Every cycle writes exactly one summary row.**
    The row must record: which agents ran, which were skipped, why, duration per agent,
    and the overall outcome.

---

## Deployment

47. **Never rely on the local filesystem for anything that must survive a restart.**
    Hosting filesystems are ephemeral. All persistent state lives in the hosted database.
    Logs, temp files, and build artefacts may go to disk; job state, listings, and scores
    must not.

48. **Every secret comes from an environment variable read in one place.**
    No secret is ever committed, printed, logged, or shown in an error message or
    traceback. If a required variable is missing, raise with a clear message naming the
    variable — never expose the value itself.

49. **The scheduled job and the dashboard are separate processes that share only the
    database.**
    The dashboard never runs a cycle; the scheduler never serves a page. They must be
    independently deployable and independently restartable.

50. **The deployed app must start and render even when the database is empty,
    unreachable, or mid-migration.**
    It shows a clear status message instead of a stack trace. A stranger must never see
    a traceback. Every database call in the dashboard is wrapped so a failure degrades
    gracefully to an empty state with a visible warning.

51. **The scheduled job is idempotent and safe to run twice.**
    Running it twice in the same window must produce the same database state as running
    it once. It must have a hard timeout and stay inside free-tier limits on every
    resource it touches (API calls, database connections, rows written).
