"""
edgedash/storage.py

THE ONLY MODULE ALLOWED TO IMPORT sqlite3 OR psycopg2.

Backend selection
-----------------
  If DATABASE_URL is set in the environment, Postgres is used.
  Otherwise, the module falls back to local SQLite (for offline dev).
  Which backend is active is logged at module import — every time.

  All public function signatures are identical for both backends.
  Dialect differences (SERIAL vs AUTOINCREMENT, %s vs ?, ON CONFLICT,
  boolean storage, timestamp types) are handled inside this module only.

CLI
---
  python -m edgedash.storage --migrate   create/update tables on Postgres
  python -m edgedash.storage --check     report backend, connectivity, row counts
"""

from __future__ import annotations

import hashlib
import json
import logging
import os
from abc import ABC, abstractmethod
from contextlib import contextmanager
from datetime import datetime, timezone
from typing import Any, Generator, Optional

logger = logging.getLogger(__name__)

# ---------------------------------------------------------------------------
# Backend abstraction
# ---------------------------------------------------------------------------

class _Row(dict):
    """Dict subclass that also supports attribute-style access so callers
    using row["key"] and row[0] both work after we normalise results."""


class _Backend(ABC):

    @property
    @abstractmethod
    def name(self) -> str: ...

    @abstractmethod
    def connect(self) -> Any: ...

    @contextmanager
    def tx(self) -> Generator[Any, None, None]:
        conn = self.connect()
        try:
            yield conn
            conn.commit()
        except Exception:
            conn.rollback()
            raise
        finally:
            conn.close()

    @abstractmethod
    def placeholder(self) -> str:
        """Return the parameter placeholder: '?' for SQLite, '%s' for Postgres."""

    def ph(self) -> str:  # short alias
        return self.placeholder()

    @abstractmethod
    def autoincrement_ddl(self) -> str:
        """Return the AUTOINCREMENT keyword for a primary-key integer column."""

    @abstractmethod
    def fetchrow(self, cursor: Any) -> Optional[dict]: ...

    @abstractmethod
    def fetchall(self, cursor: Any) -> list[dict]: ...

    @abstractmethod
    def rowcount(self, cursor: Any) -> int: ...


# ---------------------------------------------------------------------------
# SQLite backend
# ---------------------------------------------------------------------------

class _SQLiteBackend(_Backend):

    def __init__(self, path: str) -> None:
        self._path = path

    @property
    def name(self) -> str:
        return f"sqlite:{self._path}"

    def connect(self) -> Any:
        import sqlite3
        conn = sqlite3.connect(self._path)
        conn.row_factory = sqlite3.Row
        conn.execute("PRAGMA journal_mode=WAL;")
        conn.execute("PRAGMA foreign_keys=ON;")
        return conn

    def placeholder(self) -> str:
        return "?"

    def autoincrement_ddl(self) -> str:
        return "INTEGER PRIMARY KEY AUTOINCREMENT"

    def fetchrow(self, cursor: Any) -> Optional[dict]:
        row = cursor.fetchone()
        return dict(row) if row is not None else None

    def fetchall(self, cursor: Any) -> list[dict]:
        return [dict(r) for r in cursor.fetchall()]

    def rowcount(self, cursor: Any) -> int:
        return cursor.rowcount

    @contextmanager
    def tx(self) -> Generator[Any, None, None]:
        conn = self.connect()
        try:
            yield conn
            conn.commit()
        except Exception:
            conn.rollback()
            raise
        finally:
            conn.close()

    def executescript(self, conn: Any, sql: str) -> None:
        """SQLite has executescript; Postgres does not."""
        conn.executescript(sql)


# ---------------------------------------------------------------------------
# Postgres backend
# ---------------------------------------------------------------------------

class _PostgresBackend(_Backend):

    def __init__(self, dsn: str) -> None:
        self._dsn = dsn

    @property
    def name(self) -> str:
        # Redact password before logging (rule 48)
        import re
        safe = re.sub(r":[^:@]+@", ":***@", self._dsn)
        return f"postgres:{safe}"

    def connect(self) -> Any:
        try:
            import psycopg2
            import psycopg2.extras
        except ImportError as exc:
            raise ImportError(
                "psycopg2 is required for Postgres. "
                "Install it with: pip install psycopg2-binary"
            ) from exc
        conn = psycopg2.connect(self._dsn)
        conn.autocommit = False
        # Use DictCursor so rows behave like dicts
        conn.cursor_factory = psycopg2.extras.RealDictCursor
        return conn

    def placeholder(self) -> str:
        return "%s"

    def autoincrement_ddl(self) -> str:
        return "SERIAL PRIMARY KEY"

    def _normalize_row(self, row: Optional[dict]) -> Optional[dict]:
        if not row:
            return None
        res = dict(row)
        for k, v in res.items():
            if isinstance(v, datetime):
                if v.tzinfo is None:
                    v = v.replace(tzinfo=timezone.utc)
                res[k] = v.isoformat()
        return res

    def fetchrow(self, cursor: Any) -> Optional[dict]:
        row = cursor.fetchone()
        return self._normalize_row(row)

    def fetchall(self, cursor: Any) -> list[dict]:
        rows = cursor.fetchall()
        return [self._normalize_row(r) for r in rows] if rows else []

    def rowcount(self, cursor: Any) -> int:
        return cursor.rowcount

    @contextmanager
    def tx(self) -> Generator[Any, None, None]:
        conn = self.connect()
        try:
            cur = conn.cursor()
            yield cur      # yield cursor, not connection, for Postgres
            conn.commit()
        except Exception:
            conn.rollback()
            raise
        finally:
            conn.close()


# ---------------------------------------------------------------------------
# Backend selection — happens once at module import
# ---------------------------------------------------------------------------

def _select_backend(db_path: str = "edgedash.db") -> _Backend:
    url = os.environ.get("DATABASE_URL", "").strip()
    if url:
        backend: _Backend = _PostgresBackend(url)
        logger.info("[storage] Backend: POSTGRES  (%s)", backend.name)
    else:
        backend = _SQLiteBackend(db_path)
        logger.info("[storage] Backend: SQLite  (%s)", db_path)
    return backend


# ---------------------------------------------------------------------------
# Backend selection — per db_path, cached per process
# ---------------------------------------------------------------------------

_BACKEND_CACHE: dict[str, _Backend] = {}


def _select_backend(db_path: str = "edgedash.db") -> _Backend:
    # EDGEDASH_FORCE_SQLITE=1 lets tests and local dev bypass DATABASE_URL
    if os.environ.get("EDGEDASH_FORCE_SQLITE", "").strip() == "1":
        logger.info("[storage] Backend: SQLite (forced via EDGEDASH_FORCE_SQLITE)")
        return _SQLiteBackend(db_path)
    url = os.environ.get("DATABASE_URL", "").strip()
    if url:
        backend: _Backend = _PostgresBackend(url)
        logger.info("[storage] Backend: POSTGRES  (%s)", backend.name)
    else:
        backend = _SQLiteBackend(db_path)
        logger.info("[storage] Backend: SQLite  (%s)", db_path)
    return backend


def _get_backend(path: str = "edgedash.db") -> _Backend:
    if path not in _BACKEND_CACHE:
        _BACKEND_CACHE[path] = _select_backend(path)
    return _BACKEND_CACHE[path]


def _reset_backend() -> None:
    """Force re-selection on next call. Used by --check, --migrate, and tests."""
    _BACKEND_CACHE.clear()


# Keep the old name for any code that imported it directly
_BACKEND: Optional[_Backend] = None


# ---------------------------------------------------------------------------
# Unified execute helpers
# ---------------------------------------------------------------------------

def _connect(path: str = "edgedash.db") -> Any:
    """Return a raw connection. Callers must close it."""
    return _get_backend(path).connect()


@contextmanager
def _tx(path: str = "edgedash.db") -> Generator[Any, None, None]:
    """Yield a cursor/connection in a transaction; commit/rollback automatically."""
    backend = _get_backend(path)
    if isinstance(backend, _SQLiteBackend):
        # SQLite: yield connection; caller uses conn.execute(...)
        with backend.tx() as conn:
            yield conn
    else:
        # Postgres: yield cursor; caller uses cur.execute(...)
        with backend.tx() as cur:
            yield cur


def _execute(conn_or_cur: Any, sql: str, params: tuple = ()) -> Any:
    """Execute *sql* against conn_or_cur with *params*, return the cursor."""
    if hasattr(conn_or_cur, "execute"):
        res = conn_or_cur.execute(sql, params)
        # sqlite3 returns a cursor from execute(); psycopg2 returns None
        return res if res is not None else conn_or_cur
    raise TypeError(f"Expected connection or cursor, got {type(conn_or_cur)}")


def _fetchone(conn_or_cur: Any, sql: str, params: tuple = ()) -> Optional[dict]:
    cur = _execute(conn_or_cur, sql, params)
    row = cur.fetchone()
    if row is None:
        return None
    return dict(row)


def _fetchall(conn_or_cur: Any, sql: str, params: tuple = ()) -> list[dict]:
    cur = _execute(conn_or_cur, sql, params)
    rows = cur.fetchall()
    return [dict(r) for r in rows] if rows else []


def _adapt_sql(sql: str, path: str) -> str:
    """Swap ? → %s when using Postgres."""
    if isinstance(_get_backend(path), _PostgresBackend):
        return sql.replace("?", "%s")
    return sql


# ---------------------------------------------------------------------------
# DDL  (shared; dialect differences injected at runtime)
# ---------------------------------------------------------------------------

_SQLITE_DDL = """
CREATE TABLE IF NOT EXISTS listings (
    id          TEXT PRIMARY KEY,
    external_id TEXT,
    title       TEXT NOT NULL,
    company     TEXT NOT NULL,
    location    TEXT NOT NULL,
    url         TEXT NOT NULL,
    description TEXT,
    source      TEXT NOT NULL,
    posted_at   TEXT,
    fetched_at  TEXT NOT NULL,
    fit_score   INTEGER,
    fit_reason  TEXT,
    scored_at   TEXT,
    score_components TEXT
);
CREATE TABLE IF NOT EXISTS skill_gaps (
    skill     TEXT PRIMARY KEY,
    frequency INTEGER NOT NULL DEFAULT 1,
    last_seen TEXT NOT NULL
);
CREATE TABLE IF NOT EXISTS cycle_log (
    id              INTEGER PRIMARY KEY AUTOINCREMENT,
    agent           TEXT NOT NULL,
    started_at      TEXT NOT NULL,
    finished_at     TEXT,
    records_touched INTEGER NOT NULL DEFAULT 0,
    status          TEXT NOT NULL,
    notes           TEXT
);
CREATE TABLE IF NOT EXISTS extraction_cache (
    description_hash  TEXT PRIMARY KEY,
    extraction_json   TEXT NOT NULL,
    created_at        TEXT NOT NULL
);
CREATE TABLE IF NOT EXISTS gap_snapshots (
    id             INTEGER PRIMARY KEY AUTOINCREMENT,
    run_at         TEXT NOT NULL,
    skill          TEXT NOT NULL,
    weighted_score REAL NOT NULL,
    raw_frequency  INTEGER NOT NULL,
    sample_size    INTEGER NOT NULL,
    listing_ids    TEXT NOT NULL
);
CREATE TABLE IF NOT EXISTS gap_snapshots_v2 (
    id                INTEGER PRIMARY KEY AUTOINCREMENT,
    run_at            TEXT NOT NULL,
    skill             TEXT NOT NULL,
    opportunity_cost  REAL NOT NULL,
    listings_blocked  INTEGER NOT NULL,
    mean_score        REAL NOT NULL,
    top_score         INTEGER NOT NULL,
    sample_size       INTEGER NOT NULL,
    low_confidence    INTEGER NOT NULL DEFAULT 0,
    also_nice_to_have INTEGER NOT NULL DEFAULT 0,
    example_ids       TEXT NOT NULL
);
CREATE TABLE IF NOT EXISTS query_log (
    id          INTEGER PRIMARY KEY AUTOINCREMENT,
    asked_at    TEXT NOT NULL,
    question    TEXT NOT NULL,
    tool_chosen TEXT,
    params_json TEXT,
    answerable  INTEGER NOT NULL DEFAULT 0,
    duration_ms INTEGER NOT NULL DEFAULT 0
);
"""

_POSTGRES_DDL = """
CREATE TABLE IF NOT EXISTS listings (
    id          TEXT PRIMARY KEY,
    external_id TEXT,
    title       TEXT NOT NULL,
    company     TEXT NOT NULL,
    location    TEXT NOT NULL,
    url         TEXT NOT NULL,
    description TEXT,
    source      TEXT NOT NULL,
    posted_at   TIMESTAMPTZ,
    fetched_at  TIMESTAMPTZ NOT NULL,
    fit_score   INTEGER,
    fit_reason  TEXT,
    scored_at   TIMESTAMPTZ,
    score_components TEXT
);
CREATE TABLE IF NOT EXISTS skill_gaps (
    skill     TEXT PRIMARY KEY,
    frequency INTEGER NOT NULL DEFAULT 1,
    last_seen TIMESTAMPTZ NOT NULL
);
CREATE TABLE IF NOT EXISTS cycle_log (
    id              SERIAL PRIMARY KEY,
    agent           TEXT NOT NULL,
    started_at      TIMESTAMPTZ NOT NULL,
    finished_at     TIMESTAMPTZ,
    records_touched INTEGER NOT NULL DEFAULT 0,
    status          TEXT NOT NULL,
    notes           TEXT
);
CREATE TABLE IF NOT EXISTS extraction_cache (
    description_hash  TEXT PRIMARY KEY,
    extraction_json   TEXT NOT NULL,
    created_at        TIMESTAMPTZ NOT NULL
);
CREATE TABLE IF NOT EXISTS gap_snapshots (
    id             SERIAL PRIMARY KEY,
    run_at         TIMESTAMPTZ NOT NULL,
    skill          TEXT NOT NULL,
    weighted_score REAL NOT NULL,
    raw_frequency  INTEGER NOT NULL,
    sample_size    INTEGER NOT NULL,
    listing_ids    TEXT NOT NULL
);
CREATE TABLE IF NOT EXISTS gap_snapshots_v2 (
    id                SERIAL PRIMARY KEY,
    run_at            TIMESTAMPTZ NOT NULL,
    skill             TEXT NOT NULL,
    opportunity_cost  REAL NOT NULL,
    listings_blocked  INTEGER NOT NULL,
    mean_score        REAL NOT NULL,
    top_score         INTEGER NOT NULL,
    sample_size       INTEGER NOT NULL,
    low_confidence    BOOLEAN NOT NULL DEFAULT FALSE,
    also_nice_to_have BOOLEAN NOT NULL DEFAULT FALSE,
    example_ids       TEXT NOT NULL
);
CREATE TABLE IF NOT EXISTS query_log (
    id          SERIAL PRIMARY KEY,
    asked_at    TIMESTAMPTZ NOT NULL,
    question    TEXT NOT NULL,
    tool_chosen TEXT,
    params_json TEXT,
    answerable  BOOLEAN NOT NULL DEFAULT FALSE,
    duration_ms INTEGER NOT NULL DEFAULT 0
);
"""

_MIGRATIONS = [
    # (table, column, ddl_fragment)
    ("listings", "external_id",      "ALTER TABLE listings ADD COLUMN external_id TEXT"),
    ("listings", "scored_at",        "ALTER TABLE listings ADD COLUMN scored_at TEXT"),
    ("listings", "score_components", "ALTER TABLE listings ADD COLUMN score_components TEXT"),
]


# ---------------------------------------------------------------------------
# Schema init
# ---------------------------------------------------------------------------

def init_db(path: str = "edgedash.db") -> None:
    """Create all tables and apply pending migrations. Safe to call repeatedly."""
    backend = _get_backend(path)

    if isinstance(backend, _SQLiteBackend):
        with backend.tx() as conn:
            conn.executescript(_SQLITE_DDL)
            # SQLite column migrations
            cols = {row[1] for row in conn.execute("PRAGMA table_info(listings)").fetchall()}
            for _tbl, col, ddl in _MIGRATIONS:
                if col not in cols:
                    conn.execute(ddl)
                    logger.info("Migration applied: %s", ddl)
    else:
        # Postgres: run each statement separately
        conn = backend.connect()
        try:
            cur = conn.cursor()
            for stmt in _POSTGRES_DDL.strip().split(";"):
                stmt = stmt.strip()
                if stmt:
                    cur.execute(stmt)
            # Column migrations via information_schema
            for tbl, col, ddl in _MIGRATIONS:
                cur.execute(
                    "SELECT column_name FROM information_schema.columns "
                    "WHERE table_name=%s AND column_name=%s",
                    (tbl, col),
                )
                if not cur.fetchone():
                    cur.execute(ddl)
                    logger.info("Postgres migration applied: %s", ddl)
            conn.commit()
        except Exception:
            conn.rollback()
            raise
        finally:
            conn.close()

    logger.info("[storage] init_db complete (%s)", backend.name)


# ---------------------------------------------------------------------------
# Stable ID
# ---------------------------------------------------------------------------

def make_listing_id(source: str, url: str) -> str:
    """Return a stable 32-char SHA-256 hex digest from source + url."""
    payload = f"{source}::{url}".encode("utf-8")
    return hashlib.sha256(payload).hexdigest()[:32]


# ---------------------------------------------------------------------------
# Listings
# ---------------------------------------------------------------------------

def upsert_listings(path: str, rows: list[dict]) -> int:
    if not rows:
        return 0

    now = _utcnow()
    inserted = 0
    backend = _get_backend(path)
    p = backend.ph()

    with _tx(path) as cx:
        for row in rows:
            lid = make_listing_id(row["source"], row["url"])
            if isinstance(backend, _SQLiteBackend):
                sql = (
                    f"INSERT OR IGNORE INTO listings "
                    f"(id,external_id,title,company,location,url,description,"
                    f"source,posted_at,fetched_at) "
                    f"VALUES ({p},{p},{p},{p},{p},{p},{p},{p},{p},{p})"
                )
            else:
                sql = (
                    f"INSERT INTO listings "
                    f"(id,external_id,title,company,location,url,description,"
                    f"source,posted_at,fetched_at) "
                    f"VALUES ({p},{p},{p},{p},{p},{p},{p},{p},{p},{p}) "
                    f"ON CONFLICT (id) DO NOTHING"
                )
            cur = _execute(cx, sql, (
                lid,
                row.get("external_id"),
                row["title"],
                row["company"],
                row["location"],
                row["url"],
                row.get("description", ""),
                row["source"],
                row.get("posted_at"),
                row.get("fetched_at", now),
            ))
            inserted += backend.rowcount(cur)

    return inserted


def count_unscored(path: str) -> int:
    with _tx(path) as cx:
        cur = _execute(cx, "SELECT COUNT(*) FROM listings WHERE fit_score IS NULL")
        row = cur.fetchone()
        return list(row.values())[0] if row else 0


def get_unscored_listings(path: str, limit: int) -> list[dict]:
    p = _get_backend(path).ph()
    with _tx(path) as cx:
        return _fetchall(
            cx,
            f"SELECT * FROM listings WHERE fit_score IS NULL "
            f"ORDER BY fetched_at ASC LIMIT {p}",
            (limit,),
        )


def save_score(path: str, listing_id: str, score: int, reason: str, components: dict) -> None:
    p = _get_backend(path).ph()
    with _tx(path) as cx:
        _execute(
            cx,
            f"UPDATE listings SET fit_score={p}, fit_reason={p}, "
            f"score_components={p}, scored_at={p} WHERE id={p}",
            (score, reason, json.dumps(components), _utcnow(), listing_id),
        )


def null_scores(path: str, listing_ids: list[str]) -> None:
    if not listing_ids:
        return
    p = _get_backend(path).ph()
    sql = (
        f"UPDATE listings SET fit_score=NULL, fit_reason=NULL, "
        f"score_components=NULL, scored_at=NULL WHERE id={p}"
    )
    with _tx(path) as cx:
        for lid in listing_ids:
            _execute(cx, sql, (lid,))


def last_fetch_time(path: str) -> Optional[str]:
    with _tx(path) as cx:
        row = _fetchone(cx, "SELECT MAX(fetched_at) AS v FROM listings")
        return row["v"] if row else None


def last_scored_at(path: str) -> Optional[str]:
    with _tx(path) as cx:
        row = _fetchone(cx, "SELECT MAX(scored_at) AS v FROM listings")
        return row["v"] if row else None


def last_gap_run_at(path: str) -> Optional[str]:
    with _tx(path) as cx:
        row = _fetchone(cx, "SELECT MAX(run_at) AS v FROM gap_snapshots_v2")
        return row["v"] if row else None


def last_cycle_summary(path: str) -> Optional[dict]:
    p = _get_backend(path).ph()
    with _tx(path) as cx:
        return _fetchone(
            cx,
            f"SELECT agent, started_at, finished_at, status, notes "
            f"FROM cycle_log WHERE agent={p} ORDER BY id DESC LIMIT 1",
            ("cycle_summary",),
        )


def get_listings(path: str, limit: int = 100, min_score: Optional[int] = None) -> list[dict]:
    p = _get_backend(path).ph()
    with _tx(path) as cx:
        if min_score is not None:
            return _fetchall(
                cx,
                f"SELECT * FROM listings WHERE fit_score >= {p} "
                f"ORDER BY fit_score DESC LIMIT {p}",
                (min_score, limit),
            )
        return _fetchall(
            cx,
            f"SELECT * FROM listings ORDER BY fetched_at DESC LIMIT {p}",
            (limit,),
        )


def get_scored_listings_with_components(path: str) -> list[dict]:
    with _tx(path) as cx:
        return _fetchall(
            cx,
            "SELECT id, fit_score, fit_reason, score_components FROM listings "
            "WHERE fit_score IS NOT NULL AND score_components IS NOT NULL",
        )


def get_scored_listings_with_descriptions(path: str) -> list[dict]:
    with _tx(path) as cx:
        return _fetchall(
            cx,
            "SELECT id, fit_score, description FROM listings "
            "WHERE fit_score IS NOT NULL "
            "AND description IS NOT NULL AND description != ''",
        )


# ---------------------------------------------------------------------------
# Cycle log
# ---------------------------------------------------------------------------

def log_cycle(
    path: str,
    agent: str,
    started_at: str,
    finished_at: str,
    records_touched: int,
    status: str,
    notes: Optional[str] = None,
) -> None:
    p = _get_backend(path).ph()
    with _tx(path) as cx:
        _execute(
            cx,
            f"INSERT INTO cycle_log "
            f"(agent,started_at,finished_at,records_touched,status,notes) "
            f"VALUES ({p},{p},{p},{p},{p},{p})",
            (agent, started_at, finished_at, records_touched, status, notes),
        )


# ---------------------------------------------------------------------------
# Extraction cache
# ---------------------------------------------------------------------------

def get_cached_extraction(path: str, description_hash: str) -> Optional[dict]:
    p = _get_backend(path).ph()
    with _tx(path) as cx:
        row = _fetchone(
            cx,
            f"SELECT extraction_json FROM extraction_cache WHERE description_hash={p}",
            (description_hash,),
        )
    if row is None:
        return None
    return json.loads(row["extraction_json"])


def set_cached_extraction(path: str, description_hash: str, extraction: dict) -> None:
    p = _get_backend(path).ph()
    backend = _get_backend(path)
    if isinstance(backend, _SQLiteBackend):
        sql = (
            f"INSERT OR REPLACE INTO extraction_cache "
            f"(description_hash, extraction_json, created_at) VALUES ({p},{p},{p})"
        )
    else:
        sql = (
            f"INSERT INTO extraction_cache "
            f"(description_hash, extraction_json, created_at) VALUES ({p},{p},{p}) "
            f"ON CONFLICT (description_hash) DO UPDATE SET "
            f"extraction_json=EXCLUDED.extraction_json, created_at=EXCLUDED.created_at"
        )
    with _tx(path) as cx:
        _execute(cx, sql, (description_hash, json.dumps(extraction), _utcnow()))


def delete_cached_extraction(path: str, description_hash: str) -> None:
    p = _get_backend(path).ph()
    with _tx(path) as cx:
        _execute(
            cx,
            f"DELETE FROM extraction_cache WHERE description_hash={p}",
            (description_hash,),
        )


# ---------------------------------------------------------------------------
# Gap snapshots
# ---------------------------------------------------------------------------

def save_gap_snapshot(path: str, run_at: str, gaps: list[dict]) -> None:
    """Legacy v1 snapshot — kept for compatibility."""
    p = _get_backend(path).ph()
    with _tx(path) as cx:
        for gap in gaps:
            _execute(
                cx,
                f"INSERT INTO gap_snapshots "
                f"(run_at,skill,weighted_score,raw_frequency,sample_size,listing_ids) "
                f"VALUES ({p},{p},{p},{p},{p},{p})",
                (run_at, gap["skill"], gap.get("weighted_score", 0),
                 gap.get("raw_frequency", 0), gap.get("sample_size", 0),
                 json.dumps(gap.get("listing_ids", []))),
            )


def save_gap_snapshot_v2(path: str, run_at: str, gaps: list[dict]) -> None:
    p = _get_backend(path).ph()
    with _tx(path) as cx:
        for gap in gaps:
            _execute(
                cx,
                f"INSERT INTO gap_snapshots_v2 "
                f"(run_at,skill,opportunity_cost,listings_blocked,mean_score,"
                f"top_score,sample_size,low_confidence,also_nice_to_have,example_ids) "
                f"VALUES ({p},{p},{p},{p},{p},{p},{p},{p},{p},{p})",
                (
                    run_at, gap["skill"], gap["opportunity_cost"],
                    gap["listings_blocked"], gap["mean_score"], gap["top_score"],
                    gap["sample_size"], bool(gap.get("low_confidence", False)),
                    bool(gap.get("also_nice_to_have", False)), json.dumps(gap["example_ids"]),
                ),
            )


def get_latest_gap_snapshot(path: str) -> list[dict]:
    p = _get_backend(path).ph()
    with _tx(path) as cx:
        row = _fetchone(cx, "SELECT MAX(run_at) AS v FROM gap_snapshots_v2")
        run_at = row["v"] if row else None
        if not run_at:
            return []
        rows = _fetchall(
            cx,
            f"SELECT skill, opportunity_cost, listings_blocked, mean_score, "
            f"top_score, sample_size, low_confidence, also_nice_to_have, example_ids "
            f"FROM gap_snapshots_v2 WHERE run_at={p} "
            f"ORDER BY opportunity_cost DESC",
            (run_at,),
        )
    for r in rows:
        r["example_ids"] = json.loads(r["example_ids"])
        r["low_confidence"] = bool(r["low_confidence"])
    return rows


def get_gap_trend_data(path: str) -> dict:
    p = _get_backend(path).ph()
    with _tx(path) as cx:
        run_at_rows = _fetchall(
            cx, "SELECT DISTINCT run_at FROM gap_snapshots_v2 ORDER BY run_at"
        )
        if not run_at_rows:
            return {"snapshot_count": 0, "earliest_at": None, "latest_at": None,
                    "earliest": {}, "latest": {}, "latest_top10": []}

        run_ats = [r["run_at"] for r in run_at_rows]
        earliest_at, latest_at = run_ats[0], run_ats[-1]

        def _load(ra: str) -> dict[str, float]:
            rs = _fetchall(
                cx,
                f"SELECT skill, opportunity_cost FROM gap_snapshots_v2 WHERE run_at={p}",
                (ra,),
            )
            return {r["skill"]: r["opportunity_cost"] for r in rs}

        earliest_map = _load(earliest_at)
        latest_map   = _load(latest_at)

        top10 = _fetchall(
            cx,
            f"SELECT skill, opportunity_cost, listings_blocked, mean_score, "
            f"top_score, sample_size, low_confidence "
            f"FROM gap_snapshots_v2 WHERE run_at={p} "
            f"ORDER BY opportunity_cost DESC LIMIT 10",
            (latest_at,),
        )
        for r in top10:
            r["low_confidence"] = bool(r["low_confidence"])

    return {
        "snapshot_count": len(run_ats),
        "earliest_at": earliest_at, "latest_at": latest_at,
        "earliest": earliest_map, "latest": latest_map, "latest_top10": top10,
    }


# ---------------------------------------------------------------------------
# Query log
# ---------------------------------------------------------------------------

def log_query(
    path: str,
    question: str,
    tool_chosen: Optional[str],
    params_json: str,
    answerable: bool,
    duration_ms: int,
) -> None:
    p = _get_backend(path).ph()
    with _tx(path) as cx:
        _execute(
            cx,
            f"INSERT INTO query_log "
            f"(asked_at,question,tool_chosen,params_json,answerable,duration_ms) "
            f"VALUES ({p},{p},{p},{p},{p},{p})",
            (_utcnow(), question, tool_chosen, params_json,
             bool(answerable), duration_ms),
        )


# ---------------------------------------------------------------------------
# Cycle verdict (rule 38)
# ---------------------------------------------------------------------------

def get_last_passing_cycle(path: str) -> Optional[dict]:
    p = _get_backend(path).ph()
    with _tx(path) as cx:
        return _fetchone(
            cx,
            f"SELECT agent, started_at, finished_at, records_touched, notes "
            f"FROM cycle_log WHERE agent={p} AND status={p} "
            f"ORDER BY id DESC LIMIT 1",
            ("cycle_summary", "ok"),
        )


# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------

def _utcnow() -> str:
    return datetime.now(timezone.utc).isoformat()


# ---------------------------------------------------------------------------
# CLI: --migrate and --check
# ---------------------------------------------------------------------------

def _cli_migrate() -> None:
    """Create / update all tables. Safe to run on an empty or existing DB."""
    logging.basicConfig(level=logging.INFO, format="%(levelname)s %(message)s")
    _reset_backend()
    from edgedash.config import load_config
    cfg = load_config()
    init_db(cfg.db_path)
    backend = _get_backend(cfg.db_path)
    print(f"Migration complete on {backend.name}")


def _cli_check() -> None:
    """Print backend, connectivity, and row counts per table."""
    logging.basicConfig(level=logging.WARNING)
    _reset_backend()
    from edgedash.config import load_config
    cfg = load_config()
    path = cfg.db_path
    backend = _get_backend(path)

    print(f"Backend  : {backend.name}")

    tables = [
        "listings", "cycle_log", "extraction_cache",
        "gap_snapshots_v2", "query_log", "skill_gaps",
    ]
    try:
        init_db(path)
        p = backend.ph()
        with _tx(path) as cx:
            for tbl in tables:
                row = _fetchone(cx, f"SELECT COUNT(*) AS n FROM {tbl}")
                count = row["n"] if row else "?"
                print(f"  {tbl:<28} {count:>8} rows")
        print("Connection: OK")
    except Exception as exc:
        print(f"Connection: FAILED — {exc}")


if __name__ == "__main__":
    import argparse
    parser = argparse.ArgumentParser(description="EdgeDash storage CLI")
    parser.add_argument("--migrate", action="store_true",
                        help="Create/update tables on the configured database")
    parser.add_argument("--check",   action="store_true",
                        help="Report backend, connectivity, and row counts")
    args = parser.parse_args()
    if args.migrate:
        _cli_migrate()
    elif args.check:
        _cli_check()
    else:
        parser.print_help()
