"""
edgedash/agents/verifier.py

Verifier agent — runs all deterministic checks after each scoring cycle.

Reads data from storage (scores, extracted facts, gap snapshot, fetch time),
calls verification.run_all_checks(), and returns an AgentResult.

Rule 34: writes NO data other than its own cycle_log row (handled by the
orchestrator like every other agent). The verdict is carried in the notes.
"""

from __future__ import annotations

import logging
from datetime import datetime, timezone
from typing import Any

from edgedash import storage
from edgedash.agents.base import AgentResult
from edgedash.config import Config
from edgedash.planning import StopConditions
from edgedash.verification import Verdict, run_all_checks

logger = logging.getLogger(__name__)


def _utcnow_dt() -> datetime:
    return datetime.now(timezone.utc)


class Verifier:
    name: str = "Verifier"

    def run(
        self,
        config: Config,
        db_path: str,
        stop: StopConditions = StopConditions(),
    ) -> AgentResult:
        now = _utcnow_dt()

        # --- Gather data from storage (read-only) ---
        scored_rows = storage.get_scored_listings_with_components(db_path)
        scores: list[int] = [
            r["fit_score"] for r in scored_rows if r.get("fit_score") is not None
        ]

        # Extract facts from the extraction cache for scored listings
        facts_list: list[dict[str, Any]] = []
        scored_with_desc = storage.get_scored_listings_with_descriptions(db_path)
        import hashlib, json
        for listing in scored_with_desc:
            desc = (listing.get("description") or "").strip()
            if not desc:
                continue
            h = hashlib.sha256(desc.encode()).hexdigest()[:32]
            facts = storage.get_cached_extraction(db_path, h)
            if facts is not None:
                facts_list.append(facts)

        gaps = storage.get_latest_gap_snapshot(db_path)
        latest_fetch_at = storage.last_fetch_time(db_path)

        # --- Run all checks (pure functions — no I/O inside them) ---
        verdict: Verdict = run_all_checks(
            scores=scores,
            facts_list=facts_list,
            gaps=gaps,
            latest_fetch_at=latest_fetch_at,
            config=config,
            now=now,
        )

        # --- Build a compact notes string ---
        if verdict.passed:
            notes = f"VERDICT: pass — {verdict.summary}"
        else:
            detail = "; ".join(
                f"{c.name} {c.observed} (threshold: {c.threshold})"
                for c in verdict.failed_checks
            )
            notes = f"VERDICT: fail — {detail}"

        status = "ok" if verdict.passed else "failed"

        # Attach the verdict object so the orchestrator can inspect it
        # without parsing the notes string.
        result = AgentResult(
            agent=self.name,
            status=status,
            records_touched=len(scores),
            notes=notes,
        )
        # Stash for the orchestrator's retry logic (not part of the protocol,
        # but avoids re-running checks to read the verdict).
        result._verdict = verdict  # type: ignore[attr-defined]

        return result
