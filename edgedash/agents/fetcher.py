"""
edgedash/agents/fetcher.py

Real Fetcher agent. Iterates over the sources listed in config.sources,
calls each one, handles per-source failures without killing the cycle
(steering rule 12), then upserts all rows via storage.

ID computation delegates entirely to storage.make_listing_id — there is
no second implementation here.
"""

from __future__ import annotations

import logging
from datetime import datetime, timezone

from edgedash import storage
from edgedash.agents.base import AgentResult
from edgedash.config import Config
from edgedash.planning import StopConditions
from edgedash.sources.base import SOURCES
# Import all concrete source modules so their @register decorators fire.
import edgedash.sources.arbeitnow  # noqa: F401

logger = logging.getLogger(__name__)


def _utcnow() -> str:
    return datetime.now(timezone.utc).isoformat()


class Fetcher:
    name: str = "Fetcher"

    def run(
        self,
        config: Config,
        db_path: str,
        stop: StopConditions = StopConditions(),
    ) -> AgentResult:
        # Respect stop conditions set by the Orchestrator (rule 29)
        max_listings = stop.max_listings or 0   # 0 = no cap
        all_rows: list[dict] = []
        source_summaries: list[str] = []

        for source_name in config.sources:
            # Hard cap across all sources combined
            if max_listings and len(all_rows) >= max_listings:
                logger.info(
                    "Fetcher: max_listings=%d reached — stopping before source '%s'",
                    max_listings, source_name,
                )
                break

            if source_name not in SOURCES:
                msg = f"Source '{source_name}' not found in registry — skipping."
                logger.error(msg)
                storage.log_cycle(
                    path=db_path,
                    agent=f"Fetcher/{source_name}",
                    started_at=_utcnow(),
                    finished_at=_utcnow(),
                    records_touched=0,
                    status="failed",
                    notes=msg,
                )
                source_summaries.append(f"{source_name}: FAILED (not registered)")
                continue

            source_instance = SOURCES[source_name]()
            started_at = _utcnow()

            try:
                rows = source_instance.fetch(config)
            except Exception as exc:  # per-source isolation (steering rule 12)
                finished_at = _utcnow()
                err_msg = f"{type(exc).__name__}: {exc}"
                logger.error("Source '%s' raised an exception: %s", source_name, err_msg)
                print(f"  ⚠  [{source_name}] FAILED — {err_msg}")
                storage.log_cycle(
                    path=db_path,
                    agent=f"Fetcher/{source_name}",
                    started_at=started_at,
                    finished_at=finished_at,
                    records_touched=0,
                    status="failed",
                    notes=err_msg,
                )
                source_summaries.append(f"{source_name}: FAILED ({type(exc).__name__})")
                continue

            finished_at = _utcnow()
            row_count = len(rows)
            logger.info("Source '%s' returned %d rows.", source_name, row_count)
            storage.log_cycle(
                path=db_path,
                agent=f"Fetcher/{source_name}",
                started_at=started_at,
                finished_at=finished_at,
                records_touched=row_count,
                status="ok",
                notes=f"{row_count} rows fetched",
            )
            all_rows.extend(rows)
            # Partial summary — new count filled in after upsert below
            source_summaries.append((source_name, rows))

        # Upsert all collected rows and resolve per-source new counts.
        final_summaries: list[str] = []
        deferred: list[tuple[str, list[dict]]] = []

        for item in source_summaries:
            if isinstance(item, str):
                # Already a formatted failure string
                final_summaries.append(item)
            else:
                deferred.append(item)

        for source_name, rows in deferred:
            new_count = storage.upsert_listings(db_path, rows)
            final_summaries.append(
                f"{source_name}: {len(rows)} rows ({new_count} new)"
            )

        total_rows = len(all_rows)
        notes = " | ".join(final_summaries) if final_summaries else "no sources ran"

        return AgentResult(
            agent=self.name,
            status="ok",
            records_touched=total_rows,
            notes=notes,
        )
