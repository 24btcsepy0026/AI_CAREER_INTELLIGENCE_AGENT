"""
edgedash/agents/placeholders.py

Placeholder agents for Scorer and GapAnalyzer.
These log "not implemented yet" and return a skipped result.
Replace each class with the real implementation when ready — the registry
in orchestrator.py is the only line that needs changing.
"""

from __future__ import annotations

from edgedash.agents.base import Agent, AgentResult
from edgedash.config import Config
from edgedash.planning import StopConditions


class PlaceholderScorer:
    name: str = "Scorer"

    def run(
        self,
        config: Config,
        db_path: str,
        stop: StopConditions = StopConditions(),
    ) -> AgentResult:
        return AgentResult(
            agent=self.name,
            status="ok",
            records_touched=0,
            notes="NOT IMPLEMENTED YET — skipped.",
        )


class PlaceholderGapAnalyzer:
    name: str = "GapAnalyzer"

    def run(
        self,
        config: Config,
        db_path: str,
        stop: StopConditions = StopConditions(),
    ) -> AgentResult:
        return AgentResult(
            agent=self.name,
            status="ok",
            records_touched=0,
            notes="NOT IMPLEMENTED YET — skipped.",
        )
