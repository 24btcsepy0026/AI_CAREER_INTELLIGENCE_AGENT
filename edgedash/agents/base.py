"""
edgedash/agents/base.py

Defines the Agent protocol and AgentResult dataclass.
Every agent in the system must satisfy the Agent protocol.
"""

from __future__ import annotations

from dataclasses import dataclass
from typing import Protocol, runtime_checkable

from edgedash.config import Config
from edgedash.planning import StopConditions


@dataclass
class AgentResult:
    agent: str
    status: str          # "ok" | "failed"
    records_touched: int
    notes: str


@runtime_checkable
class Agent(Protocol):
    """Every agent exposes a name and a run method. Nothing more."""

    name: str

    def run(
        self,
        config: Config,
        db_path: str,
        stop: StopConditions,
    ) -> AgentResult:
        ...
