"""
edgedash/sources/base.py

Abstract base class for all job sources plus the global source registry.

Every source must:
  - expose a `name` class attribute (str)
  - implement `fetch(config) -> list[dict]` returning normalised rows
    with exactly the keys defined in NORMALISED_KEYS (steering rule 10)

To register a source, decorate its class with @register — no other
change is needed anywhere else in the codebase (steering rule 9).
"""

from __future__ import annotations

import logging
from abc import ABC, abstractmethod
from typing import Any

from edgedash.config import Config

logger = logging.getLogger(__name__)

# Canonical keys every source MUST return (steering rule 10).
# Missing values → None. Never "" or "N/A".
NORMALISED_KEYS: tuple[str, ...] = (
    "source",
    "external_id",
    "title",
    "company",
    "location",
    "url",
    "description",
    "posted_at",
    "raw",
)

# Global registry: source name → Source subclass
SOURCES: dict[str, type[Source]] = {}


def register(cls: type[Source]) -> type[Source]:
    """Class decorator that adds *cls* to the global SOURCES registry.

    Usage::

        @register
        class MySource(Source):
            name = "my_source"
            ...
    """
    if not hasattr(cls, "name") or not isinstance(cls.name, str):
        raise TypeError(f"{cls.__qualname__} must define a `name: str` class attribute.")
    if cls.name in SOURCES:
        raise ValueError(f"Source name '{cls.name}' is already registered.")
    SOURCES[cls.name] = cls
    logger.debug("Registered source: %s", cls.name)
    return cls


def normalise(row: dict[str, Any], source_name: str) -> dict[str, Any]:
    """Ensure *row* has exactly NORMALISED_KEYS; fill missing keys with None."""
    out: dict[str, Any] = {k: row.get(k) for k in NORMALISED_KEYS}
    # Coerce empty strings / "N/A" to None
    for key, val in out.items():
        if key == "raw":
            continue  # raw can be any type
        if isinstance(val, str) and val.strip() in ("", "N/A", "n/a"):
            out[key] = None
    out["source"] = source_name
    return out


class Source(ABC):
    """Base class for all EdgeDash job sources."""

    name: str  # must be overridden

    @abstractmethod
    def fetch(self, config: Config) -> list[dict[str, Any]]:
        """Fetch jobs and return a list of normalised dicts."""
