"""
run_cycle.py — entry point.

Usage:
    python run_cycle.py

Exit codes:
    0  ok, partial, or nothing_to_do  — all are successful outcomes
    1  configuration or unrecoverable startup error
"""

import logging
import sys

from edgedash.config import load_config
from edgedash.orchestrator import run_cycle

logging.basicConfig(
    level=logging.WARNING,
    format="%(levelname)s %(name)s: %(message)s",
)

if __name__ == "__main__":
    try:
        config = load_config()
        run_cycle(config)
        # nothing_to_do is a success — no special exit code needed (rule 28)
        sys.exit(0)
    except FileNotFoundError as exc:
        print(f"\n[ERROR] {exc}", file=sys.stderr)
        sys.exit(1)
    except KeyboardInterrupt:
        print("\nInterrupted.", file=sys.stderr)
        sys.exit(0)
