"""Fleet-common per-unit merge-idle heartbeat producer (task 2395, α of the
orchestrator fleet-redeploy PRD).

Every orchestrator unit writes a tiny JSON heartbeat to a fleet-common
directory, keyed by its own ``ORCH_UNIT``, on each run-loop tick.  This
module owns the on-disk contract (directory resolution, payload shape, and
the atomic writer) so producer (``Harness._write_merge_heartbeat``, this
task) and future readers — γ (the drain gate) and ε (``--report``'s
merge-idle column) — agree on a single definition with no duplicated
parsing logic.

Kept dependency-free of ``Harness``: every function here is a pure/static
helper, independently unit-testable without constructing a Harness.
"""

from __future__ import annotations

import json
import os
from collections.abc import Mapping
from pathlib import Path
from typing import Any

# Fleet-common heartbeat directory default.  Matches the stdlib watchdog's
# hardcoded REPO_DIR (scripts/orchestrator-watchdog.py:70) so this producer
# and the future reader (γ) agree with zero config coupling — all six units
# run with WorkingDirectory=/home/leo/src/dark-factory regardless of their
# own (possibly foreign) config.project_root.
DEFAULT_FLEET_DIR = Path('/home/leo/src/dark-factory/data/fleet')

# Deterministic fallback filename stem for an empty/unresolved unit, so a
# heartbeat is never written to a file literally named ``.json``.
_UNKNOWN_UNIT_STEM = 'unknown-unit'


def resolve_fleet_dir(env: Mapping[str, str] | None = None) -> Path:
    """Resolve the fleet-common heartbeat directory.

    Returns ``Path(env['ORCH_FLEET_DIR'])`` when that key is present and
    non-empty, else ``DEFAULT_FLEET_DIR``.  ``env`` defaults to
    ``os.environ`` so production call sites need no arguments while tests
    can inject an explicit mapping (or monkeypatch ``os.environ``).
    """
    if env is None:
        env = os.environ
    override = env.get('ORCH_FLEET_DIR', '')
    if override:
        return Path(override)
    return DEFAULT_FLEET_DIR
