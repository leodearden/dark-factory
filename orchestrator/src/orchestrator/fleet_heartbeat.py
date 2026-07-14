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


def build_heartbeat_payload(
    unit: str,
    merge_idle: bool,
    depth: int,
    queue_empty: bool,
    ts_epoch: float,
) -> dict[str, Any]:
    """Build the on-disk heartbeat payload.

    Returns a dict with exactly the five fields ``{unit, merge_idle, depth,
    queue_empty, ts_epoch}`` in that key order, with values passed through
    unchanged (no coercion) — this is the single definition of the on-disk
    contract shared with the future readers (γ drain gate, ε ``--report``).
    """
    return {
        'unit': unit,
        'merge_idle': merge_idle,
        'depth': depth,
        'queue_empty': queue_empty,
        'ts_epoch': ts_epoch,
    }


def write_heartbeat(fleet_dir: Path, unit: str, payload: Mapping[str, Any]) -> Path:
    """Atomically write *payload* to ``<fleet_dir>/<unit>.json``.

    Mirrors ``Scheduler._write_state_snapshot_raw`` (scheduler.py:4749):
    creates missing parent directories, writes to a ``.json.tmp`` sibling,
    then ``os.replace``s it into place so concurrent readers (γ, ε) never
    observe a partial write. An empty/unresolved ``unit`` falls back to a
    deterministic ``unknown-unit.json`` filename rather than a file
    literally named ``.json``.

    Returns the final on-disk ``Path``.
    """
    fleet_dir = Path(fleet_dir)
    fleet_dir.mkdir(parents=True, exist_ok=True)
    stem = unit if unit else _UNKNOWN_UNIT_STEM
    path = fleet_dir / f'{stem}.json'
    tmp_path = path.with_suffix('.json.tmp')
    tmp_path.write_text(json.dumps(payload), encoding='utf-8')
    os.replace(tmp_path, path)
    return path
