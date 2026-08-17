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

from shared import safe_io

# Fleet-common heartbeat directory default.  Matches the stdlib watchdog's
# hardcoded REPO_DIR (scripts/orchestrator-watchdog.py:70) so this producer
# and the future reader (γ) agree with zero config coupling — all six units
# run with WorkingDirectory=/home/leo/src/dark-factory regardless of their
# own (possibly foreign) config.project_root.
DEFAULT_FLEET_DIR = Path('/home/leo/src/dark-factory/data/fleet')


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

    Delegates to :func:`shared.safe_io.atomic_write_text` (task 3223), which
    creates missing parent directories and does the tmp + ``os.replace`` dance
    so concurrent readers (γ, ε) never observe a partial write.

    An empty or whitespace-only ``unit`` raises :class:`ValueError` BEFORE any
    filesystem work, so a writer that reached this point without a real unit
    name creates nothing at all (task 3951).  It used to fall back to a
    deterministic ``unknown-unit.json``; nothing ever read that file
    (``scripts/drain_check.py`` addresses heartbeats by name via
    ``heartbeat_path(fleet_dir, unit)`` and never enumerates the directory), so
    the fallback bought no consumer anything while turning a real
    misconfiguration into a plausible-looking file in a machine-global,
    cross-project directory.  A whitespace-only unit is REJECTED, never
    ``.strip()``ed into a "fixed" name: silently repairing a malformed unit name
    is the same silent corruption this guard exists to end.

    ``mode`` is deliberately left at the helper's umask default rather than
    narrowed: these heartbeats are read by other processes. Exceptions
    propagate — this site has no fail-open boundary.  The sole production
    caller, ``Harness._write_merge_heartbeat``, has its own ``except
    Exception`` (a heartbeat write must never stop the run loop), so in
    production this raise surfaces as a logged WARNING rather than a crash —
    while the corruption is prevented either way.

    Returns the final on-disk ``Path``.
    """
    fleet_dir = Path(fleet_dir)
    if not unit or not unit.strip():
        raise ValueError(
            f'refusing to write a heartbeat with no unit name (unit={unit!r}) '
            f'into {fleet_dir}. ORCH_UNIT is unset or blank in the process '
            'calling Harness._write_merge_heartbeat; set it to this '
            'orchestrator\'s systemd unit (e.g. orchestrator-dark-factory.service). '
            'Writing anyway would put an unattributable file in a '
            'machine-global, cross-project fleet directory.'
        )
    path = fleet_dir / f'{unit}.json'
    safe_io.atomic_write_text(
        path,
        json.dumps(payload),
        encoding='utf-8',
        mkdir=True,
    )
    return path
