#!/usr/bin/env python3
"""STDLIB-ONLY reader for α's (task 2395) per-unit merge-idle heartbeat.

Consumed by γ's drain gate in restart-all-orchestrators.sh (task 2397).
Deliberately does NOT import the `orchestrator` package, so this module
(and its CLI) runs in a deploy environment without the orchestrator venv
importable -- mirroring the stdlib watchdog's decoupled heartbeat read
(plans/orchestrator-fleet-redeploy-prd.md decision 8).

On-disk contract mirrored from orchestrator/src/orchestrator/fleet_heartbeat.py:
    {unit, merge_idle: bool, depth: int, queue_empty: bool, ts_epoch: float}
plus, from a unit running task-5371 code, two further keys:
    drain: null | {requested_ts: int|null, admission_halted: bool,
                   refused: null|str}
    verifies_in_flight: [{task_id, host, kind, started_ts, deadline_ts}, ...]
A heartbeat without ``verifies_in_flight`` comes from a pre-5371 unit; see
``classify`` for how that is read.

CLI usage (called by restart-all-orchestrators.sh's drain_gate):
    python3 scripts/drain_check.py --unit <unit> [--fleet-dir DIR]
        [--fresh-window SECS] [--now EPOCH] [--drain-requested-ts INT]
Prints exactly one of idle/busy/verifying/overdue/refused/stale/absent to
stdout and exits 0. Without --drain-requested-ts only idle/busy/stale/absent
can appear.
"""

from __future__ import annotations

import argparse
import json
import os
import sys
import time
from collections.abc import Mapping
from pathlib import Path

# Fleet-common heartbeat directory default. Hardcoded mirror of
# orchestrator.fleet_heartbeat.DEFAULT_FLEET_DIR (this module is stdlib-only,
# so it cannot import the canonical producer) -- pinned against silent drift
# by test_drain_check.py's drift test (step-3) and, across all four mirrors
# including the bash one in restart-all-orchestrators.sh, by
# tests/scripts/test_orchestrator_watchdog.py::test_fleet_dir_default_matches_across_tiers
# (task 3799).
#
# ABSOLUTE, deliberately: data/fleet/ is a MACHINE-GLOBAL, CROSS-PROJECT
# rendezvous directory (task 2395's Open-Q2) holding several projects'
# orchestrator heartbeats, living under dark-factory/data/ only because
# dark-factory is the fleet HOST. Making it repo-relative would make every
# worktree read ZERO heartbeats and silently conclude the fleet is absent --
# a fail-soft in the drain gate. Isolate tests by setting ORCH_FLEET_DIR
# (df_pytest_isolation._df_fleet_dir_redirect) instead.
DEFAULT_FLEET_DIR = Path('/home/leo/src/dark-factory/data/fleet')


def resolve_fleet_dir(env: Mapping[str, str] | None = None) -> Path:
    """Resolve the fleet-common heartbeat directory.

    Mirrors orchestrator.fleet_heartbeat.resolve_fleet_dir: returns
    ``Path(env['ORCH_FLEET_DIR'])`` when that key is present and
    non-empty, else ``DEFAULT_FLEET_DIR``. ``env`` defaults to
    ``os.environ`` so the CLI needs no arguments while tests can inject an
    explicit mapping.
    """
    if env is None:
        env = os.environ
    override = env.get('ORCH_FLEET_DIR', '')
    if override:
        return Path(override)
    return DEFAULT_FLEET_DIR


def heartbeat_path(fleet_dir: Path, unit: str) -> Path:
    """Return the on-disk heartbeat path for *unit* within *fleet_dir*.

    Mirrors the filename shape written by
    orchestrator.fleet_heartbeat.write_heartbeat: ``<fleet_dir>/<unit>.json``.
    """
    return Path(fleet_dir) / f'{unit}.json'


def _is_number(value: object) -> bool:
    return isinstance(value, (int, float)) and not isinstance(value, bool)


def _classify_drain_progress(
    heartbeat: Mapping, now: float, drain_requested_ts: int,
) -> str:
    """Verdict for a fresh, new-producer heartbeat under a drain request."""
    drain = heartbeat.get('drain')
    if not isinstance(drain, dict) or drain.get('requested_ts') != drain_requested_ts:
        return 'busy'
    if isinstance(drain.get('refused'), str):
        return 'refused'
    if drain.get('admission_halted') is not True:
        return 'busy'
    verifies = heartbeat['verifies_in_flight']
    if not isinstance(verifies, list):
        return 'busy'
    if not verifies:
        return 'idle'
    if all(
        isinstance(entry, dict)
        and _is_number(entry.get('deadline_ts'))
        and entry['deadline_ts'] <= now
        for entry in verifies
    ):
        return 'overdue'
    return 'verifying'


def classify(
    heartbeat: dict | None,
    now: float,
    fresh_window: float,
    drain_requested_ts: int | None = None,
) -> str:
    """Classify a heartbeat payload.

    Without a drain request (``drain_requested_ts is None``) the vocabulary is
    idle / busy / stale / absent:

    - idle iff the heartbeat is fresh (now - ts_epoch <= fresh_window) AND
      merge_idle is True.
    - busy iff fresh AND not idle -- an ambiguous or missing merge_idle on
      an otherwise-fresh heartbeat is conservatively treated as busy, to
      protect a possibly in-flight merge.
    - stale iff the heartbeat is a well-formed mapping with a numeric
      ts_epoch, but that ts_epoch is older than fresh_window.
    - absent iff heartbeat is None, not a mapping, or ts_epoch is missing
      or not numeric (malformed).

    With a drain request, absent and stale are unchanged, and a fresh
    heartbeat with no ``verifies_in_flight`` key (a unit still running
    pre-5371 code, which cannot see the request) is read exactly as above.
    Otherwise, first match wins:

    - busy: the request is not yet acknowledged (``drain`` is null or echoes
      another requested_ts), or admission is not yet halted, or
      ``verifies_in_flight`` is malformed -- all conservative.
    - refused: the unit examined the request and declined it.
    - idle: acknowledged, admission halted, nothing in flight (drained).
    - overdue: everything still in flight is past its own deadline_ts, so
      the unit's command timeout is about to kill it regardless.
    - verifying: at least one in-flight verify is still inside its deadline
      (an entry with a non-numeric deadline counts as inside it).
    """
    if not isinstance(heartbeat, dict):
        return 'absent'
    ts_epoch = heartbeat.get('ts_epoch')
    if not isinstance(ts_epoch, (int, float)):
        return 'absent'
    if now - ts_epoch > fresh_window:
        return 'stale'
    if drain_requested_ts is not None and 'verifies_in_flight' in heartbeat:
        return _classify_drain_progress(heartbeat, now, drain_requested_ts)
    if heartbeat.get('merge_idle') is True:
        return 'idle'
    return 'busy'


def _read_heartbeat(path: Path) -> dict | None:
    """Read and parse a heartbeat JSON file at *path*.

    Returns None if the file is missing/unreadable (OSError), its contents
    are not valid JSON (ValueError), or the parsed value is not a JSON
    object -- any of these count as "malformed" for classify()'s purposes.
    """
    try:
        text = path.read_text(encoding='utf-8')
    except OSError:
        return None
    try:
        payload = json.loads(text)
    except ValueError:
        return None
    if not isinstance(payload, dict):
        return None
    return payload


def _build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(
        description=(
            'Print the drain-gate verdict '
            '(idle/busy/verifying/overdue/refused/stale/absent) for one '
            "orchestrator unit's merge-idle heartbeat."
        ),
    )
    parser.add_argument(
        '--unit', required=True,
        help='Unit name, e.g. orchestrator-dark-factory.service',
    )
    parser.add_argument(
        '--fleet-dir', type=Path, default=resolve_fleet_dir(),
        help='Fleet-common heartbeat directory (default: resolve_fleet_dir(), '
             'honouring ORCH_FLEET_DIR)',
    )
    parser.add_argument(
        '--fresh-window', type=float, default=120.0,
        help='Freshness window in seconds (default: 120)',
    )
    parser.add_argument(
        '--now', type=float, default=time.time(),
        help='Reference "now" in epoch seconds (default: current time)',
    )
    parser.add_argument(
        '--drain-requested-ts', type=int, default=None,
        help='requested_ts of the drain request the sweep wrote for this '
             'unit; enables the verifying/overdue/refused verdicts (default: '
             'none = merge_idle-only verdicts)',
    )
    return parser


def main(argv: list[str] | None = None) -> int:
    args = _build_parser().parse_args(argv)
    path = heartbeat_path(args.fleet_dir, args.unit)
    heartbeat = _read_heartbeat(path)
    verdict = classify(
        heartbeat, args.now, args.fresh_window, args.drain_requested_ts,
    )
    print(verdict)
    return 0


if __name__ == '__main__':
    sys.exit(main())
