#!/usr/bin/env python3
"""STDLIB-ONLY reader for α's (task 2395) per-unit merge-idle heartbeat.

Consumed by γ's drain gate in restart-all-orchestrators.sh (task 2397).
Deliberately does NOT import the `orchestrator` package, so this module
(and its CLI) runs in a deploy environment without the orchestrator venv
importable -- mirroring the stdlib watchdog's decoupled heartbeat read
(plans/orchestrator-fleet-redeploy-prd.md decision 8).

On-disk contract mirrored from orchestrator/src/orchestrator/fleet_heartbeat.py:
    {unit, merge_idle: bool, depth: int, queue_empty: bool, ts_epoch: float}

CLI usage (called by restart-all-orchestrators.sh's drain_gate):
    python3 scripts/drain_check.py --unit <unit> [--fleet-dir DIR]
        [--fresh-window SECS] [--now EPOCH]
Prints exactly one of idle/busy/stale/absent to stdout and exits 0.
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
# orchestrator.fleet_heartbeat.DEFAULT_FLEET_DIR -- pinned against silent
# drift by test_drain_check.py's drift test (step-3).
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


def classify(heartbeat: dict | None, now: float, fresh_window: float) -> str:
    """Classify a heartbeat payload into idle / busy / stale / absent.

    - idle iff the heartbeat is fresh (now - ts_epoch <= fresh_window) AND
      merge_idle is True.
    - busy iff fresh AND not idle -- an ambiguous or missing merge_idle on
      an otherwise-fresh heartbeat is conservatively treated as busy, to
      protect a possibly in-flight merge.
    - stale iff the heartbeat is a well-formed mapping with a numeric
      ts_epoch, but that ts_epoch is older than fresh_window.
    - absent iff heartbeat is None, not a mapping, or ts_epoch is missing
      or not numeric (malformed).
    """
    if not isinstance(heartbeat, dict):
        return 'absent'
    ts_epoch = heartbeat.get('ts_epoch')
    if not isinstance(ts_epoch, (int, float)):
        return 'absent'
    if now - ts_epoch > fresh_window:
        return 'stale'
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
            'Print the drain-gate verdict (idle/busy/stale/absent) for one '
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
    return parser


def main(argv: list[str] | None = None) -> int:
    args = _build_parser().parse_args(argv)
    path = heartbeat_path(args.fleet_dir, args.unit)
    heartbeat = _read_heartbeat(path)
    verdict = classify(heartbeat, args.now, args.fresh_window)
    print(verdict)
    return 0


if __name__ == '__main__':
    sys.exit(main())
