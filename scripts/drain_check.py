"""STDLIB-ONLY reader for α's (task 2395) per-unit merge-idle heartbeat.

Consumed by γ's drain gate in restart-all-orchestrators.sh (task 2397).
Deliberately does NOT import the `orchestrator` package, so this module
(and its CLI) runs in a deploy environment without the orchestrator venv
importable -- mirroring the stdlib watchdog's decoupled heartbeat read
(plans/orchestrator-fleet-redeploy-prd.md decision 8).

On-disk contract mirrored from orchestrator/src/orchestrator/fleet_heartbeat.py:
    {unit, merge_idle: bool, depth: int, queue_empty: bool, ts_epoch: float}
"""

from __future__ import annotations

from pathlib import Path

# Fleet-common heartbeat directory default. Hardcoded mirror of
# orchestrator.fleet_heartbeat.DEFAULT_FLEET_DIR -- pinned against silent
# drift by test_drain_check.py's drift test (step-3).
DEFAULT_FLEET_DIR = Path('/home/leo/src/dark-factory/data/fleet')


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
