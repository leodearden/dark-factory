"""Tests for scripts/drain_check.py — the STDLIB-ONLY reader of α's (task
2395) per-unit merge-idle heartbeat, consumed by γ's drain gate in
restart-all-orchestrators.sh (task 2397).

step-1: pure classify(heartbeat, now, fresh_window) taxonomy -- idle / busy
/ stale / absent. No filesystem or subprocess I/O in this module.
"""
from __future__ import annotations

import json
import os
import subprocess
import time
from pathlib import Path

from drain_check import classify, heartbeat_path, resolve_fleet_dir

FRESH_WINDOW = 120.0
NOW = 1_000_000.0

UNIT = "orchestrator-dark-factory.service"

SCRIPT = Path(__file__).parent.parent / "drain_check.py"


def _heartbeat(**overrides):
    payload = {
        "unit": UNIT,
        "merge_idle": True,
        "depth": 0,
        "queue_empty": True,
        "ts_epoch": NOW,
    }
    payload.update(overrides)
    return payload


def test_fresh_merge_idle_true_is_idle():
    heartbeat = _heartbeat(merge_idle=True, ts_epoch=NOW)
    assert classify(heartbeat, NOW, FRESH_WINDOW) == "idle"


def test_fresh_merge_idle_false_is_busy():
    heartbeat = _heartbeat(merge_idle=False, ts_epoch=NOW)
    assert classify(heartbeat, NOW, FRESH_WINDOW) == "busy"


def test_ts_epoch_older_than_fresh_window_is_stale():
    heartbeat = _heartbeat(merge_idle=True, ts_epoch=NOW - FRESH_WINDOW - 1)
    assert classify(heartbeat, NOW, FRESH_WINDOW) == "stale"


def test_stale_even_when_merge_idle_false():
    """Staleness is decided on ts_epoch alone -- an old busy heartbeat is
    still 'stale', not 'busy' (the unknown-grace path handles it, not the
    busy/force-poll path)."""
    heartbeat = _heartbeat(merge_idle=False, ts_epoch=NOW - FRESH_WINDOW - 1)
    assert classify(heartbeat, NOW, FRESH_WINDOW) == "stale"


def test_none_heartbeat_is_absent():
    assert classify(None, NOW, FRESH_WINDOW) == "absent"


def test_missing_ts_epoch_is_absent():
    heartbeat = _heartbeat()
    del heartbeat["ts_epoch"]
    assert classify(heartbeat, NOW, FRESH_WINDOW) == "absent"


def test_non_numeric_ts_epoch_is_absent():
    heartbeat = _heartbeat(ts_epoch="not-a-number")
    assert classify(heartbeat, NOW, FRESH_WINDOW) == "absent"


def test_fresh_missing_merge_idle_is_busy():
    """Conservative: ambiguous/missing merge_idle on an otherwise-fresh
    heartbeat classifies as busy, protecting an in-flight merge."""
    heartbeat = _heartbeat(ts_epoch=NOW)
    del heartbeat["merge_idle"]
    assert classify(heartbeat, NOW, FRESH_WINDOW) == "busy"


def test_fresh_ambiguous_merge_idle_is_busy():
    heartbeat = _heartbeat(merge_idle="yes", ts_epoch=NOW)
    assert classify(heartbeat, NOW, FRESH_WINDOW) == "busy"


def test_exactly_at_fresh_window_boundary_is_still_fresh():
    """now - ts_epoch == fresh_window is fresh (<=, not <)."""
    heartbeat = _heartbeat(merge_idle=True, ts_epoch=NOW - FRESH_WINDOW)
    assert classify(heartbeat, NOW, FRESH_WINDOW) == "idle"


# ---------------------------------------------------------------------------
# step-3: resolve_fleet_dir / heartbeat_path / drift-vs-α test
# ---------------------------------------------------------------------------

def test_resolve_fleet_dir_honours_env_override():
    env = {"ORCH_FLEET_DIR": "/tmp/some-fleet-dir"}
    assert resolve_fleet_dir(env) == Path("/tmp/some-fleet-dir")


def test_resolve_fleet_dir_falls_back_to_default_when_unset():
    assert resolve_fleet_dir({}) == drain_check_default_fleet_dir()


def test_resolve_fleet_dir_falls_back_to_default_when_empty():
    assert resolve_fleet_dir({"ORCH_FLEET_DIR": ""}) == drain_check_default_fleet_dir()


def test_heartbeat_path_joins_fleet_dir_and_unit_json():
    fleet_dir = Path("/tmp/some-fleet-dir")
    assert heartbeat_path(fleet_dir, UNIT) == fleet_dir / f"{UNIT}.json"


def drain_check_default_fleet_dir():
    from drain_check import DEFAULT_FLEET_DIR

    return DEFAULT_FLEET_DIR


def test_default_fleet_dir_matches_orchestrator_fleet_heartbeat():
    """DRIFT GUARD: the stdlib-only mirror in drain_check.py must never
    silently diverge from α's canonical DEFAULT_FLEET_DIR."""
    import drain_check
    import orchestrator.fleet_heartbeat as fleet_heartbeat

    assert drain_check.DEFAULT_FLEET_DIR == fleet_heartbeat.DEFAULT_FLEET_DIR


# ---------------------------------------------------------------------------
# step-5: CLI (argparse) tests -- drive via subprocess.run
# ---------------------------------------------------------------------------

def _write_raw_heartbeat(fleet_dir: Path, unit: str, **overrides):
    fleet_dir.mkdir(parents=True, exist_ok=True)
    payload = {
        "unit": unit,
        "merge_idle": True,
        "depth": 0,
        "queue_empty": True,
        "ts_epoch": NOW,
    }
    payload.update(overrides)
    (fleet_dir / f"{unit}.json").write_text(json.dumps(payload))
    return payload


def _run_cli(*args, env=None):
    full_env = dict(os.environ)
    if env:
        full_env.update(env)
    return subprocess.run(
        ["python3", str(SCRIPT), *args],
        env=full_env,
        capture_output=True,
        text=True,
        timeout=10,
    )


def test_cli_prints_idle_for_fresh_merge_idle_heartbeat(tmp_path):
    _write_raw_heartbeat(tmp_path, UNIT, merge_idle=True, ts_epoch=NOW)

    result = _run_cli(
        "--unit", UNIT,
        "--fleet-dir", str(tmp_path),
        "--fresh-window", str(FRESH_WINDOW),
        "--now", str(NOW),
    )

    assert result.returncode == 0, f"stdout={result.stdout!r} stderr={result.stderr!r}"
    assert result.stdout.strip() == "idle"


def test_cli_prints_busy_for_fresh_merge_idle_false_heartbeat(tmp_path):
    _write_raw_heartbeat(tmp_path, UNIT, merge_idle=False, ts_epoch=NOW)

    result = _run_cli(
        "--unit", UNIT,
        "--fleet-dir", str(tmp_path),
        "--fresh-window", str(FRESH_WINDOW),
        "--now", str(NOW),
    )

    assert result.returncode == 0, f"stdout={result.stdout!r} stderr={result.stderr!r}"
    assert result.stdout.strip() == "busy"


def test_cli_prints_stale_for_old_heartbeat(tmp_path):
    _write_raw_heartbeat(tmp_path, UNIT, merge_idle=True, ts_epoch=NOW - FRESH_WINDOW - 1)

    result = _run_cli(
        "--unit", UNIT,
        "--fleet-dir", str(tmp_path),
        "--fresh-window", str(FRESH_WINDOW),
        "--now", str(NOW),
    )

    assert result.returncode == 0, f"stdout={result.stdout!r} stderr={result.stderr!r}"
    assert result.stdout.strip() == "stale"


def test_cli_prints_absent_for_missing_heartbeat_file(tmp_path):
    result = _run_cli(
        "--unit", UNIT,
        "--fleet-dir", str(tmp_path),
        "--fresh-window", str(FRESH_WINDOW),
        "--now", str(NOW),
    )

    assert result.returncode == 0, f"stdout={result.stdout!r} stderr={result.stderr!r}"
    assert result.stdout.strip() == "absent"


def test_cli_fresh_window_defaults_to_120(tmp_path):
    """Omitting --fresh-window falls back to 120 -- a 100s-old heartbeat
    (< 120) must still read idle."""
    _write_raw_heartbeat(tmp_path, UNIT, merge_idle=True, ts_epoch=NOW - 100)

    result = _run_cli(
        "--unit", UNIT,
        "--fleet-dir", str(tmp_path),
        "--now", str(NOW),
    )

    assert result.returncode == 0, f"stdout={result.stdout!r} stderr={result.stderr!r}"
    assert result.stdout.strip() == "idle"


def test_cli_now_defaults_to_current_time_when_omitted(tmp_path):
    """Omitting --now falls back to the real current time -- a heartbeat
    written far in the past reads stale against a small fresh-window, and
    one written right now reads idle."""
    _write_raw_heartbeat(tmp_path, UNIT, merge_idle=True, ts_epoch=time.time() - 10_000)

    stale_result = _run_cli(
        "--unit", UNIT,
        "--fleet-dir", str(tmp_path),
        "--fresh-window", "5",
    )
    assert stale_result.returncode == 0, (
        f"stdout={stale_result.stdout!r} stderr={stale_result.stderr!r}"
    )
    assert stale_result.stdout.strip() == "stale"

    _write_raw_heartbeat(tmp_path, UNIT, merge_idle=True, ts_epoch=time.time())

    idle_result = _run_cli(
        "--unit", UNIT,
        "--fleet-dir", str(tmp_path),
        "--fresh-window", "60",
    )
    assert idle_result.returncode == 0, (
        f"stdout={idle_result.stdout!r} stderr={idle_result.stderr!r}"
    )
    assert idle_result.stdout.strip() == "idle"


def test_cli_fleet_dir_defaults_via_resolve_fleet_dir(tmp_path):
    """Omitting --fleet-dir falls back to resolve_fleet_dir() -- honouring
    ORCH_FLEET_DIR from the environment."""
    _write_raw_heartbeat(tmp_path, UNIT, merge_idle=True, ts_epoch=NOW)

    result = _run_cli(
        "--unit", UNIT,
        "--fresh-window", str(FRESH_WINDOW),
        "--now", str(NOW),
        env={"ORCH_FLEET_DIR": str(tmp_path)},
    )

    assert result.returncode == 0, f"stdout={result.stdout!r} stderr={result.stderr!r}"
    assert result.stdout.strip() == "idle"
