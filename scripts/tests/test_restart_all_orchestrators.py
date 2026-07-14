"""Tests for restart-all-orchestrators.sh's `--drain` merge-drain gate
(task 2397, γ of the orchestrator fleet-redeploy PRD).

Drives the script via subprocess against a fake `systemctl` shimmed onto
PATH -- adapted from test_restart_orchestrator.py's single-unit harness,
EXTENDED with `list-units` support and per-unit state so the gate's
per-unit poll loop can be exercised offline. drain_check.py itself is
NOT mocked -- it runs for real against heartbeat JSON files the tests
write directly into a tmp fleet dir (ORCH_FLEET_DIR).
"""
from __future__ import annotations

import json
import os
import subprocess
import time
from pathlib import Path

import pytest

SCRIPT = Path(__file__).parent.parent / "restart-all-orchestrators.sh"
UNIT_R = "orchestrator-reify.service"

FAKE_SYSTEMCTL_SRC = '''#!/usr/bin/env python3
"""Fake `systemctl` for testing restart-all-orchestrators.sh.

Extends test_restart_orchestrator.py's single-unit fake with `list-units`
support and PER-UNIT state. State shape:
    {
      "running_units": [<unit>, ...],
      "units": {<unit>: {"MainPID": int, "ActiveState": str,
                          "ActiveEnterTimestamp": str,
                          "ActiveEnterTimestampMonotonic": int,
                          "scenario": "fresh"|"stale"}},
      "calls": [...],
    }
`list-units` prints one line per entry in running_units (first
whitespace-delimited field is the unit name, matching the real script's
`awk '{print $1}'` parse). `show -p FIELDS UNIT` and `restart UNIT`
operate on state["units"][UNIT]; a unit with no configured scenario
defaults to "stale" (restart is a no-op, mirroring the single-unit fake's
default).
"""
import json
import os
import sys

STATE_PATH = os.environ["FAKE_SYSTEMCTL_STATE"]


def _load():
    with open(STATE_PATH) as f:
        return json.load(f)


def _save(state):
    with open(STATE_PATH, "w") as f:
        json.dump(state, f)


def main(argv):
    args = [a for a in argv[1:] if a != "--user"]
    if not args:
        return 1
    verb, rest = args[0], args[1:]

    state = _load()
    state.setdefault("calls", []).append(argv[1:])

    if verb == "list-units":
        for unit in state.get("running_units", []):
            print(f"{unit} loaded active running Orchestrator")
        _save(state)
        return 0

    if verb == "restart":
        unit = rest[0] if rest else ""
        units = state.setdefault("units", {})
        ustate = units.setdefault(unit, {})
        scenario = ustate.get("scenario", "stale")
        if scenario == "fresh":
            ustate["MainPID"] = ustate.get("MainPID", 1000) + 1
            ustate["ActiveState"] = "active"
            ustate["ActiveEnterTimestampMonotonic"] = (
                ustate.get("ActiveEnterTimestampMonotonic", 0) + 5_000_000
            )
            ustate["ActiveEnterTimestamp"] = "restarted"
        _save(state)
        return 0

    if verb == "show":
        fields = None
        unit = None
        i = 0
        while i < len(rest):
            tok = rest[i]
            if tok == "-p":
                fields = rest[i + 1]
                i += 2
            elif tok.startswith("--property="):
                fields = tok.split("=", 1)[1]
                i += 1
            elif tok.startswith("-"):
                i += 1
            else:
                unit = tok
                i += 1
        ustate = state.get("units", {}).get(unit, {})
        current = {
            "MainPID": str(ustate.get("MainPID", 0)),
            "ActiveState": ustate.get("ActiveState", "active"),
            "ActiveEnterTimestamp": ustate.get("ActiveEnterTimestamp", "baseline"),
            "ActiveEnterTimestampMonotonic": str(ustate.get("ActiveEnterTimestampMonotonic", 0)),
        }
        keys = fields.split(",") if fields else list(current.keys())
        for k in keys:
            print(f"{k}={current.get(k, '')}")
        _save(state)
        return 0

    return 1


if __name__ == "__main__":
    sys.exit(main(sys.argv))
'''


# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------

def _make_fake_systemctl(tmp_path, *, running_units, units=None):
    """Write a fake multi-unit `systemctl` into <tmp_path>/bin/.

    Returns (bin_dir, state_path).
    """
    bin_dir = tmp_path / "bin"
    bin_dir.mkdir()
    fake = bin_dir / "systemctl"
    fake.write_text(FAKE_SYSTEMCTL_SRC)
    fake.chmod(0o755)

    state_path = tmp_path / "systemctl_state.json"
    state_path.write_text(json.dumps({
        "running_units": list(running_units),
        "units": units or {},
        "calls": [],
    }))
    return bin_dir, state_path


def _write_heartbeat(fleet_dir, unit, **overrides):
    fleet_dir.mkdir(parents=True, exist_ok=True)
    payload = {
        "unit": unit,
        "merge_idle": True,
        "depth": 0,
        "queue_empty": True,
        "ts_epoch": time.time(),
    }
    payload.update(overrides)
    (fleet_dir / f"{unit}.json").write_text(json.dumps(payload))


def _run_script(bin_dir, state_path, fleet_dir, *extra_args, env=None, timeout=20):
    """Run restart-all-orchestrators.sh with the fake systemctl on PATH."""
    full_env = dict(os.environ)
    full_env["PATH"] = f"{bin_dir}{os.pathsep}{full_env['PATH']}"
    full_env["FAKE_SYSTEMCTL_STATE"] = str(state_path)
    full_env["ORCH_FLEET_DIR"] = str(fleet_dir)
    if env:
        full_env.update(env)
    return subprocess.run(
        ["bash", str(SCRIPT), *extra_args],
        env=full_env,
        capture_output=True,
        text=True,
        timeout=timeout,
    )


def _load_state(state_path):
    return json.loads(state_path.read_text())


def _decode(maybe_bytes):
    """subprocess.run's TimeoutExpired attaches partial output as bytes
    even when text=True was passed to the original call -- normalize."""
    if maybe_bytes is None:
        return ""
    if isinstance(maybe_bytes, bytes):
        return maybe_bytes.decode(errors="replace")
    return maybe_bytes


# ---------------------------------------------------------------------------
# step-7: RED -- DEFER+FORCE (I3 two-way), plus idle-is-transparent
# ---------------------------------------------------------------------------

def test_force_restarts_busy_unit_when_grace_is_zero(tmp_path):
    """FORCE: a busy unit force-restarts immediately when
    ORCH_RESTART_FORCE_FIRE_AFTER_SECS=0 -- a force line is printed AND the
    restart is actually recorded AND the script exits 0."""
    fleet_dir = tmp_path / "fleet"
    bin_dir, state_path = _make_fake_systemctl(
        tmp_path, running_units=[UNIT_R], units={UNIT_R: {"scenario": "fresh"}},
    )
    _write_heartbeat(fleet_dir, UNIT_R, merge_idle=False, ts_epoch=time.time())

    result = _run_script(
        bin_dir, state_path, fleet_dir, "--drain",
        env={
            "RESTART_VERIFY_TIMEOUT": "5",
            "ORCH_RESTART_FORCE_FIRE_AFTER_SECS": "0",
        },
    )

    assert result.returncode == 0, (
        f"stdout={result.stdout!r} stderr={result.stderr!r}"
    )
    assert "force-restarting" in result.stdout.lower(), (
        f"expected a force-restart line; got stdout={result.stdout!r}"
    )
    state = _load_state(state_path)
    assert ["--user", "restart", UNIT_R] in state["calls"], (
        f"expected a restart call for {UNIT_R}; got calls={state['calls']!r}"
    )


def test_defer_withholds_restart_while_busy(tmp_path):
    """DEFER: a busy unit with a large grace is NOT restarted before the
    test's own short subprocess timeout fires -- proving the script is
    still polling, not that it happened to finish quickly."""
    fleet_dir = tmp_path / "fleet"
    bin_dir, state_path = _make_fake_systemctl(
        tmp_path, running_units=[UNIT_R], units={UNIT_R: {"scenario": "fresh"}},
    )
    _write_heartbeat(fleet_dir, UNIT_R, merge_idle=False, ts_epoch=time.time())

    with pytest.raises(subprocess.TimeoutExpired) as exc_info:
        _run_script(
            bin_dir, state_path, fleet_dir, "--drain",
            env={
                "RESTART_VERIFY_TIMEOUT": "5",
                "ORCH_RESTART_FORCE_FIRE_AFTER_SECS": "99999",
                "ORCH_DRAIN_POLL_INTERVAL_SECS": "1",
            },
            timeout=3,
        )

    stdout = _decode(exc_info.value.stdout)
    assert "deferring restart of orchestrator-reify.service: mid-merge" in stdout, (
        f"expected a stable defer-prefix line; got stdout={stdout!r}"
    )
    state = _load_state(state_path)
    assert ["--user", "restart", UNIT_R] not in state["calls"], (
        f"restart must NOT have been recorded yet; got calls={state['calls']!r}"
    )


def test_idle_unit_restarts_transparently_with_no_defer_line(tmp_path):
    """A fresh, idle unit restarts with no defer/force line at all --
    the gate is a no-op when the unit isn't mid-merge."""
    fleet_dir = tmp_path / "fleet"
    bin_dir, state_path = _make_fake_systemctl(
        tmp_path, running_units=[UNIT_R], units={UNIT_R: {"scenario": "fresh"}},
    )
    _write_heartbeat(fleet_dir, UNIT_R, merge_idle=True, ts_epoch=time.time())

    result = _run_script(
        bin_dir, state_path, fleet_dir, "--drain",
        env={"RESTART_VERIFY_TIMEOUT": "5"},
    )

    assert result.returncode == 0, (
        f"stdout={result.stdout!r} stderr={result.stderr!r}"
    )
    assert "deferring" not in result.stdout.lower(), (
        f"expected no defer line for an idle unit; got stdout={result.stdout!r}"
    )
    assert "force-restarting" not in result.stdout.lower(), (
        f"expected no force line for an idle unit; got stdout={result.stdout!r}"
    )
    state = _load_state(state_path)
    assert ["--user", "restart", UNIT_R] in state["calls"], (
        f"expected a restart call for {UNIT_R}; got calls={state['calls']!r}"
    )


# ---------------------------------------------------------------------------
# step-9: RED -- I4 (fail-toward-convergence), stale/absent branch
# ---------------------------------------------------------------------------

def test_absent_heartbeat_restarts_after_zero_grace(tmp_path):
    """ABSENT (I4 fail-toward-convergence): a unit with no heartbeat file at
    all still restarts once ORCH_DRAIN_UNKNOWN_GRACE_SECS elapses -- here 0,
    so immediately."""
    fleet_dir = tmp_path / "fleet"
    bin_dir, state_path = _make_fake_systemctl(
        tmp_path, running_units=[UNIT_R], units={UNIT_R: {"scenario": "fresh"}},
    )
    # No heartbeat file written for UNIT_R at all.

    result = _run_script(
        bin_dir, state_path, fleet_dir, "--drain",
        env={
            "RESTART_VERIFY_TIMEOUT": "5",
            "ORCH_DRAIN_UNKNOWN_GRACE_SECS": "0",
        },
    )

    assert result.returncode == 0, (
        f"stdout={result.stdout!r} stderr={result.stderr!r}"
    )
    state = _load_state(state_path)
    assert ["--user", "restart", UNIT_R] in state["calls"], (
        f"expected a restart call for {UNIT_R}; got calls={state['calls']!r}"
    )


def test_stale_heartbeat_restarts_after_zero_grace(tmp_path):
    """STALE (I4 fail-toward-convergence): a unit whose heartbeat exists but
    is older than the freshness window still restarts once
    ORCH_DRAIN_UNKNOWN_GRACE_SECS elapses -- here 0, so immediately."""
    fleet_dir = tmp_path / "fleet"
    bin_dir, state_path = _make_fake_systemctl(
        tmp_path, running_units=[UNIT_R], units={UNIT_R: {"scenario": "fresh"}},
    )
    _write_heartbeat(fleet_dir, UNIT_R, merge_idle=True, ts_epoch=time.time() - 99999)

    result = _run_script(
        bin_dir, state_path, fleet_dir, "--drain",
        env={
            "RESTART_VERIFY_TIMEOUT": "5",
            "ORCH_DRAIN_UNKNOWN_GRACE_SECS": "0",
            "ORCH_DRAIN_FRESH_WINDOW_SECS": "120",
        },
    )

    assert result.returncode == 0, (
        f"stdout={result.stdout!r} stderr={result.stderr!r}"
    )
    state = _load_state(state_path)
    assert ["--user", "restart", UNIT_R] in state["calls"], (
        f"expected a restart call for {UNIT_R}; got calls={state['calls']!r}"
    )


def test_unknown_grace_withholds_restart_while_absent(tmp_path):
    """Proves the unknown-grace wait is a REAL bounded wait, not an instant
    fail-open: an absent heartbeat with a large ORCH_DRAIN_UNKNOWN_GRACE_SECS
    is NOT restarted before the test's own short subprocess timeout fires.
    This is the opposite fail-direction from the busy/defer path (which
    protects an in-flight merge), but unknown status still gets a bounded
    grace rather than restarting on the very first check."""
    fleet_dir = tmp_path / "fleet"
    bin_dir, state_path = _make_fake_systemctl(
        tmp_path, running_units=[UNIT_R], units={UNIT_R: {"scenario": "fresh"}},
    )
    # No heartbeat file written for UNIT_R at all.

    with pytest.raises(subprocess.TimeoutExpired) as exc_info:
        _run_script(
            bin_dir, state_path, fleet_dir, "--drain",
            env={
                "RESTART_VERIFY_TIMEOUT": "5",
                "ORCH_DRAIN_UNKNOWN_GRACE_SECS": "99999",
                "ORCH_DRAIN_POLL_INTERVAL_SECS": "1",
            },
            timeout=3,
        )

    stdout = _decode(exc_info.value.stdout)
    state = _load_state(state_path)
    assert ["--user", "restart", UNIT_R] not in state["calls"], (
        f"restart must NOT have been recorded yet; got calls={state['calls']!r} "
        f"stdout={stdout!r}"
    )
