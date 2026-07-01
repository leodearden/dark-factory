"""Tests for restart-orchestrator.sh — drives the script via subprocess
against a fake `systemctl` shimmed onto PATH.

The fake `systemctl` is a small Python script written into a temp `bin/`
dir (prepended to PATH) at test time; it records every invocation (minus
the constant `--user` flag) into a JSON state file and answers `show`
from that same file, so the real script never touches a real systemd.
"""
from __future__ import annotations

import json
import os
import subprocess
from pathlib import Path

SCRIPT = Path(__file__).parent.parent / "restart-orchestrator.sh"
UNIT = "orchestrator-dark-factory.service"

FAKE_SYSTEMCTL_SRC = '''#!/usr/bin/env python3
"""Fake `systemctl` for testing restart-orchestrator.sh.

Records every invocation (minus `--user`) into a JSON state file at
$FAKE_SYSTEMCTL_STATE and answers `show`/`restart` from that file's
fields. When state["scenario"] == "fresh", a `restart` call bumps
MainPID/ActiveState/ActiveEnterTimestampMonotonic to simulate the
service coming back up. When state["scenario"] == "bumped_but_failed",
a `restart` call bumps MainPID/ActiveEnterTimestampMonotonic (as if a
new process started) but reports ActiveState="failed" -- simulating a
unit that respawned into a crash loop rather than a clean start. Any
other scenario leaves the fields unchanged (simulating a stale/failed
restart).
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

    fields = None
    i = 0
    while i < len(rest):
        tok = rest[i]
        if tok == "-p":
            fields = rest[i + 1]
            i += 2
        elif tok.startswith("--property="):
            fields = tok.split("=", 1)[1]
            i += 1
        else:
            i += 1  # unit name or unrecognized flag -- not needed by the fake

    state = _load()
    state.setdefault("calls", []).append(argv[1:])

    if verb == "restart":
        scenario = state.get("scenario")
        if scenario == "fresh":
            state["MainPID"] = state.get("MainPID", 1000) + 1
            state["ActiveState"] = "active"
            state["ActiveEnterTimestampMonotonic"] = (
                state.get("ActiveEnterTimestampMonotonic", 0) + 5_000_000
            )
            state["ActiveEnterTimestamp"] = "restarted"
        elif scenario == "bumped_but_failed":
            state["MainPID"] = state.get("MainPID", 1000) + 1
            state["ActiveState"] = "failed"
            state["ActiveEnterTimestampMonotonic"] = (
                state.get("ActiveEnterTimestampMonotonic", 0) + 5_000_000
            )
            state["ActiveEnterTimestamp"] = "restarted"
        _save(state)
        return 0

    if verb == "show":
        current = {
            "MainPID": str(state.get("MainPID", 0)),
            "ActiveState": state.get("ActiveState", "active"),
            "ActiveEnterTimestamp": state.get("ActiveEnterTimestamp", "baseline"),
            "ActiveEnterTimestampMonotonic": str(state.get("ActiveEnterTimestampMonotonic", 0)),
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

def _make_fake_systemctl(tmp_path, *, scenario="stale", main_pid=1234):
    """Write a fake `systemctl` into <tmp_path>/bin/ + its backing state file.

    scenario="fresh" makes a `restart` call bump MainPID/ActiveState/
    ActiveEnterTimestampMonotonic (simulating a successful restart);
    scenario="stale" (default) leaves them unchanged (simulating a
    failed/no-op restart).

    Returns (bin_dir, state_path).
    """
    bin_dir = tmp_path / "bin"
    bin_dir.mkdir()
    fake = bin_dir / "systemctl"
    fake.write_text(FAKE_SYSTEMCTL_SRC)
    fake.chmod(0o755)

    state_path = tmp_path / "systemctl_state.json"
    state_path.write_text(json.dumps({
        "MainPID": main_pid,
        "ActiveState": "active",
        "ActiveEnterTimestamp": "baseline",
        "ActiveEnterTimestampMonotonic": 1_000_000,
        "scenario": scenario,
        "calls": [],
    }))
    return bin_dir, state_path


def _run_script(bin_dir, state_path, *extra_args, env=None):
    """Run restart-orchestrator.sh with the fake systemctl on PATH."""
    full_env = dict(os.environ)
    full_env["PATH"] = f"{bin_dir}{os.pathsep}{full_env['PATH']}"
    full_env["FAKE_SYSTEMCTL_STATE"] = str(state_path)
    if env:
        full_env.update(env)
    return subprocess.run(
        ["bash", str(SCRIPT), *extra_args],
        env=full_env,
        capture_output=True,
        text=True,
        timeout=20,
    )


def _load_state(state_path):
    return json.loads(state_path.read_text())


# ---------------------------------------------------------------------------
# step-1: RED -- script must invoke `systemctl --user restart <unit>`
# ---------------------------------------------------------------------------

def test_invokes_systemctl_restart_on_correct_unit(tmp_path):
    """The script must call `systemctl --user restart orchestrator-dark-factory.service`."""
    bin_dir, state_path = _make_fake_systemctl(tmp_path)

    _run_script(bin_dir, state_path, env={"RESTART_VERIFY_TIMEOUT": "1"})

    state = _load_state(state_path)
    assert ["--user", "restart", UNIT] in state["calls"], (
        f"Expected a `systemctl --user restart {UNIT}` call; "
        f"got calls={state['calls']!r}"
    )


# ---------------------------------------------------------------------------
# step-3: RED -- fresh MainPID after restart must be verified and reported
# ---------------------------------------------------------------------------

def test_prints_new_main_pid_on_fresh_restart(tmp_path):
    """A restart that comes back with a fresh MainPID exits 0 and names it."""
    bin_dir, state_path = _make_fake_systemctl(tmp_path, scenario="fresh", main_pid=1234)

    result = _run_script(bin_dir, state_path, env={"RESTART_VERIFY_TIMEOUT": "5"})

    assert result.returncode == 0, (
        f"Expected exit 0 on a fresh restart; got {result.returncode}\n"
        f"stdout={result.stdout!r} stderr={result.stderr!r}"
    )
    new_pid = _load_state(state_path)["MainPID"]
    assert new_pid != 1234, "fake systemctl did not bump MainPID -- test setup is broken"
    assert str(new_pid) in result.stdout, (
        f"Expected the new MainPID ({new_pid}) in stdout; got: {result.stdout!r}"
    )


# ---------------------------------------------------------------------------
# step-5: RED -- a restart that never comes back fresh must exit non-zero
# ---------------------------------------------------------------------------

def test_exits_nonzero_when_restart_does_not_bring_back_fresh_pid(tmp_path):
    """A restart that leaves MainPID unchanged (stale/failed) exits non-zero."""
    bin_dir, state_path = _make_fake_systemctl(tmp_path, scenario="stale", main_pid=1234)

    result = _run_script(bin_dir, state_path, env={"RESTART_VERIFY_TIMEOUT": "1"})

    assert result.returncode != 0, (
        f"Expected non-zero exit when the restart never comes back fresh; got 0\n"
        f"stdout={result.stdout!r} stderr={result.stderr!r}"
    )
    assert "stale" in (result.stdout + result.stderr).lower(), (
        f"Expected a stale/failed-restart error message; got "
        f"stdout={result.stdout!r} stderr={result.stderr!r}"
    )


# ---------------------------------------------------------------------------
# step-7: RED -- a stray --drain argument must be accepted, not rejected
# ---------------------------------------------------------------------------

def test_drain_flag_is_accepted_and_ignored(tmp_path):
    """`--drain` is accepted-and-ignored; the restart+verify flow still runs."""
    bin_dir, state_path = _make_fake_systemctl(tmp_path, scenario="fresh", main_pid=1234)

    result = _run_script(bin_dir, state_path, "--drain", env={"RESTART_VERIFY_TIMEOUT": "5"})

    assert result.returncode == 0, (
        f"Expected --drain to be accepted (exit 0); got {result.returncode}\n"
        f"stdout={result.stdout!r} stderr={result.stderr!r}"
    )
    state = _load_state(state_path)
    assert ["--user", "restart", UNIT] in state["calls"], (
        f"Expected the restart+verify flow to still run with --drain; "
        f"got calls={state['calls']!r}"
    )


# ---------------------------------------------------------------------------
# amendment: unrecognized arguments must be rejected before any systemctl call
# ---------------------------------------------------------------------------

def test_rejects_unknown_argument(tmp_path):
    """An unrecognized argument exits non-zero and the script never proceeds
    to restart -- guards the `*)` branch's `exit 1` against regressions that
    would silently widen the accepted-argument set."""
    bin_dir, state_path = _make_fake_systemctl(tmp_path)

    result = _run_script(bin_dir, state_path, "--bogus", env={"RESTART_VERIFY_TIMEOUT": "1"})

    assert result.returncode != 0, (
        f"Expected non-zero exit on an unrecognized argument; got 0\n"
        f"stdout={result.stdout!r} stderr={result.stderr!r}"
    )
    state = _load_state(state_path)
    assert ["--user", "restart", UNIT] not in state["calls"], (
        f"Expected no restart call to be recorded when rejecting an unknown "
        f"argument; got calls={state['calls']!r}"
    )


# ---------------------------------------------------------------------------
# amendment: a fresh PID in a non-active unit state must still fail verify
# ---------------------------------------------------------------------------

def test_exits_nonzero_when_restart_bumps_pid_but_unit_not_active(tmp_path):
    """A restart that bumps MainPID/timestamp but lands in a non-active
    ActiveState (e.g. crash-looped into 'failed') must be treated as a
    failed restart -- guards the ActiveState=='active' conjunct against
    regressions that would key freshness off MainPID/timestamp alone."""
    bin_dir, state_path = _make_fake_systemctl(
        tmp_path, scenario="bumped_but_failed", main_pid=1234
    )

    result = _run_script(bin_dir, state_path, env={"RESTART_VERIFY_TIMEOUT": "1"})

    assert result.returncode != 0, (
        f"Expected non-zero exit when ActiveState is not 'active' even "
        f"though MainPID/timestamp advanced; got 0\n"
        f"stdout={result.stdout!r} stderr={result.stderr!r}"
    )
    assert "stale" in (result.stdout + result.stderr).lower(), (
        f"Expected a stale/failed-restart error message; got "
        f"stdout={result.stdout!r} stderr={result.stderr!r}"
    )
