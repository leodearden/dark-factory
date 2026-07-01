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
fields. State is static — a `restart` call is recorded but never
changes MainPID/ActiveState/timestamp fields.
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

def _make_fake_systemctl(tmp_path, *, main_pid=1234):
    """Write a fake `systemctl` into <tmp_path>/bin/ + its backing state file.

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

    _run_script(bin_dir, state_path, env={"RESTART_VERIFY_TIMEOUT": "5"})

    state = _load_state(state_path)
    assert ["--user", "restart", UNIT] in state["calls"], (
        f"Expected a `systemctl --user restart {UNIT}` call; "
        f"got calls={state['calls']!r}"
    )
