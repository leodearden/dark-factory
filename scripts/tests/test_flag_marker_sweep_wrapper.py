"""Tests for scripts/fused-memory-flag-marker-sweep.sh -- the committed
nightly DRAIN action for stage1_flag_marker dead-weight records (task 2693,
follow-up to task 2596's previously-unwired sweep).

Drives the wrapper via subprocess with the FLAG_MARKER_SWEEP_CMD test seam
pointed at a fake recorder executable (records its argv to a JSON state
file) -- mirrors test_install_trickle_timer.py's fake-systemctl harness.
Real uv/fused_memory/live stores are never touched.
"""
from __future__ import annotations

import json
import os
import subprocess
from pathlib import Path

WRAPPER = Path(__file__).parent.parent / "fused-memory-flag-marker-sweep.sh"


# ---------------------------------------------------------------------------
# Fake sweep-command recorder (marker-file + configurable exit code)
# ---------------------------------------------------------------------------

_FAKE_RECORDER_SRC = '''#!/usr/bin/env python3
"""Fake sweep-invocation recorder for testing
fused-memory-flag-marker-sweep.sh. Records argv[1:] (the sweep script path
and its flags) into a JSON state file at $FAKE_SWEEP_STATE, then exits with
$FAKE_SWEEP_EXIT_CODE (default 0).
"""
import json
import os
import sys

state_path = os.environ["FAKE_SWEEP_STATE"]
with open(state_path) as f:
    state = json.load(f)
state.setdefault("calls", []).append(sys.argv[1:])
with open(state_path, "w") as f:
    json.dump(state, f)

sys.exit(int(os.environ.get("FAKE_SWEEP_EXIT_CODE", "0")))
'''


def _fake_recorder(tmp_path):
    """Write an executable fake sweep-command recorder into <tmp_path>/bin/
    and its backing JSON state file. Returns (bin_dir, state_path)."""
    bin_dir = tmp_path / "bin"
    bin_dir.mkdir(exist_ok=True)
    fake = bin_dir / "fake-sweep-recorder"
    fake.write_text(_FAKE_RECORDER_SRC)
    fake.chmod(0o755)

    state_path = tmp_path / "sweep_state.json"
    state_path.write_text(json.dumps({"calls": []}))
    return bin_dir, state_path


def _recorded_calls(state_path):
    return json.loads(state_path.read_text())["calls"]


# ---------------------------------------------------------------------------
# Script driver
# ---------------------------------------------------------------------------

def _run_wrapper(tmp_path, *, exit_code=0, extra_env=None):
    """Run fused-memory-flag-marker-sweep.sh with FLAG_MARKER_SWEEP_CMD
    pointed at the fake recorder and REPO pointed at a tmp dir with no
    `.env` (so the wrapper's `source .env` is a no-op under test)."""
    bin_dir, state_path = _fake_recorder(tmp_path)

    fake_repo = tmp_path / "fake-repo"
    fake_repo.mkdir(exist_ok=True)

    env = dict(os.environ)
    env["PATH"] = f"{bin_dir}{os.pathsep}{env['PATH']}"
    env["FLAG_MARKER_SWEEP_CMD"] = "fake-sweep-recorder"
    env["FAKE_SWEEP_STATE"] = str(state_path)
    env["FAKE_SWEEP_EXIT_CODE"] = str(exit_code)
    env["REPO"] = str(fake_repo)
    if extra_env:
        env.update(extra_env)

    result = subprocess.run(
        ["bash", str(WRAPPER)],
        env=env, capture_output=True, text=True, timeout=30,
    )
    return result, state_path


# ---------------------------------------------------------------------------
# step-1: RED -- fused-memory-flag-marker-sweep.sh
# ---------------------------------------------------------------------------

def test_wrapper_is_executable():
    assert os.access(WRAPPER, os.X_OK), (
        f"Expected {WRAPPER} to be executable (os.X_OK); it is not. "
        f"Run: chmod +x {WRAPPER}"
    )


def test_wrapper_invokes_sweep_with_apply_and_terminal_drain(tmp_path):
    result, state_path = _run_wrapper(tmp_path)

    assert result.returncode == 0, (
        f"stdout={result.stdout!r} stderr={result.stderr!r}"
    )

    calls = _recorded_calls(state_path)
    assert len(calls) == 1, f"calls={calls!r}"
    argv = calls[0]
    assert any(
        a.endswith("fused-memory/scripts/sweep_orphan_flag_markers.py") for a in argv
    ), f"Expected the sweep script path in argv={argv!r}"
    assert "--apply" in argv, f"argv={argv!r}"
    assert "--terminal-drain" in argv, f"argv={argv!r}"


def test_wrapper_propagates_nonzero_exit(tmp_path):
    result, _state_path = _run_wrapper(tmp_path, exit_code=7)

    assert result.returncode != 0, (
        f"Expected a non-zero wrapper exit when the sweep command fails; "
        f"stdout={result.stdout!r} stderr={result.stderr!r}"
    )
