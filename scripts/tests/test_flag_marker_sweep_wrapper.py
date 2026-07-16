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
and its flags) and a snapshot of os.environ into a JSON state file at
$FAKE_SWEEP_STATE, then exits with $FAKE_SWEEP_EXIT_CODE (default 0).
"""
import json
import os
import sys

state_path = os.environ["FAKE_SWEEP_STATE"]
with open(state_path) as f:
    state = json.load(f)
state.setdefault("calls", []).append(sys.argv[1:])
state.setdefault("envs", []).append(dict(os.environ))
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


def _recorded_envs(state_path):
    return json.loads(state_path.read_text())["envs"]


# ---------------------------------------------------------------------------
# Script driver
# ---------------------------------------------------------------------------

def _run_wrapper(tmp_path, *, exit_code=0, extra_env=None, dotenv_contents=None):
    """Run fused-memory-flag-marker-sweep.sh with FLAG_MARKER_SWEEP_CMD
    pointed at the fake recorder and REPO pointed at a tmp dir with no
    `.env` (so the wrapper's `source .env` is a no-op under test) -- unless
    `dotenv_contents` is given, in which case a `$REPO/.env` file with that
    content is written first so the sourcing branch itself is exercised."""
    bin_dir, state_path = _fake_recorder(tmp_path)

    fake_repo = tmp_path / "fake-repo"
    fake_repo.mkdir(exist_ok=True)
    if dotenv_contents is not None:
        (fake_repo / ".env").write_text(dotenv_contents)

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


def test_wrapper_sources_dotenv_and_propagates_to_sweep(tmp_path):
    """The wrapper's stated core purpose (module docstring: it 'must run
    under the SERVICE env, not a bare shell, or the census silently
    narrows') is to source $REPO/.env and export CONFIG_PATH/PROJECT_ROOT/
    FALKORDB_URI so the sweep runs with the right service environment.
    Every other test points REPO at a directory with no .env, so the
    sourcing branch and export propagation are otherwise never exercised.
    This writes a real `.env` and confirms both an explicitly-exported var
    (FALKORDB_URI, overriding the wrapper's own default) and a plain .env
    var the wrapper never names itself (propagated only via `set -a`) reach
    the sweep process -- a regression that broke sourcing or dropped the
    `set -a` export would leave both at their fallback/absent values while
    every other test in this file stayed green."""
    result, state_path = _run_wrapper(
        tmp_path,
        dotenv_contents=(
            "FALKORDB_URI=redis://test-env-host:1234\n"
            "FLAG_MARKER_SWEEP_TEST_SENTINEL=from-dot-env\n"
        ),
    )

    assert result.returncode == 0, (
        f"stdout={result.stdout!r} stderr={result.stderr!r}"
    )

    envs = _recorded_envs(state_path)
    assert len(envs) == 1, f"envs={envs!r}"
    recorded_env = envs[0]
    assert recorded_env.get("FALKORDB_URI") == "redis://test-env-host:1234", (
        f"Expected the sourced .env's FALKORDB_URI to reach the sweep "
        f"process (overriding the wrapper's own default); "
        f"recorded_env={recorded_env!r}"
    )
    assert recorded_env.get("FLAG_MARKER_SWEEP_TEST_SENTINEL") == "from-dot-env", (
        f"Expected a plain .env var the wrapper never names itself to "
        f"still reach the sweep process via `set -a`; "
        f"recorded_env={recorded_env!r}"
    )


# ---------------------------------------------------------------------------
# Fake `uv` shim (pins the default FLAG_MARKER_SWEEP_CMD prefix)
# ---------------------------------------------------------------------------

_FAKE_UV_SRC = '''#!/usr/bin/env python3
"""Fake `uv` shim for pinning fused-memory-flag-marker-sweep.sh's default
FLAG_MARKER_SWEEP_CMD prefix (`uv run --frozen --project "$FM" python`).
Records argv[1:] into a JSON state file at $FAKE_SWEEP_STATE and exits 0.
Never invokes real uv / fused_memory / live stores.
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

sys.exit(0)
'''


def _fake_uv(tmp_path):
    """Write an executable fake `uv` into <tmp_path>/uv-bin/ and its backing
    JSON state file. Returns (bin_dir, state_path)."""
    bin_dir = tmp_path / "uv-bin"
    bin_dir.mkdir(exist_ok=True)
    fake = bin_dir / "uv"
    fake.write_text(_FAKE_UV_SRC)
    fake.chmod(0o755)

    state_path = tmp_path / "uv_state.json"
    state_path.write_text(json.dumps({"calls": []}))
    return bin_dir, state_path


def test_wrapper_default_prefix_invokes_uv_run_frozen_project(tmp_path):
    """Pins the default FLAG_MARKER_SWEEP_CMD prefix -- `uv run --frozen
    --project "$FM" python`, an unquoted array expansion with an embedded
    quoted "$FM" -- which every other test in this file bypasses by
    overriding FLAG_MARKER_SWEEP_CMD with a bare recorder. A future edit
    that broke that quoting (e.g. dropping the inner quotes around $FM)
    would pass every other test while silently mis-invoking (or
    word-splitting) the real `uv` in production. REPO is deliberately given
    a space in its path (`FM` inherits it) so a dropped-quote regression
    actually manifests here as a word-split argv rather than passing by
    coincidence -- the same case the reviewer manually verified by hand."""
    uv_bin_dir, state_path = _fake_uv(tmp_path)

    fake_repo = tmp_path / "fake repo with spaces"
    fake_repo.mkdir(exist_ok=True)
    fake_fm = fake_repo / "fused-memory"

    env = dict(os.environ)
    env["PATH"] = f"{uv_bin_dir}{os.pathsep}{env['PATH']}"
    env["FAKE_SWEEP_STATE"] = str(state_path)
    env["REPO"] = str(fake_repo)
    env.pop("FLAG_MARKER_SWEEP_CMD", None)

    result = subprocess.run(
        ["bash", str(WRAPPER)],
        env=env, capture_output=True, text=True, timeout=30,
    )
    assert result.returncode == 0, (
        f"stdout={result.stdout!r} stderr={result.stderr!r}"
    )

    calls = _recorded_calls(state_path)
    assert len(calls) == 1, f"calls={calls!r}"
    assert calls[0] == [
        "run", "--frozen", "--project", str(fake_fm), "python",
        str(fake_fm / "scripts" / "sweep_orphan_flag_markers.py"),
        "--apply", "--terminal-drain",
    ], f"argv={calls[0]!r}"
