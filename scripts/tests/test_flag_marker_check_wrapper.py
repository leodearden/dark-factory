"""Tests for scripts/fused-memory-flag-marker-check.sh -- the read-only
CHECK counterpart to fused-memory-flag-marker-sweep.sh (task 2596's
--check/backlog_verdict predicate mode).

Drives the wrapper via subprocess with the FLAG_MARKER_SWEEP_CMD test seam
pointed at a fake recorder executable (records its argv to a JSON state
file) -- mirrors test_flag_marker_sweep_wrapper.py's fake-recorder harness.
Real uv/fused_memory/live stores are never touched.

Task 4591: this wrapper had the same bare-`uv`-from-PATH fragility task
2917 fixed on the sweep wrapper (the sibling that actually tripped it via
its systemd unit's Persistent=true boot catch-up run, OBSERVED 2026-08-18
09:02:44 -- see fused-memory-flag-marker-sweep.sh's own header). This
wrapper has no systemd unit or before_done predicate wired to it today (the
watch gate is retired, task 3923), so the defect was LATENT rather than
observed here directly -- these tests pin the same absolute-uv-resolution
fix so a future re-wiring doesn't reinherit it.
"""
from __future__ import annotations

import json
import os
import subprocess
from pathlib import Path

import pytest

WRAPPER = Path(__file__).parent.parent / "fused-memory-flag-marker-check.sh"


# ---------------------------------------------------------------------------
# Fake check-command recorder (marker-file + configurable exit code)
# ---------------------------------------------------------------------------

_FAKE_RECORDER_SRC = '''#!/usr/bin/env python3
"""Fake check-invocation recorder for testing
fused-memory-flag-marker-check.sh. Records argv[1:] and a snapshot of
os.environ into a JSON state file at $FAKE_CHECK_STATE, then exits with
$FAKE_CHECK_EXIT_CODE (default 0).
"""
import json
import os
import sys

state_path = os.environ["FAKE_CHECK_STATE"]
with open(state_path) as f:
    state = json.load(f)
state.setdefault("calls", []).append(sys.argv[1:])
state.setdefault("envs", []).append(dict(os.environ))
with open(state_path, "w") as f:
    json.dump(state, f)

sys.exit(int(os.environ.get("FAKE_CHECK_EXIT_CODE", "0")))
'''


def _fake_recorder(tmp_path):
    """Write an executable fake check-command recorder into <tmp_path>/bin/
    and its backing JSON state file. Returns (bin_dir, state_path)."""
    bin_dir = tmp_path / "bin"
    bin_dir.mkdir(exist_ok=True)
    fake = bin_dir / "fake-check-recorder"
    fake.write_text(_FAKE_RECORDER_SRC)
    fake.chmod(0o755)

    state_path = tmp_path / "check_state.json"
    state_path.write_text(json.dumps({"calls": []}))
    return bin_dir, state_path


def _recorded_calls(state_path):
    return json.loads(state_path.read_text())["calls"]


def _recorded_envs(state_path):
    return json.loads(state_path.read_text())["envs"]


# ---------------------------------------------------------------------------
# Script driver
# ---------------------------------------------------------------------------

def _run_wrapper(
    tmp_path, *, exit_code=0, extra_env=None, dotenv_contents=None, args=None,
):
    """Run fused-memory-flag-marker-check.sh with FLAG_MARKER_SWEEP_CMD
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
    env["FLAG_MARKER_SWEEP_CMD"] = "fake-check-recorder"
    env["FAKE_CHECK_STATE"] = str(state_path)
    env["FAKE_CHECK_EXIT_CODE"] = str(exit_code)
    env["REPO"] = str(fake_repo)
    if extra_env:
        env.update(extra_env)

    result = subprocess.run(
        ["bash", str(WRAPPER), *(args or [])],
        env=env, capture_output=True, text=True, timeout=30,
    )
    return result, state_path


# ---------------------------------------------------------------------------
# Baseline behaviour (passthrough, --check flag, exit propagation)
# ---------------------------------------------------------------------------

def test_wrapper_is_executable():
    assert os.access(WRAPPER, os.X_OK), (
        f"Expected {WRAPPER} to be executable (os.X_OK); it is not. "
        f"Run: chmod +x {WRAPPER}"
    )


def test_wrapper_invokes_check_with_check_flag(tmp_path):
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
    assert "--check" in argv, f"argv={argv!r}"


def test_wrapper_passes_extra_args_through(tmp_path):
    result, state_path = _run_wrapper(
        tmp_path, args=["--project-id", "reify", "--max-backlog", "0"],
    )

    assert result.returncode == 0, (
        f"stdout={result.stdout!r} stderr={result.stderr!r}"
    )
    argv = _recorded_calls(state_path)[0]
    assert "--project-id" in argv and "reify" in argv, f"argv={argv!r}"
    assert "--max-backlog" in argv and "0" in argv, f"argv={argv!r}"


def test_wrapper_propagates_nonzero_exit(tmp_path):
    result, _state_path = _run_wrapper(tmp_path, exit_code=1)

    assert result.returncode != 0, (
        f"Expected a non-zero wrapper exit when the check command fails; "
        f"stdout={result.stdout!r} stderr={result.stderr!r}"
    )


def test_wrapper_sources_dotenv_and_propagates_to_check(tmp_path):
    """Mirrors the sweep wrapper's equivalent test: confirms the wrapper's
    stated core purpose (source $REPO/.env, export CONFIG_PATH/PROJECT_ROOT/
    FALKORDB_URI) actually reaches the check process."""
    result, state_path = _run_wrapper(
        tmp_path,
        dotenv_contents=(
            "FALKORDB_URI=redis://test-env-host:1234\n"
            "FLAG_MARKER_CHECK_TEST_SENTINEL=from-dot-env\n"
        ),
    )

    assert result.returncode == 0, (
        f"stdout={result.stdout!r} stderr={result.stderr!r}"
    )

    envs = _recorded_envs(state_path)
    assert len(envs) == 1, f"envs={envs!r}"
    recorded_env = envs[0]
    assert recorded_env.get("FALKORDB_URI") == "redis://test-env-host:1234", (
        f"Expected the sourced .env's FALKORDB_URI to reach the check "
        f"process; recorded_env={recorded_env!r}"
    )
    assert recorded_env.get("FLAG_MARKER_CHECK_TEST_SENTINEL") == "from-dot-env", (
        f"Expected a plain .env var to reach the check process via `set -a`; "
        f"recorded_env={recorded_env!r}"
    )


# ---------------------------------------------------------------------------
# Fake `uv` shim (pins the default FLAG_MARKER_SWEEP_CMD prefix)
# ---------------------------------------------------------------------------

_FAKE_UV_SRC = '''#!/usr/bin/env python3
"""Fake `uv` shim for pinning fused-memory-flag-marker-check.sh's default
FLAG_MARKER_SWEEP_CMD prefix (`uv run --frozen --project "$FM" python`).
Records argv[1:] into a JSON state file at $FAKE_CHECK_STATE and exits 0.
Never invokes real uv / fused_memory / live stores.
"""
import json
import os
import sys

state_path = os.environ["FAKE_CHECK_STATE"]
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
    --project "$FM" python`, built literally so "$FM" survives verbatim even
    when the repo path contains spaces."""
    uv_bin_dir, state_path = _fake_uv(tmp_path)

    fake_repo = tmp_path / "fake repo with spaces"
    fake_repo.mkdir(exist_ok=True)
    fake_fm = fake_repo / "fused-memory"

    env = dict(os.environ)
    env["PATH"] = f"{uv_bin_dir}{os.pathsep}{env['PATH']}"
    env["FAKE_CHECK_STATE"] = str(state_path)
    env["REPO"] = str(fake_repo)
    env.pop("FLAG_MARKER_SWEEP_CMD", None)

    result = subprocess.run(
        ["bash", str(WRAPPER), "--project-id", "reify"],
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
        "--check", "--project-id", "reify",
    ], f"argv={calls[0]!r}"


def test_wrapper_resolves_uv_by_absolute_path_when_path_omits_it(tmp_path):
    """Task 4591 / task 2917 EDIT 3 pattern. Pins that the wrapper resolves
    `uv` to an ABSOLUTE path rather than trusting PATH.

    Simulated by scrubbing PATH down to /usr/bin:/bin (no `uv` anywhere on
    it) while pointing the wrapper's UV_BIN override at the fake uv
    recorder. A wrapper that still relies on a bare `uv` word exits 127
    without ever invoking it -- the same `exec: uv: not found` / status=127
    the sweep wrapper's systemd unit hit on its boot catch-up run."""
    uv_bin_dir, state_path = _fake_uv(tmp_path)

    fake_repo = tmp_path / "fake-repo"
    fake_repo.mkdir(exist_ok=True)

    env = dict(os.environ)
    env["PATH"] = "/usr/bin:/bin"
    env["FAKE_CHECK_STATE"] = str(state_path)
    env["REPO"] = str(fake_repo)
    env["UV_BIN"] = str(uv_bin_dir / "uv")
    env.pop("FLAG_MARKER_SWEEP_CMD", None)

    result = subprocess.run(
        ["bash", str(WRAPPER)],
        env=env, capture_output=True, text=True, timeout=30,
    )

    assert result.returncode == 0, (
        f"Expected the wrapper to resolve uv via UV_BIN under a PATH that "
        f"omits it (returncode 127 == the regression); "
        f"stdout={result.stdout!r} stderr={result.stderr!r}"
    )
    calls = _recorded_calls(state_path)
    assert len(calls) >= 1, (
        f"Expected the absolute-path uv to actually be invoked; calls={calls!r} "
        f"stderr={result.stderr!r}"
    )


def test_wrapper_fails_loud_when_uv_cannot_be_resolved(tmp_path):
    """A missing interpreter must be DIAGNOSABLE, not a bare shell 127
    (loud-over-silent-degradation).

    Scrubs PATH to /usr/bin:/bin, leaves UV_BIN unset, and repoints HOME at a
    tmp dir so the $HOME/.local/bin/uv fallback misses too. The wrapper must
    exit non-zero AND print its own ERROR:-prefixed line naming `uv`."""
    if os.path.exists("/usr/local/bin/uv"):
        pytest.skip(
            "/usr/local/bin/uv exists on this host, so the wrapper's last-resort "
            "fallback resolves and the unresolvable-uv path cannot be exercised"
        )

    fake_repo = tmp_path / "fake-repo"
    fake_repo.mkdir(exist_ok=True)
    fake_home = tmp_path / "fake-home"
    fake_home.mkdir(exist_ok=True)

    env = dict(os.environ)
    env["PATH"] = "/usr/bin:/bin"
    env["REPO"] = str(fake_repo)
    env["HOME"] = str(fake_home)
    env.pop("UV_BIN", None)
    env.pop("FLAG_MARKER_SWEEP_CMD", None)

    result = subprocess.run(
        ["bash", str(WRAPPER)],
        env=env, capture_output=True, text=True, timeout=30,
    )

    assert result.returncode != 0, (
        f"Expected a non-zero exit when uv cannot be resolved; "
        f"stdout={result.stdout!r} stderr={result.stderr!r}"
    )
    assert "ERROR:" in result.stderr, (
        f"Expected the wrapper's own ERROR:-prefixed diagnostic rather than a "
        f"bare shell 127; stderr={result.stderr!r}"
    )
    assert "uv" in result.stderr, (
        f"Expected the diagnostic to name `uv`; stderr={result.stderr!r}"
    )
