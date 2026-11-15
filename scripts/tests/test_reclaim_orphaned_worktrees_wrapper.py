"""Tests for scripts/reclaim-orphaned-worktrees.sh -- the committed nightly
reclaim action for .worktrees-orphaned/ parkings (task 2980).

Drives the wrapper via subprocess with the RECLAIM_ORPHANED_WT_CMD test seam
pointed at a `scripts/tests/conftest.py::fake_recorder` executable and REPO
pointed at a tmp dir -- mirrors test_flag_marker_sweep_wrapper.py,
SIMPLIFIED: the reclaim script is stdlib-only so the wrapper runs
`python3 <script> --repo "$REPO"` directly (no `uv run`, no `.env` sourcing /
service-env exports that the flag-marker wrapper needs for its fused_memory
imports).
"""
from __future__ import annotations

import os
import subprocess
from pathlib import Path

WRAPPER = Path(__file__).parent.parent / "reclaim-orphaned-worktrees.sh"


# ---------------------------------------------------------------------------
# Script driver
# ---------------------------------------------------------------------------

def _run_wrapper(fake_recorder, tmp_path, *, exit_code=0, extra_env=None):
    """Run reclaim-orphaned-worktrees.sh with RECLAIM_ORPHANED_WT_CMD pointed at
    a fake recorder and REPO pointed at a tmp dir."""
    recorder = fake_recorder("fake-reclaim-recorder", exit_code=exit_code)

    fake_repo = tmp_path / "fake-repo"
    fake_repo.mkdir(exist_ok=True)

    env = dict(os.environ)
    env["PATH"] = f"{recorder.bin_dir}{os.pathsep}{env['PATH']}"
    env["RECLAIM_ORPHANED_WT_CMD"] = "fake-reclaim-recorder"
    env["REPO"] = str(fake_repo)
    if extra_env:
        env.update(extra_env)

    result = subprocess.run(
        ["bash", str(WRAPPER)],
        env=env, capture_output=True, text=True, timeout=30,
    )
    return result, recorder, fake_repo


# ---------------------------------------------------------------------------
# step-17: RED -- reclaim-orphaned-worktrees.sh
# ---------------------------------------------------------------------------

def test_wrapper_is_executable():
    assert os.access(WRAPPER, os.X_OK), (
        f"Expected {WRAPPER} to be executable (os.X_OK); it is not. "
        f"Run: chmod +x {WRAPPER}"
    )


def test_wrapper_invokes_reclaim_script_with_repo(fake_recorder, tmp_path):
    result, recorder, fake_repo = _run_wrapper(fake_recorder, tmp_path)

    assert result.returncode == 0, (
        f"stdout={result.stdout!r} stderr={result.stderr!r}"
    )

    calls = recorder.calls()
    assert len(calls) == 1, f"calls={calls!r}"
    argv = calls[0]
    assert any(
        a.endswith("scripts/reclaim_orphaned_worktrees.py") for a in argv
    ), f"Expected the reclaim script path in argv={argv!r}"
    assert "--repo" in argv, f"argv={argv!r}"
    assert argv[argv.index("--repo") + 1] == str(fake_repo), (
        f"Expected --repo <REPO> in argv={argv!r}"
    )


def test_wrapper_propagates_nonzero_exit(fake_recorder, tmp_path):
    result, _recorder, _fake_repo = _run_wrapper(fake_recorder, tmp_path, exit_code=9)

    assert result.returncode != 0, (
        f"Expected a non-zero wrapper exit when the reclaim command fails; "
        f"stdout={result.stdout!r} stderr={result.stderr!r}"
    )
