"""Tests for scripts/install-reclaim-orphaned-worktrees-timer.sh -- the
fail-loud, self-verifying installer for the .worktrees-orphaned/ reclaim
sweep's systemd user timer (task 2980).

Drives the script via subprocess with `scripts/tests/conftest.py::fake_systemctl`
first on PATH, as test_install_flag_marker_sweep_timer.py does. Real systemd
is never touched. This installer also kicks an immediate one-time drain of the
current backlog with `systemctl start`, which the fake answers with success.
"""
from __future__ import annotations

import os
import subprocess
from pathlib import Path

import pytest
from shared.whole_tree_guard_family import WHOLE_TREE_SCAN_TEST_TIMEOUT

pytestmark = pytest.mark.timeout(WHOLE_TREE_SCAN_TEST_TIMEOUT)

SCRIPT = Path(__file__).parent.parent / "install-reclaim-orphaned-worktrees-timer.sh"
TEMPLATES_DIR = Path(__file__).parent.parent  # scripts/tests/../ = scripts/

SERVICE_NAME = "reclaim-orphaned-worktrees.service"
TIMER_NAME = "reclaim-orphaned-worktrees.timer"


# ---------------------------------------------------------------------------
# Script driver
# ---------------------------------------------------------------------------

def _run_script(fake_systemctl, tmp_path, *, env=None):
    """Run install-reclaim-orphaned-worktrees-timer.sh via subprocess with a
    *fake_systemctl* first on PATH, reset on every call."""
    fake_systemctl.reset()

    full_env = dict(os.environ)
    full_env["PATH"] = f"{fake_systemctl.bin_dir}{os.pathsep}{full_env['PATH']}"
    if env:
        full_env.update(env)
    return subprocess.run(
        ["bash", str(SCRIPT)],
        env=full_env, capture_output=True, text=True, timeout=30,
    )


# ---------------------------------------------------------------------------
# step-19: RED -- install-reclaim-orphaned-worktrees-timer.sh (happy path)
# ---------------------------------------------------------------------------

def test_script_is_executable():
    assert os.access(SCRIPT, os.X_OK), (
        f"Expected {SCRIPT} to be executable (os.X_OK); it is not. "
        f"Run: chmod +x {SCRIPT}"
    )


def test_install_copies_units_enables_timer_and_kicks_drain(fake_systemctl, tmp_path):
    xdg_config = tmp_path / "xdg-config"

    result = _run_script(fake_systemctl, tmp_path, env={"XDG_CONFIG_HOME": str(xdg_config)})

    assert result.returncode == 0, (
        f"stdout={result.stdout!r} stderr={result.stderr!r}"
    )

    unit_dir = xdg_config / "systemd" / "user"
    service_path = unit_dir / SERVICE_NAME
    timer_path = unit_dir / TIMER_NAME
    assert service_path.is_file(), f"Expected {service_path} to exist after install"
    assert timer_path.is_file(), f"Expected {timer_path} to exist after install"
    assert service_path.read_bytes() == (TEMPLATES_DIR / SERVICE_NAME).read_bytes()
    assert timer_path.read_bytes() == (TEMPLATES_DIR / TIMER_NAME).read_bytes()

    calls = fake_systemctl.calls()
    assert ["daemon-reload"] in calls, f"calls={calls!r}"
    assert ["enable", "--now", TIMER_NAME] in calls, f"calls={calls!r}"
    assert ["start", SERVICE_NAME] in calls, f"calls={calls!r}"


def test_install_fails_loud_when_timer_absent_from_list_timers(fake_systemctl, tmp_path):
    xdg_config = tmp_path / "xdg-config"

    result = _run_script(
        fake_systemctl, tmp_path,
        env={
            "XDG_CONFIG_HOME": str(xdg_config),
            "FAKE_SYSTEMCTL_OMIT_LIST_TIMERS": "1",
        },
    )

    assert result.returncode != 0, (
        f"Expected a non-zero exit when list-timers omits the enabled timer "
        f"(self-verify failure); stdout={result.stdout!r} stderr={result.stderr!r}"
    )
    assert TIMER_NAME in result.stderr, (
        f"Expected stderr to name the missing timer unit {TIMER_NAME!r}; "
        f"stderr={result.stderr!r}"
    )
