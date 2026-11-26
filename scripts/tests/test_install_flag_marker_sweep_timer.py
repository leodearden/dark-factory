"""Tests for scripts/install-flag-marker-sweep-timer.sh -- the fail-loud,
self-verifying installer for the fused-memory stage1_flag_marker sweep's
systemd user timer (task 2693).

Drives the script via subprocess with `scripts/tests/conftest.py::fake_timer_systemctl`
first on PATH. Real systemd is never touched. Unlike the trickle-timer
installer, this job is a single non-templated dark_factory instance (no
project_id argument, no per-project config resolution), so the driver has no
INSTALL_TRICKLE_TIMER_PYTHON/LEGIBILITY_SEARCH_ROOTS equivalent. This
installer, unlike install-trickle-timer.sh, also kicks an immediate one-time
drain of the current backlog with `systemctl start`. FAKE_SYSTEMCTL_FAIL_START=1
makes that kick fail in systemd's own wording, so tests can pin that a failing
kick neither prevents nor hides the self-verify.
"""
from __future__ import annotations

import os
import subprocess
from pathlib import Path

import pytest
from shared.whole_tree_guard_family import WHOLE_TREE_SCAN_TEST_TIMEOUT

pytestmark = pytest.mark.timeout(WHOLE_TREE_SCAN_TEST_TIMEOUT)

SCRIPT = Path(__file__).parent.parent / "install-flag-marker-sweep-timer.sh"
TEMPLATES_DIR = Path(__file__).parent.parent  # scripts/tests/../ = scripts/

SERVICE_NAME = "fused-memory-flag-marker-sweep.service"
TIMER_NAME = "fused-memory-flag-marker-sweep.timer"


# ---------------------------------------------------------------------------
# Script driver
# ---------------------------------------------------------------------------

def _run_script(fake_timer_systemctl, *, env=None):
    """Run install-flag-marker-sweep-timer.sh via subprocess.

    Puts *fake_timer_systemctl* first on PATH, reset on every call. Callers
    supply XDG_CONFIG_HOME via *env*.
    """
    fake_timer_systemctl.reset()

    full_env = dict(os.environ)
    full_env["PATH"] = f"{fake_timer_systemctl.bin_dir}{os.pathsep}{full_env['PATH']}"
    if env:
        full_env.update(env)
    return subprocess.run(
        ["bash", str(SCRIPT)],
        env=full_env, capture_output=True, text=True, timeout=30,
    )


# ---------------------------------------------------------------------------
# step-3: RED -- install-flag-marker-sweep-timer.sh (happy path)
# ---------------------------------------------------------------------------

def test_script_is_executable():
    assert os.access(SCRIPT, os.X_OK), (
        f"Expected {SCRIPT} to be executable (os.X_OK); it is not. "
        f"Run: chmod +x {SCRIPT}"
    )


def test_install_copies_units_enables_timer_and_kicks_drain(fake_timer_systemctl, tmp_path):
    xdg_config = tmp_path / "xdg-config"

    result = _run_script(fake_timer_systemctl, env={"XDG_CONFIG_HOME": str(xdg_config)})

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

    calls = fake_timer_systemctl.calls()
    assert ["daemon-reload"] in calls, f"calls={calls!r}"
    assert ["enable", "--now", TIMER_NAME] in calls, f"calls={calls!r}"
    assert ["start", SERVICE_NAME] in calls, f"calls={calls!r}"
    assert ["list-timers", "--all"] in calls, f"calls={calls!r}"
    assert calls.index(["list-timers", "--all"]) < calls.index(
        ["start", SERVICE_NAME]
    ), (
        "Expected the list-timers self-verify to run BEFORE the one-time "
        "drain kick (systemctl start): the drain runs a blocking "
        "Type=oneshot unit, so a self-verify gated behind it would never "
        f"run if the drain kick failed; calls={calls!r}"
    )


# ---------------------------------------------------------------------------
# step-5: RED -- fail loud when the timer is absent from `list-timers`
# ---------------------------------------------------------------------------

def test_install_fails_loud_when_timer_absent_from_list_timers(fake_timer_systemctl, tmp_path):
    xdg_config = tmp_path / "xdg-config"

    result = _run_script(
        fake_timer_systemctl,
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


# ---------------------------------------------------------------------------
# task 4061 step-1/step-3: RED -- the self-verify must run before the
# one-time drain kick and must still run (and stay loud, with the failure
# attributed to the drain specifically) when the drain kick fails
# ---------------------------------------------------------------------------

def test_self_verify_runs_even_when_drain_kick_fails(fake_timer_systemctl, tmp_path):
    xdg_config = tmp_path / "xdg-config"

    result = _run_script(
        fake_timer_systemctl,
        env={
            "XDG_CONFIG_HOME": str(xdg_config),
            "FAKE_SYSTEMCTL_FAIL_START": "1",
        },
    )

    calls = fake_timer_systemctl.calls()
    assert ["list-timers", "--all"] in calls, (
        f"Expected the list-timers self-verify to run even though the "
        f"drain kick (systemctl start) failed; calls={calls!r}"
    )
    assert ["start", SERVICE_NAME] in calls, f"calls={calls!r}"
    assert calls.index(["list-timers", "--all"]) < calls.index(
        ["start", SERVICE_NAME]
    ), (
        "Expected the list-timers self-verify to precede the (failing) "
        f"one-time drain kick; calls={calls!r}"
    )
    assert TIMER_NAME in result.stdout and "installed and enabled" in result.stdout, (
        f"Expected the install-succeeded signal (naming {TIMER_NAME!r}) to "
        f"still be emitted on stdout even though the drain kick failed -- "
        f"the operator must be told the timer IS armed regardless; "
        f"stdout={result.stdout!r}"
    )
    assert result.returncode != 0, (
        f"Expected a failing drain kick to still exit non-zero; "
        f"stdout={result.stdout!r} stderr={result.stderr!r}"
    )

    # step-3: the failure must be attributed to the drain kick by the
    # SCRIPT itself, not just visible via the fake systemctl's generic
    # 'Job for <unit> failed...' message (which is already in stderr today
    # and would make a bare `SERVICE_NAME in result.stderr` check a false
    # GREEN). Select the script's own line by its `ERROR:` prefix -- the
    # fake's message carries no such prefix -- rather than pinning exact
    # prose, so a benign rewording of the message doesn't break this test.
    own_error_lines = [
        line for line in result.stderr.splitlines() if line.startswith("ERROR:")
    ]
    assert own_error_lines and SERVICE_NAME in own_error_lines[0], (
        f"Expected the script's own ERROR: line to attribute the failure "
        f"to the one-time drain kick and name {SERVICE_NAME!r}, distinct "
        f"from the fake systemctl's generic 'Job for ... failed' message; "
        f"stderr={result.stderr!r}"
    )


def test_installed_service_unit_carries_explicit_path_for_boot_catchup(fake_timer_systemctl, tmp_path):
    """Task 2917 EDIT 3, unit half.

    The .timer unit sets `Persistent=true`, so a nightly firing missed while
    the machine was asleep/offline is caught up at BOOT -- and a boot-time
    systemd user service gets a minimal PATH,
    because the login session has not yet pushed the user PATH into the
    systemd user manager. OBSERVED consequence (journalctl --user -u
    fused-memory-flag-marker-sweep.service):

        Aug 18 09:02:44 ... fused-memory-flag-marker-sweep.sh[65377]:
            .../fused-memory-flag-marker-sweep.sh: line 46: exec: uv: not found
        Aug 18 09:02:44 ... fused-memory-flag-marker-sweep.service:
            Main process exited, code=exited, status=127/n/a

    Asserted on the unit the installer actually COPIED into place rather
    than on the in-repo template, so this is genuinely "the installer sets
    it up correctly at install time" coverage. /home/leo/.local/bin is where
    `uv` measurably lives; /usr/bin is needed so `systemctl`, `grep` and
    `cut` -- which the wrapper also shells out to -- resolve under the same
    minimal boot env."""
    xdg_config = tmp_path / "xdg-config"

    result = _run_script(fake_timer_systemctl, env={"XDG_CONFIG_HOME": str(xdg_config)})
    assert result.returncode == 0, (
        f"stdout={result.stdout!r} stderr={result.stderr!r}"
    )

    service_path = xdg_config / "systemd" / "user" / SERVICE_NAME
    installed = service_path.read_text()

    path_lines = [
        ln for ln in installed.splitlines() if ln.startswith("Environment=PATH=")
    ]
    assert path_lines, (
        f"Expected the INSTALLED unit to pin an explicit Environment=PATH= so "
        f"the Persistent=true boot catch-up run can find uv (OBSERVED "
        f"status=127 without it); installed unit=\n{installed}"
    )
    path_value = path_lines[0].split("=", 2)[2]
    assert "/home/leo/.local/bin" in path_value, (
        f"Expected the pinned PATH to include /home/leo/.local/bin, where uv "
        f"actually lives; path_value={path_value!r}"
    )
    assert "/usr/bin" in path_value, (
        f"Expected the pinned PATH to include /usr/bin so systemctl/grep/cut "
        f"resolve under the boot catch-up's minimal env; "
        f"path_value={path_value!r}"
    )
