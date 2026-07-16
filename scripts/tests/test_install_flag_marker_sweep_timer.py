"""Tests for scripts/install-flag-marker-sweep-timer.sh -- the fail-loud,
self-verifying installer for the fused-memory stage1_flag_marker sweep's
systemd user timer (task 2693).

Drives the script via subprocess with a FAKE `systemctl` shimmed onto PATH
(records every invocation, minus `--user`, into a shared JSON state file --
mirrors scripts/tests/test_install_trickle_timer.py). Real systemd is never
touched. Unlike the trickle-timer installer, this job is a single
non-templated dark_factory instance (no project_id argument, no per-project
config resolution), so the driver has no
INSTALL_TRICKLE_TIMER_PYTHON/LEGIBILITY_SEARCH_ROOTS equivalent. The fake
systemctl additionally handles `start <unit>` (always succeeds) since this
installer, unlike install-trickle-timer.sh, also kicks an immediate
one-time drain of the current backlog.
"""
from __future__ import annotations

import json
import os
import subprocess
from pathlib import Path

SCRIPT = Path(__file__).parent.parent / "install-flag-marker-sweep-timer.sh"
TEMPLATES_DIR = Path(__file__).parent.parent  # scripts/tests/../ = scripts/

SERVICE_NAME = "fused-memory-flag-marker-sweep.service"
TIMER_NAME = "fused-memory-flag-marker-sweep.timer"


# ---------------------------------------------------------------------------
# Fake systemctl (marker-file + canned enable/start/list-timers responses)
# ---------------------------------------------------------------------------

_FAKE_SYSTEMCTL_SRC = '''#!/usr/bin/env python3
"""Fake `systemctl` for testing install-flag-marker-sweep-timer.sh.

Records every invocation (minus `--user`) into a JSON state file at
$FAKE_SYSTEMCTL_STATE. `enable --now <unit>` marks each non-flag arg as an
enabled unit; `start <unit>` is recorded and always succeeds (the immediate
one-time drain kick); `list-timers` echoes back one line per *.timer unit
enabled so far THIS RUN, unless FAKE_SYSTEMCTL_OMIT_LIST_TIMERS=1 --
simulating a self-verify failure where `enable` nominally succeeded but the
unit is absent from `list-timers`.
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
    state.setdefault("calls", []).append(args)

    if verb == "daemon-reload":
        _save(state)
        return 0

    if verb == "enable":
        units = [a for a in rest if not a.startswith("-")]
        enabled = state.setdefault("enabled_timers", [])
        for u in units:
            if u not in enabled:
                enabled.append(u)
        _save(state)
        return 0

    if verb == "start":
        _save(state)
        return 0

    if verb == "list-timers":
        _save(state)
        if os.environ.get("FAKE_SYSTEMCTL_OMIT_LIST_TIMERS") == "1":
            print("0 timers listed.")
            return 0
        enabled = state.get("enabled_timers", [])
        for unit in enabled:
            service = unit.replace(".timer", ".service")
            print(f"Mon 2026-07-14 03:30:00 UTC  8h left  n/a  n/a  {unit}  {service}")
        print(f"{len(enabled)} timers listed.")
        return 0

    return 1


if __name__ == "__main__":
    sys.exit(main(sys.argv))
'''


def _fake_systemctl(tmp_path):
    """Write an executable fake `systemctl` into <tmp_path>/bin/systemctl and
    its backing JSON state file. Returns (bin_dir, state_path)."""
    bin_dir = tmp_path / "bin"
    bin_dir.mkdir(exist_ok=True)
    fake = bin_dir / "systemctl"
    fake.write_text(_FAKE_SYSTEMCTL_SRC)
    fake.chmod(0o755)

    state_path = tmp_path / "systemctl_state.json"
    state_path.write_text(json.dumps({"calls": [], "enabled_timers": []}))
    return bin_dir, state_path


def _systemctl_calls(tmp_path):
    state_path = tmp_path / "systemctl_state.json"
    if not state_path.is_file():
        return []
    return json.loads(state_path.read_text())["calls"]


# ---------------------------------------------------------------------------
# Script driver
# ---------------------------------------------------------------------------

def _run_script(tmp_path, *, env=None):
    """Run install-flag-marker-sweep-timer.sh via subprocess.

    Puts a fresh fake `systemctl` on PATH (state reset each call). Callers
    supply XDG_CONFIG_HOME via *env*.
    """
    bin_dir, state_path = _fake_systemctl(tmp_path)

    full_env = dict(os.environ)
    full_env["PATH"] = f"{bin_dir}{os.pathsep}{full_env['PATH']}"
    full_env["FAKE_SYSTEMCTL_STATE"] = str(state_path)
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


def test_install_copies_units_enables_timer_and_kicks_drain(tmp_path):
    xdg_config = tmp_path / "xdg-config"

    result = _run_script(tmp_path, env={"XDG_CONFIG_HOME": str(xdg_config)})

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

    calls = _systemctl_calls(tmp_path)
    assert ["daemon-reload"] in calls, f"calls={calls!r}"
    assert ["enable", "--now", TIMER_NAME] in calls, f"calls={calls!r}"
    assert ["start", SERVICE_NAME] in calls, f"calls={calls!r}"
