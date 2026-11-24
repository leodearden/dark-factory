"""The files fake_user_systemctl.py keeps under $FAKE_SYSTEMD_STATE.

The fake itself and the tests that drive it find those files here. Tests
cannot import the fake, which reads $FAKE_SYSTEMD_STATE when imported.
"""
from __future__ import annotations

import contextlib
import os
import signal
from pathlib import Path

PID_SUFFIX = ".pid"


def calls_log(state: Path) -> Path:
    """Every invocation's argv, one JSON list per line."""
    return state / "systemctl.log"


def pid_file(state: Path, unit: str) -> Path:
    return state / f"{unit}{PID_SUFFIX}"


def log_file(state: Path, unit: str) -> Path:
    """The service's stdout and stderr, across every start."""
    return state / f"{unit}.log"


def enabled_marker(state: Path, unit: str) -> Path:
    return state / f"{unit}.enabled"


def unit_pid(state: Path, unit: str) -> int:
    return int(pid_file(state, unit).read_text())


def kill_all_units(state: Path) -> None:
    """SIGKILL the process group of every service the fake recorded a pid for."""
    for recorded in state.glob(f"*{PID_SUFFIX}"):
        with contextlib.suppress(ProcessLookupError, ValueError):
            os.killpg(int(recorded.read_text()), signal.SIGKILL)
