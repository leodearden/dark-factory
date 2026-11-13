"""The watchdog's pid_alive copy answers exactly as shared.proc_liveness.pid_alive.

The watchdog runs as a bare ``python3 scripts/orchestrator-watchdog.py`` with no
shared bootstrap, so it keeps a copy; this file is what keeps that copy honest.
"""

from __future__ import annotations

import importlib.util
import os
import pathlib
import types

import pytest
from shared.proc_liveness import pid_alive
from shared.whole_tree_guard_family import WHOLE_TREE_SCAN_TEST_TIMEOUT

pytestmark = pytest.mark.timeout(WHOLE_TREE_SCAN_TEST_TIMEOUT)

REPO_ROOT = pathlib.Path(__file__).parents[2]
WATCHDOG_PATH = REPO_ROOT / "scripts" / "orchestrator-watchdog.py"
_SELF = "SELF"
_MOCKED_PID = 12345


def _load_watchdog() -> types.ModuleType:
    spec = importlib.util.spec_from_file_location("orchestrator_watchdog", WATCHDOG_PATH)
    assert spec is not None, f"Could not build spec from {WATCHDOG_PATH}"
    assert spec.loader is not None
    mod = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(mod)
    return mod


@pytest.fixture(scope="module")
def watchdog_pid_alive():
    return _load_watchdog()._pid_alive


@pytest.mark.parametrize(
    "value",
    [0, -1, True, False, None, "123", 1.5, _SELF, 2**31 - 1, 10**20],
    ids=["zero", "negative", "true", "false", "none", "str", "float", "self", "dead", "unrepresentable"],
)
def test_copy_matches_canonical_on_the_real_syscall(watchdog_pid_alive, value: object) -> None:
    if value == _SELF:
        value = os.getpid()

    assert watchdog_pid_alive(value) is pid_alive(value)


def _kill_raising(exc: BaseException | None):
    def fake_kill(pid: int, sig: int) -> None:
        if exc is not None:
            raise exc

    return fake_kill


@pytest.mark.parametrize(
    "exc",
    [None, ProcessLookupError(), PermissionError(), OSError(5, "io"), OverflowError()],
    ids=["signal-delivered", "no-such-process", "not-permitted", "other-oserror", "overflow"],
)
def test_copy_matches_canonical_on_every_syscall_outcome(
    watchdog_pid_alive, monkeypatch: pytest.MonkeyPatch, exc: BaseException | None
) -> None:
    monkeypatch.setattr(os, "kill", _kill_raising(exc))

    assert watchdog_pid_alive(_MOCKED_PID) is pid_alive(_MOCKED_PID)
