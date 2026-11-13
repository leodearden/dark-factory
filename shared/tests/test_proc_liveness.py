"""shared.proc_liveness.pid_alive: the one pid-only liveness predicate.

The predicate is fed values parsed from untrusted input, so the contract is
that it answers a bool for ANY value and never raises.
"""

from __future__ import annotations

import os

import pytest

from shared.proc_liveness import pid_alive

_DEAD_PID = 2**31 - 1
_UNREPRESENTABLE_PIDS = (10**20, 2**62)
_MOCKED_PID = 12345


def test_this_process_is_alive() -> None:
    assert pid_alive(os.getpid()) is True


def test_a_pid_no_process_holds_is_dead() -> None:
    assert pid_alive(_DEAD_PID) is False


@pytest.mark.parametrize('pid', _UNREPRESENTABLE_PIDS, ids=['10**20', '2**62'])
def test_a_pid_no_pid_t_can_hold_is_dead_without_raising(pid: int) -> None:
    assert pid_alive(pid) is False


@pytest.mark.parametrize(
    'value',
    [0, -1, True, False, None, '123', 1.5],
    ids=['zero', 'negative', 'true', 'false', 'none', 'str', 'float'],
)
def test_a_non_positive_or_non_int_value_never_reaches_the_syscall(
    monkeypatch: pytest.MonkeyPatch, value: object
) -> None:
    signalled: list[tuple[int, int]] = []
    monkeypatch.setattr(os, 'kill', lambda pid, sig: signalled.append((pid, sig)))

    assert pid_alive(value) is False
    assert signalled == []


def _kill_raising(exc: BaseException | None):
    def fake_kill(pid: int, sig: int) -> None:
        assert (pid, sig) == (_MOCKED_PID, 0)
        if exc is not None:
            raise exc

    return fake_kill


@pytest.mark.parametrize(
    ('exc', 'alive'),
    [
        (None, True),
        (ProcessLookupError(), False),
        (PermissionError(), True),
        (OSError(5, 'io'), False),
        (OverflowError(), False),
    ],
    ids=['signal-delivered', 'no-such-process', 'not-permitted', 'other-oserror', 'overflow'],
)
def test_the_syscall_outcome_decides_liveness(
    monkeypatch: pytest.MonkeyPatch, exc: BaseException | None, alive: bool
) -> None:
    monkeypatch.setattr(os, 'kill', _kill_raising(exc))

    assert pid_alive(_MOCKED_PID) is alive
