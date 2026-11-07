"""shared.session_liveness's judge and capture_for_record against a real child process and fake proc roots.

Fake proc roots go through the public ``proc_root`` keyword and the legacy pid
predicate through the public ``pid_alive`` keyword; nothing private is patched.
"""

from __future__ import annotations

import dataclasses
import logging
import socket
import subprocess
from collections.abc import Callable, Iterator

import pytest
from _child_processes import spawned_children

from shared.process_identity import Liveness, ProcessIdentity, capture
from shared.session_liveness import capture_for_record, judge

_FORGED_BOOT_ID = '00000000-0000-0000-0000-000000000000'
_OTHER_HOST = 'other-host.invalid'


@pytest.fixture
def live_child() -> Iterator[subprocess.Popen]:
    with spawned_children() as spawn:
        yield spawn(['sleep', '600'])


def _captured(pid: int) -> ProcessIdentity:
    identity = capture(pid)
    assert identity is not None, f'pid {pid} should be capturable while it runs'
    return identity


def _never_called(pid: int) -> bool:
    raise AssertionError(f'pid_alive({pid}) called although an identity decides')


def _recording(answer: bool) -> tuple[Callable[[int], bool], list[int]]:
    asked: list[int] = []

    def pid_alive(pid: int) -> bool:
        asked.append(pid)
        return answer

    return pid_alive, asked


def test_a_running_identity_is_alive_without_asking_the_pid_predicate(live_child):
    identity = _captured(live_child.pid)

    assert judge(identity, live_child.pid, pid_alive=_never_called) is Liveness.ALIVE


def test_a_reused_pid_is_dead_while_the_pid_still_runs(live_child):
    identity = _captured(live_child.pid)
    reused = dataclasses.replace(identity, start_ticks=identity.start_ticks + 1)

    assert judge(reused, live_child.pid, pid_alive=_never_called) is Liveness.DEAD


def test_a_reboot_is_dead_while_the_pid_still_runs(live_child):
    rebooted = dataclasses.replace(_captured(live_child.pid), boot_id=_FORGED_BOOT_ID)

    assert judge(rebooted, live_child.pid, pid_alive=_never_called) is Liveness.DEAD


def test_an_exited_and_reaped_identity_is_dead(live_child):
    identity = _captured(live_child.pid)

    live_child.kill()
    live_child.wait()

    assert judge(identity, live_child.pid, pid_alive=_never_called) is Liveness.DEAD


def test_another_host_is_other_host_before_any_proc_read(live_child, tmp_path):
    elsewhere = dataclasses.replace(_captured(live_child.pid), host=_OTHER_HOST)

    assert judge(elsewhere, live_child.pid, pid_alive=_never_called) is Liveness.OTHER_HOST
    assert (
        judge(elsewhere, live_child.pid, pid_alive=_never_called, proc_root=tmp_path)
        is Liveness.OTHER_HOST
    )


@pytest.mark.parametrize(
    ('answer', 'expected'), [(True, Liveness.ALIVE), (False, Liveness.DEAD)]
)
def test_no_identity_is_judged_by_the_injected_pid_predicate(answer, expected):
    pid_alive, asked = _recording(answer)

    assert judge(None, 4242, pid_alive=pid_alive) is expected
    assert asked == [4242]


def test_an_unreadable_proc_is_alive_and_warns(live_child, tmp_path, caplog):
    identity = _captured(live_child.pid)

    with caplog.at_level(logging.WARNING, logger='shared.session_liveness'):
        verdict = judge(identity, live_child.pid, pid_alive=_never_called, proc_root=tmp_path)

    assert verdict is Liveness.ALIVE
    assert [r.levelno for r in caplog.records] == [logging.WARNING]


def test_an_identity_naming_another_pid_is_refused_with_both_pids(live_child):
    identity = _captured(live_child.pid)
    other_pid = live_child.pid + 1

    with pytest.raises(ValueError) as refused:
        judge(identity, other_pid, pid_alive=_never_called)

    assert str(live_child.pid) in str(refused.value)
    assert str(other_pid) in str(refused.value)


def test_capture_for_record_is_the_live_identity(live_child):
    identity = capture_for_record(live_child.pid)

    assert identity is not None
    assert identity == _captured(live_child.pid)
    assert identity.host == socket.gethostname()


def test_capture_for_record_of_an_exited_and_reaped_pid_is_none(live_child):
    live_child.kill()
    live_child.wait()

    assert capture_for_record(live_child.pid) is None


def test_capture_for_record_reads_an_unreadable_proc_as_none_and_warns(
    live_child, tmp_path, caplog
):
    with caplog.at_level(logging.WARNING, logger='shared.session_liveness'):
        identity = capture_for_record(live_child.pid, proc_root=tmp_path)

    assert identity is None
    assert [r.levelno for r in caplog.records] == [logging.WARNING]
