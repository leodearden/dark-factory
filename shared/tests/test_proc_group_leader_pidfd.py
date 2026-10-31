"""shared.proc_group's leader-pidfd route, exercised with real processes.

A pidfd held on a group's leader names the group itself, not its recyclable
pgid number, so it can still SIGKILL the group's survivors after the leader
is reaped (task 6251).
"""
from __future__ import annotations

import asyncio
import contextlib
import errno
import os
import signal
import sys
import time
from collections.abc import AsyncIterator
from types import SimpleNamespace
from typing import cast

import pytest

from shared.pidfd import PIDFD_SIGNAL_PROCESS_GROUP, pidfd_open, pidfd_send_signal
from shared.proc_group import (
    SigkillDispatch,
    open_leader_pidfd,
    pid_running,
    sigkill_process_group,
)

pytestmark = pytest.mark.skipif(sys.platform != 'linux', reason='pidfds are Linux-only')


def _kernel_signals_groups_through_pidfds() -> bool:
    """Signal 0 to our own pidfd's group: EINVAL before Linux 6.9, success or ESRCH after."""
    try:
        fd = pidfd_open(os.getpid())
    except OSError:
        return False
    try:
        pidfd_send_signal(fd, 0, flags=PIDFD_SIGNAL_PROCESS_GROUP)
    except ProcessLookupError:
        return True
    except OSError:
        return False
    finally:
        os.close(fd)
    return True


_needs_group_pidfd_signals = pytest.mark.skipif(
    sys.platform != 'linux' or not _kernel_signals_groups_through_pidfds(),
    reason='the kernel predates PIDFD_SIGNAL_PROCESS_GROUP (Linux 6.9)',
)


@contextlib.asynccontextmanager
async def _session_leader(script: str) -> AsyncIterator[asyncio.subprocess.Process]:
    """``sh -c script`` leading a new session, its stdin piped; killed and reaped on exit."""
    proc = await asyncio.create_subprocess_exec(
        'sh', '-c', script,
        stdin=asyncio.subprocess.PIPE,
        stdout=asyncio.subprocess.PIPE,
        stderr=asyncio.subprocess.DEVNULL,
        start_new_session=True,
    )
    try:
        yield proc
    finally:
        if proc.returncode is None:
            with contextlib.suppress(ProcessLookupError):
                proc.kill()
            await proc.wait()


def _wait_not_running(pid: int, timeout: float = 5.0) -> bool:
    """Poll until *pid* is gone or a zombie awaiting its subreaper; False at *timeout*."""
    deadline = time.monotonic() + timeout
    while pid_running(pid):
        if time.monotonic() >= deadline:
            return False
        time.sleep(0.05)
    return True


@_needs_group_pidfd_signals
async def test_a_held_leader_pidfd_sigkills_the_group_after_the_leader_is_reaped() -> None:
    helper = None
    try:
        async with _session_leader('sleep 60 >/dev/null 2>&1 & echo $!; read _') as proc:
            pgid = proc.pid
            with open_leader_pidfd(proc) as fd:
                assert fd is not None
                out, _ = await proc.communicate(b'go\n')
                helper = int(out)
                assert proc.returncode == 0
                assert pid_running(helper), 'the helper died on its own; nothing left to prove'

                dispatch = sigkill_process_group(proc, pgid, leader_pidfd=fd)

            assert dispatch is SigkillDispatch.GROUP_KILLED
            assert await asyncio.to_thread(_wait_not_running, helper), (
                f'helper pid {helper} survived a group SIGKILL through the pidfd of its reaped leader'
            )
    finally:
        if helper is not None and pid_running(helper):
            os.kill(helper, signal.SIGKILL)


@_needs_group_pidfd_signals
async def test_a_held_leader_pidfd_on_an_emptied_group_dispatches_nothing() -> None:
    async with _session_leader('read _') as proc:
        pgid = proc.pid
        with open_leader_pidfd(proc) as fd:
            await proc.communicate(b'go\n')

            dispatch = sigkill_process_group(proc, pgid, leader_pidfd=fd)

    assert dispatch is SigkillDispatch.NOT_SIGNALLED


async def test_a_leader_pidfd_the_kernel_cannot_signal_through_falls_back_to_killpg() -> None:
    """A kernel before Linux 6.9 refuses the group send (EINVAL); an fd that is no pidfd is refused too (EBADF)."""
    not_a_pidfd = os.open(os.devnull, os.O_RDONLY)
    try:
        async with _session_leader('read _') as proc:
            dispatch = sigkill_process_group(proc, proc.pid, leader_pidfd=not_a_pidfd)

            assert dispatch is SigkillDispatch.GROUP_KILLED
            assert await proc.wait() == -signal.SIGKILL
    finally:
        os.close(not_a_pidfd)


async def test_a_child_that_does_not_lead_its_own_group_gets_no_leader_pidfd_and_is_still_killed() -> None:
    """Its pidfd would name a group with no members, so a group send through it would signal nobody."""
    proc = await asyncio.create_subprocess_exec(
        'sleep', '60', stdout=asyncio.subprocess.DEVNULL, stderr=asyncio.subprocess.DEVNULL,
    )
    try:
        with open_leader_pidfd(proc) as fd:
            dispatch = sigkill_process_group(proc, proc.pid, leader_pidfd=fd)

        assert fd is None
        assert dispatch is SigkillDispatch.DIRECT_KILLED
        assert await proc.wait() == -signal.SIGKILL
    finally:
        if proc.returncode is None:
            with contextlib.suppress(ProcessLookupError):
                proc.kill()
            await proc.wait()


async def test_no_pidfd_is_opened_for_an_already_reaped_leader() -> None:
    """A reaped leader's number may already be a stranger's."""
    async with _session_leader('exit 0') as proc:
        await proc.wait()

        with open_leader_pidfd(proc) as fd:
            assert fd is None


def test_no_pidfd_is_handed_out_for_a_pid_that_is_not_this_process_s_child() -> None:
    stranger = cast(asyncio.subprocess.Process, SimpleNamespace(pid=os.getppid(), returncode=None))

    with open_leader_pidfd(stranger) as fd:
        assert fd is None


async def test_the_pidfd_is_closed_when_the_context_exits() -> None:
    async with _session_leader('read _') as proc:
        with open_leader_pidfd(proc) as fd:
            held = fd
        assert held is not None

        with pytest.raises(OSError) as closed:
            os.fstat(held)
        assert closed.value.errno == errno.EBADF
        await proc.communicate(b'go\n')
