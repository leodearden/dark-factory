"""shared.proc_group's leader-pidfd route, exercised with real processes.

A pidfd held on a group's leader names the group itself, not its recyclable
pgid number, so it can still SIGKILL the group's survivors after the leader
is reaped (task 6251).
"""
from __future__ import annotations

import asyncio
import contextlib
import errno
import logging
import os
import signal
import sys
from collections.abc import AsyncIterator
from dataclasses import dataclass
from types import SimpleNamespace
from typing import cast
from unittest.mock import Mock

import pytest
from _pidfd_kernel import needs_group_pidfd_signals
from df_pytest_isolation import wait_pid_gone

from shared.proc_group import (
    LeaderPidfd,
    SigkillDispatch,
    open_leader_pidfd,
    pid_running,
    sigkill_process_group,
)

pytestmark = pytest.mark.skipif(sys.platform != 'linux', reason='pidfds are Linux-only')


@dataclass(frozen=True)
class _ChildView:
    """A child as :func:`open_leader_pidfd` sees it, for a pid no real Process handle names."""

    pid: int
    returncode: int | None


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


@needs_group_pidfd_signals
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
            assert await asyncio.to_thread(wait_pid_gone, helper, timeout=5.0), (
                f'helper pid {helper} survived a group SIGKILL through the pidfd of its reaped leader'
            )
    finally:
        if helper is not None and pid_running(helper):
            os.kill(helper, signal.SIGKILL)


@needs_group_pidfd_signals
async def test_a_held_leader_pidfd_on_an_emptied_group_dispatches_nothing() -> None:
    """The stand-in's returncode still reads None, as between the watcher's reap and
    asyncio's exit callback, so only the pidfd stops a killpg of the freed number."""
    async with _session_leader('read _') as proc:
        pgid = proc.pid
        with open_leader_pidfd(proc) as fd:
            await proc.communicate(b'go\n')
            kill = Mock()
            not_yet_marked_reaped = cast(
                asyncio.subprocess.Process,
                SimpleNamespace(pid=proc.pid, returncode=None, kill=kill),
            )

            dispatch = sigkill_process_group(not_yet_marked_reaped, pgid, leader_pidfd=fd)

    assert dispatch is SigkillDispatch.NOT_SIGNALLED
    kill.assert_not_called()


async def test_a_held_leader_pidfd_does_not_bypass_the_pgid_sanity_check() -> None:
    async with _session_leader('read _') as proc:
        with open_leader_pidfd(proc) as fd:
            assert fd is not None

            dispatch = sigkill_process_group(proc, os.getpgrp(), leader_pidfd=fd)

        assert dispatch is SigkillDispatch.DIRECT_KILLED
        assert await proc.wait() == -signal.SIGKILL


async def test_a_leader_pidfd_the_kernel_cannot_signal_through_falls_back_to_killpg() -> None:
    """A kernel before Linux 6.9 refuses the group send (EINVAL); an fd that is no pidfd is refused too (EBADF)."""
    not_a_pidfd = os.open(os.devnull, os.O_RDONLY)
    try:
        async with _session_leader('read _') as proc:
            dispatch = sigkill_process_group(proc, proc.pid, leader_pidfd=LeaderPidfd(not_a_pidfd))

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
    with open_leader_pidfd(_ChildView(pid=os.getppid(), returncode=None)) as fd:
        assert fd is None


async def test_a_pidfd_open_failure_leaves_the_same_debug_line_as_every_refusal(
    caplog: pytest.LogCaptureFixture,
) -> None:
    """A reaped pid whose handle has not yet seen its returncode: pidfd_open answers ESRCH."""
    async with _session_leader('exit 0') as proc:
        await proc.wait()
        caplog.set_level(logging.DEBUG, logger='shared.proc_group')

        with open_leader_pidfd(_ChildView(pid=proc.pid, returncode=None)) as fd:
            assert fd is None

    assert f'open_leader_pidfd: no pidfd for pid {proc.pid} (' in caplog.text


async def test_the_pidfd_is_closed_when_the_context_exits() -> None:
    async with _session_leader('read _') as proc:
        with open_leader_pidfd(proc) as fd:
            held = fd
        assert held is not None

        with pytest.raises(OSError) as closed:
            os.fstat(held)
        assert closed.value.errno == errno.EBADF
        await proc.communicate(b'go\n')
