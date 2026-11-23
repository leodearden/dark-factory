"""The rig's wait for a verify-task's recorded cold worktree (task 11470).

``await_recorded_cold_worktree`` waits on states, not on a clock: the record
naming a worktree that holds its cold ``.task/``, the verify-task child having
exited, or the watched call having ended.  Fixtures go through the production
record seam in ``orchestrator.verify_cancel``, and the exited child is the real
CLI, found by its request through ``ChildCli``.
"""

from __future__ import annotations

import asyncio
import contextlib
import functools
from collections.abc import AsyncIterator
from pathlib import Path
from typing import Any

import pytest
from _task_verify_integration_rig import (
    ChildCli,
    ChildExitedBeforeRecording,
    await_recorded_cold_worktree,
)

from orchestrator.verify_cancel import pgid_file, write_worktree_path

pytestmark = pytest.mark.asyncio


@contextlib.asynccontextmanager
async def a_call_still_running() -> AsyncIterator[asyncio.Future[Any]]:
    call = asyncio.ensure_future(asyncio.Event().wait())
    try:
        yield call
    finally:
        call.cancel()
        with contextlib.suppress(asyncio.CancelledError):
            await call


async def test_returns_the_recorded_worktree_once_it_holds_its_cold_task_dir(tmp_path: Path) -> None:
    pgf = pgid_file(tmp_path / 'base', 'req1')
    worktree = tmp_path / 'wt'
    worktree.mkdir()
    write_worktree_path(pgf, worktree)

    def provision_while_running() -> int | None:
        (worktree / '.task').mkdir(exist_ok=True)
        return None

    async with a_call_still_running() as call:
        recorded = await await_recorded_cold_worktree(
            pgf, child_returncode=provision_while_running, watching=call,
        )

    assert recorded == worktree
    assert (recorded / '.task').is_dir()


async def test_a_verify_task_child_that_exits_before_recording_fails_at_once_with_its_returncode(
    tmp_path: Path,
) -> None:
    pgf = pgid_file(tmp_path / 'base', 'req1')
    children = ChildCli()
    try:
        rc, _, _ = await children.invoke(['verify-task', '--request-id', 'req1', '--no-such-option'])
    finally:
        await children.close()

    assert rc != 0
    assert children.returncode_of('req1') == rc
    assert children.returncode_of('another-request') is None
    async with a_call_still_running() as call:
        with pytest.raises(ChildExitedBeforeRecording) as raised:
            await await_recorded_cold_worktree(
                pgf,
                child_returncode=functools.partial(children.returncode_of, 'req1'),
                watching=call,
            )

    assert raised.value.returncode == rc
    assert raised.value.pgf == pgf


async def test_a_watched_call_that_ends_first_reraises_its_error(tmp_path: Path) -> None:
    pgf = pgid_file(tmp_path / 'base', 'req1')

    async def dispatch_that_fails() -> None:
        raise RuntimeError('the dispatch failed')

    call = asyncio.ensure_future(dispatch_that_fails())
    with pytest.raises(RuntimeError) as ended:
        await call

    with pytest.raises(RuntimeError) as raised:
        await await_recorded_cold_worktree(pgf, child_returncode=lambda: None, watching=call)

    assert raised.value is ended.value
