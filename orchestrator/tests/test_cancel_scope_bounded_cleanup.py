"""CancellationScope bounds its terminal cleanup once a hard-cancel arrives (task 6660).

Before 6660 the scope's ``on_terminal`` cleanup absorbed every outer cancel
with no bound, so one cleanup entry that survived a cancel kept the slot task
alive for ever: the harness's ``hard_cancel_workflow`` polls were swallowed and
the slot and its module locks were never released. Now the first outer cancel
starts a grace clock for the running entry and every later one; an entry still
running when its grace runs out is cancelled under a bound, named in
``abandoned_entries``, and the remaining entries still run.
"""

from __future__ import annotations

import asyncio
from collections.abc import Awaitable, Callable

import pytest
from _orch_helpers import wait_responsive

from orchestrator.workflow_types import CancellationScope, WorkflowCancelled

_GRACE = 0.1
_BARRIER = 5.0

Entry = tuple[str, Callable[[str | None], Awaitable[None]]]


def _recorder(name: str, ran: list[str]) -> Entry:
    async def _fn(_kind: str | None) -> None:
        ran.append(name)

    return name, _fn


def _cancel_proof(name: str, entered: asyncio.Event, release: asyncio.Event, survivals: int) -> Entry:
    """An entry that ignores the first *survivals* cancels it is sent.

    ``survivals=1`` is the shape of ``_leaked_task_drain_probe.py``; a large
    value is a fully cancel-immune await. *release* lets the test retire it.
    """

    async def _fn(_kind: str | None) -> None:
        entered.set()
        absorbed = 0
        while not release.is_set():
            try:
                await release.wait()
            except asyncio.CancelledError:
                if absorbed == survivals:
                    raise
                absorbed += 1

    return name, _fn


async def _hard_cancel_until_done(outer: asyncio.Task, *, every: float, budget: float) -> None:
    """Call ``outer.cancel()`` every *every* seconds, as the watcher's polls do."""
    loop = asyncio.get_running_loop()
    deadline = loop.time() + budget
    while not outer.done() and loop.time() < deadline:
        outer.cancel()
        await asyncio.wait({outer}, timeout=every)


async def _body_returns() -> str:
    return 'body-result'


@pytest.mark.asyncio
async def test_one_cancel_proof_entry_is_abandoned_after_grace_and_later_entries_run():
    ran: list[str] = []
    entered = asyncio.Event()
    release = asyncio.Event()
    scope = CancellationScope(
        asyncio.Event(),
        [
            _recorder('a', ran),
            _cancel_proof('wedge', entered, release, survivals=1),
            _recorder('c', ran),
        ],
        cancel_grace_secs=_GRACE,
    )

    outer = asyncio.create_task(scope.supervise(_body_returns()))
    await wait_responsive(entered.wait(), timeout=_BARRIER, label='the cancel-proof entry starts')
    loop = asyncio.get_running_loop()
    started = loop.time()
    await _hard_cancel_until_done(outer, every=_GRACE / 4, budget=_BARRIER)

    assert outer.done()
    assert loop.time() - started < 4 * _GRACE + 1.0
    assert scope.abandoned_entries == ('wedge',)
    assert ran == ['a', 'c']
    assert outer.result() == 'body-result'
    release.set()


@pytest.mark.asyncio
async def test_fully_cancel_immune_entry_is_abandoned_and_scope_still_returns():
    ran: list[str] = []
    entered = asyncio.Event()
    release = asyncio.Event()
    scope = CancellationScope(
        asyncio.Event(),
        [_cancel_proof('immune', entered, release, survivals=1_000_000), _recorder('after', ran)],
        cancel_grace_secs=_GRACE,
    )

    async def _body_waits() -> None:
        await asyncio.sleep(3600)

    outer = asyncio.create_task(scope.supervise(_body_waits()))
    await asyncio.sleep(0)
    outer.cancel()
    await wait_responsive(entered.wait(), timeout=_BARRIER, label='the cancel-proof entry starts')
    await _hard_cancel_until_done(outer, every=_GRACE / 4, budget=_BARRIER)

    assert outer.done()
    exc = outer.exception()
    assert isinstance(exc, WorkflowCancelled)
    assert exc.kind == 'hard'
    assert scope.abandoned_entries == ('immune',)
    assert ran == ['after']
    release.set()
    await asyncio.sleep(0)


@pytest.mark.asyncio
async def test_no_outer_cancel_means_no_bound():
    ran: list[str] = []

    async def _slow(_kind: str | None) -> None:
        await asyncio.sleep(2 * _GRACE)
        ran.append('slow')

    scope = CancellationScope(
        asyncio.Event(), [('slow', _slow), _recorder('after', ran)], cancel_grace_secs=_GRACE,
    )

    assert await scope.supervise(_body_returns()) == 'body-result'
    assert ran == ['slow', 'after']
    assert scope.abandoned_entries == ()
