"""``SlotLease``: a slot's idempotent release, lending its permit to a borrower, and the slot-ended signal."""

from __future__ import annotations

import asyncio
from collections.abc import Coroutine
from typing import Any

import pytest
from _orch_helpers import wait_responsive

from orchestrator.slot_lease import SlotLease, SlotLendingError

pytestmark = pytest.mark.asyncio

LENDER = 'lender-7'
BORROWER = 'borrower-8'
SECOND = 'second-9'


class _RecordingScheduler:
    """Satisfies the lease's lock releaser and records every lock release."""

    def __init__(self) -> None:
        self.calls: list[tuple[str, bool]] = []

    def release(self, task_id: str, *, requeued: bool = False) -> None:
        self.calls.append((task_id, requeued))


class _RaisingScheduler(_RecordingScheduler):
    def release(self, task_id: str, *, requeued: bool = False) -> None:
        super().release(task_id, requeued=requeued)
        raise RuntimeError(f'lock release failed for {task_id}')


async def free_permits(sem: asyncio.Semaphore) -> int:
    """How many permits *sem* can hand out now, counted by taking and returning them."""
    taken = 0
    while not sem.locked():
        await sem.acquire()
        taken += 1
    for _ in range(taken):
        sem.release()
    return taken


async def _claimed(task_id: str, scheduler: _RecordingScheduler, pool: asyncio.Semaphore) -> SlotLease:
    await pool.acquire()
    return SlotLease(task_id, scheduler, pool)


def _completes_without_suspending(coro: Coroutine[Any, Any, None]) -> bool:
    try:
        coro.send(None)
    except StopIteration:
        return True
    coro.close()
    return False


@pytest.fixture
def scheduler() -> _RecordingScheduler:
    return _RecordingScheduler()


@pytest.fixture
def pool() -> asyncio.Semaphore:
    return asyncio.Semaphore(3)


async def test_a_holding_lease_returns_one_permit_and_releases_its_locks_once(
    scheduler: _RecordingScheduler, pool: asyncio.Semaphore,
) -> None:
    lease = await _claimed(LENDER, scheduler, pool)
    assert lease.task_id == LENDER
    assert lease.holds_permit
    assert lease.lender is None
    assert await free_permits(pool) == 2

    assert lease.release(requeued=True) is True
    assert await free_permits(pool) == 3
    assert scheduler.calls == [(LENDER, True)]
    assert lease.released
    assert not lease.holds_permit

    assert lease.release(requeued=False) is False
    assert await free_permits(pool) == 3
    assert scheduler.calls == [(LENDER, True)]


async def test_lend_builds_a_borrower_that_takes_no_permit(
    scheduler: _RecordingScheduler, pool: asyncio.Semaphore,
) -> None:
    lender = await _claimed(LENDER, scheduler, pool)
    before = await free_permits(pool)

    borrower = lender.lend(BORROWER)

    assert borrower.task_id == BORROWER
    assert not borrower.holds_permit
    assert borrower.lender is lender
    assert borrower.borrower is None
    assert not borrower.released
    assert lender.borrower is borrower
    assert lender.holds_permit
    assert await free_permits(pool) == before


async def test_a_lender_released_first_hands_its_permit_to_the_live_borrower(
    scheduler: _RecordingScheduler, pool: asyncio.Semaphore,
) -> None:
    lender = await _claimed(LENDER, scheduler, pool)
    borrower = lender.lend(BORROWER)
    before = await free_permits(pool)

    assert lender.release(requeued=True) is True

    assert await free_permits(pool) == before
    assert borrower.holds_permit
    assert borrower.lender is lender
    assert lender.borrower is None
    assert not lender.holds_permit
    assert scheduler.calls == [(LENDER, True)]

    assert borrower.release(requeued=False) is True
    assert await free_permits(pool) == before + 1
    assert scheduler.calls == [(LENDER, True), (BORROWER, False)]


async def test_a_borrower_released_first_returns_nothing_and_frees_its_lender_to_lend_again(
    scheduler: _RecordingScheduler, pool: asyncio.Semaphore,
) -> None:
    lender = await _claimed(LENDER, scheduler, pool)
    borrower = lender.lend(BORROWER)
    before = await free_permits(pool)

    assert borrower.release(requeued=False) is True

    assert await free_permits(pool) == before
    assert not borrower.holds_permit
    assert lender.holds_permit
    assert lender.borrower is None
    assert scheduler.calls == [(BORROWER, False)]

    second = lender.lend(SECOND)
    assert lender.borrower is second
    assert second.release(requeued=False) is True
    assert await free_permits(pool) == before
    assert lender.borrower is None

    assert lender.release(requeued=False) is True
    assert await free_permits(pool) == before + 1
    assert scheduler.calls == [(BORROWER, False), (SECOND, False), (LENDER, False)]


class TestLendRefusals:

    async def test_a_released_lease_cannot_lend(
        self, scheduler: _RecordingScheduler, pool: asyncio.Semaphore,
    ) -> None:
        lender = await _claimed(LENDER, scheduler, pool)
        lender.release(requeued=False)
        before = await free_permits(pool)

        with pytest.raises(SlotLendingError) as refused:
            lender.lend(BORROWER)

        message = str(refused.value)
        assert 'already released' in message
        assert LENDER in message and BORROWER in message
        assert lender.borrower is None
        assert await free_permits(pool) == before

    async def test_a_lease_with_a_live_borrower_cannot_lend_a_second_time(
        self, scheduler: _RecordingScheduler, pool: asyncio.Semaphore,
    ) -> None:
        lender = await _claimed(LENDER, scheduler, pool)
        borrower = lender.lend(BORROWER)
        before = await free_permits(pool)

        with pytest.raises(SlotLendingError) as refused:
            lender.lend(SECOND)

        message = str(refused.value)
        assert 'already lends' in message
        assert LENDER in message and SECOND in message and BORROWER in message
        assert lender.borrower is borrower
        assert lender.holds_permit and not borrower.holds_permit
        assert not lender.released and not borrower.released
        assert await free_permits(pool) == before

    async def test_a_borrower_that_has_not_inherited_cannot_lend(
        self, scheduler: _RecordingScheduler, pool: asyncio.Semaphore,
    ) -> None:
        lender = await _claimed(LENDER, scheduler, pool)
        borrower = lender.lend(BORROWER)
        before = await free_permits(pool)

        with pytest.raises(SlotLendingError) as refused:
            borrower.lend(SECOND)

        message = str(refused.value)
        assert 'holds no permit' in message
        assert BORROWER in message and SECOND in message
        assert borrower.borrower is None
        assert lender.borrower is borrower
        assert lender.holds_permit and not borrower.holds_permit
        assert await free_permits(pool) == before


@pytest.mark.parametrize('end_order', [('D', 'D2'), ('D2', 'D')])
async def test_a_borrower_that_inherited_lends_on_and_the_chain_conserves_the_permit(
    scheduler: _RecordingScheduler, end_order: tuple[str, str],
) -> None:
    pool = asyncio.Semaphore(1)
    p = await _claimed('P', scheduler, pool)
    d = p.lend('D')
    p.release(requeued=False)
    d2 = d.lend('D2')
    assert d2.lender is d and d.borrower is d2
    assert await free_permits(pool) == 0

    leases = {'D': d, 'D2': d2}
    for name in end_order:
        assert leases[name].release(requeued=False) is True
        if name != end_order[-1]:
            assert await free_permits(pool) == 0

    assert await free_permits(pool) == 1
    assert all(lease.released for lease in (p, d, d2))
    assert sorted(task_id for task_id, _ in scheduler.calls) == ['D', 'D2', 'P']


class TestSlotEndedSignal:

    async def test_the_waiter_wakes_after_the_permit_is_returned_and_the_locks_released(
        self, scheduler: _RecordingScheduler,
    ) -> None:
        pool = asyncio.Semaphore(1)
        lease = await _claimed(LENDER, scheduler, pool)
        seen: list[tuple[bool, list[tuple[str, bool]]]] = []

        async def waiter() -> None:
            await lease.wait_released()
            seen.append((pool.locked(), list(scheduler.calls)))

        waiting = asyncio.create_task(waiter())
        for _ in range(3):
            await asyncio.sleep(0)
        assert not waiting.done()

        lease.release(requeued=False)
        await wait_responsive(waiting, label='the slot-ended waiter wakes')

        assert seen == [(False, [(LENDER, False)])]

    async def test_the_waiter_wakes_after_the_permit_passes_to_the_borrower(
        self, scheduler: _RecordingScheduler,
    ) -> None:
        pool = asyncio.Semaphore(1)
        lender = await _claimed(LENDER, scheduler, pool)
        borrower = lender.lend(BORROWER)
        seen: list[tuple[bool, bool]] = []

        async def waiter() -> None:
            await lender.wait_released()
            seen.append((borrower.holds_permit, lender.borrower is None))

        waiting = asyncio.create_task(waiter())
        for _ in range(3):
            await asyncio.sleep(0)
        assert not waiting.done()

        lender.release(requeued=False)
        await wait_responsive(waiting, label='the slot-ended waiter wakes')

        assert seen == [(True, True)]
        assert pool.locked()

    async def test_waiting_on_a_released_lease_returns_at_once(
        self, scheduler: _RecordingScheduler, pool: asyncio.Semaphore,
    ) -> None:
        lease = await _claimed(LENDER, scheduler, pool)
        assert not _completes_without_suspending(lease.wait_released())

        lease.release(requeued=False)

        assert _completes_without_suspending(lease.wait_released())


class TestLockReleaseFailure:

    async def test_a_raising_lock_release_still_returns_the_permit_and_sets_the_signal(self) -> None:
        scheduler = _RaisingScheduler()
        pool = asyncio.Semaphore(1)
        lease = await _claimed(LENDER, scheduler, pool)

        with pytest.raises(RuntimeError, match='lock release failed'):
            lease.release(requeued=False)

        assert lease.released
        assert await free_permits(pool) == 1
        assert _completes_without_suspending(lease.wait_released())

    async def test_a_raising_lock_release_still_hands_the_permit_over_and_sets_the_signal(self) -> None:
        scheduler = _RaisingScheduler()
        pool = asyncio.Semaphore(1)
        lender = await _claimed(LENDER, scheduler, pool)
        borrower = lender.lend(BORROWER)

        with pytest.raises(RuntimeError, match='lock release failed'):
            lender.release(requeued=False)

        assert borrower.holds_permit
        assert lender.borrower is None
        assert await free_permits(pool) == 0
        assert _completes_without_suspending(lender.wait_released())
