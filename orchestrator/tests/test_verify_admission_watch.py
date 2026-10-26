"""The live admission watch turns the gate's snapshot into runs.db rows (task 5628).

PRD plans/verify-admission-fifo-prd.md, boundary scenario 12: while any waiter is
queued, each tick writes one ``verify_admission_heartbeat``; the oldest waiter
past the alarm age is surfaced once as ``verify_admission_wait_aged``, and again
only when that waiter's identity changes. Driven through ``gate_for`` and
``AdmissionWatch.tick`` with a real EventStore.
"""

from __future__ import annotations

import asyncio
from collections.abc import Callable
from pathlib import Path

import pytest

from orchestrator.event_store import EventStore, EventType
from orchestrator.verify_admission_gate import (
    AdmissionPlan,
    AdmissionPolicy,
    AdmissionRequest,
    gate_for,
)
from orchestrator.verify_admission_watch import AdmissionWatch

_ONE_SLOT_ARRIVAL = AdmissionPolicy(slots=1, order='arrival', promote_after_secs=7200.0)


async def _until(
    predicate: Callable[[], bool], *, watching: asyncio.Task, timeout: float = 5.0,
) -> None:
    async with asyncio.timeout(timeout):
        while not predicate():
            if watching.done():
                watching.result()
                raise AssertionError('the leg finished before reaching the awaited state')
            await asyncio.sleep(0.005)


class _ParkedLegs:
    """Legs admitted on one gate, each in its own task, parked until released."""

    def __init__(self, slots_dir: Path) -> None:
        self.slots_dir = slots_dir
        self.gate = gate_for(slots_dir)
        self.requests: dict[str, AdmissionRequest] = {}
        self._tasks: dict[str, asyncio.Task] = {}
        self._release: dict[str, asyncio.Event] = {}

    def _on_gate(self, task_id: str) -> bool:
        snapshot = self.gate.snapshot()
        views = (*snapshot.waiters, *snapshot.holders)
        return any(view.request.task_id == task_id for view in views)

    async def start(self, task_id: str, module: str) -> None:
        request = AdmissionPlan.begin(role='task', task_id=task_id).request(
            module=module, scope_kind=None,
        )
        release = asyncio.Event()

        async def leg() -> None:
            async with self.gate.admit(request, lambda: _ONE_SLOT_ARRIVAL):
                await release.wait()

        task = asyncio.create_task(leg())
        self.requests[task_id] = request
        self._tasks[task_id] = task
        self._release[task_id] = release
        await _until(lambda: self._on_gate(task_id), watching=task)

    async def cancel(self, task_id: str) -> None:
        task = self._tasks.pop(task_id)
        task.cancel()
        with pytest.raises(asyncio.CancelledError):
            async with asyncio.timeout(5):
                await task

    async def finish(self) -> None:
        for release in self._release.values():
            release.set()
        async with asyncio.timeout(5):
            await asyncio.gather(*self._tasks.values())
        self._tasks.clear()


def _heartbeats(store: EventStore) -> list[dict]:
    return [row['data'] for row in store.fetch_events_by_type(EventType.verify_admission_heartbeat)]


def _aged(store: EventStore) -> list[dict]:
    return [row['data'] for row in store.fetch_events_by_type(EventType.verify_admission_wait_aged)]


@pytest.fixture
def store(tmp_path: Path) -> EventStore:
    return EventStore(tmp_path / 'runs.db', run_id='t')


@pytest.mark.asyncio
async def test_a_queued_waiter_gives_one_heartbeat_naming_the_oldest_and_the_holders(
    tmp_path: Path, store: EventStore,
):
    legs = _ParkedLegs(tmp_path / 'slots')
    try:
        await legs.start('H', 'm1')
        await legs.start('W', 'm2')

        AdmissionWatch().tick(store, slots_dir=legs.slots_dir, alarm_secs=3600, order='arrival')

        [beat] = _heartbeats(store)
        assert beat['depth'] == 1
        assert beat['order'] == 'arrival'
        assert beat['oldest_wait_secs'] > 0
        assert beat['oldest'] == {
            'task_id': 'W', 'module': 'm2', 'role': 'task',
            'plan_id': legs.requests['W'].plan_id,
        }
        [holder] = beat['holders']
        assert holder['task_id'] == 'H'
        assert holder['module'] == 'm1'
        assert holder['role'] == 'task'
        assert holder['held_secs'] >= 0
        assert _aged(store) == []
    finally:
        await legs.finish()


@pytest.mark.asyncio
async def test_an_aged_waiter_is_surfaced_once_however_many_ticks_see_it(
    tmp_path: Path, store: EventStore,
):
    legs = _ParkedLegs(tmp_path / 'slots')
    try:
        await legs.start('H', 'm1')
        await legs.start('W', 'm2')
        await asyncio.sleep(0.1)

        watch = AdmissionWatch()
        for _ in range(3):
            watch.tick(store, slots_dir=legs.slots_dir, alarm_secs=0.05, order='arrival')

        assert len(_heartbeats(store)) == 3
        [aged] = _aged(store)
        assert aged['waiter']['task_id'] == 'W'
        assert aged['waiter']['module'] == 'm2'
        assert aged['waiter']['role'] == 'task'
        assert aged['waiter']['plan_id'] == legs.requests['W'].plan_id
        assert aged['waiter']['wait_secs'] >= 0.05
        assert aged['aged_count'] == 1
        assert aged['alarm_secs'] == 0.05
    finally:
        await legs.finish()


@pytest.mark.asyncio
async def test_a_new_oldest_aged_waiter_is_surfaced_again(tmp_path: Path, store: EventStore):
    legs = _ParkedLegs(tmp_path / 'slots')
    watch = AdmissionWatch()

    def tick() -> None:
        watch.tick(store, slots_dir=legs.slots_dir, alarm_secs=0.05, order='arrival')

    try:
        await legs.start('H', 'm1')
        await legs.start('W', 'm2')
        await asyncio.sleep(0.1)
        tick()
        await legs.start('W2', 'm3')
        await asyncio.sleep(0.1)
        tick()
        assert [row['waiter']['task_id'] for row in _aged(store)] == ['W']

        await legs.cancel('W')
        tick()
        tick()

        assert [row['waiter']['task_id'] for row in _aged(store)] == ['W', 'W2']
        assert _aged(store)[-1]['aged_count'] == 1
    finally:
        await legs.finish()


@pytest.mark.asyncio
async def test_no_queued_waiter_writes_no_heartbeat(tmp_path: Path, store: EventStore):
    legs = _ParkedLegs(tmp_path / 'slots')
    watch = AdmissionWatch()
    try:
        await legs.start('H', 'm1')
        watch.tick(store, slots_dir=legs.slots_dir, alarm_secs=0.05, order='arrival')
        await legs.start('W', 'm2')
    finally:
        await legs.finish()

    watch.tick(store, slots_dir=legs.slots_dir, alarm_secs=0.05, order='arrival')

    assert _heartbeats(store) == []
    assert _aged(store) == []
