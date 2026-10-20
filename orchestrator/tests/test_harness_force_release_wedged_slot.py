"""The harness force-releases a terminal task's slot that a hard-cancel never ended (task 6660).

The terminal-status watcher soft-cancels a workflow whose task row is already
done/cancelled, then hard-cancels it. A workflow that ignores every cancel used
to keep its slot and module locks until restart, because only ``_run_slot``'s
own ``finally`` released them. Now, ``terminal_status_force_release_polls``
polls past the hard-cancel, the watcher releases the slot and the locks itself,
emits ``workflow_slot_force_released`` and files one ``workflow_slot_wedged`` L1.
When the zombie finally does unwind, its ``finally`` must neither release the
slot a second time nor unregister a fresh dispatch of the same task.
"""

from __future__ import annotations

import asyncio
from collections.abc import Iterator
from contextlib import contextmanager
from pathlib import Path
from unittest.mock import AsyncMock, MagicMock, patch

import pytest
from _workflow_helpers import _build_harness
from escalation.categories import EscalationCategory
from escalation.queue import EscalationQueue

from orchestrator.config import OrchestratorConfig
from orchestrator.event_store import EventType
from orchestrator.harness import Harness
from orchestrator.scheduler import Scheduler, TaskAssignment

TID = '42'
HARD_CANCEL_POLLS = 2
FORCE_RELEASE_POLLS = 3
BARRIER = 5.0


class _CancelImmuneWorkflow:
    """A workflow whose ``run()`` ignores every cancel until *release* is set,
    then lets the cancel through."""

    def __init__(self, release: asyncio.Event) -> None:
        self._release = release
        self._steward = None

    async def run(self) -> None:
        while not self._release.is_set():
            try:
                await self._release.wait()
            except asyncio.CancelledError:
                continue
        raise asyncio.CancelledError()


class _Rig:
    def __init__(
        self, harness: Harness, scheduler: Scheduler, queue: EscalationQueue, events: MagicMock,
    ) -> None:
        self.harness = harness
        self.scheduler = scheduler
        self.queue = queue
        self.events = events
        self.releases: list[asyncio.Event] = []
        self.slots: list[asyncio.Task] = []

    def immune_workflow(self) -> tuple[_CancelImmuneWorkflow, asyncio.Event]:
        release = asyncio.Event()
        self.releases.append(release)
        return _CancelImmuneWorkflow(release), release

    async def retire(self) -> None:
        """Let every immune workflow through and wait for the slots, so a failing
        assertion can never leave the event loop holding a task that ignores cancels."""
        for release in self.releases:
            release.set()
        if self.slots:
            await asyncio.wait(self.slots, timeout=BARRIER)

    def start_slot(self, sem: asyncio.Semaphore) -> asyncio.Task:
        assignment = TaskAssignment(task_id=TID, task={'id': TID, 'title': 'wedged'}, modules=['lib'])
        slot = asyncio.create_task(self.harness._run_slot(assignment, sem), name=f'workflow-{TID}')
        self.slots.append(slot)
        return slot

    async def wait_active(self) -> None:
        loop = asyncio.get_running_loop()
        deadline = loop.time() + BARRIER
        while not self.harness.is_workflow_active(TID):
            assert loop.time() < deadline, 'the slot never registered'
            await asyncio.sleep(0.01)

    async def scan(self, polls: int) -> None:
        for _ in range(polls):
            await self.harness._scan_for_terminal_active_tasks()

    def emitted(self, event_type: EventType) -> list:
        return [c for c in self.events.emit.call_args_list if c.args and c.args[0] is event_type]

    def wedged_l1s(self) -> list:
        return [
            e for e in self.queue.get_by_task(TID, status='pending')
            if e.category == EscalationCategory.WORKFLOW_SLOT_WEDGED
        ]


@pytest.fixture
def rig(tmp_path: Path, monkeypatch: pytest.MonkeyPatch) -> _Rig:
    config = OrchestratorConfig(
        project_root=tmp_path,
        terminal_status_hard_cancel_polls=HARD_CANCEL_POLLS,
        terminal_status_force_release_polls=FORCE_RELEASE_POLLS,
    )
    harness = _build_harness(config)
    scheduler = Scheduler(config)
    monkeypatch.setattr(scheduler, 'dispatch_tool', AsyncMock(return_value={}))
    monkeypatch.setattr(scheduler, 'get_statuses', AsyncMock(return_value=({TID: 'done'}, None)))
    harness.scheduler = scheduler
    events = MagicMock()
    harness.event_store = events
    queue = EscalationQueue(tmp_path / 'escalations')
    monkeypatch.setattr(harness, '_escalation_queue', queue)
    return _Rig(harness, scheduler, queue, events)


@contextmanager
def _built_workflows(*workflows: object) -> Iterator[None]:
    with patch('orchestrator.harness.build_workflow', side_effect=list(workflows)):
        yield


async def _value_is_exactly_one(sem: asyncio.Semaphore) -> bool:
    if sem.locked():
        return False
    await sem.acquire()
    try:
        return sem.locked()
    finally:
        sem.release()


@pytest.mark.asyncio
async def test_watcher_force_releases_a_cancel_immune_terminal_slot(rig: _Rig):
    workflow, release = rig.immune_workflow()
    sem = asyncio.Semaphore(0)
    assert rig.scheduler.lock_table.try_acquire(TID, ['lib'])
    try:
        with _built_workflows(workflow):
            slot = rig.start_slot(sem)
            await rig.wait_active()

            await rig.scan(HARD_CANCEL_POLLS + FORCE_RELEASE_POLLS - 1)
            assert sem.locked(), 'released before the backstop threshold'
            assert rig.scheduler.lock_table.is_held(TID)

            await rig.scan(1)

            assert not slot.done(), 'the stub must still be wedged: the harness released it, not its finally'
            assert await _value_is_exactly_one(sem)
            assert not rig.scheduler.lock_table.is_held(TID)
            assert not rig.harness.is_workflow_active(TID)
            released = rig.emitted(EventType.workflow_slot_force_released)
            assert len(released) == 1
            assert released[0].kwargs['task_id'] == TID
            assert released[0].kwargs['data']['status'] == 'done'
            assert released[0].kwargs['data']['polls'] == HARD_CANCEL_POLLS + FORCE_RELEASE_POLLS
            assert released[0].kwargs['data']['modules'] == ['lib']
            l1s = rig.wedged_l1s()
            assert len(l1s) == 1
            assert l1s[0].level == 1
            assert l1s[0].severity == 'blocking'
            assert 'test_harness_force_release_wedged_slot.py' in l1s[0].detail, 'the L1 names where the slot task is stuck'
            assert ' in run\n' in l1s[0].detail

            await rig.scan(3)
            assert len(rig.emitted(EventType.workflow_slot_force_released)) == 1
            assert len(rig.wedged_l1s()) == 1
    finally:
        await rig.retire()
    assert slot.done()


@pytest.mark.asyncio
async def test_zombie_finally_after_force_release_is_a_no_op(rig: _Rig):
    workflow, release = rig.immune_workflow()
    sem = asyncio.Semaphore(0)
    assert rig.scheduler.lock_table.try_acquire(TID, ['lib'])
    try:
        with _built_workflows(workflow):
            slot = rig.start_slot(sem)
            await rig.wait_active()
            await rig.scan(HARD_CANCEL_POLLS + FORCE_RELEASE_POLLS)
            assert await _value_is_exactly_one(sem)

            release.set()
            await asyncio.wait({slot}, timeout=BARRIER)
    finally:
        await rig.retire()

    assert slot.done()
    assert slot.exception() is None
    assert await _value_is_exactly_one(sem), 'the zombie finally released the slot a second time'
    assert not rig.scheduler.lock_table.is_held(TID)


@pytest.mark.asyncio
async def test_zombie_finally_does_not_unregister_a_redispatch(rig: _Rig):
    zombie_workflow, zombie_release = rig.immune_workflow()
    successor_workflow, _successor_release = rig.immune_workflow()
    zombie_sem = asyncio.Semaphore(0)
    successor_sem = asyncio.Semaphore(0)
    assert rig.scheduler.lock_table.try_acquire(TID, ['lib'])
    try:
        with _built_workflows(zombie_workflow, successor_workflow):
            zombie = rig.start_slot(zombie_sem)
            await rig.wait_active()
            await rig.scan(HARD_CANCEL_POLLS + FORCE_RELEASE_POLLS)
            assert not rig.harness.is_workflow_active(TID)

            assert rig.scheduler.lock_table.try_acquire(TID, ['lib'])
            successor = rig.start_slot(successor_sem)
            await rig.wait_active()

            zombie_release.set()
            await asyncio.wait({zombie}, timeout=BARRIER)
            assert zombie.done()

            assert rig.harness.is_workflow_active(TID), "the zombie's finally unregistered the successor"
            assert rig.scheduler.lock_table.is_held(TID), "the zombie's finally released the successor's locks"
            assert successor_sem.locked(), "the zombie's finally released the successor's slot"
            assert rig.harness.hard_cancel_workflow(TID), "the successor's slot task is no longer registered"
    finally:
        await rig.retire()

    assert successor.done()
    assert await _value_is_exactly_one(successor_sem)
    assert not rig.scheduler.lock_table.is_held(TID)
