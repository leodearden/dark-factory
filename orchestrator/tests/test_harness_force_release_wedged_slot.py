"""The harness force-releases a terminal task's slot that a hard-cancel never ended (task 6660).

The terminal-status watcher soft-cancels a workflow whose task row is already
done/cancelled, then hard-cancels it. A workflow that ignores every cancel used
to keep its slot and module locks until restart, because only ``_run_slot``'s
own ``finally`` released them. Now, ``terminal_status_force_release_polls``
polls past the hard-cancel, the watcher releases the slot and the locks itself,
emits ``workflow_slot_force_released`` and files one ``workflow_slot_wedged`` L1
naming the await that holds the slot. The wedges here are the two real shapes:
a terminal-cleanup entry and the body drain, each in its own child task of a
real ``CancellationScope``. When the zombie finally does unwind it must neither
release the slot again nor touch a fresh dispatch of the same task.
"""

from __future__ import annotations

import asyncio
import contextlib
from collections.abc import Awaitable, Callable, Iterator
from contextlib import contextmanager
from pathlib import Path
from unittest.mock import AsyncMock, MagicMock, patch

import pytest
from _workflow_helpers import _build_harness
from escalation.categories import EscalationCategory
from escalation.queue import EscalationQueue

from orchestrator.config import OrchestratorConfig
from orchestrator.event_store import EventType
from orchestrator.harness import Harness, TaskReport
from orchestrator.scheduler import Scheduler, TaskAssignment
from orchestrator.workflow_types import CancellationScope, WorkflowCancelled, WorkflowOutcome

TID = '42'
SLOT = f'workflow-{TID}'
HARD_CANCEL_POLLS = 2
FORCE_RELEASE_POLLS = 3
CLEANUP_GRACE = 10.0
BARRIER = 5.0
THIS_FILE = Path(__file__).name


async def _ignore_cancels_until(release: asyncio.Event) -> None:
    while not release.is_set():
        try:
            await release.wait()
        except asyncio.CancelledError:
            continue


class _ScopedWorkflow:
    """A workflow whose ``run()`` is a real CancellationScope over *body* with one
    terminal-cleanup entry, ``stop_claimant_heartbeat``; a hard-cancel exit is
    surfaced as the slot's CancelledError."""

    def __init__(
        self,
        body: Callable[[], Awaitable[None]],
        cleanup: Callable[[str | None], Awaitable[None]],
    ) -> None:
        self._body = body
        self._cleanup = cleanup
        self._steward = None

    async def run(self) -> None:
        scope = CancellationScope(
            asyncio.Event(), [('stop_claimant_heartbeat', self._cleanup)],
            cancel_grace_secs=CLEANUP_GRACE,
        )
        with contextlib.suppress(WorkflowCancelled):
            await scope.supervise(self._body())
        raise asyncio.CancelledError()


class _Rig:
    def __init__(
        self, harness: Harness, scheduler: Scheduler, queue: EscalationQueue, events: MagicMock,
    ) -> None:
        self.harness = harness
        self.scheduler = scheduler
        self.queue = queue
        self.events = events
        self.reports: list[TaskReport] = []
        self.releases: list[asyncio.Event] = []
        self.slots: list[asyncio.Task] = []

    def release_event(self) -> asyncio.Event:
        release = asyncio.Event()
        self.releases.append(release)
        return release

    def wedged_in_cleanup(self) -> tuple[_ScopedWorkflow, asyncio.Event]:
        release = self.release_event()

        async def _body() -> None:
            return None

        async def _cleanup_entry_ignoring_cancels(_kind: str | None) -> None:
            await _ignore_cancels_until(release)

        return _ScopedWorkflow(_body, _cleanup_entry_ignoring_cancels), release

    def wedged_in_body_drain(self) -> tuple[_ScopedWorkflow, asyncio.Event]:
        release = self.release_event()

        async def _body_ignoring_cancels() -> None:
            await _ignore_cancels_until(release)

        async def _cleanup(_kind: str | None) -> None:
            return None

        return _ScopedWorkflow(_body_ignoring_cancels, _cleanup), release

    async def retire(self) -> None:
        """Let every wedge through and wait for the slots, so a failing assertion
        can never leave the event loop holding a task that ignores cancels."""
        for release in self.releases:
            release.set()
        if self.slots:
            await asyncio.wait(self.slots, timeout=BARRIER)

    def launch(self, sem: asyncio.Semaphore) -> asyncio.Task:
        """Dispatch a slot for TID the way the harness's dispatch loop does."""
        assignment = TaskAssignment(task_id=TID, task={'id': TID, 'title': 'wedged'}, modules=['lib'])
        slot = self.harness._launch_slot(assignment, sem, self.reports)
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
            await asyncio.sleep(0)

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
        terminal_cleanup_cancel_grace_secs=CLEANUP_GRACE,
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
async def test_watcher_force_releases_a_slot_wedged_in_a_terminal_cleanup_entry(rig: _Rig):
    workflow, _release = rig.wedged_in_cleanup()
    sem = asyncio.Semaphore(0)
    assert rig.scheduler.lock_table.try_acquire(TID, ['lib'])
    try:
        with _built_workflows(workflow):
            slot = rig.launch(sem)
            await rig.wait_active()

            await rig.scan(HARD_CANCEL_POLLS + FORCE_RELEASE_POLLS - 1)
            assert sem.locked(), 'released before the backstop threshold'
            assert rig.scheduler.lock_table.is_held(TID)

            await rig.scan(1)

            assert not slot.done(), 'the slot must still be wedged: the harness released it, not its finally'
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
            culprit_line = l1s[0].detail.splitlines()[0]
            assert culprit_line.startswith(f'Culprit: {SLOT}/stop_claimant_heartbeat at ')
            assert THIS_FILE in culprit_line
            assert culprit_line.endswith(' in _ignore_cancels_until')
            assert f'{SLOT}:\n' in l1s[0].detail, 'the slot task itself is listed too'

            await rig.scan(3)
            assert len(rig.emitted(EventType.workflow_slot_force_released)) == 1
            assert len(rig.wedged_l1s()) == 1
    finally:
        await rig.retire()
    assert slot.done()


@pytest.mark.asyncio
async def test_watcher_force_releases_a_slot_wedged_draining_its_body(rig: _Rig):
    workflow, _release = rig.wedged_in_body_drain()
    sem = asyncio.Semaphore(0)
    assert rig.scheduler.lock_table.try_acquire(TID, ['lib'])
    try:
        with _built_workflows(workflow):
            slot = rig.launch(sem)
            await rig.wait_active()
            await rig.scan(HARD_CANCEL_POLLS + FORCE_RELEASE_POLLS)

            assert not slot.done()
            assert await _value_is_exactly_one(sem)
            assert not rig.scheduler.lock_table.is_held(TID)
            culprit_line = rig.wedged_l1s()[0].detail.splitlines()[0]
            assert culprit_line.startswith(f'Culprit: {SLOT}/body at ')
            assert THIS_FILE in culprit_line
    finally:
        await rig.retire()
    assert slot.done()


@pytest.mark.asyncio
async def test_zombie_finally_after_force_release_is_a_no_op(rig: _Rig):
    workflow, release = rig.wedged_in_cleanup()
    sem = asyncio.Semaphore(0)
    assert rig.scheduler.lock_table.try_acquire(TID, ['lib'])
    try:
        with _built_workflows(workflow):
            slot = rig.launch(sem)
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
    assert rig.reports == [], "the zombie's late report was recorded"


@pytest.mark.asyncio
async def test_zombie_unwinding_beside_a_redispatch_leaves_the_redispatch_alone(
    rig: _Rig, monkeypatch: pytest.MonkeyPatch,
):
    retry_cap_calls: list[tuple[str, WorkflowOutcome]] = []

    async def _recording_retry_cap(task_id: str, report: TaskReport, requeued: bool) -> bool:
        retry_cap_calls.append((task_id, report.outcome))
        return requeued

    monkeypatch.setattr(rig.harness, '_apply_retry_cap', _recording_retry_cap)
    zombie_workflow, zombie_release = rig.wedged_in_cleanup()
    successor_workflow, successor_release = rig.wedged_in_body_drain()
    zombie_sem = asyncio.Semaphore(0)
    successor_sem = asyncio.Semaphore(0)
    assert rig.scheduler.lock_table.try_acquire(TID, ['lib'])
    try:
        with _built_workflows(zombie_workflow, successor_workflow):
            zombie = rig.launch(zombie_sem)
            await rig.wait_active()
            await rig.scan(HARD_CANCEL_POLLS + FORCE_RELEASE_POLLS)
            assert not rig.harness.is_workflow_active(TID)

            assert rig.scheduler.lock_table.try_acquire(TID, ['lib'])
            successor = rig.launch(successor_sem)
            await rig.wait_active()
            successor_escalation_event = rig.harness._escalation_events.get(TID)
            assert successor_escalation_event is not None
            assert rig.harness.hard_cancel_workflow(TID, reason='action_teardown:restart')

            zombie_release.set()
            await asyncio.wait({zombie}, timeout=BARRIER)
            assert zombie.done()

            assert rig.harness.is_workflow_active(TID), "the zombie's finally unregistered the successor"
            assert rig.harness._escalation_events.get(TID) is successor_escalation_event, (
                "the zombie's dispatch callback dropped the successor's escalation event"
            )
            assert rig.scheduler.lock_table.is_held(TID), "the zombie's finally released the successor's locks"
            assert successor_sem.locked(), "the zombie's finally released the successor's slot"
            assert retry_cap_calls == [], "the zombie's finally ran the retry cap against the successor"
            assert rig.reports == [], "the zombie's late report was recorded"

            successor_release.set()
            await asyncio.wait({successor}, timeout=BARRIER)
    finally:
        await rig.retire()

    assert successor.done()
    assert await _value_is_exactly_one(successor_sem)
    assert not rig.scheduler.lock_table.is_held(TID)
    assert retry_cap_calls == [(TID, WorkflowOutcome.CANCELLED)]
    assert [(r.task_id, r.block_reason) for r in rig.reports] == [(TID, 'action_teardown:restart')], (
        "the zombie consumed the successor's cancel cause"
    )
