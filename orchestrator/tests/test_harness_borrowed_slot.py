"""Slots launched on a lease, and a slot borrowed from a live one (task 10475).

Every slot runs on the ``SlotLease`` it was launched with, built at dispatch.
A borrowing lease takes no permit: it inherits its lender's when the lender
ends first, and returns nothing when it ends first
(``plans/early-start-on-predecessor-branch-prd.md`` section 5.3 "Slot",
invariant 4). These tests drive the real ``_run_slot`` over a real
``Scheduler`` and a real ``RunSlots`` pool, with ``build_workflow`` patched to
a gated stub, and the real ``Harness.run`` loop through ``_run_loop_board``.
"""

from __future__ import annotations

import asyncio
import contextlib
from collections.abc import AsyncIterator
from enum import Enum
from pathlib import Path
from unittest.mock import MagicMock

import pytest
import pytest_asyncio
from _orch_helpers import HermeticMcpSession, wait_responsive
from _run_loop_board import (
    build_board_harness,
    fleet_dir_env,
    plain,
    run_driven,
    run_ready_callbacks,
    until,
)
from _workflow_helpers import _build_harness

from orchestrator.config import OrchestratorConfig
from orchestrator.harness import Harness, TaskReport
from orchestrator.run_slots import DispatchScope, RunSlots
from orchestrator.scheduler import Scheduler, TaskAssignment
from orchestrator.slot_lease import LendingRule, SlotLease, SlotLendingError
from orchestrator.workflow import TerminalReport, WorkflowOutcome, WorkflowState
from orchestrator.workflow_types import CancellationScope, WorkflowCancelled

HARD_CANCEL_POLLS = 2
FORCE_RELEASE_POLLS = 3
CLEANUP_GRACE = 10.0
_UNIT = 'orchestrator-borrowed-slot-test.service'


@pytest.fixture(autouse=True)
def _fleet_dir(tmp_path: Path, monkeypatch: pytest.MonkeyPatch) -> None:
    fleet_dir_env(monkeypatch, tmp_path, _UNIT)


class _Exit(Enum):
    """How a slot ends."""

    DONE = 'done'
    REQUEUE = 'requeue'
    EXCEPTION = 'exception'
    SHUTDOWN = 'shutdown'
    FORCED_RELEASE = 'forced-release'


_EXITS = list(_Exit)


def _exit_id(exit_: _Exit) -> str:
    return exit_.value


class _GatedWorkflow:
    """Stands in for ``TaskWorkflow``: ``run()`` waits for its task's gate, then ends as *exit_* says."""

    def __init__(self, gate: asyncio.Event, exit_: _Exit) -> None:
        self._gate = gate
        self._exit = exit_
        self._steward = None
        self.metrics = MagicMock(
            total_cost_usd=0.0,
            total_duration_ms=0,
            agent_invocations=0,
            execute_iterations=0,
            verify_attempts=0,
            review_cycles=0,
        )

    async def run(self) -> TerminalReport:
        await self._gate.wait()
        if self._exit is _Exit.EXCEPTION:
            raise RuntimeError('the workflow failed')
        if self._exit is _Exit.REQUEUE:
            return TerminalReport(
                outcome=WorkflowOutcome.REQUEUED, reason='', phase=WorkflowState.PLAN, detail='', category=None,
            )
        return TerminalReport(
            outcome=WorkflowOutcome.DONE, reason='', phase=WorkflowState.DONE, detail='', category=None,
        )


async def _ignore_cancels_until(release: asyncio.Event) -> None:
    while not release.is_set():
        try:
            await release.wait()
        except asyncio.CancelledError:
            continue


class _WedgedWorkflow:
    """A real ``CancellationScope`` whose body ignores every cancel until *release*.

    A hard-cancel exit is surfaced as the slot's CancelledError.
    """

    def __init__(self, release: asyncio.Event) -> None:
        self._release = release
        self._steward = None

    async def run(self) -> None:
        scope = CancellationScope(
            asyncio.Event(), [('stop_claimant_heartbeat', self._cleanup)], cancel_grace_secs=CLEANUP_GRACE,
        )
        with contextlib.suppress(WorkflowCancelled):
            await scope.supervise(_ignore_cancels_until(self._release))
        raise asyncio.CancelledError()

    async def _cleanup(self, _kind: str | None) -> None:
        return None


def _assignment(tid: str) -> TaskAssignment:
    module = f'{tid}.py'
    task = {'id': tid, 'title': f'Task {tid}', 'status': 'pending', 'dependencies': []}
    return TaskAssignment(task_id=tid, task=task, modules=[module])


class _Rig:
    """A real harness slot path over a real scheduler and a one-permit ``RunSlots`` pool."""

    def __init__(self, harness: Harness, scheduler: Scheduler, slots: RunSlots) -> None:
        self.harness = harness
        self.scheduler = scheduler
        self.slots = slots
        self.reports: list[TaskReport] = []
        self.exits: dict[str, _Exit] = {}
        self.statuses: dict[str, str] = {}
        self.gates: dict[str, asyncio.Event] = {}
        self.wedge_releases: list[asyncio.Event] = []
        self.built: list[str] = []
        self.slot_of: dict[str, asyncio.Task] = {}

    def gate(self, tid: str) -> asyncio.Event:
        return self.gates.setdefault(tid, asyncio.Event())

    def build_workflow(self, *, assignment: TaskAssignment, **_kwargs: object) -> _GatedWorkflow | _WedgedWorkflow:
        tid = assignment.task_id
        self.built.append(tid)
        exit_ = self.exits.get(tid, _Exit.DONE)
        if exit_ is _Exit.FORCED_RELEASE:
            release = asyncio.Event()
            self.wedge_releases.append(release)
            return _WedgedWorkflow(release)
        return _GatedWorkflow(self.gate(tid), exit_)

    async def get_statuses(self, *_args: object, **_kwargs: object) -> tuple[dict[str, str], None]:
        return dict(self.statuses), None

    def assignment(self, tid: str) -> TaskAssignment:
        """*tid*'s assignment, with its module locks taken as the scheduler's dispatch takes them."""
        assignment = _assignment(tid)
        assert self.scheduler.lock_table.try_acquire(tid, list(assignment.modules))
        return assignment

    async def holding_lease(self, tid: str) -> SlotLease:
        sem = await self.slots.claim(DispatchScope.ANY, {'id': tid})
        return SlotLease(tid, self.scheduler, sem)

    async def outside_lease(self, tid: str) -> SlotLease:
        """A lease on a permit outside the rig's pool, so *tid* runs beside P without taking P's permit."""
        pool = asyncio.Semaphore(1)
        await pool.acquire()
        return SlotLease(tid, self.scheduler, pool)

    def launch(self, assignment: TaskAssignment, lease: SlotLease) -> asyncio.Task:
        slot = self.harness._launch_slot(assignment, lease, self.reports)
        self.slot_of[assignment.task_id] = slot
        return slot

    def launch_borrowing(self, assignment: TaskAssignment, lender_tid: str) -> asyncio.Task:
        slot = self.harness._launch_borrowing_slot(assignment, lender_tid, self.reports)
        self.slot_of[assignment.task_id] = slot
        return slot

    async def launch_lender(self) -> SlotLease:
        """Lender P on a holding lease, running; returns its lease."""
        p_lease = await self.holding_lease('P')
        self.launch(self.assignment('P'), p_lease)
        await self.wait_active('P')
        return p_lease

    async def launch_pair(self) -> SlotLease:
        """Lender P on a holding lease, then D borrowing from it; returns P's lease."""
        p_lease = await self.launch_lender()
        self.launch_borrowing(self.assignment('D'), 'P')
        await self.wait_active('D')
        return p_lease

    async def wait_active(self, tid: str) -> None:
        await until(lambda: self.harness.is_workflow_active(tid), f'slot {tid} registered')

    async def scan(self, polls: int) -> None:
        for _ in range(polls):
            await self.harness._scan_for_terminal_active_tasks()
            await asyncio.sleep(0)

    async def end(self, tid: str) -> None:
        """End *tid*'s slot the way ``exits[tid]`` says."""
        slot = self.slot_of[tid]
        exit_ = self.exits.get(tid, _Exit.DONE)
        if exit_ is _Exit.FORCED_RELEASE:
            self.statuses[tid] = 'done'
            await self.scan(HARD_CANCEL_POLLS + FORCE_RELEASE_POLLS)
            assert not slot.done(), f'slot {tid} must still be wedged: the watcher released it, not its finally'
        else:
            if exit_ is _Exit.SHUTDOWN:
                slot.cancel()
            else:
                self.gate(tid).set()
            await wait_responsive(asyncio.wait({slot}), label=f'slot {tid} ends')
        assert not self.harness.is_workflow_active(tid)

    async def assert_never_launched(self, tid: str) -> None:
        await run_ready_callbacks()
        assert tid not in self.built
        assert tid not in self.slot_of
        assert not self.harness.is_workflow_active(tid)

    async def exactly_one_permit_free(self) -> bool:
        if self.slots.free_scope() is not DispatchScope.ANY:
            return False
        probe = await self.slots.claim(DispatchScope.ANY, {'id': 'probe'})
        try:
            return self.slots.free_scope() is None
        finally:
            probe.release()

    async def retire(self) -> None:
        """Let every gate and wedge through and wait for the slots, so no task that ignores cancels outlives the test."""
        for gate in self.gates.values():
            gate.set()
        for release in self.wedge_releases:
            release.set()
        if self.slot_of:
            await wait_responsive(asyncio.wait(set(self.slot_of.values())), label='every slot ends')


@pytest_asyncio.fixture
async def rig(tmp_path: Path, monkeypatch: pytest.MonkeyPatch) -> AsyncIterator[_Rig]:
    config = OrchestratorConfig(
        project_root=tmp_path,
        max_concurrent_tasks=1,
        terminal_status_hard_cancel_polls=HARD_CANCEL_POLLS,
        terminal_status_force_release_polls=FORCE_RELEASE_POLLS,
        terminal_cleanup_cancel_grace_secs=CLEANUP_GRACE,
    )
    harness = _build_harness(config)
    scheduler = Scheduler(config, mcp_session=HermeticMcpSession())
    harness.scheduler = scheduler
    rig = _Rig(harness, scheduler, RunSlots(general=1, main_health_fix_reserve=0))
    monkeypatch.setattr(scheduler, 'get_statuses', rig.get_statuses)
    monkeypatch.setattr('orchestrator.harness.build_workflow', rig.build_workflow)
    try:
        yield rig
    finally:
        await rig.retire()


@pytest.mark.asyncio
class TestSlotOnALease:

    async def test_a_slot_cancelled_before_its_first_step_releases_its_lease(self, rig: _Rig) -> None:
        assignment = rig.assignment('P')
        slot = rig.launch(assignment, await rig.holding_lease('P'))
        slot.cancel()
        await asyncio.gather(slot, return_exceptions=True)

        assert slot.cancelled()
        assert await rig.exactly_one_permit_free()
        assert not rig.scheduler.lock_table.is_held('P')

    async def test_a_borrower_cancelled_before_its_first_step_detaches_and_the_lender_returns_its_permit(
        self, rig: _Rig,
    ) -> None:
        p_lease = await rig.launch_lender()
        slot = rig.launch_borrowing(rig.assignment('D'), 'P')
        slot.cancel()
        await asyncio.gather(slot, return_exceptions=True)

        assert slot.cancelled()
        assert 'D' not in rig.built
        assert p_lease.borrower is None
        assert not rig.scheduler.lock_table.is_held('D')
        assert rig.slots.free_scope() is None

        await rig.end('P')
        assert await rig.exactly_one_permit_free()

    async def test_a_holding_lease_slot_that_ends_returns_its_permit_and_frees_its_locks(
        self, rig: _Rig,
    ) -> None:
        slot = rig.launch(rig.assignment('P'), await rig.holding_lease('P'))
        await rig.wait_active('P')
        assert rig.slots.free_scope() is None

        rig.gate('P').set()
        report = await wait_responsive(slot, label='slot P ends')

        assert report is not None and report.outcome is WorkflowOutcome.DONE
        assert await rig.exactly_one_permit_free()
        assert not rig.scheduler.lock_table.is_held('P')


_GENERAL_SLOTS = 2


@pytest.mark.asyncio
@pytest.mark.parametrize('first_to_end', ['lender', 'borrower'])
async def test_a_borrower_in_the_run_loop_takes_no_permit_and_the_waiting_task_starts_only_on_a_returned_one(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch, first_to_end: str,
) -> None:
    h, board = build_board_harness(
        tmp_path, monkeypatch,
        general_slots=_GENERAL_SLOTS, reserve=0, tasks=[plain('1'), plain('2'), plain('3')],
    )

    async def drive() -> None:
        await board.until(lambda: len(board.live) == _GENERAL_SLOTS, 'every general slot is taken')
        lender = min(board.live)
        [waiting] = [tid for tid in ('1', '2', '3') if not board.started(tid)]

        h._launch_slot(
            TaskAssignment(task_id='D', task=plain('D'), modules=[]), board.leases[lender].lend('D'), [],
        )
        await board.until(lambda: board.started('D'), 'the borrower starts')
        await board.settle()
        assert not board.started(waiting), 'the borrower took a permit'

        first, second = (lender, 'D') if first_to_end == 'lender' else ('D', lender)
        board.gates[first].set()
        await board.until(lambda: first not in board.live, f'{first} ends')
        await board.settle()
        assert not board.started(waiting), f'{first} ending returned a permit while {second} still runs'

        board.gates[second].set()
        await board.until(lambda: board.started(waiting), 'the waiting task starts on the returned permit')

    await run_driven(h, board, drive)
    assert board.max_live <= _GENERAL_SLOTS + 1



def _other(tid: str) -> str:
    return 'D' if tid == 'P' else 'P'


@pytest.mark.asyncio
class TestBorrowedSlotExits:

    @pytest.mark.parametrize('second_exit', _EXITS, ids=_exit_id)
    @pytest.mark.parametrize('first_exit', _EXITS, ids=_exit_id)
    @pytest.mark.parametrize('first', ['P', 'D'], ids=['lender-first', 'borrower-first'])
    async def test_every_exit_in_either_order_conserves_the_one_permit(
        self, rig: _Rig, first: str, first_exit: _Exit, second_exit: _Exit,
    ) -> None:
        second = _other(first)
        rig.exits.update({first: first_exit, second: second_exit})
        await rig.launch_pair()
        assert rig.slots.free_scope() is None

        await rig.end(first)
        assert rig.slots.free_scope() is None, f'{first} ending returned a permit while {second} still runs'
        assert not rig.scheduler.lock_table.is_held(first)
        assert rig.harness.is_workflow_active(second)

        await rig.end(second)
        assert await rig.exactly_one_permit_free()
        assert not rig.scheduler.lock_table.is_held('P')
        assert not rig.scheduler.lock_table.is_held('D')

        await rig.retire()
        assert await rig.exactly_one_permit_free(), 'a force-released slot released its lease again when it ended'

    @pytest.mark.parametrize('first', ['P', 'D'], ids=['lender-first', 'borrower-first'])
    async def test_shutting_down_both_slots_returns_exactly_one_permit(self, rig: _Rig, first: str) -> None:
        await rig.launch_pair()

        rig.slot_of[first].cancel()
        rig.slot_of[_other(first)].cancel()
        await asyncio.gather(rig.slot_of['P'], rig.slot_of['D'], return_exceptions=True)

        assert await rig.exactly_one_permit_free()
        assert not rig.scheduler.lock_table.is_held('P')
        assert not rig.scheduler.lock_table.is_held('D')


@pytest.mark.asyncio
class TestBorrowingRefusals:

    async def test_a_lender_lends_to_one_borrower_at_a_time(self, rig: _Rig) -> None:
        await rig.launch_pair()

        with pytest.raises(SlotLendingError) as refused:
            rig.launch_borrowing(_assignment('D2'), 'P')
        assert refused.value.rule is LendingRule.ALREADY_LENDS
        await rig.assert_never_launched('D2')
        assert rig.slots.free_scope() is None

        await rig.end('D')
        rig.launch_borrowing(rig.assignment('D2'), 'P')
        await rig.wait_active('D2')
        assert rig.slots.free_scope() is None

        await rig.end('D2')
        await rig.end('P')
        assert await rig.exactly_one_permit_free()

    async def test_a_borrower_lends_only_after_it_inherits(self, rig: _Rig) -> None:
        await rig.launch_pair()

        with pytest.raises(SlotLendingError) as refused:
            rig.launch_borrowing(_assignment('D3'), 'D')
        assert refused.value.rule is LendingRule.NO_OWN_PERMIT
        await rig.assert_never_launched('D3')
        assert rig.slots.free_scope() is None

        await rig.end('P')
        rig.launch_borrowing(rig.assignment('D3'), 'D')
        await rig.wait_active('D3')
        assert rig.slots.free_scope() is None

        await rig.end('D')
        assert rig.slots.free_scope() is None, 'D3 inherits the permit D inherited'
        await rig.end('D3')
        assert await rig.exactly_one_permit_free()

    async def test_borrowing_from_a_task_with_no_live_slot_is_refused(self, rig: _Rig) -> None:
        with pytest.raises(SlotLendingError) as refused:
            rig.launch_borrowing(_assignment('X'), 'nobody')

        assert refused.value.rule is LendingRule.NO_LIVE_SLOT
        assert (refused.value.lender_task_id, refused.value.borrower_task_id) == ('nobody', 'X')
        await rig.assert_never_launched('X')
        assert await rig.exactly_one_permit_free()

    async def test_a_task_with_a_live_slot_cannot_borrow_a_second_one(self, rig: _Rig) -> None:
        rig.launch(rig.assignment('Q'), await rig.outside_lease('Q'))
        await rig.wait_active('Q')
        p_lease = await rig.launch_lender()

        with pytest.raises(SlotLendingError) as refused:
            rig.launch_borrowing(_assignment('Q'), 'P')

        assert refused.value.rule is LendingRule.BORROWER_HAS_SLOT
        await run_ready_callbacks()
        assert rig.built.count('Q') == 1
        assert p_lease.borrower is None

        await rig.end('Q')
        assert not rig.scheduler.lock_table.is_held('Q')
        assert rig.slots.free_scope() is None
        await rig.end('P')
        assert await rig.exactly_one_permit_free()


@pytest.mark.asyncio
class TestSlotEndedSignal:

    @pytest.mark.parametrize('exit_', _EXITS, ids=_exit_id)
    async def test_a_waiter_on_the_lenders_lease_wakes_once_its_permit_has_passed(
        self, rig: _Rig, exit_: _Exit,
    ) -> None:
        rig.exits['P'] = exit_
        p_lease = await rig.launch_pair()
        seen: list[tuple[bool, DispatchScope | None, bool]] = []

        async def waiter() -> None:
            await p_lease.wait_released()
            seen.append((
                rig.scheduler.lock_table.is_held('P'),
                rig.slots.free_scope(),
                rig.harness.is_workflow_active('D'),
            ))

        waiting = asyncio.create_task(waiter())
        try:
            await run_ready_callbacks()
            assert not waiting.done(), 'the signal fired while P still runs'

            await rig.end('P')
            await wait_responsive(waiting, label='the slot-ended waiter wakes')
        finally:
            waiting.cancel()

        assert seen == [(False, None, True)]
