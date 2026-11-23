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
from collections.abc import AsyncIterator
from pathlib import Path
from unittest.mock import MagicMock

import pytest
import pytest_asyncio
from _orch_helpers import HermeticMcpSession
from _run_loop_board import (
    build_board_harness,
    fleet_dir_env,
    plain,
    run_driven,
    settle,
    until,
)
from _workflow_helpers import _build_harness

from orchestrator.config import OrchestratorConfig
from orchestrator.harness import Harness, TaskReport
from orchestrator.run_slots import DispatchScope, RunSlots
from orchestrator.scheduler import Scheduler, TaskAssignment
from orchestrator.slot_lease import SlotLease
from orchestrator.workflow import TerminalReport, WorkflowOutcome, WorkflowState

BARRIER = 5.0
_UNIT = 'orchestrator-borrowed-slot-test.service'


@pytest.fixture(autouse=True)
def _fleet_dir(tmp_path: Path, monkeypatch: pytest.MonkeyPatch) -> None:
    fleet_dir_env(monkeypatch, tmp_path, _UNIT)


class _GatedWorkflow:
    """Stands in for ``TaskWorkflow``: ``run()`` waits for its task's gate, then reports DONE."""

    def __init__(self, gate: asyncio.Event) -> None:
        self._gate = gate
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
        return TerminalReport(
            outcome=WorkflowOutcome.DONE, reason='', phase=WorkflowState.DONE, detail='', category=None,
        )


class _Rig:
    """A real harness slot path over a real scheduler and a one-permit ``RunSlots`` pool."""

    def __init__(self, harness: Harness, scheduler: Scheduler, slots: RunSlots) -> None:
        self.harness = harness
        self.scheduler = scheduler
        self.slots = slots
        self.reports: list[TaskReport] = []
        self.gates: dict[str, asyncio.Event] = {}
        self.launched: list[asyncio.Task] = []

    def gate(self, tid: str) -> asyncio.Event:
        return self.gates.setdefault(tid, asyncio.Event())

    def build_workflow(self, *, assignment: TaskAssignment, **_kwargs: object) -> _GatedWorkflow:
        return _GatedWorkflow(self.gate(assignment.task_id))

    def assignment(self, tid: str) -> TaskAssignment:
        """*tid*'s assignment, with its module locks taken as the scheduler's dispatch takes them."""
        module = f'{tid}.py'
        assert self.scheduler.lock_table.try_acquire(tid, [module])
        task = {'id': tid, 'title': f'Task {tid}', 'status': 'pending', 'dependencies': []}
        return TaskAssignment(task_id=tid, task=task, modules=[module])

    async def holding_lease(self, tid: str) -> SlotLease:
        sem = await self.slots.claim(DispatchScope.ANY, {'id': tid})
        return SlotLease(tid, self.scheduler, sem)

    def launch(self, assignment: TaskAssignment, lease: SlotLease) -> asyncio.Task:
        slot = self.harness._launch_slot(assignment, lease, self.reports)
        self.launched.append(slot)
        return slot

    async def wait_active(self, tid: str) -> None:
        await until(lambda: self.harness.is_workflow_active(tid), f'slot {tid} registered')

    async def exactly_one_permit_free(self) -> bool:
        if self.slots.free_scope() is not DispatchScope.ANY:
            return False
        probe = await self.slots.claim(DispatchScope.ANY, {'id': 'probe'})
        try:
            return self.slots.free_scope() is None
        finally:
            probe.release()

    async def retire(self) -> None:
        for gate in self.gates.values():
            gate.set()
        if self.launched:
            await asyncio.wait(self.launched, timeout=BARRIER)


@pytest_asyncio.fixture
async def rig(tmp_path: Path, monkeypatch: pytest.MonkeyPatch) -> AsyncIterator[_Rig]:
    config = OrchestratorConfig(project_root=tmp_path, max_concurrent_tasks=1)
    harness = _build_harness(config)
    scheduler = Scheduler(config, mcp_session=HermeticMcpSession())
    harness.scheduler = scheduler
    rig = _Rig(harness, scheduler, RunSlots(general=1, main_health_fix_reserve=0))
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

    async def test_a_holding_lease_slot_that_ends_returns_its_permit_and_frees_its_locks(
        self, rig: _Rig,
    ) -> None:
        slot = rig.launch(rig.assignment('P'), await rig.holding_lease('P'))
        await rig.wait_active('P')
        assert rig.slots.free_scope() is None

        rig.gate('P').set()
        report = await asyncio.wait_for(slot, BARRIER)

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
        await until(lambda: len(board.live) == _GENERAL_SLOTS, 'every general slot is taken')
        lender = min(board.live)
        [waiting] = [tid for tid in ('1', '2', '3') if not board.started(tid)]

        h._launch_slot(
            TaskAssignment(task_id='D', task=plain('D'), modules=[]), board.leases[lender].lend('D'), [],
        )
        await until(lambda: board.started('D'), 'the borrower starts')
        await settle()
        assert not board.started(waiting), 'the borrower took a permit'

        first, second = (lender, 'D') if first_to_end == 'lender' else ('D', lender)
        board.gates[first].set()
        await until(lambda: first not in board.live, f'{first} ends')
        await settle()
        assert not board.started(waiting), f'{first} ending returned a permit while {second} still runs'

        board.gates[second].set()
        await until(lambda: board.started(waiting), 'the waiting task starts on the returned permit')

    await run_driven(h, board, drive)
    assert board.max_live <= _GENERAL_SLOTS + 1

