"""A rig that drives the real ``Harness.run`` loop over a real ``Scheduler`` with gated fake slots.

The scheduler is a REAL ``Scheduler`` over an in-test task list.  The slot
body is stubbed at its one seam, ``_run_slot``: each slot holds its run slot
until the test opens its gate.  Startup collaborators are stubbed the way
``test_harness_parked_loop.py`` stubs them.

``Board`` records every dispatch tick and every wait for a slot to turn over;
``build_board_harness`` wires it into a harness; ``run_driven`` runs ``h.run()``
while a test coroutine steers it.  Nothing here sleeps in real time:
``Board.until`` wakes on each change the board records, ``Board.settle`` counts
the loop's own polls, and ``until`` polls by yielding, for a predicate over
state the board does not record.
"""

from __future__ import annotations

import asyncio
from collections import defaultdict
from collections.abc import Awaitable, Callable
from dataclasses import dataclass
from pathlib import Path
from unittest.mock import AsyncMock, MagicMock, patch

import pytest
from _orch_helpers import HermeticMcpSession, wait_responsive

from orchestrator.config import OrchestratorConfig, PsiAdmissionConfig, ReviewConfig
from orchestrator.harness import Harness, TaskReport
from orchestrator.main_health_fix_owner import SIGNATURE_KEY
from orchestrator.run_slots import DispatchScope
from orchestrator.scheduler import Scheduler, TaskAssignment
from orchestrator.slot_lease import SlotLease
from orchestrator.workflow import WorkflowOutcome

TICK = 0.01
SETTLE_POLLS = 10
READY_CALLBACK_ROUNDS = 10
_GATE_OFF = PsiAdmissionConfig(enabled=False)


def task(tid: str, priority: str, *, signature: str | None = None) -> dict:
    metadata: dict = {'files': [f'mod_{tid}.py']}
    if signature is not None:
        metadata.update({SIGNATURE_KEY: signature, 'merge_lane': 'high'})
    return {
        'id': tid,
        'title': f'Task {tid}',
        'status': 'pending',
        'priority': priority,
        'dependencies': [],
        'metadata': metadata,
    }


def plain(tid: str, priority: str = 'medium') -> dict:
    return task(tid, priority)


@dataclass(frozen=True)
class Tick:
    """One ``acquire_next`` call: the board epoch it started in, its scope and its pick."""

    epoch: int
    scope: DispatchScope
    picked: str | None


class Board:
    """The task list, the gated slots, and a record of every dispatch tick."""

    def __init__(self, scheduler: Scheduler, tasks: list[dict], monkeypatch: pytest.MonkeyPatch) -> None:
        self.scheduler = scheduler
        self.tasks = list(tasks)
        self.epoch = 0
        self.ticks: list[Tick] = []
        self.gates: defaultdict[str, asyncio.Event] = defaultdict(asyncio.Event)
        self.live: set[str] = set()
        self.live_at_start: dict[str, int] = {}
        self.max_live = 0
        self.leases: dict[str, SlotLease] = {}
        self.turnover_waits = 0
        self._change = asyncio.Event()
        self._all_open = False
        self._inside_next_restricted_tick: Callable[[], Awaitable[None]] | None = None
        real_acquire = scheduler.acquire_next

        async def recorded_acquire(scope: DispatchScope = DispatchScope.ANY) -> TaskAssignment | None:
            epoch = self.epoch
            if scope is DispatchScope.MAIN_HEALTH_FIX and self._inside_next_restricted_tick is not None:
                action, self._inside_next_restricted_tick = self._inside_next_restricted_tick, None
                await action()
            assignment = await real_acquire(scope)
            self.ticks.append(Tick(epoch, scope, assignment.task_id if assignment else None))
            self._changed()
            return assignment

        scheduler.get_tasks = AsyncMock(side_effect=lambda **_: list(self.tasks))
        scheduler.get_statuses = AsyncMock(return_value=({}, None))
        monkeypatch.setattr(scheduler, 'acquire_next', recorded_acquire)

    def mark(self) -> int:
        """Start a new epoch; every tick that begins from now on is recorded in it."""
        self.epoch += 1
        return self.epoch

    def file(self, *tasks: dict) -> int:
        """Put *tasks* on the board; returns the epoch whose ticks are the first to see them."""
        self.tasks.extend(tasks)
        return self.mark()

    def inside_next_restricted_tick(self, action: Callable[[], Awaitable[None]]) -> None:
        """Run *action* once, inside the next MAIN_HEALTH_FIX tick, before that tick reads the board."""
        self._inside_next_restricted_tick = action

    def ticks_since(self, epoch: int) -> list[Tick]:
        return [tick for tick in self.ticks if tick.epoch >= epoch]

    def polls(self) -> int:
        """The run loop's polls so far: every dispatch tick and every wait for a slot to turn over."""
        return len(self.ticks) + self.turnover_waits

    def count_turnover_wait(self) -> None:
        self.turnover_waits += 1
        self._changed()

    def _changed(self) -> None:
        change, self._change = self._change, asyncio.Event()
        change.set()

    async def until(self, predicate: Callable[[], bool], label: str) -> None:
        """Wait until *predicate* over the board holds, re-checking on each change the board records.

        Fails by *label* on the responsive-time deadline.
        """
        async def wait() -> None:
            while not predicate():
                await self._change.wait()

        await wait_responsive(wait(), label=label)

    async def settle(self) -> None:
        """Let the run loop make ``SETTLE_POLLS`` more polls."""
        target = self.polls() + SETTLE_POLLS
        await self.until(lambda: self.polls() >= target, f'the run loop makes {SETTLE_POLLS} more polls')

    def holds_locks(self, tid: str) -> bool:
        return self.scheduler.lock_table.is_held(tid)

    def started(self, tid: str) -> bool:
        return tid in self.live_at_start

    def open_all(self) -> None:
        self._all_open = True
        for gate in self.gates.values():
            gate.set()

    async def run_slot(self, assignment: TaskAssignment, lease: SlotLease) -> TaskReport:
        """Holds its run slot until the test opens its gate; records the lease it runs on."""
        tid = assignment.task_id
        self.leases[tid] = lease
        self.live.add(tid)
        self.live_at_start[tid] = len(self.live)
        self.max_live = max(self.max_live, len(self.live))
        self._changed()
        if self._all_open:
            self.gates[tid].set()
        try:
            await self.gates[tid].wait()
            return TaskReport(task_id=tid, title=tid, outcome=WorkflowOutcome.DONE)
        except asyncio.CancelledError:
            return TaskReport(task_id=tid, title=tid, outcome=WorkflowOutcome.CANCELLED)
        finally:
            self.live.discard(tid)
            self.tasks = [t for t in self.tasks if t['id'] != tid]
            lease.release(requeued=False)
            self._changed()


async def until(predicate: Callable[[], bool], label: str) -> None:
    """Yield to the loop until *predicate* holds, failing by *label* on the responsive-time deadline.

    For state the board does not record; a predicate over the board waits in ``Board.until``.
    """
    async def poll() -> None:
        while not predicate():
            await asyncio.sleep(0)

    await wait_responsive(poll(), label=label)


async def run_ready_callbacks() -> None:
    """Yield ``READY_CALLBACK_ROUNDS`` times, so every callback already scheduled, and those it schedules, runs."""
    for _ in range(READY_CALLBACK_ROUNDS):
        await asyncio.sleep(0)


def build_board_harness(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
    *,
    general_slots: int,
    reserve: int,
    tasks: list[dict],
    idle_poll_secs: float = TICK,
    psi_admission: PsiAdmissionConfig = _GATE_OFF,
) -> tuple[Harness, Board]:
    config = OrchestratorConfig(
        project_root=tmp_path,
        max_concurrent_tasks=general_slots,
        main_health_fix_reserved_slots=reserve,
        idle_poll_secs=idle_poll_secs,
        watcher_supervisor_enabled=False,
        review=ReviewConfig(full_review_on_complete=False),
        psi_admission=psi_admission,
    )
    scheduler = Scheduler(config, mcp_session=HermeticMcpSession())
    board = Board(scheduler, tasks, monkeypatch)
    with patch('orchestrator.harness.McpLifecycle') as mcp_cls, \
         patch('orchestrator.harness.Scheduler', return_value=scheduler), \
         patch('orchestrator.harness.BriefingAssembler'):
        h = Harness(config)
    mcp_cls.return_value.start = AsyncMock()
    mcp_cls.return_value.stop = AsyncMock()

    h.git_ops = MagicMock()
    h.git_ops.disable_shared_repo_auto_maintenance = AsyncMock()
    h.git_ops.has_dirty_working_tree = AsyncMock(return_value=None)
    h.git_ops.reap_interactive_worktrees = AsyncMock(return_value=[])
    h.git_ops.resolve_branch_sha = AsyncMock(return_value=None)
    h.git_ops.find_merge_marker = AsyncMock(return_value=None)
    h.git_ops.worktree_base = tmp_path / '.worktrees'
    h.usage_gate = None
    h.review_checkpoint = None
    wait_for_slot_turnover = h._wait_for_slot_turnover

    async def counted_wait_for_slot_turnover() -> None:
        await wait_for_slot_turnover()
        board.count_turnover_wait()

    stubs: tuple[tuple[str, object], ...] = (
        ('_start_escalation_server', AsyncMock()),
        ('_start_merge_worker', AsyncMock()),
        ('_dismiss_stale_escalations', AsyncMock()),
        ('_recover_crashed_tasks', AsyncMock()),
        ('_reconcile_lane_checkouts', AsyncMock()),
        ('_reconcile_stranded_in_progress', AsyncMock(return_value=0)),
        ('_run_slot', board.run_slot),
        ('_wait_for_slot_turnover', counted_wait_for_slot_turnover),
    )
    for name, stub in stubs:
        monkeypatch.setattr(h, name, stub)
    return h, board


async def run_driven(h: Harness, board: Board, drive: Callable[[], Awaitable[None]]) -> None:
    """Run ``h.run()`` in this task while ``drive`` steers it; a finished or failed drive drains every slot."""

    async def driver() -> None:
        try:
            await drive()
        finally:
            board.open_all()

    steering = asyncio.create_task(driver())
    await h.run(until_idle=True)
    await steering


def fleet_dir_env(monkeypatch: pytest.MonkeyPatch, tmp_path: Path, unit: str) -> None:
    monkeypatch.setenv('ORCH_UNIT', unit)
    monkeypatch.setenv('ORCH_FLEET_DIR', str(tmp_path / 'fleet'))
