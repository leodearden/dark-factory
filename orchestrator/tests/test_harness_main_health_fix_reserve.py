"""The 2026-11-04 / 2026-11-15 deadlock replayed through the real ``Harness.run`` loop (task 8480).

Every general run slot holds a merge-waiter that cannot land until main is
healed.  A main-health fix filed in that state must dispatch on the reserved
slot at the next tick, while the loop picks nothing, and so locks nothing,
whenever every slot is held.

The scheduler is a REAL ``Scheduler`` over an in-test task list.  The slot
body is stubbed at its one seam, ``_run_slot``: each slot is a merge-waiter
that ends only when the test opens its gate.  Startup collaborators are
stubbed the way ``test_harness_parked_loop.py`` stubs them.
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
from orchestrator.workflow import WorkflowOutcome

_TICK = 0.01
_SETTLE_TICKS = 10
_GENERAL_SLOTS = 18
_UNIT = 'orchestrator-reserve-test.service'


def _task(tid: str, priority: str, *, signature: str | None = None) -> dict:
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


def _plain(tid: str, priority: str = 'medium') -> dict:
    return _task(tid, priority)


def _fix(tid: str) -> dict:
    return _task(tid, 'high', signature=f'sig-{tid}')


@dataclass(frozen=True)
class _Tick:
    """One ``acquire_next`` call: the board epoch it started in, its scope and its pick."""

    epoch: int
    scope: DispatchScope
    picked: str | None


class _Board:
    """The task list, the merge-waiter slots, and a record of every dispatch tick."""

    def __init__(self, scheduler: Scheduler, tasks: list[dict], monkeypatch: pytest.MonkeyPatch) -> None:
        self.scheduler = scheduler
        self.tasks = list(tasks)
        self.epoch = 0
        self.ticks: list[_Tick] = []
        self.gates: defaultdict[str, asyncio.Event] = defaultdict(asyncio.Event)
        self.live: set[str] = set()
        self.live_at_start: dict[str, int] = {}
        self.max_live = 0
        self._all_open = False
        real_acquire = scheduler.acquire_next

        async def recorded_acquire(scope: DispatchScope = DispatchScope.ANY) -> TaskAssignment | None:
            epoch = self.epoch
            assignment = await real_acquire(scope)
            self.ticks.append(_Tick(epoch, scope, assignment.task_id if assignment else None))
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

    def ticks_since(self, epoch: int) -> list[_Tick]:
        return [tick for tick in self.ticks if tick.epoch >= epoch]

    def holds_locks(self, tid: str) -> bool:
        return self.scheduler.lock_table.is_held(tid)

    def started(self, tid: str) -> bool:
        return tid in self.live_at_start

    def open_all(self) -> None:
        self._all_open = True
        for gate in self.gates.values():
            gate.set()

    async def run_slot(self, assignment: TaskAssignment, sem: asyncio.Semaphore) -> TaskReport:
        """A merge-waiter: holds its run slot until the test opens its gate."""
        tid = assignment.task_id
        self.live.add(tid)
        self.live_at_start[tid] = len(self.live)
        self.max_live = max(self.max_live, len(self.live))
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
            self.scheduler.release(tid)
            sem.release()


async def _until(predicate: Callable[[], bool], label: str) -> None:
    async def poll() -> None:
        while not predicate():
            await asyncio.sleep(_TICK)

    await wait_responsive(poll(), label=label)


async def _settle() -> None:
    """Let the loop run several idle polls."""
    await asyncio.sleep(_TICK * _SETTLE_TICKS)


def _harness(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch, *, reserve: int, tasks: list[dict],
) -> tuple[Harness, _Board]:
    config = OrchestratorConfig(
        project_root=tmp_path,
        max_concurrent_tasks=_GENERAL_SLOTS,
        main_health_fix_reserved_slots=reserve,
        idle_poll_secs=_TICK,
        watcher_supervisor_enabled=False,
        review=ReviewConfig(full_review_on_complete=False),
        psi_admission=PsiAdmissionConfig(enabled=False),
    )
    scheduler = Scheduler(config, mcp_session=HermeticMcpSession())
    board = _Board(scheduler, tasks, monkeypatch)
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

    stubs: tuple[tuple[str, object], ...] = (
        ('_start_escalation_server', AsyncMock()),
        ('_start_merge_worker', AsyncMock()),
        ('_dismiss_stale_escalations', AsyncMock()),
        ('_recover_crashed_tasks', AsyncMock()),
        ('_reconcile_lane_checkouts', AsyncMock()),
        ('_reconcile_stranded_in_progress', AsyncMock(return_value=0)),
        ('_run_slot', board.run_slot),
    )
    for name, stub in stubs:
        monkeypatch.setattr(h, name, stub)
    return h, board


def _merge_waiters() -> list[dict]:
    return [_plain(str(n)) for n in range(1, _GENERAL_SLOTS + 1)]


async def _run_driven(h: Harness, board: _Board, drive: Callable[[], Awaitable[None]]) -> None:
    """Run ``h.run()`` in this task while ``drive`` steers it; a finished or failed drive drains every slot."""

    async def driver() -> None:
        try:
            await drive()
        finally:
            board.open_all()

    steering = asyncio.create_task(driver())
    await h.run(until_idle=True)
    await steering


@pytest.fixture(autouse=True)
def _fleet_dir(tmp_path: Path, monkeypatch: pytest.MonkeyPatch) -> None:
    monkeypatch.setenv('ORCH_UNIT', _UNIT)
    monkeypatch.setenv('ORCH_FLEET_DIR', str(tmp_path / 'fleet'))


async def _saturate(board: _Board) -> None:
    await _until(lambda: len(board.live) == _GENERAL_SLOTS, 'every general slot holds a merge-waiter')


@pytest.mark.asyncio
class TestMainHealthFixReserve:

    async def test_fix_filed_while_every_general_slot_holds_a_merge_waiter_dispatches_next_tick(
        self, tmp_path: Path, monkeypatch: pytest.MonkeyPatch,
    ) -> None:
        h, board = _harness(tmp_path, monkeypatch, reserve=1, tasks=_merge_waiters())

        async def drive() -> None:
            await _saturate(board)
            filed = board.file(_plain('50', 'critical'), _fix('60'))

            await _until(lambda: board.started('60'), 'the fix starts on the reserve')
            assert board.live_at_start['60'] == _GENERAL_SLOTS + 1
            first = board.ticks_since(filed)[0]
            assert (first.scope, first.picked) == (DispatchScope.MAIN_HEALTH_FIX, '60')

            await _settle()
            assert not board.started('50')
            assert not board.holds_locks('50')

        await _run_driven(h, board, drive)

    async def test_loop_picks_nothing_while_every_slot_is_held(
        self, tmp_path: Path, monkeypatch: pytest.MonkeyPatch,
    ) -> None:
        h, board = _harness(tmp_path, monkeypatch, reserve=1, tasks=_merge_waiters())

        async def drive() -> None:
            await _saturate(board)
            board.file(_fix('60'))
            await _until(lambda: board.started('60'), 'the fix takes the reserve')

            ticks_when_full = len(board.ticks)
            board.file(_plain('70'))
            await _settle()
            assert len(board.ticks) == ticks_when_full, 'no tick while every slot is held'
            assert not board.holds_locks('70')

            freed = board.mark()
            board.gates['1'].set()
            await _until(lambda: board.started('70'), 'the next task starts on the freed slot')
            first = board.ticks_since(freed)[0]
            assert (first.scope, first.picked) == (DispatchScope.ANY, '70')

        await _run_driven(h, board, drive)

    async def test_reserve_disabled_keeps_the_old_cap(
        self, tmp_path: Path, monkeypatch: pytest.MonkeyPatch,
    ) -> None:
        h, board = _harness(tmp_path, monkeypatch, reserve=0, tasks=_merge_waiters())

        async def drive() -> None:
            await _saturate(board)
            board.file(_fix('60'))
            await _settle()

            assert not board.started('60')
            assert not board.holds_locks('60')
            assert len(board.ticks) == _GENERAL_SLOTS, 'no tick while every slot is held'

        await _run_driven(h, board, drive)

    async def test_a_second_fix_does_not_take_a_second_extra_slot(
        self, tmp_path: Path, monkeypatch: pytest.MonkeyPatch,
    ) -> None:
        h, board = _harness(tmp_path, monkeypatch, reserve=1, tasks=_merge_waiters())

        async def drive() -> None:
            await _saturate(board)
            board.file(_fix('60'), _fix('61'))
            await _until(lambda: board.started('60') or board.started('61'), 'one fix starts')
            await _settle()

            [first_fix] = [tid for tid in ('60', '61') if board.started(tid)]
            second_fix = '61' if first_fix == '60' else '60'
            assert board.max_live == _GENERAL_SLOTS + 1
            assert not board.holds_locks(second_fix)

            freed = board.mark()
            board.gates[first_fix].set()
            await _until(lambda: board.started(second_fix), 'the second fix takes the freed reserve')
            assert board.live_at_start[second_fix] == _GENERAL_SLOTS + 1
            first = board.ticks_since(freed)[0]
            assert (first.scope, first.picked) == (DispatchScope.MAIN_HEALTH_FIX, second_fix)

        await _run_driven(h, board, drive)
