"""Harness.run() with its dispatch loop PARKED (task 5344, INV-10).

Under saturation the loop sits in ``await sem.acquire()`` (a slot is busy and
``acquire_next`` keeps handing out work) rather than in a rest branch.  These
tests drive the real ``run()`` in that shape and assert what must still happen
there: a finishing slot's report is collected, an owed service restart
force-fires, and the fleet merge heartbeat keeps advancing.

``max_concurrent_tasks=1`` is what parks the loop: once slot ``'1'`` holds the
semaphore, the next assignment blocks in ``sem.acquire()`` until that slot
ends.  Startup collaborators are stubbed the way
``test_harness_scheduler_seam.py::TestFinishStartupCalledBeforeFirstAcquireNext``
does; the slot body is stubbed at its one seam, ``_run_slot``.
"""

from __future__ import annotations

import asyncio
import json
from collections import defaultdict
from collections.abc import Awaitable, Callable
from pathlib import Path
from unittest.mock import AsyncMock, MagicMock, patch

import pytest
from _orch_helpers import wait_responsive

from orchestrator.config import OrchestratorConfig
from orchestrator.harness import Harness, HarnessReport, TaskReport
from orchestrator.scheduler import TaskAssignment
from orchestrator.service_restart import StaleServiceRestartCoordinator
from orchestrator.workflow import WorkflowOutcome

_TICK = 0.01
_FORCE_FIRE_AFTER_SECS = 900.0
_UNIT = 'orchestrator-parked-test.service'


class _Slots:
    """Stand-in slot runner: each slot ends when the test opens its gate.

    Mirrors ``Harness._run_slot``'s contract: it releases the semaphore on
    exit and turns a cancel into a synthetic CANCELLED report.
    """

    def __init__(self) -> None:
        self.gates: defaultdict[str, asyncio.Event] = defaultdict(asyncio.Event)
        self._all_open = False

    def open_all(self) -> None:
        self._all_open = True
        for gate in self.gates.values():
            gate.set()

    async def run(self, assignment: TaskAssignment, sem: asyncio.Semaphore) -> TaskReport:
        tid = assignment.task_id
        if self._all_open:
            self.gates[tid].set()
        try:
            await self.gates[tid].wait()
            return TaskReport(task_id=tid, title=tid, outcome=WorkflowOutcome.DONE)
        except asyncio.CancelledError:
            return TaskReport(task_id=tid, title=tid, outcome=WorkflowOutcome.CANCELLED)
        finally:
            sem.release()


class _Dispatch:
    """``acquire_next`` stand-in handing out ``task_ids`` in order, then None."""

    def __init__(self, task_ids: list[str]) -> None:
        self._pending = list(task_ids)
        self.calls = 0
        self.stop_when: Callable[[], bool] = lambda: False

    async def acquire_next(self) -> TaskAssignment | None:
        self.calls += 1
        if self.stop_when():
            raise _Stop
        if not self._pending:
            return None
        tid = self._pending.pop(0)
        return TaskAssignment(task_id=tid, task={'title': tid}, modules=[])


class _Stop(Exception):
    pass


class _Clock:
    """Monotonic clock for a coordinator; counts its reads (one per pass)."""

    def __init__(self) -> None:
        self.now = 1000.0
        self.reads = 0

    def __call__(self) -> float:
        self.reads += 1
        return self.now


async def _until(predicate: Callable[[], bool], label: str) -> None:
    async def poll() -> None:
        while not predicate():
            await asyncio.sleep(_TICK)

    await wait_responsive(poll(), label=label)


def _harness(tmp_path: Path, dispatch: _Dispatch, slots: _Slots) -> Harness:
    config = OrchestratorConfig(
        project_root=tmp_path,
        max_concurrent_tasks=1,
        idle_poll_secs=_TICK,
        merge_heartbeat_interval_secs=_TICK,
        stale_service_restart_poll_secs=_TICK,
        watcher_supervisor_enabled=False,
        review={'full_review_on_complete': False},
    )
    with patch('orchestrator.harness.McpLifecycle') as mcp_cls, \
         patch('orchestrator.harness.Scheduler'), \
         patch('orchestrator.harness.BriefingAssembler'):
        h = Harness(config)
    mcp_cls.return_value.start = AsyncMock()
    mcp_cls.return_value.stop = AsyncMock()

    h.git_ops = MagicMock()
    h.git_ops.disable_shared_repo_auto_maintenance = AsyncMock()
    h.git_ops.has_dirty_working_tree = AsyncMock(return_value=None)
    h.git_ops.worktree_base = tmp_path / '.worktrees'
    h.usage_gate = None
    h.review_checkpoint = MagicMock()
    h.review_checkpoint.should_trigger.return_value = False

    h.scheduler = MagicMock()
    h.scheduler.is_paused = False
    h.scheduler.get_statuses = AsyncMock(return_value=({}, None))
    h.scheduler.get_tasks = AsyncMock(return_value=[])
    h.scheduler.acquire_next = dispatch.acquire_next

    h._start_escalation_server = AsyncMock()  # type: ignore[method-assign]
    h._start_merge_worker = AsyncMock()  # type: ignore[method-assign]
    h._dismiss_stale_escalations = AsyncMock()  # type: ignore[method-assign]
    h._recover_crashed_tasks = AsyncMock()  # type: ignore[method-assign]
    h._reconcile_lane_checkouts = AsyncMock()  # type: ignore[method-assign]
    h._reconcile_stranded_in_progress = AsyncMock(return_value=0)  # type: ignore[method-assign]
    h._run_slot = slots.run  # type: ignore[method-assign]
    return h


async def _owed_fused_memory_restart(
    h: Harness, clock: _Clock, fired: asyncio.Event,
) -> StaleServiceRestartCoordinator:
    async def executor() -> None:
        fired.set()

    coord = StaleServiceRestartCoordinator(
        git_ops=MagicMock(),
        event_store=None,
        debounce_secs=0.0,
        force_fire_after_secs=_FORCE_FIRE_AFTER_SECS,
        restart_executor=executor,
        clock=clock,
    )
    armed = await coord.note_merge(
        '7', 'base', 'head', prefetched_diff=['fused-memory/src/fused_memory/x.py'],
    )
    assert armed
    h._service_restart_coordinators = [coord]
    return coord


def _heartbeat_ts(fleet_dir: Path) -> float | None:
    path = fleet_dir / f'{_UNIT}.json'
    if not path.exists():
        return None
    return json.loads(path.read_text())['ts_epoch']


@pytest.fixture
def fleet_dir(tmp_path: Path, monkeypatch: pytest.MonkeyPatch) -> Path:
    fleet = tmp_path / 'fleet'
    monkeypatch.setenv('ORCH_UNIT', _UNIT)
    monkeypatch.setenv('ORCH_FLEET_DIR', str(fleet))
    return fleet




async def _run_driven(
    h: Harness,
    slots: _Slots,
    drive: Callable[[], Awaitable[None]],
    *,
    until_idle: bool = True,
) -> HarnessReport:
    """Run ``h.run()`` in this task while ``drive`` steers it from another.

    ``run()`` cancels every other live task at shutdown, so it must own the
    test's task.  A failed drive step opens every slot gate so ``run()``
    still exits, then re-raises here.
    """

    async def driver() -> None:
        try:
            await drive()
        finally:
            slots.open_all()

    steering = asyncio.create_task(driver())
    report = await h.run(until_idle=until_idle)
    await steering
    return report


@pytest.mark.asyncio
class TestParkedLoop:

    async def test_slot_finishing_while_parked_is_collected(
        self, tmp_path: Path, fleet_dir: Path,
    ) -> None:
        dispatch, slots = _Dispatch(['1', '2', '3']), _Slots()
        h = _harness(tmp_path, dispatch, slots)

        async def drive() -> None:
            for tid, calls in (('1', 2), ('2', 3), ('3', 4)):
                await _until(lambda n=calls: dispatch.calls >= n, f'dispatch call {calls}')
                slots.gates[tid].set()

        report = await _run_driven(h, slots, drive)

        assert sorted(r.task_id for r in report.task_reports) == ['1', '2', '3']
        assert report.completed == 3
        assert h.review_checkpoint is not None
        assert h.review_checkpoint.record_merge.call_count == 3

    async def test_owed_restart_force_fires_while_parked(
        self, tmp_path: Path, fleet_dir: Path,
    ) -> None:
        dispatch, slots = _Dispatch(['1', '2']), _Slots()
        h = _harness(tmp_path, dispatch, slots)
        clock, fired = _Clock(), asyncio.Event()
        await _owed_fused_memory_restart(h, clock, fired)

        async def drive() -> None:
            await _until(lambda: dispatch.calls >= 2, 'loop parked on slot 1')
            reads_when_parked = clock.reads
            await _until(lambda: clock.reads >= reads_when_parked + 3, 'restart polled while parked')
            assert not fired.is_set(), 'a polite restart must wait while a slot is live'
            clock.now += _FORCE_FIRE_AFTER_SECS
            await _until(fired.is_set, 'owed restart force-fired while parked')
            assert dispatch.calls == 2, 'the loop must still be parked when it fires'

        await _run_driven(h, slots, drive)

    async def test_heartbeat_advances_while_parked(
        self, tmp_path: Path, fleet_dir: Path,
    ) -> None:
        dispatch, slots = _Dispatch(['1', '2']), _Slots()
        h = _harness(tmp_path, dispatch, slots)

        async def drive() -> None:
            await _until(lambda: dispatch.calls >= 2, 'loop parked on slot 1')
            await _until(lambda: _heartbeat_ts(fleet_dir) is not None, 'first heartbeat')
            first = _heartbeat_ts(fleet_dir) or 0.0
            await _until(lambda: (_heartbeat_ts(fleet_dir) or 0.0) > first, 'heartbeat advanced')
            assert dispatch.calls == 2, 'the loop must still be parked'

        await _run_driven(h, slots, drive)

    async def test_slot_cancelled_at_shutdown_is_collected(
        self, tmp_path: Path, fleet_dir: Path,
    ) -> None:
        dispatch, slots = _Dispatch(['1', '2']), _Slots()
        h = _harness(tmp_path, dispatch, slots)
        main = asyncio.current_task()
        assert main is not None

        async def sigterm_while_parked() -> None:
            await _until(lambda: dispatch.calls >= 2, 'loop parked on slot 1')
            main.cancel()

        with pytest.raises(asyncio.CancelledError):
            await _run_driven(h, slots, sigterm_while_parked)

        outcomes = {r.task_id: r.outcome for r in h.report.task_reports}
        assert outcomes == {'1': WorkflowOutcome.CANCELLED}

    async def test_polite_restart_waits_for_the_dispatch_loop(
        self, tmp_path: Path, fleet_dir: Path,
    ) -> None:
        dispatch, slots = _Dispatch([]), _Slots()
        h = _harness(tmp_path, dispatch, slots)
        clock, fired = _Clock(), asyncio.Event()
        await _owed_fused_memory_restart(h, clock, fired)

        async def startup_recovery() -> None:
            await _until(lambda: clock.reads >= 4, 'restart polled during startup')
            assert not fired.is_set(), 'no polite restart during startup recovery'

        h._recover_crashed_tasks = startup_recovery  # type: ignore[method-assign]
        dispatch.stop_when = fired.is_set

        with pytest.raises(_Stop):
            await h.run()
        assert fired.is_set()
