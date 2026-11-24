"""Harness.run() with its dispatch loop PARKED (task 5344, INV-10).

Under saturation the loop waits for a live slot to end without picking a task,
so no picked task holds its module locks while it waits (task 8480).  These
tests drive the real ``run()`` in that shape and assert what must still happen
there: a finishing slot's report is collected, an owed service restart
force-fires, and the fleet merge heartbeat keeps advancing.

``max_concurrent_tasks=1`` with no main-health-fix reserve is what parks the
loop: once slot ``'1'`` holds the only run slot, the loop makes no further
``acquire_next`` call until that slot ends.  Startup collaborators are stubbed the way
``test_harness_scheduler_seam.py::TestFinishStartupCalledBeforeFirstAcquireNext``
does; the slot body is stubbed at its one seam, ``_run_slot``.
"""

from __future__ import annotations

import asyncio
import json
import logging
from collections import defaultdict
from collections.abc import Awaitable, Callable
from pathlib import Path
from unittest.mock import AsyncMock, MagicMock, patch

import pytest
from _orch_helpers import wait_responsive

from orchestrator.config import OrchestratorConfig, ReviewConfig
from orchestrator.harness import Harness, HarnessReport, TaskReport
from orchestrator.run_slots import DispatchScope
from orchestrator.scheduler import TaskAssignment
from orchestrator.service_restart import StaleServiceRestartCoordinator
from orchestrator.slot_lease import SlotLease
from orchestrator.workflow import WorkflowOutcome

_TICK = 0.01
_FORCE_FIRE_AFTER_SECS = 900.0
_UNIT = 'orchestrator-parked-test.service'
_SHUTDOWN_DRAIN_SECS = 0.2
_HANG_GUARD_SECS = 60.0


class _Slots:
    """Stand-in slot runner: each slot ends when the test opens its gate.

    Mirrors ``Harness._run_slot``'s contract: it releases its lease on exit
    and turns a cancel into a synthetic CANCELLED report.  A slot whose id is
    in ``immune`` ignores every cancel and every gate, ``open_all``'s
    included, until ``immune_release`` is set.  A slot whose id is in
    ``absorbs_one_cancel`` ignores its gate and its first cancel, noting
    ``'slot <id> absorbed a cancel'`` in ``events``, and ends on the next.
    """

    def __init__(self) -> None:
        self.gates: defaultdict[str, asyncio.Event] = defaultdict(asyncio.Event)
        self.started: set[str] = set()
        self.finished: set[str] = set()
        self.immune: set[str] = set()
        self.immune_release = asyncio.Event()
        self.absorbs_one_cancel: set[str] = set()
        self.events: list[str] = []
        self._all_open = False

    def open_all(self) -> None:
        self._all_open = True
        for gate in self.gates.values():
            gate.set()

    async def run(self, assignment: TaskAssignment, lease: SlotLease) -> TaskReport:
        tid = assignment.task_id
        self.started.add(tid)
        if self._all_open:
            self.gates[tid].set()
        try:
            if tid in self.immune:
                await _ignore_cancels_until(self.immune_release)
                return TaskReport(task_id=tid, title=tid, outcome=WorkflowOutcome.CANCELLED)
            if tid in self.absorbs_one_cancel:
                await _absorb_one_cancel(lambda: self.events.append(f'slot {tid} absorbed a cancel'))
            await self.gates[tid].wait()
            return TaskReport(task_id=tid, title=tid, outcome=WorkflowOutcome.DONE)
        except asyncio.CancelledError:
            self.events.append(f'slot {tid} unwound')
            return TaskReport(task_id=tid, title=tid, outcome=WorkflowOutcome.CANCELLED)
        finally:
            self.finished.add(tid)
            lease.release(requeued=False)


class _HangGuard:
    """Frees a cancel-immune task if shutdown would otherwise wait on it forever.

    Armed at the cancel, so its budget bounds the shutdown alone.  A timer
    handle, not a Task, so run()'s straggler sweep cannot cancel it.
    """

    def __init__(self, release: asyncio.Event) -> None:
        self._release = release
        self._handle: asyncio.TimerHandle | None = None
        self.fired = False

    def arm(self) -> None:
        self._handle = asyncio.get_running_loop().call_later(_HANG_GUARD_SECS, self._fire)

    def _fire(self) -> None:
        self.fired = True
        self._release.set()

    def cancel(self) -> None:
        if self._handle is not None:
            self._handle.cancel()


async def _absorb_one_cancel(on_absorbed: Callable[[], None]) -> None:
    try:
        await asyncio.Event().wait()
    except asyncio.CancelledError:
        on_absorbed()
    await asyncio.Event().wait()


async def _ignore_cancels_until(release: asyncio.Event) -> None:
    while not release.is_set():
        try:
            await release.wait()
        except asyncio.CancelledError:
            continue


class _Dispatch:
    """``acquire_next`` stand-in handing out ``task_ids`` in order, then None.

    Its tasks are not main-health fixes, so a restricted tick hands out none.
    Call number N blocks until ``held_ticks[N]`` is set, like a long tick.
    """

    def __init__(self, task_ids: list[str]) -> None:
        self._pending = list(task_ids)
        self.calls = 0
        self.scopes: list[DispatchScope] = []
        self.stop_when: Callable[[], bool] = lambda: False
        self.held_ticks: dict[int, asyncio.Event] = {}

    async def acquire_next(self, scope: DispatchScope = DispatchScope.ANY) -> TaskAssignment | None:
        self.calls += 1
        self.scopes.append(scope)
        if self.calls in self.held_ticks:
            await self.held_ticks[self.calls].wait()
        if self.stop_when():
            raise _Stop
        if scope is not DispatchScope.ANY or not self._pending:
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


def _harness(
    tmp_path: Path, dispatch: _Dispatch, slots: _Slots, *,
    max_concurrent_tasks: int = 1, main_health_fix_reserved_slots: int = 0,
) -> Harness:
    config = OrchestratorConfig(
        project_root=tmp_path,
        max_concurrent_tasks=max_concurrent_tasks,
        main_health_fix_reserved_slots=main_health_fix_reserved_slots,
        idle_poll_secs=_TICK,
        merge_heartbeat_interval_secs=_TICK,
        stale_service_restart_interval_secs=_TICK,
        watcher_supervisor_enabled=False,
        review=ReviewConfig(full_review_on_complete=False),
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
    h.git_ops.reap_interactive_worktrees = AsyncMock(return_value=[])
    h.git_ops.worktree_base = tmp_path / '.worktrees'
    h.usage_gate = None
    h.review_checkpoint = None

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


def _fused_memory_restart(
    h: Harness, clock: _Clock, fired: asyncio.Event,
    *, force_fire_after_secs: float = _FORCE_FIRE_AFTER_SECS,
) -> StaleServiceRestartCoordinator:
    async def executor() -> None:
        fired.set()

    coord = StaleServiceRestartCoordinator(
        git_ops=MagicMock(),
        event_store=None,
        debounce_secs=0.0,
        force_fire_after_secs=force_fire_after_secs,
        restart_executor=executor,
        clock=clock,
    )
    h._service_restart_coordinators = [coord]
    return coord


async def _arm(coord: StaleServiceRestartCoordinator) -> None:
    armed = await coord.note_merge(
        '7', 'base', 'head', prefetched_diff=['fused-memory/src/fused_memory/x.py'],
    )
    assert armed


def _heartbeat_ts(fleet_dir: Path) -> float | None:
    path = fleet_dir / f'{_UNIT}.json'
    if not path.exists():
        return None
    return json.loads(path.read_text())['ts_epoch']


def _shrink_shutdown_drains(h: Harness, monkeypatch: pytest.MonkeyPatch) -> None:
    monkeypatch.setattr(h, '_SHUTDOWN_WORKFLOW_DRAIN_SECS', _SHUTDOWN_DRAIN_SECS)
    monkeypatch.setattr(h, '_SHUTDOWN_STRAGGLER_DRAIN_SECS', _SHUTDOWN_DRAIN_SECS)


def _harness_errors_naming(caplog: pytest.LogCaptureFixture, name: str) -> list[str]:
    return [
        r.getMessage() for r in caplog.records
        if r.name == 'orchestrator.harness' and r.levelno == logging.ERROR and name in r.getMessage()
    ]


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
        review = MagicMock()
        review.should_trigger.return_value = False
        h.review_checkpoint = review

        async def drive() -> None:
            for tid, calls in (('1', 1), ('2', 2), ('3', 3)):
                await _until(lambda n=calls: dispatch.calls >= n, f'dispatch call {calls}')
                slots.gates[tid].set()

        report = await _run_driven(h, slots, drive)

        assert sorted(r.task_id for r in report.task_reports) == ['1', '2', '3']
        assert report.completed == 3
        assert review.record_merge.call_count == 3

    async def test_slot_finishing_during_a_long_acquire_next_tick_is_collected(
        self, tmp_path: Path, fleet_dir: Path,
    ) -> None:
        dispatch, slots = _Dispatch(['1']), _Slots()
        tick = asyncio.Event()
        dispatch.held_ticks[2] = tick
        h = _harness(tmp_path, dispatch, slots, main_health_fix_reserved_slots=1)

        async def drive() -> None:
            try:
                await _until(lambda: dispatch.calls >= 2, 'acquire_next tick in progress')
                slots.gates['1'].set()
                await _until(lambda: '1' in slots.finished, 'slot 1 finished mid-tick')
            finally:
                tick.set()

        report = await _run_driven(h, slots, drive)

        assert [r.task_id for r in report.task_reports] == ['1']
        assert dispatch.scopes[:2] == [DispatchScope.ANY, DispatchScope.MAIN_HEALTH_FIX], (
            'tick 2 is the reserve tick, run while slot 1 holds the only general slot'
        )

    async def test_owed_restart_force_fires_while_parked(
        self, tmp_path: Path, fleet_dir: Path,
    ) -> None:
        dispatch, slots = _Dispatch(['1', '2']), _Slots()
        h = _harness(tmp_path, dispatch, slots)
        clock, fired = _Clock(), asyncio.Event()
        coord = _fused_memory_restart(h, clock, fired)

        async def drive() -> None:
            await _until(lambda: dispatch.calls >= 1, 'loop parked on slot 1')
            await _arm(coord)
            reads_when_armed = clock.reads
            await _until(lambda: clock.reads >= reads_when_armed + 3, 'restart polled while parked')
            assert not fired.is_set(), 'a polite restart must wait while a slot is live'
            clock.now += _FORCE_FIRE_AFTER_SECS
            await _until(fired.is_set, 'owed restart force-fired while parked')
            assert dispatch.calls == 1, 'the loop must still be parked when it fires'

        await _run_driven(h, slots, drive)

    async def test_heartbeat_advances_while_parked(
        self, tmp_path: Path, fleet_dir: Path,
    ) -> None:
        dispatch, slots = _Dispatch(['1', '2']), _Slots()
        h = _harness(tmp_path, dispatch, slots)

        async def drive() -> None:
            await _until(lambda: dispatch.calls >= 1, 'loop parked on slot 1')
            await _until(lambda: _heartbeat_ts(fleet_dir) is not None, 'first heartbeat')
            first = _heartbeat_ts(fleet_dir) or 0.0
            await _until(lambda: (_heartbeat_ts(fleet_dir) or 0.0) > first, 'heartbeat advanced')
            assert dispatch.calls == 1, 'the loop must still be parked'

        await _run_driven(h, slots, drive)

    async def test_slot_cancelled_at_shutdown_is_collected(
        self, tmp_path: Path, fleet_dir: Path,
    ) -> None:
        dispatch, slots = _Dispatch(['1', '2']), _Slots()
        h = _harness(tmp_path, dispatch, slots)
        main = asyncio.current_task()
        assert main is not None

        async def sigterm_while_parked() -> None:
            await _until(lambda: dispatch.calls >= 1, 'loop parked on slot 1')
            main.cancel()

        with pytest.raises(asyncio.CancelledError):
            await _run_driven(h, slots, sigterm_while_parked)

        outcomes = {r.task_id: r.outcome for r in h.report.task_reports}
        assert outcomes == {'1': WorkflowOutcome.CANCELLED}

    async def test_cancel_immune_bystander_does_not_hang_the_straggler_sweep(
        self, tmp_path: Path, fleet_dir: Path,
        monkeypatch: pytest.MonkeyPatch, caplog: pytest.LogCaptureFixture,
    ) -> None:
        caplog.set_level(logging.ERROR, logger='orchestrator.harness')
        dispatch, slots = _Dispatch(['1', '2']), _Slots()
        h = _harness(tmp_path, dispatch, slots)
        _shrink_shutdown_drains(h, monkeypatch)
        release = asyncio.Event()
        guard = _HangGuard(release)
        main = asyncio.current_task()
        assert main is not None
        bystanders: list[asyncio.Task] = []

        async def sigterm_beside_an_immune_bystander() -> None:
            await _until(lambda: dispatch.calls >= 1, 'loop parked on slot 1')
            bystanders.append(asyncio.create_task(_ignore_cancels_until(release), name='immune-bystander'))
            await asyncio.sleep(0)
            guard.arm()
            main.cancel()

        try:
            with pytest.raises(asyncio.CancelledError):
                await _run_driven(h, slots, sigterm_beside_an_immune_bystander)

            assert not guard.fired, 'shutdown waited on the cancel-immune bystander until the hang guard freed it'
            [bystander] = bystanders
            assert not bystander.done(), 'shutdown must not wait for a task that ignores its cancel'
            assert _harness_errors_naming(caplog, 'immune-bystander')
            outcomes = {r.task_id: r.outcome for r in h.report.task_reports}
            assert outcomes == {'1': WorkflowOutcome.CANCELLED}
        finally:
            guard.cancel()
            release.set()
            if bystanders:
                await asyncio.wait(bystanders)

    async def test_cancel_immune_slot_is_detached_and_its_sibling_still_collected(
        self, tmp_path: Path, fleet_dir: Path,
        monkeypatch: pytest.MonkeyPatch, caplog: pytest.LogCaptureFixture,
    ) -> None:
        caplog.set_level(logging.ERROR, logger='orchestrator.harness')
        dispatch, slots = _Dispatch(['1', '2', '3']), _Slots()
        slots.immune = {'1'}
        h = _harness(tmp_path, dispatch, slots, max_concurrent_tasks=2)
        _shrink_shutdown_drains(h, monkeypatch)
        guard = _HangGuard(slots.immune_release)
        main = asyncio.current_task()
        assert main is not None
        immune_slots: list[asyncio.Task] = []

        async def sigterm_with_both_slots_running() -> None:
            await _until(lambda: {'1', '2'} <= slots.started, 'slots 1 and 2 running')
            immune_slots.append(next(t for t in asyncio.all_tasks() if t.get_name() == 'workflow-1'))
            guard.arm()
            main.cancel()

        try:
            with pytest.raises(asyncio.CancelledError):
                await _run_driven(h, slots, sigterm_with_both_slots_running)

            assert not guard.fired, 'shutdown waited on the cancel-immune slot until the hang guard freed it'
            [immune_slot] = immune_slots
            assert not immune_slot.done(), 'shutdown must not wait for a slot that ignores its cancel'
            assert immune_slot not in h._live_tasks
            assert _harness_errors_naming(caplog, 'workflow-1')
            outcomes = {r.task_id: r.outcome for r in h.report.task_reports}
            assert outcomes == {'2': WorkflowOutcome.CANCELLED}
        finally:
            guard.cancel()
            slots.immune_release.set()
            if immune_slots:
                await asyncio.wait(immune_slots)

    async def test_slot_surviving_one_cancel_unwinds_in_the_workflow_drain(
        self, tmp_path: Path, fleet_dir: Path,
        monkeypatch: pytest.MonkeyPatch, caplog: pytest.LogCaptureFixture,
    ) -> None:
        caplog.set_level(logging.ERROR, logger='orchestrator.harness')
        dispatch, slots = _Dispatch(['1', '2']), _Slots()
        slots.absorbs_one_cancel = {'1'}
        h = _harness(tmp_path, dispatch, slots)
        _shrink_shutdown_drains(h, monkeypatch)
        monkeypatch.setattr(
            h.mcp, 'stop', AsyncMock(side_effect=lambda: slots.events.append('mcp.stop')),
        )
        main = asyncio.current_task()
        assert main is not None

        async def sigterm_while_parked() -> None:
            await _until(lambda: dispatch.calls >= 1, 'loop parked on slot 1')
            main.cancel()

        with pytest.raises(asyncio.CancelledError):
            await _run_driven(h, slots, sigterm_while_parked)

        assert slots.events == ['slot 1 absorbed a cancel', 'slot 1 unwound', 'mcp.stop']
        assert _harness_errors_naming(caplog, 'workflow-1') == []
        outcomes = {r.task_id: r.outcome for r in h.report.task_reports}
        assert outcomes == {'1': WorkflowOutcome.CANCELLED}

    async def test_polite_restart_waits_for_the_dispatch_loop(
        self, tmp_path: Path, fleet_dir: Path,
    ) -> None:
        dispatch, slots = _Dispatch([]), _Slots()
        h = _harness(tmp_path, dispatch, slots)
        clock, fired = _Clock(), asyncio.Event()
        await _arm(_fused_memory_restart(h, clock, fired))

        async def startup_recovery() -> None:
            await _until(lambda: clock.reads >= 4, 'restart polled during startup')
            assert not fired.is_set(), 'no polite restart during startup recovery'

        h._recover_crashed_tasks = startup_recovery  # type: ignore[method-assign]
        dispatch.stop_when = fired.is_set

        with pytest.raises(_Stop):
            await h.run()
        assert fired.is_set()

    async def test_polite_restart_waits_out_an_inline_full_review(
        self, tmp_path: Path, fleet_dir: Path,
    ) -> None:
        dispatch, slots = _Dispatch(['1']), _Slots()
        h = _harness(tmp_path, dispatch, slots)
        h.config.review.full_review_on_complete = True
        review_started, review_done = asyncio.Event(), asyncio.Event()

        async def run_full() -> MagicMock:
            review_started.set()
            await review_done.wait()
            return MagicMock(findings_count=0, tasks_created=[], cost_usd=0.0, parse_failed=False)

        review = MagicMock()
        review.should_trigger.return_value = False
        review.should_run_full.return_value = True
        review.run_full = run_full
        h.review_checkpoint = review
        clock, fired = _Clock(), asyncio.Event()
        coord = _fused_memory_restart(h, clock, fired, force_fire_after_secs=0.0)
        observed: dict[str, bool] = {}

        async def drive() -> None:
            await _until(lambda: dispatch.calls >= 1, 'slot 1 dispatched')
            slots.gates['1'].set()
            try:
                await _until(review_started.is_set, 'full review running inline')
                await _arm(coord)
                reads = clock.reads
                await _until(
                    lambda: fired.is_set() or clock.reads >= reads + 3,
                    'restart polled during the full review',
                )
                observed['fired_during_review'] = fired.is_set()
            finally:
                review_done.set()

        dispatch.stop_when = fired.is_set
        with pytest.raises(_Stop):
            await _run_driven(h, slots, drive, until_idle=False)

        assert observed == {'fired_during_review': False}
        assert fired.is_set(), 'the polite restart fires once the loop idles'
