"""Tests for the scheduler's dispatch-admission gate (task 2328, DA3 of PRD
docs/prds/dispatch-admission-load-cap.md).

Covers:
  step-1  (RED)   Throttle direction (scored loop) + event payload: a HEAVY
                  (normal-kind) candidate is deferred under PSI saturation
                  while a DETERMINISTIC candidate scored lower in the same
                  tick still dispatches; exactly one ``dispatch_deferred``
                  event fires with the ``{metric, value, in_flight, floor}``
                  payload.

Mirrors test_scheduler_landed_dispatch_gate.py / test_scheduler_deterministic.py
conventions: module-local task-dict helpers, ``scheduler.get_tasks =
AsyncMock(return_value=[...])``, driving ``acquire_next()`` directly, and
``_RecordingEventStore`` for event assertions.
"""

from __future__ import annotations

from unittest.mock import AsyncMock

import pytest

from orchestrator.config import OrchestratorConfig, PsiAdmissionConfig
from orchestrator.scheduler import Scheduler
from shared.psi import PsiSample

from _recording_event_store import _RecordingEventStore


# ---------------------------------------------------------------------------
# Minimal task-dict + PSI-sample helpers
# ---------------------------------------------------------------------------


def _heavy_task(
    task_id: str,
    priority: str = 'medium',
    files: list[str] | None = None,
) -> dict:
    """Minimal normal-kind (heavy) pending-task dict carrying every field
    acquire_next reads."""
    return {
        'id': task_id,
        'title': f'Task {task_id}',
        'status': 'pending',
        'priority': priority,
        'dependencies': [],
        'metadata': {'task_kind': 'normal', 'files': files or [f'mod{task_id}']},
    }


def _det_task(
    task_id: str,
    priority: str = 'medium',
    files: list[str] | None = None,
) -> dict:
    """Minimal deterministic-kind pending-task dict carrying every field
    acquire_next reads."""
    return {
        'id': task_id,
        'title': f'Task {task_id}',
        'status': 'pending',
        'priority': priority,
        'dependencies': [],
        'metadata': {'task_kind': 'deterministic', 'files': files or [f'mod{task_id}']},
    }


def _psi(
    *,
    cpu_some10: float = 0.0,
    mem_some10: float = 0.0,
    mem_full10: float = 0.0,
    io_some10: float = 0.0,
    read_ok: bool = True,
) -> PsiSample:
    """Build a PsiSample; all metrics idle (0.0) unless overridden."""
    return PsiSample(
        cpu_some10=cpu_some10,
        mem_some10=mem_some10,
        mem_full10=mem_full10,
        io_some10=io_some10,
        read_ok=read_ok,
    )


# ---------------------------------------------------------------------------
# step-1 — Throttle direction (scored loop) + event payload
# ---------------------------------------------------------------------------


@pytest.mark.asyncio
class TestDispatchAdmissionThrottleDirection:
    """A saturated host defers a HEAVY candidate but still dispatches a
    DETERMINISTIC candidate scored lower in the same tick."""

    async def test_heavy_deferred_deterministic_dispatches_same_tick(self) -> None:
        event_store = _RecordingEventStore()
        scheduler = Scheduler(OrchestratorConfig(max_per_module=1), event_store=event_store)
        # cpu_some10=90 is OVER the default cpu_some_avg10=85 threshold.
        scheduler._read_psi_sample = lambda: _psi(cpu_some10=90.0, read_ok=True)
        # >= default min_inflight_floor=1, so the floor does not suppress the hold.
        scheduler._dispatched = {'inflight1'}

        # H must outscore D so the scored loop tries H first (skips it, then
        # dispatches D) — otherwise D would dispatch and return before H is
        # ever evaluated, and no dispatch_deferred event would fire.
        task_h = _heavy_task('H', priority='high')
        task_d = _det_task('D', priority='medium')
        scheduler.get_tasks = AsyncMock(return_value=[task_h, task_d])

        result = await scheduler.acquire_next()

        assert result is not None
        assert result.task_id == 'D', 'deterministic candidate must still dispatch this tick'
        assert 'H' not in scheduler._dispatched, 'heavy candidate must be deferred under saturation'
        assert 'D' in scheduler._dispatched

        deferred_events = [e for e in event_store.events if e[0] == 'dispatch_deferred']
        assert len(deferred_events) == 1, 'exactly one dispatch_deferred event this tick'
        assert deferred_events[0][1]['data'] == {
            'metric': 'cpu_some_avg10',
            'value': 90.0,
            'in_flight': 1,
            'floor': 1,
        }


# ---------------------------------------------------------------------------
# step-3 — Work-conserving + floor/deadlock-freedom + metric ranking
# ---------------------------------------------------------------------------


@pytest.mark.asyncio
class TestDispatchAdmissionWorkConserving:
    """An idle host never throttles — ALL heavy candidates dispatch across
    successive ticks, with zero dispatch_deferred events, even when
    in-flight is already at/above the floor."""

    async def test_idle_psi_dispatches_all_heavy_candidates(self) -> None:
        event_store = _RecordingEventStore()
        scheduler = Scheduler(OrchestratorConfig(max_per_module=1), event_store=event_store)
        scheduler._read_psi_sample = lambda: _psi(read_ok=True)  # all metrics 0.0 — idle
        scheduler._dispatched = {'inflight1'}  # already >= default floor=1

        tasks = [
            _heavy_task('H1', files=['src/h1.py']),
            _heavy_task('H2', files=['src/h2.py']),
            _heavy_task('H3', files=['src/h3.py']),
        ]
        scheduler.get_tasks = AsyncMock(return_value=tasks)

        dispatched_ids = []
        for _ in range(3):
            result = await scheduler.acquire_next()
            assert result is not None
            dispatched_ids.append(result.task_id)

        assert set(dispatched_ids) == {'H1', 'H2', 'H3'}
        assert {'H1', 'H2', 'H3'} <= scheduler._dispatched
        assert [e for e in event_store.events if e[0] == 'dispatch_deferred'] == []


@pytest.mark.asyncio
class TestDispatchAdmissionFloorDeadlockFreedom:
    """The anti-deadlock floor: a hold can never engage with fewer than
    ``min_inflight_floor`` tasks in flight on this orchestrator — "hold with
    nothing running" is unreachable by construction."""

    async def test_zero_in_flight_still_dispatches_under_saturation(self) -> None:
        event_store = _RecordingEventStore()
        scheduler = Scheduler(OrchestratorConfig(max_per_module=1), event_store=event_store)
        scheduler._read_psi_sample = lambda: _psi(cpu_some10=90.0, read_ok=True)
        # scheduler._dispatched left empty — in_flight=0 < default floor=1.

        task_h = _heavy_task('H')
        scheduler.get_tasks = AsyncMock(return_value=[task_h])

        result = await scheduler.acquire_next()

        assert result is not None
        assert result.task_id == 'H'
        assert 'H' in scheduler._dispatched
        assert [e for e in event_store.events if e[0] == 'dispatch_deferred'] == []

    async def test_in_flight_below_configured_floor_still_dispatches(self) -> None:
        event_store = _RecordingEventStore()
        config = OrchestratorConfig(
            max_per_module=1,
            psi_admission=PsiAdmissionConfig(min_inflight_floor=2),
        )
        scheduler = Scheduler(config, event_store=event_store)
        scheduler._read_psi_sample = lambda: _psi(cpu_some10=90.0, read_ok=True)
        scheduler._dispatched = {'inflight1'}  # 1 < configured floor=2

        task_h = _heavy_task('H')
        scheduler.get_tasks = AsyncMock(return_value=[task_h])

        result = await scheduler.acquire_next()

        assert result is not None
        assert result.task_id == 'H'
        assert 'H' in scheduler._dispatched
        assert [e for e in event_store.events if e[0] == 'dispatch_deferred'] == []


@pytest.mark.asyncio
class TestDispatchAdmissionMetricRanking:
    """The gating metric+value reported in the dispatch_deferred event
    follows DA-D1 ranked order: cpu_some_avg10 > mem_some_avg10 >
    mem_full_avg10 > io_some_avg10."""

    async def test_cpu_and_io_both_saturated_reports_cpu(self) -> None:
        event_store = _RecordingEventStore()
        scheduler = Scheduler(OrchestratorConfig(max_per_module=1), event_store=event_store)
        scheduler._read_psi_sample = lambda: _psi(cpu_some10=90.0, io_some10=50.0, read_ok=True)
        scheduler._dispatched = {'inflight1'}

        task_h = _heavy_task('H')
        scheduler.get_tasks = AsyncMock(return_value=[task_h])

        await scheduler.acquire_next()

        deferred_events = [e for e in event_store.events if e[0] == 'dispatch_deferred']
        assert len(deferred_events) == 1
        assert deferred_events[0][1]['data']['metric'] == 'cpu_some_avg10'

    async def test_io_only_saturated_reports_io(self) -> None:
        event_store = _RecordingEventStore()
        scheduler = Scheduler(OrchestratorConfig(max_per_module=1), event_store=event_store)
        scheduler._read_psi_sample = lambda: _psi(io_some10=50.0, read_ok=True)
        scheduler._dispatched = {'inflight1'}

        task_h = _heavy_task('H')
        scheduler.get_tasks = AsyncMock(return_value=[task_h])

        await scheduler.acquire_next()

        deferred_events = [e for e in event_store.events if e[0] == 'dispatch_deferred']
        assert len(deferred_events) == 1
        assert deferred_events[0][1]['data']['metric'] == 'io_some_avg10'
