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

from orchestrator.config import OrchestratorConfig
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
