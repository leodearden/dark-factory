"""End-to-end scheduler boundary test — PSI saturation transition (task 2329,
DA4 of PRD docs/prds/dispatch-admission-load-cap.md).

DA4 is the batch leaf / integration gate for the L3b dispatch-admission load
cap: it drives the REAL ``Scheduler.acquire_next()`` selection + a real
``ModuleLockTable`` + a real SQLite ``EventStore`` across a PSI saturation
TRANSITION (idle -> saturated -> idle) with a burst of ready HEAVY
(normal-kind) tasks plus at least one DETERMINISTIC task, proving the PRD §6
work-conserving + deadlock-free invariants end-to-end — not just the DA3 gate
(task 2328)'s isolated single-tick unit tests in
``test_scheduler_dispatch_admission.py``.

This is a TEST-ONLY task — no new production module.  The "impl" steps in
this file build a reusable multi-tick transition harness (``_PsiFeed`` +
``_Driver``) that models the harness main loop itself (live in-flight
tracking, max_concurrent enforcement, and task completion via the REAL
``scheduler.release()``), since ``acquire_next()`` itself enforces neither.

Covers three scenarios (see the ``TestPsiSaturationTransition`` class):
  test_full_transition_idle_saturated_idle
      Scenario 1 — full idle -> saturated -> idle transition end-to-end.
  test_floor_holds_under_sustained_saturation
      Scenario 2 — deadlock-freedom at the peak of SUSTAINED saturation.
  test_recovery_restores_full_dispatch_up_to_cap
      Scenario 3 — work-conserving restore up to max_concurrent_tasks with
      NO residual hold after recovery.

Helpers are MODULE-LOCAL (not conftest.py) per the DA2/DA3 rationale: a
conftest.py edit trips verify.py's has_conftest scoped-verify fallback.
"""

from __future__ import annotations

from unittest.mock import AsyncMock

import pytest
from shared.psi import PsiSample

from orchestrator.config import OrchestratorConfig, PsiAdmissionConfig
from orchestrator.event_store import EventStore
from orchestrator.scheduler import Scheduler

# ---------------------------------------------------------------------------
# Minimal task-dict + PSI-sample helpers (mirrored from
# test_scheduler_dispatch_admission.py:44-93 — same minimal shape acquire_next
# reads, so DA4's fixtures match DA3's unit-test fixtures field-for-field).
# ---------------------------------------------------------------------------


def _heavy_task(
    task_id: str,
    priority: str = 'medium',
    files: list[str] | None = None,
) -> dict:
    """Minimal normal-kind (heavy) pending-task dict carrying every field
    acquire_next reads.  Defaults to a file DISTINCT per task_id so
    max_per_module=1 never masks the PSI gate with lock contention."""
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
# Scenario 1 — full idle -> saturated -> idle transition end-to-end
#
# RED (step-1): drives ticks via ``_FakeClock`` / ``_PsiFeed`` / ``_Driver``,
# none of which exist yet.  This is deliberate write-the-API-you-wish-you-had
# TDD for the test-harness infrastructure itself (this task adds no
# production module) — the NameError below IS the RED failure; step-2 builds
# exactly the harness this test exercises.
# ---------------------------------------------------------------------------


@pytest.mark.asyncio
class TestPsiSaturationTransition:
    """DA4 integration gate: multi-tick PSI saturation transitions driven
    against the REAL Scheduler.acquire_next() + ModuleLockTable + SQLite
    EventStore (task 2329)."""

    async def test_full_transition_idle_saturated_idle(self, tmp_path) -> None:
        """Scenario 1: idle -> saturated -> idle, full transition.

        Backlog: 4 heavy (H1..H4, distinct files/modules) + 1 deterministic
        (D1).  Heavies outrank D1 (priority='high' vs D1's default 'medium')
        so the idle phase dispatches heavies first, never D1 — isolating D1's
        later dispatch as evidence of the deterministic exemption rather than
        plain scoring luck.
        """
        event_store = EventStore(db_path=tmp_path / 'psi.db', run_id='da4-s1')
        clock = _FakeClock()
        config = OrchestratorConfig(max_per_module=1, max_concurrent_tasks=6)

        tasks = [
            _heavy_task('H1', priority='high', files=['src/h1.py']),
            _heavy_task('H2', priority='high', files=['src/h2.py']),
            _heavy_task('H3', priority='high', files=['src/h3.py']),
            _heavy_task('H4', priority='high', files=['src/h4.py']),
            _det_task('D1', files=['src/d1.py']),
        ]

        feed = _PsiFeed()  # starts idle
        driver = _Driver.build(
            config, tasks, event_store=event_store, time_source=clock, psi_feed=feed,
        )

        # --- (A) idle: heavies dispatch, zero dispatch_deferred ---
        await driver.run_ticks(2)
        assert driver.heavy_in_flight() == 2, (
            'two heavy tasks should be in flight after 2 idle ticks'
        )
        assert driver.deferred_events() == [], 'an idle host must never defer dispatch'

        # --- (B) saturated: heavies held, deterministic still dispatches ---
        feed.set_saturated(cpu_some10=90.0)
        await driver.run_ticks(4)

        assert driver.heavy_in_flight() == 2, 'saturation must not admit new heavy dispatch'
        assert 'D1' in driver.dispatched_ids(), (
            'deterministic tasks are exempt from the PSI hold'
        )

        deferred_during_saturation = driver.deferred_events()
        assert len(deferred_during_saturation) >= 1, (
            'saturation must defer at least one heavy dispatch'
        )
        assert deferred_during_saturation[0]['data'] == {
            'metric': 'cpu_some_avg10',
            'value': 90.0,
            'in_flight': 2,
            'floor': 1,
        }
        assert all(
            e['task_id'] in {'H1', 'H2', 'H3', 'H4'} for e in deferred_during_saturation
        ), 'dispatch_deferred must only ever name a HEAVY candidate, never the deterministic task'
        assert driver.heavy_in_flight() >= config.psi_admission.min_inflight_floor, (
            'in-flight heavy must never fall below the anti-deadlock floor while held'
        )

        # --- (C) recovery: remaining heavies dispatch, no residual hold ---
        feed.set_idle()
        await driver.run_ticks(2)

        assert driver.heavy_in_flight() == 4, (
            'recovery must dispatch the remaining heavies (work-conserving)'
        )
        assert driver.dispatched_ids() == {'H1', 'H2', 'H3', 'H4', 'D1'}
        assert driver.deferred_events() == deferred_during_saturation, (
            'no NEW dispatch_deferred events may occur after the recovery flip'
        )
