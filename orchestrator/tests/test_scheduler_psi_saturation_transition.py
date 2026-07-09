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
