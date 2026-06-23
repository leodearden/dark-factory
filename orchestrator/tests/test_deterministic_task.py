"""B+H integration-gate suite for the deterministic task kind, B1–B12.

Exercises the full integration of the deterministic task kind over the landed
β(1899)/γ(1900)/ε(1902)/α(1898)/δ(1901) PRD implementations:
 - Pure gate path: born-at-L2 escalation, quiescence, proceed/no-go resolution,
   restart stamp-clear re-fire, orchestrator-restart replay, strand-reaper
   invisibility, no-lock (B1–B5, B11, B12).
 - Auto-deploy path: cross-unit success, failure+reaper no-rerun, self-restart
   scheduled, δ submit CLI L2, α validation rejection corners (B6–B10).
"""

from __future__ import annotations

import asyncio
from pathlib import Path
from unittest.mock import AsyncMock, MagicMock, patch

import pytest
from escalation.models import Escalation
from escalation.queue import EscalationQueue

from orchestrator.deterministic_runner import DeterministicRunner
from orchestrator.harness import Harness
from orchestrator.scheduler import Scheduler, TaskAssignment
from orchestrator.workflow import WorkflowOutcome


# ---------------------------------------------------------------------------
# Task-dict builders
# ---------------------------------------------------------------------------

def _gate_task(
    task_id: str = 'gate-99',
    deps: list | None = None,
    title: str = 'Release gate',
    **stamps,
) -> dict:
    """Build a minimal deterministic pure-gate task dict.

    ``stamps`` are merged directly into ``metadata`` (e.g. gate_escalated_at=...).
    ``deps`` is a list of dependency dicts or bare string/int ids; None = no deps.
    """
    metadata: dict = {
        'task_kind': 'deterministic',
        'always_escalates': True,
        'before_done': None,
        **stamps,
    }
    task: dict = {
        'id': task_id,
        'title': title,
        'description': f'Gate: {title}',
        'metadata': metadata,
        'status': 'pending',
    }
    if deps is not None:
        task['dependencies'] = [
            {'id': str(d)} if not isinstance(d, dict) else d
            for d in deps
        ]
    return task


def _make_assignment(task: dict) -> TaskAssignment:
    """Build a TaskAssignment(modules=[]) for a deterministic (no-lock) task."""
    return TaskAssignment(task_id=str(task['id']), task=task, modules=[])


# ---------------------------------------------------------------------------
# _StoreScheduler — persisting in-memory scheduler stand-in
# ---------------------------------------------------------------------------

class _StoreScheduler:
    """Persisting in-memory scheduler stand-in for B1–B12 integration tests.

    The existing ``_workflow_helpers.FakeScheduler`` cannot be used here because
    its ``update_task`` is a no-op that never persists stamps — but B3/B4/B7/B11
    require stamps (gate_escalated_at, before_done_ran_at, …) to survive across
    re-dispatch calls and simulated orchestrator restarts.

    Implements the exact surface that DeterministicRunner and
    ``Harness._run_slot`` / ``_action_teardown_and_set_status`` touch:
      - ``get_task(task_id)`` — returns the current task dict (async)
      - ``get_status(task_id)`` — returns the latest status from history (async)
      - ``set_task_status(task_id, status, *, done_provenance=None)`` — appends
        to status history, persists done_provenance in task metadata (async)
      - ``update_task(task_id, fields, *, metadata_mode='merge')`` — merges
        fields into task['metadata'] (async)
      - ``release(task_id, *, requeued=False)`` — no-op (sync)
      - ``is_deterministic`` — staticmethod delegating to Scheduler.is_deterministic
    """

    is_deterministic = staticmethod(Scheduler.is_deterministic)

    def __init__(self) -> None:
        self._tasks: dict[str, dict] = {}
        self._status_history: dict[str, list[str]] = {}

    def seed(self, task: dict) -> None:
        """Seed a task dict into the store (used during test setup)."""
        tid = str(task['id'])
        # Deep-copy the dict so tests can mutate without aliasing
        import copy
        self._tasks[tid] = copy.deepcopy(task)
        init_status = task.get('status', 'pending')
        self._status_history[tid] = [init_status]

    async def get_task(self, task_id: str) -> dict | None:
        return self._tasks.get(str(task_id))

    async def get_status(self, task_id: str) -> str | None:
        hist = self._status_history.get(str(task_id), [])
        return hist[-1] if hist else None

    async def set_task_status(
        self,
        task_id: str,
        status: str,
        *,
        done_provenance: dict | None = None,
    ) -> None:
        tid = str(task_id)
        self._status_history.setdefault(tid, []).append(status)
        if tid in self._tasks:
            self._tasks[tid]['status'] = status
            if done_provenance is not None:
                self._tasks[tid].setdefault('metadata', {})['done_provenance'] = done_provenance

    async def update_task(
        self,
        task_id: str,
        fields: dict,
        *,
        metadata_mode: str = 'merge',
    ) -> bool:
        tid = str(task_id)
        if tid not in self._tasks:
            return False
        meta = self._tasks[tid].setdefault('metadata', {})
        if metadata_mode == 'merge':
            meta.update(fields)
        else:
            self._tasks[tid]['metadata'] = dict(fields)
        return True

    def release(self, task_id: str, *, requeued: bool = False) -> None:
        """No-op — no module lock table in this stand-in."""


# ---------------------------------------------------------------------------
# det_harness fixture and _dispatch helper
# ---------------------------------------------------------------------------

def _build_harness(mock_orch_config, queue_dir: Path, store: _StoreScheduler) -> Harness:
    """Build a Harness with patched internals and wired stand-ins.

    Follows the fixture pattern from test_harness_deterministic_dispatch.py:
    - Patch McpLifecycle/OverrideStore/Scheduler/BriefingAssembler at ctor.
    - Replace h.scheduler with the given persisting _StoreScheduler.
    - Wire h._escalation_queue with a real file-backed EscalationQueue.
    - Wire h.git_ops with a MagicMock (AsyncMock release_lane_for_terminal_task).
    - Set h._run_id=None so _apply_retry_cap no-ops.
    """
    with (
        patch('orchestrator.harness.McpLifecycle'),
        patch('orchestrator.harness.OverrideStore'),
        patch('orchestrator.harness.Scheduler'),
        patch('orchestrator.harness.BriefingAssembler'),
    ):
        h = Harness(mock_orch_config)

    h.scheduler = store
    h._escalation_queue = EscalationQueue(queue_dir)
    git_ops = MagicMock()
    git_ops.release_lane_for_terminal_task = AsyncMock()
    h.git_ops = git_ops
    h._run_id = None
    return h


@pytest.fixture
def det_harness(tmp_path: Path, mock_orch_config) -> Harness:
    """Harness + real EscalationQueue + _StoreScheduler for gate-row integration tests.

    Access the stand-in via ``h.scheduler`` (cast to _StoreScheduler).
    Access the queue via ``h._escalation_queue``.
    The queue_dir is ``tmp_path / 'queue'``; pass it to a second _build_harness
    call in B11 to simulate an orchestrator restart over the same queue.
    """
    store = _StoreScheduler()
    return _build_harness(mock_orch_config, tmp_path / 'queue', store)


async def _dispatch(h: Harness, task_id: str):
    """Re-read the current task from the store and dispatch it through _run_slot.

    Rebuilds the assignment from the store's latest task dict so that stamps
    written by prior dispatches (gate_escalated_at, before_done_ran_at, …)
    are visible to the runner on re-dispatch.  Passes a fresh Semaphore(1);
    the finally block releases it (the value increments to 2, harmless).

    Returns the TaskReport returned by _run_slot.
    """
    task = await h.scheduler.get_task(str(task_id))
    assignment = _make_assignment(task)
    return await h._run_slot(assignment, asyncio.Semaphore(1))


# ---------------------------------------------------------------------------
# Smoke tests (step-1): fixture builds + _StoreScheduler round-trips
# ---------------------------------------------------------------------------

@pytest.mark.asyncio
class TestSmoke:
    """Smoke: det_harness builds, _StoreScheduler metadata round-trips persist."""

    async def test_det_harness_builds(self, det_harness: Harness) -> None:
        """det_harness returns a Harness with _StoreScheduler and real EscalationQueue."""
        h = det_harness
        assert isinstance(h, Harness)
        assert isinstance(h.scheduler, _StoreScheduler)
        assert isinstance(h._escalation_queue, EscalationQueue)
        assert h._run_id is None

    async def test_store_scheduler_update_task_merges_and_persists(
        self, det_harness: Harness
    ) -> None:
        """update_task(metadata_mode='merge') merges fields and preserves existing fields."""
        store: _StoreScheduler = det_harness.scheduler
        task = _gate_task(task_id='smoke-1')
        store.seed(task)

        stamp = '2026-01-01T00:00:00+00:00'
        await store.update_task('smoke-1', {'gate_escalated_at': stamp})

        retrieved = await store.get_task('smoke-1')
        assert retrieved is not None
        assert retrieved['metadata']['gate_escalated_at'] == stamp
        # Existing metadata fields are preserved (not wiped)
        assert retrieved['metadata']['task_kind'] == 'deterministic'
        assert retrieved['metadata']['always_escalates'] is True

    async def test_store_scheduler_set_status_persists(self, det_harness: Harness) -> None:
        """set_task_status persists status transitions and done_provenance."""
        store: _StoreScheduler = det_harness.scheduler
        task = _gate_task(task_id='smoke-2')
        store.seed(task)

        await store.set_task_status('smoke-2', 'blocked')
        assert await store.get_status('smoke-2') == 'blocked'

        provenance = {'kind': 'deterministic-deploy', 'pid': 12345}
        await store.set_task_status('smoke-2', 'done', done_provenance=provenance)
        assert await store.get_status('smoke-2') == 'done'
        retrieved = await store.get_task('smoke-2')
        assert retrieved is not None
        assert retrieved['metadata']['done_provenance'] == provenance
