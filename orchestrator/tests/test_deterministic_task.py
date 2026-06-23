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


# ---------------------------------------------------------------------------
# B1 — gate, deps unsatisfied (step-2)
# ---------------------------------------------------------------------------

class TestB1DepsUnsatisfied:
    """B1: a deterministic gate with unsatisfied deps is not dispatch-eligible.

    Confirms that _deps_satisfied returns False for a gate whose dependency is
    still pending, and that no escalation is filed while the gate waits.
    """

    def test_gate_not_deps_satisfied_when_dep_pending(self, tmp_path: Path) -> None:
        """Scheduler._deps_satisfied returns False when a dep is pending."""
        from orchestrator.config import OrchestratorConfig
        config = OrchestratorConfig(project_root=tmp_path)
        scheduler = Scheduler(config)

        gate = _gate_task(task_id='gate-b1', deps=[10])
        # Dep 10 is still pending
        status_map = {'10': 'pending'}
        assert scheduler._deps_satisfied(gate, status_map) is False

    def test_no_escalation_filed_before_dispatch(self, tmp_path: Path) -> None:
        """No escalation is filed for a gate that has never been dispatched."""
        queue = EscalationQueue(tmp_path / 'queue')
        # Gate was never dispatched — queue is empty
        assert queue.get_by_task('gate-b1', status='pending') == []

    def test_gate_deps_satisfied_when_dep_done(self, tmp_path: Path) -> None:
        """Once the dep is done, _deps_satisfied returns True (gate now eligible)."""
        from orchestrator.config import OrchestratorConfig
        config = OrchestratorConfig(project_root=tmp_path)
        scheduler = Scheduler(config)

        gate = _gate_task(task_id='gate-b1', deps=[10])
        status_map = {'10': 'done'}
        assert scheduler._deps_satisfied(gate, status_map) is True


# ---------------------------------------------------------------------------
# B2 — gate deps satisfied → exactly one born-at-L2 (step-3)
# ---------------------------------------------------------------------------

@pytest.mark.asyncio
class TestB2GateDepsOk:
    """B2: first dispatch of an eligible gate files exactly one born-at-L2 and blocks.

    Drives the REAL production path: _run_slot → _run_deterministic_slot →
    real DeterministicRunner + real file-backed EscalationQueue.
    The ONLY fake is the task backend (_StoreScheduler).
    """

    async def test_dispatch_returns_blocked_report(self, det_harness: Harness) -> None:
        """_run_slot returns BLOCKED with block_reason='deterministic_gate'."""
        store: _StoreScheduler = det_harness.scheduler
        task = _gate_task(task_id='gate-b2', title='Ship v2 gate')
        store.seed(task)

        report = await _dispatch(det_harness, 'gate-b2')

        assert report is not None
        assert report.outcome == WorkflowOutcome.BLOCKED
        assert report.block_reason == 'deterministic_gate'

    async def test_store_status_becomes_blocked(self, det_harness: Harness) -> None:
        """_StoreScheduler reflects 'blocked' status after gate dispatch."""
        store: _StoreScheduler = det_harness.scheduler
        task = _gate_task(task_id='gate-b2', title='Ship v2 gate')
        store.seed(task)

        await _dispatch(det_harness, 'gate-b2')

        assert await store.get_status('gate-b2') == 'blocked'

    async def test_exactly_one_born_at_l2_escalation(self, det_harness: Harness) -> None:
        """Exactly one pending L2 milestone_gate escalation is filed."""
        store: _StoreScheduler = det_harness.scheduler
        task = _gate_task(task_id='gate-b2', title='Ship v2 gate')
        store.seed(task)

        await _dispatch(det_harness, 'gate-b2')

        queue: EscalationQueue = det_harness._escalation_queue
        pending = queue.get_by_task('gate-b2', status='pending')
        assert len(pending) == 1, f'Expected 1 pending escalation, got {len(pending)}'

        esc = pending[0]
        assert esc.level == 2
        assert esc.agent_role == 'orchestrator-deterministic'
        assert esc.category == 'milestone_gate'
        assert esc.summary == 'Ship v2 gate'[:200]

    async def test_gate_escalated_at_stamped(self, det_harness: Harness) -> None:
        """metadata.gate_escalated_at is stamped in the store after dispatch."""
        store: _StoreScheduler = det_harness.scheduler
        task = _gate_task(task_id='gate-b2', title='Ship v2 gate')
        store.seed(task)

        await _dispatch(det_harness, 'gate-b2')

        retrieved = await store.get_task('gate-b2')
        assert retrieved is not None
        stamp = retrieved['metadata'].get('gate_escalated_at')
        assert stamp is not None, 'gate_escalated_at must be stamped after first dispatch'

    async def test_no_worktree_or_branch_created(self, det_harness: Harness) -> None:
        """No worktree/branch/agent is created — runner is built without git_ops (I4/B2)."""
        store: _StoreScheduler = det_harness.scheduler
        task = _gate_task(task_id='gate-b2', title='Ship v2 gate')
        store.seed(task)

        await _dispatch(det_harness, 'gate-b2')

        # git_ops is a MagicMock; the runner is constructed without git_ops
        # so none of the worktree-creation methods should be called.
        git_ops = det_harness.git_ops
        for method_name in ('create_worktree', 'checkout', 'clone_worktree'):
            mock_attr = getattr(git_ops, method_name, None)
            if mock_attr is not None and hasattr(mock_attr, 'call_count'):
                assert mock_attr.call_count == 0, (
                    f'git_ops.{method_name} must not be called for a deterministic gate'
                )


# ---------------------------------------------------------------------------
# B3 — gate quiescence: repeated sweeps do NOT file a second L2 (step-4)
# ---------------------------------------------------------------------------

@pytest.mark.asyncio
class TestB3Quiescence:
    """B3: re-dispatch of a blocked gate with a pending L2 stays BLOCKED without re-escalating.

    Drives the production quiescence guard in DeterministicRunner section 1:
    gate_escalated_at set + pending escalation → return BLOCKED (no re-file).
    """

    async def _reach_b2_state(self, h: Harness) -> None:
        """Seed a gate and dispatch once to reach the B2 blocked+L2 state."""
        store: _StoreScheduler = h.scheduler
        task = _gate_task(task_id='gate-b3', title='Quiescence gate')
        store.seed(task)
        await _dispatch(h, 'gate-b3')

    async def test_re_dispatches_all_return_blocked(self, det_harness: Harness) -> None:
        """Each re-dispatch returns BLOCKED (quiescence — still open L2)."""
        await self._reach_b2_state(det_harness)

        for _ in range(2):
            report = await _dispatch(det_harness, 'gate-b3')
            assert report is not None
            assert report.outcome == WorkflowOutcome.BLOCKED
            assert report.block_reason == 'deterministic_gate'

    async def test_still_exactly_one_escalation_after_re_dispatches(
        self, det_harness: Harness
    ) -> None:
        """queue.get_by_task still has exactly ONE pending escalation after 3 sweeps."""
        await self._reach_b2_state(det_harness)

        # Two additional sweeps (total 3 dispatches)
        await _dispatch(det_harness, 'gate-b3')
        await _dispatch(det_harness, 'gate-b3')

        queue: EscalationQueue = det_harness._escalation_queue
        pending = queue.get_by_task('gate-b3', status='pending')
        assert len(pending) == 1, (
            f'Expected exactly 1 pending escalation after quiescent re-dispatches, '
            f'got {len(pending)} (quiescence guard not firing?)'
        )

    async def test_status_remains_blocked_after_re_dispatches(
        self, det_harness: Harness
    ) -> None:
        """Status stays 'blocked' through all quiescent sweeps."""
        await self._reach_b2_state(det_harness)

        await _dispatch(det_harness, 'gate-b3')
        await _dispatch(det_harness, 'gate-b3')

        store: _StoreScheduler = det_harness.scheduler
        assert await store.get_status('gate-b3') == 'blocked'


# ---------------------------------------------------------------------------
# B4 — proceed: resume → done, no re-escalate, dependent eligible (step-5)
# ---------------------------------------------------------------------------

@pytest.mark.asyncio
class TestB4Proceed:
    """B4: resolving the L2 (proceed) and re-dispatching drives the gate to done.

    Production path:
    1. Dispatch → BLOCKED (B2 state).
    2. queue.resolve() → L2 archived (no longer pending).
    3. _action_teardown_and_set_status('resume') → gate re-pended, stamps PRESERVED.
    4. _dispatch() → runner section 1 sees gate_escalated_at set + no pending L2
       → drive to done (I2/B4).

    Note: we call _action_teardown_and_set_status directly (the production method
    invoked by _cascade_unblock_member) instead of wiring the full
    _on_escalation_resolved callback (which requires _schedule_coro_threadsafe /
    event-loop plumbing that is out of scope for unit tests).
    """

    async def _reach_resolved_state(self, h: Harness, task_id: str) -> None:
        """Dispatch → B2 state, then resolve the L2 and re-pend via resume."""
        store: _StoreScheduler = h.scheduler
        task = _gate_task(task_id=task_id, title='Proceed gate')
        store.seed(task)

        # First dispatch → BLOCKED + L2 filed
        await _dispatch(h, task_id)

        # Resolve the L2 so it leaves the pending set
        queue: EscalationQueue = h._escalation_queue
        pending = queue.get_by_task(task_id, status='pending')
        assert pending, 'Pre-condition: must have a pending L2 after first dispatch'
        queue.resolve(pending[0].id, resolution='proceed')

        # Re-pend via the production resume path (stamps preserved — NOT cleared)
        await h._action_teardown_and_set_status(task_id, 'pending', 'resume')

    async def test_resume_dispatch_returns_done(self, det_harness: Harness) -> None:
        """Re-dispatch after resolve → report.outcome == DONE."""
        await self._reach_resolved_state(det_harness, 'gate-b4')

        report = await _dispatch(det_harness, 'gate-b4')

        assert report is not None
        assert report.outcome == WorkflowOutcome.DONE

    async def test_store_status_done_after_resume(self, det_harness: Harness) -> None:
        """Store status is 'done' after the resume re-dispatch."""
        await self._reach_resolved_state(det_harness, 'gate-b4')
        await _dispatch(det_harness, 'gate-b4')

        store: _StoreScheduler = det_harness.scheduler
        assert await store.get_status('gate-b4') == 'done'

    async def test_no_new_escalation_after_resume(self, det_harness: Harness) -> None:
        """No new escalation is filed on the resume re-dispatch (gate resolved)."""
        await self._reach_resolved_state(det_harness, 'gate-b4')
        await _dispatch(det_harness, 'gate-b4')

        queue: EscalationQueue = det_harness._escalation_queue
        # The original L2 is archived (resolved); no new pending L2 should appear
        assert queue.get_by_task('gate-b4', status='pending') == []

    async def test_dependent_eligible_after_gate_done(
        self, det_harness: Harness, tmp_path: Path
    ) -> None:
        """A dependent task is _deps_satisfied when the gate is done (B4 follow-on)."""
        from orchestrator.config import OrchestratorConfig
        config = OrchestratorConfig(project_root=tmp_path)
        scheduler = Scheduler(config)

        await self._reach_resolved_state(det_harness, 'gate-b4')
        await _dispatch(det_harness, 'gate-b4')

        dep_task = {
            'id': 'worker-b4',
            'title': 'Worker depending on gate',
            'metadata': {'task_kind': 'normal'},
            'dependencies': [{'id': 'gate-b4'}],
        }
        status_map = {'gate-b4': 'done'}
        assert scheduler._deps_satisfied(dep_task, status_map) is True


# ---------------------------------------------------------------------------
# B5 — no-go re-pend with new deps: restart + re-fire (step-6)
# ---------------------------------------------------------------------------

@pytest.mark.asyncio
class TestB5NoGo:
    """B5: no-go restart clears stamps, re-deps the gate, re-fires when new task lands.

    Production path (cross-phase no-go):
    1. Dispatch → BLOCKED (B2 state).
    2. Resolve L2 + add new design task N as dep + action='restart'
       → gate_escalated_at cleared, gate re-pended.
    3. With N pending: _deps_satisfied False (gate not re-dispatched).
    4. Land N (done): _deps_satisfied True + re-dispatch → fresh L2 + BLOCKED (re-fires).

    Design decision (plan §5): uses 'restart' (clears stamps) not 'resume' (preserves stamps),
    because the re-depended gate must RE-ESCALATE on next dispatch, not drive to done.
    """

    async def _setup_b5_restart(
        self, h: Harness, gate_id: str, n_id: str,
    ) -> None:
        """Reach B2 blocked state, then resolve + restart to clear stamps + add dep."""
        store: _StoreScheduler = h.scheduler
        task = _gate_task(task_id=gate_id, title='No-go gate')
        store.seed(task)

        # First dispatch → BLOCKED + L2 filed
        await _dispatch(h, gate_id)

        # Resolve the L2 (operator decides: no-go, re-examine)
        queue: EscalationQueue = h._escalation_queue
        pending = queue.get_by_task(gate_id, status='pending')
        assert pending, 'Pre-condition: pending L2 must exist after first dispatch'
        queue.resolve(pending[0].id, resolution='no-go')

        # Add the new design task N into the store
        n_task = {
            'id': n_id,
            'title': 'New design task',
            'metadata': {'task_kind': 'normal'},
            'status': 'pending',
        }
        store.seed(n_task)

        # Add N as a dependency of the gate (operator updates deps during review)
        gate_in_store = await store.get_task(gate_id)
        assert gate_in_store is not None
        gate_in_store.setdefault('dependencies', []).append({'id': str(n_id)})

        # Restart: clears gate_escalated_at + sets gate status to 'pending'
        await h._action_teardown_and_set_status(gate_id, 'pending', 'restart')

    async def test_restart_clears_gate_escalated_at(self, det_harness: Harness) -> None:
        """gate_escalated_at is cleared after restart (stamp-clear mechanism, I2/B5)."""
        await self._setup_b5_restart(det_harness, 'gate-b5', 'N-1')

        store: _StoreScheduler = det_harness.scheduler
        gate = await store.get_task('gate-b5')
        assert gate is not None
        assert gate['metadata'].get('gate_escalated_at') is None

    async def test_gate_repended_after_restart(self, det_harness: Harness) -> None:
        """Gate status is 'pending' after restart (line survives — not done/cancelled)."""
        await self._setup_b5_restart(det_harness, 'gate-b5', 'N-1')
        store: _StoreScheduler = det_harness.scheduler
        assert await store.get_status('gate-b5') == 'pending'

    async def test_original_l2_no_longer_pending_after_restart(
        self, det_harness: Harness
    ) -> None:
        """The original L2 is resolved before restart — no pending escalation remains."""
        await self._setup_b5_restart(det_harness, 'gate-b5', 'N-1')
        queue: EscalationQueue = det_harness._escalation_queue
        assert queue.get_by_task('gate-b5', status='pending') == []

    async def test_deps_not_satisfied_while_n_pending(
        self, det_harness: Harness, tmp_path: Path
    ) -> None:
        """Scheduler._deps_satisfied(gate, {N:'pending'}) is False (re-gated)."""
        from orchestrator.config import OrchestratorConfig
        config = OrchestratorConfig(project_root=tmp_path)
        scheduler = Scheduler(config)
        await self._setup_b5_restart(det_harness, 'gate-b5', 'N-1')

        store: _StoreScheduler = det_harness.scheduler
        gate = await store.get_task('gate-b5')
        assert gate is not None
        assert scheduler._deps_satisfied(gate, {'N-1': 'pending'}) is False

    async def test_re_escalates_when_n_lands(
        self, det_harness: Harness, tmp_path: Path
    ) -> None:
        """When N lands (done), re-dispatch RE-ESCALATES (fresh L2) and status is blocked."""
        from orchestrator.config import OrchestratorConfig
        config = OrchestratorConfig(project_root=tmp_path)
        scheduler = Scheduler(config)
        await self._setup_b5_restart(det_harness, 'gate-b5', 'N-1')

        store: _StoreScheduler = det_harness.scheduler
        # Land N
        await store.set_task_status('N-1', 'done')
        gate = await store.get_task('gate-b5')
        assert gate is not None
        assert scheduler._deps_satisfied(gate, {'N-1': 'done'}) is True

        # Re-dispatch: gate_escalated_at cleared + N done → runs section 3 → new L2
        report = await _dispatch(det_harness, 'gate-b5')
        assert report is not None
        assert report.outcome == WorkflowOutcome.BLOCKED

        queue: EscalationQueue = det_harness._escalation_queue
        pending = queue.get_by_task('gate-b5', status='pending')
        assert len(pending) == 1, (
            f'Expected fresh L2 after re-dispatch, got {len(pending)} pending'
        )
        assert pending[0].level == 2
        assert await store.get_status('gate-b5') == 'blocked'


# ---------------------------------------------------------------------------
# Deploy-row scaffolding (step-8)
# ---------------------------------------------------------------------------

def _deploy_task(
    task_id: str = 'deploy-200',
    target_unit: str = 'orchestrator-reify.service',
    script: str = '/tmp/test-deploy.sh',
    args: list | None = None,
    env: dict | None = None,
    cwd: str = '/tmp',
    timeout_secs: int = 30,
    before_done_ran_at: str | None = None,
    before_done_verified_at: str | None = None,
    before_done_verified_pid: int | None = None,
    before_done_scheduled_at: dict | None = None,
) -> dict:
    """Build a deterministic deploy task dict (before_done set, always_escalates=False).

    Adapted from orchestrator/tests/test_deterministic_runner.py: _deploy_task.
    """
    before_done: dict = {
        'script': script,
        'args': args if args is not None else [],
        'env': env if env is not None else {},
        'cwd': cwd,
        'timeout_secs': timeout_secs,
        'target_unit': target_unit,
    }
    metadata: dict = {
        'task_kind': 'deterministic',
        'always_escalates': False,
        'before_done': before_done,
    }
    if before_done_ran_at is not None:
        metadata['before_done_ran_at'] = before_done_ran_at
    if before_done_verified_at is not None:
        metadata['before_done_verified_at'] = before_done_verified_at
    if before_done_verified_pid is not None:
        metadata['before_done_verified_pid'] = before_done_verified_pid
    if before_done_scheduled_at is not None:
        metadata['before_done_scheduled_at'] = before_done_scheduled_at
    return {
        'id': task_id,
        'title': f'Deploy {target_unit}',
        'description': f'Cross-unit deploy of {target_unit}',
        'status': 'pending',
        'metadata': metadata,
    }


# Unit states for B6/B7 deploy tests (adapted from test_deterministic_runner.py)
_BASELINE_UNIT_STATE: dict = {
    'MainPID': 100,
    'ActiveState': 'active',
    'ActiveEnterTimestamp': 'Mon 2026-06-23 10:00:00 UTC',
    'ActiveEnterTimestampMonotonic': 1_000_000,
}
_FRESH_UNIT_STATE: dict = {
    'MainPID': 200,
    'ActiveState': 'active',
    'ActiveEnterTimestamp': 'Mon 2026-06-23 10:01:00 UTC',
    'ActiveEnterTimestampMonotonic': 2_000_000,
}


def _build_runner(
    store: _StoreScheduler,
    queue: EscalationQueue,
    *,
    unit_inspector=None,
    script_runner=None,
    own_unit_resolver=None,
    restart_scheduler=None,
) -> DeterministicRunner:
    """Build a real DeterministicRunner with injected systemd seams.

    Uses the real DeterministicRunner and a real file-backed EscalationQueue;
    the only fakes are the systemd/script edges (unit_inspector, script_runner,
    own_unit_resolver, restart_scheduler).  Missing seams default to None
    (runner uses its built-in default, which hits real systemd — always pass
    seams for cross-unit and self-restart tests).

    Adapted from the injection pattern in orchestrator/tests/test_deterministic_runner.py.
    """
    return DeterministicRunner(
        scheduler=store,
        escalation_queue=queue,
        unit_inspector=unit_inspector,
        script_runner=script_runner,
        own_unit_resolver=own_unit_resolver,
        restart_scheduler=restart_scheduler,
    )


class TestDeployScaffoldSmoke:
    """Smoke: _build_runner returns a DeterministicRunner with seams wired."""

    def test_build_runner_returns_deterministic_runner(self, tmp_path: Path) -> None:
        """_build_runner returns a DeterministicRunner instance."""
        store = _StoreScheduler()
        queue = EscalationQueue(tmp_path / 'queue')
        script_runner = AsyncMock(name='script_runner')
        runner = _build_runner(store, queue, script_runner=script_runner)
        assert isinstance(runner, DeterministicRunner)
        assert runner._script_runner is script_runner

    def test_build_runner_seams_wired(self, tmp_path: Path) -> None:
        """All injected seams are accessible on the runner."""
        store = _StoreScheduler()
        queue = EscalationQueue(tmp_path / 'q2')
        ui = AsyncMock(name='unit_inspector')
        sr = AsyncMock(name='script_runner')
        our = lambda: 'my-unit.service'  # noqa: E731
        rs = AsyncMock(name='restart_scheduler')
        runner = _build_runner(store, queue, unit_inspector=ui, script_runner=sr,
                               own_unit_resolver=our, restart_scheduler=rs)
        assert runner._unit_inspector is ui
        assert runner._script_runner is sr
        assert runner._own_unit_resolver is our
        assert runner._restart_scheduler is rs


# ---------------------------------------------------------------------------
# B11 — restart-window replay: stamps/L2 survive orchestrator restart (step-7)
# ---------------------------------------------------------------------------

@pytest.mark.asyncio
class TestB11RestartReplay:
    """B11: after an orchestrator restart, a blocked gate stays quiescent.

    Two variants:
    (a) Pure gate — reach B2 blocked state, build a FRESH Harness over the
        SAME queue_dir and SAME _StoreScheduler, re-dispatch.  Runner section 1
        sees gate_escalated_at + pending L2 → BLOCKED without re-escalating.
    (b) Deploy — a deploy task with before_done_ran_at pre-stamped and a pending
        infra_issue L2 (crash mid-deploy before verify).  On re-dispatch via a
        fresh DeterministicRunner with injected seams, runner section 2 sees
        before_done_ran_at + pending L2 → BLOCKED (I1: script_runner not called).
    """

    async def test_pure_gate_restart_quiescent(
        self, tmp_path: Path, mock_orch_config,
    ) -> None:
        """Pure gate: restart with same store+queue → quiescent BLOCKED, 1 pending L2."""
        store = _StoreScheduler()
        queue_dir = tmp_path / 'queue'
        h1 = _build_harness(mock_orch_config, queue_dir, store)

        task = _gate_task(task_id='gate-b11', title='B11 restart gate')
        store.seed(task)
        report1 = await _dispatch(h1, 'gate-b11')
        assert report1.outcome == WorkflowOutcome.BLOCKED

        # Verify B2 preconditions
        gate = await store.get_task('gate-b11')
        assert gate is not None
        assert gate['metadata'].get('gate_escalated_at') is not None
        q = EscalationQueue(queue_dir)
        assert len(q.get_by_task('gate-b11', status='pending')) == 1

        # Simulate orchestrator restart: fresh Harness, SAME queue_dir + SAME store
        h2 = _build_harness(mock_orch_config, queue_dir, store)
        report2 = await _dispatch(h2, 'gate-b11')

        # Section 1 quiescence: gate_escalated_at set + pending L2 → BLOCKED
        assert report2.outcome == WorkflowOutcome.BLOCKED
        assert report2.block_reason == 'deterministic_gate'

        # gate_escalated_at STILL persisted (not cleared by quiescence)
        gate2 = await store.get_task('gate-b11')
        assert gate2 is not None
        assert gate2['metadata'].get('gate_escalated_at') is not None

        # STILL exactly one pending L2 (no second escalation filed)
        q2 = EscalationQueue(queue_dir)
        pending = q2.get_by_task('gate-b11', status='pending')
        assert len(pending) == 1

    async def test_deploy_restart_no_rerun_quiescent(
        self, tmp_path: Path,
    ) -> None:
        """Deploy (I1): restart with before_done_ran_at + pending L2 → no rerun, BLOCKED.

        Simulates a crash mid-deploy: before_done_ran_at stamped but neither
        before_done_verified_at nor a resolved escalation exists.  A pending
        infra_issue L2 was left in the queue.  On rehydrate re-dispatch the
        script_runner must NOT be called a second time (I1 once-only).
        """
        task_id = 'deploy-b11'
        store = _StoreScheduler()
        queue_dir = tmp_path / 'queue2'
        before_done = {
            'script': '/tmp/deploy.sh',
            'args': [],
            'env': {},
            'cwd': '/tmp',
            'timeout_secs': 30,
            'target_unit': 'orchestrator-reify.service',
        }
        task = {
            'id': task_id,
            'title': 'B11 deploy restart',
            'description': 'Deploy before restart test',
            'status': 'blocked',
            'metadata': {
                'task_kind': 'deterministic',
                'always_escalates': False,
                'before_done': before_done,
                'before_done_ran_at': '2026-01-01T00:00:00+00:00',
            },
        }
        store.seed(task)

        # Submit a pending infra_issue L2 (left by the prior (crashed) dispatch)
        queue = EscalationQueue(queue_dir)
        esc = Escalation(
            id=queue.make_id(task_id),
            task_id=task_id,
            agent_role='orchestrator-deterministic',
            severity='critical',
            category='infra_issue',
            summary='Deploy failed: orchestrator-reify.service',
            level=2,
        )
        queue.submit(esc)
        assert len(queue.get_by_task(task_id, status='pending')) == 1

        # Build runner directly with injected seams (no real systemd)
        script_runner = AsyncMock(name='script_runner')
        task_dict = await store.get_task(task_id)
        assignment = _make_assignment(task_dict)
        runner = DeterministicRunner(
            scheduler=store,
            escalation_queue=queue,
            script_runner=script_runner,
        )

        outcome = await runner.run(assignment)

        # I1: script_runner must NOT have been called (before_done_ran_at stamped + pending L2)
        script_runner.assert_not_awaited()
        assert outcome == WorkflowOutcome.BLOCKED

        # Queue STILL has exactly ONE pending L2 (no second escalation filed)
        remaining = queue.get_by_task(task_id, status='pending')
        assert len(remaining) == 1
