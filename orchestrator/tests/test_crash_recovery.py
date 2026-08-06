"""Tests for crash recovery — surviving worktree detection and plan injection."""

import json
import logging
import os
import shutil
from datetime import UTC, datetime, timedelta
from pathlib import Path
from unittest.mock import AsyncMock, MagicMock, patch

import pytest

from orchestrator.artifacts import TaskArtifacts
from orchestrator.config import SessionResumeConfig, TranscriptArchiveConfig
from orchestrator.event_store import EventType
from orchestrator.harness import Harness
from orchestrator.lane_lifecycle import LaneLifecycle
from orchestrator.lane_lifecycle import LaneState as DurableLaneState
from orchestrator.verify_cancel import (
    acquire_merge_verify_flock,
    lane_lock_path,
    release_merge_verify_flock,
    remove_lock_holder_pgid,
    write_lock_holder_pgid,
)
from orchestrator.warm_lane_pool import LaneState, WarmLanePool


@pytest.fixture
def harness(tmp_path: Path, mock_orch_config):
    """Create a Harness with mocked internals for unit testing recovery."""
    with patch('orchestrator.harness.McpLifecycle'), \
         patch('orchestrator.harness.Scheduler'), \
         patch('orchestrator.harness.BriefingAssembler'):
        h = Harness(mock_orch_config)

    # Replace scheduler with async mocks
    h.scheduler = MagicMock()
    h.scheduler.get_tasks = AsyncMock(return_value=[])
    h.scheduler.set_task_status = AsyncMock()
    # Fix C identity guard: get_task feeds the live title.  Default returns a
    # title-less dict ({} is non-None → no defer; no title → identities_match
    # fails open → adopt), so the pre-Fix-C recovery tests behave unchanged.
    h.scheduler.get_task = AsyncMock(return_value={})
    # T10 amplifier (task 1881): get_status is awaited inside the warm-lane
    # recovery branch.  Default None → "transient/None → fall through to
    # restore" safe path (harness.py:1718-1719); all warm-lane RED tests assert
    # restore/preserve, none assert release.
    h.scheduler.get_status = AsyncMock(return_value=None)
    h.scheduler._dispatched = set()
    # Substrate gate: _run_slot now calls substrate_gate.carries_substrate_probe
    # (module-level, not a Scheduler method — task 2121) directly on
    # assignment.task. This file's task dicts carry no 'metadata' key, so the
    # real predicate already returns False and the D4 gate is skipped without
    # needing to stub anything on the mocked scheduler.
    # Deterministic dispatch (task 1899): is_deterministic is a sync @staticmethod
    # predicate checked at top of _run_slot (harness.py:3728).  Stub False so the
    # 4 _run_slot tests skip _run_deterministic_slot.
    h.scheduler.is_deterministic = MagicMock(return_value=False)

    # Replace git_ops cleanup/quarantine with async mocks; keep worktree_base real
    h.git_ops.worktree_base = (tmp_path / '.worktrees').resolve()
    # Task 2099: mark pool storage present by default so the new
    # mount-presence guard on _recover_crashed_tasks does not false-trip
    # across this file's recovery-routing tests, which all assume an
    # already-mounted host — an orthogonal concern to plan recovery/cleanup
    # routing. The dedicated storage-absent tests remove the sentinel
    # explicitly to exercise the guard itself.
    h.git_ops.mark_pool_storage_present()
    h.git_ops.cleanup_worktree = AsyncMock()
    h.git_ops.quarantine_worktree = AsyncMock(return_value=None)
    # W11 delta: GitOps.__init__ built _lane_lifecycle against the ORIGINAL
    # worktree_base (before the reassignment above) — rebind it so the
    # record-driven recovery path reads/writes the same .lane-state dir
    # every other test helper here (_attach_pool, _setup_lane) targets.
    h.git_ops._lane_lifecycle = LaneLifecycle(
        h.git_ops.worktree_base, quarantine_worktree=h.git_ops.quarantine_worktree,
    )
    # Registration guard (reify 4655/4947): default to "still registered" so
    # the existing warm-lane restore-path tests (fabricated via mkdir, never
    # `git worktree add`ed) keep exercising the positive (non-terminal +
    # registered -> restore) path. Tests for the orphaned-lane invariant
    # override this to False.
    h.git_ops._is_registered_worktree = AsyncMock(return_value=True)
    # Exercise the (best-effort) event emits without a real store.
    h.event_store = MagicMock()

    return h


def _make_plan(
    steps_done: int,
    steps_total: int,
    task_id: str = 'test',
    *,
    session_id: str | None = None,
) -> dict:
    """Build a plan dict with the given step completion counts.

    When ``session_id`` is provided, the plan is provenance-stamped (mirrors
    artifacts.stamp_plan_provenance), which signals the recovery path that
    the architect already produced this plan and the worktree should be
    preserved for revalidation rather than wiped.
    """
    steps = []
    for i in range(steps_total):
        steps.append({
            'id': f'step-{i + 1}',
            'description': f'Step {i + 1}',
            'status': 'done' if i < steps_done else 'pending',
            'commit': f'abc{i}' if i < steps_done else None,
        })
    plan: dict = {
        'task_id': task_id,
        'title': 'Test Task',
        'steps': steps,
    }
    if session_id is not None:
        plan['_session_id'] = session_id
    return plan


def _setup_worktree(base: Path, task_id: str, plan: dict | None = None):
    """Create a fake worktree directory, optionally with a plan."""
    wt = base / task_id
    wt.mkdir(parents=True, exist_ok=True)
    if plan is not None:
        task_dir = wt / '.task'
        task_dir.mkdir(exist_ok=True)
        (task_dir / 'plan.json').write_text(json.dumps(plan))
    return wt


# ── Session-resume γ guard helpers (task 2774) ───────────────────────────────
def _make_transcript(base: Path, session_id: str) -> Path:
    """Create a real ``<cfg>/projects/<slug>/<session_id>.jsonl`` transcript and
    return the ``<cfg>`` claude-config dir path.

    Mirrors the on-disk layout that ``transcript_exists(config_dir,
    session_id)`` globs (``<config_dir>/projects/*/<session_id>.jsonl``), so a
    stashed ``_recovered_session_config_dirs`` entry pointing at the returned
    dir corroborates the session as eligible.
    """
    cfg = base / f'claude-config-{session_id}'
    proj = cfg / 'projects' / 'some-slug'
    proj.mkdir(parents=True, exist_ok=True)
    (proj / f'{session_id}.jsonl').write_text('{"type": "summary"}\n')
    return cfg


def _session_resume_emits(harness: Harness) -> list[tuple]:
    """Return ``[(event_type, kwargs), ...]`` for every session_resume* emit.

    Referencing the new EventType members lives here (call-time), never at
    module scope, so a missing member in the RED phase fails only these tests
    rather than breaking collection of the whole module.
    """
    wanted = {
        EventType.session_resume,
        EventType.session_resume_fallback,
        EventType.session_resume_capped,
    }
    out: list[tuple] = []
    for call in harness.event_store.emit.call_args_list:  # type: ignore[attr-defined]
        if call.args and call.args[0] in wanted:
            out.append((call.args[0], call.kwargs))
    return out


async def _drive_session_slot(
    harness: Harness,
    task_id: str,
    session: dict,
    *,
    config_dir: Path | str | None = None,
):
    """Populate recovered-session state and run ``_run_slot`` with
    ``build_workflow`` patched; return the ``resume_session_id`` kwarg it saw.
    """
    harness._recovered_sessions[task_id] = session
    if config_dir is not None:
        harness._recovered_session_config_dirs[task_id] = str(config_dir)

    assignment = MagicMock()
    assignment.task_id = task_id
    assignment.task = {'title': f'task {task_id}'}
    sem = MagicMock()
    sem.release = MagicMock()

    with patch('orchestrator.harness.build_workflow') as MockWorkflow:
        mock_wf = AsyncMock()
        mock_wf.run.return_value = MagicMock(value='done')
        mock_wf.metrics = MagicMock(
            total_cost_usd=0.0, total_duration_ms=0, agent_invocations=0,
        )
        MockWorkflow.return_value = mock_wf
        await harness._run_slot(assignment, sem)
        return MockWorkflow.call_args.kwargs['resume_session_id']


@pytest.mark.asyncio
class TestRecoverCrashedTasks:
    async def test_recover_worktree_with_completed_steps(self, harness: Harness):
        """Worktree with plan (3/5 steps done) -> plan stored in _recovered_plans."""
        plan = _make_plan(steps_done=3, steps_total=5, task_id='35')
        _setup_worktree(harness.git_ops.worktree_base, '35', plan)

        await harness._recover_crashed_tasks()

        assert '35' in harness._recovered_plans
        recovered = harness._recovered_plans['35']
        assert len(recovered['steps']) == 5
        done = [s for s in recovered['steps'] if s['status'] == 'done']
        assert len(done) == 3
        harness.git_ops.cleanup_worktree.assert_not_called()  # type: ignore[attr-defined]

    async def test_recover_cold_worktree_plan_adopts_v2_sidecar_session(
        self, harness: Harness,
    ):
        """Task 2772 (session-resume beta): cold worktree with plan (3/5
        done) AND a co-located v2 agent_session.json sidecar -> the
        heuristic plan-present site (~2943) must ALSO populate
        _recovered_sessions[task_id], not just _recovered_plans, so the
        already-wired _run_slot injection can --resume the prior session.
        """
        plan = _make_plan(steps_done=3, steps_total=5, task_id='35')
        wt = _setup_worktree(harness.git_ops.worktree_base, '35', plan)
        sidecar = {
            'session_id': 'uuid-cold',
            'role': 'implementer',
            'started_at': '2026-07-19T09:00:00+00:00',
            'owner_pid': 4242,
            'task_id': '35',
            'resume_count': 0,
            'schema_version': 2,
        }
        (wt / '.task' / 'agent_session.json').write_text(json.dumps(sidecar))

        await harness._recover_crashed_tasks()

        assert '35' in harness._recovered_plans
        assert '35' in harness._recovered_sessions
        assert harness._recovered_sessions['35']['session_id'] == 'uuid-cold'
        assert harness._recovered_sessions['35']['role'] == 'implementer'
        harness.git_ops.cleanup_worktree.assert_not_called()  # type: ignore[attr-defined]

    async def test_recover_planless_worktree_cleaned_up(self, harness: Harness):
        """Worktree with no .task/ dir -> cleaned up."""
        wt = _setup_worktree(harness.git_ops.worktree_base, '36')

        await harness._recover_crashed_tasks()

        assert '36' not in harness._recovered_plans
        harness.git_ops.cleanup_worktree.assert_called_once_with(wt, '36')  # type: ignore[attr-defined]

    async def test_recover_plan_no_progress_cleaned_up(self, harness: Harness):
        """Unstamped plan with all steps pending -> cleaned up.

        The predicate is ``_session_id`` presence rather than step-count alone:
        an unstamped plan represents a half-written architect output (the
        stamp is applied AFTER successful create_plan), so there is nothing
        worth preserving.
        """
        plan = _make_plan(steps_done=0, steps_total=4)
        # Predicate-shape lock: this scenario must hit the "unstamped" branch.
        assert '_session_id' not in plan, (
            '_make_plan default must produce an unstamped plan'
        )
        wt = _setup_worktree(harness.git_ops.worktree_base, '37', plan)

        await harness._recover_crashed_tasks()

        assert '37' not in harness._recovered_plans
        assert '37' not in harness._preserved_worktrees
        harness.git_ops.cleanup_worktree.assert_called_once_with(wt, '37')  # type: ignore[attr-defined]

    async def test_recover_stamped_no_done_preserved(self, harness: Harness):
        """Stamped plan with 0 done steps -> worktree kept, lock cleared,
        added to _preserved_worktrees but NOT _recovered_plans.

        Stamped pre-EXECUTE plans usually arrive here via the blast-radius
        lock-conflict requeue: architect ran, plan was stamped, scheduler
        rejected the expanded module set, task was re-pended.  Wiping the
        worktree wastes the architect call; preserving it lets the next
        acquisition take the revalidation branch in _plan().
        """
        plan = _make_plan(
            steps_done=0, steps_total=4, task_id='38',
            session_id='38-deadbeefcafe',
        )
        wt = _setup_worktree(harness.git_ops.worktree_base, '38', plan)
        # Seed a stale plan.lock to verify it's unlinked on preservation.
        lock_path = wt / '.task' / 'plan.lock'
        lock_path.write_text(json.dumps({'session_id': 'old', 'owner_pid': 1}))

        await harness._recover_crashed_tasks()

        # Worktree dir survives — cleanup_worktree must NOT be called.
        harness.git_ops.cleanup_worktree.assert_not_called()  # type: ignore[attr-defined]
        assert wt.exists()
        # NOT pre-loaded into _recovered_plans (we want _plan() to revalidate).
        assert '38' not in harness._recovered_plans
        # Marked preserved so _reconcile_stranded_in_progress won't wipe it.
        assert '38' in harness._preserved_worktrees
        # Stale lock cleared.
        assert not lock_path.exists()

    async def test_recover_corrupt_plan_cleaned_up(self, harness: Harness):
        """Invalid JSON in plan.json -> cleaned up with warning."""
        wt = harness.git_ops.worktree_base / '38'
        wt.mkdir(parents=True)
        task_dir = wt / '.task'
        task_dir.mkdir()
        (task_dir / 'plan.json').write_text('{not valid json!!!')

        await harness._recover_crashed_tasks()

        assert '38' not in harness._recovered_plans
        harness.git_ops.cleanup_worktree.assert_called_once_with(wt, '38')  # type: ignore[attr-defined]

    async def test_recover_no_worktrees_dir_noop(self, harness: Harness):
        """Worktree base doesn't exist -> no-op, no errors."""
        # The fixture's mark_pool_storage_present() call creates worktree_base
        # as a side effect (task 2099) — remove it again so this test still
        # exercises the pre-existing "base missing entirely" guard, distinct
        # from the pool-storage-absent guard (base exists, sentinel absent)
        # covered by the dedicated storage-absent tests.
        shutil.rmtree(harness.git_ops.worktree_base)
        assert not harness.git_ops.worktree_base.exists()

        await harness._recover_crashed_tasks()

        assert harness._recovered_plans == {}
        harness.git_ops.cleanup_worktree.assert_not_called()  # type: ignore[attr-defined]

    async def test_in_progress_tasks_left_for_reconcile_sweep(self, harness: Harness):
        """_recover_crashed_tasks does NOT reset in-progress tasks to pending.

        Status reconciliation for stranded in-progress tasks is handled by the
        separate _reconcile_stranded_in_progress() sweep that runs immediately
        after this method in Harness.run().
        """
        harness.git_ops.worktree_base.mkdir(parents=True, exist_ok=True)
        harness.scheduler.get_tasks.return_value = [  # type: ignore[attr-defined]
            {'id': 10, 'status': 'in-progress', 'title': 'Stuck task'},
            {'id': 11, 'status': 'pending', 'title': 'Normal task'},
            {'id': 12, 'status': 'done', 'title': 'Done task'},
            {'id': 13, 'status': 'in-progress', 'title': 'Another stuck'},
        ]

        await harness._recover_crashed_tasks()

        # set_task_status must NOT be called — status reconciliation is
        # delegated to _reconcile_stranded_in_progress.
        harness.scheduler.set_task_status.assert_not_called()  # type: ignore[attr-defined]

    async def test_recovered_plan_injected_in_run_slot(self, harness: Harness):
        """Plan consumed from _recovered_plans and passed as initial_plan."""
        plan = _make_plan(steps_done=3, steps_total=5)
        harness._recovered_plans['42'] = plan

        assignment = MagicMock()
        assignment.task_id = '42'
        assignment.task = {'title': 'Recovered task'}

        sem = MagicMock()
        sem.release = MagicMock()

        with patch('orchestrator.harness.build_workflow') as MockWorkflow:
            mock_wf = AsyncMock()
            mock_wf.run.return_value = MagicMock(value='done')
            mock_wf.metrics = MagicMock(
                total_cost_usd=0.0,
                total_duration_ms=0,
                agent_invocations=0,
            )
            MockWorkflow.return_value = mock_wf

            await harness._run_slot(assignment, sem)

            # Verify TaskWorkflow was created with the recovered plan
            call_kwargs = MockWorkflow.call_args.kwargs
            assert call_kwargs['initial_plan'] is plan

        # Plan should be consumed (popped)
        assert '42' not in harness._recovered_plans

    async def test_no_injection_without_recovered_plan(self, harness: Harness):
        """Without a recovered plan, initial_plan should be None."""
        assignment = MagicMock()
        assignment.task_id = '99'
        assignment.task = {'title': 'Fresh task'}

        sem = MagicMock()
        sem.release = MagicMock()

        with patch('orchestrator.harness.build_workflow') as MockWorkflow:
            mock_wf = AsyncMock()
            mock_wf.run.return_value = MagicMock(value='done')
            mock_wf.metrics = MagicMock(
                total_cost_usd=0.0,
                total_duration_ms=0,
                agent_invocations=0,
            )
            MockWorkflow.return_value = mock_wf

            await harness._run_slot(assignment, sem)

            call_kwargs = MockWorkflow.call_args.kwargs
            assert call_kwargs['initial_plan'] is None

    async def test_run_slot_clears_preserved_marker(self, harness: Harness):
        """When the slot picks up a preserved-worktree task, the marker must
        be discarded so a subsequent reconcile sweep doesn't see it as still
        stranded."""
        harness._preserved_worktrees.add('77')

        assignment = MagicMock()
        assignment.task_id = '77'
        assignment.task = {'title': 'Preserved task'}

        sem = MagicMock()
        sem.release = MagicMock()

        with patch('orchestrator.harness.build_workflow') as MockWorkflow:
            mock_wf = AsyncMock()
            mock_wf.run.return_value = MagicMock(value='done')
            mock_wf.metrics = MagicMock(
                total_cost_usd=0.0,
                total_duration_ms=0,
                agent_invocations=0,
            )
            MockWorkflow.return_value = mock_wf

            await harness._run_slot(assignment, sem)

        # Marker cleared — _plan() will reuse the worktree on its own and
        # the next reconcile sweep should not see the task as preserved.
        assert '77' not in harness._preserved_worktrees

    async def test_recover_plan_task_id_mismatch_cleaned_up(self, harness: Harness):
        """Plan whose task_id doesn't match the worktree dir -> cleaned up."""
        plan = _make_plan(steps_done=3, steps_total=5, task_id='216')
        wt = _setup_worktree(harness.git_ops.worktree_base, '369', plan)

        await harness._recover_crashed_tasks()

        assert '369' not in harness._recovered_plans
        harness.git_ops.cleanup_worktree.assert_called_once_with(wt, '369')  # type: ignore[attr-defined]

    async def test_recover_sidecar_no_plan_preserved(self, harness: Harness):
        """Worktree with agent_session.json sidecar but NO plan.json — preserved
        for resume, session info recorded, worktree NOT cleaned up."""
        wt = harness.git_ops.worktree_base / '88'
        task_dir = wt / '.task'
        task_dir.mkdir(parents=True)
        sidecar = {
            'session_id': 'uuid-mid-flight',
            'role': 'architect',
            'started_at': '2026-05-12T10:00:00+00:00',
            'owner_pid': 4242,
        }
        (task_dir / 'agent_session.json').write_text(json.dumps(sidecar))

        await harness._recover_crashed_tasks()

        # Worktree survives — no cleanup
        harness.git_ops.cleanup_worktree.assert_not_called()  # type: ignore[attr-defined]
        assert wt.exists()
        # Session captured for the next slot
        assert '88' in harness._recovered_sessions
        assert harness._recovered_sessions['88']['session_id'] == 'uuid-mid-flight'
        assert harness._recovered_sessions['88']['role'] == 'architect'
        # Preserved so the stranded sweep doesn't wipe it
        assert '88' in harness._preserved_worktrees
        # No plan was recovered (the architect never wrote one)
        assert '88' not in harness._recovered_plans

    async def test_recover_corrupt_sidecar_falls_back_to_cleanup(self, harness: Harness):
        """Unreadable sidecar -> log warning and clean up like a planless worktree."""
        wt = harness.git_ops.worktree_base / '89'
        task_dir = wt / '.task'
        task_dir.mkdir(parents=True)
        (task_dir / 'agent_session.json').write_text('{not json')

        await harness._recover_crashed_tasks()

        assert '89' not in harness._recovered_sessions
        assert '89' not in harness._preserved_worktrees
        harness.git_ops.cleanup_worktree.assert_called_once_with(wt, '89')  # type: ignore[attr-defined]

    async def test_run_slot_passes_recovered_session(
        self, harness: Harness, tmp_path: Path
    ):
        """An ELIGIBLE recovered session flows through to TaskWorkflow as
        resume_session_id (γ guard keeps it: fresh + under-cap + transcript on
        disk). Updated for task 2774 — the pre-γ setup (stale started_at, no
        transcript) is now ineligible, so the guard needs a corroborated
        session for this assertion to hold.
        """
        session_dict = {
            'session_id': 'uuid-resume-me',
            'role': 'implementer',
            'started_at': datetime.now(UTC).isoformat(),
            'owner_pid': 9999,
            'resume_count': 0,
        }
        cfg = _make_transcript(tmp_path, 'uuid-resume-me')
        harness.config.session_resume = SessionResumeConfig()
        harness._recovered_sessions['55'] = session_dict
        harness._recovered_session_config_dirs['55'] = str(cfg)
        harness._preserved_worktrees.add('55')

        assignment = MagicMock()
        assignment.task_id = '55'
        assignment.task = {'title': 'Resumable task'}

        sem = MagicMock()
        sem.release = MagicMock()

        with patch('orchestrator.harness.build_workflow') as MockWorkflow:
            mock_wf = AsyncMock()
            mock_wf.run.return_value = MagicMock(value='done')
            mock_wf.metrics = MagicMock(
                total_cost_usd=0.0,
                total_duration_ms=0,
                agent_invocations=0,
            )
            MockWorkflow.return_value = mock_wf

            await harness._run_slot(assignment, sem)

            call_kwargs = MockWorkflow.call_args.kwargs
            assert call_kwargs['resume_session_id'] is session_dict

        # Consumed by the slot
        assert '55' not in harness._recovered_sessions
        assert '55' not in harness._preserved_worktrees

    async def test_recover_stamped_plan_clears_stale_sidecar(self, harness: Harness):
        """A stamped plan path takes precedence; any sidecar is stale and cleared."""
        plan = _make_plan(
            steps_done=0, steps_total=2, task_id='66',
            session_id='66-aaaabbbbcccc',
        )
        wt = _setup_worktree(harness.git_ops.worktree_base, '66', plan)
        sidecar_path = wt / '.task' / 'agent_session.json'
        sidecar_path.write_text(json.dumps({
            'session_id': 'stale-uuid', 'role': 'reviewer',
            'started_at': 'whenever', 'owner_pid': 1,
        }))

        await harness._recover_crashed_tasks()

        # Stamped plan branch wins; sidecar cleared to avoid confusing next slot
        assert '66' in harness._preserved_worktrees
        assert not sidecar_path.exists()
        assert '66' not in harness._recovered_sessions

    async def test_multiple_worktrees_mixed(self, harness: Harness):
        """Multiple worktrees: one recovered, one cleaned, one no-progress."""
        base = harness.git_ops.worktree_base

        # Task with progress — should be recovered
        plan_good = _make_plan(steps_done=2, steps_total=4, task_id='50')
        _setup_worktree(base, '50', plan_good)

        # Task with no plan — should be cleaned
        wt_noplan = _setup_worktree(base, '51')

        # Task with no progress — should be cleaned
        plan_empty = _make_plan(steps_done=0, steps_total=3)
        wt_noprog = _setup_worktree(base, '52', plan_empty)

        await harness._recover_crashed_tasks()

        assert '50' in harness._recovered_plans
        assert '51' not in harness._recovered_plans
        assert '52' not in harness._recovered_plans

        cleanup_calls = harness.git_ops.cleanup_worktree.call_args_list  # type: ignore[attr-defined]
        cleaned_paths = {str(c.args[0]) for c in cleanup_calls}
        assert str(wt_noplan) in cleaned_paths
        assert str(wt_noprog) in cleaned_paths
        assert len(cleanup_calls) == 2


def _setup_worktree_with_meta(base: Path, task_id: str, plan: dict, *, title: str):
    """Worktree with a plan AND a .task/metadata.json carrying ``title``."""
    wt = _setup_worktree(base, task_id, plan)
    (wt / '.task' / 'metadata.json').write_text(
        json.dumps({'task_id': task_id, 'title': title})
    )
    return wt


@pytest.mark.asyncio
class TestRecoverIdentityGuard:
    """Fix C: semantic identity guard on the crash-recovery path.

    The numeric guard only proves ``plan.task_id == dirname``; for a recycled
    id both equal the new task's id.  These tests cover the title comparison
    against the live DB task — the exact check that would have caught reify
    task 3770.
    """

    async def test_quarantines_on_title_mismatch(self, harness: Harness):
        """The 3770 scenario: worktree holds a trajectory plan but the live
        (recycled-id) task is the cycle-breaker → quarantine, do not adopt."""
        plan = _make_plan(steps_done=3, steps_total=5, task_id='3770')
        wt = _setup_worktree_with_meta(
            harness.git_ops.worktree_base, '3770', plan,
            title='Trajectory beta: spline solver',
        )
        harness.scheduler.get_task = AsyncMock(
            return_value={'id': '3770', 'title': 'Cycle-breaker beta: dedup edges'},
        )

        await harness._recover_crashed_tasks()

        assert '3770' not in harness._recovered_plans
        harness.git_ops.quarantine_worktree.assert_called_once_with(  # type: ignore[attr-defined]
            wt, '3770', 'recovery-identity-mismatch',
        )
        emitted = {c.args[0] for c in harness.event_store.emit.call_args_list}  # type: ignore[attr-defined]
        assert EventType.worktree_quarantined in emitted

    async def test_adopts_on_match_with_autoeval_prefix(self, harness: Harness):
        """A benign ``[auto-eval redo] `` prefix normalises away → adopt."""
        plan = _make_plan(steps_done=3, steps_total=5, task_id='40')
        _setup_worktree_with_meta(
            harness.git_ops.worktree_base, '40', plan, title='Fix the widget',
        )
        harness.scheduler.get_task = AsyncMock(
            return_value={'id': '40', 'title': '[auto-eval redo] Fix the widget'},
        )

        await harness._recover_crashed_tasks()

        assert '40' in harness._recovered_plans
        harness.git_ops.quarantine_worktree.assert_not_called()  # type: ignore[attr-defined]

    async def test_defers_when_get_task_none(self, harness: Harness):
        """get_task None (deleted OR transient error) → no adopt, no destroy."""
        plan = _make_plan(steps_done=3, steps_total=5, task_id='41')
        wt = _setup_worktree_with_meta(
            harness.git_ops.worktree_base, '41', plan, title='Whatever',
        )
        harness.scheduler.get_task = AsyncMock(return_value=None)

        await harness._recover_crashed_tasks()

        assert '41' not in harness._recovered_plans
        harness.git_ops.quarantine_worktree.assert_not_called()  # type: ignore[attr-defined]
        harness.git_ops.cleanup_worktree.assert_not_called()  # type: ignore[attr-defined]
        assert wt.exists()  # deferred to the reaper, untouched

    async def test_adopts_when_no_stored_title(self, harness: Harness):
        """No readable stored title → identities_match fails open → adopt."""
        plan = _make_plan(steps_done=3, steps_total=5, task_id='42')
        plan.pop('title', None)
        _setup_worktree(harness.git_ops.worktree_base, '42', plan)  # no metadata.json
        harness.scheduler.get_task = AsyncMock(
            return_value={'id': '42', 'title': 'Some live title'},
        )

        await harness._recover_crashed_tasks()

        assert '42' in harness._recovered_plans
        harness.git_ops.quarantine_worktree.assert_not_called()  # type: ignore[attr-defined]

    async def test_disabled_flag_skips_check(self, harness: Harness):
        """Flag off → the title comparison is skipped entirely (get_task unused)."""
        harness.config.worktree_identity_guard_enabled = False
        plan = _make_plan(steps_done=3, steps_total=5, task_id='43')
        _setup_worktree_with_meta(
            harness.git_ops.worktree_base, '43', plan, title='Mismatch A',
        )
        harness.scheduler.get_task = AsyncMock(
            return_value={'id': '43', 'title': 'Mismatch B'},
        )

        await harness._recover_crashed_tasks()

        assert '43' in harness._recovered_plans  # adopted despite mismatch
        harness.scheduler.get_task.assert_not_called()
        harness.git_ops.quarantine_worktree.assert_not_called()  # type: ignore[attr-defined]


# ===========================================================================
# Step-3 RED: warm-lane recovery — _recover_crashed_tasks with WarmLanePool
# ===========================================================================


def _attach_pool(harness: Harness, size: int = 2) -> WarmLanePool:
    """Attach a WarmLanePool to harness.git_ops.warm_lane_pool.

    The pool must be constructed against the same worktree_base that was
    assigned to harness.git_ops AFTER GitOps construction (test_crash_recovery
    fixture does h.git_ops.worktree_base = (tmp_path / '.worktrees').resolve()
    at line ~32), so is_lane/assignment_for path comparisons match.
    """
    base = harness.git_ops.worktree_base
    base.mkdir(parents=True, exist_ok=True)
    pool = WarmLanePool(worktree_base=base, size=size)
    harness.git_ops.warm_lane_pool = pool
    return pool


def _setup_lane(base: Path, lane_name: str, plan: dict) -> Path:
    """Create a lane dir (e.g. '_lane-0') with the given plan.json."""
    lane = base / lane_name
    task_dir = lane / '.task'
    task_dir.mkdir(parents=True, exist_ok=True)
    (task_dir / 'plan.json').write_text(json.dumps(plan))
    return lane


def _seed_lane_record(
    lifecycle: LaneLifecycle, lane: Path, *, task_id: str, branch: str | None = None,
) -> None:
    """Bring *lane*'s durable record to ASSIGNED:*task_id* via the legal
    seed-up ladder (None -> SEED -> REGISTERED -> ASSIGNED), mirroring
    GitOps._note_assigned_via_route's climb. ``branch=None`` seeds a
    branchless record, which trivially satisfies the recovery path's
    ``rec.branch is None or ...`` branch-match check regardless of what
    ``lane_branch_checkouts()`` reports.
    """
    lifecycle.transition(lane, DurableLaneState.SEED, seeded_from_sha='abc')
    lifecycle.transition(lane, DurableLaneState.REGISTERED, branch=branch)
    lifecycle.transition(
        lane, DurableLaneState.ASSIGNED, task_id=task_id, branch=branch,
    )


def _setup_lane_meta_plan(base: Path, lane_name: str, plan: dict) -> Path:
    """Write plan.json under the NEW `.task-meta` root (W11 beta relocation),
    a SIBLING of the lane dir rather than nested inside it.  Returns the
    `.task-meta/<lane_name>` dir.
    """
    meta_dir = TaskArtifacts.meta_root_for(base, lane_name)
    meta_dir.mkdir(parents=True, exist_ok=True)
    (meta_dir / 'plan.json').write_text(json.dumps(plan))
    return meta_dir


@pytest.mark.asyncio
@pytest.mark.timeout(180)  # task 2376: heavy class, widened from the 60s default to tolerate host oversubscription
class TestRecoverCrashedTasksWarmLane:
    """_recover_crashed_tasks must correctly recover warm-lane worktrees.

    A lane dir is named '_lane-0' but plan.json['task_id'] holds the real
    task id ('42').  The cold numeric-mismatch branch would clean it up
    (plan_task_id='42' != dir_name='_lane-0' → cleanup) and lose the work.
    With the warm-lane path, recovery uses plan.json's task_id as the key.
    """

    async def test_warm_lane_plan_keyed_under_real_task_id(
        self, harness: Harness,
    ):
        """Plan recovered from _lane-0 is stored under '42', not '_lane-0'."""
        _pool = _attach_pool(harness, size=2)
        base = harness.git_ops.worktree_base
        plan = _make_plan(steps_done=3, steps_total=5, task_id='42')
        _setup_lane(base, '_lane-0', plan)

        await harness._recover_crashed_tasks()

        assert '42' in harness._recovered_plans, (
            'Plan must be keyed under real task_id, not lane dir name'
        )
        assert '_lane-0' not in harness._recovered_plans

    async def test_warm_lane_recordless_plan_adopts_session_keyed_by_task_id(
        self, harness: Harness,
    ):
        """Task 2772: record-less warm lane (heuristic path) with plan.json
        (task_id='42') AND a co-located v2 sidecar -> session is adopted
        under the real task id, not the lane dir name.
        """
        _pool = _attach_pool(harness, size=2)
        base = harness.git_ops.worktree_base
        plan = _make_plan(steps_done=3, steps_total=5, task_id='42')
        lane = _setup_lane(base, '_lane-0', plan)
        sidecar = {
            'session_id': 'uuid-warm-heuristic',
            'role': 'implementer',
            'started_at': '2026-07-19T09:00:00+00:00',
            'owner_pid': 4242,
            'task_id': '42',
            'resume_count': 0,
            'schema_version': 2,
        }
        (lane / '.task' / 'agent_session.json').write_text(json.dumps(sidecar))

        await harness._recover_crashed_tasks()

        assert '42' in harness._recovered_plans
        assert '42' in harness._recovered_sessions, (
            'Session must be keyed under real task_id, not lane dir name'
        )
        assert harness._recovered_sessions['42']['session_id'] == 'uuid-warm-heuristic'
        assert '_lane-0' not in harness._recovered_sessions

    async def test_warm_lane_recordless_plan_adopts_v1_sidecar_via_plan_task_id(
        self, harness: Harness,
    ):
        """Task 2772 (B11): a v1 sidecar (no task_id key) on a record-less
        warm lane is still adopted, keyed via plan.json's task_id rather
        than the sidecar itself (which has no id to key by)."""
        _pool = _attach_pool(harness, size=2)
        base = harness.git_ops.worktree_base
        plan = _make_plan(steps_done=3, steps_total=5, task_id='42')
        lane = _setup_lane(base, '_lane-0', plan)
        sidecar = {
            'session_id': 'uuid-warm-v1',
            'role': 'implementer',
            'started_at': '2026-07-19T09:00:00+00:00',
            'owner_pid': 4242,
        }
        (lane / '.task' / 'agent_session.json').write_text(json.dumps(sidecar))

        await harness._recover_crashed_tasks()

        assert '42' in harness._recovered_sessions
        assert harness._recovered_sessions['42'] == sidecar

    async def test_warm_lane_cleanup_not_called(self, harness: Harness):
        """cleanup_worktree must NOT be called for a lane with recoverable work."""
        _pool = _attach_pool(harness, size=2)
        base = harness.git_ops.worktree_base
        plan = _make_plan(steps_done=3, steps_total=5, task_id='42')
        _setup_lane(base, '_lane-0', plan)

        await harness._recover_crashed_tasks()

        harness.git_ops.cleanup_worktree.assert_not_called()  # type: ignore[attr-defined]

    async def test_warm_lane_pool_assignment_restored(self, harness: Harness):
        """After recovery, pool.assignment_for('42') == base/'_lane-0'.

        Record-driven (W11 delta): the pin now only happens via the ADOPT
        path, which requires a durable ASSIGNED record whose git reality
        matches (registered + branch checks out).
        """
        pool = _attach_pool(harness, size=2)
        base = harness.git_ops.worktree_base
        plan = _make_plan(steps_done=3, steps_total=5, task_id='42')
        lane_path = _setup_lane(base, '_lane-0', plan)
        _seed_lane_record(
            harness.git_ops._lane_lifecycle, lane_path, task_id='42', branch='task/42',
        )
        harness.git_ops.lane_branch_checkouts = AsyncMock(
            return_value={'42': lane_path},
        )

        await harness._recover_crashed_tasks()

        assert pool.assignment_for('42') == lane_path

    async def test_warm_lane_pool_state_assigned(self, harness: Harness):
        """After recovery, pool.state(base/'_lane-0') == LaneState.ASSIGNED.

        Record-driven (W11 delta): see test_warm_lane_pool_assignment_restored.
        """
        pool = _attach_pool(harness, size=2)
        base = harness.git_ops.worktree_base
        plan = _make_plan(steps_done=3, steps_total=5, task_id='42')
        lane_path = _setup_lane(base, '_lane-0', plan)
        _seed_lane_record(
            harness.git_ops._lane_lifecycle, lane_path, task_id='42', branch='task/42',
        )
        harness.git_ops.lane_branch_checkouts = AsyncMock(
            return_value={'42': lane_path},
        )

        await harness._recover_crashed_tasks()

        assert pool.state(lane_path) == LaneState.ASSIGNED

    async def test_warm_lane_cold_path_unaffected(self, harness: Harness):
        """Cold (non-lane) worktrees still recover normally alongside lane dirs."""
        _pool = _attach_pool(harness, size=2)
        base = harness.git_ops.worktree_base
        # Lane with completed work → recover under real task_id
        plan_lane = _make_plan(steps_done=2, steps_total=4, task_id='42')
        _setup_lane(base, '_lane-0', plan_lane)
        # Cold worktree with completed work → recover under dir name
        plan_cold = _make_plan(steps_done=1, steps_total=3, task_id='55')
        _setup_worktree(base, '55', plan_cold)

        await harness._recover_crashed_tasks()

        assert '42' in harness._recovered_plans
        assert '55' in harness._recovered_plans

    @pytest.mark.parametrize('term_status', ['done', 'cancelled'])
    async def test_warm_lane_terminal_task_released(
        self, harness: Harness, term_status: str
    ):
        """T10 amplifier: task already terminal → lane released, not restored.

        task 1881 regression lock: when get_status returns a terminal status
        ('done' or 'cancelled'), recovery must call cleanup_worktree instead of
        restore_assignment, preventing a dead lane from consuming a pool slot on
        every harness restart.  Both arms of the predicate are exercised via
        parametrize so a regression narrowing the check to only one value is caught.
        """
        pool = _attach_pool(harness, size=2)
        base = harness.git_ops.worktree_base
        plan = _make_plan(steps_done=3, steps_total=5, task_id='42')
        lane_path = _setup_lane(base, '_lane-0', plan)
        # Per-test override: drive the terminal (release) branch for each status
        harness.scheduler.get_status = AsyncMock(return_value=term_status)

        await harness._recover_crashed_tasks()

        # Lane was released (cleanup), not restored
        harness.git_ops.cleanup_worktree.assert_called_once_with(  # type: ignore[attr-defined]
            lane_path, '42'
        )
        # Plan NOT injected (released task needs no recovery)
        assert '42' not in harness._recovered_plans
        # Pool assignment NOT created (release path skips restore_assignment)
        assert pool.assignment_for('42') is None
        # Lane state must remain FREE — restore_assignment was bypassed
        assert pool.state(lane_path) == LaneState.FREE

    async def test_warm_lane_orphaned_registration_not_pinned(
        self, harness: Harness, caplog,
    ):
        """reify 4655/4947 (record-driven, W11 delta): an ORPHANED lane (a
        durable record ASSIGNED:'42', but no longer a registered git
        worktree) must be QUARANTINED, never re-pinned.  Restoring the
        assignment unconditionally would re-ASSIGN a broken lane on every
        restart, forcing the next dispatch down the faulting reuse
        fast-path and shielding the lane from the create-once self-heal
        forever.  Quarantining relocates the worktree out of the pool's
        way entirely rather than merely leaving it FREE-with-plan (PRD
        dec.4: any divergence quarantines, never adopt-on-doubt).

        Sibling coverage: test_harness_warm_lane_wiring.py::
        TestRecoveryTerminalTaskLaneRelease::test_recovery_skips_pin_for_unregistered_lane
        exercises the same invariant against a real git repo + GitOps pool;
        keep both in sync if this invariant ever changes.
        """
        pool = _attach_pool(harness, size=2)
        base = harness.git_ops.worktree_base
        plan = _make_plan(steps_done=3, steps_total=5, task_id='42')
        lane_path = _setup_lane(base, '_lane-0', plan)
        _seed_lane_record(
            harness.git_ops._lane_lifecycle, lane_path, task_id='42', branch='task/42',
        )
        harness.git_ops._is_registered_worktree = AsyncMock(return_value=False)
        # get_status left at fixture default (None) — non-terminal path.

        with caplog.at_level(logging.WARNING, logger='orchestrator.harness'):
            await harness._recover_crashed_tasks()

        # No pin: assignment map untouched, lane remains FREE
        assert pool.assignment_for('42') is None, (
            'unregistered lane must not be pinned to the task'
        )
        assert pool.state(lane_path) == LaneState.FREE, (
            'unregistered lane must stay FREE (quarantine relocates the '
            'git worktree; the pool cache is never pinned)'
        )
        # Plan is NOT recovered — the quarantine path skips plan recovery
        # entirely (never adopt-on-doubt).
        assert '42' not in harness._recovered_plans
        # Quarantined via the two-explicit-steps route (git_ops.quarantine_worktree
        # then the durable transition), never cleaned up (that's the
        # terminal-task release branch, a different cell).
        harness.git_ops.quarantine_worktree.assert_called_once_with(  # type: ignore[attr-defined]
            lane_path, 'task/42', 'recovery-record-divergence',
        )
        harness.git_ops.cleanup_worktree.assert_not_called()  # type: ignore[attr-defined]
        record = harness.git_ops._lane_lifecycle.read(lane_path)
        assert record is not None
        assert record.state == DurableLaneState.QUARANTINED
        # Loud post-crash integrity signal naming the task and the quarantine.
        assert any(
            rec.levelno == logging.WARNING
            and '42' in rec.getMessage()
            and 'quarantin' in rec.getMessage().lower()
            for rec in caplog.records
        ), f'expected a quarantine warning naming task 42; got: {caplog.text!r}'

    async def test_warm_lane_registration_check_exception_falls_back_to_pin(
        self, harness: Harness, caplog,
    ):
        """If the registration check itself raises (e.g. a WorktreeMissing/
        OSError from a git subprocess hiccup), recovery must not propagate the
        exception (which would abort recovery for every other worktree) and
        must not treat the failure as conclusive "unregistered" — it falls
        back to the pre-guard safe default (pin), mirroring the transient-None
        term_status handling above.  See harness.py:_recover_crashed_tasks for
        why the rc!=0-collapsed-to-False case inside _is_registered_worktree
        itself is a separate, out-of-scope concern (would need a git_ops.py
        contract change); this test locks only the exception-safety net that
        IS addressable from this call site.

        Record-driven (W11 delta): the exception only surfaces via the
        record-driven ADOPT/QUARANTINE decision, so this now requires a
        durable ASSIGNED record.  Seeded branchless (branch=None) since the
        safe-default fallback being exercised here is orthogonal to branch
        matching.
        """
        pool = _attach_pool(harness, size=2)
        base = harness.git_ops.worktree_base
        plan = _make_plan(steps_done=3, steps_total=5, task_id='42')
        lane_path = _setup_lane(base, '_lane-0', plan)
        _seed_lane_record(harness.git_ops._lane_lifecycle, lane_path, task_id='42')
        harness.git_ops._is_registered_worktree = AsyncMock(
            side_effect=OSError('git worktree list failed')
        )
        # get_status left at fixture default (None) — non-terminal path.

        with caplog.at_level(logging.WARNING, logger='orchestrator.harness'):
            await harness._recover_crashed_tasks()  # must not raise

        # Safe default: pin IS restored despite the raised exception
        assert pool.assignment_for('42') == lane_path
        assert pool.state(lane_path) == LaneState.ASSIGNED
        assert '42' in harness._recovered_plans
        harness.git_ops.quarantine_worktree.assert_not_called()  # type: ignore[attr-defined]
        record = harness.git_ops._lane_lifecycle.read(lane_path)
        assert record is not None
        assert record.state == DurableLaneState.ASSIGNED
        assert any(
            rec.levelno == logging.WARNING
            and '42' in rec.getMessage()
            and 'registration check raised' in rec.getMessage()
            for rec in caplog.records
        ), f'expected a registration-check-raised warning; got: {caplog.text!r}'


# ===========================================================================
# Step-5 RED: warm-lane edge cases
# (a) Stamped-but-no-progress lane
# (b) Plan-less lane (only agent_session.json sidecar, no plan.json)
# ===========================================================================


@pytest.mark.asyncio
@pytest.mark.timeout(180)  # task 2376: heavy class, widened from the 60s default to tolerate host oversubscription
class TestRecoverCrashedTasksWarmLaneEdgeCases:
    """Edge cases for warm-lane crash recovery."""

    async def test_stamped_no_progress_lane_preserved(self, harness: Harness):
        """(a) Stamped plan + 0 done steps on a lane → '77' in _preserved_worktrees,
        plan.lock removed, lane path NOT in _preserved_worktrees (stored by real id).

        Record-driven compat (W11 delta, PRD dec.5): this lane carries NO
        durable ``.lane-state`` record (a pre-W11 seed), so it takes the
        compat path — its plan is still recovered (here, preserved for
        revalidation) but the pool pin must NEVER be silently restored.
        """
        pool = _attach_pool(harness, size=2)
        base = harness.git_ops.worktree_base
        plan = _make_plan(
            steps_done=0, steps_total=4, task_id='77',
            session_id='77-aabbccddeeff',
        )
        lane_path = _setup_lane(base, '_lane-1', plan)
        lock_path = lane_path / '.task' / 'plan.lock'
        lock_path.write_text(json.dumps({'session_id': 'old', 'owner_pid': 1}))

        await harness._recover_crashed_tasks()

        # Preserved under the real task_id, not the lane dir name
        assert '77' in harness._preserved_worktrees
        assert '_lane-1' not in harness._preserved_worktrees
        # Stale lock cleared
        assert not lock_path.exists()
        # Record-less lane: never silently pinned (PRD dec.5)
        assert pool.assignment_for('77') is None
        assert pool.state(lane_path) == LaneState.FREE
        # cleanup NOT called (worktree preserved)
        harness.git_ops.cleanup_worktree.assert_not_called()  # type: ignore[attr-defined]

    async def test_planless_lane_released_to_pool(self, harness: Harness):
        """(b) Lane with only agent_session.json sidecar (no plan.json) →
        cleanup_worktree called with (base/'_lane-0', '_lane-0') and
        NEITHER '_lane-0' nor any session stored in _recovered_sessions/
        _recovered_plans.
        """
        _pool = _attach_pool(harness, size=2)
        base = harness.git_ops.worktree_base
        lane_path = base / '_lane-0'
        task_dir = lane_path / '.task'
        task_dir.mkdir(parents=True)
        sidecar = {
            'session_id': 'uuid-lane-mid-flight',
            'role': 'architect',
            'started_at': '2026-06-18T10:00:00+00:00',
            'owner_pid': 4242,
        }
        (task_dir / 'agent_session.json').write_text(json.dumps(sidecar))

        await harness._recover_crashed_tasks()

        # cleanup_worktree called for the lane (routes to release_warm_lane)
        harness.git_ops.cleanup_worktree.assert_called_once_with(  # type: ignore[attr-defined]
            lane_path, '_lane-0'
        )
        # Sidecar NOT stored (no task_id → can't key it)
        assert '_lane-0' not in harness._recovered_sessions
        assert '_lane-0' not in harness._preserved_worktrees
        assert '_lane-0' not in harness._recovered_plans

    async def test_planless_lane_with_v2_sidecar_adopts_session(self, harness: Harness):
        """Task 2772 (B3): a no-plan lane whose sidecar IS v2 (carries its
        own task_id) -> the session is adopted (keyed by the sidecar's
        task_id, the only source available on a no-plan lane), but the lane
        DISPOSITION is unchanged -- still released back to the pool via
        cleanup_worktree, same as the planless case above.
        """
        _pool = _attach_pool(harness, size=2)
        base = harness.git_ops.worktree_base
        lane_path = base / '_lane-0'
        task_dir = lane_path / '.task'
        task_dir.mkdir(parents=True)
        sidecar = {
            'session_id': 'uuid-lane-b3',
            'role': 'architect',
            'started_at': '2026-06-18T10:00:00+00:00',
            'owner_pid': 4242,
            'task_id': '73',
            'resume_count': 0,
            'schema_version': 2,
        }
        (task_dir / 'agent_session.json').write_text(json.dumps(sidecar))

        await harness._recover_crashed_tasks()

        assert '73' in harness._recovered_sessions
        assert harness._recovered_sessions['73']['session_id'] == 'uuid-lane-b3'
        assert '73' not in harness._recovered_plans
        # Disposition unchanged: still released back to the pool.
        harness.git_ops.cleanup_worktree.assert_called_once_with(  # type: ignore[attr-defined]
            lane_path, '_lane-0'
        )
        assert '_lane-0' not in harness._recovered_sessions

    async def test_planless_lane_corrupt_sidecar_falls_back_to_cleanup(
        self, harness: Harness,
    ):
        """A no-plan lane whose sidecar is malformed JSON -> the exception
        branch inside `_adopt_recovered_session` (harness.py:2401-2408) is
        hit via the task_id=None call site (harness.py:2777, only reachable
        from a lane) and returns None: nothing is adopted and disposition is
        unchanged (still released back to the pool), exactly like the
        cold-worktree analog `test_recover_corrupt_sidecar_falls_back_to_cleanup`
        above -- but that test only exercises the task_id-given call site
        (harness.py:2786), leaving this task_id=None site previously
        untested.
        """
        _pool = _attach_pool(harness, size=2)
        base = harness.git_ops.worktree_base
        lane_path = base / '_lane-0'
        task_dir = lane_path / '.task'
        task_dir.mkdir(parents=True)
        (task_dir / 'agent_session.json').write_text('{not json')

        await harness._recover_crashed_tasks()

        # Corrupt sidecar -> nothing adopted (no key was ever parseable)
        assert harness._recovered_sessions == {}
        assert '_lane-0' not in harness._preserved_worktrees
        # Disposition unchanged: still released back to the pool.
        harness.git_ops.cleanup_worktree.assert_called_once_with(  # type: ignore[attr-defined]
            lane_path, '_lane-0'
        )


@pytest.mark.asyncio
@pytest.mark.timeout(180)  # task 2376: heavy class, widened from the 60s default to tolerate host oversubscription
class TestRecoverCrashedTasksPoolStorageAbsentGuard:
    """_recover_crashed_tasks must defer — not clean up — when pool storage
    is absent (task 2099).

    Direct regression guard for the Jul-3 incident: an unmounted mountpoint
    dir must never let crash recovery treat every mount-resident worktree as
    planless/corrupt and destroy potentially-recoverable work.
    """

    async def test_storage_absent_defers_no_plan_worktree(self, harness: Harness):
        from orchestrator.git_ops import POOL_ROOT_SENTINEL

        # The `harness` fixture marks pool storage present by default —
        # remove the sentinel to simulate an unmounted mountpoint with a
        # live, empty (from git's perspective) mount dir.
        (harness.git_ops.worktree_base / POOL_ROOT_SENTINEL).unlink()
        # A pool must be configured for this guard to fire (step-17
        # review-fix): pool_storage_present() is permanently False on a
        # pool-less host by design, so pool_in_use() is what distinguishes
        # a real mount-down incident from that (see
        # TestRecoverCrashedTasksNoPoolConfiguredNoOp below).
        harness.git_ops.warm_lane_pool = WarmLanePool(
            worktree_base=harness.git_ops.worktree_base, size=1,
        )
        assert not harness.git_ops.pool_storage_present()

        wt = _setup_worktree(harness.git_ops.worktree_base, '36')  # no plan.json

        harness._file_pool_storage_absent_escalation = MagicMock()

        await harness._recover_crashed_tasks()

        harness.git_ops.cleanup_worktree.assert_not_called()  # type: ignore[attr-defined]
        assert harness._recovered_plans == {}
        harness._file_pool_storage_absent_escalation.assert_called_once()  # type: ignore[attr-defined]
        assert wt.exists()

    async def test_storage_absent_defers_corrupt_plan_worktree(self, harness: Harness):
        from orchestrator.git_ops import POOL_ROOT_SENTINEL

        (harness.git_ops.worktree_base / POOL_ROOT_SENTINEL).unlink()
        harness.git_ops.warm_lane_pool = WarmLanePool(
            worktree_base=harness.git_ops.worktree_base, size=1,
        )
        assert not harness.git_ops.pool_storage_present()

        wt = harness.git_ops.worktree_base / '38'
        task_dir = wt / '.task'
        task_dir.mkdir(parents=True)
        (task_dir / 'plan.json').write_text('{not valid json!!!')

        harness._file_pool_storage_absent_escalation = MagicMock()

        await harness._recover_crashed_tasks()

        harness.git_ops.cleanup_worktree.assert_not_called()  # type: ignore[attr-defined]
        assert harness._recovered_plans == {}
        harness._file_pool_storage_absent_escalation.assert_called_once()  # type: ignore[attr-defined]
        assert wt.exists()

    async def test_storage_absent_defers_recoverable_plan_worktree(
        self, harness: Harness,
    ):
        """Even a worktree WITH recoverable progress must not be scanned —
        the guard returns before the iterdir() loop, so _recovered_plans
        stays empty rather than getting a false sense of what's live."""
        from orchestrator.git_ops import POOL_ROOT_SENTINEL

        (harness.git_ops.worktree_base / POOL_ROOT_SENTINEL).unlink()
        harness.git_ops.warm_lane_pool = WarmLanePool(
            worktree_base=harness.git_ops.worktree_base, size=1,
        )
        assert not harness.git_ops.pool_storage_present()

        plan = _make_plan(steps_done=3, steps_total=5, task_id='35')
        _setup_worktree(harness.git_ops.worktree_base, '35', plan)

        harness._file_pool_storage_absent_escalation = MagicMock()

        await harness._recover_crashed_tasks()

        assert harness._recovered_plans == {}
        harness._file_pool_storage_absent_escalation.assert_called_once()  # type: ignore[attr-defined]

    async def test_storage_present_control_recovery_unchanged(
        self, harness: Harness,
    ):
        """Regression guard: sentinel present (the fixture default) — the
        existing recover-plan / cleanup-no-plan behavior is unchanged."""
        assert harness.git_ops.pool_storage_present()
        plan = _make_plan(steps_done=3, steps_total=5, task_id='35')
        _setup_worktree(harness.git_ops.worktree_base, '35', plan)
        wt_noplan = _setup_worktree(harness.git_ops.worktree_base, '36')

        await harness._recover_crashed_tasks()

        assert '35' in harness._recovered_plans
        harness.git_ops.cleanup_worktree.assert_called_once_with(  # type: ignore[attr-defined]
            wt_noplan, '36',
        )


@pytest.mark.asyncio
@pytest.mark.timeout(180)  # task 2376: heavy class, widened from the 60s default to tolerate host oversubscription
class TestRecoverCrashedTasksNoPoolConfiguredNoOp:
    """_recover_crashed_tasks() must proceed normally on a pool-less default
    host even though `.pool-root` is absent (step-17 review-fix).

    ``create_worktree`` places COLD worktrees directly at
    ``worktree_base/<branch>`` (git_ops.py:1234), so ``worktree_base.exists()``
    is True on any pool-disabled host that has ever run a task, while
    ``.pool-root`` (whose only writer, ``_seed_warm_lane`` on ``rc == 0``,
    requires a configured pool) is never written. Pre-fix, the guard
    deferred the ENTIRE recovery pass — destroying no work, but also
    recovering nothing — at every startup on such a host. Gating on
    ``pool_in_use()`` (task 2099 step-16) restores normal recovery when no
    pool is configured.
    """

    async def test_recovers_normally_when_no_pool_configured(self, harness: Harness):
        from orchestrator.git_ops import POOL_ROOT_SENTINEL

        (harness.git_ops.worktree_base / POOL_ROOT_SENTINEL).unlink()
        assert harness.git_ops.warm_lane_pool is None
        assert harness.git_ops.spec_warm_lane_pool is None
        assert not harness.git_ops.pool_storage_present()

        wt = _setup_worktree(harness.git_ops.worktree_base, '36')  # no plan.json
        harness._file_pool_storage_absent_escalation = MagicMock()

        await harness._recover_crashed_tasks()

        harness.git_ops.cleanup_worktree.assert_called_once_with(  # type: ignore[attr-defined]
            wt, '36',
        )
        harness._file_pool_storage_absent_escalation.assert_not_called()  # type: ignore[attr-defined]


# ===========================================================================
# Task 2257 (W11 delta) step-5 RED: record-driven crash recovery
# (git_ops._lane_lifecycle read -> verify-git -> adopt/quarantine),
# superseding the plan.json-only heuristic above for lanes carrying a
# durable LaneLifecycle record. See plans/worktree-lane-lifecycle-prd.md,
# task delta, mechanism 1.
# ===========================================================================


@pytest.mark.asyncio
@pytest.mark.timeout(180)  # task 2376: heavy class, widened from the 60s default to tolerate host oversubscription
class TestRecordDrivenRecovery:
    """B1 adopt / B2 quarantine / terminal-release / branch-mismatch / compat
    / .task-meta-relocation contracts for the record-driven lane recovery
    path.  Each test seeds a durable ``.lane-state/<lane>.json`` record via
    ``LaneLifecycle`` directly (the harness fixture already rebinds
    ``git_ops._lane_lifecycle`` to the test ``worktree_base`` — see the
    ``harness`` fixture above) rather than relying on plan.json heuristics.
    """

    async def test_adopt_on_exact_record_git_match(self, harness: Harness):
        """B1: durable record ASSIGNED:'42' branch='task/42', git reality
        matches (still registered + checked-out branch matches the record)
        and the task is non-terminal -> ADOPT: pool rebound to the lane,
        plan recovered from .task-meta, durable record left ASSIGNED:'42',
        and neither quarantine nor cleanup is invoked.
        """
        pool = _attach_pool(harness, size=2)
        base = harness.git_ops.worktree_base
        lane = base / '_lane-0'
        lane.mkdir(parents=True, exist_ok=True)  # the lane dir itself (no .task/ needed)
        lifecycle = harness.git_ops._lane_lifecycle
        _seed_lane_record(lifecycle, lane, task_id='42', branch='task/42')

        plan = _make_plan(steps_done=3, steps_total=5, task_id='42')
        _setup_lane_meta_plan(base, '_lane-0', plan)

        harness.git_ops._is_registered_worktree = AsyncMock(return_value=True)
        # lane_branch_checkouts returns {bare_id: lane} (branch_prefix
        # stripped — see GitOps.lane_branch_checkouts's documented
        # contract); the harness reconstructs the full branch name via
        # config.branch_prefix ('task/' by default) to compare against
        # rec.branch.
        harness.git_ops.lane_branch_checkouts = AsyncMock(
            return_value={'42': lane},
        )
        harness.scheduler.get_status = AsyncMock(return_value=None)  # non-terminal

        await harness._recover_crashed_tasks()

        assert pool.assignment_for('42') == lane
        assert pool.state(lane) == LaneState.ASSIGNED
        assert '42' in harness._recovered_plans

        record = lifecycle.read(lane)
        assert record is not None
        assert record.state == DurableLaneState.ASSIGNED
        assert record.task_id == '42'

        harness.git_ops.quarantine_worktree.assert_not_called()  # type: ignore[attr-defined]
        harness.git_ops.cleanup_worktree.assert_not_called()  # type: ignore[attr-defined]

    async def test_adopt_on_exact_record_git_match_also_adopts_session(
        self, harness: Harness,
    ):
        """Task 2772 (B1): same ADOPT shape as
        test_adopt_on_exact_record_git_match, but with a v2
        agent_session.json sidecar co-located under the SAME
        .task-meta/_lane-0 new path -- the record-driven completed-steps
        branch (~2635) must ALSO populate _recovered_sessions, not just
        _recovered_plans, without disturbing the pool pin.
        """
        pool = _attach_pool(harness, size=2)
        base = harness.git_ops.worktree_base
        lane = base / '_lane-0'
        lane.mkdir(parents=True, exist_ok=True)  # the lane dir itself (no .task/ needed)
        lifecycle = harness.git_ops._lane_lifecycle
        _seed_lane_record(lifecycle, lane, task_id='42', branch='task/42')

        plan = _make_plan(steps_done=3, steps_total=5, task_id='42')
        meta_dir = _setup_lane_meta_plan(base, '_lane-0', plan)
        sidecar = {
            'session_id': 'uuid-warm-b1',
            'role': 'implementer',
            'started_at': '2026-07-19T09:00:00+00:00',
            'owner_pid': 4242,
            'task_id': '42',
            'resume_count': 0,
            'schema_version': 2,
        }
        (meta_dir / 'agent_session.json').write_text(json.dumps(sidecar))

        harness.git_ops._is_registered_worktree = AsyncMock(return_value=True)
        harness.git_ops.lane_branch_checkouts = AsyncMock(
            return_value={'42': lane},
        )
        harness.scheduler.get_status = AsyncMock(return_value=None)  # non-terminal

        await harness._recover_crashed_tasks()

        assert '42' in harness._recovered_plans
        assert '42' in harness._recovered_sessions
        assert harness._recovered_sessions['42']['session_id'] == 'uuid-warm-b1'
        # Pin unchanged from the B1 baseline.
        assert pool.assignment_for('42') == lane

    async def test_quarantine_on_registration_divergence(
        self, harness: Harness, caplog,
    ):
        """B2: durable record ASSIGNED:'42' branch='task/42' but the git
        admin entry is gone (the 2097/2098 orphaned-worktree divergence) ->
        QUARANTINE, never re-pinned.  Quarantine is two explicit steps
        (git_ops.quarantine_worktree then git_ops._lane_lifecycle.transition
        to QUARANTINED -- see the design decision on the stale injected
        callable), the lane is left FREE in the pool cache (never restored/
        re-pinned), and the next dispatch for a different task must still
        find a usable lane.
        """
        pool = _attach_pool(harness, size=2)
        base = harness.git_ops.worktree_base
        lane = base / '_lane-0'
        lane.mkdir(parents=True, exist_ok=True)  # the lane dir itself (no .task/ needed)
        lifecycle = harness.git_ops._lane_lifecycle
        _seed_lane_record(lifecycle, lane, task_id='42', branch='task/42')

        harness.git_ops._is_registered_worktree = AsyncMock(return_value=False)
        harness.scheduler.get_status = AsyncMock(return_value=None)  # non-terminal

        with caplog.at_level(logging.WARNING, logger='orchestrator.harness'):
            await harness._recover_crashed_tasks()

        harness.git_ops.quarantine_worktree.assert_called_once_with(  # type: ignore[attr-defined]
            lane, 'task/42', 'recovery-record-divergence',
        )
        harness.git_ops.cleanup_worktree.assert_not_called()  # type: ignore[attr-defined]

        record = lifecycle.read(lane)
        assert record is not None
        assert record.state == DurableLaneState.QUARANTINED

        # Never re-pinned: no assignment, lane stays FREE in the cache, and
        # no plan is recovered for the divergent record.
        assert pool.assignment_for('42') is None
        assert pool.state(lane) == LaneState.FREE
        assert '42' not in harness._recovered_plans

        emitted = {c.args[0] for c in harness.event_store.emit.call_args_list}  # type: ignore[attr-defined]
        assert EventType.worktree_quarantined in emitted

        assert any(
            rec.levelno == logging.WARNING
            and '42' in rec.getMessage()
            and 'quarantin' in rec.getMessage().lower()
            for rec in caplog.records
        ), f'expected a quarantine warning naming task 42; got: {caplog.text!r}'

        # Next dispatch is clean: a different task can still acquire a lane
        # (no lingering assignment/exhaustion from the quarantined lane).
        result = await pool.acquire_for('task/99')
        assert result is not None, (
            'pool must not be stuck exhausted after quarantining a lane'
        )

    async def test_terminal_task_releases_lane_not_adopted(self, harness: Harness):
        """Terminal task (T10 amplifier): durable record ASSIGNED:'42'
        branch='task/42' and git reality still matches (registered, branch
        OK), but the task itself is already terminal (scheduler.get_status
        -> 'done').  The lane must be RELEASED (cleanup_worktree), NOT
        adopted -- re-pinning a dead task's lane on every restart would
        shrink the pool forever.  This is resolved BEFORE the git-reality
        adopt/quarantine decision, so it must win even though registration
        and branch both check out fine.

        step-16 review-fix regression guard: cleanup_worktree's side_effect
        performs the REAL durable RELEASED write (mirroring what
        release_warm_lane -> pool.release -> _note_released_durable does for a
        real warm lane) BEFORE the harness's own explicit transition runs, so
        an unconditional second RELEASED -> RELEASED transition would raise
        IllegalLaneTransition uncaught out of the recovery loop.  A second,
        non-terminal lane seeded AFTER this one proves recovery keeps going
        instead of aborting mid-loop.
        """
        pool = _attach_pool(harness, size=2)
        base = harness.git_ops.worktree_base
        lane = base / '_lane-0'
        lane.mkdir(parents=True, exist_ok=True)
        lifecycle = harness.git_ops._lane_lifecycle
        _seed_lane_record(lifecycle, lane, task_id='42', branch='task/42')

        # Second, non-terminal lane seeded AFTER the terminal one -- proves
        # the loop doesn't abort mid-recovery on the redundant-transition bug.
        lane2 = base / '_lane-1'
        plan2 = _make_plan(steps_done=1, steps_total=2, task_id='99')
        _setup_lane(base, '_lane-1', plan2)
        _seed_lane_record(lifecycle, lane2, task_id='99', branch='task/99')

        harness.git_ops._is_registered_worktree = AsyncMock(return_value=True)
        harness.git_ops.lane_branch_checkouts = AsyncMock(
            return_value={'42': lane, '99': lane2},
        )

        async def _get_status(task_id):
            return 'done' if task_id == '42' else None

        harness.scheduler.get_status = AsyncMock(side_effect=_get_status)
        harness.git_ops.cleanup_worktree.side_effect = (  # type: ignore[attr-defined]
            lambda entry, tid: harness.git_ops._lane_lifecycle.transition(
                entry, DurableLaneState.RELEASED,
            )
        )

        await harness._recover_crashed_tasks()  # must not raise

        harness.git_ops.cleanup_worktree.assert_called_once_with(  # type: ignore[attr-defined]
            lane, '42',
        )
        harness.git_ops.quarantine_worktree.assert_not_called()  # type: ignore[attr-defined]

        record = lifecycle.read(lane)
        assert record is not None
        assert record.state == DurableLaneState.RELEASED

        assert pool.assignment_for('42') is None
        assert '42' not in harness._recovered_plans

        # The second, non-terminal lane is still adopted -- recovery did not
        # abort mid-loop when it hit the terminal lane's redundant transition.
        assert pool.assignment_for('99') == lane2
        assert '99' in harness._recovered_plans

    async def test_quarantine_on_branch_mismatch(self, harness: Harness):
        """The 2062 detached-HEAD/stale-branch collision: durable record
        ASSIGNED:'42' branch='task/42', the admin entry is still registered,
        but the lane is ACTUALLY checked out onto a different task's branch
        (task/99) -- a stale-branch collision, not an orphan.  Must route to
        the SAME quarantine cell as the not-registered divergence (each
        historical bug collapses to one QUARANTINE cell), never adopt.
        """
        pool = _attach_pool(harness, size=2)
        base = harness.git_ops.worktree_base
        lane = base / '_lane-0'
        lane.mkdir(parents=True, exist_ok=True)
        lifecycle = harness.git_ops._lane_lifecycle
        _seed_lane_record(lifecycle, lane, task_id='42', branch='task/42')

        harness.git_ops._is_registered_worktree = AsyncMock(return_value=True)
        # lane_branch_checkouts returns {bare_id: lane} (branch_prefix
        # already stripped -- see GitOps.lane_branch_checkouts's documented
        # contract, and the comment on the matching adopt-path test above).
        # bare_id '99' reconstructs to 'task/99', which disagrees with the
        # record's 'task/42' -- the lane is actually checked out on a
        # DIFFERENT task's branch than its own durable record claims.
        harness.git_ops.lane_branch_checkouts = AsyncMock(
            return_value={'99': lane},
        )
        harness.scheduler.get_status = AsyncMock(return_value=None)  # non-terminal

        await harness._recover_crashed_tasks()

        harness.git_ops.quarantine_worktree.assert_called_once_with(  # type: ignore[attr-defined]
            lane, 'task/42', 'recovery-record-divergence',
        )
        harness.git_ops.cleanup_worktree.assert_not_called()  # type: ignore[attr-defined]

        record = lifecycle.read(lane)
        assert record is not None
        assert record.state == DurableLaneState.QUARANTINED

        assert pool.assignment_for('42') is None
        assert '42' not in harness._recovered_plans

    async def test_adopts_on_unresolvable_branch_read(self, harness: Harness):
        """Fail-safe: lane_branch_checkouts() returns None when `git worktree
        list` errors (its documented contract -- "never mass-mutate on an
        unreliable read").  A record ASSIGNED:'42' branch='task/42' whose admin
        entry IS registered must NOT be quarantined merely because the branch
        read was unresolvable; a transient git hiccup at startup would otherwise
        quarantine every assigned lane carrying a branch record (the normal
        case) and drop its recovered plan.  The lane is ADOPTED and its plan
        recovered, mirroring the is_registered OSError fail-safe.
        """
        pool = _attach_pool(harness, size=2)
        base = harness.git_ops.worktree_base
        plan = _make_plan(steps_done=3, steps_total=5, task_id='42')
        lane = _setup_lane(base, '_lane-0', plan)
        lifecycle = harness.git_ops._lane_lifecycle
        _seed_lane_record(lifecycle, lane, task_id='42', branch='task/42')

        harness.git_ops._is_registered_worktree = AsyncMock(return_value=True)
        # None => unresolvable read (git error / pool disabled), NOT a
        # resolved-but-absent lane.  Must fall through to adopt.
        harness.git_ops.lane_branch_checkouts = AsyncMock(return_value=None)
        harness.scheduler.get_status = AsyncMock(return_value=None)

        await harness._recover_crashed_tasks()

        harness.git_ops.quarantine_worktree.assert_not_called()  # type: ignore[attr-defined]
        harness.git_ops.cleanup_worktree.assert_not_called()  # type: ignore[attr-defined]
        assert pool.assignment_for('42') == lane
        assert '42' in harness._recovered_plans
        record = lifecycle.read(lane)
        assert record is not None
        assert record.state == DurableLaneState.ASSIGNED

    async def test_adopt_stamped_zero_completed_preserves_not_preloads(
        self, harness: Harness,
    ):
        """ADOPT + a stamped plan with zero completed steps (the
        blast-radius lock-conflict requeue shape, workflow.py:1071-1088)
        must NOT be pre-loaded into _recovered_plans -- workflow.py treats a
        pre-loaded plan as initial_plan and skips _plan() entirely, so a
        stale plan would never be revalidated against a possibly-advanced
        main.  It must instead land in _preserved_worktrees, mirroring the
        heuristic path's stamped-preservation branch (~2415 below), so the
        next acquisition still takes _plan()'s revalidation branch.  The
        lane is still ADOPTED (pinned) regardless -- only the pre-load
        decision changes.
        """
        pool = _attach_pool(harness, size=2)
        base = harness.git_ops.worktree_base
        plan = _make_plan(
            steps_done=0, steps_total=4, task_id='42', session_id='sess-abc',
        )
        lane = _setup_lane(base, '_lane-0', plan)
        lifecycle = harness.git_ops._lane_lifecycle
        _seed_lane_record(lifecycle, lane, task_id='42', branch='task/42')

        harness.git_ops._is_registered_worktree = AsyncMock(return_value=True)
        harness.git_ops.lane_branch_checkouts = AsyncMock(
            return_value={'42': lane},
        )
        harness.scheduler.get_status = AsyncMock(return_value=None)  # non-terminal

        await harness._recover_crashed_tasks()

        # Still adopted/pinned -- the stamped-zero-completed shape only
        # changes whether the plan is pre-loaded, not the pin decision.
        assert pool.assignment_for('42') == lane
        assert pool.state(lane) == LaneState.ASSIGNED

        assert '42' not in harness._recovered_plans
        assert '42' in harness._preserved_worktrees

        record = lifecycle.read(lane)
        assert record is not None
        assert record.state == DurableLaneState.ASSIGNED
        assert record.task_id == '42'

        harness.git_ops.quarantine_worktree.assert_not_called()  # type: ignore[attr-defined]
        harness.git_ops.cleanup_worktree.assert_not_called()  # type: ignore[attr-defined]


# ===========================================================================
# Task 2257 (W11 delta) step-11 RED: compat (never-silently-re-pin a
# record-less lane) + .task-meta read relocation.
# ===========================================================================


@pytest.mark.asyncio
@pytest.mark.timeout(180)  # task 2376: heavy class, widened from the 60s default to tolerate host oversubscription
class TestRecordDrivenRecoveryCompatAndRelocation:
    """(a) compat: a record-less lane recovers its plan but is never pinned.
    (b) .task-meta-only artifacts are read/cleared on the ADOPT path.
    (c) legacy-only artifacts (<wt>/.task, no .task-meta) still recover.
    """

    async def test_compat_no_record_lane_recovers_plan_without_pinning(
        self, harness: Harness,
    ):
        """(a) COMPAT: a pool lane with plan.json (task_id='42', 3/5 steps
        done) but NO .lane-state record -> the plan IS recovered but the
        lane is NOT pinned (PRD dec.5: never silently re-pin a record-less
        lane) and NOT quarantined/cleaned up.
        """
        pool = _attach_pool(harness, size=2)
        base = harness.git_ops.worktree_base
        plan = _make_plan(steps_done=3, steps_total=5, task_id='42')
        lane = _setup_lane(base, '_lane-0', plan)  # legacy path, no record

        await harness._recover_crashed_tasks()

        assert '42' in harness._recovered_plans
        assert pool.assignment_for('42') is None, (
            'a record-less lane must never be silently pinned'
        )
        assert pool.state(lane) == LaneState.FREE
        harness.git_ops.quarantine_worktree.assert_not_called()  # type: ignore[attr-defined]
        harness.git_ops.cleanup_worktree.assert_not_called()  # type: ignore[attr-defined]

    async def test_compat_planless_lane_released(self, harness: Harness):
        """(a) COMPAT, planless variant: no plan.json anywhere (new or
        legacy) and no record -> released back to the pool, never pinned.
        """
        pool = _attach_pool(harness, size=2)
        base = harness.git_ops.worktree_base
        lane = base / '_lane-0'
        lane.mkdir(parents=True, exist_ok=True)

        await harness._recover_crashed_tasks()

        harness.git_ops.cleanup_worktree.assert_called_once_with(  # type: ignore[attr-defined]
            lane, '_lane-0',
        )
        assert pool.assignment_for('42') is None

    async def test_adopt_reads_and_clears_task_meta_new_path_only(
        self, harness: Harness,
    ):
        """(b) A lane whose ASSIGNED record matches git and whose
        plan.json/plan.lock live ONLY under
        TaskArtifacts.meta_root_for(base, name) (new path, not <wt>/.task)
        is adopted, its plan recovered from the new path, and plan.lock at
        the new path is cleared.
        """
        pool = _attach_pool(harness, size=2)
        base = harness.git_ops.worktree_base
        lane = base / '_lane-0'
        lane.mkdir(parents=True, exist_ok=True)
        lifecycle = harness.git_ops._lane_lifecycle
        _seed_lane_record(lifecycle, lane, task_id='42', branch='task/42')

        plan = _make_plan(steps_done=3, steps_total=5, task_id='42')
        meta_dir = _setup_lane_meta_plan(base, '_lane-0', plan)
        (meta_dir / 'plan.lock').write_text(
            json.dumps({'session_id': 'x', 'owner_pid': 1})
        )

        harness.git_ops._is_registered_worktree = AsyncMock(return_value=True)
        harness.git_ops.lane_branch_checkouts = AsyncMock(
            return_value={'42': lane},
        )
        harness.scheduler.get_status = AsyncMock(return_value=None)

        await harness._recover_crashed_tasks()

        assert pool.assignment_for('42') == lane
        assert '42' in harness._recovered_plans
        assert not (meta_dir / 'plan.lock').exists()
        # legacy .task dir was never created for this lane
        assert not (lane / '.task').exists()

    async def test_adopt_falls_back_to_legacy_task_dir(self, harness: Harness):
        """(c) Legacy fallback: a lane with artifacts ONLY under
        <wt>/.task (no .task-meta at all) still recovers via new-then-old
        resolution.
        """
        pool = _attach_pool(harness, size=2)
        base = harness.git_ops.worktree_base
        plan = _make_plan(steps_done=3, steps_total=5, task_id='42')
        lane = _setup_lane(base, '_lane-0', plan)  # legacy .task/plan.json only
        lifecycle = harness.git_ops._lane_lifecycle
        _seed_lane_record(lifecycle, lane, task_id='42', branch='task/42')

        harness.git_ops._is_registered_worktree = AsyncMock(return_value=True)
        harness.git_ops.lane_branch_checkouts = AsyncMock(
            return_value={'42': lane},
        )
        harness.scheduler.get_status = AsyncMock(return_value=None)

        await harness._recover_crashed_tasks()

        assert pool.assignment_for('42') == lane
        assert '42' in harness._recovered_plans


@pytest.mark.asyncio
class TestSessionResumeGuard:
    """γ eligibility guard in _run_slot (task 2774): an ineligible recovered
    session degrades to fresh dispatch (resume_session_id=None) with a
    reason-carrying event (B4/B5/B7); an eligible one is injected and emits a
    session_resume event. The kill switch (enabled=False) degrades silently
    with no event (B6). The guard is fail-safe (I3) — every ineligible path
    is no-worse than today's fresh dispatch.
    """

    async def test_eligible_keeps_session_and_emits(self, harness: Harness, tmp_path: Path):
        session = {
            'session_id': 'uuid-elig',
            'role': 'implementer',
            'started_at': datetime.now(UTC).isoformat(),
            'resume_count': 0,
        }
        cfg = _make_transcript(tmp_path, 'uuid-elig')
        harness.config.session_resume = SessionResumeConfig()

        resume_id = await _drive_session_slot(harness, 'e1', session, config_dir=cfg)

        assert resume_id is session
        emits = _session_resume_emits(harness)
        assert [et for et, _ in emits] == [EventType.session_resume]

    async def test_disabled_falls_back_silently(self, harness: Harness, tmp_path: Path):
        """enabled=False → no --resume injected AND no session_resume_* event (B6)."""
        session = {
            'session_id': 'uuid-dis',
            'role': 'implementer',
            'started_at': datetime.now(UTC).isoformat(),
            'resume_count': 0,
        }
        cfg = _make_transcript(tmp_path, 'uuid-dis')
        harness.config.session_resume = SessionResumeConfig(enabled=False)

        resume_id = await _drive_session_slot(harness, 'd1', session, config_dir=cfg)

        assert resume_id is None
        assert _session_resume_emits(harness) == []

    async def test_stale_falls_back(self, harness: Harness, tmp_path: Path):
        """Sidecar older than freshness_window → fallback reason 'stale' (B5)."""
        real = SessionResumeConfig()
        stale = datetime.now(UTC) - timedelta(
            seconds=2 * real.freshness_window_secs
        )
        session = {
            'session_id': 'uuid-stale',
            'role': 'implementer',
            'started_at': stale.isoformat(),
            'resume_count': 0,
        }
        cfg = _make_transcript(tmp_path, 'uuid-stale')
        harness.config.session_resume = real

        resume_id = await _drive_session_slot(harness, 's1', session, config_dir=cfg)

        assert resume_id is None
        emits = _session_resume_emits(harness)
        assert len(emits) == 1
        et, kwargs = emits[0]
        assert et == EventType.session_resume_fallback
        assert kwargs['data']['reason'] == 'stale'

    async def test_unparseable_or_missing_started_at_falls_back_stale(
        self, harness: Harness, tmp_path: Path
    ):
        """A garbage or absent started_at fails the freshness parse (fail-safe)
        → fallback reason 'stale', BEFORE the transcript leg is reached.
        """
        harness.config.session_resume = SessionResumeConfig()

        cfg1 = _make_transcript(tmp_path, 'uuid-bad')
        s1 = {
            'session_id': 'uuid-bad', 'role': 'r',
            'started_at': 'not-a-date', 'resume_count': 0,
        }
        rid1 = await _drive_session_slot(harness, 'u1', s1, config_dir=cfg1)
        assert rid1 is None

        cfg2 = _make_transcript(tmp_path, 'uuid-bad2')
        s2 = {'session_id': 'uuid-bad2', 'role': 'r', 'resume_count': 0}  # no started_at
        rid2 = await _drive_session_slot(harness, 'u2', s2, config_dir=cfg2)
        assert rid2 is None

        emits = _session_resume_emits(harness)
        assert len(emits) == 2
        for et, kwargs in emits:
            assert et == EventType.session_resume_fallback
            assert kwargs['data']['reason'] == 'stale'

    async def test_transcript_absent_falls_back_no_transcript(
        self, harness: Harness, tmp_path: Path
    ):
        """The config dir SURVIVES on disk but this session's transcript jsonl
        is absent → fallback reason 'no_transcript' (a genuine corroboration
        failure — distinct from the whole-store wipe, which is 'reseeded').
        """
        session = {
            'session_id': 'uuid-notr',
            'role': 'implementer',
            'started_at': datetime.now(UTC).isoformat(),
            'resume_count': 0,
        }
        empty_cfg = tmp_path / 'claude-config-empty'
        (empty_cfg / 'projects').mkdir(parents=True)
        harness.config.session_resume = SessionResumeConfig()

        resume_id = await _drive_session_slot(harness, 'n1', session, config_dir=empty_cfg)

        assert resume_id is None
        emits = _session_resume_emits(harness)
        assert len(emits) == 1
        et, kwargs = emits[0]
        assert et == EventType.session_resume_fallback
        assert kwargs['data']['reason'] == 'no_transcript'

    async def test_wiped_config_dir_falls_back_reseeded(
        self, harness: Harness, tmp_path: Path
    ):
        """The stashed config dir is GONE from disk → fallback reason
        'reseeded' (task 3256).

        Models the lane having been re-seeded between boot-time adoption and
        re-dispatch: warm-lane acquire ALWAYS re-seeds from base, which wipes
        ``<lane>/.task/`` (``git clean -xfd`` on the RECYCLE route,
        ``rmtree(lane/'.task')`` on RESET_IN_PLACE_REATTACH) and with it the
        whole ``claude-config-*`` transcript store. That is the always-reseed
        invariant working as designed — an EXPECTED fallback, not a
        corroboration failure — so it must be classified apart from
        'no_transcript'.
        """
        session = {
            'session_id': 'uuid-reseed',
            'role': 'implementer',
            'started_at': datetime.now(UTC).isoformat(),
            'resume_count': 0,
        }
        gone = tmp_path / 'gone' / 'claude-config-x'
        assert not gone.exists()  # the reseed already swept the lane
        harness.config.session_resume = SessionResumeConfig()

        resume_id = await _drive_session_slot(harness, 'rs1', session, config_dir=gone)

        assert resume_id is None
        emits = _session_resume_emits(harness)
        assert len(emits) == 1
        et, kwargs = emits[0]
        # The event type is UNCHANGED — the downgrade suppresses the
        # escalation, not the telemetry channel (PRD open question 3).
        assert et == EventType.session_resume_fallback
        assert kwargs['data']['reason'] == 'reseeded'

    async def test_surviving_config_dir_missing_transcript_stays_no_transcript(
        self, harness: Harness, tmp_path: Path
    ):
        """A config dir that SURVIVES the lane's lifetime but has lost only
        this session's transcript must NOT be swallowed by the new 'reseeded'
        branch (task 3256) — it is a real, distinct failure mode and stays a
        loud 'no_transcript' corroboration failure.
        """
        session = {
            'session_id': 'uuid-survivor',
            'role': 'implementer',
            'started_at': datetime.now(UTC).isoformat(),
            'resume_count': 0,
        }
        # A REAL claude-config dir, with a projects/ tree holding some OTHER
        # session's transcript but not this one's.
        survivor = _make_transcript(tmp_path, 'uuid-other')
        assert survivor.is_dir()
        harness.config.session_resume = SessionResumeConfig()

        resume_id = await _drive_session_slot(
            harness, 'sv1', session, config_dir=survivor
        )

        assert resume_id is None
        emits = _session_resume_emits(harness)
        assert len(emits) == 1
        et, kwargs = emits[0]
        assert et == EventType.session_resume_fallback
        assert kwargs['data']['reason'] == 'no_transcript'

    async def test_unreadable_config_dir_stays_no_transcript(
        self, harness: Harness, tmp_path: Path, monkeypatch: pytest.MonkeyPatch
    ):
        """A config dir that is PRESENT but UNREADABLE (EACCES on a parent,
        ELOOP, a stale NFS handle) is a filesystem fault, not a lane wipe, and
        must stay a loud 'no_transcript' (task 3256 amendment).

        Keying the split on ``Path.exists()`` is wrong in BOTH directions: it
        swallows exactly {ENOENT, ENOTDIR, EBADF, ELOOP} into False, filing a
        symlink loop or bad fd under the silent, storm-exempt 'reseeded' arm,
        and it RE-RAISES every other OSError — the EACCES injected here — out
        through the guard's documented never-raises I3 contract. Either way a
        genuine filesystem fault stops surfacing as the systematic breakage
        INV-4 exists to catch. Only ENOENT/ENOTDIR is evidence of a wipe.

        The error is INJECTED rather than produced with ``chmod(0o000)`` so
        the test is deterministic and still meaningful when the suite runs as
        root (a root process can stat through a 0o000 parent).
        """
        session = {
            'session_id': 'uuid-eacces',
            'role': 'implementer',
            'started_at': datetime.now(UTC).isoformat(),
            'resume_count': 0,
        }
        blocked = tmp_path / 'unreadable' / 'claude-config-x'
        blocked.mkdir(parents=True)
        real_stat = Path.stat

        def fake_stat(self: Path, *args, **kwargs):
            if self == blocked:
                raise PermissionError(13, 'Permission denied')
            return real_stat(self, *args, **kwargs)

        monkeypatch.setattr(Path, 'stat', fake_stat)
        harness.config.session_resume = SessionResumeConfig()

        resume_id = await _drive_session_slot(
            harness, 'ur1', session, config_dir=blocked
        )

        assert resume_id is None
        emits = _session_resume_emits(harness)
        assert len(emits) == 1
        et, kwargs = emits[0]
        assert et == EventType.session_resume_fallback
        assert kwargs['data']['reason'] == 'no_transcript'

    async def test_no_config_dir_falls_back_no_transcript(self, harness: Harness):
        """No stashed config_dir at all → cannot corroborate → 'no_transcript'."""
        session = {
            'session_id': 'uuid-nocfg',
            'role': 'implementer',
            'started_at': datetime.now(UTC).isoformat(),
            'resume_count': 0,
        }
        harness.config.session_resume = SessionResumeConfig()

        resume_id = await _drive_session_slot(harness, 'nc1', session)  # config_dir=None

        assert resume_id is None
        emits = _session_resume_emits(harness)
        assert len(emits) == 1
        assert emits[0][0] == EventType.session_resume_fallback
        assert emits[0][1]['data']['reason'] == 'no_transcript'

    async def test_capped_emits_capped(self, harness: Harness, tmp_path: Path):
        """resume_count at the cap → fresh dispatch + session_resume_capped (B7).

        Distinct from a fallback: capped is by-design throttling, its own event.
        """
        session = {
            'session_id': 'uuid-cap',
            'role': 'implementer',
            'started_at': datetime.now(UTC).isoformat(),
            'resume_count': 3,
        }
        cfg = _make_transcript(tmp_path, 'uuid-cap')
        harness.config.session_resume = SessionResumeConfig(max_resumes_per_task=3)

        resume_id = await _drive_session_slot(harness, 'c1', session, config_dir=cfg)

        assert resume_id is None
        emits = _session_resume_emits(harness)
        assert len(emits) == 1
        assert emits[0][0] == EventType.session_resume_capped


@pytest.mark.asyncio
class TestSessionResumeStorm:
    """γ fallback-storm escape (INV-4, task 2774, narrowed by task 3256): a RUN
    of UNEXPLAINED session_resume_fallback degradations reaching
    fallback_storm_threshold files ONE deduped L1 escalation.

    The streak is a rolling CHAIN, not a per-boot running total: consecutive
    means chained within storm_window_secs, so a gap at least that long decays
    it to 0 (an eligible resume also resets it outright, and clears the chain's
    comparison stamp). Three degradations are excluded from it — 'disabled'
    (silent kill switch), 'capped' (by-design throttling) and 'reseeded'
    (by-design lane wipe) — leaving only the genuine corroboration failures
    {stale, no_transcript}. Filing is best-effort — a None queue never raises
    (I3).
    """

    @staticmethod
    def _stale_session(sid: str) -> dict:
        stale = datetime.now(UTC) - timedelta(seconds=2 * 86400)
        return {
            'session_id': sid,
            'role': 'implementer',
            'started_at': stale.isoformat(),
            'resume_count': 0,
        }

    @staticmethod
    def _fresh_session(sid: str) -> dict:
        """A session that passes freshness + cap, so its only failing leg is
        transcript corroboration (used to drive the 'reseeded' arm).
        """
        return {
            'session_id': sid,
            'role': 'implementer',
            'started_at': datetime.now(UTC).isoformat(),
            'resume_count': 0,
        }

    @staticmethod
    def _queue() -> MagicMock:
        q = MagicMock()
        q.has_open_l1 = MagicMock(return_value=False)
        q.make_id = MagicMock(return_value='sr-storm')
        return q

    async def test_reseeded_fallbacks_never_file_l1(
        self, harness: Harness, tmp_path: Path
    ):
        """Reseed-explained fallbacks are EXPECTED and must never trip the
        storm escape, however many of them arrive (task 3256).

        threshold=1 makes the very first genuine fallback fire, so a zero
        submit count across three reseeded dispatches proves they do not feed
        the streak at all. The telemetry channel must SURVIVE the downgrade:
        this is noise suppression of the ESCALATION, not of the event.
        """
        harness.config.session_resume = SessionResumeConfig(fallback_storm_threshold=1)
        harness._escalation_queue = self._queue()

        for i in range(3):
            await _drive_session_slot(
                harness, f'rs{i}', self._fresh_session(f'uuid-rs{i}'),
                config_dir=tmp_path / f'gone{i}' / 'claude-config-x',
            )

        assert harness._escalation_queue.submit.call_count == 0
        assert harness._session_resume_fallback_streak == 0
        emits = _session_resume_emits(harness)
        assert len(emits) == 3
        for et, kwargs in emits:
            assert et == EventType.session_resume_fallback
            assert kwargs['data']['reason'] == 'reseeded'

    async def test_genuine_failures_still_file_l1_across_reseeded(
        self, harness: Harness, tmp_path: Path
    ):
        """A reseeded fallback neither counts toward NOR resets the genuine
        streak — the same semantics as ``capped`` (task 3256).

        Interleaving reseeds between ``threshold`` stale fallbacks must still
        file exactly one L1: a drip of by-design reseeds cannot mask a genuine
        systematic failure hiding between them.
        """
        harness.config.session_resume = SessionResumeConfig(fallback_storm_threshold=3)
        harness._escalation_queue = self._queue()

        for i in range(3):
            await _drive_session_slot(harness, f'g{i}', self._stale_session(f'uuid-g{i}'))
            await _drive_session_slot(
                harness, f'ir{i}', self._fresh_session(f'uuid-ir{i}'),
                config_dir=tmp_path / f'gone{i}' / 'claude-config-x',
            )

        assert harness._escalation_queue.submit.call_count == 1
        esc = harness._escalation_queue.submit.call_args.args[0]
        assert esc.level == 1
        assert 'resume' in esc.summary.lower()

    async def test_streak_files_one_l1_at_threshold(self, harness: Harness):
        harness.config.session_resume = SessionResumeConfig(fallback_storm_threshold=3)
        harness._escalation_queue = self._queue()

        for i in range(3):
            await _drive_session_slot(harness, f'st{i}', self._stale_session(f'uuid-st{i}'))

        assert harness._escalation_queue.submit.call_count == 1
        esc = harness._escalation_queue.submit.call_args.args[0]
        assert esc.level == 1
        assert 'resume' in esc.summary.lower()

    async def test_dedup_no_second_submit_when_l1_open(self, harness: Harness):
        harness.config.session_resume = SessionResumeConfig(fallback_storm_threshold=3)
        harness._escalation_queue = self._queue()

        for i in range(3):
            await _drive_session_slot(harness, f'st{i}', self._stale_session(f'uuid-st{i}'))
        assert harness._escalation_queue.submit.call_count == 1

        # L1 now open → further fallbacks must NOT re-submit (has_open_l1 dedup).
        harness._escalation_queue.has_open_l1 = MagicMock(return_value=True)
        for i in range(3, 6):
            await _drive_session_slot(harness, f'st{i}', self._stale_session(f'uuid-st{i}'))
        assert harness._escalation_queue.submit.call_count == 1

    async def test_streak_is_consecutive_reset_by_eligible(
        self, harness: Harness, tmp_path: Path
    ):
        harness.config.session_resume = SessionResumeConfig(fallback_storm_threshold=3)
        harness._escalation_queue = self._queue()

        # 2 stale fallbacks (streak=2)...
        for i in range(2):
            await _drive_session_slot(harness, f'a{i}', self._stale_session(f'uuid-a{i}'))
        # ...then an ELIGIBLE resume resets the streak to 0.
        cfg = _make_transcript(tmp_path, 'uuid-ok')
        await _drive_session_slot(
            harness, 'ok1',
            {
                'session_id': 'uuid-ok', 'role': 'implementer',
                'started_at': datetime.now(UTC).isoformat(),
                'resume_count': 0,
            },
            config_dir=cfg,
        )
        # 2 more stale after the reset → streak=2 (<3) → still no L1.
        for i in range(2):
            await _drive_session_slot(harness, f'b{i}', self._stale_session(f'uuid-b{i}'))
        assert harness._escalation_queue.submit.call_count == 0

        # A 3rd consecutive stale AFTER the reset reaches threshold → fires once,
        # proving the streak resumed from 0 (consecutive, not cumulative).
        await _drive_session_slot(harness, 'b2', self._stale_session('uuid-b2'))
        assert harness._escalation_queue.submit.call_count == 1

    async def test_capped_does_not_feed_streak(self, harness: Harness, tmp_path: Path):
        """resume_count-capped degradations are by-design throttling and must
        NOT count toward the storm streak (design decision, task 2774).
        """
        harness.config.session_resume = SessionResumeConfig(
            fallback_storm_threshold=2, max_resumes_per_task=1,
        )
        harness._escalation_queue = self._queue()

        # Three capped dispatches (resume_count=1 == max) — never fire the storm.
        for i in range(3):
            cfg = _make_transcript(tmp_path, f'uuid-cap{i}')
            await _drive_session_slot(
                harness, f'cap{i}',
                {
                    'session_id': f'uuid-cap{i}', 'role': 'implementer',
                    'started_at': datetime.now(UTC).isoformat(),
                    'resume_count': 1,
                },
                config_dir=cfg,
            )
        assert harness._escalation_queue.submit.call_count == 0

    async def test_streak_decays_after_storm_window(self, harness: Harness):
        """A gap of >= storm_window_secs between two genuine fallbacks decays
        the streak to 0, so an isolated drip can never accumulate into a false
        storm (task 3256 — the addendum's second defect).

        The clock is advanced by rewinding the harness's own monotonic stamp,
        NOT by monkeypatching time.monotonic globally: the rewind is
        deterministic and perturbs no unrelated timer.
        """
        harness.config.session_resume = SessionResumeConfig(
            fallback_storm_threshold=3, storm_window_secs=60,
        )
        harness._escalation_queue = self._queue()

        # Two genuine fallbacks inside the window → streak=2.
        for i in range(2):
            await _drive_session_slot(harness, f'd{i}', self._stale_session(f'uuid-d{i}'))
        assert harness._session_resume_fallback_streak == 2

        # ...then the clock jumps past the window before the 3rd arrives.
        # (The assert also pins that a genuine fallback stamped the chain point.)
        assert harness._last_session_resume_fallback_at is not None
        harness._last_session_resume_fallback_at -= 120

        await _drive_session_slot(harness, 'd2', self._stale_session('uuid-d2'))

        # Decayed to 0, then re-incremented — NOT 3, so no L1.
        assert harness._session_resume_fallback_streak == 1
        assert harness._escalation_queue.submit.call_count == 0

    async def test_reseeded_fallbacks_do_not_refresh_the_chain_stamp(
        self, harness: Harness, tmp_path: Path
    ):
        """A reseeded fallback must leave ``_last_session_resume_fallback_at``
        untouched, not just the streak counter (task 3256 amendment).

        The stamp is the chain's comparison point, so refreshing it on a
        by-design reseed would keep the chain alive across an arbitrarily long
        drip: two genuine failures hours apart could then still chain into a
        false storm, re-opening the exact hole the decay window closes. That
        is one refactor away — hoisting ``now = time.monotonic()`` above the
        reason dispatch — and every other test in this class stays green
        under it, so the stamp identity is pinned directly here.
        """
        harness.config.session_resume = SessionResumeConfig(
            fallback_storm_threshold=2, storm_window_secs=60,
        )
        harness._escalation_queue = self._queue()

        # One genuine fallback opens the chain: streak=1, stamp set.
        await _drive_session_slot(harness, 'cs0', self._stale_session('uuid-cs0'))
        assert harness._session_resume_fallback_streak == 1
        stamp = harness._last_session_resume_fallback_at
        assert stamp is not None

        # A drip of reseeds moves NEITHER the counter nor the stamp.
        for i in range(3):
            await _drive_session_slot(
                harness, f'csr{i}', self._fresh_session(f'uuid-csr{i}'),
                config_dir=tmp_path / f'gone{i}' / 'claude-config-x',
            )
        assert harness._session_resume_fallback_streak == 1
        assert harness._last_session_resume_fallback_at == stamp

        # So when the clock passes the window, the NEXT genuine fallback is
        # measured against the first one and decays — reaching 1, not the
        # threshold of 2.
        harness._last_session_resume_fallback_at = stamp - 120
        await _drive_session_slot(harness, 'cs1', self._stale_session('uuid-cs1'))

        assert harness._session_resume_fallback_streak == 1
        assert harness._escalation_queue.submit.call_count == 0

    async def test_streak_survives_within_storm_window(self, harness: Harness):
        """The decay must NOT neuter INV-4: a genuine tight burst still reaches
        the threshold and files exactly one L1 (task 3256).
        """
        harness.config.session_resume = SessionResumeConfig(
            fallback_storm_threshold=3, storm_window_secs=60,
        )
        harness._escalation_queue = self._queue()

        # Three back-to-back fallbacks — no rewind, so all chain inside the window.
        for i in range(3):
            await _drive_session_slot(harness, f'w{i}', self._stale_session(f'uuid-w{i}'))

        assert harness._session_resume_fallback_streak == 3
        assert harness._escalation_queue.submit.call_count == 1
        esc = harness._escalation_queue.submit.call_args.args[0]
        assert esc.level == 1

    async def test_eligible_resume_clears_fallback_timestamp(
        self, harness: Harness, tmp_path: Path
    ):
        """An eligible resume clears the chain's comparison point as well as
        the streak, so the next fallback starts a fresh run rather than
        chaining off a pre-reset stamp (task 3256).
        """
        harness.config.session_resume = SessionResumeConfig(fallback_storm_threshold=3)
        harness._escalation_queue = self._queue()

        await _drive_session_slot(harness, 'ts0', self._stale_session('uuid-ts0'))
        assert harness._session_resume_fallback_streak == 1
        assert harness._last_session_resume_fallback_at is not None

        cfg = _make_transcript(tmp_path, 'uuid-ts-ok')
        await _drive_session_slot(
            harness, 'ts-ok', self._fresh_session('uuid-ts-ok'), config_dir=cfg,
        )

        assert harness._session_resume_fallback_streak == 0
        assert harness._last_session_resume_fallback_at is None

    async def test_no_escalation_queue_never_raises(self, harness: Harness):
        """A bare harness (no escalation queue) must never raise on a fallback
        that would otherwise trip the storm filer (fail-safe totality, I3).
        """
        harness.config.session_resume = SessionResumeConfig(fallback_storm_threshold=1)
        harness._escalation_queue = None

        # threshold=1 → the very first fallback trips the filer, which must
        # early-return on the absent queue rather than raising.
        await _drive_session_slot(harness, 'x1', self._stale_session('uuid-x1'))


@pytest.mark.asyncio
class TestMarkInProgressDoneRecoveryStateCleanup:
    """_mark_in_progress_done must drop ALL THREE parallel recovery stashes for
    the task it terminates, not just two of them (task 3256).

    Why this matters now: ``_recovered_session_config_dirs`` was left behind
    while ``_recovered_plans`` / ``_recovered_sessions`` were popped. BEFORE the
    'reseeded' split that orphan was at worst a harmless mis-corroboration that
    still emitted a LOUD fallback. AFTER it, a stale stash pointing at a
    long-deleted path classifies as ``reason='reseeded'`` and is SILENTLY
    suppressed — the leak turns from benign into a silent-degradation path,
    against the repo's loud-over-silent / no-silent-fail-soft invariants. So
    closing it is a correctness PRECONDITION for the downgrade being sound.
    """

    async def test_mark_in_progress_done_clears_recovered_session_config_dir(
        self, harness: Harness, tmp_path: Path
    ):
        tid = '9256'
        harness._recovered_plans[tid] = _make_plan(3, 5, tid)
        harness._recovered_sessions[tid] = {
            'session_id': 'uuid-leak', 'role': 'implementer',
            'started_at': datetime.now(UTC).isoformat(), 'resume_count': 0,
        }
        harness._recovered_session_config_dirs[tid] = str(
            tmp_path / 'long-deleted' / 'claude-config-x'
        )

        harness.scheduler.mark_done = AsyncMock()
        harness.git_ops.release_lane_for_terminal_task = AsyncMock()
        # Resolve to a path that does NOT exist, so the cleanup_worktree branch
        # is skipped and this stays a pure recovery-state assertion.
        harness._resolve_task_worktree = MagicMock(return_value=tmp_path / 'nope')

        marked = await harness._mark_in_progress_done(
            tid, sha='deadbeef', note='test-leak', reason='found-on-main',
        )

        assert marked is True
        harness.scheduler.mark_done.assert_awaited_once()
        assert tid not in harness._recovered_plans
        assert tid not in harness._recovered_sessions
        assert tid not in harness._recovered_session_config_dirs, (
            'the config-dir stash must be dropped in lockstep with its session '
            "— a surviving orphan would later classify as 'reseeded' and be "
            'silently suppressed instead of surfacing'
        )


class TestCrashRecoveryPromptNote:
    """γ L0-dismissal note (task 2774): adding the escalation auto-dismissal
    warning to the shared crash-recovery resume prompt must NOT flip
    resume_delivers_prompt off its False default (I4 / task-1462 regression
    class), and a crash-recovery resume (resume_delivers_prompt=False) must
    still DELIVER CRASH_RECOVERY_RESUME_PROMPT to the underlying invocation
    (the behavior the signature default protects).
    """

    def test_resume_delivers_prompt_default_stays_false(self):
        """I4 / task-1462 regression guard: the prompt note must NOT flip
        resume_delivers_prompt; its default in invoke_with_cap_retry stays False.
        """
        import inspect

        from shared.cli_invoke import invoke_with_cap_retry

        sig = inspect.signature(invoke_with_cap_retry)
        assert sig.parameters['resume_delivers_prompt'].default is False

    @pytest.mark.asyncio
    async def test_crash_recovery_resume_delivers_recovery_prompt(self):
        """Behavioral complement to the signature guard: a caller-initiated
        resume (resume_session_id pre-set) with the default
        resume_delivers_prompt=False must deliver CRASH_RECOVERY_RESUME_PROMPT
        to the underlying invocation — NOT the real task prompt, which is kept
        only as original_prompt for fresh-fallback (I4 / task-1462 contract).

        Asserts BEHAVIOR (the prompt actually delivered) rather than a
        signature detail, so a refactor that preserved the default but changed
        the delivery path would still be caught.
        """
        from shared.cli_invoke import (
            CRASH_RECOVERY_RESUME_PROMPT,
            AgentResult,
            invoke_with_cap_retry,
        )

        seen: dict = {}

        async def _fake_invoke(**kwargs) -> AgentResult:
            seen.update(kwargs)
            return AgentResult(success=True, output='ok')

        # usage_gate=None → the single-invocation fast path; invoke_fn is the
        # public injection seam, so no subprocess/gate machinery is exercised.
        await invoke_with_cap_retry(
            None, 'lbl',
            prompt='REAL TASK CONTEXT — kept only as original_prompt',
            resume_session_id='sess-crash-1',
            invoke_fn=_fake_invoke,
        )
        assert seen['prompt'] == CRASH_RECOVERY_RESUME_PROMPT
        assert seen['resume_session_id'] == 'sess-crash-1'

    @pytest.mark.asyncio
    async def test_live_continuation_delivers_real_prompt(self):
        """Contrast case proving the fork is real: resume_delivers_prompt=True
        (the steward's live continuation) delivers the caller's REAL prompt,
        NOT the crash-recovery prompt — so the False-default guard above pins a
        genuine behavioral branch, not a no-op that would pass regardless.
        """
        from shared.cli_invoke import (
            CRASH_RECOVERY_RESUME_PROMPT,
            AgentResult,
            invoke_with_cap_retry,
        )

        seen: dict = {}

        async def _fake_invoke(**kwargs) -> AgentResult:
            seen.update(kwargs)
            return AgentResult(success=True, output='ok')

        real = 'REAL CONTINUATION PROMPT the resumed session has not seen'
        await invoke_with_cap_retry(
            None, 'lbl',
            prompt=real,
            resume_session_id='sess-live-1',
            resume_delivers_prompt=True,
            invoke_fn=_fake_invoke,
        )
        assert seen['prompt'] == real
        assert seen['prompt'] != CRASH_RECOVERY_RESUME_PROMPT


@pytest.mark.asyncio
class TestRecoverCrashedTasksC2Namespace:
    """C2 namespace invariant in _recover_crashed_tasks (task 2925, beta).

    PRD: docs/prds/merge-worktree-lifecycle-integrity.md §4 Contract C2.

    The crash-recovery sweep must classify each non-lane worktree_base entry
    by the positive-match namespace rule (classify_worktree_entry) BEFORE the
    no-plan cleanup heuristic: `_merge-*` is REPORTED to the merge reaper
    (never removed by the sweep — the 2026-07-22 task/5326 incident, where a
    persistent `_merge-verify` with a LIVE verify lease was force-removed 21s
    after a verify was dispatched into it), every other `_`/`.`-prefixed
    entry is left to its owner, and only a task-id-shaped entry is subject to
    the existing plan.json/cleanup logic.
    """

    async def test_infra_and_merge_survive_sweep_only_task_shaped_cleaned(
        self, harness: Harness, caplog,
    ):
        base = harness.git_ops.worktree_base

        # ── Merge band: plant `_merge-verify` (persistent) + `_merge-<uuid>`
        # each with a LIVE merge-verify lease, faithfully replaying the 5326
        # timing (a verify holds the lease while the sweep runs). C2 skips
        # them by NAME, but the live lease future-proofs against any impl
        # that also consults the lease.
        merge_verify = base / '_merge-verify'
        merge_verify.mkdir()
        merge_uuid = base / '_merge-ba97f10a'
        merge_uuid.mkdir()

        # ── Infra band: plain infra dirs the sweep must leave to their owner.
        # `.lane-state`/`.task-meta` are the durable-state dirs whose former
        # dedicated per-name skip is now SUBSUMED by C2's `.`-prefix rule —
        # planted here as the regression guard for that removal.
        infra_dirs = {
            name: (base / name)
            for name in (
                '.reseed-trash', '_mainprobe-x', '.lane-state',
                '.task-meta', '_offline-deep',
            )
        }
        for d in infra_dirs.values():
            d.mkdir()

        # ── Task band (positive control): a task-id-shaped PLANLESS dir must
        # still be cleaned. An inert sweep that skips everything fails HERE.
        wt_task = base / '999'
        wt_task.mkdir()

        fd_verify = acquire_merge_verify_flock(lane_lock_path(merge_verify), 5.0)
        fd_uuid = acquire_merge_verify_flock(lane_lock_path(merge_uuid), 5.0)
        assert fd_verify is not None and fd_uuid is not None, (
            'test setup: must be able to acquire both merge-verify leases'
        )
        write_lock_holder_pgid(base, os.getpgrp())
        try:
            with caplog.at_level(logging.INFO, logger='orchestrator.harness'):
                await harness._recover_crashed_tasks()
        finally:
            release_merge_verify_flock(fd_verify)
            release_merge_verify_flock(fd_uuid)
            remove_lock_holder_pgid(base)

        # Positive control: the task-shaped planless dir WAS cleaned — and it
        # is the ONLY cleanup_worktree call (any infra/merge cleanup would
        # push the count past one, the 5326 "Cleaned up worktree _merge-verify"
        # regression).
        harness.git_ops.cleanup_worktree.assert_called_once_with(wt_task, '999')  # type: ignore[attr-defined]

        # Explicit regression guard on the cleaned set: no merge/infra path.
        cleaned_paths = {
            c.args[0] for c in harness.git_ops.cleanup_worktree.call_args_list  # type: ignore[attr-defined]
        }
        protected = {merge_verify, merge_uuid, *infra_dirs.values()}
        assert cleaned_paths.isdisjoint(protected), (
            f'C2 violated — sweep cleaned protected entries: '
            f'{cleaned_paths & protected}'
        )
        # All merge/infra dirs still on disk.
        for d in protected:
            assert d.exists(), f'{d.name} must survive the recovery sweep'

        # Skip disposition OBSERVED (not silence): every protected entry is
        # named in an explicit INFO record, per PRD §1 (operators must see a
        # skip/report line instead of the 5326 "Cleaned up worktree
        # _merge-verify" signature). We assert the STABLE structured signal —
        # an INFO record mentions the entry name — NOT the exact human-facing
        # prose of the disposition lines, which may be reworded without any
        # change to the disposition. (The classifier's task/merge/infra
        # verdict is unit-pinned in test_worktree_namespace_c2.py.)
        info_messages = [
            r.getMessage() for r in caplog.records if r.levelno >= logging.INFO
        ]
        for name in ('_merge-verify', '_merge-ba97f10a', '_mainprobe-x',
                     '_offline-deep', '.reseed-trash', '.lane-state',
                     '.task-meta'):
            assert any(name in m for m in info_messages), (
                f'missing explicit skip/report line naming {name}'
            )


# ── archive_available instrumentation helpers (task 3727) ────────────────────
# A lane-encoded project dir DELIBERATELY unlike any config dir these tests
# build, so a passing lookup proves the encoded-cwd component was globbed
# rather than reconstructed from the caller's own cwd (I-B).
_ARCHIVE_ENC = '-home-leo-src-dark-factory--worktrees-9999'


def _make_archive(project_root: Path, task_id: str, session_id: str) -> Path:
    """Lay down one archived transcript at the real producer layout.

    ``<project_root>/data/orchestrator/agent-transcripts/<task_id>/<enc>/
    <session_id>.jsonl.gz`` — the path shared.transcript_archive._archive_one
    writes and durable_archive_path globs.
    """
    dest = (
        project_root
        / 'data/orchestrator/agent-transcripts'
        / task_id
        / _ARCHIVE_ENC
        / f'{session_id}.jsonl.gz'
    )
    dest.parent.mkdir(parents=True, exist_ok=True)
    dest.write_bytes(b'archived-transcript-bytes')
    return dest


@pytest.mark.asyncio
class TestSessionResumeArchiveAvailable:
    """archive_available on session_resume_fallback (task 3727, PRD §8).

    Every fallback emit reports whether that session was actually RECOVERABLE
    from the durable transcript archive — the measurement task 3619 will move
    and leaf δ may later gate on. It is instrumentation ONLY (D8 / INV-3
    instrument-before-acting): it must never change what dispatches, so every
    assertion here also pins that the reason, the resume decision and the
    storm streak are exactly what they were before the field existed.
    """

    async def test_no_transcript_reports_archive_present(
        self, harness: Harness, tmp_path: Path
    ):
        """Archive PRESENT under a foreign lane → archive_available is True.

        The recoverable population: the transcript is gone from the live
        config dir but survives in the durable archive, under an encoded-cwd
        dir belonging to no lane this test uses.
        """
        session = {
            'session_id': 'uuid-arch-yes',
            'role': 'implementer',
            'started_at': datetime.now(UTC).isoformat(),
            'resume_count': 0,
        }
        empty_cfg = tmp_path / 'claude-config-empty-yes'
        (empty_cfg / 'projects').mkdir(parents=True)
        harness.config.session_resume = SessionResumeConfig()
        harness.config.transcript_archive = TranscriptArchiveConfig()
        _make_archive(harness.config.project_root, 'ar1', 'uuid-arch-yes')

        resume_id = await _drive_session_slot(
            harness, 'ar1', session, config_dir=empty_cfg
        )

        # D8: the resume decision and the reason are untouched.
        assert resume_id is None
        emits = _session_resume_emits(harness)
        assert len(emits) == 1
        et, kwargs = emits[0]
        assert et == EventType.session_resume_fallback
        assert kwargs['data']['reason'] == 'no_transcript'
        # `is True`, not truthy: the field must be a real JSON bool for
        # json_extract(data, '$.archive_available') to be queryable in runs.db.
        assert kwargs['data']['archive_available'] is True
        # D8: a genuine corroboration failure still feeds the storm streak.
        assert harness._session_resume_fallback_streak == 1

    async def test_no_transcript_reports_archive_absent(
        self, harness: Harness, tmp_path: Path
    ):
        """Empty archive root → archive_available is False."""
        session = {
            'session_id': 'uuid-arch-no',
            'role': 'implementer',
            'started_at': datetime.now(UTC).isoformat(),
            'resume_count': 0,
        }
        empty_cfg = tmp_path / 'claude-config-empty-no'
        (empty_cfg / 'projects').mkdir(parents=True)
        harness.config.session_resume = SessionResumeConfig()
        harness.config.transcript_archive = TranscriptArchiveConfig()

        resume_id = await _drive_session_slot(
            harness, 'ar2', session, config_dir=empty_cfg
        )

        assert resume_id is None
        emits = _session_resume_emits(harness)
        assert len(emits) == 1
        et, kwargs = emits[0]
        assert et == EventType.session_resume_fallback
        assert kwargs['data']['reason'] == 'no_transcript'
        assert kwargs['data']['archive_available'] is False
        assert harness._session_resume_fallback_streak == 1

    async def test_stale_also_carries_the_field(
        self, harness: Harness, tmp_path: Path
    ):
        """reason == 'stale' carries it too — it rides the BRANCH, not one reason."""
        real = SessionResumeConfig()
        stale_at = datetime.now(UTC) - timedelta(seconds=2 * real.freshness_window_secs)
        session = {
            'session_id': 'uuid-arch-stale',
            'role': 'implementer',
            'started_at': stale_at.isoformat(),
            'resume_count': 0,
        }
        cfg = _make_transcript(tmp_path, 'uuid-arch-stale')
        harness.config.session_resume = SessionResumeConfig()
        harness.config.transcript_archive = TranscriptArchiveConfig()
        _make_archive(harness.config.project_root, 'ar3', 'uuid-arch-stale')

        resume_id = await _drive_session_slot(harness, 'ar3', session, config_dir=cfg)

        assert resume_id is None
        emits = _session_resume_emits(harness)
        assert len(emits) == 1
        et, kwargs = emits[0]
        assert et == EventType.session_resume_fallback
        assert kwargs['data']['reason'] == 'stale'
        assert kwargs['data']['archive_available'] is True
        assert harness._session_resume_fallback_streak == 1
