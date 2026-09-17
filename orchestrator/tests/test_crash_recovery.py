"""Tests for crash recovery — surviving worktree detection and plan injection."""

import json
import logging
import os
import re
import shutil
from collections.abc import Callable
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


def _reasons_for(
    harness: Harness,
    session: object,
    config_dir: str | None,
    *,
    archive_available: bool = False,
) -> frozenset[str]:
    """Call the eligibility predicate directly, off the ``_run_slot`` path.

    The ONE place this suite names ``_session_resume_reasons``. The predicate is
    module-internal, so every case routing through a single seam keeps the
    coupling to that name at one line rather than one per case — a rename costs
    an edit here instead of twenty.

    ``archive_available`` defaults to False, the pre-δ answer, so a case that
    says nothing about the archive reads as one whose outcome does not turn on
    it; the δ cases below pass it explicitly.
    """
    return harness._session_resume_reasons(
        session, config_dir, archive_available=archive_available
    )


# ── Storm-streak state, read through ONE seam each ───────────────────────────
# The `_reasons_for` convention above, applied to the streak. Every row that
# touches this state goes through these, so the suite names each private field
# ONCE: a rename costs one edit instead of forty, and the coupling a reader has
# to hold in their head is one line rather than one per assertion. That is also
# what the merge-lane ratchet's `private_reads` measure is counting.


def _streak(harness: Harness) -> int:
    """The current run of genuine session-resume failures."""
    return harness._session_resume_fallback_streak


def _chain_stamp(harness: Harness) -> float | None:
    """The chain's monotonic comparison point; None means no run in progress."""
    return harness._last_session_resume_fallback_at


def _set_chain_stamp(harness: Harness, value: float | None) -> None:
    return setattr(harness, '_last_session_resume_fallback_at', value)


def _rewind_chain(harness: Harness, secs: float) -> None:
    """Advance the clock by *secs* against the chain's own stamp.

    Rewinding the harness's stamp, never monkeypatching ``time.monotonic``: the
    rewind is deterministic and perturbs no unrelated timer.
    """
    stamp = _chain_stamp(harness)
    assert stamp is not None, 'no chain in progress — nothing to rewind'
    _set_chain_stamp(harness, stamp - secs)


def _recorded_failures(harness: Harness) -> list:
    """The eligible-but-FAILED resumes backing the current run."""
    return list(harness._eligible_but_failed_resumes)


async def _drive_session_slot(
    harness: Harness,
    task_id: str,
    session: dict,
    *,
    config_dir: Path | str | None = None,
    workflow_reports: Callable[[object], None] | None = None,
):
    """Populate recovered-session state and run ``_run_slot`` with
    ``build_workflow`` patched; return the ``resume_session_id`` kwarg it saw.

    The FULL kwarg set is stashed on the harness as
    ``_last_build_workflow_kwargs`` for the rows that need a different one (ε's
    sink wiring). Stashed rather than returned because ~20 existing rows read
    the return value as a session id, and per-harness rather than per-module
    because the fixture is function-scoped, so nothing leaks between rows.

    ``workflow_reports`` COMPOSES the guard with the arm seam (task ε/3733).
    The stand-in workflow calls it, from inside its ``run()``, with the
    ``resume_outcome_sink`` kwarg ``_run_slot`` actually handed
    ``build_workflow`` — the object and the moment production's ``_invoke``
    reports through. That is what puts the REAL eligibility pass and the REAL
    sink on ONE streak in a single dispatch, in production's own order; the
    rows that drive either seam alone cannot observe how they interleave, and
    the reset-on-eligible defect lived exactly in that gap. Left None, nothing
    reports and every existing row is byte-identical.
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
        if workflow_reports is not None:
            async def _run(*_args, **_kwargs):
                workflow_reports(
                    MockWorkflow.call_args.kwargs['resume_outcome_sink']
                )
                return MagicMock(value='done')

            mock_wf.run = _run
        await harness._run_slot(assignment, sem)
        harness._last_build_workflow_kwargs = MockWorkflow.call_args.kwargs  # type: ignore[attr-defined]
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



@pytest.mark.asyncio
class TestAdoptNonDictSidecar:
    """A sidecar that PARSED as JSON but is not an OBJECT must be rejected at
    ADOPTION — the reachability boundary — not merely survived downstream.

    ``_adopt_recovered_session`` is the SOLE writer of ``_recovered_sessions``
    (harness.py::Harness._adopt_recovered_session), so it is the one place that
    can make the ``_run_slot`` guard's ``recovered_session.get('session_id')``
    sound. Today it does ``json.loads`` with NO dict validation and stores the
    payload verbatim, so a truncated/garbage ``agent_session.json`` holding
    ``[]``, ``"str"`` or ``7`` reaches the dispatch path intact.

    Two live consequences, both contradicting docstrings already in the tree:
      - the no-plan-lane call site does ``session_data.get('task_id')`` with no
        enclosing try/except, so a non-dict raises straight out of
        ``_recover_crashed_tasks`` — a method whose own docstring promises
        "Never raises" and promises it reads the sidecar "as a RAW dict". That
        one is PRE-EXISTING, not introduced by task 3728;
      - ``_run_slot``'s guard builds ``resume_event_data`` from
        ``recovered_session.get(...)`` immediately after the predicate call, so
        the same payload raises there too — which is why the method-level guard
        alone does not restore the guard's observable "never a stall, never a
        scheduler-visible error" contract.
    """

    @pytest.mark.parametrize(
        'payload', [[], 'str', 7], ids=['list', 'str', 'int'],
    )
    @pytest.mark.parametrize(
        'task_id', ['35', None], ids=['keyed-by-task-id', 'no-plan-lane'],
    )
    async def test_non_dict_sidecar_is_not_adopted(
        self, harness: Harness, caplog, payload, task_id
    ):
        """BOTH keying paths reject the payload, adopt nothing, and say why.

        Parametrized over the two ``task_id`` arities because they are the two
        branches of the single ``key = task_id if task_id is not None else
        session_data.get('task_id')`` line: only a check placed BEFORE it
        covers both, and the ``None`` arity is the one that raises today.
        """
        wt = _setup_worktree(harness.git_ops.worktree_base, '35')
        task_dir = wt / '.task'
        task_dir.mkdir(exist_ok=True)
        (task_dir / 'agent_session.json').write_text(json.dumps(payload))

        with caplog.at_level(logging.WARNING, logger='orchestrator.harness'):
            adopted = harness._adopt_recovered_session(wt, task_id)

        assert adopted is None
        assert harness._recovered_sessions == {}
        assert harness._recovered_session_config_dirs == {}
        assert any(
            type(payload).__name__ in rec.getMessage() for rec in caplog.records
        ), (
            f'expected a warning naming the offending type '
            f'{type(payload).__name__!r}; got: {caplog.text!r}'
        )

    async def test_recover_crashed_tasks_survives_a_non_dict_sidecar(
        self, harness: Harness
    ):
        """The no-plan lane drives the live unguarded ``.get`` through the real
        ``_recover_crashed_tasks``, whose docstring promises it never raises.
        """
        wt = harness.git_ops.worktree_base / '90'
        task_dir = wt / '.task'
        task_dir.mkdir(parents=True)
        (task_dir / 'agent_session.json').write_text(json.dumps(['a']))

        await harness._recover_crashed_tasks()  # must not raise

        assert harness._recovered_sessions == {}
        assert '90' not in harness._preserved_worktrees

    async def test_non_dict_sidecar_degrades_to_a_fresh_dispatch(
        self, harness: Harness
    ):
        """END-TO-END (I3): a corrupt sidecar on disk yields a FRESH dispatch —
        ``_run_slot`` completes, and hands ``resume_session_id=None`` to the
        workflow.

        Driven through the REAL adoption path rather than by injecting the
        non-dict straight into ``_recovered_sessions``, because adoption is the
        sole writer of that map and therefore the only place a fix can make
        this assertion hold; injecting past it would pin a guard that, by the
        deliberate two-guard design, does not exist. RED today for exactly the
        reason this class exists: the payload IS adopted, so the guard's
        ``recovered_session.get('session_id')`` sees a list.
        """
        wt = _setup_worktree(harness.git_ops.worktree_base, '91')
        task_dir = wt / '.task'
        task_dir.mkdir(exist_ok=True)
        (task_dir / 'agent_session.json').write_text(json.dumps(['a']))
        harness._adopt_recovered_session(wt, '91')

        assignment = MagicMock()
        assignment.task_id = '91'
        assignment.task = {'title': 'task 91'}
        sem = MagicMock()
        sem.release = MagicMock()

        with patch('orchestrator.harness.build_workflow') as MockWorkflow:
            mock_wf = AsyncMock()
            mock_wf.run.return_value = MagicMock(value='done')
            mock_wf.metrics = MagicMock(
                total_cost_usd=0.0, total_duration_ms=0, agent_invocations=0,
            )
            MockWorkflow.return_value = mock_wf
            await harness._run_slot(assignment, sem)  # must not raise

        assert MockWorkflow.call_args.kwargs['resume_session_id'] is None


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


class TestSessionResumeReasons:
    """The composite eligibility predicate itself (task 3728, β / D5).

    ``_session_resume_reasons`` returns a ``frozenset[str]`` of EVERY reason a
    recovered session is ineligible; the EMPTY set means eligible. Called
    directly here (no ``_run_slot``) because the point under test is the
    predicate's own reporting fidelity, not the caller's routing.

    The predecessor ``_session_resume_eligible`` returned ``(bool, reason)``
    on the FIRST matching branch, so a session that was both stale AND
    uncorroborated reported only ``stale`` — sending an operator to check NTP
    for a session whose transcript had also vanished. Case (a) below is that
    hidden co-occurrence, asserted directly.
    """

    def test_stale_and_no_transcript_co_occur(self, harness: Harness, tmp_path: Path):
        """(a) HEADLINE — an AGED sidecar whose transcript is ALSO gone reports
        BOTH reasons, not just the first one branch order happened to reach.
        """
        cfg = SessionResumeConfig()
        harness.config.session_resume = cfg
        session = {
            'session_id': 'uuid-both',
            'role': 'implementer',
            'started_at': (
                datetime.now(UTC) - timedelta(seconds=2 * cfg.freshness_window_secs)
            ).isoformat(),
            'resume_count': 0,
        }
        # The config dir SURVIVES (so this is 'no_transcript', not 'reseeded')
        # but holds no transcript for this session — mirrors
        # test_transcript_absent_falls_back_no_transcript's fixture.
        empty_cfg = tmp_path / 'claude-config-both'
        (empty_cfg / 'projects').mkdir(parents=True)

        reasons = _reasons_for(harness, session, str(empty_cfg))

        assert reasons == frozenset({'stale', 'no_transcript'})

    def test_eligible_is_the_empty_set(self, harness: Harness, tmp_path: Path):
        """(b) Fresh + under cap + corroborated → NO reasons. Empty == eligible;
        there is no 'eligible' pseudo-reason to disagree with a bool.
        """
        harness.config.session_resume = SessionResumeConfig()
        session = {
            'session_id': 'uuid-ok',
            'role': 'implementer',
            'started_at': datetime.now(UTC).isoformat(),
            'resume_count': 0,
        }
        cfg_dir = _make_transcript(tmp_path, 'uuid-ok')

        reasons = _reasons_for(harness, session, str(cfg_dir))

        assert reasons == frozenset()
        assert not reasons  # the eligibility predicate itself

    def test_three_reasons_co_occur(self, harness: Harness, tmp_path: Path):
        """(c) Aged AND capped AND provably-wiped store → all three."""
        harness.config.session_resume = SessionResumeConfig(max_resumes_per_task=3)
        session = {
            'session_id': 'uuid-three',
            'role': 'implementer',
            'started_at': (datetime.now(UTC) - timedelta(days=2)).isoformat(),
            'resume_count': 3,
        }
        gone = tmp_path / 'gone-three' / 'claude-config-x'
        assert not gone.exists()  # provably ENOENT → the 'reseeded' arm

        reasons = _reasons_for(harness, session, str(gone))

        assert reasons == frozenset({'stale', 'capped', 'reseeded'})

    def test_disabled_short_circuits_exactly(self, harness: Harness, tmp_path: Path):
        """(d) The B6 kill switch is a property of the FEATURE, not the session:
        it reports 'disabled' ALONE over the very inputs that would otherwise
        yield {'stale','no_transcript'} (case a). Exact equality, so the
        short-circuit exemption cannot silently widen to other predicates.
        """
        real = SessionResumeConfig()
        harness.config.session_resume = SessionResumeConfig(enabled=False)
        session = {
            'session_id': 'uuid-dis',
            'role': 'implementer',
            'started_at': (
                datetime.now(UTC) - timedelta(seconds=2 * real.freshness_window_secs)
            ).isoformat(),
            'resume_count': 0,
        }
        empty_cfg = tmp_path / 'claude-config-dis'
        (empty_cfg / 'projects').mkdir(parents=True)

        reasons = _reasons_for(harness, session, str(empty_cfg))

        assert reasons == frozenset({'disabled'})

    def test_unparseable_or_absent_started_at_is_stale_only(
        self, harness: Harness, tmp_path: Path
    ):
        """(e) Fail-safe freshness still degrades to 'stale' — and now WITHOUT
        swallowing the corroboration leg: over a CORROBORATED dir the set is
        exactly {'stale'}, proving the parse failure no longer returns early.
        """
        harness.config.session_resume = SessionResumeConfig()

        cfg1 = _make_transcript(tmp_path, 'uuid-bad')
        r1 = _reasons_for(
            harness,
            {'session_id': 'uuid-bad', 'role': 'r',
             'started_at': 'not-a-date', 'resume_count': 0},
            str(cfg1),
        )
        assert 'stale' in r1
        assert 'no_transcript' not in r1

        cfg2 = _make_transcript(tmp_path, 'uuid-bad2')
        r2 = _reasons_for(
            harness,
            {'session_id': 'uuid-bad2', 'role': 'r', 'resume_count': 0},  # no started_at
            str(cfg2),
        )
        assert 'stale' in r2
        assert 'no_transcript' not in r2

    def test_missing_corroboration_inputs_are_no_transcript(self, harness: Harness):
        """(f1) No stashed config_dir, and no session_id, both stay 'no_transcript'
        — an adopted session with nothing to corroborate against is pathological
        and must stay LOUD rather than land in the silent 'reseeded' arm.
        """
        harness.config.session_resume = SessionResumeConfig()
        fresh = datetime.now(UTC).isoformat()

        no_dir = _reasons_for(
            harness,
            {'session_id': 'uuid-nocfg', 'role': 'r',
             'started_at': fresh, 'resume_count': 0},
            None,
        )
        assert 'no_transcript' in no_dir

        no_sid = _reasons_for(
            harness,
            {'session_id': None, 'role': 'r', 'started_at': fresh, 'resume_count': 0},
            '/some/where',
        )
        assert 'no_transcript' in no_sid

    def test_unreadable_config_dir_stays_no_transcript_and_never_raises(
        self, harness: Harness, tmp_path: Path, monkeypatch: pytest.MonkeyPatch
    ):
        """(f2) I3 totality survives the rewrite: a PRESENT-but-UNREADABLE config
        dir is a filesystem fault, not a lane wipe, so it stays a loud
        'no_transcript' and the call does not raise.

        The EACCES is INJECTED rather than produced with ``chmod(0o000)``, for
        the reason :meth:`TestSessionResumeGuard.
        test_unreadable_config_dir_stays_no_transcript` records: a root process
        can stat through a 0o000 parent, which would make the row vacuous.
        """
        blocked = tmp_path / 'unreadable-r' / 'claude-config-x'
        blocked.mkdir(parents=True)
        real_stat = Path.stat

        def fake_stat(self: Path, *args, **kwargs):
            if self == blocked:
                raise PermissionError(13, 'Permission denied')
            return real_stat(self, *args, **kwargs)

        monkeypatch.setattr(Path, 'stat', fake_stat)
        harness.config.session_resume = SessionResumeConfig()

        reasons = _reasons_for(
            harness,
            {'session_id': 'uuid-eacces', 'role': 'r',
             'started_at': datetime.now(UTC).isoformat(), 'resume_count': 0},
            str(blocked),
        )

        assert 'no_transcript' in reasons
        assert 'reseeded' not in reasons

    @pytest.mark.parametrize(
        'bad_session',
        [['a'], 'str', 7, 3.5, True, [], ()],
        ids=['list', 'str', 'int', 'float', 'bool', 'empty-list', 'tuple'],
    )
    def test_non_dict_session_is_total_and_stays_by_design(
        self, harness: Harness, tmp_path: Path, bad_session
    ):
        """(f3) I3 totality for a sidecar that PARSED as JSON but is not an
        OBJECT — a truncated/garbage ``agent_session.json`` holding ``[]``,
        ``"str"`` or ``7``.

        The predecessor ``_session_resume_eligible`` was total for this input
        only by ACCIDENT of first-match ordering: ``session['started_at']``
        raised ``TypeError`` straight into the freshness leg's own ``except``,
        which returned ``(False, 'stale')`` before any ``.get`` ran. A
        composite predicate does not return early by construction, so it
        cannot inherit that accident — it falls through to
        ``session.get('resume_count', 0)`` and raises ``AttributeError``,
        contradicting the docstring's "this method NEVER raises".

        Four properties, because "does not raise" alone would be satisfied by
        a spurious eligibility:
          - NON-EMPTY, so the caller degrades to a fresh dispatch rather than
            resuming a corrupt session;
          - a ``frozenset`` of ``str``, the declared return shape;
          - a SUBSET of the by-design vocabulary, so one corrupt sidecar can
            never feed the INV-4 storm streak and page an operator.
        """
        from orchestrator.harness import _BY_DESIGN_SESSION_RESUME_REASONS

        harness.config.session_resume = SessionResumeConfig()

        reasons = _reasons_for(
            harness,
            bad_session, str(tmp_path)
        )

        assert isinstance(reasons, frozenset)
        assert all(isinstance(r, str) for r in reasons)
        assert reasons, (
            'a non-dict session must be INELIGIBLE (non-empty reasons): the '
            'empty set is the eligibility predicate, so returning it here '
            'would resume a corrupt sidecar'
        )
        assert reasons <= _BY_DESIGN_SESSION_RESUME_REASONS, (
            f'a corrupt sidecar reported {sorted(reasons)}, which escapes '
            '_BY_DESIGN_SESSION_RESUME_REASONS and would therefore feed the '
            'fallback-storm streak — one garbage file must not page an operator'
        )

    def test_disabled_still_wins_over_a_non_dict_session(
        self, harness: Harness, tmp_path: Path
    ):
        """(f4) The kill switch short-circuits BEFORE the non-dict degradation.

        Pins the guard's placement — after ``enabled``, before the freshness
        leg — executably, so D2's "'disabled' is returned ALONE" exemption
        cannot silently widen into a mixed set when the session is garbage.
        """
        harness.config.session_resume = SessionResumeConfig(enabled=False)

        reasons = _reasons_for(
            harness,
            ['a'], str(tmp_path)
        )

        assert reasons == frozenset({'disabled'})

    # ── δ (task 3730 / D2): the durable archive as a SECOND source of
    #    reachability, and freshness demoted to the no-archive case ─────────

    def test_aged_but_archived_and_config_dir_less_is_eligible(
        self, harness: Harness
    ):
        """(a) B8 HEADLINE — the real crash-recovery shape is now ELIGIBLE.

        An aged sidecar, NO live config dir at all, but the session is still
        in the durable transcript archive: the empty set. Neither 'stale' nor
        'no_transcript'.

        This is the shape production actually reaches, which is the whole
        point. ``run()``'s finally executes an unconditional
        ``cleanup_config_dir`` teardown while ``session_preserved`` keeps the
        sidecar, so on every crash-recovery path the config dir is GONE and
        ``_adopt_recovered_session``'s glob hands the guard ``config_dir=None``.
        Before δ that combination was uncorroborable by construction, which is
        why ~91% of post-3578 fallbacks (92 of 101, measured 2026-09-04) had a
        recoverable archive the predicate never consulted.

        Reachability outranks freshness: an archived transcript does not decay
        with wall-clock, so "how old is it" is the wrong question for a
        session that is still reachable. The absolute backstop (D3) is what
        keeps that from meaning "no age limit at all".
        """
        cfg = SessionResumeConfig()
        harness.config.session_resume = cfg
        session = {
            'session_id': 'uuid-archived',
            'role': 'implementer',
            'started_at': (
                datetime.now(UTC) - timedelta(seconds=2 * cfg.freshness_window_secs)
            ).isoformat(),
            'resume_count': 0,
        }
        assert 2 * cfg.freshness_window_secs < cfg.absolute_resume_age_secs, (
            'this row must sit BETWEEN the two thresholds, or it is testing '
            'the backstop instead of the freshness demotion'
        )

        reasons = _reasons_for(harness, session, None, archive_available=True)

        assert reasons == frozenset()
        assert not reasons  # the eligibility predicate itself

    def test_the_archive_is_the_only_thing_that_changed_the_answer(
        self, harness: Harness
    ):
        """(b) THE CONTROL for (a) — same session, archive_available=False,
        and the answer is exactly today's {'stale', 'no_transcript'}.

        Run against byte-identical inputs so the archive is demonstrably the
        only variable. Without this, (a) would be consistent with δ having
        loosened something else.
        """
        cfg = SessionResumeConfig()
        harness.config.session_resume = cfg
        session = {
            'session_id': 'uuid-archived',
            'role': 'implementer',
            'started_at': (
                datetime.now(UTC) - timedelta(seconds=2 * cfg.freshness_window_secs)
            ).isoformat(),
            'resume_count': 0,
        }

        # Explicit, though False is the helper's default: this row IS the
        # archive variable held at False, so spelling it makes the contrast
        # with (a) legible without cross-referencing the helper.
        reasons = _reasons_for(harness, session, None, archive_available=False)

        assert reasons == frozenset({'stale', 'no_transcript'})

    def test_fresh_and_archived_without_a_live_transcript_is_eligible(
        self, harness: Harness
    ):
        """(c) Reachability ALONE suffices — the age was never the objection.

        A FRESH sidecar with no live transcript is ineligible today purely on
        corroboration. With an archive it is reachable, so it is eligible;
        this separates the corroboration change from the freshness change,
        which (a) exercises together.
        """
        harness.config.session_resume = SessionResumeConfig()
        session = {
            'session_id': 'uuid-fresh-arch',
            'role': 'implementer',
            'started_at': datetime.now(UTC).isoformat(),
            'resume_count': 0,
        }

        reasons = _reasons_for(harness, session, None, archive_available=True)

        assert reasons == frozenset()

    def test_a_live_transcript_still_corroborates_on_its_own(
        self, harness: Harness, tmp_path: Path
    ):
        """(d) The archive is an ADDITIONAL source of reachability, not a
        replacement: a live transcript still corroborates with no archive.

        Pins that δ WIDENED the corroboration leg rather than moving it onto
        the archive — a rewrite that made the archive the only accepted source
        would leave every warm-lane resume (the population γ shipped for)
        newly ineligible, and nothing else in this class would notice.
        """
        harness.config.session_resume = SessionResumeConfig()
        session = {
            'session_id': 'uuid-live-only',
            'role': 'implementer',
            'started_at': datetime.now(UTC).isoformat(),
            'resume_count': 0,
        }
        cfg_dir = _make_transcript(tmp_path, 'uuid-live-only')

        reasons = _reasons_for(
            harness,
            session, str(cfg_dir)
        )

        assert reasons == frozenset()


    # ── δ (task 3730 / D3): the absolute backstop — "archive outranks age"
    #    must not become "no age limit at all" ────────────────────────────

    def _aged(self, cfg: SessionResumeConfig, age_secs: float) -> dict:
        """A sidecar back-dated *age_secs*, driven off the config knobs."""
        return {
            'session_id': 'uuid-aged',
            'role': 'implementer',
            'started_at': (
                datetime.now(UTC) - timedelta(seconds=age_secs)
            ).isoformat(),
            'resume_count': 0,
        }

    def test_aged_out_rejects_even_an_archived_session(self, harness: Harness):
        """(a) B9 HEADLINE — reachability outranks FRESHNESS, not the BACKSTOP.

        Past absolute_resume_age_secs a session is rejected however reachable
        it is, and reports 'aged_out'. Without this leg D2 would read as "an
        archive exempts a session from age entirely", and a sidecar surviving
        an arbitrarily long outage would resume into a world that had moved on.

        Back-dated off the CONFIG FIELD rather than a literal, so a re-tuned
        bound re-tunes this row with it.
        """
        cfg = SessionResumeConfig()
        harness.config.session_resume = cfg
        session = self._aged(cfg, 2 * cfg.absolute_resume_age_secs)

        reasons = _reasons_for(harness, session, None, archive_available=True)

        assert 'aged_out' in reasons
        assert reasons  # ineligible, whatever else co-occurs

    def test_aged_out_boundary_is_closed_at_the_bound(self, harness: Harness):
        """(b) `>=`, matching the freshness leg's existing convention.

        AT the bound is rejected; comfortably below it, with an archive, is
        eligible — which also proves this row is exercising the BACKSTOP and
        not the freshness window it sits above.
        """
        cfg = SessionResumeConfig()
        harness.config.session_resume = cfg

        at_bound = _reasons_for(
            harness,
            self._aged(cfg, cfg.absolute_resume_age_secs),
            None,
            archive_available=True,
        )
        assert 'aged_out' in at_bound

        below = _reasons_for(
            harness,
            self._aged(cfg, cfg.absolute_resume_age_secs - 3600),
            None,
            archive_available=True,
        )
        assert below == frozenset()

    def test_aged_out_co_occurs_rather_than_replacing(self, harness: Harness):
        """(c) D5 survives the new token: aged out AND capped reports BOTH.

        The predicate accumulates; a new leg that returned early would undo
        exactly the co-occurrence reporting task 3728 exists to provide.
        """
        cfg = SessionResumeConfig()
        harness.config.session_resume = cfg
        session = self._aged(cfg, 2 * cfg.absolute_resume_age_secs)
        session['resume_count'] = cfg.max_resumes_per_task

        reasons = _reasons_for(harness, session, None, archive_available=True)

        assert 'aged_out' in reasons
        assert 'capped' in reasons

    @pytest.mark.parametrize(
        'started_at',
        ['not-a-date', None, 12345, ['2026-01-01']],
        ids=['unparseable', 'none', 'int', 'list'],
    )
    def test_an_undateable_session_is_never_laundered_by_the_archive(
        self, harness: Harness, started_at
    ):
        """(d) THE FAIL-SAFE CARVE-OUT — an archive cannot redeem an UNDATEABLE
        sidecar.

        D2's argument for suppressing 'stale' is that an archived transcript
        does not decay with wall-clock, so age is the wrong question. That
        applies only to a session whose age we KNOW. With started_at missing,
        unparseable or the wrong type the age is unknown, so nothing bounds it
        — and the D3 backstop cannot be evaluated either, because there is no
        age to compare against. Suppressing 'stale' here would make an
        undateable sidecar FULLY ELIGIBLE on the strength of an archive: a
        fail-OPEN regression against the I3 contract.

        The sharpest way to get δ wrong is to write the suppression as a
        single `if not archive_available` around the whole freshness leg,
        which passes every other row in this class. This is the row that
        catches it.
        """
        harness.config.session_resume = SessionResumeConfig()
        session = {'session_id': 'uuid-undateable', 'role': 'r',
                   'resume_count': 0}
        if started_at is not None:
            session['started_at'] = started_at

        reasons = _reasons_for(harness, session, None, archive_available=True)

        assert 'stale' in reasons, (
            'an undateable sidecar must stay ineligible however reachable it '
            'is: its age cannot be bounded and the absolute backstop cannot '
            'be evaluated, so nothing is left to stop it resuming'
        )
        # ...and the backstop is NOT claimed, because nothing was compared.
        assert 'aged_out' not in reasons

    def test_no_archive_reports_both_thresholds_separately(self, harness: Harness):
        """(e) On the no-archive path an aged-out session reports 'stale' AND
        'aged_out', so a runs.db census can still tell the two apart.

        They answer different operator questions and are actioned differently:
        'stale' is "old, with no archive to redeem it" (worth asking why the
        archive is missing — the U2 population), 'aged_out' is "old past the
        point resuming is safe regardless of reachability" (the backstop
        working). Collapsing them into one token would destroy the
        co-occurrence census D5 built the reason SET to enable.
        """
        cfg = SessionResumeConfig()
        harness.config.session_resume = cfg

        reasons = _reasons_for(
            harness,
            self._aged(cfg, 2 * cfg.absolute_resume_age_secs),
            None,
        )

        assert 'stale' in reasons
        assert 'aged_out' in reasons


@pytest.mark.asyncio
class TestSessionResumeGuard:
    """γ eligibility guard in _run_slot (task 2774): an ineligible recovered
    session degrades to fresh dispatch (resume_session_id=None) with a
    reason-carrying event (B4/B5/B7); an eligible one is injected and emits a
    session_resume event. The kill switch (enabled=False) degrades silently
    with no event (B6). The guard is fail-safe (I3) — every ineligible path
    is no-worse than today's fresh dispatch.

    Since task 3730 (δ) the guard also HOISTS the durable-archive lookup and
    feeds it to the predicate as an eligibility input, so two rows here drive
    the archive-mediated outcomes end to end: an aged, config-dir-less,
    archive-backed session is now ADOPTED (D2), and the same session past the
    absolute bound still falls back with 'aged_out' (D3). Every pre-δ row
    above is unchanged in meaning: none of them seeds an archive, so the
    hoisted lookup answers False and they exercise exactly the no-archive
    world they always did.
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
        assert kwargs['data']['reasons'] == ['stale']

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
            assert kwargs['data']['reasons'] == ['stale']

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
        assert kwargs['data']['reasons'] == ['no_transcript']

    async def test_aged_and_uncorroborated_reports_both_reasons(
        self, harness: Harness, tmp_path: Path
    ):
        """The user-observable signal of task 3728, end to end through the
        guard: a sidecar that is BOTH too old AND missing its transcript emits
        ONE fallback carrying BOTH reasons.

        Before the composite rewrite this event said ``reason='stale'`` and
        nothing else, because freshness was checked first — so an operator
        triaging it went to check host clock skew for a session that had also
        lost its transcript, and the second cause was not merely unranked but
        absent from the wire entirely.

        The list is SORTED (alphabetical, hence no_transcript before stale) so
        ``json_extract(data,'$.reasons')`` is a stable composite group key, and
        it is a real ``list`` — a set/frozenset would not survive the JSON
        round-trip into runs.db.
        """
        cfg = SessionResumeConfig()
        harness.config.session_resume = cfg
        harness.config.transcript_archive = TranscriptArchiveConfig()
        session = {
            'session_id': 'uuid-both-wire',
            'role': 'implementer',
            'started_at': (
                datetime.now(UTC) - timedelta(seconds=2 * cfg.freshness_window_secs)
            ).isoformat(),
            'resume_count': 0,
        }
        empty_cfg = tmp_path / 'claude-config-both-wire'
        (empty_cfg / 'projects').mkdir(parents=True)

        resume_id = await _drive_session_slot(
            harness, 'both1', session, config_dir=empty_cfg
        )

        assert resume_id is None
        emits = _session_resume_emits(harness)
        assert len(emits) == 1
        et, kwargs = emits[0]
        assert et == EventType.session_resume_fallback
        data = kwargs['data']
        assert data['reasons'] == ['no_transcript', 'stale']
        assert isinstance(data['reasons'], list)
        # The first-match scalar is GONE, not kept as an alias: retaining it
        # would mean retaining the projection this task exists to kill, and
        # operators would keep reading the field that misdirects them.
        assert 'reason' not in data
        # α's instrument still rides the same emit (no archive was written).
        assert data['archive_available'] is False

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
        assert kwargs['data']['reasons'] == ['reseeded']

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
        assert kwargs['data']['reasons'] == ['no_transcript']

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
        assert kwargs['data']['reasons'] == ['no_transcript']

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
        assert emits[0][1]['data']['reasons'] == ['no_transcript']

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

    async def test_capped_plus_another_reason_emits_fallback_not_capped(
        self, harness: Harness, tmp_path: Path
    ):
        """A capped session that ALSO fails corroboration emits
        session_resume_fallback carrying BOTH reasons — NOT session_resume_capped.

        This pins the caller's ``reasons == {'capped'}`` EXACT-equality routing
        (task 3728), which every other capped test leaves unexercised because
        they all drive a fresh, corroborated session whose only failing leg is
        the cap. Relaxing that equality to ``'capped' in reasons`` passes the
        whole rest of the suite, and it would silently restore two defects at
        once: config.py documents session_resume_capped as throttling of an
        otherwise HEALTHY resumable session, so the throttle population would
        be overstated by sessions that could not have resumed anyway, and the
        co-occurring corroboration failure would be BURIED — first-match
        reporting reintroduced one level up, in the event routing rather than
        in the reason string. It would also move these dispatches between the
        two events the event_store.py ratio recipe counts.

        The streak stays 0 either way: both reasons are by-design (D4), so the
        richer event costs nothing in escalation noise.
        """
        session = {
            'session_id': 'uuid-cap-notr',
            'role': 'implementer',
            'started_at': datetime.now(UTC).isoformat(),   # fresh: not 'stale'
            'resume_count': 3,                             # at the cap
        }
        # The config dir SURVIVES but holds no transcript for this session, so
        # corroboration yields 'no_transcript' (the surviving-dir arm) rather
        # than 'reseeded'.
        empty_cfg = tmp_path / 'claude-config-cap-notr'
        (empty_cfg / 'projects').mkdir(parents=True)
        harness.config.session_resume = SessionResumeConfig(max_resumes_per_task=3)
        harness.config.transcript_archive = TranscriptArchiveConfig()

        resume_id = await _drive_session_slot(
            harness, 'cn1', session, config_dir=empty_cfg
        )

        assert resume_id is None
        emits = _session_resume_emits(harness)
        assert len(emits) == 1
        et, kwargs = emits[0]
        assert et == EventType.session_resume_fallback
        # 'capped' stays visible in the set, so routing the dispatch here
        # rather than to session_resume_capped loses no information.
        assert kwargs['data']['reasons'] == ['capped', 'no_transcript']
        assert _streak(harness) == 0

    async def test_archive_backed_session_with_no_config_dir_is_adopted(
        self, harness: Harness
    ):
        """δ END TO END (task 3730 / D2), at the GUARD rather than the
        predicate: the real crash-recovery shape — sidecar PRESERVED, config
        dir GONE, transcript recoverable only from the durable archive — is
        adopted, injected as ``resume_session_id``, and emits
        ``session_resume`` instead of ``session_resume_fallback``.

        THE BAR THIS ROW SETS, and why it is not "any session_resume". The
        task's Tier-1 signal is a resume attributable to the ARCHIVE-MEDIATED
        predicate; a live-dir resume proves nothing about D2, because that
        path was already eligible before δ (see
        :meth:`test_eligible_keeps_session_and_emits`, unchanged). So this row
        seeds NO config dir at all and back-dates the sidecar PAST
        ``freshness_window_secs``: before δ that combination was doubly
        ineligible (``{'no_transcript', 'stale'}``), and it is the shape
        production reaches on EVERY crash-recovery path — ``run()``'s finally
        executes an unconditional ``cleanup_config_dir`` teardown
        (registered by ``workflow.py::TaskWorkflow._on_terminal_cleanups``,
        run on every terminal exit) while ``session_preserved`` keeps
        the sidecar, so ``_adopt_recovered_session``'s glob hands the guard
        ``config_dir=None``. ~91% of post-3578 fallbacks (92 of 101, measured
        2026-09-04) had exactly this recoverable archive the predicate never
        consulted.

        Also pins what adoption does NOT do: a storm run in progress survives
        it untouched (task ε/3733). Adoption is an ELIGIBILITY verdict reached
        before the restore, so on this very path — archive-mediated, the one δ
        opened — the resume it adopts may still fault a phase later. Only a
        resume that survived retires the run.
        """
        cfg = SessionResumeConfig()
        harness.config.session_resume = cfg
        harness.config.transcript_archive = TranscriptArchiveConfig()
        assert 2 * cfg.freshness_window_secs < cfg.absolute_resume_age_secs, (
            'this row must sit BETWEEN the two thresholds, or it is measuring '
            'the backstop rather than the freshness demotion'
        )
        session = {
            'session_id': 'uuid-delta-e2e',
            'role': 'implementer',
            'started_at': (
                datetime.now(UTC) - timedelta(seconds=2 * cfg.freshness_window_secs)
            ).isoformat(),
            'resume_count': 0,
        }
        _make_archive(harness.config.project_root, 'dz1', 'uuid-delta-e2e')
        harness._session_resume_fallback_streak = 2  # a run in progress

        resume_id = await _drive_session_slot(harness, 'dz1', session)  # no config_dir

        assert resume_id is session
        emits = _session_resume_emits(harness)
        assert [et for et, _ in emits] == [EventType.session_resume]
        assert _streak(harness) == 2

    async def test_aged_out_archive_backed_session_still_falls_back(
        self, harness: Harness
    ):
        """The D3 backstop at the guard: the SAME archive-backed, config-dir-less
        shape adopted above still falls back once it is past
        ``absolute_resume_age_secs``, and the event carries 'aged_out'.

        Reachability outranks freshness, NOT the backstop — without this row
        the change above would read as "an archive exempts a session from age
        entirely". 'aged_out' rather than 'stale' so the two thresholds stay
        distinguishable in runs.db, and by-design so a batch of week-old
        sidecars after a long outage cannot page an operator (D4/D5).

        Back-dated off the CONFIG FIELD, never a literal, so a re-derived
        bound re-tunes this row with it.
        """
        cfg = SessionResumeConfig()
        harness.config.session_resume = cfg
        harness.config.transcript_archive = TranscriptArchiveConfig()
        session = {
            'session_id': 'uuid-delta-agedout',
            'role': 'implementer',
            'started_at': (
                datetime.now(UTC) - timedelta(seconds=2 * cfg.absolute_resume_age_secs)
            ).isoformat(),
            'resume_count': 0,
        }
        _make_archive(harness.config.project_root, 'dz2', 'uuid-delta-agedout')

        resume_id = await _drive_session_slot(harness, 'dz2', session)  # no config_dir

        assert resume_id is None
        emits = _session_resume_emits(harness)
        assert len(emits) == 1
        et, kwargs = emits[0]
        assert et == EventType.session_resume_fallback
        # 'stale' is suppressed by the archive; only the backstop fires, and
        # the corroboration leg is satisfied by the archive too.
        assert kwargs['data']['reasons'] == ['aged_out']
        assert kwargs['data']['archive_available'] is True
        assert _streak(harness) == 0  # by design (D4)

    async def test_restore_kill_switch_withholds_the_archive_from_eligibility(
        self, harness: Harness
    ):
        """THE NARROW KILL SWITCH STILL REVERTS δ (task 3578's
        ``restore_from_archive``, at δ's guard).

        Same archive-backed, config-dir-less shape adopted two rows above, with
        restoration disabled — the switch an operator pulls precisely when they
        suspect a restore regression. It must put that session back on its
        pre-δ path: an archive nothing will rehydrate does not make a session
        reachable.

        WHAT GOES WRONG IF ELIGIBILITY IGNORES THE SWITCH. The guard would
        adopt the session and emit ``session_resume``; the arm site would then
        skip rehydration (``restore_outcome='disabled'``), fail
        re-corroboration against the fresh config dir, veto, and dispatch fresh
        with ``session_resume_failed(stage='pre_flight')``. Every
        archive-mediated session would move from ``session_resume_fallback`` to
        ``session_resume`` while none of them actually resumed — so D8's ratio
        recipe, and the OPERATIONS.md §14 instruction to watch
        ``session_resume`` rise, would read 100% resumed at 0% resumed. The
        switch would have made the signal it exists to preserve actively
        misleading.

        The INSTRUMENT is NOT withheld with it: ``archive_available`` still
        reports what is on disk, because 'restore switched off' and 'no archive
        at all' are different operator situations and this field is the only
        thing in runs.db that tells them apart. That is the same field
        ``restore_from_archive``'s own description promises not to go blind on.
        """
        cfg = SessionResumeConfig(restore_from_archive=False)
        harness.config.session_resume = cfg
        harness.config.transcript_archive = TranscriptArchiveConfig()
        session = {
            'session_id': 'uuid-delta-norestore',
            'role': 'implementer',
            'started_at': (
                datetime.now(UTC) - timedelta(seconds=2 * cfg.freshness_window_secs)
            ).isoformat(),
            'resume_count': 0,
        }
        _make_archive(harness.config.project_root, 'dz3', 'uuid-delta-norestore')

        resume_id = await _drive_session_slot(harness, 'dz3', session)  # no config_dir

        assert resume_id is None
        emits = _session_resume_emits(harness)
        assert len(emits) == 1
        et, kwargs = emits[0]
        assert et == EventType.session_resume_fallback
        # EXACTLY the pre-δ answer for this shape (cf.
        # test_the_archive_is_the_only_thing_that_changed_the_answer).
        assert kwargs['data']['reasons'] == ['no_transcript', 'stale']
        assert kwargs['data']['archive_available'] is True


@pytest.mark.asyncio
class TestSessionResumeStorm:
    """γ fallback-storm escape (INV-4, task 2774; narrowed by 3256, carved out
    by 3728): a RUN of GENUINE session-resume failures reaching
    fallback_storm_threshold files ONE deduped L1 escalation.

    The streak is a rolling CHAIN, not a per-boot running total: consecutive
    means chained within storm_window_secs, so a gap at least that long decays
    it to 0 (a resume that ADOPTED AND SURVIVED also retires the run outright,
    clearing the chain's comparison stamp and the recorded failures with it —
    a resume merely judged ELIGIBLE does not, since that predicate runs before
    the restore and settles nothing about its outcome).

    WHAT FEEDS IT is the part task 3728 changed. EVERY by-design outcome is now
    excluded by construction — 'disabled' (silent kill switch), 'capped'
    (throttling), 'reseeded' (lane wipe) and, newly, 'stale' and
    'no_transcript'. The last two were classified genuine while
    ``_session_resume_reasons``' own docstring described them as the
    anticipated reseed/wipe/clock cases, so the L1 they filed sent operators to
    check NTP for a population the system expects to see. After the carve-out
    NOTHING the predicate can produce feeds the streak, so the rows driving it
    through ``_run_slot`` use a SYNTHETIC feeder (:meth:`_arm_synthetic_feeder`),
    which doubles as the executable statement of the extension contract.

    THE LIVE FEEDER is ε's (task 3733), and it is a different seam: every armed
    resume that did not survive, reported from ``TaskWorkflow._invoke`` to
    :meth:`Harness.note_resume_failed` and classified against
    ``_BY_DESIGN_RESTORE_OUTCOMES``. The ε rows at the end of this class drive
    those two methods directly against the same streak, the same decay and the
    same filer. Filing is best-effort — a None queue never raises (I3).
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

    @staticmethod
    def _arm_synthetic_feeder(
        harness: Harness, reasons: frozenset[str] = frozenset({'restore_failed'}),
    ) -> None:
        """Make every dispatch report *reasons*, standing in for ε's feeder.

        After the 3728 carve-out no reason the real predicate can produce feeds
        the streak, so the increment / threshold / filing path has no reachable
        production input until PRD leaf ε (task 3733) adds one. Patching the
        predicate is the narrowest injection point that still exercises the
        REAL caller, the REAL streak, the REAL dedup and the REAL filer — and
        it states ε's extension contract executably: a reason absent from
        ``_BY_DESIGN_SESSION_RESUME_REASONS`` feeds the storm with no second
        edit anywhere.

        ``restore_failed`` is NOT the name ε ended up using, and the reason is
        structural rather than a change of mind: ε's feeder is the ARM SEAM,
        not a predicate reason. ``_session_resume_reasons`` is evaluated in
        ``_run_slot`` BEFORE the resume injection and takes ``archive_available``
        as a bool specifically so it acquires no filesystem dependency of its
        own, while ``restore_archived_transcript`` has exactly ONE call site —
        inside ``TaskWorkflow._invoke``, a whole process-phase downstream. A
        predicate that runs before the restore, and is contractually forbidden
        I/O, cannot report that a restore failed; making it do so would corrupt
        an ELIGIBILITY predicate into an outcome log.

        So this row is KEPT, and the name with it, as the executable statement
        of the EXTENSION CONTRACT: a reason absent from
        ``_BY_DESIGN_SESSION_RESUME_REASONS`` feeds the storm with no second
        edit anywhere. ε's own carve-out
        (``_BY_DESIGN_RESTORE_OUTCOMES``) states the same rule one seam later.
        """
        def _reasons(
            session: dict, config_dir: str | None, *, archive_available: bool
        ) -> frozenset[str]:
            # Accepts δ's archive_available (task 3730) so the stub tracks the
            # real signature — a **kwargs sponge would keep passing if the
            # caller stopped supplying it, which is the one thing the hoist
            # rows in TestSessionResumeArchiveAvailable exist to catch.
            return reasons

        harness._session_resume_reasons = _reasons  # type: ignore[method-assign]

    # ── NEGATIVE half: no by-design outcome can trip the escape ──────────────

    @pytest.mark.parametrize(
        'reason', ['stale', 'no_transcript', 'reseeded', 'capped'],
    )
    async def test_by_design_reasons_never_file_l1(
        self, harness: Harness, tmp_path: Path, reason: str,
    ):
        """EVERY by-design outcome is excluded from the streak by construction
        (task 3728 D4), however many of them arrive.

        threshold=1 makes the very first GENUINE fallback fire, so a zero
        submit count across four dispatches proves the reason does not feed the
        streak at all. ``stale`` and ``no_transcript`` are the two this task
        RECLASSIFIED — they used to file the L1 that told operators to check
        NTP; ``reseeded`` and ``capped`` were already excluded (task 3256 /
        2774) and are re-asserted here under the new set-based feeder.

        The telemetry channel must SURVIVE the carve-out: this is noise
        suppression of the ESCALATION, never of the event. One event per
        dispatch, carrying the reason, exactly as before.
        """
        harness.config.session_resume = SessionResumeConfig(
            fallback_storm_threshold=1, max_resumes_per_task=3,
        )
        harness._escalation_queue = self._queue()

        expect_capped = reason == 'capped'
        for i in range(4):
            sid = f'uuid-bd-{reason}-{i}'
            if reason == 'stale':
                # Corroborated dir, so staleness is the SOLE reason.
                session, cfg = self._stale_session(sid), _make_transcript(tmp_path, sid)
            elif reason == 'no_transcript':
                # The dir SURVIVES but holds no transcript for this session.
                session = self._fresh_session(sid)
                cfg = tmp_path / f'empty-{i}'
                (cfg / 'projects').mkdir(parents=True)
            elif reason == 'reseeded':
                session = self._fresh_session(sid)
                cfg = tmp_path / f'gone{i}' / 'claude-config-x'
            else:  # capped
                session = self._fresh_session(sid) | {'resume_count': 3}
                cfg = _make_transcript(tmp_path, sid)

            await _drive_session_slot(harness, f'bd{i}', session, config_dir=cfg)
            # Asserted INSIDE the loop: the counter must never transiently
            # rise, not merely end at 0.
            assert _streak(harness) == 0
            assert harness._escalation_queue.submit.call_count == 0

        emits = _session_resume_emits(harness)
        assert len(emits) == 4
        for et, kwargs in emits:
            if expect_capped:
                assert et == EventType.session_resume_capped
            else:
                assert et == EventType.session_resume_fallback
                assert kwargs['data']['reasons'] == [reason]

    async def test_by_design_constant_classifies_every_producible_reason(self):
        """``_BY_DESIGN_SESSION_RESUME_REASONS`` covers EXACTLY the vocabulary
        ``_session_resume_reasons`` can currently produce.

        The classification defaults to GENUINE — a reason absent from the
        constant feeds the storm — which is the fail-LOUD direction, but it
        makes "forgot to think about it" and "deliberately genuine"
        indistinguishable at the constant. This row closes that: the producible
        vocabulary is read structurally out of the predicate's own source, so
        adding a reason without classifying it fails HERE.

        The read is deliberately FAIL-CLOSED, and that is the whole design of
        it. An earlier version collected only string constants sitting inside
        ``set.add(...)`` / ``frozenset(...)`` calls, which failed OPEN for
        every other spelling: ``reasons |= {'x'}``, ``reasons.update({'x'})``
        and a reason routed through a module constant were all invisible, so a
        new unclassified reason would have slipped past this row and gone
        straight onto the INV-4 storm feeder as escalation noise. So instead of
        guessing which syntax introduces a reason, this collects EVERY string
        literal in the method (docstrings excluded) plus any module-global the
        method references that is itself a string or a collection of strings,
        and requires the total to partition exactly into declared non-reasons
        and classified reasons.

        The cost of fail-closed is that an unrelated new literal — say a fourth
        sidecar key — also fails this row. That is intended: the fix is one
        line in ``non_reason_literals`` below, and it forces a human to state
        which kind of string was just added. Fail-open, by contrast, is silent.

        ε (task 3733) was expected to trip this by adding ``restore_failed``
        to the predicate, and did not: it keys its feeder on
        ``TaskWorkflow._invoke``'s arm seam instead, for the structural reason
        recorded on :meth:`_arm_synthetic_feeder` above. The restore vocabulary
        gets its own sibling constant and its own fail-closed structural row —
        ``_BY_DESIGN_RESTORE_OUTCOMES`` and
        ``test_by_design_restore_constant_classifies_every_producible_outcome``
        — so this one still covers EXACTLY the predicate, which is what makes
        its equality assertion meaningful.
        """
        import ast  # noqa: PLC0415 — structural read of one method's source
        import inspect  # noqa: PLC0415
        import textwrap  # noqa: PLC0415

        import orchestrator.harness as harness_mod  # noqa: PLC0415
        from orchestrator.harness import _BY_DESIGN_SESSION_RESUME_REASONS

        # Strings the method uses that are NOT reasons. Every one is a key read
        # off the recovered-session sidecar dict; nothing here reaches the
        # returned set.
        non_reason_literals = {'started_at', 'resume_count', 'session_id'}

        fn = ast.parse(
            textwrap.dedent(inspect.getsource(Harness._session_resume_reasons))
        ).body[0]

        # Docstrings are string constants too — identify them by position (the
        # first statement of any scope) and skip exactly those nodes.
        docstrings = {
            id(scope.body[0].value)
            for scope in ast.walk(fn)
            if isinstance(scope, ast.FunctionDef | ast.AsyncFunctionDef | ast.ClassDef)
            and scope.body
            and isinstance(scope.body[0], ast.Expr)
            and isinstance(scope.body[0].value, ast.Constant)
            and isinstance(scope.body[0].value.value, str)
        }
        strings: set[str] = {
            node.value
            for node in ast.walk(fn)
            if isinstance(node, ast.Constant)
            and isinstance(node.value, str)
            and id(node) not in docstrings
        }
        # ...and any module-level constant the method reads that could CARRY a
        # reason, so routing one through an indirection does not hide it.
        for node in ast.walk(fn):
            if not isinstance(node, ast.Name):
                continue
            value = getattr(harness_mod, node.id, None)
            if isinstance(value, str):
                strings.add(value)
            elif isinstance(value, set | frozenset | tuple | list) and all(
                isinstance(item, str) for item in value
            ):
                strings |= set(value)

        producible = strings - non_reason_literals
        assert producible == {'disabled', 'stale', 'capped', 'no_transcript',
                              'reseeded', 'aged_out'}, (
            'the string literals in _session_resume_reasons no longer partition '
            'into the declared non-reasons and the known reason vocabulary. If '
            'you added a REASON, classify it in '
            '_BY_DESIGN_SESSION_RESUME_REASONS (or deliberately leave it out, '
            'making it a genuine storm feeder) and extend this assertion; if '
            'you added a non-reason string, add it to non_reason_literals '
            f'above. Saw: {sorted(producible)}'
        )
        assert producible == _BY_DESIGN_SESSION_RESUME_REASONS

    async def test_by_design_restore_constant_classifies_every_producible_outcome(self):
        """``_BY_DESIGN_RESTORE_OUTCOMES`` classifies EXACTLY the restore
        vocabulary ``TaskWorkflow._invoke``'s arm block can produce (ε/3733).

        The sibling of the row above, one seam downstream. β's constant covers
        the ELIGIBILITY predicate, evaluated before dispatch on a sidecar; this
        one covers the archive RESTORE, evaluated inside ``_invoke`` against
        the filesystem. Because ε keys its feeder on the ARM SEAM rather than
        on a predicate reason, this — not the reason set — is where an
        unclassified new value would slip onto the INV-4 storm feeder.

        Fail-CLOSED in the same way, and for the same reason: every string
        literal anywhere inside a statement that ASSIGNS ``restore_outcome`` is
        collected, rather than matching one syntax. A fifth outcome introduced
        as a plain constant, as a ternary arm (which ``'published' if … else
        'miss'`` already is), as a walrus, or routed through a module-level
        constant therefore fails HERE instead of becoming a silent feeder.

        The carve-out must also stay a STRICT subset: were it ever to equal the
        producible set, nothing could feed the streak and INV-4's escape would
        be inert again — precisely the condition ε exists to end.
        """
        import ast  # noqa: PLC0415 — structural read of one method's source
        import inspect  # noqa: PLC0415
        import textwrap  # noqa: PLC0415

        import orchestrator.workflow as workflow_mod  # noqa: PLC0415
        from orchestrator.harness import _BY_DESIGN_RESTORE_OUTCOMES  # noqa: PLC0415
        from orchestrator.workflow import TaskWorkflow  # noqa: PLC0415

        fn = ast.parse(
            textwrap.dedent(inspect.getsource(TaskWorkflow._invoke))
        ).body[0]

        producible: set[str] = set()
        for node in ast.walk(fn):
            if isinstance(node, ast.Assign):
                targets, value = node.targets, node.value
            elif isinstance(node, ast.AnnAssign | ast.AugAssign | ast.NamedExpr):
                targets, value = [node.target], node.value
            else:
                continue
            if value is None or not any(
                isinstance(t, ast.Name) and t.id == 'restore_outcome' for t in targets
            ):
                continue
            # The WHOLE value subtree, not just a top-level Constant: the
            # shipped vocabulary already routes two outcomes through a ternary.
            for sub in ast.walk(value):
                if isinstance(sub, ast.Constant) and isinstance(sub.value, str):
                    producible.add(sub.value)
                elif isinstance(sub, ast.Name):
                    # ...and a module-global indirection must not hide one.
                    resolved = getattr(workflow_mod, sub.id, None)
                    if isinstance(resolved, str):
                        producible.add(resolved)
                    elif isinstance(resolved, set | frozenset | tuple | list) and all(
                        isinstance(item, str) for item in resolved
                    ):
                        producible |= set(resolved)

        assert producible == {'disabled', 'miss', 'fault', 'published'}, (
            'the restore outcomes assigned in TaskWorkflow._invoke no longer '
            'match the known vocabulary. If you added an OUTCOME, classify it '
            'in harness.py::_BY_DESIGN_RESTORE_OUTCOMES (or deliberately leave '
            'it out, making it a genuine storm feeder) and extend this '
            f'assertion. Saw: {sorted(producible)}'
        )
        assert isinstance(_BY_DESIGN_RESTORE_OUTCOMES, frozenset)
        carved_out = set(_BY_DESIGN_RESTORE_OUTCOMES)
        assert carved_out == {'disabled', 'miss'}
        assert carved_out < producible, (
            'the by-design restore carve-out must be a STRICT subset of the '
            'producible vocabulary — if it covers everything, no restore '
            'outcome can feed the streak and the INV-4 escape is inert again'
        )

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
        assert _streak(harness) == 0
        emits = _session_resume_emits(harness)
        assert len(emits) == 3
        for et, kwargs in emits:
            assert et == EventType.session_resume_fallback
            assert kwargs['data']['reasons'] == ['reseeded']

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

    # ── POSITIVE half: the RETAINED mechanism, driven by ε's stand-in ────────

    async def test_genuine_feeder_files_one_l1_at_threshold(self, harness: Harness):
        """A reason OUTSIDE the by-design vocabulary reaches the threshold and
        files exactly one L1 — the mechanism is retained, not deleted (F3).
        """
        harness.config.session_resume = SessionResumeConfig(fallback_storm_threshold=3)
        harness._escalation_queue = self._queue()
        self._arm_synthetic_feeder(harness)

        for i in range(3):
            await _drive_session_slot(harness, f'st{i}', self._fresh_session(f'uuid-st{i}'))

        assert _streak(harness) == 3
        assert harness._escalation_queue.submit.call_count == 1
        esc = harness._escalation_queue.submit.call_args.args[0]
        assert esc.level == 1
        assert 'resume' in esc.summary.lower()
        # The telemetry rides the same emit, carrying the unclassified reason.
        emits = _session_resume_emits(harness)
        assert len(emits) == 3
        for et, kwargs in emits:
            assert et == EventType.session_resume_fallback
            assert kwargs['data']['reasons'] == ['restore_failed']

    async def test_dedup_no_second_submit_when_l1_open(self, harness: Harness):
        harness.config.session_resume = SessionResumeConfig(fallback_storm_threshold=3)
        harness._escalation_queue = self._queue()
        self._arm_synthetic_feeder(harness)

        for i in range(3):
            await _drive_session_slot(harness, f'st{i}', self._fresh_session(f'uuid-st{i}'))
        assert harness._escalation_queue.submit.call_count == 1

        # L1 now open → further fallbacks must NOT re-submit (has_open_l1 dedup).
        harness._escalation_queue.has_open_l1 = MagicMock(return_value=True)
        for i in range(3, 6):
            await _drive_session_slot(harness, f'st{i}', self._fresh_session(f'uuid-st{i}'))
        assert harness._escalation_queue.submit.call_count == 1

    async def test_a_by_design_reason_cannot_launder_a_genuine_one(
        self, harness: Harness
    ):
        """D4 ∧ D5 — a MIXED set still feeds the streak.

        This is the conjunction that makes the carve-out safe. Under first-match
        reporting a session that was BOTH stale and (say) restore-failed
        reported only ``stale``; carving ``stale`` out of the feeder on top of
        that would have SILENCED the genuine failure entirely — a by-design
        reason laundering a real one, which is strictly worse than the
        misdirection this task set out to fix. Because the caller subtracts the
        by-design SET from the full reason set, what is left is non-empty and
        the streak feeds exactly as it would on the genuine reason alone.
        """
        harness.config.session_resume = SessionResumeConfig(fallback_storm_threshold=2)
        harness._escalation_queue = self._queue()
        self._arm_synthetic_feeder(harness, frozenset({'stale', 'restore_failed'}))

        for i in range(2):
            await _drive_session_slot(harness, f'mx{i}', self._fresh_session(f'uuid-mx{i}'))

        assert _streak(harness) == 2
        assert harness._escalation_queue.submit.call_count == 1
        # ...and BOTH reasons are on the wire, so the operator sees the
        # by-design co-occurrence rather than inferring it.
        et, kwargs = _session_resume_emits(harness)[0]
        assert et == EventType.session_resume_fallback
        assert kwargs['data']['reasons'] == ['restore_failed', 'stale']

    async def test_storm_l1_names_the_failures_rather_than_a_census(
        self, harness: Harness
    ):
        """The filed L1 must send the operator to the EVIDENCE first.

        Its detail used to name ``stale`` and ``no_transcript`` as the only
        surviving causes and send the operator to check NTP FIRST. Task 3728
        excluded those by construction, and this row then pinned the
        replacement: a SQL census the operator had to run, and a reason to
        guess at from its output.

        ε (task 3733) removes the guess entirely. The streak now has a live
        feeder that records WHICH resumes failed, so the escalation can name
        them — task, session, role, stage, restore outcome, archive path — and
        the census prose this row used to pin is gone. What survives unchanged
        is the CONTRACT it was really asserting: the L1 is filed at level 1 and
        its first directive starts from evidence rather than from a guess. The
        wording around it stays free to improve.
        """
        harness.config.session_resume = SessionResumeConfig(fallback_storm_threshold=1)
        harness._escalation_queue = self._queue()
        self._arm_synthetic_feeder(harness)

        await _drive_session_slot(harness, 'p1', self._fresh_session('uuid-p1'))

        esc = harness._escalation_queue.submit.call_args.args[0]
        assert esc.level == 1
        # The failure is NAMED — the same renderer the arm-seam feeder uses,
        # so there is one streak and one L1 (SPOT).
        assert 'p1' in esc.detail
        assert 'uuid-p1' in esc.detail
        assert 'restore_failed' in esc.detail
        # ...and the operator is no longer handed a census to run and read.
        assert "json_extract(data,'$.reasons')" not in esc.detail
        first_directive = esc.suggested_action.split('.')[0].lower()
        assert 'query' not in first_directive

    async def test_genuine_failures_still_file_l1_across_by_design(
        self, harness: Harness, tmp_path: Path
    ):
        """A by-design fallback neither counts toward NOR resets the genuine
        streak (task 3256's anti-masking rule, generalised by 3728).

        Interleaving reseeds between ``threshold`` genuine fallbacks must still
        file exactly one L1: a drip of by-design outcomes cannot mask a genuine
        systematic failure hiding between them. The genuine half now comes from
        the synthetic feeder, since stale no longer qualifies — so the
        interleaving is driven by swapping the predicate rather than by two
        different fixtures.
        """
        harness.config.session_resume = SessionResumeConfig(fallback_storm_threshold=3)
        harness._escalation_queue = self._queue()

        for i in range(3):
            self._arm_synthetic_feeder(harness)
            await _drive_session_slot(harness, f'g{i}', self._fresh_session(f'uuid-g{i}'))
            # A by-design reseed between each genuine failure.
            self._arm_synthetic_feeder(harness, frozenset({'reseeded'}))
            await _drive_session_slot(harness, f'ir{i}', self._fresh_session(f'uuid-ir{i}'))

        assert harness._escalation_queue.submit.call_count == 1
        esc = harness._escalation_queue.submit.call_args.args[0]
        assert esc.level == 1
        assert 'resume' in esc.summary.lower()

    async def test_streak_is_consecutive_reset_by_a_surviving_resume(
        self, harness: Harness, tmp_path: Path
    ):
        """The streak is CONSECUTIVE, not cumulative — and what breaks a run is
        a resume that SURVIVED, not one merely judged eligible.

        β wrote this row against the guard's eligibility branch, and it was
        sound while the predicate was the only feeder: "eligible" and "not a
        failure" were then ONE proposition, on one input, at one instant. ε
        moves the feeder a whole process-phase downstream — into the restore
        ``_invoke`` performs after this predicate has already run — so the two
        are no longer the same claim, and only the arm seam can report that a
        resume actually worked.

        Both halves of the original contract survive: the run resumes from 0
        after a genuine reset (the count is consecutive), and an eligible
        dispatch interleaved mid-run does NOT break the chain.
        """
        harness.config.session_resume = SessionResumeConfig(fallback_storm_threshold=3)
        harness._escalation_queue = self._queue()
        self._arm_synthetic_feeder(harness)

        # 2 genuine fallbacks (streak=2)...
        for i in range(2):
            await _drive_session_slot(harness, f'a{i}', self._fresh_session(f'uuid-a{i}'))
        # ...then a resume that ADOPTED AND SURVIVED retires the run.
        harness.note_resume_succeeded()
        assert _streak(harness) == 0

        # 2 more genuine after the reset → streak=2 (<3) → still no L1.
        for i in range(2):
            await _drive_session_slot(harness, f'b{i}', self._fresh_session(f'uuid-b{i}'))
        assert harness._escalation_queue.submit.call_count == 0

        # An ELIGIBLE dispatch interleaved here leaves the run alone — it is a
        # pre-dispatch predicate, not a verdict on any restore. The real
        # predicate has to be back in place for a resume to BE eligible.
        del harness._session_resume_reasons
        cfg = _make_transcript(tmp_path, 'uuid-ok')
        await _drive_session_slot(
            harness, 'ok1', self._fresh_session('uuid-ok'), config_dir=cfg,
        )
        assert _streak(harness) == 2

        # A 3rd consecutive genuine fallback AFTER the reset reaches threshold →
        # fires once, proving the streak resumed from 0 (consecutive, not
        # cumulative).
        self._arm_synthetic_feeder(harness)
        await _drive_session_slot(harness, 'b2', self._fresh_session('uuid-b2'))
        assert harness._escalation_queue.submit.call_count == 1

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
        self._arm_synthetic_feeder(harness)

        # Two genuine fallbacks inside the window → streak=2.
        for i in range(2):
            await _drive_session_slot(harness, f'd{i}', self._fresh_session(f'uuid-d{i}'))
        assert _streak(harness) == 2

        # ...then the clock jumps past the window before the 3rd arrives.
        # (The assert also pins that a genuine fallback stamped the chain point.)
        assert _chain_stamp(harness) is not None
        _rewind_chain(harness, 120)

        await _drive_session_slot(harness, 'd2', self._fresh_session('uuid-d2'))

        # Decayed to 0, then re-incremented — NOT 3, so no L1.
        assert _streak(harness) == 1
        assert harness._escalation_queue.submit.call_count == 0

    async def test_window_retires_the_run_on_a_by_design_dispatch(
        self, harness: Harness, tmp_path: Path
    ):
        """The counter must be correct when READ, not merely when incremented
        (task 3256's addendum defect, in scope for 3728).

        The decay is only meaningful if the passage of the window RETIRES a run
        — but it is computed inside the increment branch, so with no genuine
        feeder arriving the counter holds its last value indefinitely and any
        other reader sees a run that ended long ago. ε's re-armed feeder makes
        that reader real; today it is already a lie about state.

        Drive threshold-1 genuine failures, let the window pass, then dispatch
        ONE ordinary by-design (stale) session: the run must be gone — streak 0
        AND the chain's comparison stamp cleared — even though that dispatch
        contributed nothing itself.
        """
        harness.config.session_resume = SessionResumeConfig(
            fallback_storm_threshold=3, storm_window_secs=60,
        )
        harness._escalation_queue = self._queue()

        self._arm_synthetic_feeder(harness)
        for i in range(2):  # threshold - 1
            await _drive_session_slot(harness, f'rq{i}', self._fresh_session(f'uuid-rq{i}'))
        assert _streak(harness) == 2
        assert _chain_stamp(harness) is not None

        # The clock passes the window with no further genuine failure.
        _rewind_chain(harness, 120)

        # A perfectly ordinary by-design dispatch (real predicate, corroborated
        # dir, aged sidecar → {'stale'}) is enough to observe the expiry.
        del harness._session_resume_reasons
        cfg = _make_transcript(tmp_path, 'uuid-rq-stale')
        await _drive_session_slot(
            harness, 'rq-stale', self._stale_session('uuid-rq-stale'), config_dir=cfg,
        )

        assert _streak(harness) == 0
        assert _chain_stamp(harness) is None
        assert harness._escalation_queue.submit.call_count == 0

    async def test_by_design_dispatch_inside_the_window_decays_nothing(
        self, harness: Harness, tmp_path: Path
    ):
        """...and evaluating the decay per dispatch must not become a RESET per
        dispatch.

        The same interleaving WITHOUT the clock passing leaves both the counter
        and the stamp exactly as they were: a by-design outcome still neither
        FEEDS the streak nor RESETS it (task 3256's anti-masking rule — a drip
        of expected fallbacks must not launder a genuine systematic failure
        interleaved between them), and it still does not refresh the chain
        stamp, which would keep a run alive across an arbitrarily long gap.
        """
        harness.config.session_resume = SessionResumeConfig(
            fallback_storm_threshold=3, storm_window_secs=60,
        )
        harness._escalation_queue = self._queue()

        self._arm_synthetic_feeder(harness)
        for i in range(2):
            await _drive_session_slot(harness, f'nw{i}', self._fresh_session(f'uuid-nw{i}'))
        stamp = _chain_stamp(harness)
        assert _streak(harness) == 2
        assert stamp is not None

        del harness._session_resume_reasons
        cfg = _make_transcript(tmp_path, 'uuid-nw-stale')
        await _drive_session_slot(
            harness, 'nw-stale', self._stale_session('uuid-nw-stale'), config_dir=cfg,
        )

        assert _streak(harness) == 2
        assert _chain_stamp(harness) == stamp

        # And the run is still live: one more genuine failure reaches the
        # threshold, so the per-dispatch decay did not quietly neuter INV-4.
        self._arm_synthetic_feeder(harness)
        await _drive_session_slot(harness, 'nw2', self._fresh_session('uuid-nw2'))
        assert _streak(harness) == 3
        assert harness._escalation_queue.submit.call_count == 1

    async def test_by_design_fallbacks_do_not_refresh_the_chain_stamp(
        self, harness: Harness, tmp_path: Path
    ):
        """A by-design fallback must leave ``_last_session_resume_fallback_at``
        untouched, not just the streak counter (task 3256 amendment).

        The stamp is the chain's comparison point, so refreshing it on a
        by-design outcome would keep the chain alive across an arbitrarily long
        drip: two genuine failures hours apart could then still chain into a
        false storm, re-opening the exact hole the decay window closes.
        """
        harness.config.session_resume = SessionResumeConfig(
            fallback_storm_threshold=2, storm_window_secs=60,
        )
        harness._escalation_queue = self._queue()

        # One genuine fallback opens the chain: streak=1, stamp set.
        self._arm_synthetic_feeder(harness)
        await _drive_session_slot(harness, 'cs0', self._fresh_session('uuid-cs0'))
        assert _streak(harness) == 1
        stamp = _chain_stamp(harness)
        assert stamp is not None

        # A drip of REAL by-design reseeds moves NEITHER the counter nor the
        # stamp (real predicate, so this is not an artifact of the stand-in).
        del harness._session_resume_reasons
        for i in range(3):
            await _drive_session_slot(
                harness, f'csr{i}', self._fresh_session(f'uuid-csr{i}'),
                config_dir=tmp_path / f'gone{i}' / 'claude-config-x',
            )
        assert _streak(harness) == 1
        assert _chain_stamp(harness) == stamp

        # So when the clock passes the window, the NEXT genuine fallback is
        # measured against the first one and decays — reaching 1, not the
        # threshold of 2.
        _set_chain_stamp(harness, stamp - 120)
        self._arm_synthetic_feeder(harness)
        await _drive_session_slot(harness, 'cs1', self._fresh_session('uuid-cs1'))

        assert _streak(harness) == 1
        assert harness._escalation_queue.submit.call_count == 0

    async def test_streak_survives_within_storm_window(self, harness: Harness):
        """The decay must NOT neuter INV-4: a genuine tight burst still reaches
        the threshold and files exactly one L1 (task 3256).
        """
        harness.config.session_resume = SessionResumeConfig(
            fallback_storm_threshold=3, storm_window_secs=60,
        )
        harness._escalation_queue = self._queue()
        self._arm_synthetic_feeder(harness)

        # Three back-to-back fallbacks — no rewind, so all chain inside the window.
        for i in range(3):
            await _drive_session_slot(harness, f'w{i}', self._fresh_session(f'uuid-w{i}'))

        assert _streak(harness) == 3
        assert harness._escalation_queue.submit.call_count == 1
        esc = harness._escalation_queue.submit.call_args.args[0]
        assert esc.level == 1

    # β's `test_eligible_resume_clears_fallback_timestamp` stood here. Its
    # whole subject was the retirement task ε/3733 deleted, and its real
    # contract — that a reset clears the chain's comparison point and not only
    # the streak (task 3256) — moved with the reset to the arm seam, where
    # `test_success_report_retires_the_run` asserts it verbatim. Re-pointing it
    # would have produced a duplicate of that row, or of
    # `test_an_eligible_dispatch_never_retires_a_run_in_progress`, which
    # asserts the stamp's survival along with the other two pieces of run
    # state; neither is a second kind of coverage (SPOT).

    async def test_no_escalation_queue_never_raises(self, harness: Harness):
        """A bare harness (no escalation queue) must never raise on a fallback
        that would otherwise trip the storm filer (fail-safe totality, I3).
        """
        harness.config.session_resume = SessionResumeConfig(fallback_storm_threshold=1)
        harness._escalation_queue = None
        self._arm_synthetic_feeder(harness)

        # threshold=1 → the very first genuine fallback trips the filer, which
        # must early-return on the absent queue rather than raising.
        await _drive_session_slot(harness, 'x1', self._fresh_session('uuid-x1'))

    # ── ε (task 3733): the ARM-SEAM feeder ──────────────────────────────────
    #
    # β's synthetic feeder above stands in at the ELIGIBILITY predicate. ε's
    # real one reports from `TaskWorkflow._invoke`'s arm seam instead — the
    # single place both resume producers converge and the only place the
    # restore actually happens — so these rows drive `note_resume_failed` /
    # `note_resume_succeeded` directly against the REAL streak, the REAL decay
    # and the REAL filer, exactly as the synthetic rows do at the other seam.

    @staticmethod
    def _report(n: int = 0, **over):
        """Build one `ResumeFailure`, genuine (restore='fault') by default.

        Imported at CALL time, never at module scope, so a missing name in a
        RED phase fails only these rows instead of collection (the idiom
        ``_session_resume_emits`` already uses for EventType members).
        """
        from orchestrator.workflow import ResumeFailure  # noqa: PLC0415
        fields = {
            'task_id': f'task-{n}',
            'session_id': f'uuid-rf-{n}',
            'role': 'implementer',
            'stage': 'pre_flight',
            'restore': 'fault',
            'archive_root': f'/archive/root-{n}',
            'archive_path': f'/archive/root-{n}/sess-{n}.jsonl.gz',
            'detail': f'OSError: [Errno 28] No space left on device #{n}',
        }
        return ResumeFailure(**(fields | over))

    async def test_sink_files_one_l1_at_threshold_and_dedups(self, harness: Harness):
        """A RUN of genuine arm-seam failures reaches the threshold and files
        EXACTLY ONE L1, whatever follows it.

        The run deliberately mixes both stages: two pre_flight restore faults
        and one cli-stage rejection. A cli-stage report carries no restore
        outcome at all (the restore happened a phase earlier), so it can never
        be in the carve-out — every CLI rejection of a resume WE armed is
        genuine by construction.
        """
        harness.config.session_resume = SessionResumeConfig(
            fallback_storm_threshold=3, storm_window_secs=60,
        )
        q = self._queue()
        # The dedup is only observable if the queue answers "open" once one is
        # filed; _queue()'s stand-in always says False.
        q.has_open_l1 = MagicMock(side_effect=lambda _s: q.submit.call_count > 0)
        harness._escalation_queue = q

        harness.note_resume_failed(self._report(0))
        assert _streak(harness) == 1
        assert q.submit.call_count == 0
        harness.note_resume_failed(self._report(1))
        harness.note_resume_failed(self._report(2, stage='cli', restore=None))

        assert _streak(harness) == 3
        assert q.submit.call_count == 1
        assert q.submit.call_args.args[0].level == 1

        # A fourth keeps counting but files nothing — one open storm L1 at a time.
        harness.note_resume_failed(self._report(3))
        assert _streak(harness) == 4
        assert q.submit.call_count == 1

    @pytest.mark.parametrize('outcome', ['miss', 'disabled'])
    async def test_sink_ignores_by_design_restore_outcomes(
        self, harness: Harness, outcome: str,
    ):
        """A by-design restore outcome neither feeds the streak nor records a
        failure, however many arrive (the D4 carve-out at the arm seam).

        threshold=1 makes the very first GENUINE report fire, so a zero submit
        count across four reports proves these do not feed it at all. 'miss' is
        the archive-COVERAGE signal β assigns to a future RATE watch;
        'disabled' is the restore_from_archive kill switch.
        """
        harness.config.session_resume = SessionResumeConfig(
            fallback_storm_threshold=1, storm_window_secs=60,
        )
        harness._escalation_queue = self._queue()

        for i in range(4):
            harness.note_resume_failed(self._report(i, restore=outcome))
            # Asserted INSIDE the loop: the counter must never transiently
            # rise, not merely end at 0.
            assert _streak(harness) == 0
            assert _chain_stamp(harness) is None
            assert not _recorded_failures(harness)
            assert harness._escalation_queue.submit.call_count == 0

    async def test_success_report_retires_the_run(self, harness: Harness):
        """A corroborated resume that survived resets the streak AND clears the
        chain's comparison stamp and the recorded failures.

        Clearing all three together is what keeps the L1's detail truthful: a
        surviving record from a retired run would name a failure that is no
        longer part of the run being escalated.
        """
        harness.config.session_resume = SessionResumeConfig(
            fallback_storm_threshold=2, storm_window_secs=60,
        )
        harness._escalation_queue = self._queue()

        harness.note_resume_failed(self._report(0))
        assert _streak(harness) == 1
        assert _chain_stamp(harness) is not None
        assert len(_recorded_failures(harness)) == 1

        harness.note_resume_succeeded()
        assert _streak(harness) == 0
        assert _chain_stamp(harness) is None
        assert not _recorded_failures(harness)

        # ...so the next failure opens a FRESH run and never reaches 2.
        harness.note_resume_failed(self._report(1))
        assert _streak(harness) == 1
        assert harness._escalation_queue.submit.call_count == 0

    async def test_sink_run_decays_after_storm_window(self, harness: Harness):
        """A gap of >= storm_window_secs between two genuine reports retires
        the run, so an isolated drip can never accumulate into a false storm.

        The clock is advanced by rewinding the harness's own monotonic stamp,
        NOT by monkeypatching time.monotonic — deterministic, and it perturbs
        no unrelated timer (β's idiom, reused).
        """
        harness.config.session_resume = SessionResumeConfig(
            fallback_storm_threshold=3, storm_window_secs=60,
        )
        harness._escalation_queue = self._queue()

        for i in range(2):
            harness.note_resume_failed(self._report(i))
        assert _streak(harness) == 2
        assert _chain_stamp(harness) is not None

        _rewind_chain(harness, 120)

        harness.note_resume_failed(self._report(2))

        # Decayed to 0, then re-incremented — NOT 3, so no L1 — and the
        # retired run's records went with it.
        assert _streak(harness) == 1
        assert harness._escalation_queue.submit.call_count == 0
        assert [f.session_id for f in _recorded_failures(harness)] == [
            'uuid-rf-2'
        ]

    async def test_a_fresh_boot_carries_no_run(self, harness: Harness):
        """The run is PER-BOOT: a newly constructed Harness starts with no
        streak, no comparison stamp and no recorded failures.

        Asserted structurally as well as by value — the three fields are
        per-INSTANCE state initialised in ``__init__``, not class attributes a
        second orchestrator process (or a second Harness in one interpreter)
        could inherit a half-finished run from.
        """
        assert _streak(harness) == 0
        assert _chain_stamp(harness) is None
        assert not _recorded_failures(harness)

        harness.config.session_resume = SessionResumeConfig(
            fallback_storm_threshold=3, storm_window_secs=60,
        )
        harness._escalation_queue = self._queue()
        harness.note_resume_failed(self._report(0))
        assert _streak(harness) == 1

        for field in ('_session_resume_fallback_streak',
                      '_last_session_resume_fallback_at',
                      '_eligible_but_failed_resumes'):
            assert field in vars(harness), f'{field} must be per-instance state'
            assert not hasattr(Harness, field), (
                f'{field} is a class attribute — a run would leak across boots'
            )

    async def test_sink_is_total(self, harness: Harness):
        """Instrumentation must never break a dispatch (I3).

        Two independent ways it could: a bare harness with no escalation queue
        at the moment the threshold trips, and a malformed report. Both are
        swallowed; neither reaches the caller, which is the production
        ``_invoke`` path.
        """
        harness.config.session_resume = SessionResumeConfig(
            fallback_storm_threshold=1, storm_window_secs=60,
        )
        harness._escalation_queue = None

        # threshold=1 → the first genuine report trips the filer, which must
        # early-return on the absent queue rather than raising.
        harness.note_resume_failed(self._report(0))
        assert _streak(harness) == 1

        harness.note_resume_failed(object())  # type: ignore[arg-type]
        harness.note_resume_succeeded()

    async def test_run_slot_wires_the_harness_as_the_resume_outcome_sink(
        self, harness: Harness, tmp_path: Path,
    ):
        """The production dispatch path is CONNECTED, not merely connectable.

        ``resume_outcome_sink`` is OPTIONAL on ``build_workflow`` — which is
        what keeps eval dispatch unedited and un-drifted, and also what makes
        "nobody ever passed one" a silent, fully-green failure mode: every
        arm-seam report would be dropped on the floor and INV-4's escape would
        be inert again while every other row in this class still passed.

        So this pins the wiring itself: ``_run_slot`` passes the Harness
        ITSELF, the object that owns the streak, the carve-outs and the
        escalation queue.
        """
        harness.config.session_resume = SessionResumeConfig()
        cfg = _make_transcript(tmp_path, 'uuid-sink')
        await _drive_session_slot(
            harness, 'sink1', self._fresh_session('uuid-sink'), config_dir=cfg,
        )

        kwargs = harness._last_build_workflow_kwargs  # type: ignore[attr-defined]
        assert kwargs['resume_outcome_sink'] is harness

    async def test_an_eligible_dispatch_never_retires_a_run_in_progress(
        self, harness: Harness, tmp_path: Path,
    ):
        """ELIGIBILITY IS NOT SURVIVAL: the guard must leave the run alone.

        Eligibility is a PRE-DISPATCH predicate — evaluated one process-phase
        BEFORE the restore whose failure is this task's headline feeder, in the
        SAME dispatch (``build_workflow(..., resume_outcome_sink=self)`` sits a
        few lines below it, and ``TaskWorkflow._invoke``'s restore later
        still). While the guard retired the run here, an archive-backed session
        whose restore then faults ran ``eligible → streak:=0 → fault →
        streak:=1`` forever: serially the streak could never exceed 1 and
        INV-4's escape could never fire, which is exactly the "green, and
        measuring nothing" shape this task exists to end.

        All THREE pieces of run state are asserted, not just the streak: they
        are retired together, so a partial survival would leave a later L1
        naming failures from a run it is not about. The ``session_resume`` emit
        is asserted too — the guard keeps doing its own job; only the
        retirement goes.
        """
        harness.config.session_resume = SessionResumeConfig(
            fallback_storm_threshold=5, storm_window_secs=60,
        )
        harness._escalation_queue = self._queue()
        for i in range(2):
            harness.note_resume_failed(self._report(i))
        run_before = (
            _streak(harness), _chain_stamp(harness), _recorded_failures(harness),
        )
        assert run_before[0] == 2
        assert run_before[1] is not None
        assert len(run_before[2]) == 2

        # No synthetic feeder is armed, so the predicate is the REAL one, and a
        # live transcript makes the session genuinely eligible.
        cfg = _make_transcript(tmp_path, 'uuid-elig')
        await _drive_session_slot(
            harness, 'elig1', self._fresh_session('uuid-elig'), config_dir=cfg,
        )

        assert [et for et, _ in _session_resume_emits(harness)] == [
            EventType.session_resume
        ]
        assert (
            _streak(harness), _chain_stamp(harness), _recorded_failures(harness),
        ) == run_before

    async def test_a_run_of_archive_backed_restore_faults_still_pages(
        self, harness: Harness, tmp_path: Path,
    ):
        """THE COMPOSED REGRESSION — the real guard and the real sink, one
        streak, in production's per-dispatch order.

        Every other row in this class drives ONE of the two seams: the β rows
        drive the predicate through ``_run_slot``, the ε rows drive
        ``note_resume_failed`` directly, and the gate's end-to-end rows drive
        ``_invoke``. None of them can see how the two INTERLEAVE inside a
        single dispatch, and the reset-on-eligible defect lived precisely in
        that gap — every one of them stayed green while production measured
        nothing.

        So this row runs the headline feeder's exact production shape: a
        session eligible BECAUSE δ/3730's hoisted lookup finds an ARCHIVE (the
        live config dir survives and simply does not hold the transcript),
        whose restore then faults and reports back through the very
        ``resume_outcome_sink`` the guard handed ``build_workflow``. The emits
        pin that eligibility really was the path taken, so the row cannot pass
        by quietly falling back instead.
        """
        config = SessionResumeConfig(
            fallback_storm_threshold=3, storm_window_secs=60,
        )
        harness.config.session_resume = config
        harness.config.transcript_archive = TranscriptArchiveConfig()
        harness._escalation_queue = self._queue()

        for i in range(config.fallback_storm_threshold):
            task_id, session_id = f'af{i}', f'uuid-arch-fault-{i}'
            _make_archive(harness.config.project_root, task_id, session_id)
            empty_cfg = tmp_path / f'claude-config-empty-{i}'
            (empty_cfg / 'projects').mkdir(parents=True)
            await _drive_session_slot(
                harness, task_id, self._fresh_session(session_id),
                config_dir=empty_cfg,
                workflow_reports=(
                    lambda sink, n=i: sink.note_resume_failed(self._report(n))
                ),
            )

        assert [et for et, _ in _session_resume_emits(harness)] == (
            [EventType.session_resume] * config.fallback_storm_threshold
        )
        assert _streak(harness) == config.fallback_storm_threshold
        assert harness._escalation_queue.submit.call_count == 1

    async def test_the_shipped_window_admits_the_population_that_exists(
        self, harness: Harness,
    ):
        """At the SHIPPED defaults, a run arriving at the fleet's MEASURED
        spacing still chains and files the L1 (task ε/3733).

        The other storm rows all pick a small window and drive it, which proves
        the mechanism but says nothing about whether the SHIPPED number can
        ever fire. At the pre-ε 3600 s it could not: the smallest interval
        between two eligible-but-FAILED resumes the fleet has ever produced is
        5.82 h, so no two of them chained at ANY threshold and INV-4's escape
        was green and unfireable.

        The spacing is read from ``storm_window_bound.MEASURED_MIN_GAP_SECS``
        rather than re-typed here, so this row and the derivation cannot drift
        (SPOT), and the loop count is the shipped threshold — no arithmetic on
        literals the test controls on both sides. Both halves run the REAL
        rolling-window decay: the clock is advanced by rewinding the harness's
        own monotonic stamp, β's idiom.
        """
        from orchestrator.storm_window_bound import (  # noqa: PLC0415
            MEASURED_MIN_GAP_SECS,
        )

        config = SessionResumeConfig()   # SHIPPED defaults, no overrides
        harness.config.session_resume = config
        harness._escalation_queue = self._queue()

        for i in range(config.fallback_storm_threshold):
            if i:
                _rewind_chain(harness, MEASURED_MIN_GAP_SECS)
            harness.note_resume_failed(self._report(i))

        assert _streak(harness) == (
            config.fallback_storm_threshold
        )
        assert harness._escalation_queue.submit.call_count == 1

        # ...and the decay is still LIVE, not merely wide: the same run spaced
        # a full window apart never chains, so the alarm did not become a
        # cumulative per-boot counter on the way to being fireable.
        harness.note_resume_succeeded()
        harness._escalation_queue = self._queue()
        for i in range(config.fallback_storm_threshold):
            if i:
                _rewind_chain(harness, config.storm_window_secs)
            harness.note_resume_failed(self._report(100 + i))

        assert _streak(harness) == 1
        assert harness._escalation_queue.submit.call_count == 0

    async def test_storm_l1_names_every_recorded_failure(self, harness: Harness):
        """INV-2 structured-facts-at-failure: the L1 names what actually failed.

        Every recorded field of every failure in the run appears VERBATIM, so
        an operator can go straight to the archive root and the session that
        broke instead of reconstructing the run from a census. The run here
        deliberately varies every dimension — two stages, two restore outcomes,
        three roles, three tasks — because a renderer that collapsed any of
        them would still pass a single-failure assertion.

        A missing archive_root/archive_path is rendered as an explicit "none
        located", never as a bare ``None`` and never by dropping the line: the
        difference between "the archive was checked and held nothing" and "the
        lookup itself faulted" is exactly what an operator needs, and a
        silently absent line reads as the former.
        """
        harness.config.session_resume = SessionResumeConfig(
            fallback_storm_threshold=3, storm_window_secs=60,
        )
        harness._escalation_queue = self._queue()

        reports = [
            self._report(
                0, task_id='4101', session_id='uuid-aa', role='implementer',
                stage='pre_flight', restore='fault',
                archive_root='/srv/archive-aa',
                archive_path='/srv/archive-aa/4101/uuid-aa.jsonl.gz',
                detail='OSError: [Errno 28] No space left on device',
            ),
            # Archive-root COMPOSITION faulted, so neither path was ever
            # located — the best-effort lookup must not masquerade as a miss.
            self._report(
                1, task_id='4102', session_id='uuid-bb', role='architect',
                stage='pre_flight', restore='fault',
                archive_root=None, archive_path=None,
                detail='TypeError: unsupported operand type(s) for /',
            ),
            # A cli-stage rejection: no restore outcome at all (it ran a phase
            # earlier), which is why every one of these is genuine.
            self._report(
                2, task_id='4103', session_id='uuid-cc', role='reviewer',
                stage='cli', restore=None,
                archive_root='/srv/archive-cc', archive_path=None,
                detail='CLI rejected the armed session after 1 fallback',
            ),
        ]
        for report in reports:
            harness.note_resume_failed(report)

        assert harness._escalation_queue.has_open_l1.called, (
            'the dedup must still be consulted — one open storm L1 at a time'
        )
        esc = harness._escalation_queue.submit.call_args.args[0]
        assert esc.level == 1
        assert 'session-resume' in esc.summary.lower()
        assert 'storm' in esc.summary.lower()

        for report in reports:
            for value in (report.task_id, report.session_id, report.role,
                          report.stage, report.restore, report.archive_root,
                          report.archive_path, report.detail):
                if value is not None:
                    assert value in esc.detail, f'{value!r} missing from detail'

        # Three of the six archive fields above are absent; each says so.
        assert esc.detail.count('none located') == 3
        assert 'None' not in esc.detail, (
            'a raw None reached the operator-facing detail'
        )

    async def test_storm_l1_never_sends_the_operator_to_ntp(self, harness: Harness):
        """The misdirection task 3728 removed must not come back (INV-2).

        Pinned where the misdirection would actually reach an operator: in the
        text of an L1 that was really filed. Neither its detail nor its
        suggested action may mention clock skew or NTP — the streak's feeder is
        archive-restore failure, which has nothing to do with either.

        Deliberately NOT pinned by searching harness.py's source for the
        literal that used to carry it. That assertion matched raw module text,
        comments included, so it failed in both directions at once: harness.py
        discusses clock skew on purpose (the monotonic-clock rationale in the
        rolling-window decay, and in ``note_resume_failed``), so a legitimate
        comment would have turned the suite red for no defect, while any
        reworded misdirection — "check NTP sync", "the wall clock may have
        jumped" — is the regression this row exists to stop and would have
        stayed green. The task's delivered-check gate is where that literal's
        absence is enforced, and it runs on every task.
        """
        harness.config.session_resume = SessionResumeConfig(
            fallback_storm_threshold=1, storm_window_secs=60,
        )
        harness._escalation_queue = self._queue()
        harness.note_resume_failed(self._report(0))

        esc = harness._escalation_queue.submit.call_args.args[0]
        operator_text = f'{esc.summary}\n{esc.detail}\n{esc.suggested_action}'
        assert not re.search('clock skew|NTP', operator_text, re.IGNORECASE)


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

    ``<project_root>/<TranscriptArchiveConfig.root>/<task_id>/<enc>/
    <session_id>.jsonl.gz`` — the path shared.transcript_archive._archive_one
    writes and durable_archive_path globs.

    The root is READ OFF the config default rather than hardcoded: a change to
    ``TranscriptArchiveConfig.root`` would otherwise silently desynchronise
    this helper from the lookup under test, and every row here would fail as a
    baffling ``archive_available is False`` instead of at the seam that moved.
    """
    dest = (
        project_root
        / TranscriptArchiveConfig().root
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
    from the durable transcript archive. Task 3727 added it as instrumentation
    ONLY (D8 / INV-3 instrument-before-acting) — measure the recoverable
    population before gating on it — and these rows pinned that it moved
    nothing.

    TASK 3730 (δ) DELIBERATELY ENDS THAT, which is the whole point of the
    instrument-then-act sequence: the measured signal (~91% of post-3578
    fallbacks were recoverable, 92 of 101 on 2026-09-04) is now an ELIGIBILITY
    input, so an archive-backed session that used to fall back is adopted.
    Four rows below therefore assert the OPPOSITE of what they asserted under
    3727, each saying so in its own docstring; they are updated rather than
    deleted because the population each one describes is exactly the
    population δ moves, and a row that watched it fall back is the right place
    to record that it no longer does.

    What survives unchanged from 3727: the field is still emitted on the
    fallback branch ONLY (never on session_resume / session_resume_capped), it
    is still a real JSON bool, and it is still False-on-fault so a broken
    lookup cannot break dispatch. What is NEW under δ is that the lookup is
    hoisted into the guard and happens EXACTLY ONCE per dispatch — the value
    the predicate consumed and the value the event reports are structurally
    the same bool — and that the disabled path still pays no filesystem I/O.
    """

    async def test_archive_present_now_resumes_instead_of_falling_back(
        self, harness: Harness, tmp_path: Path
    ):
        """THE POPULATION 3727 COUNTED AND δ MOVES (task 3730 / D2).

        Archive PRESENT under a foreign lane, live transcript gone: under 3727
        this emitted ``session_resume_fallback`` with ``reasons=['no_transcript']``
        and ``archive_available: true`` — an ineligible dispatch that the
        instrument could see was recoverable and was not allowed to act on.
        The measurement it produced (92 of 101 such fallbacks, 2026-09-04) is
        what authorised δ, so the row now asserts the conversion: the same
        inputs are ADOPTED, and no fallback is emitted at all.

        A surviving-but-empty config dir rather than an absent one, so this
        row is distinct from the guard's crash-recovery-shape row: it pins
        that reachability is answered by the archive even when the live dir is
        present and simply does not hold this session's transcript.
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

        assert resume_id is session
        emits = _session_resume_emits(harness)
        assert [et for et, _ in emits] == [EventType.session_resume]
        # The eligible event stays byte-identical (D8's surviving half): the
        # field rides the fallback branch only, so event_store.py's ratio
        # recipe keeps its denominator.
        assert 'archive_available' not in emits[0][1]['data']
        assert _streak(harness) == 0

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
        assert kwargs['data']['reasons'] == ['no_transcript']
        assert kwargs['data']['archive_available'] is False
        assert _streak(harness) == 0  # by design (3728)

    async def test_stale_also_carries_the_field(
        self, harness: Harness, tmp_path: Path
    ):
        """reason == 'stale' carries it too — it rides the BRANCH, not one reason.

        RETARGETED BY δ (task 3730), deliberately and not incidentally. This
        row used to drive an AGE-derived 'stale' with the archive present,
        which is precisely the combination D2 now makes eligible — so keeping
        it would have asserted the defect δ removes. It drives the OTHER
        'stale' instead: an UNPARSEABLE ``started_at``, the one the archive
        never suppresses, because an undateable session cannot be bounded by
        the absolute backstop either (fail-safe direction: cannot date it,
        cannot resume it).

        That keeps the row's original claim exactly — the field rides the
        fallback BRANCH rather than any one reason, so it is present on a
        'stale' emit and not only on a corroboration failure — while making it
        a live statement about δ rather than a fossil of pre-δ behaviour.
        """
        session = {
            'session_id': 'uuid-arch-stale',
            'role': 'implementer',
            'started_at': 'not-a-date',  # undateable: 'stale', never suppressed
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
        assert kwargs['data']['reasons'] == ['stale']
        assert kwargs['data']['archive_available'] is True
        assert _streak(harness) == 0  # by design (3728)

    async def test_reseeded_lane_with_an_archive_is_now_adopted(
        self, harness: Harness, tmp_path: Path
    ):
        """A RESEEDED lane whose transcript survives in the archive resumes (δ).

        3727 called this "the branch where the field matters MOST: a reseeded
        lane is precisely the population task 3619 will move". δ is what moves
        it. Warm-lane acquire always re-seeds from base, wiping
        ``<lane>/.task/`` and the whole live transcript store with it — but
        the archival pass copied the transcript OUT of that store before the
        wipe, so the session is still reachable and the wipe is no longer a
        reason to dispatch fresh.

        The reseeded/no_transcript discrimination itself is untouched and
        still pinned by :meth:`test_reseeded_reports_absent_archive`, which
        drives the same wiped-lane shape with an EMPTY archive: that is where
        the split still decides the answer.
        """
        session = {
            'session_id': 'uuid-arch-reseed',
            'role': 'implementer',
            'started_at': datetime.now(UTC).isoformat(),
            'resume_count': 0,
        }
        gone = tmp_path / 'gone-arch' / 'claude-config-x'
        assert not gone.exists()  # the reseed already swept the lane
        harness.config.session_resume = SessionResumeConfig()
        harness.config.transcript_archive = TranscriptArchiveConfig()
        _make_archive(harness.config.project_root, 'ar4', 'uuid-arch-reseed')

        resume_id = await _drive_session_slot(harness, 'ar4', session, config_dir=gone)

        assert resume_id is session
        emits = _session_resume_emits(harness)
        assert [et for et, _ in emits] == [EventType.session_resume]
        assert _streak(harness) == 0

    async def test_reseeded_reports_absent_archive(
        self, harness: Harness, tmp_path: Path
    ):
        """Same branch, empty archive root → False."""
        session = {
            'session_id': 'uuid-arch-reseed-no',
            'role': 'implementer',
            'started_at': datetime.now(UTC).isoformat(),
            'resume_count': 0,
        }
        gone = tmp_path / 'gone-arch-no' / 'claude-config-x'
        harness.config.session_resume = SessionResumeConfig()
        harness.config.transcript_archive = TranscriptArchiveConfig()

        resume_id = await _drive_session_slot(harness, 'ar5', session, config_dir=gone)

        assert resume_id is None
        emits = _session_resume_emits(harness)
        assert len(emits) == 1
        et, kwargs = emits[0]
        assert et == EventType.session_resume_fallback
        assert kwargs['data']['reasons'] == ['reseeded']
        assert kwargs['data']['archive_available'] is False

    async def test_unconfigured_transcript_archive_degrades_to_false(
        self, harness: Harness, tmp_path: Path
    ):
        """FAIL-SAFE: a config regression must not become a dispatch fault.

        ``transcript_archive`` is left as the bare spec_set MagicMock the
        conftest fixture yields (it never assigns the field). _run_slot must
        still complete, the fallback must still be emitted with its correct
        reason, and the instrument must read False — never propagate.

        MECHANISM, measured rather than assumed (and NOT the one the original
        plan rationale asserted): ``project_root / <MagicMock>.root`` does not
        raise. ``MagicMock`` implements ``__fspath__``, so the composition
        succeeds into a nonsense-but-well-formed path
        (``<project_root>/MagicMock/mock.root/<id>``) that simply matches
        nothing, and the lookup returns None. The genuinely-raising composition
        is pinned separately by the ``project_root = None`` rows in
        :meth:`test_archive_available_is_total_against_broken_config` and
        :meth:`test_instrument_fault_is_reported_loudly_exactly_once`. Both
        routes have to land on False, which is what this row and those rows
        together establish.
        """
        session = {
            'session_id': 'uuid-arch-mock',
            'role': 'implementer',
            'started_at': datetime.now(UTC).isoformat(),
            'resume_count': 0,
        }
        empty_cfg = tmp_path / 'claude-config-empty-mock'
        (empty_cfg / 'projects').mkdir(parents=True)
        harness.config.session_resume = SessionResumeConfig()
        # DELIBERATELY not assigning harness.config.transcript_archive.

        resume_id = await _drive_session_slot(
            harness, 'ar6', session, config_dir=empty_cfg
        )

        assert resume_id is None
        emits = _session_resume_emits(harness)
        assert len(emits) == 1
        et, kwargs = emits[0]
        assert et == EventType.session_resume_fallback
        assert kwargs['data']['reasons'] == ['no_transcript']
        assert kwargs['data']['archive_available'] is False

    async def test_archive_available_is_total_against_broken_config(
        self, harness: Harness
    ):
        """The helper itself is total, called directly, on a broken config."""
        # Bare conftest MagicMock for transcript_archive: composes (MagicMock is
        # os.PathLike) into a path that matches nothing — no raise, still False.
        assert harness._archive_available('42', 'sid') is False

        # project_root = None: the composition itself raises TypeError. THIS is
        # the row that exercises the guard, not the MagicMock one above.
        harness.config.transcript_archive = TranscriptArchiveConfig()
        harness.config.project_root = None  # type: ignore[assignment]
        assert harness._archive_available('42', 'sid') is False

    async def test_instrument_fault_is_reported_loudly_exactly_once(
        self, harness: Harness, caplog
    ):
        """A faulted instrument must not be indistinguishable from an empty archive.

        False-on-fault is the right DISPATCH behaviour, but reporting it
        silently is the silent degradation INV-2/INV-4 forbid: "no archive" and
        "the lookup is broken" would both read ``archive_available: false``, so
        a config regression would render the measurement this task exists to
        produce as a confident "0% recoverable" with nothing above DEBUG
        dissenting — on the very signal task 3619 is judged by.

        The faults reaching this handler are the ones durable_archive_path
        CANNOT log (it logs its own at WARNING): they fail before the lookup, in
        the archive-root composition, and are persistent rather than transient.
        Hence loud, and hence rate-limited to once per process — a persistent
        fault fires on every dispatch, and a fallback storm must not become a
        log flood.
        """
        # A project_root the composition genuinely cannot divide. MEASURED, not
        # assumed: the bare conftest MagicMock does NOT fault here (see
        # test_unconfigured_transcript_archive_degrades_to_false), so using it
        # would make this test vacuous.
        harness.config.transcript_archive = TranscriptArchiveConfig()
        harness.config.project_root = None  # type: ignore[assignment]

        with caplog.at_level(logging.DEBUG, logger='orchestrator.harness'):
            assert harness._archive_available('42', 'sid-1') is False
            assert harness._archive_available('42', 'sid-2') is False
            assert harness._archive_available('42', 'sid-3') is False

        warnings = [
            r for r in caplog.records
            if r.levelno >= logging.WARNING and 'archive_available' in r.getMessage()
        ]
        assert len(warnings) == 1, (
            f'expected exactly one WARNING across 3 faults, got {len(warnings)}'
        )
        # It has to be actionable on its own: name the field so an operator
        # greps it, and say the reported rate is not to be trusted.
        msg = warnings[0].getMessage()
        assert 'archive_available' in msg
        assert 'sid-1' in msg  # the FIRST fault is the one that speaks
        assert warnings[0].exc_info is not None  # the traceback is retained

        # The suppressed repeats are still recoverable at DEBUG, not dropped.
        debugs = [
            r for r in caplog.records
            if r.levelno == logging.DEBUG and 'archive_available' in r.getMessage()
        ]
        assert len(debugs) == 2

    async def test_disabled_archival_still_reports_an_existing_archive(
        self, harness: Harness, tmp_path: Path
    ):
        """The FILESYSTEM is the source of truth, not ``transcript_archive.enabled``.

        Pins the docstring's explicit design decision, which nothing else
        covered: adding an ``if not ...enabled: return False`` short-circuit
        looks like a free optimisation and would still leave the whole suite
        green, while silently inverting the contract. Turning archival OFF
        today does not un-archive what was written while it was ON, and those
        sessions are exactly the recoverable population this lookup exists to
        find — a config-derived answer would bias the measurement toward
        "nothing is recoverable" and mislead an operator triaging the storm L1.

        δ RAISES THE STAKES rather than changing the property (task 3730). The
        lookup is now an ELIGIBILITY input, so the short-circuit would cost
        real resumes rather than only a wrong telemetry field: flipping
        ``transcript_archive.enabled`` off would retroactively make every
        already-archived session ineligible. The observable therefore moves
        from ``archive_available is True`` on a fallback to the session being
        ADOPTED — a strictly louder statement of the same contract.
        """
        session = {
            'session_id': 'uuid-arch-disabled',
            'role': 'implementer',
            'started_at': datetime.now(UTC).isoformat(),
            'resume_count': 0,
        }
        empty_cfg = tmp_path / 'claude-config-empty-disabled'
        (empty_cfg / 'projects').mkdir(parents=True)
        harness.config.session_resume = SessionResumeConfig()
        harness.config.transcript_archive = TranscriptArchiveConfig(enabled=False)
        _make_archive(harness.config.project_root, 'ar11', 'uuid-arch-disabled')

        resume_id = await _drive_session_slot(
            harness, 'ar11', session, config_dir=empty_cfg
        )

        assert resume_id is session
        emits = _session_resume_emits(harness)
        assert [et for et, _ in emits] == [EventType.session_resume]

    async def test_null_session_id_reports_false_without_raising(
        self, harness: Harness, tmp_path: Path
    ):
        """A recovered session with no session_id still emits, reporting False."""
        session = {
            'session_id': None,
            'role': 'implementer',
            'started_at': datetime.now(UTC).isoformat(),
            'resume_count': 0,
        }
        cfg = _make_transcript(tmp_path, 'uuid-unrelated')
        harness.config.session_resume = SessionResumeConfig()
        harness.config.transcript_archive = TranscriptArchiveConfig()

        resume_id = await _drive_session_slot(harness, 'ar7', session, config_dir=cfg)

        assert resume_id is None
        emits = _session_resume_emits(harness)
        assert len(emits) == 1
        et, kwargs = emits[0]
        assert et == EventType.session_resume_fallback
        assert kwargs['data']['reasons'] == ['no_transcript']
        assert kwargs['data']['archive_available'] is False

    async def test_eligible_and_capped_events_do_not_carry_the_field(
        self, harness: Harness, tmp_path: Path
    ):
        """D8 blast radius: the instrument went onto EXACTLY one event.

        event_store.py's documented ratio recipe reads attempts as the sum of
        the three outcome events; session_resume and session_resume_capped
        must stay byte-identical so that denominator keeps its meaning — and
        so neither path pays for a filesystem glob it does not use.
        """
        harness.config.session_resume = SessionResumeConfig()
        harness.config.transcript_archive = TranscriptArchiveConfig()

        # Eligible → session_resume.
        elig = {
            'session_id': 'uuid-arch-elig',
            'role': 'implementer',
            'started_at': datetime.now(UTC).isoformat(),
            'resume_count': 0,
        }
        cfg = _make_transcript(tmp_path, 'uuid-arch-elig')
        _make_archive(harness.config.project_root, 'ar8', 'uuid-arch-elig')
        assert await _drive_session_slot(harness, 'ar8', elig, config_dir=cfg) is elig

        # Capped → session_resume_capped.
        real = SessionResumeConfig()
        capped = {
            'session_id': 'uuid-arch-capped',
            'role': 'implementer',
            'started_at': datetime.now(UTC).isoformat(),
            'resume_count': real.max_resumes_per_task,
        }
        cfg2 = _make_transcript(tmp_path, 'uuid-arch-capped')
        _make_archive(harness.config.project_root, 'ar9', 'uuid-arch-capped')
        assert await _drive_session_slot(harness, 'ar9', capped, config_dir=cfg2) is None

        emits = _session_resume_emits(harness)
        assert [et for et, _ in emits] == [
            EventType.session_resume,
            EventType.session_resume_capped,
        ]
        for _et, kwargs in emits:
            assert 'archive_available' not in kwargs['data']

    async def test_one_archive_lookup_per_dispatch_feeds_both_consumers(
        self, harness: Harness
    ):
        """δ WIRING (task 3730 / D-hoist): ONE lookup per dispatch, and the
        value the predicate consumed IS the value the event reports.

        Before δ the fallback emit did its own ``_archive_available`` call,
        independent of the predicate (which did none at all). Leaving it that
        way once the predicate gates on the archive would mean TWO lookups of
        the same fact per dispatch, and — because an archival pass can land
        between them — a ``session_resume_fallback`` whose
        ``archive_available`` contradicts the ``reasons`` printed beside it.
        An operator reading ``archive_available: true`` next to
        ``reasons: ['no_transcript']`` would be looking at a state δ makes
        impossible, with no way to tell it was a race.

        So this row counts the calls (exactly one) AND compares the two
        consumers' views of the same bool. Counting alone is not enough: a
        single lookup wired to only ONE of the two consumers, with the other
        left on a hardcoded default, would also count one.
        """
        cfg = SessionResumeConfig()
        harness.config.session_resume = cfg
        harness.config.transcript_archive = TranscriptArchiveConfig()
        session = {
            'session_id': 'uuid-arch-once',
            'role': 'implementer',
            'started_at': (
                datetime.now(UTC) - timedelta(seconds=2 * cfg.absolute_resume_age_secs)
            ).isoformat(),
            'resume_count': 0,
        }
        # Aged OUT, so the dispatch still reaches the fallback emit and the
        # event is observable at all — an adopted session emits no field.
        _make_archive(harness.config.project_root, 'ar12', 'uuid-arch-once')

        lookups: list[tuple] = []
        real_lookup = harness._archive_available

        def counting_lookup(task_id, session_id):
            lookups.append((task_id, session_id))
            return real_lookup(task_id, session_id)

        consumed: list[object] = []
        real_reasons = harness._session_resume_reasons

        def spying_reasons(session_arg, config_dir, *, archive_available):
            consumed.append(archive_available)
            return real_reasons(
                session_arg, config_dir, archive_available=archive_available
            )

        harness._archive_available = counting_lookup  # type: ignore[method-assign]
        harness._session_resume_reasons = spying_reasons  # type: ignore[method-assign]

        resume_id = await _drive_session_slot(harness, 'ar12', session)

        assert resume_id is None
        assert lookups == [('ar12', 'uuid-arch-once')], (
            f'expected exactly one hoisted archive lookup, got {lookups!r}'
        )
        emits = _session_resume_emits(harness)
        assert len(emits) == 1
        et, kwargs = emits[0]
        assert et == EventType.session_resume_fallback
        assert kwargs['data']['reasons'] == ['aged_out']
        # The predicate saw it, the event reports it, and they are the SAME
        # bool — not merely equal by luck of a second lookup agreeing.
        assert consumed == [True]
        assert kwargs['data']['archive_available'] is consumed[0]

    async def test_kill_switch_does_no_archive_lookup_at_all(
        self, harness: Harness, tmp_path: Path
    ):
        """B6, PRESERVED under δ: with ``session_resume.enabled`` False the
        hoisted lookup never runs — zero filesystem I/O, as today.

        The lookup is on the dispatch path now, not only on the fallback emit,
        so hoisting it carelessly (above the ``enabled`` check rather than
        inside it) would make the kill switch cost a glob per dispatch for a
        feature that is switched off. The predicate returns ``{'disabled'}``
        alone without consulting the archive, so there is nothing for the
        lookup to inform.

        This is also the one D8 zero-I/O property δ genuinely preserves: the
        ELIGIBLE path now pays one glob, and the plan records that as a stated
        regression rather than an oversight.
        """
        session = {
            'session_id': 'uuid-arch-killed',
            'role': 'implementer',
            'started_at': datetime.now(UTC).isoformat(),
            'resume_count': 0,
        }
        cfg = _make_transcript(tmp_path, 'uuid-arch-killed')
        harness.config.session_resume = SessionResumeConfig(enabled=False)
        harness.config.transcript_archive = TranscriptArchiveConfig()
        _make_archive(harness.config.project_root, 'ar13', 'uuid-arch-killed')

        lookups: list[tuple] = []
        real_lookup = harness._archive_available

        def counting_lookup(task_id, session_id):
            lookups.append((task_id, session_id))
            return real_lookup(task_id, session_id)

        harness._archive_available = counting_lookup  # type: ignore[method-assign]

        resume_id = await _drive_session_slot(harness, 'ar13', session, config_dir=cfg)

        assert resume_id is None
        assert lookups == [], (
            f'the disabled path must touch the filesystem 0 times, saw {lookups!r}'
        )
        assert _session_resume_emits(harness) == []
