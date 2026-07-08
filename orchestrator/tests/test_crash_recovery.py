"""Tests for crash recovery — surviving worktree detection and plan injection."""

import json
import logging
import shutil
from pathlib import Path
from unittest.mock import AsyncMock, MagicMock, patch

import pytest

from orchestrator.artifacts import TaskArtifacts
from orchestrator.event_store import EventType
from orchestrator.harness import Harness
from orchestrator.lane_lifecycle import LaneLifecycle
from orchestrator.lane_lifecycle import LaneState as DurableLaneState
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

        with patch('orchestrator.harness.TaskWorkflow') as MockWorkflow:
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

        with patch('orchestrator.harness.TaskWorkflow') as MockWorkflow:
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

        with patch('orchestrator.harness.TaskWorkflow') as MockWorkflow:
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

    async def test_run_slot_passes_recovered_session(self, harness: Harness):
        """A recovered session dict flows through to TaskWorkflow as resume_session_id."""
        session_dict = {
            'session_id': 'uuid-resume-me',
            'role': 'implementer',
            'started_at': '2026-05-12T10:00:00+00:00',
            'owner_pid': 9999,
        }
        harness._recovered_sessions['55'] = session_dict
        harness._preserved_worktrees.add('55')

        assignment = MagicMock()
        assignment.task_id = '55'
        assignment.task = {'title': 'Resumable task'}

        sem = MagicMock()
        sem.release = MagicMock()

        with patch('orchestrator.harness.TaskWorkflow') as MockWorkflow:
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


@pytest.mark.asyncio
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
        release_warm_lane -> _lifecycle_note_released does for a real warm
        lane) BEFORE the harness's own explicit transition runs, so an
        unconditional second RELEASED -> RELEASED transition would raise
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
            lambda entry, tid: harness.git_ops._lifecycle_note_released(entry)
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


# ===========================================================================
# Task 2257 (W11 delta) step-11 RED: compat (never-silently-re-pin a
# record-less lane) + .task-meta read relocation.
# ===========================================================================


@pytest.mark.asyncio
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
