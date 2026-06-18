"""Tests for crash recovery — surviving worktree detection and plan injection."""

import json
from pathlib import Path
from unittest.mock import AsyncMock, MagicMock, patch

import pytest

from orchestrator.event_store import EventType
from orchestrator.harness import Harness
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
    h.scheduler._dispatched = set()

    # Replace git_ops cleanup/quarantine with async mocks; keep worktree_base real
    h.git_ops.worktree_base = (tmp_path / '.worktrees').resolve()
    h.git_ops.cleanup_worktree = AsyncMock()
    h.git_ops.quarantine_worktree = AsyncMock(return_value=None)
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
        # Don't create the worktree base dir
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
        pool = _attach_pool(harness, size=2)
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
        pool = _attach_pool(harness, size=2)
        base = harness.git_ops.worktree_base
        plan = _make_plan(steps_done=3, steps_total=5, task_id='42')
        _setup_lane(base, '_lane-0', plan)

        await harness._recover_crashed_tasks()

        harness.git_ops.cleanup_worktree.assert_not_called()  # type: ignore[attr-defined]

    async def test_warm_lane_pool_assignment_restored(self, harness: Harness):
        """After recovery, pool.assignment_for('42') == base/'_lane-0'."""
        pool = _attach_pool(harness, size=2)
        base = harness.git_ops.worktree_base
        plan = _make_plan(steps_done=3, steps_total=5, task_id='42')
        lane_path = _setup_lane(base, '_lane-0', plan)

        await harness._recover_crashed_tasks()

        assert pool.assignment_for('42') == lane_path

    async def test_warm_lane_pool_state_assigned(self, harness: Harness):
        """After recovery, pool.state(base/'_lane-0') == LaneState.ASSIGNED."""
        pool = _attach_pool(harness, size=2)
        base = harness.git_ops.worktree_base
        plan = _make_plan(steps_done=3, steps_total=5, task_id='42')
        lane_path = _setup_lane(base, '_lane-0', plan)

        await harness._recover_crashed_tasks()

        assert pool.state(lane_path) == LaneState.ASSIGNED

    async def test_warm_lane_cold_path_unaffected(self, harness: Harness):
        """Cold (non-lane) worktrees still recover normally alongside lane dirs."""
        pool = _attach_pool(harness, size=2)
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
