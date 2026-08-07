"""Tests for the post-_wait_for_resolution status preservation guard (Fix 1).

When the steward resolves an L0 escalation by setting the task to a terminal
or preserve status (done / cancelled / deferred / blocked), the orchestrator
must NOT resume the implementer.  Without this guard, the resume loop kept
invoking the implementer/debugger until verify-attempt budget exhausted
(see workflow.py ESCALATED branch in run()).

The guard mirrors the inline returns inside _mark_blocked (~L3125–3168) but
runs on the hot resume path so the steward's terminal decision is honored
before the orchestrator burns another iteration's worth of budget.
"""

from __future__ import annotations

import asyncio
import logging
from pathlib import Path
from unittest.mock import AsyncMock, MagicMock, patch

import pytest
from _workflow_helpers import (
    FakeBriefing,
    FakeMcp,
    FakeScheduler,
    _make_resolving_steward,
    _make_status_setting_steward,
)
from escalation.models import Escalation
from escalation.queue import EscalationQueue

from orchestrator.agents.invoke import AgentResult
from orchestrator.artifacts import TaskArtifacts
from orchestrator.config import GitConfig, OrchestratorConfig
from orchestrator.git_ops import GitOps, _run
from orchestrator.scheduler import TaskAssignment, files_to_modules
from orchestrator.workflow import (
    IMPLEMENTER,
    TaskWorkflow,
    WorkflowOutcome,
    WorkflowState,
    _PriorImplStatus,
)
from orchestrator.workflow_types import StewardResolved

# ---------------------------------------------------------------------------
# Fixtures (kept local — these tests don't share runtime with test_workflow_e2e)
# ---------------------------------------------------------------------------


def _same_module_siblings(lock_depth: int) -> tuple[str, str]:
    """Two DISTINCT file paths guaranteed to share a module at *lock_depth*.

    ``shared.locking.normalize_lock`` truncates a path to its first ``depth``
    components, so two files share a lock module iff those components agree.
    A fixed literal like ``a/b/c/d/{e,f}.py`` satisfies that only while
    ``lock_depth <= 4``: at a deeper setting the two paths normalize to
    themselves and become DIFFERENT modules, making the "same-module widen"
    premise unsatisfiable.

    That is not hypothetical — it is why these tests went red on main when
    ``dark-factory-orchestrator.yaml`` moved ``lock_depth`` 4 -> 12 (commit
    094d634465, deliberately making module locks file-granular). The autouse
    ``_isolate_orch_config`` fixture in conftest.py pins ``ORCH_CONFIG_PATH``
    at the LIVE operational config, so ``config.lock_depth`` here is the real
    deployed value by design, not a code default.

    Deriving the package prefix FROM ``lock_depth`` keeps the premise true BY
    CONSTRUCTION at any depth, so these stay tests of same-module widen
    behaviour instead of silently becoming tests of the knob's current value.
    """
    package = '/'.join(f'p{i}' for i in range(lock_depth))
    return f'{package}/e.py', f'{package}/f.py'


@pytest.fixture
def git_repo(tmp_path: Path) -> Path:
    repo = tmp_path / 'repo'
    repo.mkdir()
    asyncio.run(_init_repo(repo))
    return repo


async def _init_repo(repo: Path) -> None:
    await _run(['git', 'init', '-b', 'main'], cwd=repo)
    await _run(['git', 'config', 'user.email', 'test@test.com'], cwd=repo)
    await _run(['git', 'config', 'user.name', 'Test'], cwd=repo)
    (repo / 'lib.py').write_text('def greet(name): return name\n')
    await _run(['git', 'add', '-A'], cwd=repo)
    await _run(['git', 'commit', '-m', 'Initial commit'], cwd=repo)


@pytest.fixture
def config(git_repo: Path) -> OrchestratorConfig:
    return OrchestratorConfig(
        project_root=git_repo,
        max_concurrent_tasks=1,
        git=GitConfig(
            main_branch='main',
            branch_prefix='task/',
            remote='origin',
            worktree_dir='.worktrees',
        ),
    )


@pytest.fixture
def git_ops(config: OrchestratorConfig) -> GitOps:
    return GitOps(config.git, config.project_root)


@pytest.fixture
def task_assignment() -> TaskAssignment:
    return TaskAssignment(
        task_id='42',
        task={
            'id': '42',
            'title': 'X',
            'description': 'Y',
            'status': 'pending',
            'metadata': {'files': ['lib']},
            'dependencies': [],
        },
        modules=['lib'],
    )


PLAN = {
    'task_id': '42',
    'title': 'X',
    'files': ['lib.py'],
    'analysis': '',
    'prerequisites': [],
    'steps': [
        {
            'id': 'step-1',
            'type': 'impl',
            'description': '',
            'status': 'pending',
            'commit': None,
        },
    ],
    'design_decisions': [],
    'reuse': [],
}


# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------


async def _make_advanced_worktree(git_ops: GitOps, task_id: str) -> Path:
    """Create a worktree on a fresh task branch and add a commit so the
    branch is ONE commit ahead of main.

    Without this, wt_head == main_sha and the already-on-main short-circuit
    fires before Fix 1's status check, masking the behaviour under test.
    """
    wt_info = await git_ops.create_worktree(task_id)
    wt = wt_info.path
    (wt / 'precommit.txt').write_text('test marker\n')
    await _run(['git', 'add', 'precommit.txt'], cwd=wt)
    await _run(['git', 'commit', '-m', 'test marker commit'], cwd=wt)
    return wt


def _build_workflow(
    config: OrchestratorConfig,
    git_ops: GitOps,
    assignment: TaskAssignment,
    queue: EscalationQueue,
    worktree: Path,
) -> tuple[TaskWorkflow, FakeScheduler]:
    """Wire a TaskWorkflow with all fakes for these tests.

    Worktree is set externally (eval-mode path) so run() skips create_worktree
    AND skips merge phase (post-success path runs unconditionally though).
    """
    scheduler = FakeScheduler()
    workflow = TaskWorkflow(
        assignment=assignment,
        config=config,
        git_ops=git_ops,
        scheduler=scheduler,  # type: ignore[arg-type]
        briefing=FakeBriefing(),  # type: ignore[arg-type]
        mcp=FakeMcp(),  # type: ignore[arg-type]
        escalation_queue=queue,
        initial_plan=dict(PLAN),
    )
    # Pre-set worktree so run() goes through the eval-mode external path.
    # _worktree_external is then set to True inside run() — MERGE block is
    # skipped, the SUCCESS path still runs after a normal break/loop exit.
    workflow.worktree = worktree
    return workflow, scheduler


def _submit_l0(queue: EscalationQueue, task_id: str, *, category: str = 'task_failure') -> str:
    esc = Escalation(
        id=queue.make_id(task_id),
        task_id=task_id,
        agent_role='implementer',
        severity='blocking',
        category=category,
        summary='synthetic blocker',
        detail='synthetic detail',
    )
    queue.submit(esc)
    return esc.id


def _make_evrl_returner(returns: list[WorkflowOutcome]):
    """Return an AsyncMock that pops successive WorkflowOutcomes per call.

    The last value is reused if the list is exhausted, so a misconfigured
    test does not hang the workflow's outer while-true.
    """
    state = {'count': 0, 'queue': list(returns)}

    async def fake_evrl():
        state['count'] += 1
        q = state['queue']
        if len(q) > 1:
            return q.pop(0)
        return q[0] if q else WorkflowOutcome.DONE

    mock = AsyncMock(side_effect=fake_evrl)
    return mock, state


def _make_granting_steward(
    queue: EscalationQueue, task_id: str, granted_files: list[str],
) -> type:
    """Return a steward class that resolves pending L0s with a structured
    ``granted_files`` scope-expansion grant (task 2505).

    Mirrors ``_workflow_helpers._make_resolving_steward`` but forwards the
    grant through ``queue.resolve(..., granted_files=granted_files)`` so the
    resume loop's ``_collect_granted_files``/``_set_task_scope`` consumption
    can be exercised end-to-end via ``workflow.run()``.
    """

    class _FakeSteward:
        def __init__(self, wt_path, cfg_dir):  # noqa: ARG002
            self._outcome_channel = None
            self._wip_probe = None

        def set_outcome_channel(self, channel) -> None:
            self._outcome_channel = channel

        def set_wip_probe(self, probe) -> None:
            self._wip_probe = probe

        async def start(self) -> None:
            pending = queue.get_by_task(task_id, status='pending', level=0)
            assert pending, 'expected at least one pending L0 escalation to resolve'
            for esc in pending:
                queue.resolve(
                    esc.id, 'Granted scope expansion',
                    resolved_by='fake-steward', granted_files=granted_files,
                )
            if self._outcome_channel is not None:
                self._outcome_channel.put_nowait(
                    StewardResolved(resolution_text='Granted scope expansion'),
                )

        async def stop(self) -> None:
            pass

    return _FakeSteward


# ---------------------------------------------------------------------------
# Tests
# ---------------------------------------------------------------------------


@pytest.mark.asyncio
class TestStatusPreservationOnResume:
    """Fix 1: workflow honors steward terminal decisions before resuming.

    Each test drives the ESCALATED branch of run() exactly once via a
    monkeypatched _execute_verify_review_loop, then asserts the outcome
    after the steward has set a particular task status.
    """

    async def test_resume_after_steward_set_deferred_returns_blocked_no_l1(
        self, config, git_ops, task_assignment, tmp_path,
    ):
        """Steward sets status='deferred' → BLOCKED with no L1 and no resume."""
        wt = await _make_advanced_worktree(git_ops, task_assignment.task_id)
        queue = EscalationQueue(tmp_path / 'queue')
        _submit_l0(queue, task_assignment.task_id)
        workflow, scheduler = _build_workflow(
            config, git_ops, task_assignment, queue, wt,
        )
        workflow._steward_factory = _make_status_setting_steward(
            queue, scheduler, task_assignment.task_id, 'deferred',
        )
        evrl_mock, state = _make_evrl_returner(
            [WorkflowOutcome.ESCALATED, WorkflowOutcome.DONE],
        )
        workflow._execute_verify_review_loop = evrl_mock  # type: ignore[method-assign]
        invoke_mock = AsyncMock(return_value=AgentResult(success=True, output=''))
        workflow._invoke = invoke_mock  # type: ignore[method-assign]

        outcome = (await workflow.run()).outcome

        assert outcome == WorkflowOutcome.BLOCKED, (
            f'Expected BLOCKED, got {outcome!r}; '
            'Fix 1 must short-circuit on steward-set deferred.'
        )
        assert state['count'] == 1, (
            f'_execute_verify_review_loop must be entered exactly once; '
            f'got {state["count"]} entries — resume happened despite deferred.'
        )
        assert invoke_mock.await_count == 0, (
            'Implementer resume must NOT be invoked when steward set deferred.'
        )
        assert not queue.has_open_l1(task_assignment.task_id), (
            'No L1 must be filed for an intentional steward terminal decision.'
        )
        statuses = scheduler.statuses.get(task_assignment.task_id, [])
        assert statuses[-1] == 'deferred', (
            f"Final status must remain 'deferred': {statuses}"
        )
        assert workflow.state == WorkflowState.BLOCKED

    async def test_resume_after_steward_set_cancelled_returns_blocked_no_l1(
        self, config, git_ops, task_assignment, tmp_path,
    ):
        """Steward sets status='cancelled' → BLOCKED with no L1 and no resume."""
        wt = await _make_advanced_worktree(git_ops, task_assignment.task_id)
        queue = EscalationQueue(tmp_path / 'queue')
        _submit_l0(queue, task_assignment.task_id)
        workflow, scheduler = _build_workflow(
            config, git_ops, task_assignment, queue, wt,
        )
        workflow._steward_factory = _make_status_setting_steward(
            queue, scheduler, task_assignment.task_id, 'cancelled',
        )
        evrl_mock, state = _make_evrl_returner(
            [WorkflowOutcome.ESCALATED, WorkflowOutcome.DONE],
        )
        workflow._execute_verify_review_loop = evrl_mock  # type: ignore[method-assign]
        invoke_mock = AsyncMock(return_value=AgentResult(success=True, output=''))
        workflow._invoke = invoke_mock  # type: ignore[method-assign]

        outcome = (await workflow.run()).outcome

        assert outcome == WorkflowOutcome.BLOCKED
        assert state['count'] == 1
        assert invoke_mock.await_count == 0
        assert not queue.has_open_l1(task_assignment.task_id)
        statuses = scheduler.statuses.get(task_assignment.task_id, [])
        assert statuses[-1] == 'cancelled'
        assert workflow.state == WorkflowState.BLOCKED

    async def test_resume_after_steward_set_done_returns_done(
        self, config, git_ops, task_assignment, tmp_path,
    ):
        """Steward sets status='done' → DONE short-circuit, no resume."""
        wt = await _make_advanced_worktree(git_ops, task_assignment.task_id)
        queue = EscalationQueue(tmp_path / 'queue')
        _submit_l0(queue, task_assignment.task_id)
        workflow, scheduler = _build_workflow(
            config, git_ops, task_assignment, queue, wt,
        )
        workflow._steward_factory = _make_status_setting_steward(
            queue, scheduler, task_assignment.task_id, 'done',
        )
        evrl_mock, state = _make_evrl_returner(
            [WorkflowOutcome.ESCALATED, WorkflowOutcome.DONE],
        )
        workflow._execute_verify_review_loop = evrl_mock  # type: ignore[method-assign]
        invoke_mock = AsyncMock(return_value=AgentResult(success=True, output=''))
        workflow._invoke = invoke_mock  # type: ignore[method-assign]

        outcome = (await workflow.run()).outcome

        assert outcome == WorkflowOutcome.DONE, (
            f'Expected DONE, got {outcome!r}; '
            'Fix 1 must short-circuit on steward-set done.'
        )
        assert state['count'] == 1
        assert invoke_mock.await_count == 0
        assert not queue.has_open_l1(task_assignment.task_id)
        assert workflow.state == WorkflowState.DONE

    async def test_resume_with_status_in_progress_continues_implementer(
        self, config, git_ops, task_assignment, tmp_path,
    ):
        """Steward leaves status as in-progress → resume the implementer.

        This is the unchanged happy path: when the steward only resolves the
        L0 (does not set a terminal/preserve status), Fix 1 falls through and
        the existing resume-implementer code runs.
        """
        wt = await _make_advanced_worktree(git_ops, task_assignment.task_id)
        queue = EscalationQueue(tmp_path / 'queue')
        _submit_l0(queue, task_assignment.task_id)
        workflow, scheduler = _build_workflow(
            config, git_ops, task_assignment, queue, wt,
        )
        # Resolving steward — does not touch task status.
        workflow._steward_factory = _make_resolving_steward(
            queue, task_assignment.task_id,
        )
        # ESCALATED first, then DONE on the post-resume re-entry.
        evrl_mock, state = _make_evrl_returner(
            [WorkflowOutcome.ESCALATED, WorkflowOutcome.DONE],
        )
        workflow._execute_verify_review_loop = evrl_mock  # type: ignore[method-assign]
        invoke_mock = AsyncMock(return_value=AgentResult(success=True, output=''))
        workflow._invoke = invoke_mock  # type: ignore[method-assign]

        outcome = (await workflow.run()).outcome

        # After resume + DONE second loop iteration + skip-merge (eval-mode),
        # the SUCCESS path runs and writes 'done' to the scheduler.
        assert outcome == WorkflowOutcome.DONE
        assert state['count'] == 2, (
            f'_execute_verify_review_loop must be entered twice (initial '
            f'ESCALATED + post-resume DONE); got {state["count"]}.'
        )
        assert invoke_mock.await_count == 1, (
            'Implementer resume must be invoked exactly once when status is '
            'in-progress (no terminal/preserve status set).'
        )
        # The single _invoke call should be the implementer (resume).
        call_args = invoke_mock.await_args_list[0]
        # First positional arg is the AgentRole; check its name.
        role = call_args.args[0]
        assert getattr(role, 'name', '') == 'implementer', (
            f'Resume invocation must use IMPLEMENTER role; got {role!r}'
        )

    async def test_scope_violation_resume_clean_tree_reinvokes_implementer(
        self, config, git_ops, task_assignment, tmp_path,
    ):
        """A scope_violation L0 resolved with a clean/empty branch (tip ==
        base, zero commits) must still resume the implementer — task 2504.

        Before the fix, the ESCALATED-resume guard did a raw
        ``is_ancestor(wt_head, main_sha)`` check with no has-work guard. An
        empty branch trivially satisfies that check (its tip IS main's
        HEAD), so the guard wrongly concluded "already merged" and broke
        straight to MERGE, skipping implementer resume (and verify)
        entirely — the un-implemented branch then failed MERGE's
        plan-files-touched gate. This mirrors
        test_resume_with_status_in_progress_continues_implementer but uses
        a fresh/empty worktree (is_ancestor True, has_work naturally False)
        instead of one commit ahead of main, and a scope_violation category
        to match the observed incident.
        """
        wt_info = await git_ops.create_worktree(task_assignment.task_id)
        wt = wt_info.path
        queue = EscalationQueue(tmp_path / 'queue')
        _submit_l0(queue, task_assignment.task_id, category='scope_violation')
        workflow, scheduler = _build_workflow(
            config, git_ops, task_assignment, queue, wt,
        )
        # Resolving steward — does not touch task status, so it stays
        # in-progress and the resume path is live.
        workflow._steward_factory = _make_resolving_steward(
            queue, task_assignment.task_id,
        )
        # ESCALATED first, then DONE on the post-resume re-entry.
        evrl_mock, state = _make_evrl_returner(
            [WorkflowOutcome.ESCALATED, WorkflowOutcome.DONE],
        )
        workflow._execute_verify_review_loop = evrl_mock  # type: ignore[method-assign]
        invoke_mock = AsyncMock(return_value=AgentResult(success=True, output=''))
        workflow._invoke = invoke_mock  # type: ignore[method-assign]

        outcome = (await workflow.run()).outcome

        assert invoke_mock.await_count >= 1, (
            'Implementer must be resumed on a clean/empty-tree branch — the '
            'raw is_ancestor check must not short-circuit straight to MERGE '
            'for a branch with no prior implementation work.'
        )
        call_args = invoke_mock.await_args_list[0]
        role = call_args.args[0]
        assert getattr(role, 'name', '') == 'implementer', (
            f'Resume invocation must use IMPLEMENTER role; got {role!r}'
        )
        assert outcome == WorkflowOutcome.DONE

    async def test_scope_violation_resume_with_granted_files_widens_scope(
        self, config, git_ops, task_assignment, tmp_path,
    ):
        """A scope_violation L0 resolved with a structured ``granted_files``
        grant must fold the granted file into plan.files/metadata.files/
        locks BEFORE the implementer resumes — task 2505.

        Before the fix, the resume loop never consumed ``granted_files``: the
        grant lived only as free-text resolution prose, plan.json/
        metadata.files/locks were never updated, and the resumed implementer's
        briefing would not reflect the expanded scope.
        """
        wt_info = await git_ops.create_worktree(task_assignment.task_id)
        wt = wt_info.path
        queue = EscalationQueue(tmp_path / 'queue')
        _submit_l0(queue, task_assignment.task_id, category='scope_violation')
        workflow, scheduler = _build_workflow(
            config, git_ops, task_assignment, queue, wt,
        )
        # Granting steward — lock is free (FakeScheduler.blast_radius_result
        # defaults to True), so the scope-widen must succeed.
        workflow._steward_factory = _make_granting_steward(
            queue, task_assignment.task_id, ['new.py'],
        )
        # ESCALATED first, then DONE on the post-resume re-entry.
        evrl_mock, state = _make_evrl_returner(
            [WorkflowOutcome.ESCALATED, WorkflowOutcome.DONE],
        )
        workflow._execute_verify_review_loop = evrl_mock  # type: ignore[method-assign]
        invoke_mock = AsyncMock(return_value=AgentResult(success=True, output=''))
        workflow._invoke = invoke_mock  # type: ignore[method-assign]

        outcome = (await workflow.run()).outcome

        assert outcome == WorkflowOutcome.DONE
        assert 'new.py' in workflow.plan['files'], (
            f"granted file 'new.py' must be folded into plan.files; got "
            f"{workflow.plan['files']!r}"
        )
        assert any(
            'new.py' in (persist_files or [])
            for _current, _needed, persist_files in scheduler.blast_radius_calls
        ), (
            'expected a handle_blast_radius_expansion call with persist_files '
            f"including 'new.py'; got {scheduler.blast_radius_calls!r}"
        )
        assert invoke_mock.await_count >= 1, (
            'Implementer must be resumed after the scope grant is consumed.'
        )
        call_args = invoke_mock.await_args_list[0]
        role = call_args.args[0]
        assert getattr(role, 'name', '') == 'implementer', (
            f'Resume invocation must use IMPLEMENTER role; got {role!r}'
        )

    async def test_resume_consumes_same_module_grant(
        self, config, git_ops, task_assignment, tmp_path,
    ):
        """A grant for a same-package sibling of an existing plan file must
        STILL be folded into plan.files/metadata.files on resume — task 2505
        (reviewer regression).

        The pre-fix resume-loop gate keyed on ``set(new_modules) !=
        set(self.modules)``; for a same-module grant that set is unchanged, so
        ``_set_task_scope`` was never called and neither plan.files nor
        metadata.files ever reflected the grant. The fix decouples the
        plan.files write from the module-change gate (fires whenever the file
        set actually widens) and persists metadata.files directly when the
        module set is unchanged.
        """
        # Same-package siblings: identical module set at config.lock_depth.
        f1, f2 = _same_module_siblings(config.lock_depth)
        assert files_to_modules([f1], config.lock_depth) == files_to_modules(
            [f1, f2], config.lock_depth,
        ), (
            'precondition: F1 and F2 must map to the SAME module set at '
            f'lock_depth={config.lock_depth}'
        )

        wt_info = await git_ops.create_worktree(task_assignment.task_id)
        wt = wt_info.path
        queue = EscalationQueue(tmp_path / 'queue')
        _submit_l0(queue, task_assignment.task_id, category='scope_violation')
        workflow, scheduler = _build_workflow(
            config, git_ops, task_assignment, queue, wt,
        )
        # Seed the plan with F1 only and the matching (single-module) lock set,
        # so the granted sibling F2 does NOT change the module set.
        workflow.initial_plan = {**dict(PLAN), 'files': [f1]}
        workflow.modules = files_to_modules([f1], config.lock_depth)
        # Granting steward — grant the same-package sibling; lock is free
        # (blast_radius_result defaults True, though for a same-module widen no
        # blast-radius call is expected at all).
        workflow._steward_factory = _make_granting_steward(
            queue, task_assignment.task_id, [f2],
        )
        evrl_mock, state = _make_evrl_returner(
            [WorkflowOutcome.ESCALATED, WorkflowOutcome.DONE],
        )
        workflow._execute_verify_review_loop = evrl_mock  # type: ignore[method-assign]
        invoke_mock = AsyncMock(return_value=AgentResult(success=True, output=''))
        workflow._invoke = invoke_mock  # type: ignore[method-assign]

        outcome = (await workflow.run()).outcome

        assert outcome == WorkflowOutcome.DONE
        assert f2 in workflow.plan['files'], (
            f"granted same-module sibling {f2!r} must be folded into "
            f"plan.files; got {workflow.plan['files']!r}"
        )
        # Module set unchanged → NO handle_blast_radius_expansion call for the
        # grant (the reconcile no-ops when the module set doesn't change).
        assert all(
            f2 not in (persist_files or [])
            for _current, _needed, persist_files in scheduler.blast_radius_calls
        ), (
            'a same-module grant must not trigger a blast-radius call for the '
            f'grant; got {scheduler.blast_radius_calls!r}'
        )
        # metadata.files persisted DIRECTLY via update_task (blast-radius
        # no-op'd, so the persist must come from _set_task_scope's same-module
        # branch) — plan.files and metadata.files stay in lockstep.
        persisted_files = [
            md.get('files') for _tid, md in scheduler.update_task_calls
            if isinstance(md, dict)
        ]
        assert any(f2 in (files or []) for files in persisted_files), (
            'metadata.files must be persisted including the granted '
            f'sibling {f2!r}; update_task_calls={scheduler.update_task_calls!r}'
        )
        assert invoke_mock.await_count >= 1, (
            'Implementer must be resumed after the same-module grant is consumed.'
        )
        call_args = invoke_mock.await_args_list[0]
        role = call_args.args[0]
        assert getattr(role, 'name', '') == 'implementer', (
            f'Resume invocation must use IMPLEMENTER role; got {role!r}'
        )

    async def test_scope_violation_resume_granted_files_lock_conflict_requeues(
        self, config, git_ops, task_assignment, tmp_path,
    ):
        """A granted scope expansion whose lock is held by a sibling task
        must REQUEUE rather than resume the implementer under a foreign
        lock — task 2505.

        Before the fix, ``_set_task_scope``'s ``False`` return (lock
        conflict) was never checked in the resume loop, so the implementer
        would still be resumed even though the scheduler had already
        requeued the task to pending on its own conflicting-lock branch.
        """
        wt_info = await git_ops.create_worktree(task_assignment.task_id)
        wt = wt_info.path
        queue = EscalationQueue(tmp_path / 'queue')
        _submit_l0(queue, task_assignment.task_id, category='scope_violation')
        workflow, scheduler = _build_workflow(
            config, git_ops, task_assignment, queue, wt,
        )
        # Sibling holds the additional lock the grant needs.
        scheduler.blast_radius_result = False
        workflow._steward_factory = _make_granting_steward(
            queue, task_assignment.task_id, ['new.py'],
        )
        evrl_mock, state = _make_evrl_returner(
            [WorkflowOutcome.ESCALATED, WorkflowOutcome.DONE],
        )
        workflow._execute_verify_review_loop = evrl_mock  # type: ignore[method-assign]
        invoke_mock = AsyncMock(return_value=AgentResult(success=True, output=''))
        workflow._invoke = invoke_mock  # type: ignore[method-assign]

        report = await workflow.run()

        assert report.outcome == WorkflowOutcome.REQUEUED, (
            f'Expected REQUEUED on a scope-grant lock conflict, got '
            f'{report.outcome!r}'
        )
        assert state['count'] == 1, (
            '_execute_verify_review_loop must be entered exactly once — no '
            'resume re-entry once the scope grant hits a lock conflict.'
        )
        assert invoke_mock.await_count == 0, (
            'Implementer must NOT be resumed while the granted file lock '
            'is held by a sibling task.'
        )

    async def test_already_on_main_short_circuit_runs_before_status_check(
        self, config, git_ops, task_assignment, tmp_path,
    ):
        """If branch is already on main at resolution time, the existing
        already-on-main short-circuit must run BEFORE Fix 1's status check.

        Otherwise a steward-driven merge (which sets status to 'deferred' as a
        bookkeeping move while the merge lands externally) would be misread as
        'preserve and exit BLOCKED' instead of taking the already-merged path.
        """
        # Use a fresh worktree (HEAD == main) so is_ancestor returns True.
        wt_info = await git_ops.create_worktree(task_assignment.task_id)
        wt = wt_info.path
        queue = EscalationQueue(tmp_path / 'queue')
        _submit_l0(queue, task_assignment.task_id)
        workflow, scheduler = _build_workflow(
            config, git_ops, task_assignment, queue, wt,
        )
        # Steward sets deferred — Fix 1 would otherwise return BLOCKED here.
        workflow._steward_factory = _make_status_setting_steward(
            queue, scheduler, task_assignment.task_id, 'deferred',
        )
        # This fixture's worktree is fresh/empty (no commits), so it has no
        # real implementation work — in eval mode, _setup_worktree_and_artifacts
        # re-stamps base_commit to the current worktree HEAD (task 2504
        # subtlety B), making the SHA-primary has-work signal unreachable
        # through the natural run() flow. This test's purpose is ORDERING
        # (already-on-main short-circuit runs before the Fix-1 status
        # check), not the has-work predicate itself, so stub has_work=True
        # to keep it decoupled from the (now-fixed) empty-branch semantics
        # while still exercising the real is_ancestor check.
        #
        # Cross-ref: test_scope_violation_resume_clean_tree_reinvokes_implementer
        # (above) is the canonical coverage for the has_work=False resume
        # path — it uses no stub, so a regression there can't be masked by
        # this test's has_work=True stub. Together the two tests cover both
        # the short-circuit ordering (this test) and the has-work predicate
        # itself (that test).
        workflow._has_prior_implementation = MagicMock(  # type: ignore[method-assign]
            return_value=_PriorImplStatus(
                has_work=True,
                entries=[{'agent': 'implementer', 'steps_completed': ['step-1']}],
                base_commit='base',
            ),
        )
        evrl_mock, state = _make_evrl_returner(
            [WorkflowOutcome.ESCALATED, WorkflowOutcome.DONE],
        )
        workflow._execute_verify_review_loop = evrl_mock  # type: ignore[method-assign]
        invoke_mock = AsyncMock(return_value=AgentResult(success=True, output=''))
        workflow._invoke = invoke_mock  # type: ignore[method-assign]

        outcome = (await workflow.run()).outcome

        # already-on-main path → break → eval-mode skip MERGE → SUCCESS → DONE.
        # If Fix 1 ran first (regression) the outcome would be BLOCKED.
        assert outcome == WorkflowOutcome.DONE, (
            f'Expected DONE via already-on-main path, got {outcome!r}; '
            'Fix 1 must NOT preempt the already-on-main short-circuit.'
        )
        assert invoke_mock.await_count == 0, (
            'No implementer resume on the already-on-main path.'
        )


# ---------------------------------------------------------------------------
# _set_task_scope — orchestrator-side plan.files widen (task 2505)
# ---------------------------------------------------------------------------


@pytest.mark.asyncio
class TestSetTaskScope:
    """Unit tests for ``TaskWorkflow._set_task_scope`` — the single choke
    point that widens plan.json (via ``artifacts.set_plan_files``) and
    reconciles the scheduler's file-locks/metadata.files
    (``_reconcile_scope_locks``) outside the architect/plan-boundary flow.
    Consumed by the resume-path scope-grant logic added later in task 2505.
    """

    async def _build(self, config, git_ops, task_assignment, tmp_path):
        """Shared setup: a workflow with artifacts/plan wired directly
        (no run()), stamped-owned by the workflow's own session_id."""
        wt_info = await git_ops.create_worktree(task_assignment.task_id)
        wt = wt_info.path
        queue = EscalationQueue(tmp_path / 'queue')
        workflow, scheduler = _build_workflow(
            config, git_ops, task_assignment, queue, wt,
        )
        workflow.artifacts = TaskArtifacts(wt)
        workflow.artifacts.init(task_assignment.task_id, 'X', 'Y')
        plan = dict(PLAN)
        plan['files'] = ['lib.py']
        workflow.artifacts.write_plan(plan)
        workflow.artifacts.stamp_plan_provenance(workflow.session_id)
        workflow.plan = workflow.artifacts.read_plan()
        return workflow, scheduler

    async def test_success_widens_plan_and_modules(
        self, config, git_ops, task_assignment, tmp_path,
    ):
        workflow, scheduler = await self._build(
            config, git_ops, task_assignment, tmp_path,
        )
        artifacts = workflow.artifacts
        assert artifacts is not None

        result = await workflow._set_task_scope(['lib.py', 'new.py'])

        assert result is True
        on_disk = artifacts.read_plan()
        assert on_disk['files'] == ['lib.py', 'new.py']
        assert artifacts.validate_plan_owner(workflow.session_id) is True
        assert len(scheduler.blast_radius_calls) == 1, (
            'exactly one handle_blast_radius_expansion call expected'
        )
        _current, _needed, persist_files = scheduler.blast_radius_calls[0]
        assert persist_files == ['lib.py', 'new.py']
        assert workflow.modules == files_to_modules(
            ['lib.py', 'new.py'], config.lock_depth,
        )

    async def test_lock_conflict_leaves_modules_unchanged_but_widens_plan(
        self, config, git_ops, task_assignment, tmp_path,
    ):
        workflow, scheduler = await self._build(
            config, git_ops, task_assignment, tmp_path,
        )
        artifacts = workflow.artifacts
        assert artifacts is not None
        original_modules = list(workflow.modules)
        scheduler.blast_radius_result = False

        result = await workflow._set_task_scope(['lib.py', 'new.py'])

        assert result is False
        assert workflow.modules == original_modules, (
            'self.modules must stay unchanged on a lock conflict — the '
            'scheduler already requeued the task holding the ORIGINAL lock.'
        )
        on_disk = artifacts.read_plan()
        assert on_disk['files'] == ['lib.py', 'new.py'], (
            'plan.json must still be widened on conflict — the scheduler '
            'persists metadata.files=new_files on its requeue branch too, '
            'so plan.files must match metadata.files even when the lock '
            'could not be acquired.'
        )

    async def test_same_module_widen_persists_plan_and_metadata_without_blast(
        self, config, git_ops, task_assignment, tmp_path,
    ):
        """A grant whose file maps to a module the task already locks must
        still persist metadata.files DIRECTLY — task 2505 (reviewer regression).

        ``handle_blast_radius_expansion`` persists ``metadata.files`` ONLY on a
        module change; a same-module widen no-ops that call. Without the
        direct persist, plan.files would be widened while metadata.files stayed
        behind — the exact divergence the MERGE-entry ``_check_scope_invariant``
        tripwire is built to catch.
        """
        # Two files co-located in the same package: at config.lock_depth the
        # module set is identical with or without F2 — a genuine same-module
        # widen (self-validating precondition asserted first).
        f1, f2 = _same_module_siblings(config.lock_depth)
        assert files_to_modules([f1], config.lock_depth) == files_to_modules(
            [f1, f2], config.lock_depth,
        ), (
            'precondition: F1 and F2 must map to the SAME module set at '
            f'lock_depth={config.lock_depth}; got '
            f'{files_to_modules([f1], config.lock_depth)!r} vs '
            f'{files_to_modules([f1, f2], config.lock_depth)!r}'
        )

        wt_info = await git_ops.create_worktree(task_assignment.task_id)
        wt = wt_info.path
        queue = EscalationQueue(tmp_path / 'queue')
        workflow, scheduler = _build_workflow(
            config, git_ops, task_assignment, queue, wt,
        )
        workflow.artifacts = TaskArtifacts(wt)
        workflow.artifacts.init(task_assignment.task_id, 'X', 'Y')
        plan = dict(PLAN)
        plan['files'] = [f1]
        workflow.artifacts.write_plan(plan)
        workflow.artifacts.stamp_plan_provenance(workflow.session_id)
        workflow.plan = workflow.artifacts.read_plan()
        workflow.modules = files_to_modules([f1], config.lock_depth)
        modules_before = list(workflow.modules)

        result = await workflow._set_task_scope([f1, f2])

        assert result is True
        assert workflow.modules == modules_before, (
            'self.modules must be UNCHANGED on a same-module widen'
        )
        assert scheduler.blast_radius_calls == [], (
            'a same-module widen must NOT call handle_blast_radius_expansion '
            f'(module set unchanged); got {scheduler.blast_radius_calls!r}'
        )
        on_disk = workflow.artifacts.read_plan()
        assert on_disk['files'] == [f1, f2], (
            f'plan.json must be widened to include {f2!r}; got {on_disk["files"]!r}'
        )
        assert workflow.artifacts.validate_plan_owner(workflow.session_id) is True
        persisted_files = [
            md.get('files') for _tid, md in scheduler.update_task_calls
            if isinstance(md, dict)
        ]
        assert any(f2 in (files or []) for files in persisted_files), (
            'metadata.files must be persisted DIRECTLY (via update_task) '
            f'including {f2!r} on a same-module widen, since '
            'handle_blast_radius_expansion no-ops and never persists it; '
            f'update_task_calls={scheduler.update_task_calls!r}'
        )


# ---------------------------------------------------------------------------
# _check_scope_invariant — plan.files vs metadata.files tripwire (task 2505)
# ---------------------------------------------------------------------------


@pytest.mark.asyncio
class TestCheckScopeInvariant:
    """Unit tests for ``TaskWorkflow._check_scope_invariant`` — the MERGE-entry
    tripwire that surfaces a plan.files/metadata.files divergence with a
    WARNING + infra_issue escalation, fail-safe when the task can't be read.
    """

    def _build(self, config, git_ops, task_assignment, tmp_path):
        queue = EscalationQueue(tmp_path / 'queue')
        scheduler = FakeScheduler()
        workflow = TaskWorkflow(
            assignment=task_assignment,
            config=config,
            git_ops=git_ops,
            scheduler=scheduler,  # type: ignore[arg-type]
            briefing=FakeBriefing(),  # type: ignore[arg-type]
            mcp=FakeMcp(),  # type: ignore[arg-type]
            escalation_queue=queue,
        )
        return workflow, scheduler, queue

    async def test_divergent_plan_and_metadata_files_escalates(
        self, config, git_ops, task_assignment, tmp_path,
    ):
        # A genuine cross-MODULE divergence must escalate. At lock_depth=2 the
        # top-level files 'a.py' and 'b.py' normalize to DISTINCT modules, so
        # {'a.py','b.py'} != {'a.py'} at module granularity too — the tripwire
        # (now module-granular, task 2505 reviewer amendment) still fires.
        # PLAN-EXTRA / UNSAFE direction (task 3429): plan.files needs a lock
        # module that metadata.files does not cover — this is the direction
        # the narrowed containment check must still catch.
        plan_files = ['a.py', 'b.py']
        metadata_files = ['a.py']
        plan_modules = set(files_to_modules(plan_files, config.lock_depth))
        metadata_modules = set(files_to_modules(metadata_files, config.lock_depth))
        assert plan_modules != metadata_modules, (
            'precondition: this test must exercise a genuine cross-MODULE '
            f'divergence at lock_depth={config.lock_depth}'
        )
        assert plan_modules - metadata_modules, (
            'precondition: this must be a PLAN-EXTRA divergence — plan.files '
            'needs a lock module metadata.files does not cover — at '
            f'lock_depth={config.lock_depth}; plan_modules={plan_modules!r} '
            f'metadata_modules={metadata_modules!r}'
        )
        workflow, scheduler, queue = self._build(
            config, git_ops, task_assignment, tmp_path,
        )
        workflow.plan = {'files': plan_files}
        scheduler.task_data[task_assignment.task_id] = {
            'id': task_assignment.task_id,
            'metadata': {'files': metadata_files},
        }

        await workflow._check_scope_invariant()

        pending = queue.get_by_task(task_assignment.task_id, status='pending')
        infra_issues = [e for e in pending if e.category == 'infra_issue']
        assert infra_issues, (
            'expected an infra_issue escalation for the plan/metadata files '
            f'divergence; pending escalations: {pending!r}'
        )

    async def test_same_module_file_divergence_no_escalation(
        self, config, git_ops, task_assignment, tmp_path,
    ):
        """A same-module file-level difference must NOT escalate — task 2505
        reviewer amendment (false-positive fix).

        ``metadata.files`` is only persisted by ``handle_blast_radius_expansion``,
        which no-ops when the module set is unchanged, so the normal architect
        widen — author declares one file; architect plans that plus a
        same-package sibling — legitimately leaves plan.files and
        metadata.files differing at file level while the LOCK-MODULE sets (and
        thus the locks and the merge) agree. The MERGE-entry tripwire must not
        fire a blocking infra_issue for that benign case. This FAILS under the
        old file-granularity comparison (which would escalate) and passes under
        the module-granularity comparison.
        """
        _f1, _f2 = _same_module_siblings(config.lock_depth)
        plan_files = [_f1, _f2]
        metadata_files = [_f1]
        # Self-validating precondition: genuinely same-module but file-divergent.
        assert files_to_modules(plan_files, config.lock_depth) == files_to_modules(
            metadata_files, config.lock_depth,
        ), (
            'precondition: F-set must map to the SAME module set at '
            f'lock_depth={config.lock_depth}'
        )
        assert set(plan_files) != set(metadata_files), (
            'precondition: the file lists must genuinely differ (else this '
            'would collapse into the consistent-files case)'
        )
        workflow, scheduler, queue = self._build(
            config, git_ops, task_assignment, tmp_path,
        )
        workflow.plan = {'files': plan_files}
        scheduler.task_data[task_assignment.task_id] = {
            'id': task_assignment.task_id,
            'metadata': {'files': metadata_files},
        }

        await workflow._check_scope_invariant()

        assert queue.get_by_task(task_assignment.task_id, status='pending') == [], (
            'a benign same-module file addition (locks unaffected) must NOT '
            'fire the blocking infra_issue tripwire — comparison is at '
            'lock-module granularity, not file granularity'
        )

    async def test_consistent_plan_and_metadata_files_no_escalation(
        self, config, git_ops, task_assignment, tmp_path,
    ):
        workflow, scheduler, queue = self._build(
            config, git_ops, task_assignment, tmp_path,
        )
        workflow.plan = {'files': ['a.py', 'b.py']}
        scheduler.task_data[task_assignment.task_id] = {
            'id': task_assignment.task_id,
            'metadata': {'files': ['a.py', 'b.py']},
        }

        await workflow._check_scope_invariant()

        assert queue.get_by_task(task_assignment.task_id, status='pending') == [], (
            'no escalation should be filed when plan.files and '
            'metadata.files already agree'
        )

    async def test_get_task_none_is_fail_safe_no_escalation_no_raise(
        self, config, git_ops, task_assignment, tmp_path,
    ):
        workflow, scheduler, queue = self._build(
            config, git_ops, task_assignment, tmp_path,
        )
        workflow.plan = {'files': ['a.py', 'b.py']}
        # scheduler.task_data intentionally left empty -> get_task() -> None.

        await workflow._check_scope_invariant()  # must not raise

        assert queue.get_by_task(task_assignment.task_id, status='pending') == [], (
            'an unreadable task must be treated as "cannot check", not '
            '"divergent" — no escalation on a transient read failure'
        )

    async def test_metadata_module_superset_of_plan_no_escalation(
        self, config, git_ops, task_assignment, tmp_path, caplog,
    ):
        """SAFE direction (task 3429): metadata.files a strict module
        SUPERSET of plan.files must NOT escalate.

        Mirrors the measured structural false-positive class (7 of the 9
        PLAN-MISSING divergence strands on 2026-08-06): the extra file is the
        task's OWN TEST FILE living in a sibling directory. metadata.files is
        the sole input the scheduler derives this task's file locks from, so
        a module SUPERSET means the held locks are wider than the plan
        needs — that cannot let two tasks collide, it can only
        over-serialise — so this must not fire the blocking L0 that gates
        the merge. The safe direction must still stay observable, so it is
        logged at INFO instead of vanishing silently.
        """
        plan_files = ['scripts/foo.sh']
        metadata_files = ['scripts/foo.sh', 'scripts/tests/test_foo.py']
        plan_modules = set(files_to_modules(plan_files, config.lock_depth))
        metadata_modules = set(
            files_to_modules(metadata_files, config.lock_depth)
        )
        assert plan_modules < metadata_modules, (
            'precondition: metadata_modules must be a STRICT superset of '
            f'plan_modules at lock_depth={config.lock_depth}; '
            f'plan_modules={plan_modules!r} metadata_modules={metadata_modules!r}'
        )
        workflow, scheduler, queue = self._build(
            config, git_ops, task_assignment, tmp_path,
        )
        workflow.plan = {'files': plan_files}
        scheduler.task_data[task_assignment.task_id] = {
            'id': task_assignment.task_id,
            'metadata': {'files': metadata_files},
        }

        with caplog.at_level(logging.INFO, logger='orchestrator.workflow'):
            await workflow._check_scope_invariant()

        assert queue.get_by_task(task_assignment.task_id, status='pending') == [], (
            'a metadata.files module SUPERSET is a wider-than-needed lock '
            'set — it cannot let two tasks collide, only over-serialise — '
            'so it must NOT file the blocking L0 that gates the merge'
        )
        extra_modules = sorted(metadata_modules - plan_modules)
        info_messages = [
            r.getMessage() for r in caplog.records if r.levelno == logging.INFO
        ]
        matching = [
            msg for msg in info_messages
            if task_assignment.task_id in msg
            and all(mod in msg for mod in extra_modules)
        ]
        assert matching, (
            'expected an INFO log naming the task id and the extra metadata '
            f'module(s) {extra_modules!r} so the benign-superset direction '
            f'stays observable; INFO records: {info_messages!r}'
        )


# ---------------------------------------------------------------------------
# Regression: workflow passes real prompt (not 'continue') to invoke_with_cap_retry
# ---------------------------------------------------------------------------


@pytest.mark.asyncio
class TestRealPromptPassedToCliInvokeOnResume:
    """Regression for task 1462: workflow must pass the real task prompt to
    invoke_with_cap_retry even when resuming via a recovered sidecar session.

    Before the fix, workflow.py:3753 overwrote invoke_prompt = 'continue' and
    passed that as prompt= to invoke_with_cap_retry.  cli_invoke then captured
    original_prompt = 'continue', so on fresh-fallback the agent started with
    zero context.  After the fix, workflow always passes the real prompt and
    cli_invoke owns the substitution.
    """

    async def test_invoke_passes_real_prompt_when_resuming_via_sidecar(
        self, config, git_ops, task_assignment,
    ):
        """_invoke forwards the real task prompt to invoke_with_cap_retry alongside resume_session_id."""
        wt_info = await git_ops.create_worktree(task_assignment.task_id)
        cwd = wt_info.path

        workflow = TaskWorkflow(
            assignment=task_assignment,
            config=config,
            git_ops=git_ops,
            scheduler=FakeScheduler(),  # type: ignore[arg-type]
            briefing=FakeBriefing(),  # type: ignore[arg-type]
            mcp=None,
            resume_session_id={'session_id': 'sid-1', 'role': 'implementer'},
        )
        # Ensure artifacts is None so write_agent_session is skipped.
        workflow.artifacts = None

        with patch(
            'orchestrator.workflow.invoke_with_cap_retry',
            new_callable=AsyncMock,
            return_value=AgentResult(success=True, output=''),
        ) as mock_cap_retry:
            await workflow._invoke(IMPLEMENTER, 'REAL TASK PROMPT', cwd)

        assert mock_cap_retry.await_count == 1, 'invoke_with_cap_retry must be called once'
        assert mock_cap_retry.await_args is not None, 'await_args must not be None'
        call_kwargs = mock_cap_retry.await_args.kwargs
        assert call_kwargs.get('prompt') == 'REAL TASK PROMPT', (
            f"Expected prompt='REAL TASK PROMPT' but got {call_kwargs.get('prompt')!r}. "
            'workflow._invoke must pass the real task prompt, not the continuation string.'
        )
        assert call_kwargs.get('resume_session_id') == 'sid-1', (
            f"Expected resume_session_id='sid-1' but got {call_kwargs.get('resume_session_id')!r}."
        )


# ---------------------------------------------------------------------------
# Regression: config.timeouts.startup_grace_secs reaches the watchdog
# ---------------------------------------------------------------------------


@pytest.mark.asyncio
class TestStartupGraceSecsReachesWatchdog:
    """config.timeouts.startup_grace_secs must be forwarded to invoke_with_cap_retry.

    Step-13 RED: _invoke does not yet pass startup_grace_secs, so the config
    field is a dead knob — editing config.yaml has no effect on the watchdog.
    After step-14 it is threaded end-to-end:
    config.yaml → TimeoutsConfig.startup_grace_secs → _invoke →
    invoke_with_cap_retry → invoke_agent → _invoke_claude_with_sandbox →
    {sandbox _run_subprocess | invoke_claude_agent → _invoke_claude → _run_subprocess}
    """

    async def test_config_startup_grace_secs_reaches_invoke_with_cap_retry(
        self, config, git_ops, task_assignment,
    ):
        """_invoke forwards config.timeouts.startup_grace_secs to invoke_with_cap_retry.

        Fails today: _invoke does not pass startup_grace_secs, so the captured
        kwarg is absent/None != 77.0.
        """
        config.timeouts.startup_grace_secs = 77.0

        wt_info = await git_ops.create_worktree(task_assignment.task_id)
        cwd = wt_info.path

        workflow = TaskWorkflow(
            assignment=task_assignment,
            config=config,
            git_ops=git_ops,
            scheduler=FakeScheduler(),  # type: ignore[arg-type]
            briefing=FakeBriefing(),  # type: ignore[arg-type]
            mcp=None,
        )
        workflow.artifacts = None

        with patch(
            'orchestrator.workflow.invoke_with_cap_retry',
            new_callable=AsyncMock,
            return_value=AgentResult(success=True, output=''),
        ) as mock_cap_retry:
            await workflow._invoke(IMPLEMENTER, 'PROMPT', cwd)

        assert mock_cap_retry.await_count == 1, 'invoke_with_cap_retry must be called once'
        assert mock_cap_retry.await_args is not None, 'await_args must be set after one await'
        call_kwargs = mock_cap_retry.await_args.kwargs
        assert call_kwargs.get('startup_grace_secs') == 77.0, (
            f'Expected startup_grace_secs=77.0 forwarded to invoke_with_cap_retry; '
            f'got {call_kwargs.get("startup_grace_secs")!r}. '
            'workflow._invoke must thread config.timeouts.startup_grace_secs end-to-end.'
        )
