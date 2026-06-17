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
from pathlib import Path
from unittest.mock import AsyncMock, patch

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
from orchestrator.config import GitConfig, OrchestratorConfig
from orchestrator.git_ops import GitOps, _run
from orchestrator.scheduler import TaskAssignment
from orchestrator.workflow import (
    IMPLEMENTER,
    TaskWorkflow,
    WorkflowOutcome,
    WorkflowState,
)

# ---------------------------------------------------------------------------
# Fixtures (kept local — these tests don't share runtime with test_workflow_e2e)
# ---------------------------------------------------------------------------


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

        outcome = await workflow.run()

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

        outcome = await workflow.run()

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

        outcome = await workflow.run()

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

        outcome = await workflow.run()

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
        evrl_mock, state = _make_evrl_returner(
            [WorkflowOutcome.ESCALATED, WorkflowOutcome.DONE],
        )
        workflow._execute_verify_review_loop = evrl_mock  # type: ignore[method-assign]
        invoke_mock = AsyncMock(return_value=AgentResult(success=True, output=''))
        workflow._invoke = invoke_mock  # type: ignore[method-assign]

        outcome = await workflow.run()

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
        call_kwargs = mock_cap_retry.await_args.kwargs
        assert call_kwargs.get('startup_grace_secs') == 77.0, (
            f'Expected startup_grace_secs=77.0 forwarded to invoke_with_cap_retry; '
            f'got {call_kwargs.get("startup_grace_secs")!r}. '
            'workflow._invoke must thread config.timeouts.startup_grace_secs end-to-end.'
        )
