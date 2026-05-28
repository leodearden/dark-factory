"""Tests for the opaque-infra-timeout fast-fail path in ``_verify_debugfix_loop``.

When the verifier's own injected ``Command timed out after Ns: …`` wrapper
string is the only cause hint, retrying gives the debugger nothing to act
on.  After ``max_opaque_timeout_attempts`` such failures the loop must
short-circuit to ``WorkflowOutcome.BLOCKED`` instead of burning the full
``max_verify_attempts`` budget.

The streamed-output fix in ``_run_cmd`` (Change 2) means a real cause hint
should surface on attempt 1 for any genuine in-test hang; this guard is the
escalation hatch for the residual case where even the streamed buffer is
empty.
"""

from __future__ import annotations

import asyncio
from pathlib import Path
from unittest.mock import AsyncMock, MagicMock

import pytest

from orchestrator.artifacts import TaskArtifacts
from orchestrator.config import GitConfig, OrchestratorConfig
from orchestrator.git_ops import GitOps, _run
from orchestrator.scheduler import TaskAssignment
from orchestrator.verify import VerifyResult
from orchestrator.workflow import TaskWorkflow, WorkflowOutcome


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
    (repo / 'lib.py').write_text('x = 1\n')
    await _run(['git', 'add', '-A'], cwd=repo)
    await _run(['git', 'commit', '-m', 'Initial'], cwd=repo)


@pytest.fixture
def config(git_repo: Path) -> OrchestratorConfig:
    return OrchestratorConfig(
        project_root=git_repo,
        max_concurrent_tasks=1,
        # Generous attempt cap so the test asserts the opaque-timeout cap,
        # not the global cap.
        max_verify_attempts=5,
        max_opaque_timeout_attempts=2,
        # Raise the signature-repetition cap well above max_verify_attempts so
        # the opaque-timeout fast-fail tests are not affected by the new
        # signature guard — these tests are solely about the opaque-timeout cap.
        max_failure_signature_repeat=100,
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
            'id': '42', 'title': 'X', 'description': '',
            'status': 'pending', 'metadata': {'files': ['lib']},
            'dependencies': [],
        },
        modules=['lib'],
    )


def _make_workflow(
    config: OrchestratorConfig,
    git_ops: GitOps,
    assignment: TaskAssignment,
    worktree: Path,
) -> tuple[TaskWorkflow, TaskArtifacts]:
    workflow = TaskWorkflow(
        assignment=assignment,
        config=config,
        git_ops=git_ops,
        scheduler=MagicMock(),  # type: ignore[arg-type]
        briefing=MagicMock(),  # type: ignore[arg-type]
        mcp=MagicMock(),  # type: ignore[arg-type]
    )
    workflow.worktree = worktree
    artifacts = TaskArtifacts(worktree)
    artifacts.init('42', 'X', 'desc', base_commit='base-sha-old')
    workflow.artifacts = artifacts
    workflow.plan = {'task_id': '42', 'steps': []}
    workflow._check_escalations = MagicMock(return_value=[])  # type: ignore[method-assign]
    workflow.briefing.build_debugger_prompt = AsyncMock(return_value='debug')  # type: ignore[attr-defined]

    from orchestrator.agents.invoke import AgentResult

    workflow._invoke = AsyncMock(  # type: ignore[method-assign]
        return_value=AgentResult(success=True, output=''),
    )
    workflow._get_head_commit = AsyncMock(return_value='head-sha')  # type: ignore[method-assign]
    return workflow, artifacts


def _opaque_timeout_result() -> VerifyResult:
    return VerifyResult(
        passed=False,
        test_output='',
        lint_output='',
        type_output='',
        summary='Verification timed out',
        timed_out=True,
        cause_hint='Command timed out after 600.0s: pytest -q',
        category='infra_timeout',
    )


def _actionable_test_failure_result() -> VerifyResult:
    return VerifyResult(
        passed=False,
        test_output='FAILED tests/test_x.py::test_y\n',
        lint_output='',
        type_output='',
        summary='Failures: tests failed',
        timed_out=False,
        cause_hint='FAILED tests/test_x.py::test_y',
        category='test_failure',
    )


@pytest.mark.asyncio
class TestOpaqueTimeoutFastFail:
    async def test_opaque_infra_timeout_escalates_at_attempt_2(
        self, config, git_ops, task_assignment, monkeypatch,
    ):
        """Two opaque ``Command timed out after Ns: …`` results in a row → BLOCKED.

        Without the fast-fail guard the loop would invoke the debugger and
        retry through ``max_verify_attempts``.
        """
        wt_info = await git_ops.create_worktree(task_assignment.task_id)
        wt = wt_info.path
        workflow, _artifacts = _make_workflow(config, git_ops, task_assignment, wt)
        # Don't rebase mid-loop; that introduces unrelated git work.
        workflow._inter_iteration_rebase = AsyncMock(return_value=None)  # type: ignore[method-assign]

        verify_mock = AsyncMock(side_effect=[
            _opaque_timeout_result(),
            _opaque_timeout_result(),
            _opaque_timeout_result(),
        ])
        monkeypatch.setattr(
            'orchestrator.workflow.run_scoped_verification', verify_mock,
        )

        outcome = await workflow._verify_debugfix_loop()
        assert outcome == WorkflowOutcome.BLOCKED
        # Verify was called exactly max_opaque_timeout_attempts (2) times;
        # third call MUST NOT happen.
        assert verify_mock.await_count == 2, (
            f'expected fast-fail at attempt 2, but verify was invoked '
            f'{verify_mock.await_count} times'
        )

    async def test_actionable_cause_hint_still_retries_to_global_cap(
        self, config, git_ops, task_assignment, monkeypatch,
    ):
        """A populated cause_hint (real test failure) keeps retrying through
        the global ``max_verify_attempts`` budget, unaffected by the
        opaque-timeout cap.
        """
        wt_info = await git_ops.create_worktree(task_assignment.task_id)
        wt = wt_info.path
        workflow, _artifacts = _make_workflow(config, git_ops, task_assignment, wt)
        workflow._inter_iteration_rebase = AsyncMock(return_value=None)  # type: ignore[method-assign]

        # Provide enough failure results to fully exhaust max_verify_attempts.
        verify_mock = AsyncMock(side_effect=[
            _actionable_test_failure_result() for _ in range(config.max_verify_attempts)
        ])
        monkeypatch.setattr(
            'orchestrator.workflow.run_scoped_verification', verify_mock,
        )

        outcome = await workflow._verify_debugfix_loop()
        assert outcome == WorkflowOutcome.BLOCKED
        # Real test failures retry up to max_verify_attempts.
        assert verify_mock.await_count == config.max_verify_attempts, (
            f'expected full {config.max_verify_attempts}-attempt budget, '
            f'got {verify_mock.await_count}'
        )

    async def test_mixed_timeout_then_actionable_does_not_fast_fail(
        self, config, git_ops, task_assignment, monkeypatch,
    ):
        """Opaque-timeout on attempt 1 followed by an actionable failure
        on attempt 2 should NOT short-circuit — only consecutive opaque
        timeouts hit the cap.

        The simplest reading of the guard counts ``verify_attempt`` against
        the opaque cap regardless of intervening attempts.  This test pins
        whichever semantics we ship; if the implementation chooses the
        consecutive-counter variant later, update the expectation.
        """
        wt_info = await git_ops.create_worktree(task_assignment.task_id)
        wt = wt_info.path
        workflow, _artifacts = _make_workflow(config, git_ops, task_assignment, wt)
        workflow._inter_iteration_rebase = AsyncMock(return_value=None)  # type: ignore[method-assign]

        # Sequence: opaque, actionable, actionable, actionable, actionable (5)
        side_effects = [
            _opaque_timeout_result(),
            _actionable_test_failure_result(),
            _actionable_test_failure_result(),
            _actionable_test_failure_result(),
            _actionable_test_failure_result(),
        ]
        verify_mock = AsyncMock(side_effect=side_effects)
        monkeypatch.setattr(
            'orchestrator.workflow.run_scoped_verification', verify_mock,
        )

        outcome = await workflow._verify_debugfix_loop()
        # We expect a full max_verify_attempts (5) loop because the
        # plan's simple guard checks attempt-count + opaque category +
        # opaque regex.  After the first opaque attempt verify_attempt=1
        # ( < 2 ), so we don't short-circuit; attempts 2..5 are actionable
        # so the regex check fails too.
        assert outcome == WorkflowOutcome.BLOCKED
        assert verify_mock.await_count == config.max_verify_attempts
