"""Tests for Fix 3 — rebase before each verify retry.

Closes the verify-only-retry rebase gap: when main advances mid-task (e.g.
a sibling task fixes the env collision causing verify to fail), the
existing ``_inter_iteration_rebase`` only fires from the EXECUTE loop and
cannot pick up new main commits while the workflow is cycling between
verify and the debugger.

The fix wires ``_inter_iteration_rebase`` (now parametrized with an
``event_label``) into ``_verify_debugfix_loop`` ahead of every
``run_scoped_verification`` call, gated on ``config.rebase_before_verify``.
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
from orchestrator.workflow import (
    TaskWorkflow,
    WorkflowOutcome,
)


# ---------------------------------------------------------------------------
# Fixtures + helpers
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
    (repo / 'lib.py').write_text('x = 1\n')
    await _run(['git', 'add', '-A'], cwd=repo)
    await _run(['git', 'commit', '-m', 'Initial'], cwd=repo)


@pytest.fixture
def config(git_repo: Path) -> OrchestratorConfig:
    return OrchestratorConfig(
        project_root=git_repo,
        max_concurrent_tasks=1,
        max_verify_attempts=2,  # tight budget so failure tests don't loop forever
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
            'status': 'pending', 'metadata': {'modules': ['lib']},
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
    """Wire a minimal TaskWorkflow with all heavy collaborators mocked.

    Returns ``(workflow, artifacts)`` so callers can read/mutate artifacts
    without re-deriving narrowing for ``workflow.artifacts: TaskArtifacts | None``.

    Mocks ``_invoke`` (debugger), ``_check_escalations`` (none), and
    ``briefing`` calls so ``_verify_debugfix_loop`` can run end-to-end
    without real agents.
    """
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
    # No escalations by default; tests that want to assert otherwise can
    # override via monkeypatching.
    workflow._check_escalations = MagicMock(return_value=[])  # type: ignore[method-assign]
    # Briefing's debugger prompt builder must be awaitable.
    workflow.briefing.build_debugger_prompt = AsyncMock(return_value='debug')  # type: ignore[attr-defined]
    # Debugger calls go through _invoke; stub it.
    from orchestrator.agents.invoke import AgentResult
    workflow._invoke = AsyncMock(  # type: ignore[method-assign]
        return_value=AgentResult(success=True, output=''),
    )
    workflow._get_head_commit = AsyncMock(return_value='head-sha')  # type: ignore[method-assign]
    return workflow, artifacts


# ---------------------------------------------------------------------------
# Direct tests on _inter_iteration_rebase — event_label propagation + clean tree
# ---------------------------------------------------------------------------


@pytest.mark.asyncio
class TestInterIterationRebaseEventLabel:
    async def test_event_label_propagates_to_iteration_log(
        self, config, git_ops, task_assignment, tmp_path,
    ):
        """A successful rebase logs the supplied event_label."""
        wt_info = await git_ops.create_worktree(task_assignment.task_id)
        wt = wt_info.path
        # Advance main: write & commit a file directly on main (in repo).
        repo = config.project_root
        (repo / 'sibling.txt').write_text('sibling fix\n')
        await _run(['git', 'add', 'sibling.txt'], cwd=repo)
        await _run(['git', 'commit', '-m', 'sibling fix'], cwd=repo)

        workflow, artifacts = _make_workflow(config, git_ops, task_assignment, wt)
        # Set base_commit to the original main sha (pre-sibling-fix).
        artifacts.update_base_commit(wt_info.base_commit)

        result = await workflow._inter_iteration_rebase(
            event_label='verify_phase_rebase',
        )

        assert result is not None, 'Rebase should have happened (main advanced).'
        entries, _ = artifacts.read_iteration_log()
        rebase_entries = [e for e in entries if e.get('event') == 'verify_phase_rebase']
        assert len(rebase_entries) == 1, (
            f'Expected 1 verify_phase_rebase log entry, got {len(rebase_entries)}: {entries}'
        )
        # Default label preserved when not specified.
        assert not any(e.get('event') == 'rebase' for e in entries), (
            'No legacy "rebase"-labelled entry should be present.'
        )

    async def test_default_event_label_is_rebase(
        self, config, git_ops, task_assignment, tmp_path,
    ):
        """Calling without event_label preserves the original 'rebase' label."""
        wt_info = await git_ops.create_worktree(task_assignment.task_id)
        wt = wt_info.path
        repo = config.project_root
        (repo / 'sibling2.txt').write_text('sibling fix 2\n')
        await _run(['git', 'add', 'sibling2.txt'], cwd=repo)
        await _run(['git', 'commit', '-m', 'sibling fix 2'], cwd=repo)

        workflow, artifacts = _make_workflow(config, git_ops, task_assignment, wt)
        artifacts.update_base_commit(wt_info.base_commit)

        result = await workflow._inter_iteration_rebase()  # default

        assert result is not None
        entries, _ = artifacts.read_iteration_log()
        labels = [e.get('event') for e in entries]
        assert 'rebase' in labels and 'verify_phase_rebase' not in labels, (
            f'Default call must use "rebase"; entries: {entries}'
        )

    async def test_rebase_on_clean_tree_does_not_create_empty_commit(
        self, config, git_ops, task_assignment, tmp_path,
    ):
        """Verify-phase callers run on a clean tree.  ``commit()`` must
        no-op so we don't pollute history with empty WIP commits."""
        wt_info = await git_ops.create_worktree(task_assignment.task_id)
        wt = wt_info.path
        # Advance main so the rebase actually runs.
        repo = config.project_root
        (repo / 'sibling3.txt').write_text('clean-tree fix\n')
        await _run(['git', 'add', 'sibling3.txt'], cwd=repo)
        await _run(['git', 'commit', '-m', 'clean-tree fix'], cwd=repo)

        workflow, artifacts = _make_workflow(config, git_ops, task_assignment, wt)
        artifacts.update_base_commit(wt_info.base_commit)

        # Capture log SHAs before.
        _, before_shas, _ = await _run(
            ['git', 'log', '--oneline', '--format=%H'], cwd=wt,
        )
        before = before_shas.strip().splitlines()

        result = await workflow._inter_iteration_rebase(
            event_label='verify_phase_rebase',
        )
        assert result is not None

        # After: branch should have only fast-forwarded; no WIP commit
        # introduced. The branch's local commits == 0 (it's purely main).
        _, after_shas, _ = await _run(
            ['git', 'log', '--oneline', '--format=%s'], cwd=wt,
        )
        subjects = after_shas.strip().splitlines()
        assert not any(
            'save WIP before inter-iteration rebase' in s for s in subjects
        ), f'Empty WIP commit was created on a clean tree: {subjects}'


# ---------------------------------------------------------------------------
# Tests on _verify_debugfix_loop — config gate + happy path + skip path
# ---------------------------------------------------------------------------


@pytest.mark.asyncio
class TestVerifyPhaseRebaseInLoop:
    async def test_rebase_called_before_verify_when_enabled(
        self, config, git_ops, task_assignment, tmp_path, monkeypatch,
    ):
        wt_info = await git_ops.create_worktree(task_assignment.task_id)
        wt = wt_info.path
        workflow, _artifacts = _make_workflow(config, git_ops, task_assignment, wt)

        rebase_mock = AsyncMock(return_value=None)
        workflow._inter_iteration_rebase = rebase_mock  # type: ignore[method-assign]
        # Verify passes immediately so the loop exits after one iteration.
        monkeypatch.setattr(
            'orchestrator.workflow.run_scoped_verification',
            AsyncMock(return_value=VerifyResult(
                passed=True, test_output='ok', lint_output='', type_output='',
                summary='passed',
            )),
        )

        outcome = await workflow._verify_debugfix_loop()
        assert outcome == WorkflowOutcome.DONE
        assert rebase_mock.await_count == 1
        # Must be called with the correct event label.
        assert rebase_mock.await_args is not None
        _, kwargs = rebase_mock.await_args
        assert kwargs.get('event_label') == 'verify_phase_rebase', (
            f'Verify-phase calls must pass event_label=verify_phase_rebase; '
            f'got kwargs={kwargs}'
        )

    async def test_rebase_skipped_when_config_disabled(
        self, config, git_ops, task_assignment, tmp_path, monkeypatch,
    ):
        config.rebase_before_verify = False
        wt_info = await git_ops.create_worktree(task_assignment.task_id)
        wt = wt_info.path
        workflow, _artifacts = _make_workflow(config, git_ops, task_assignment, wt)

        rebase_mock = AsyncMock(return_value=None)
        workflow._inter_iteration_rebase = rebase_mock  # type: ignore[method-assign]
        monkeypatch.setattr(
            'orchestrator.workflow.run_scoped_verification',
            AsyncMock(return_value=VerifyResult(
                passed=True, test_output='ok', lint_output='', type_output='',
                summary='passed',
            )),
        )

        outcome = await workflow._verify_debugfix_loop()
        assert outcome == WorkflowOutcome.DONE
        assert rebase_mock.await_count == 0, (
            'rebase_before_verify=False must skip the rebase entirely.'
        )

    async def test_rebase_skipped_via_helper_when_main_unchanged(
        self, config, git_ops, task_assignment, tmp_path, monkeypatch,
    ):
        """Fast-path: when main has not advanced past base, the helper
        returns None immediately and verify proceeds.

        Drives the real ``_inter_iteration_rebase`` (no monkeypatch) and
        confirms the no-op fast-path: no log entry written, no WIP commit.
        """
        wt_info = await git_ops.create_worktree(task_assignment.task_id)
        wt = wt_info.path
        workflow, artifacts = _make_workflow(config, git_ops, task_assignment, wt)
        artifacts.update_base_commit(wt_info.base_commit)

        # Verify passes so loop exits after one iteration.
        monkeypatch.setattr(
            'orchestrator.workflow.run_scoped_verification',
            AsyncMock(return_value=VerifyResult(
                passed=True, test_output='ok', lint_output='', type_output='',
                summary='passed',
            )),
        )

        outcome = await workflow._verify_debugfix_loop()
        assert outcome == WorkflowOutcome.DONE
        entries, _ = artifacts.read_iteration_log()
        assert not any(
            e.get('event') == 'verify_phase_rebase' for e in entries
        ), (
            f'No rebase log entry should be written when main is unchanged; '
            f'entries: {entries}'
        )

    async def test_rebase_failure_does_not_block_verify_continues(
        self, config, git_ops, task_assignment, tmp_path, monkeypatch,
    ):
        """If the rebase helper fails (returns None silently), the verify
        loop must still proceed — failure is non-blocking."""
        wt_info = await git_ops.create_worktree(task_assignment.task_id)
        wt = wt_info.path
        workflow, _artifacts = _make_workflow(config, git_ops, task_assignment, wt)

        # Helper returns None (simulating "no rebase needed" or "rebase
        # failed silently"); verify proceeds and passes.
        workflow._inter_iteration_rebase = AsyncMock(return_value=None)  # type: ignore[method-assign]
        verify_mock = AsyncMock(return_value=VerifyResult(
            passed=True, test_output='ok', lint_output='', type_output='',
            summary='passed',
        ))
        monkeypatch.setattr(
            'orchestrator.workflow.run_scoped_verification', verify_mock,
        )

        outcome = await workflow._verify_debugfix_loop()
        assert outcome == WorkflowOutcome.DONE
        assert verify_mock.await_count == 1, (
            'Verify must run even when the pre-verify rebase returned None.'
        )

    async def test_rebase_fires_on_each_retry(
        self, config, git_ops, task_assignment, tmp_path, monkeypatch,
    ):
        """Rebase fires before EVERY verify call, not just the first.

        With max_verify_attempts=2 and verify failing once then passing,
        we expect 2 calls to _inter_iteration_rebase (one before each
        verify) and one debugger invocation in between.
        """
        wt_info = await git_ops.create_worktree(task_assignment.task_id)
        wt = wt_info.path
        workflow, _artifacts = _make_workflow(config, git_ops, task_assignment, wt)

        rebase_mock = AsyncMock(return_value=None)
        workflow._inter_iteration_rebase = rebase_mock  # type: ignore[method-assign]
        results = [
            VerifyResult(
                passed=False, test_output='boom', lint_output='', type_output='',
                summary='env collision',
            ),
            VerifyResult(
                passed=True, test_output='ok', lint_output='', type_output='',
                summary='passed',
            ),
        ]
        verify_mock = AsyncMock(side_effect=results)
        monkeypatch.setattr(
            'orchestrator.workflow.run_scoped_verification', verify_mock,
        )

        outcome = await workflow._verify_debugfix_loop()
        assert outcome == WorkflowOutcome.DONE
        assert rebase_mock.await_count == 2, (
            f'Expected 2 rebase calls (one per verify retry); '
            f'got {rebase_mock.await_count}'
        )
        assert verify_mock.await_count == 2
