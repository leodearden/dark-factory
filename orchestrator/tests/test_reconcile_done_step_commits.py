"""Tests for reconciling a done plan step's stale ``commit`` after an
inter-iteration / warm-lane rebase orphans it (task 2386).

``TaskWorkflow._detect_tip_wip_commits`` (task 2051) surfaces WIP
safety-commits at HEAD for still-PENDING steps, but explicitly dedups any
sha already recorded as a DONE step's ``commit`` — so a done step whose
recorded commit was rewritten/orphaned by ``_inter_iteration_rebase``
(workflow.py) or a warm-lane/requeue rebase (git_ops.py) is invisible to
that detector. This suite covers the new reconciliation machinery that
fixes that:

  - GitOps.get_commit_changed_files (git_ops.py)
  - TaskWorkflow._reconcile_done_step_commits (workflow.py)
  - _execute_iterations wiring the reconciler in before the WIP detector
"""

from __future__ import annotations

import asyncio
from pathlib import Path
from unittest.mock import AsyncMock, MagicMock

import pytest

from orchestrator.agents.invoke import AgentResult
from orchestrator.artifacts import TaskArtifacts
from orchestrator.config import GitConfig, OrchestratorConfig
from orchestrator.git_ops import GitOps, _run
from orchestrator.scheduler import TaskAssignment
from orchestrator.workflow import TaskWorkflow, WorkflowOutcome

# ---------------------------------------------------------------------------
# step-1 RED: GitOps.get_commit_changed_files
# ---------------------------------------------------------------------------
#
# Real temp git repo pattern, mirroring test_harness_wip_step_detection.py's
# git_repo/config/git_ops fixtures.


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


@pytest.mark.asyncio
class TestGetCommitChangedFiles:
    async def test_normal_commit_returns_changed_files_vs_parent(self, git_repo, git_ops):
        """A normal commit's changed-file set is computed vs its sole parent."""
        (git_repo / 'a.txt').write_text('a\n')
        await _run(['git', 'add', '-A'], cwd=git_repo)
        await _run(['git', 'commit', '-m', 'add a'], cwd=git_repo)
        _, sha, _ = await _run(['git', 'rev-parse', 'HEAD'], cwd=git_repo)
        sha = sha.strip()

        result = await git_ops.get_commit_changed_files(sha)

        assert result == ['a.txt']

    async def test_root_commit_returns_its_own_files(self, git_repo, git_ops):
        """A ROOT commit (no parent) must still return the files it introduced.

        Plain `git diff-tree <sha>` (without --root) shows nothing for a
        root commit since it has no parent to diff against — the
        implementation must handle this so an orphaned original commit that
        happens to be the repo's first commit is not silently treated as
        unresolvable.
        """
        _, root_sha, _ = await _run(['git', 'rev-parse', 'HEAD'], cwd=git_repo)
        root_sha = root_sha.strip()

        result = await git_ops.get_commit_changed_files(root_sha)

        assert result == ['lib.py']

    async def test_nonexistent_sha_returns_empty_list(self, git_repo, git_ops):
        """A garbage/nonexistent SHA returns [] defensively, never raises."""
        result = await git_ops.get_commit_changed_files(
            'deadbeefdeadbeefdeadbeefdeadbeefdeadbeef',
        )

        assert result == []

    async def test_multi_file_commit_returns_all_files(self, git_repo, git_ops):
        """A commit touching several files returns the full changed-file set."""
        (git_repo / 'b.txt').write_text('b\n')
        (git_repo / 'c.txt').write_text('c\n')
        await _run(['git', 'add', '-A'], cwd=git_repo)
        await _run(['git', 'commit', '-m', 'add b and c'], cwd=git_repo)
        _, sha, _ = await _run(['git', 'rev-parse', 'HEAD'], cwd=git_repo)
        sha = sha.strip()

        result = await git_ops.get_commit_changed_files(sha)

        assert sorted(result) == ['b.txt', 'c.txt']


# ---------------------------------------------------------------------------
# step-3 RED: TaskWorkflow._reconcile_done_step_commits (core / auto-reconcile)
# ---------------------------------------------------------------------------
#
# Real-git _make_workflow pattern, mirroring test_harness_wip_step_detection.py
# — real GitOps + a real worktree created via git_ops.create_worktree, heavy
# collaborators (scheduler/briefing/mcp) mocked.


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
    *,
    escalation_queue=None,
) -> tuple[TaskWorkflow, TaskArtifacts]:
    """Wire a minimal TaskWorkflow with heavy collaborators mocked.

    Mirrors the pattern in test_harness_wip_step_detection.py._make_workflow.
    """
    workflow = TaskWorkflow(
        assignment=assignment,
        config=config,
        git_ops=git_ops,
        scheduler=MagicMock(),  # type: ignore[arg-type]
        briefing=MagicMock(),  # type: ignore[arg-type]
        mcp=MagicMock(),  # type: ignore[arg-type]
        escalation_queue=escalation_queue,
    )
    workflow.worktree = worktree
    artifacts = TaskArtifacts(worktree)
    artifacts.init('42', 'X', 'desc')
    workflow.artifacts = artifacts
    workflow.plan = {'task_id': '42', 'steps': [], 'prerequisites': []}
    return workflow, artifacts


def _write_done_step_plan(artifacts: TaskArtifacts, step_id: str, commit: str) -> dict:
    """Persist a single-step plan (status='done', the given commit) and
    return the re-read plan so ``workflow.plan`` mirrors what's on disk —
    required because ``update_step_status`` reads/writes plan.json directly,
    independent of any in-memory ``workflow.plan`` the caller also sets."""
    plan = {
        'task_id': '42',
        'title': 'X',
        'analysis': 'A',
        'prerequisites': [],
        'steps': [
            {'id': step_id, 'type': 'impl', 'status': 'done', 'commit': commit},
        ],
    }
    artifacts.write_plan(plan)
    return artifacts.read_plan()


@pytest.mark.asyncio
class TestReconcileDoneStepCommits:
    async def test_orphaned_done_step_commit_matching_wip_tip_is_reconciled(
        self, config, git_ops, task_assignment,
    ):
        """HAPPY PATH: a done step's commit was orphaned (rebase reset the
        branch back to base), but its exact file(s) reappear in a WIP
        safety-commit now sitting at HEAD -> auto-reconcile the step's
        commit to the WIP tip sha."""
        wt_info = await git_ops.create_worktree(task_assignment.task_id)
        wt = wt_info.path
        workflow, artifacts = _make_workflow(config, git_ops, task_assignment, wt)
        artifacts.update_base_commit(wt_info.base_commit)

        # Step-1's original implementation commit.
        (wt / 'feature.py').write_text('original implementation\n')
        step_commit = await git_ops.commit(wt, 'feat: GREEN — step-1 implementation')
        assert step_commit, 'Setup: expected a real commit to be made'

        workflow.plan = _write_done_step_plan(artifacts, 'step-1', step_commit)

        # Simulate the rebase orphaning step_commit: reset hard to base, then
        # re-land the same file as a WIP safety-commit at HEAD (mirrors
        # _inter_iteration_rebase's squash-then-rebase sequence).
        await _run(['git', 'reset', '--hard', wt_info.base_commit], cwd=wt)
        (wt / 'feature.py').write_text('original implementation\n')
        wip_sha = await git_ops.commit(wt, 'chore: save WIP before inter-iteration rebase')
        assert wip_sha, 'Setup: expected a real WIP commit at HEAD'

        await workflow._reconcile_done_step_commits()

        reconciled = artifacts.read_plan()
        assert reconciled['steps'][0]['commit'] == wip_sha, (
            f'Expected step-1 commit re-pointed to WIP tip {wip_sha}, '
            f"got {reconciled['steps'][0]['commit']}"
        )
        assert reconciled['steps'][0]['status'] == 'done'

    async def test_reachable_done_step_commit_is_left_unchanged(
        self, config, git_ops, task_assignment,
    ):
        """A done step's recorded commit that IS reachable from HEAD (the
        ordinary, non-orphaned case) must not be touched."""
        wt_info = await git_ops.create_worktree(task_assignment.task_id)
        wt = wt_info.path
        workflow, artifacts = _make_workflow(config, git_ops, task_assignment, wt)
        artifacts.update_base_commit(wt_info.base_commit)

        (wt / 'feature.py').write_text('implementation\n')
        step_commit = await git_ops.commit(wt, 'feat: GREEN — step-1 implementation')
        assert step_commit

        workflow.plan = _write_done_step_plan(artifacts, 'step-1', step_commit)

        await workflow._reconcile_done_step_commits()

        unchanged = artifacts.read_plan()
        assert unchanged['steps'][0]['commit'] == step_commit

    async def test_worktree_none_is_a_noop(self, config, git_ops, task_assignment):
        wt_info = await git_ops.create_worktree(task_assignment.task_id)
        workflow, artifacts = _make_workflow(config, git_ops, task_assignment, wt_info.path)
        artifacts.update_base_commit(wt_info.base_commit)
        workflow.plan = _write_done_step_plan(artifacts, 'step-1', 'deadbeef')
        workflow.worktree = None

        await workflow._reconcile_done_step_commits()  # must not raise

        assert artifacts.read_plan()['steps'][0]['commit'] == 'deadbeef'

    async def test_git_ops_none_is_a_noop(self, config, git_ops, task_assignment):
        wt_info = await git_ops.create_worktree(task_assignment.task_id)
        workflow, artifacts = _make_workflow(config, git_ops, task_assignment, wt_info.path)
        artifacts.update_base_commit(wt_info.base_commit)
        workflow.plan = _write_done_step_plan(artifacts, 'step-1', 'deadbeef')
        workflow.git_ops = None  # type: ignore[assignment]

        await workflow._reconcile_done_step_commits()  # must not raise

        assert artifacts.read_plan()['steps'][0]['commit'] == 'deadbeef'

    async def test_base_commit_unset_is_a_noop(self, config, git_ops, task_assignment):
        wt_info = await git_ops.create_worktree(task_assignment.task_id)
        wt = wt_info.path
        workflow = TaskWorkflow(
            assignment=task_assignment,
            config=config,
            git_ops=git_ops,
            scheduler=MagicMock(),  # type: ignore[arg-type]
            briefing=MagicMock(),  # type: ignore[arg-type]
            mcp=MagicMock(),  # type: ignore[arg-type]
        )
        workflow.worktree = wt
        artifacts = TaskArtifacts(wt)
        artifacts.init('42', 'X', 'desc')  # no base_commit
        workflow.artifacts = artifacts
        workflow.plan = _write_done_step_plan(artifacts, 'step-1', 'deadbeef')

        await workflow._reconcile_done_step_commits()  # must not raise

        assert artifacts.read_plan()['steps'][0]['commit'] == 'deadbeef'
