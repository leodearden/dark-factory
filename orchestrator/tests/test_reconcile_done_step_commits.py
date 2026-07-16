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
