"""Tests for surfacing already-committed WIP safety-commits to the implementer (task 2051).

The harness auto-commits uncommitted work as a safety net before several
rebase/requeue/reclaim operations (see git_ops.py and workflow.py). Any of
these can land a still-"pending" plan step's complete implementation at
branch HEAD *before* mark_step_done is called for that step, so every
session has had to re-discover the workaround ad-hoc. This suite covers the
new detection + prompt-surfacing machinery that fixes that:

  - is_wip_safety_commit / WIP_SAFETY_COMMIT_PREFIXES (git_ops.py)
  - GitOps.get_commit_subjects (git_ops.py)
  - TaskWorkflow._detect_tip_wip_commits (workflow.py)
  - BriefingAssembler.build_implementer_prompt's wip_notice rendering (briefing.py)
  - _execute_iterations wiring the detector into the prompt builder (workflow.py)
"""

from __future__ import annotations

import asyncio
from pathlib import Path

import pytest

from orchestrator.config import GitConfig, OrchestratorConfig
from orchestrator.git_ops import WIP_SAFETY_COMMIT_PREFIXES, GitOps, _run, is_wip_safety_commit

# ---------------------------------------------------------------------------
# step-1 RED: is_wip_safety_commit predicate + WIP_SAFETY_COMMIT_PREFIXES
# ---------------------------------------------------------------------------


class TestIsWipSafetyCommit:
    """Pins the predicate to the exact literal subjects the harness produces.

    Producing sites (must stay recognized):
      - workflow.py:4579  'chore: save WIP before inter-iteration rebase'
      - git_ops.py:1134/2496 'chore: save WIP before requeue rebase'
      - git_ops.py:2050  'chore: save WIP before warm-lane reclaim (task 1933)'
    """

    def test_recognizes_requeue_rebase_literal(self):
        assert is_wip_safety_commit('chore: save WIP before requeue rebase') is True

    def test_recognizes_inter_iteration_rebase_literal(self):
        assert is_wip_safety_commit('chore: save WIP before inter-iteration rebase') is True

    def test_recognizes_warm_lane_reclaim_literal(self):
        assert is_wip_safety_commit(
            'chore: save WIP before warm-lane reclaim (task 1933)',
        ) is True

    def test_rejects_normal_feat_commit(self):
        assert is_wip_safety_commit('feat: GREEN — x') is False

    def test_rejects_normal_test_commit(self):
        assert is_wip_safety_commit('test: RED — y') is False

    def test_rejects_empty_string(self):
        assert is_wip_safety_commit('') is False

    def test_rejects_near_miss_subject(self):
        assert is_wip_safety_commit('chore: save the world') is False

    def test_prefixes_tuple_matches_producing_literal(self):
        """WIP_SAFETY_COMMIT_PREFIXES is the single source of truth for the prefix."""
        assert WIP_SAFETY_COMMIT_PREFIXES == ('chore: save WIP before ',)


# ---------------------------------------------------------------------------
# step-3 RED: GitOps.get_commit_subjects
# ---------------------------------------------------------------------------
#
# Real temp git repo pattern, mirroring test_rebase_verify_cost.py's git_repo
# fixture — a real repo is simplest and least brittle for exercising actual
# `git log` output.


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
class TestGetCommitSubjects:
    async def test_head_first_order_and_subjects_match(self, git_repo, git_ops):
        """Two commits on top of base: HEAD-first (sha, subject) tuples, subjects match."""
        _, base_sha, _ = await _run(['git', 'rev-parse', 'HEAD'], cwd=git_repo)
        base_sha = base_sha.strip()

        (git_repo / 'a.txt').write_text('a\n')
        await _run(['git', 'add', '-A'], cwd=git_repo)
        await _run(['git', 'commit', '-m', 'second commit'], cwd=git_repo)
        _, second_sha, _ = await _run(['git', 'rev-parse', 'HEAD'], cwd=git_repo)
        second_sha = second_sha.strip()

        (git_repo / 'b.txt').write_text('b\n')
        await _run(['git', 'add', '-A'], cwd=git_repo)
        await _run(['git', 'commit', '-m', 'third commit'], cwd=git_repo)
        _, third_sha, _ = await _run(['git', 'rev-parse', 'HEAD'], cwd=git_repo)
        third_sha = third_sha.strip()

        result = await git_ops.get_commit_subjects(git_repo, base_sha)

        assert result == [
            (third_sha, 'third commit'),
            (second_sha, 'second commit'),
        ], f'Expected HEAD-first pairs for third/second commit, got {result}'

    async def test_excludes_commits_at_or_before_base(self, git_repo, git_ops):
        """Commits at/before base_sha (e.g. 'Initial') must not appear in the result."""
        _, base_sha, _ = await _run(['git', 'rev-parse', 'HEAD'], cwd=git_repo)
        base_sha = base_sha.strip()

        (git_repo / 'a.txt').write_text('a\n')
        await _run(['git', 'add', '-A'], cwd=git_repo)
        await _run(['git', 'commit', '-m', 'only new commit'], cwd=git_repo)

        result = await git_ops.get_commit_subjects(git_repo, base_sha)

        subjects = [subject for _sha, subject in result]
        assert subjects == ['only new commit']
        assert 'Initial' not in subjects

    async def test_empty_range_returns_empty_list(self, git_repo, git_ops):
        """base_sha == HEAD (no commits since base) must return []."""
        _, head_sha, _ = await _run(['git', 'rev-parse', 'HEAD'], cwd=git_repo)
        head_sha = head_sha.strip()

        result = await git_ops.get_commit_subjects(git_repo, head_sha)

        assert result == []
