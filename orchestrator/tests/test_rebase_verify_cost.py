"""Tests for rebase-distance → verify-cost instrumentation (task 1802).

Covers:
  - GitOps.get_rebase_distance
  - classify_rebase_cohort (pure function)
  - VerifyResult.duration_secs + _verify_duration_secs helper
  - _aggregate_results duration propagation
  - EventStore.fetch_events_by_type
  - _inter_iteration_rebase enrichment (distance_commits + cohort)
  - _verify_debugfix_loop join + emit
  - summarize_rebase_verify_cost readout
"""

from __future__ import annotations

import asyncio
from pathlib import Path
from unittest.mock import AsyncMock, MagicMock

import pytest

from orchestrator.config import GitConfig, OrchestratorConfig
from orchestrator.git_ops import GitOps, _run


# ---------------------------------------------------------------------------
# Shared git repo fixtures (same pattern as test_verify_phase_rebase.py)
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
        max_verify_attempts=2,
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


# ---------------------------------------------------------------------------
# step-01 RED: GitOps.get_rebase_distance
# ---------------------------------------------------------------------------


@pytest.mark.asyncio
class TestGetRebaseDistance:
    async def test_exact_range_count(self, config, git_ops):
        """get_rebase_distance returns the exact git rev-list count."""
        repo = config.project_root
        # Capture old_base before adding more commits.
        _, old_base, _ = await _run(['git', 'rev-parse', 'HEAD'], cwd=repo)
        old_base = old_base.strip()

        # Add N=3 commits on main beyond old_base.
        n = 3
        for i in range(n):
            (repo / f'extra_{i}.txt').write_text(f'content {i}\n')
            await _run(['git', 'add', '-A'], cwd=repo)
            await _run(['git', 'commit', '-m', f'extra commit {i}'], cwd=repo)

        _, new_base, _ = await _run(['git', 'rev-parse', 'HEAD'], cwd=repo)
        new_base = new_base.strip()

        distance = await git_ops.get_rebase_distance(old_base, new_base)
        assert distance == n, (
            f'Expected distance {n}, got {distance} for {old_base}..{new_base}'
        )

    async def test_equal_refs_returns_zero(self, config, git_ops):
        """When old_base == new_base the distance is 0."""
        repo = config.project_root
        _, sha, _ = await _run(['git', 'rev-parse', 'HEAD'], cwd=repo)
        sha = sha.strip()
        distance = await git_ops.get_rebase_distance(sha, sha)
        assert distance == 0

    async def test_bogus_ref_returns_minus_one(self, git_ops):
        """Unknown/bogus ref must return -1 (fail-safe sentinel)."""
        distance = await git_ops.get_rebase_distance(
            'deadbeefdeadbeefdeadbeefdeadbeefdeadbeef',
            'cafecafecafecafecafecafecafecafecafecafe',
        )
        assert distance == -1, (
            f'Expected -1 sentinel for bogus refs, got {distance}'
        )
