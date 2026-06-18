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


# ---------------------------------------------------------------------------
# step-03 RED: classify_rebase_cohort pure function
# ---------------------------------------------------------------------------


# ---------------------------------------------------------------------------
# step-05 RED: VerifyResult.duration_secs field + _verify_duration_secs helper
# ---------------------------------------------------------------------------


def test_verify_result_has_duration_secs_default_zero():
    """VerifyResult must accept a duration_secs field defaulting to 0.0."""
    from orchestrator.verify import VerifyResult
    r = VerifyResult(passed=True, test_output='', lint_output='', type_output='', summary='ok')
    assert hasattr(r, 'duration_secs'), 'VerifyResult must have duration_secs field'
    assert r.duration_secs == 0.0, f'Expected 0.0 default, got {r.duration_secs}'


def test_verify_result_duration_secs_settable():
    """duration_secs can be set explicitly on construction."""
    from orchestrator.verify import VerifyResult
    r = VerifyResult(
        passed=True, test_output='', lint_output='', type_output='',
        summary='ok', duration_secs=3.14,
    )
    assert r.duration_secs == pytest.approx(3.14)


@pytest.mark.parametrize('runs,expected', [
    # Three runs with mixed durations
    (
        [{'duration_secs': 1.5}, {'duration_secs': 2.0}, {'duration_secs': 0.0}],
        3.5,
    ),
    # Single run
    ([{'duration_secs': 4.2}], 4.2),
    # Empty list → 0.0
    ([], 0.0),
    # Runs without duration_secs key → treated as 0.0 per entry
    ([{'rc': 0}, {'rc': 0}], 0.0),
    # Missing key in some entries
    ([{'duration_secs': 1.0}, {'rc': 0}], 1.0),
])
def test_verify_duration_secs(runs, expected):
    """_verify_duration_secs sums per-command duration_secs (default 0.0 per entry)."""
    from orchestrator.verify import _verify_duration_secs
    result = _verify_duration_secs(runs)
    assert result == pytest.approx(expected), (
        f'_verify_duration_secs({runs}) => {result}, expected {expected}'
    )


# ---------------------------------------------------------------------------
# step-03 RED: classify_rebase_cohort pure function
# ---------------------------------------------------------------------------


@pytest.mark.parametrize('distance,is_first,threshold,expected', [
    # distance < 0 → 'unknown'
    (-1, True, 25, 'unknown'),
    (-1, False, 25, 'unknown'),
    # 0 <= distance < threshold → 'continuous' (ignores is_first_rebase)
    (0, True, 25, 'continuous'),
    (0, False, 25, 'continuous'),
    (24, True, 25, 'continuous'),
    (24, False, 25, 'continuous'),
    (1, True, 25, 'continuous'),
    # distance == threshold and is_first → 'post-unblock' (exact boundary)
    (25, True, 25, 'post-unblock'),
    # distance == threshold and not is_first → 'big-jump' (exact boundary)
    (25, False, 25, 'big-jump'),
    # distance > threshold and is_first → 'post-unblock'
    (100, True, 25, 'post-unblock'),
    # distance > threshold and not is_first → 'big-jump'
    (100, False, 25, 'big-jump'),
    # Different threshold value
    (10, True, 10, 'post-unblock'),
    (9, True, 10, 'continuous'),
    (9, False, 10, 'continuous'),
])
def test_classify_rebase_cohort(distance, is_first, threshold, expected):
    from orchestrator.workflow import classify_rebase_cohort
    result = classify_rebase_cohort(distance, is_first, threshold)
    assert result == expected, (
        f'classify_rebase_cohort({distance}, {is_first}, {threshold}) '
        f'=> {result!r}, expected {expected!r}'
    )
