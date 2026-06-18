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
# step-07 RED: _aggregate_results duration_secs propagation
# ---------------------------------------------------------------------------


def _make_verify_result(passed: bool = True, duration_secs: float = 0.0) -> 'VerifyResult':
    from orchestrator.verify import VerifyResult
    return VerifyResult(
        passed=passed, test_output='', lint_output='', type_output='',
        summary='ok', duration_secs=duration_secs,
    )


def test_aggregate_results_multi_children_duration_is_max():
    """_aggregate_results sets duration_secs to max across children (wall approx)."""
    from orchestrator.verify import _aggregate_results
    children = [
        _make_verify_result(duration_secs=1.0),
        _make_verify_result(duration_secs=4.0),
        _make_verify_result(duration_secs=2.0),
    ]
    result = _aggregate_results(children)
    assert result.duration_secs == pytest.approx(4.0), (
        f'Expected max 4.0, got {result.duration_secs}'
    )


def test_aggregate_results_single_child_preserved():
    """_aggregate_results fast-path (len==1) returns child unchanged, duration preserved."""
    from orchestrator.verify import _aggregate_results
    child = _make_verify_result(duration_secs=7.5)
    result = _aggregate_results([child])
    assert result is child, 'Fast path must return the same object'
    assert result.duration_secs == pytest.approx(7.5)


def test_aggregate_results_all_zero_duration():
    """_aggregate_results with all-zero children returns 0.0."""
    from orchestrator.verify import _aggregate_results
    children = [_make_verify_result(duration_secs=0.0)] * 3
    result = _aggregate_results(children)
    assert result.duration_secs == pytest.approx(0.0)


# ---------------------------------------------------------------------------
# step-09 RED: EventStore.fetch_events_by_type
# ---------------------------------------------------------------------------


def test_fetch_events_by_type_returns_matching_rows(tmp_path):
    """fetch_events_by_type returns only rebase_verify_cost rows for this run_id."""
    from orchestrator.event_store import EventStore, EventType

    db = tmp_path / 'events.db'
    store = EventStore(db_path=db, run_id='run-abc')

    data1 = {'old_base': 'aaa', 'new_base': 'bbb', 'distance_commits': 3, 'cohort': 'continuous'}
    data2 = {'old_base': 'bbb', 'new_base': 'ccc', 'distance_commits': 30, 'cohort': 'post-unblock'}

    store.emit(EventType.rebase_verify_cost, task_id='42', phase='verify', data=data1)
    store.emit(EventType.rebase_verify_cost, task_id='42', phase='verify', data=data2)
    # Unrelated event (different type) — must not appear in results
    store.emit(EventType.phase_enter, task_id='42', phase='verify', data={'phase': 'verify'})

    rows = store.fetch_events_by_type('rebase_verify_cost')
    assert len(rows) == 2, f'Expected 2 rows, got {len(rows)}: {rows}'

    # Each row must have its data column parsed back to a dict.
    assert isinstance(rows[0]['data'], dict), 'data must be a dict, not a string'
    assert rows[0]['data']['cohort'] == 'continuous'
    assert rows[1]['data']['cohort'] == 'post-unblock'


def test_fetch_events_by_type_empty_for_unknown_type(tmp_path):
    """fetch_events_by_type returns [] for a type with no rows."""
    from orchestrator.event_store import EventStore

    db = tmp_path / 'events_empty.db'
    store = EventStore(db_path=db, run_id='run-xyz')
    store.emit(
        __import__('orchestrator.event_store', fromlist=['EventType']).EventType.phase_enter,
        task_id='1', phase='execute', data={},
    )
    rows = store.fetch_events_by_type('rebase_verify_cost')
    assert rows == [], f'Expected empty list, got {rows}'


def test_fetch_events_by_type_scoped_to_run_id(tmp_path):
    """fetch_events_by_type is scoped to the current run_id (no cross-run leakage)."""
    from orchestrator.event_store import EventStore, EventType

    db = tmp_path / 'events_scoped.db'
    store_a = EventStore(db_path=db, run_id='run-A')
    store_b = EventStore(db_path=db, run_id='run-B')

    # run-A emits a rebase_verify_cost event.
    store_a.emit(EventType.rebase_verify_cost, task_id='1', data={'cohort': 'continuous'})
    # run-B should NOT see run-A's event.
    rows = store_b.fetch_events_by_type('rebase_verify_cost')
    assert rows == [], f'run-B must not see run-A events; got {rows}'

    # run-A should see exactly its own event.
    rows_a = store_a.fetch_events_by_type('rebase_verify_cost')
    assert len(rows_a) == 1
    assert rows_a[0]['data']['cohort'] == 'continuous'


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
