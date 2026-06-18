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

from orchestrator.artifacts import TaskArtifacts
from orchestrator.config import GitConfig, OrchestratorConfig
from orchestrator.git_ops import GitOps, _run
from orchestrator.scheduler import TaskAssignment
from orchestrator.verify import VerifyResult
from orchestrator.workflow import TaskWorkflow

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
    event_store=None,
) -> tuple[TaskWorkflow, TaskArtifacts]:
    """Wire a minimal TaskWorkflow with heavy collaborators mocked.

    Mirrors the pattern in test_verify_phase_rebase.py._make_workflow.
    Accepts an optional event_store so step-13+ tests can wire a real one.
    """
    from orchestrator.agents.invoke import AgentResult

    workflow = TaskWorkflow(
        assignment=assignment,
        config=config,
        git_ops=git_ops,
        scheduler=MagicMock(),  # type: ignore[arg-type]
        briefing=MagicMock(),  # type: ignore[arg-type]
        mcp=MagicMock(),  # type: ignore[arg-type]
        event_store=event_store,
    )
    workflow.worktree = worktree
    artifacts = TaskArtifacts(worktree)
    artifacts.init('42', 'X', 'desc', base_commit='base-sha-old')
    workflow.artifacts = artifacts
    workflow.plan = {'task_id': '42', 'steps': []}
    workflow._check_escalations = MagicMock(return_value=[])  # type: ignore[method-assign]
    workflow.briefing.build_debugger_prompt = AsyncMock(return_value='debug')  # type: ignore[attr-defined]
    workflow._invoke = AsyncMock(  # type: ignore[method-assign]
        return_value=AgentResult(success=True, output=''),
    )
    workflow._get_head_commit = AsyncMock(return_value='head-sha')  # type: ignore[method-assign]
    return workflow, artifacts


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


def _make_verify_result(passed: bool = True, duration_secs: float = 0.0) -> VerifyResult:
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


# ---------------------------------------------------------------------------
# step-11 RED: _inter_iteration_rebase distance+cohort enrichment
# ---------------------------------------------------------------------------


@pytest.mark.asyncio
class TestInterIterationRebaseEnrichment:
    async def test_iteration_log_has_distance_and_cohort(
        self, config, git_ops, task_assignment,
    ):
        """After a real rebase, the iteration log entry gains distance_commits + cohort."""
        # Use threshold=2 so 3 commits always lands in a non-continuous cohort.
        config.rebase_reseed_distance_threshold = 2
        repo = config.project_root

        wt_info = await git_ops.create_worktree(task_assignment.task_id)
        wt = wt_info.path

        # Advance main by N=3 commits beyond the worktree's base.
        n = 3
        for i in range(n):
            (repo / f'enrich_{i}.txt').write_text(f'{i}\n')
            await _run(['git', 'add', '-A'], cwd=repo)
            await _run(['git', 'commit', '-m', f'enrich {i}'], cwd=repo)

        workflow, artifacts = _make_workflow(config, git_ops, task_assignment, wt)
        artifacts.update_base_commit(wt_info.base_commit)

        result = await workflow._inter_iteration_rebase(event_label='verify_phase_rebase')

        assert result is not None, 'Expected a real rebase (main advanced by 3 commits)'

        # (a) iteration log entry must carry distance_commits and cohort
        entries, _ = artifacts.read_iteration_log()
        rebase_entries = [e for e in entries if e.get('event') == 'verify_phase_rebase']
        assert len(rebase_entries) == 1, f'Expected 1 rebase entry; got: {entries}'
        entry = rebase_entries[0]
        assert entry.get('distance_commits') == n, (
            f'Expected distance_commits={n}, got {entry.get("distance_commits")}'
        )
        assert isinstance(entry.get('cohort'), str), (
            f'Expected cohort to be a string, got {entry.get("cohort")!r}'
        )

        # (b) returned dict must include distance_commits, cohort, is_first_rebase
        assert result.get('distance_commits') == n, (
            f'Returned dict: expected distance_commits={n}, got {result.get("distance_commits")}'
        )
        assert isinstance(result.get('cohort'), str), (
            f'Returned dict: expected cohort string, got {result.get("cohort")!r}'
        )
        assert result.get('is_first_rebase') is True, (
            f'Returned dict: expected is_first_rebase=True on first rebase, '
            f'got {result.get("is_first_rebase")!r}'
        )

    async def test_first_rebase_above_threshold_is_post_unblock(
        self, config, git_ops, task_assignment,
    ):
        """First rebase with distance >= threshold → cohort 'post-unblock'."""
        config.rebase_reseed_distance_threshold = 2
        repo = config.project_root

        wt_info = await git_ops.create_worktree(task_assignment.task_id)
        wt = wt_info.path

        # 3 commits, threshold=2: 3>=2 + is_first → 'post-unblock'
        for i in range(3):
            (repo / f'pu_{i}.txt').write_text(f'{i}\n')
            await _run(['git', 'add', '-A'], cwd=repo)
            await _run(['git', 'commit', '-m', f'pu {i}'], cwd=repo)

        workflow, artifacts = _make_workflow(config, git_ops, task_assignment, wt)
        artifacts.update_base_commit(wt_info.base_commit)

        result = await workflow._inter_iteration_rebase(event_label='verify_phase_rebase')

        assert result is not None
        assert result['cohort'] == 'post-unblock', (
            f'Expected post-unblock (first rebase, distance 3 >= threshold 2); '
            f'got {result["cohort"]!r}'
        )
        assert result['is_first_rebase'] is True

    async def test_first_rebase_below_threshold_is_continuous(
        self, config, git_ops, task_assignment,
    ):
        """First rebase with distance < threshold → cohort 'continuous'."""
        config.rebase_reseed_distance_threshold = 5
        repo = config.project_root

        wt_info = await git_ops.create_worktree(task_assignment.task_id)
        wt = wt_info.path

        # 2 commits, threshold=5: 2 < 5 → 'continuous'
        for i in range(2):
            (repo / f'cont_{i}.txt').write_text(f'{i}\n')
            await _run(['git', 'add', '-A'], cwd=repo)
            await _run(['git', 'commit', '-m', f'cont {i}'], cwd=repo)

        workflow, artifacts = _make_workflow(config, git_ops, task_assignment, wt)
        artifacts.update_base_commit(wt_info.base_commit)

        result = await workflow._inter_iteration_rebase(event_label='verify_phase_rebase')

        assert result is not None
        assert result['cohort'] == 'continuous', (
            f'Expected continuous (distance 2 < threshold 5); got {result["cohort"]!r}'
        )


# ---------------------------------------------------------------------------
# step-13 RED: _verify_debugfix_loop rebase→verify join + emit
# ---------------------------------------------------------------------------


@pytest.mark.asyncio
class TestVerifyDebugfixLoopEmit:
    async def test_emits_one_rebase_verify_cost_event_when_main_advanced(
        self, config, git_ops, task_assignment, tmp_path, monkeypatch,
    ):
        """When main advanced, _verify_debugfix_loop emits ONE rebase_verify_cost event.

        Checks: data contains old_base, new_base, distance_commits, files_changed_on_main
        (as a count), next_verify_wall_secs, cohort, and verify_scope.
        """
        from orchestrator.event_store import EventStore
        from orchestrator.verify import VerifyResult

        config.rebase_reseed_distance_threshold = 2  # small threshold

        db = tmp_path / 'events.db'
        store = EventStore(db_path=db, run_id='run-join-1')

        wt_info = await git_ops.create_worktree(task_assignment.task_id)
        wt = wt_info.path

        # Advance main by 3 commits (>= threshold=2 → post-unblock for first rebase).
        repo = config.project_root
        n = 3
        for i in range(n):
            (repo / f'join_{i}.txt').write_text(f'{i}\n')
            await _run(['git', 'add', '-A'], cwd=repo)
            await _run(['git', 'commit', '-m', f'join {i}'], cwd=repo)

        workflow, artifacts = _make_workflow(config, git_ops, task_assignment, wt, event_store=store)
        artifacts.update_base_commit(wt_info.base_commit)

        expected_duration = 5.0
        monkeypatch.setattr(
            'orchestrator.workflow.run_scoped_verification',
            AsyncMock(return_value=VerifyResult(
                passed=True, test_output='ok', lint_output='', type_output='',
                summary='passed', duration_secs=expected_duration,
            )),
        )

        from orchestrator.workflow import WorkflowOutcome
        outcome = await workflow._verify_debugfix_loop()
        assert outcome == WorkflowOutcome.DONE

        rows = store.fetch_events_by_type('rebase_verify_cost')
        assert len(rows) == 1, f'Expected exactly 1 rebase_verify_cost event; got {len(rows)}: {rows}'

        data = rows[0]['data']
        assert 'old_base' in data, f'data missing old_base: {data}'
        assert 'new_base' in data, f'data missing new_base: {data}'
        assert data.get('distance_commits') == n, (
            f'Expected distance_commits={n}, got {data.get("distance_commits")}'
        )
        assert isinstance(data.get('files_changed_on_main'), int), (
            f'files_changed_on_main must be int count; got {data.get("files_changed_on_main")!r}'
        )
        assert data.get('next_verify_wall_secs') == pytest.approx(expected_duration), (
            f'Expected next_verify_wall_secs={expected_duration}, got {data.get("next_verify_wall_secs")}'
        )
        assert isinstance(data.get('cohort'), str), (
            f'cohort must be a string; got {data.get("cohort")!r}'
        )
        assert 'verify_scope' in data, f'data missing verify_scope: {data}'
        assert isinstance(data['verify_scope'], dict), (
            f'verify_scope must be a dict; got {data["verify_scope"]!r}'
        )

    async def test_emits_zero_events_when_main_unchanged(
        self, config, git_ops, task_assignment, tmp_path, monkeypatch,
    ):
        """When main has NOT advanced (rebase short-circuits to None), no event is emitted."""
        from orchestrator.event_store import EventStore
        from orchestrator.verify import VerifyResult

        db = tmp_path / 'events_noop.db'
        store = EventStore(db_path=db, run_id='run-join-2')

        wt_info = await git_ops.create_worktree(task_assignment.task_id)
        wt = wt_info.path

        workflow, artifacts = _make_workflow(config, git_ops, task_assignment, wt, event_store=store)
        # Set base_commit to the CURRENT main sha — so main == base → no rebase.
        artifacts.update_base_commit(wt_info.base_commit)

        monkeypatch.setattr(
            'orchestrator.workflow.run_scoped_verification',
            AsyncMock(return_value=VerifyResult(
                passed=True, test_output='ok', lint_output='', type_output='',
                summary='passed', duration_secs=2.0,
            )),
        )

        from orchestrator.workflow import WorkflowOutcome
        outcome = await workflow._verify_debugfix_loop()
        assert outcome == WorkflowOutcome.DONE

        rows = store.fetch_events_by_type('rebase_verify_cost')
        assert rows == [], (
            f'Expected no rebase_verify_cost events when main unchanged; got {rows}'
        )


# ---------------------------------------------------------------------------
# step-15 RED: summarize_rebase_verify_cost pure readout function
# ---------------------------------------------------------------------------


def _make_row(distance: int, verify_secs: float, cohort: str) -> dict:
    """Construct a synthetic fetch_events_by_type-style row."""
    return {
        'data': {
            'distance_commits': distance,
            'next_verify_wall_secs': verify_secs,
            'cohort': cohort,
        }
    }


def test_summarize_rebase_verify_cost_groups_by_cohort():
    """summarize_rebase_verify_cost groups by cohort with n, distance_p50, verify_secs_p50."""
    from orchestrator.rebase_cost_readout import summarize_rebase_verify_cost

    rows = [
        # continuous cohort: 3 rows
        _make_row(1, 10.0, 'continuous'),
        _make_row(3, 20.0, 'continuous'),
        _make_row(5, 30.0, 'continuous'),
        # post-unblock cohort: 2 rows
        _make_row(100, 120.0, 'post-unblock'),
        _make_row(200, 240.0, 'post-unblock'),
    ]

    summary = summarize_rebase_verify_cost(rows)

    assert 'continuous' in summary, f'Expected continuous key; got {summary.keys()}'
    assert 'post-unblock' in summary, f'Expected post-unblock key; got {summary.keys()}'

    cont = summary['continuous']
    assert cont['n'] == 3, f'Expected n=3 for continuous; got {cont}'
    assert cont['distance_p50'] == pytest.approx(3.0), (
        f'Expected distance_p50=3.0; got {cont["distance_p50"]}'
    )
    assert cont['verify_secs_p50'] == pytest.approx(20.0), (
        f'Expected verify_secs_p50=20.0; got {cont["verify_secs_p50"]}'
    )

    pu = summary['post-unblock']
    assert pu['n'] == 2, f'Expected n=2 for post-unblock; got {pu}'
    # median of [100, 200] = 150.0
    assert pu['distance_p50'] == pytest.approx(150.0), (
        f'Expected distance_p50=150.0; got {pu["distance_p50"]}'
    )
    assert pu['verify_secs_p50'] == pytest.approx(180.0), (
        f'Expected verify_secs_p50=180.0; got {pu["verify_secs_p50"]}'
    )


def test_summarize_rebase_verify_cost_empty_input():
    """summarize_rebase_verify_cost returns {} for empty input (no crash)."""
    from orchestrator.rebase_cost_readout import summarize_rebase_verify_cost
    result = summarize_rebase_verify_cost([])
    assert result == {}, f'Expected empty dict; got {result}'


def test_summarize_rebase_verify_cost_single_cohort_single_row():
    """Single row: n=1, median == the value itself."""
    from orchestrator.rebase_cost_readout import summarize_rebase_verify_cost
    rows = [_make_row(7, 3.5, 'big-jump')]
    summary = summarize_rebase_verify_cost(rows)
    assert 'big-jump' in summary
    bj = summary['big-jump']
    assert bj['n'] == 1
    assert bj['distance_p50'] == pytest.approx(7.0)
    assert bj['verify_secs_p50'] == pytest.approx(3.5)


def test_summarize_rebase_verify_cost_flat_rows_without_data_wrapper():
    """Rows without a 'data' key (flat dict) should also be handled."""
    from orchestrator.rebase_cost_readout import summarize_rebase_verify_cost
    # Flat dict: no 'data' wrapper — the impl reads from row directly if 'data' absent.
    flat_rows = [
        {'distance_commits': 2, 'next_verify_wall_secs': 5.0, 'cohort': 'continuous'},
        {'distance_commits': 4, 'next_verify_wall_secs': 7.0, 'cohort': 'continuous'},
    ]
    summary = summarize_rebase_verify_cost(flat_rows)
    assert 'continuous' in summary
    assert summary['continuous']['n'] == 2
    assert summary['continuous']['distance_p50'] == pytest.approx(3.0)
