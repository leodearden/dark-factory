"""The hand-driven merge-worker support module's own contract (``_merge_worker_drive``)."""
from __future__ import annotations

import asyncio
from pathlib import Path

import pytest
from _git_fixtures import seed_repo
from _merge_lane_fakes import FakeVerifier, lane_scene_config
from _merge_worker_drive import (
    HEAVY_BARRIER_MEASURED_UNDER_LOAD_WORST_CASE,
    HEAVY_BARRIER_TEST_TIMEOUT,
    make_merge_request,
    make_real_merge_item,
    stop_merge_worker,
)
from _orch_helpers import (
    PYPROJECT_DEFAULT_TIMEOUT,
    STOP_WORKER_JOIN_TIMEOUT,
    UNDER_LOAD_HEADROOM_FACTOR,
)
from _wait_budget_audit import item_wait_budgets
from _wait_hygiene_scan import unguarded_worker_teardown_items

from orchestrator.config import GitConfig
from orchestrator.git_ops import GitOps
from orchestrator.merge_lane import MergeLane


@pytest.fixture
def git_config() -> GitConfig:
    return GitConfig(
        main_branch='main',
        branch_prefix='task/',
        remote='origin',
        worktree_dir='.worktrees',
        push_after_advance=False,
    )


@pytest.fixture
def git_ops(tmp_path: Path, git_config: GitConfig) -> GitOps:
    return GitOps(git_config, seed_repo(tmp_path / 'repo'))


class TestMakeMergeRequest:
    def test_builds_from_a_sync_body_on_a_placeholder_future(
        self, tmp_path: Path, git_config: GitConfig,
    ) -> None:
        config = lane_scene_config(tmp_path, git_config)

        req = make_merge_request('t1', 'task/t1', tmp_path, config)

        assert req.task_id == 't1'
        assert req.worktree == tmp_path
        assert req.pre_rebased is False
        assert req.lane == 'normal'
        assert not req.result.done()

    @pytest.mark.asyncio
    async def test_binds_its_future_to_the_running_loop(
        self, tmp_path: Path, git_config: GitConfig,
    ) -> None:
        config = lane_scene_config(tmp_path, git_config)

        req = make_merge_request('t1', 'task/t1', tmp_path, config)

        assert req.result.get_loop() is asyncio.get_running_loop()


class TestMakeRealMergeItem:
    def test_builds_a_dispatch_ready_non_speculative_item(
        self, git_ops: GitOps, git_config: GitConfig,
    ) -> None:
        config = lane_scene_config(git_ops.project_root, git_config)

        item = make_real_merge_item(git_ops, config, 'stop-a', 'aaa')

        expected_wt = git_ops.project_root / '.worktrees' / 'stop-a'
        assert item.speculative is False
        assert item.base_sha == 'aaa'
        assert item.merge_wt == expected_wt
        assert item.merge_result.merge_worktree == expected_wt
        assert item.merge_result.success is True
        assert item.request.task_id == 'stop-a'


async def _run_then_fail(lane: MergeLane) -> None:
    await lane.run()
    raise RuntimeError('the run task failed after the lane stopped')


@pytest.mark.asyncio
class TestStopMergeWorker:
    async def test_joins_the_run_task(self, git_ops: GitOps) -> None:
        lane = MergeLane(git_ops, asyncio.Queue(), verifier=FakeVerifier())
        run_task = asyncio.create_task(lane.run())
        await asyncio.sleep(0)

        await stop_merge_worker(lane, run_task)

        assert run_task.done()

    async def test_a_failing_join_does_not_propagate(self, git_ops: GitOps) -> None:
        lane = MergeLane(git_ops, asyncio.Queue(), verifier=FakeVerifier())
        run_task = asyncio.create_task(_run_then_fail(lane))
        await asyncio.sleep(0)

        await stop_merge_worker(lane, run_task)

        assert run_task.done()
        assert isinstance(run_task.exception(), RuntimeError)


class TestScannersRecogniseTheTeardownHelper:
    """Both wait scanners match the teardown helper by name, so each source
    here spells it through ``stop_merge_worker.__name__``: renaming the helper
    reddens this class instead of silently disabling the two guards.

    Only recognition is pinned here. The billing and finally-guard rules are
    pinned in test_wait_budget_audit.py and test_wait_hygiene_scan.py.
    """

    def test_wait_budget_audit_bills_the_default_join(self) -> None:
        source = f'''
class TestDefaultJoin:
    async def test_it(self):
        await {stop_merge_worker.__name__}(worker, worker_task)
'''
        budgets = item_wait_budgets(source)

        assert budgets == {'TestDefaultJoin::test_it': STOP_WORKER_JOIN_TIMEOUT}

    def test_teardown_scan_reports_a_straight_line_call(self) -> None:
        source = f'''
class TestUnguarded:
    async def test_it(self):
        await wait_responsive(fut, timeout=5.0, label='x')
        await {stop_merge_worker.__name__}(worker, worker_task)
'''
        offenders = unguarded_worker_teardown_items(source)

        assert offenders == ['TestUnguarded::test_it']


class TestHeavyBarrierTimeoutConstant:
    """HEAVY_BARRIER_TEST_TIMEOUT is a literal held up by two floors.

    Whether each marked test's waits fit under it is checked per item by
    test_wait_budget_coverage_guard.py, not here.
    """

    def test_clears_the_measured_wall_clock_with_headroom(self) -> None:
        """The constant keeps UNDER_LOAD_HEADROOM_FACTOR over the measured worst case."""
        required = HEAVY_BARRIER_MEASURED_UNDER_LOAD_WORST_CASE * UNDER_LOAD_HEADROOM_FACTOR
        assert required <= HEAVY_BARRIER_TEST_TIMEOUT, (
            f'HEAVY_BARRIER_TEST_TIMEOUT ({HEAVY_BARRIER_TEST_TIMEOUT}s) no longer '
            f'clears {UNDER_LOAD_HEADROOM_FACTOR}x the measured-under-load worst '
            f'case ({HEAVY_BARRIER_MEASURED_UNDER_LOAD_WORST_CASE}s = {required}s). '
            'Raise the constant (and re-record the measurement) rather than '
            'relaxing this arithmetic.'
        )

    def test_never_narrows_the_ini_default(self) -> None:
        """The constant never falls below PYPROJECT_DEFAULT_TIMEOUT."""
        assert HEAVY_BARRIER_TEST_TIMEOUT >= PYPROJECT_DEFAULT_TIMEOUT, (
            f'HEAVY_BARRIER_TEST_TIMEOUT ({HEAVY_BARRIER_TEST_TIMEOUT}s) is below '
            f'the pyproject default ({PYPROJECT_DEFAULT_TIMEOUT}s). A timeout mark '
            'REPLACES the ambient budget rather than flooring it (see '
            '_orch_helpers.VERIFY_CLI_PER_TEST_TIMEOUT), so below the ini default '
            'the merge-barrier classes carrying this mark run TIGHTER than every '
            'unmarked test on a bare local or agent run. Raise the constant.'
        )
