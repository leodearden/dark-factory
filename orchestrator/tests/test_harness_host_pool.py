"""δ (task 6670): the harness owns the task path's HostPool and its host health.

The harness builds one quarantine set, one :class:`HostPool` over it and one
:class:`TaskVerifyDispatcher` for the process lifetime, independent of the
merge worker's start (PRD ``plans/remote-task-leg-verify-prd.md`` §5.6,
decision 4).  A host the task path benches lands on the merge worker's
RunnerUnavailable tracker (one set, one bench), and a readmission on the merge
lane re-offers the host to the task pool's waiters.

The merge worker is real; the remote host is never reached: no case dispatches
a verify, and the reprobe sweep sleeps for minutes before its first probe.
"""

from __future__ import annotations

import asyncio
import contextlib
import logging
from collections.abc import AsyncIterator
from pathlib import Path
from typing import Any
from unittest.mock import MagicMock, patch

import pytest
import pytest_asyncio
from _gate_fixtures import TEST_DEPLOYED_SHA
from _git_fixtures import seed_repo
from _host_health_helpers import transport_down, until

from orchestrator.config import GitConfig, OrchestratorConfig, VerifyRunnerConfig
from orchestrator.harness import Harness
from orchestrator.merge_lane.liveness import MergeLivenessConfigError
from orchestrator.scheduler import TaskAssignment
from orchestrator.task_verify_dispatch import TaskVerifyDispatcher
from orchestrator.verify_admission_gate import AdmissionPlan

pytestmark = pytest.mark.usefixtures('code_default_config')

_GIT = GitConfig(
    main_branch='main', branch_prefix='task/', remote='origin',
    worktree_dir='.worktrees', push_after_advance=False,
)
_HOST = 'laptop'
_HARNESS_LOGGER = 'orchestrator.harness'


@pytest.fixture(autouse=True)
def _fleet_dir(monkeypatch: pytest.MonkeyPatch, tmp_path: Path) -> None:
    """No heartbeat from a started merge worker lands in the live data/fleet."""
    monkeypatch.setenv('ORCH_FLEET_DIR', str(tmp_path / 'fleet'))


@pytest.fixture
def repo(tmp_path: Path) -> Path:
    return seed_repo(tmp_path / 'repo')


def _laptop() -> VerifyRunnerConfig:
    return VerifyRunnerConfig(
        name=_HOST, ssh_host='laptop.invalid', git_remote='laptop-remote',
        df_checkout_path='/home/laptop/dark-factory',
    )


def _config(repo: Path, **overrides: Any) -> OrchestratorConfig:
    fields: dict[str, Any] = {
        'project_root': repo, 'git': _GIT, 'verify_runners': [_laptop()],
        'verify_host_unreachable_escalate_after_n': 1,
        'verify_admission_order': 'arrival', 'verify_admission_task_slots': 1,
    }
    return OrchestratorConfig(**{**fields, **overrides})


def _harness(config: OrchestratorConfig, *, deployed_sha: Any = TEST_DEPLOYED_SHA) -> Harness:
    with patch('orchestrator.harness.McpLifecycle'), \
         patch('orchestrator.harness.Scheduler'), \
         patch('orchestrator.harness.BriefingAssembler'):
        return Harness(config, deployed_sha=deployed_sha)


@pytest.fixture
def harness(repo: Path) -> Harness:
    return _harness(_config(repo))


@pytest_asyncio.fixture
async def started(harness: Harness) -> AsyncIterator[Harness]:
    await harness._start_merge_worker()
    await harness._launch_merge_worker_loop()
    try:
        yield harness
    finally:
        await harness._stop_merge_worker()


def _laptop_row(harness: Harness) -> dict[str, Any]:
    worker = harness._merge_worker
    assert worker is not None
    rows = {row['name']: row for row in worker.snapshot()['hosts']}
    return rows[_HOST]


class TestBeforeAnyMergeWorker:
    def test_the_pool_and_dispatcher_exist_and_a_bench_is_a_logged_no_op(
        self, harness: Harness, caplog: pytest.LogCaptureFixture,
    ) -> None:
        assert harness._host_pool.hosts == (_HOST,)
        assert isinstance(harness._task_verify_dispatcher, TaskVerifyDispatcher)

        with caplog.at_level(logging.WARNING, logger='orchestrator.harness'):
            harness._host_pool.mark_unavailable(_HOST, transport_down())

        assert [r for r in caplog.records if _HOST in r.getMessage()], caplog.text
        assert harness._host_pool.policy(_HOST, harness.config).slots > 0


@pytest.mark.asyncio
class TestWithTheMergeWorker:
    async def test_the_host_is_managed_at_start_and_shows_its_task_holder(
        self, started: Harness,
    ) -> None:
        assert _laptop_row(started)['task_held_by'] is None

        with started._host_pool.holding(_HOST, '42'):
            row = _laptop_row(started)
            assert (row['task_held_by'], row['slot_state']) == ('task:42', 'free')

    async def test_a_pool_bench_is_one_merge_lane_bench(self, started: Harness) -> None:
        started._host_pool.mark_unavailable(_HOST, transport_down())

        row = _laptop_row(started)
        assert (row['quarantined'], row['quarantine_class'], row['streak']) == (True, 'ru', 1)
        assert started._host_pool.policy(_HOST, started.config).slots == 0

    async def test_a_merge_lane_fault_in_the_bench_never_reaches_the_task_verify(
        self, started: Harness, caplog: pytest.LogCaptureFixture,
    ) -> None:
        """The task lane's call into the merge lane is fault-isolated, as the readmit hook is."""
        assert started._merge_worker is not None
        fault = OSError('escalation queue write failed')

        with patch.object(started._merge_worker, 'bench_unavailable_runner', side_effect=fault), \
             caplog.at_level(logging.ERROR, logger=_HARNESS_LOGGER):
            started._host_pool.mark_unavailable(_HOST, transport_down())

        [error] = [r for r in caplog.records if r.name == _HARNESS_LOGGER and r.levelno == logging.ERROR]
        assert _HOST in error.getMessage()
        assert error.exc_info is not None and error.exc_info[1] is fault

    async def test_a_bench_after_the_worker_stops_is_a_logged_no_op(
        self, started: Harness, caplog: pytest.LogCaptureFixture,
    ) -> None:
        """A stopped worker has no reprobe sweep, so a bench there would strand the host."""
        await started._stop_merge_worker()

        with caplog.at_level(logging.WARNING, logger=_HARNESS_LOGGER):
            started._host_pool.mark_unavailable(_HOST, transport_down())

        assert [r for r in caplog.records if r.name == _HARNESS_LOGGER and _HOST in r.getMessage()]
        assert started._host_pool.policy(_HOST, started.config).slots > 0

    async def test_the_bench_survives_a_merge_worker_rebuild(self, started: Harness) -> None:
        started._host_pool.mark_unavailable(_HOST, transport_down())
        first = started._merge_worker

        await started._stop_merge_worker()
        await started._start_merge_worker()

        assert started._merge_worker is not first
        assert _laptop_row(started)['quarantined'] is True
        assert started._host_pool.policy(_HOST, started.config).slots == 0

    async def test_an_operator_clear_grants_the_oldest_task_waiter_at_once(
        self, started: Harness,
    ) -> None:
        """Row 32: the merge lane's readmission reaches the task pool."""
        pool = started._host_pool
        pool.mark_unavailable(_HOST, transport_down())
        gate = pool.gate(_HOST)
        older = AdmissionPlan.begin(role='task', task_id='old')
        younger = AdmissionPlan.begin(role='task', task_id='young')
        release = asyncio.Event()

        async def wait_turn(plan: AdmissionPlan) -> None:
            request = plan.request(module=None, scope_kind=None)
            async with gate.admit(request, lambda: pool.policy(_HOST, started.config)):
                await release.wait()

        waiters = [asyncio.create_task(wait_turn(p)) for p in (younger, older)]
        await until(lambda: gate.snapshot().depth == 2, what='both task waiters to queue')
        assert gate.snapshot().holders == ()

        assert started._merge_worker is not None
        outcome = await started._merge_worker.clear_host_quarantine(_HOST)

        assert outcome['cleared'] is True
        assert [h.request.task_id for h in gate.snapshot().holders] == ['old']
        release.set()
        async with asyncio.timeout(5):
            await asyncio.gather(*waiters)


@pytest.mark.asyncio
class TestMergeWorkerRefusal:
    async def test_the_pool_outlives_a_refused_merge_worker_start(self, harness: Harness) -> None:
        with patch('orchestrator.merge_lane.liveness.INFLIGHT_MERGE_WORKTREE_LIVENESS_SECS', 1.0), \
             pytest.raises(MergeLivenessConfigError, match='startup refused'):
            await harness._start_merge_worker()

        assert harness._merge_worker is None
        assert harness._host_pool.hosts == (_HOST,)
        harness._host_pool.mark_unavailable(_HOST, transport_down())


class TestNoRemoteHosts:
    def test_no_deployed_sha_leaves_the_pool_empty(self, repo: Path) -> None:
        harness = _harness(_config(repo), deployed_sha=None)

        assert harness._host_pool.hosts == ()
        assert isinstance(harness._task_verify_dispatcher, TaskVerifyDispatcher)

    def test_no_enabled_runner_leaves_the_pool_empty(self, repo: Path) -> None:
        harness = _harness(_config(repo, verify_runners=[]))

        assert harness._host_pool.hosts == ()
        assert isinstance(harness._task_verify_dispatcher, TaskVerifyDispatcher)


@pytest.mark.asyncio
class TestRunSlotInjectsTheDispatcher:
    async def test_build_workflow_receives_the_harness_dispatcher(self, harness: Harness) -> None:
        harness.scheduler.is_deterministic = MagicMock(return_value=False)
        assignment = TaskAssignment(task_id='42', task={'title': 't'}, modules=[])

        with patch('orchestrator.harness.build_workflow') as build:
            async def wedge() -> None:
                await asyncio.sleep(3600)

            build.return_value.run = wedge
            slot = asyncio.create_task(harness._run_slot(assignment, asyncio.Semaphore(0)))
            try:
                await until(lambda: build.called, what='the slot to build its workflow')
                assert build.call_args.kwargs['task_verify'] is harness._task_verify_dispatcher
            finally:
                harness.hard_cancel_workflow('42')
                slot.cancel()
                with contextlib.suppress(asyncio.CancelledError, Exception):
                    await slot
