"""δ (task 6670): the merge worker's host-health seams for the harness-owned pool.

The harness owns one quarantine set for the process lifetime and hands it to
each merge worker it builds (PRD ``plans/remote-task-leg-verify-prd.md``
decision 4, §5.6), together with a reader of the task holds on each remote
host.  So a host the task path benches is out of the merge pool too, before
the first merge, and the merge snapshot shows which task holds a host.

Driven through public seams only: constructor keywords,
``ensure_host_allocator``, ``bench_unavailable_runner``, ``snapshot()`` and
the allocator's public ``acquire_remote`` / ``is_quarantined``.  The worker
runs over a real ``GitOps`` on a temporary repository and a real escalation
queue; the remote is a fake whose answers the test sets.
"""

from __future__ import annotations

import asyncio
import logging
from pathlib import Path
from typing import Any, ClassVar

import pytest
from _gate_fixtures import TEST_DEPLOYED_SHA
from _git_fixtures import seed_repo
from _host_health_helpers import transport_down, until
from _merge_lane_fakes import (
    FakeClock,
    FakeVerifier,
    healthy_verify_clock,
    lane_scene_config,
    make_lane,
    passes,
    running_lane,
)
from _orch_helpers import wait_responsive
from escalation.queue import EscalationQueue

from orchestrator.config import GitConfig, OrchestratorConfig, VerifyRunnerConfig
from orchestrator.git_ops import GitOps
from orchestrator.merge_lane.types import QueuedBranch
from orchestrator.merge_lane.worker import MergeRequest, SpeculativeMergeWorker
from orchestrator.verify import VerifyResult
from orchestrator.verify_runner import MergeVerifySpec

pytestmark = pytest.mark.usefixtures('code_default_config')

_GIT = GitConfig(
    main_branch='main', branch_prefix='task/', remote='origin',
    worktree_dir='.worktrees', push_after_advance=False,
)
_HOST = 'laptop'


@pytest.fixture
def repo(tmp_path: Path) -> Path:
    return seed_repo(tmp_path / 'repo')


def _config(repo: Path, **overrides: Any) -> OrchestratorConfig:
    return lane_scene_config(repo, _GIT, **overrides)


class _Laptop:
    """A remote verify host whose reachability and verify the test controls."""

    is_local: ClassVar[bool] = False

    def __init__(self, name: str = _HOST, *, release: asyncio.Event | None = None) -> None:
        self.name = name
        self.healthy = True
        self.clean = True
        self.cancel_rc = 0
        self.release = release
        self.entered = asyncio.Event()
        self.verified: list[str] = []

    async def health(self) -> bool:
        return self.healthy

    async def ensure_gate(self, *, event_store: Any = None, task_id: str | None = None) -> None:
        return None

    async def probe_clean(self) -> bool:
        return self.clean

    async def cancel_verify(self) -> int:
        return self.cancel_rc

    async def reap_idle_gates(self) -> None:
        return None

    async def run_merge_verify(self, merge_sha: str, spec: MergeVerifySpec) -> VerifyResult:
        self.verified.append(merge_sha)
        self.entered.set()
        if self.release is not None:
            await self.release.wait()
        return passes().result


def _worker(repo: Path, **kwargs: Any) -> SpeculativeMergeWorker:
    return SpeculativeMergeWorker(
        GitOps(_GIT, repo), asyncio.Queue(), verifier=FakeVerifier(), clock=FakeClock(),
        deployed_sha=TEST_DEPLOYED_SHA, **kwargs,
    )


def _rows(worker: SpeculativeMergeWorker) -> dict[str, dict[str, Any]]:
    return {row['name']: row for row in worker.snapshot()['hosts']}


def _laptop_runner_config() -> VerifyRunnerConfig:
    return VerifyRunnerConfig(
        name=_HOST, ssh_host='laptop.local', git_remote='laptop-remote',
        df_checkout_path='/home/laptop/dark-factory',
    )


class TestInjectedQuarantine:
    def test_the_allocator_reads_the_injected_set_by_reference(self, repo: Path) -> None:
        quarantine: set[str] = set()
        worker = _worker(repo, runner_quarantine=quarantine, remote_runners=[_Laptop()])

        allocator = worker.ensure_host_allocator(_config(repo))

        assert worker.ensure_host_allocator(_config(repo)) is allocator
        assert not allocator.is_quarantined(_HOST)
        quarantine.add(_HOST)
        assert allocator.is_quarantined(_HOST)

    def test_a_bench_lands_in_the_injected_set(self, repo: Path) -> None:
        quarantine: set[str] = set()
        worker = _worker(repo, runner_quarantine=quarantine, remote_runners=[_Laptop()])
        worker.ensure_host_allocator(_config(repo))

        worker.bench_unavailable_runner(_HOST, transport_down())

        assert quarantine == {_HOST}


class TestBenchBeforeAnyMerge:
    def test_the_config_thresholds_govern_the_first_bench(
        self, repo: Path, tmp_path: Path,
    ) -> None:
        escalations = EscalationQueue(tmp_path / 'escalations')
        worker = _worker(repo, escalation_queue=escalations, remote_runners=[_Laptop()])
        worker.ensure_host_allocator(_config(repo, verify_host_unreachable_escalate_after_n=1))

        worker.bench_unavailable_runner(_HOST, transport_down())

        l1s = [esc for esc in escalations.get_pending() if esc.level == 1]
        assert [esc.category for esc in l1s] == ['verify_host_unreachable']
        row = _rows(worker)[_HOST]
        assert (row['quarantined'], row['quarantine_class'], row['streak']) == (True, 'ru', 1)

    def test_a_host_benched_before_the_first_build_is_still_managed(self, repo: Path) -> None:
        worker = _worker(repo)
        config = _config(repo, verify_runners=[_laptop_runner_config()])

        worker.bench_unavailable_runner(_HOST, transport_down())
        allocator = worker.ensure_host_allocator(config)

        assert _HOST in allocator.host_names
        names = [row['name'] for row in worker.snapshot()['hosts']]
        assert names.count(_HOST) == 1
        row = _rows(worker)[_HOST]
        assert row['slot_state'] is not None, 'an orphan row: the build left the host out'
        assert (row['quarantined'], row['quarantine_class']) == (True, 'ru')


class TestTaskHeldBy:
    def test_every_row_names_the_task_holding_its_host(self, repo: Path) -> None:
        worker = _worker(
            repo, remote_runners=[_Laptop()], host_holder_reader=lambda: {_HOST: 'task:42'},
        )
        allocator = worker.ensure_host_allocator(_config(repo))

        rows = _rows(worker)
        assert rows['local']['task_held_by'] is None
        assert (rows[_HOST]['slot_state'], rows[_HOST]['task_held_by']) == ('free', 'task:42')

        assert allocator.acquire_remote() is not None

        row = _rows(worker)[_HOST]
        assert (row['slot_state'], row['task_held_by']) == ('busy', 'task:42')

    def test_an_orphan_row_names_its_holder_too(self, repo: Path) -> None:
        worker = _worker(
            repo, remote_runners=[_Laptop()], host_holder_reader=lambda: {'gone': 'task:9'},
        )
        worker.ensure_host_allocator(_config(repo))

        worker.bench_unavailable_runner('gone', transport_down('gone is gone'))

        orphan = _rows(worker)['gone']
        assert (orphan['slot_state'], orphan['task_held_by']) == (None, 'task:9')

    def test_no_reader_holds_nothing(self, repo: Path) -> None:
        worker = _worker(repo, remote_runners=[_Laptop()])
        worker.ensure_host_allocator(_config(repo))

        rows = worker.snapshot()['hosts']
        assert [row['task_held_by'] for row in rows] == [None] * len(rows)


async def _branch_with_file(git_ops: GitOps, task_id: str, filename: str) -> Path:
    worktree = (await git_ops.create_worktree(task_id)).path
    (worktree / filename).write_text(f'{task_id} = 1\n')
    await git_ops.commit(worktree, f'Add {filename}')
    return worktree


async def _request(git_ops: GitOps, config: OrchestratorConfig, task_id: str) -> MergeRequest:
    worktree = await _branch_with_file(git_ops, task_id, f'{task_id}.py')
    return MergeRequest(
        task_id=task_id,
        branch=QueuedBranch.parse(f'task/{task_id}', config.git.branch_prefix),
        worktree=worktree,
        pre_rebased=False,
        task_files=None,
        module_configs=[],
        config=config,
        result=asyncio.get_running_loop().create_future(),
    )


@pytest.mark.asyncio
class TestSymmetricHost:
    async def test_a_merge_verify_runs_on_a_host_a_task_holds(self, repo: Path) -> None:
        """Row 7: a task hold never takes a merge slot; both show on the one row."""
        release = asyncio.Event()
        laptop = _Laptop(release=release)
        git_ops = GitOps(_GIT, repo)
        config = _config(repo, verify_host_policy='prefer_remote')
        queue: asyncio.Queue[Any] = asyncio.Queue()
        lane = make_lane(
            git_ops, queue, clock=healthy_verify_clock(), remote_runners=[laptop],
            deployed_sha=TEST_DEPLOYED_SHA, host_holder_reader=lambda: {_HOST: 'task:42'},
        )
        request = await _request(git_ops, config, 'row7')

        async with running_lane(lane) as run:
            await queue.put(request)
            await wait_responsive(laptop.entered.wait(), label='the merge verify to reach the laptop')

            row = _rows(lane)[_HOST]
            assert (row['slot_state'], row['task_held_by']) == ('busy', 'task:42')

            release.set()
            outcome = await run.outcome(request)

        assert outcome.status == 'done', outcome
        assert len(laptop.verified) == 1


# ---------------------------------------------------------------------------
# Readmission reaches the task pool through on_host_readmitted
# ---------------------------------------------------------------------------

_LANE_LOGGER = 'orchestrator.merge_lane'


def _benched_worker(
    repo: Path, laptop: _Laptop, **kwargs: Any,
) -> tuple[SpeculativeMergeWorker, Any]:
    worker = _worker(repo, remote_runners=[laptop], **kwargs)
    allocator = worker.ensure_host_allocator(_config(repo))
    worker.bench_unavailable_runner(_HOST, transport_down())
    return worker, allocator


@pytest.mark.asyncio
class TestReadmissionHook:
    async def test_an_operator_clear_calls_the_hook_once(self, repo: Path) -> None:
        readmitted: list[str] = []
        worker, _ = _benched_worker(repo, _Laptop(), on_host_readmitted=readmitted.append)

        outcome = await worker.clear_host_quarantine(_HOST)

        assert outcome['cleared'] is True
        assert readmitted == [_HOST]
        assert _rows(worker)[_HOST]['quarantined'] is False

    async def test_a_reprobe_readmission_calls_the_hook(self, repo: Path, tmp_path: Path) -> None:
        """Row 9 at worker level: no merge is dispatched, the sweep alone readmits."""
        readmitted: list[str] = []
        escalations = EscalationQueue(tmp_path / 'escalations')
        laptop = _Laptop()
        laptop.healthy = False
        lane = make_lane(
            GitOps(_GIT, repo), remote_runners=[laptop], deployed_sha=TEST_DEPLOYED_SHA,
            escalation_queue=escalations, on_host_readmitted=readmitted.append,
        )

        async with running_lane(lane):
            lane.ensure_host_allocator(_config(
                repo, verify_host_unreachable_escalate_after_n=1, verify_host_reprobe_interval_s=1.0,
            ))
            lane.bench_unavailable_runner(_HOST, transport_down())
            row = _rows(lane)[_HOST]
            assert (row['quarantine_class'], row['streak']) == ('ru', 1)
            [alarm] = [esc for esc in escalations.get_pending() if esc.level == 1]
            assert alarm.category == 'verify_host_unreachable'

            laptop.healthy = True
            await until(
                lambda: _rows(lane)[_HOST]['quarantined'] is False,
                what='the reprobe sweep to readmit the laptop',
            )

        assert readmitted == [_HOST]
        [resolved] = [
            esc for esc in escalations.get_by_task(alarm.task_id) if esc.id == alarm.id
        ]
        assert resolved.status == 'resolved'

    async def test_a_refused_readmission_does_not_call_the_hook(self, repo: Path) -> None:
        readmitted: list[str] = []
        laptop = _Laptop()
        worker = _worker(repo, remote_runners=[laptop], on_host_readmitted=readmitted.append)
        allocator = worker.ensure_host_allocator(_config(repo))
        lease = allocator.acquire_remote()
        assert lease is not None
        laptop.cancel_rc, laptop.clean = 1, False
        await allocator.cancel_and_release(lease, max_attempts=1)
        worker.bench_unavailable_runner(_HOST, transport_down())

        outcome = await worker.clear_host_quarantine(_HOST)

        assert outcome['cleared'] is False
        assert readmitted == []
        assert _rows(worker)[_HOST]['quarantined'] is True

    async def test_a_hook_that_raises_is_logged_and_the_readmission_stands(
        self, repo: Path, caplog: pytest.LogCaptureFixture,
    ) -> None:
        def broken_hook(name: str) -> None:
            raise RuntimeError(f'task pool fault for {name}')

        caplog.set_level(logging.ERROR, logger=_LANE_LOGGER)
        worker, _ = _benched_worker(repo, _Laptop(), on_host_readmitted=broken_hook)

        outcome = await worker.clear_host_quarantine(_HOST)

        assert outcome['cleared'] is True
        assert _rows(worker)[_HOST]['quarantined'] is False
        errors = [
            r for r in caplog.records if r.name == _LANE_LOGGER and r.levelno == logging.ERROR
        ]
        assert len(errors) == 1
        assert _HOST in errors[0].getMessage()
        assert errors[0].exc_info is not None

    async def test_no_hook_readmits_as_before(self, repo: Path) -> None:
        worker, _ = _benched_worker(repo, _Laptop())

        outcome = await worker.clear_host_quarantine(_HOST)

        assert outcome['cleared'] is True
        assert _rows(worker)[_HOST]['quarantined'] is False
