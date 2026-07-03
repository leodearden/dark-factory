"""Tests for SpeculativeMergeWorker request-liveness checks + heartbeat wiring
(MQ-invariants eta / task 1992).

Steps covered:
  step-7  RED   — _check_request_liveness unit tests (bare worker)
  step-8  GREEN — implement _check_request_liveness
  step-9  RED   — heartbeat-wiring: liveness check runs before depth==0 early-return
  step-10 GREEN — call _check_request_liveness from the top of _maybe_log_queue_heartbeat
  step-11 RED   — end-to-end wedged-verify integration
  step-12 GREEN — wire the arm hook at the merger-loop head
  step-13 RED   — operator-halt requeue no-false-alarm
  step-14 GREEN — wire on_requeued at the 3 put_nowait sites

This module intentionally imports orchestrator.merge_queue LOCALLY inside each
test method (not at module scope) so a not-yet-implemented symbol
(_check_request_liveness, before step-8) never breaks collection of the rest
of the file — mirrors test_merge_request_ledger.py's step-5 convention.
"""

from __future__ import annotations

import asyncio
import logging
from pathlib import Path

import pytest

from orchestrator.config import GitConfig, OrchestratorConfig
from orchestrator.git_ops import GitOps, _run
from orchestrator.merge_types import MergeOutcome, MergeRequest

# ---------------------------------------------------------------------------
# Fixtures + helpers (per-file duplication convention — see
# test_merge_queue_resolve_release.py / test_merge_queue_concurrent_verify.py)
# ---------------------------------------------------------------------------


async def _setup_repo(repo: Path) -> None:
    """Initialise a bare git repository with a single commit on main."""
    await _run(['git', 'init', '-b', 'main'], cwd=repo)
    await _run(['git', 'config', 'user.email', 'test@test.com'], cwd=repo)
    await _run(['git', 'config', 'user.name', 'Test'], cwd=repo)
    (repo / 'README.md').write_text('# Test\n')
    await _run(['git', 'add', '-A'], cwd=repo)
    await _run(['git', 'commit', '-m', 'Initial commit'], cwd=repo)


@pytest.fixture
def git_repo(tmp_path: Path) -> Path:
    """Create a temporary git repository with an initial commit."""
    repo = tmp_path / 'repo'
    repo.mkdir()
    asyncio.run(_setup_repo(repo))
    return repo


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
def git_ops(git_config: GitConfig, git_repo: Path) -> GitOps:
    return GitOps(git_config, git_repo)


@pytest.fixture
def config(git_repo: Path, git_config: GitConfig) -> OrchestratorConfig:
    """Single-host (no verify_runners) OrchestratorConfig."""
    return OrchestratorConfig(project_root=git_repo, git=git_config)


def _make_request(
    task_id: str,
    branch: str,
    worktree: Path,
    config: OrchestratorConfig,
) -> MergeRequest:
    """Build a MergeRequest with a fresh Future for the running event loop."""
    return MergeRequest(
        task_id=task_id,
        branch=branch,
        worktree=worktree,
        pre_rebased=False,
        task_files=None,
        module_configs=[],
        config=config,
        result=asyncio.get_running_loop().create_future(),
        lane='normal',
    )


class _FakeEscalationQueue:
    """Minimal fake escalation queue (copied from
    test_merge_queue_multihost_wiring.py:1200 — per-file duplication
    convention).
    """

    def __init__(self, *, open_l1: bool = False):
        self._open_l1 = open_l1
        self._seq = 0
        self.submitted: list = []

    def has_open_l1(self, task_id: str) -> bool:  # noqa: ARG002
        return self._open_l1

    def make_id(self, task_id: str) -> str:
        self._seq += 1
        return f'esc-{self._seq}'

    def submit(self, esc) -> None:
        self.submitted.append(esc)

    def open_it(self):
        """Simulate a prior open L1 (for dedup tests)."""
        self._open_l1 = True


# ---------------------------------------------------------------------------
# step-7 RED / step-8 GREEN: SpeculativeMergeWorker._check_request_liveness
# ---------------------------------------------------------------------------


@pytest.mark.asyncio
class TestCheckRequestLiveness:
    """Unit tests for SpeculativeMergeWorker._check_request_liveness(now).

    RED until step-8 GREEN adds the method to merge_queue.py.
    """

    async def test_stuck_request_warns_and_alarms_once(
        self,
        tmp_path: Path,
        config: OrchestratorConfig,
        git_ops: GitOps,
        caplog: pytest.LogCaptureFixture,
    ) -> None:
        from orchestrator.merge_queue import SpeculativeMergeWorker

        fake_eq = _FakeEscalationQueue(open_l1=False)
        queue: asyncio.Queue = asyncio.Queue()
        worker = SpeculativeMergeWorker(git_ops, queue, escalation_queue=fake_eq)

        wt = tmp_path / 'wt'
        wt.mkdir()
        req = _make_request('stuck-task', 'stuck-task', wt, config)

        t0 = 1_000_000.0
        worker._request_ledger.on_dequeue(req, now=t0)

        with caplog.at_level(logging.WARNING, logger='orchestrator.merge_queue'):
            worker._check_request_liveness(t0 + 2000, threshold_s=1000)

        warnings = [r for r in caplog.records if r.levelno == logging.WARNING]
        assert len(warnings) == 1, f'expected exactly one WARNING, got: {caplog.text}'
        msg = warnings[0].message
        assert req.request_id in msg
        assert req.branch in msg
        assert '2000' in msg

        assert len(fake_eq.submitted) == 1
        esc = fake_eq.submitted[0]
        assert esc.category == 'merge_request_stuck'
        assert req.request_id in esc.summary

    async def test_second_call_is_deduped_by_open_l1(
        self,
        tmp_path: Path,
        config: OrchestratorConfig,
        git_ops: GitOps,
        caplog: pytest.LogCaptureFixture,
    ) -> None:
        from orchestrator.merge_queue import SpeculativeMergeWorker

        fake_eq = _FakeEscalationQueue(open_l1=False)
        queue: asyncio.Queue = asyncio.Queue()
        worker = SpeculativeMergeWorker(git_ops, queue, escalation_queue=fake_eq)

        wt = tmp_path / 'wt'
        wt.mkdir()
        req = _make_request('stuck-task-2', 'stuck-task-2', wt, config)

        t0 = 1_000_000.0
        worker._request_ledger.on_dequeue(req, now=t0)

        worker._check_request_liveness(t0 + 2000, threshold_s=1000)
        assert len(fake_eq.submitted) == 1

        # An open L1 now exists for this request's sentinel (real escalation
        # queues would report has_open_l1 True after the submit above); mirror
        # that with the fake's open_it() and confirm no duplicate is filed.
        fake_eq.open_it()
        worker._check_request_liveness(t0 + 3000, threshold_s=1000)
        assert len(fake_eq.submitted) == 1, 'second call must not submit a duplicate escalation'

    async def test_observation_only_never_resolves_or_halts(
        self,
        tmp_path: Path,
        config: OrchestratorConfig,
        git_ops: GitOps,
    ) -> None:
        from orchestrator.merge_queue import SpeculativeMergeWorker

        fake_eq = _FakeEscalationQueue(open_l1=False)
        queue: asyncio.Queue = asyncio.Queue()
        worker = SpeculativeMergeWorker(git_ops, queue, escalation_queue=fake_eq)

        wt = tmp_path / 'wt'
        wt.mkdir()
        req = _make_request('stuck-task-3', 'stuck-task-3', wt, config)

        t0 = 1_000_000.0
        worker._request_ledger.on_dequeue(req, now=t0)

        worker._check_request_liveness(t0 + 2000, threshold_s=1000)

        assert not req.result.done()
        assert worker._queue.empty()
        assert not worker._operator_halt.is_set()

    async def test_resolved_request_does_not_alarm(
        self,
        tmp_path: Path,
        config: OrchestratorConfig,
        git_ops: GitOps,
        caplog: pytest.LogCaptureFixture,
    ) -> None:
        from orchestrator.merge_queue import SpeculativeMergeWorker

        fake_eq = _FakeEscalationQueue(open_l1=False)
        queue: asyncio.Queue = asyncio.Queue()
        worker = SpeculativeMergeWorker(git_ops, queue, escalation_queue=fake_eq)

        wt = tmp_path / 'wt'
        wt.mkdir()
        req = _make_request('resolved-task', 'resolved-task', wt, config)

        t0 = 1_000_000.0
        worker._request_ledger.on_dequeue(req, now=t0)
        req.result.set_result(MergeOutcome('done'))

        with caplog.at_level(logging.WARNING, logger='orchestrator.merge_queue'):
            worker._check_request_liveness(t0 + 2000, threshold_s=1000)

        assert len([r for r in caplog.records if r.levelno == logging.WARNING]) == 0
        assert len(fake_eq.submitted) == 0
        assert worker._request_ledger.is_empty()  # swept as resolved


# ---------------------------------------------------------------------------
# step-9 RED / step-10 GREEN: heartbeat wiring runs the liveness check first
# ---------------------------------------------------------------------------


@pytest.mark.asyncio
class TestHeartbeatWiringRunsLivenessCheckFirst:
    """_maybe_log_queue_heartbeat must run the liveness check BEFORE its
    depth==0 / rate-limit early-returns (task 1992 step-9).

    RED until step-10 GREEN wires the call at the top of
    _maybe_log_queue_heartbeat.
    """

    async def test_leaked_request_alarms_even_though_heartbeat_stays_idle(
        self,
        tmp_path: Path,
        config: OrchestratorConfig,
        git_ops: GitOps,
        caplog: pytest.LogCaptureFixture,
    ) -> None:
        from orchestrator.merge_queue import SpeculativeMergeWorker

        fake_eq = _FakeEscalationQueue(open_l1=False)
        queue: asyncio.Queue = asyncio.Queue()
        worker = SpeculativeMergeWorker(git_ops, queue, escalation_queue=fake_eq)
        # High interval: this test's single call must not be the thing that
        # rate-limits the depth heartbeat — depth==0 is what must short-circuit it.
        worker._heartbeat_interval_s = 1_000_000.0

        wt = tmp_path / 'wt'
        wt.mkdir()
        req = _make_request('leaked-task', 'leaked-task', wt, config)

        t0 = 1_000_000.0
        worker._request_ledger.on_dequeue(req, now=t0)
        # worker._queue is left EMPTY and the request is never dispatched into
        # _inflight — it is absent from snapshot()['entries'] entirely, so
        # snapshot()['depth'] == 0 (the leaked-request shape: fallen out of
        # every pipeline structure).

        far_future = t0 + 20_000.0  # past the 16200s (1.5x) default stuck threshold

        with caplog.at_level(logging.WARNING, logger='orchestrator.merge_queue'):
            result = worker._maybe_log_queue_heartbeat(far_future)

        assert result is False, 'heartbeat must stay idle — no depth to report (snapshot depth==0)'

        warnings = [r for r in caplog.records if r.levelno == logging.WARNING]
        assert len(warnings) == 1, f'expected exactly one stuck-request WARNING, got: {caplog.text}'
        assert req.request_id in warnings[0].message

        assert len(fake_eq.submitted) == 1
        assert fake_eq.submitted[0].category == 'merge_request_stuck'
