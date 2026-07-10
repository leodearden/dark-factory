"""Tests for SpeculativeMergeWorker request-liveness checks + heartbeat wiring
(MQ-invariants eta / task 1992), and the merge-worktree content-mtime
no-progress budget (task 2420, DEFECT 1 split from 2357; extends #1728).

Steps covered:
  step-7  RED   — _check_request_liveness unit tests (bare worker)
  step-8  GREEN — implement _check_request_liveness
  step-9  RED   — heartbeat-wiring: liveness check runs before depth==0 early-return
  step-10 GREEN — call _check_request_liveness from the top of _maybe_log_queue_heartbeat
  step-11 RED   — end-to-end wedged-verify integration
  step-12 GREEN — wire the arm hook at the merger-loop head
  step-13 RED   — operator-halt requeue no-false-alarm
  step-14 GREEN — wire on_requeued at the 3 put_nowait sites

  task 2420 (DEFECT 1 split from 2357; extends #1728):
  step-3  RED   — dead LOCAL in-flight verify (no content progress) is
                  aborted + re-dispatched within a bounded no-progress budget
  step-4  GREEN — elapsed-only abort trigger 3 (minimal; no local-only gate
                  or progress-reset yet)
  step-5  RED   — healthy LOCAL verify that keeps writing is NOT aborted;
                  REMOTE lease is never progress-aborted (scope fence)
  step-6  GREEN — convert trigger 3 to a LOCAL-only no-PROGRESS budget
  step-7  RED   — repeated dead verify converts to 'blocked' (busy-loop cap);
                  a subsequent success clears the per-task counter
  step-8  GREEN — per-task MAX_INFLIGHT_DEAD_VERIFY_ABORTS busy-loop guard

This module intentionally imports orchestrator.merge_queue LOCALLY inside each
test method (not at module scope) so a not-yet-implemented symbol
(_check_request_liveness, before step-8) never breaks collection of the rest
of the file — mirrors test_merge_request_ledger.py's step-5 convention.
"""

from __future__ import annotations

import asyncio
import contextlib
import logging
import time
from pathlib import Path
from typing import Any
from unittest.mock import AsyncMock, MagicMock, patch

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


async def _make_branch_with_file(
    git_ops: GitOps,
    branch_name: str,
    filename: str,
    content: str,
) -> Path:
    """Create a worktree branch with one committed file and return its path.

    Duplicated from test_merge_queue_concurrent_verify.py / test_merge_queue_resolve_release.py
    (per-file duplication convention — see this file's module docstring).
    """
    worktree = (await git_ops.create_worktree(branch_name)).path
    (worktree / filename).write_text(content)
    await git_ops.commit(worktree, f'Add {filename}')
    return worktree


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

        with caplog.at_level(logging.WARNING, logger='orchestrator.merge_queue'):
            worker._check_request_liveness(t0 + 2000, threshold_s=1000)
            assert len(fake_eq.submitted) == 1

            warnings_after_first = [r for r in caplog.records if r.levelno == logging.WARNING]
            assert len(warnings_after_first) == 1, 'first call must warn exactly once'

            # An open L1 now exists for this request's sentinel (real escalation
            # queues would report has_open_l1 True after the submit above); mirror
            # that with the fake's open_it() and confirm no duplicate is filed.
            fake_eq.open_it()
            worker._check_request_liveness(t0 + 3000, threshold_s=1000)
            assert len(fake_eq.submitted) == 1, 'second call must not submit a duplicate escalation'

            # The WARNING log is dedup'd exactly like the escalation: a second
            # sweep of the SAME still-open (never resolved/requeued) episode
            # must not re-log — otherwise a genuinely leaked/wedged request
            # would spam an identical WARNING on every heartbeat poll forever.
            warnings_after_second = [r for r in caplog.records if r.levelno == logging.WARNING]
            assert len(warnings_after_second) == 1, (
                'second call for the same open stuck episode must not re-warn '
                '— the WARNING log must be dedup\'d per episode like the escalation'
            )

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

    async def test_warning_relogs_after_requeue_and_redequeue_restarts_episode(
        self,
        tmp_path: Path,
        config: OrchestratorConfig,
        git_ops: GitOps,
        caplog: pytest.LogCaptureFixture,
    ) -> None:
        """Amendment (post-verification review, task 1992): the WARNING log
        dedup is per-EPISODE, not permanent — a requeue (on_requeued) followed
        by a later re-dequeue (on_dequeue) must re-warn, mirroring the
        escalation's has_open_l1 contract which likewise only dedups within a
        single open episode.
        """
        from orchestrator.merge_queue import SpeculativeMergeWorker

        fake_eq = _FakeEscalationQueue(open_l1=False)
        queue: asyncio.Queue = asyncio.Queue()
        worker = SpeculativeMergeWorker(git_ops, queue, escalation_queue=fake_eq)

        wt = tmp_path / 'wt'
        wt.mkdir()
        req = _make_request('requeued-episode-task', 'requeued-episode-task', wt, config)

        t0 = 1_000_000.0
        worker._request_ledger.on_dequeue(req, now=t0)

        with caplog.at_level(logging.WARNING, logger='orchestrator.merge_queue'):
            worker._check_request_liveness(t0 + 2000, threshold_s=1000)
        assert len([r for r in caplog.records if r.levelno == logging.WARNING]) == 1

        # Requeue clears the entry (operator halt / cascade); a later
        # re-dequeue re-arms it fresh — a brand-new episode.
        worker._request_ledger.on_requeued(req.request_id)
        t2 = t0 + 10_000.0
        worker._request_ledger.on_dequeue(req, now=t2)

        caplog.clear()
        with caplog.at_level(logging.WARNING, logger='orchestrator.merge_queue'):
            worker._check_request_liveness(t2 + 2000, threshold_s=1000)

        warnings = [r for r in caplog.records if r.levelno == logging.WARNING]
        assert len(warnings) == 1, (
            'a fresh episode after on_requeued + re-on_dequeue must re-warn '
            'once, not stay silently suppressed by the PRIOR episode\'s '
            'warned flag'
        )


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


# ---------------------------------------------------------------------------
# step-11 RED / step-12 GREEN: end-to-end wedged-verify integration
# ---------------------------------------------------------------------------


@pytest.mark.asyncio
class TestWedgedVerifyIntegration:
    """A REAL dequeued MergeRequest wedged in in-flight verify (PRD boundary
    #5) must be armed by the merger-loop dequeue hook and detected by
    ``_check_request_liveness`` while still genuinely owned by an
    ``_inflight`` slot (``state == 'verifying'``) — distinct from the
    leaked/unowned shape covered by ``TestHeartbeatWiringRunsLivenessCheckFirst``
    above.

    RED until step-12 GREEN wires ``self._request_ledger.on_dequeue(...)``
    at the merger-loop head: today a real ``worker.run()`` dequeue never
    arms the ledger, so it stays empty and the liveness check finds nothing.
    """

    async def test_wedged_verify_is_armed_and_alarmed_then_resolves_cleanly(
        self,
        git_ops: GitOps,
        config: OrchestratorConfig,
        caplog: pytest.LogCaptureFixture,
    ) -> None:
        from orchestrator.merge_queue import (
            INFLIGHT_MERGE_WORKTREE_LIVENESS_SECS,
            SpeculativeMergeWorker,
        )

        gate_release = asyncio.Event()
        gate_entered = asyncio.Event()

        async def _gated_local_verify(*args: object, **kwargs: object) -> MagicMock:
            gate_entered.set()
            await gate_release.wait()
            return MagicMock(
                passed=True, summary='ok', test_output='ok',
                lint_output='', type_output='', category='',
                timed_out=False, verify_skipped=False,
            )

        wt = await _make_branch_with_file(
            git_ops, 'task/wedged-verify', 'wedged.py', 'x = 1\n',
        )

        fake_eq = _FakeEscalationQueue(open_l1=False)
        q: asyncio.Queue = asyncio.Queue()
        worker = SpeculativeMergeWorker(git_ops, q, escalation_queue=fake_eq)

        req = _make_request('wedged-verify', 'task/wedged-verify', wt, config)

        with patch('orchestrator.merge_queue.run_scoped_verification', _gated_local_verify):
            worker_task = asyncio.create_task(worker.run())

            try:
                await q.put(req)
                await asyncio.wait_for(gate_entered.wait(), timeout=15.0)
            except TimeoutError:
                gate_release.set()
                with contextlib.suppress(Exception):
                    await asyncio.wait_for(worker.stop(), timeout=5.0)
                with contextlib.suppress(Exception):
                    await asyncio.wait_for(worker_task, timeout=5.0)
                raise

            # The request must be genuinely owned by an in-flight verify slot
            # (boundary #5 — wedged, not leaked) before we probe liveness.
            snap = worker.snapshot()
            matching = [e for e in snap['entries'] if e['request_id'] == req.request_id]
            assert len(matching) == 1 and matching[0]['state'] == 'verifying', (
                f"Expected req in an in-flight 'verifying' entry, got: {snap['entries']!r}"
            )

            # The merger-loop dequeue hook must already have armed the ledger —
            # this is the crux of the RED/GREEN split for step-11/step-12.
            assert req.request_id in worker._request_ledger.open_request_ids(), (
                'merger-loop dequeue hook not wired — ledger never armed for a '
                'real dequeue, so the wedged request is invisible to the '
                'liveness sweep (RED until step-12)'
            )

            threshold_s = 1.5 * INFLIGHT_MERGE_WORKTREE_LIVENESS_SECS
            now = time.time() + threshold_s + 60.0  # comfortably past threshold

            with caplog.at_level(logging.WARNING, logger='orchestrator.merge_queue'):
                worker._check_request_liveness(now)

            warnings = [r for r in caplog.records if r.levelno == logging.WARNING]
            assert len(warnings) == 1, f'expected exactly one WARNING, got: {caplog.text}'
            msg = warnings[0].message
            assert req.request_id in msg
            assert req.branch in msg

            assert len(fake_eq.submitted) == 1
            esc = fake_eq.submitted[0]
            assert esc.category == 'merge_request_stuck'
            assert req.request_id in esc.summary

            # Observation-only: still wedged, nothing mutated or halted.
            assert not req.result.done()
            assert not worker._operator_halt.is_set()

            # ── Release the gate and confirm clean shutdown ────────────────
            gate_release.set()
            outcome = await asyncio.wait_for(req.result, timeout=15.0)
            assert outcome.status == 'done', f'expected clean resolution, got {outcome!r}'

            await worker.stop()

        with contextlib.suppress(Exception):
            await asyncio.wait_for(worker_task, timeout=10.0)

        # Passive resolution: production only sweeps a resolved entry on the
        # NEXT liveness check, never eagerly on resolve — so trigger that
        # sweep explicitly and confirm the ledger is left clean.
        worker._request_ledger.sweep_resolved()
        assert worker._request_ledger.is_empty(), (
            'ledger must be swept empty after the request resolves cleanly'
        )


# ---------------------------------------------------------------------------
# step-13 RED / step-14 GREEN: operator-halt requeue must not false-alarm
# ---------------------------------------------------------------------------


@pytest.mark.asyncio
class TestOperatorHaltRequeueNoFalseAlarm:
    """A requeued request (operator halt / cascade) deliberately leaves its
    Future pending — passive resolution can never see it as done. Without an
    explicit ``on_requeued`` hook, a request parked on ``_queue`` for a long
    halt would eventually false-alarm even though it is not actually stuck.

    Drives the pre-dispatch operator-halt branch of ``_dispatch_item``
    directly (bare worker, no real ``worker.run()`` loop) — one of the 3
    ``put_nowait`` requeue sites; step-14 wires ``on_requeued`` at all 3.

    RED until step-14 GREEN wires ``on_requeued`` at the 3 ``put_nowait`` sites.
    """

    async def test_predispatch_operator_halt_requeue_clears_ledger_and_restarts_clock(
        self,
        tmp_path: Path,
        config: OrchestratorConfig,
        git_ops: GitOps,
    ) -> None:
        from orchestrator.merge_queue import (
            InflightStatus,
            RealMergeItem,
            SpeculativeMergeWorker,
        )

        fake_eq = _FakeEscalationQueue(open_l1=False)
        queue: asyncio.Queue = asyncio.Queue()
        worker = SpeculativeMergeWorker(git_ops, queue, escalation_queue=fake_eq)

        wt = tmp_path / 'wt'
        wt.mkdir()
        req = _make_request('halt-task', 'halt-task', wt, config)

        t0 = 1_000_000.0
        worker._request_ledger.on_dequeue(req, now=t0)

        item = RealMergeItem(
            request=req,
            merge_result=MagicMock(),
            merge_wt=tmp_path / 'merge_wt',
            base_sha='deadbeef',
            speculative=False,
        )

        worker._operator_halt.set()
        entry = await worker._dispatch_item(item)

        assert entry is not None and entry.status == InflightStatus.REQUEUED_PREDISPATCH
        assert not queue.empty(), 'req must be put back on _queue by the halt branch'

        # The stale T0 entry must be gone — a legitimately halted/parked
        # request must never age out while parked.
        assert req.request_id not in worker._request_ledger.open_request_ids(), (
            'requeue hook not wired — stale T0 ledger entry survives the '
            'operator halt (RED until step-14)'
        )

        worker._check_request_liveness(t0 + 100_000.0, threshold_s=1000)
        assert len(fake_eq.submitted) == 0, (
            'parked request must never alarm — its ledger entry should be gone'
        )

        # Re-dequeue: the NEXT on_dequeue must re-arm with a FRESH dequeued_at.
        t2 = t0 + 50_000.0
        worker._request_ledger.on_dequeue(req, now=t2)

        # No alarm shortly after re-dequeue — the age clock restarted from T2,
        # NOT the stale T0 (which would already be ~50000s old at t2+10, and
        # thus WOULD exceed threshold_s=1000 if on_dequeue had kept the stale
        # T0 timestamp instead of re-arming fresh).
        worker._check_request_liveness(t2 + 10.0, threshold_s=1000)
        assert len(fake_eq.submitted) == 0, (
            'age clock must restart from T2 on re-dequeue, not resume from the '
            'stale T0 — a bug here would immediately alarm (RED until step-14)'
        )

        # Sanity: the re-armed entry genuinely is back in the ledger (not
        # silently dropped by the requeue/re-dequeue sequence).
        assert req.request_id in worker._request_ledger.open_request_ids()


# ---------------------------------------------------------------------------
# task 2420 step-3 RED / step-4 GREEN: dead LOCAL in-flight verify (no content
# progress) is aborted + re-dispatched within a bounded no-progress budget
# ---------------------------------------------------------------------------


async def _make_merged_item(
    git_ops: GitOps,
    config: OrchestratorConfig,
    branch: str,
    filename: str,
    content: str,
    *,
    task_id: str | None = None,
):
    """Create a merged RealMergeItem on a fresh branch (real git, no mocks).

    *task_id* defaults to *branch* (existing call sites); pass it explicitly
    to drive multiple separate merged items against the SAME task_id (e.g.
    to accumulate a per-task counter across repeated dispatch attempts).

    Duplicated from test_merge_queue_concurrent_verify.py's
    TestRunInflightVerifyAbortPoll._make_merged_item (per-file duplication
    convention — see this file's module docstring).
    """
    from orchestrator.merge_queue import RealMergeItem

    wt = await _make_branch_with_file(git_ops, branch, filename, content)
    loop = asyncio.get_running_loop()
    req = MergeRequest(
        task_id=task_id if task_id is not None else branch,
        branch=branch,
        worktree=wt,
        pre_rebased=False,
        task_files=None,
        module_configs=[],
        config=config,
        result=loop.create_future(),
        lane='normal',
    )
    merge_result = await git_ops.merge_to_main(wt, branch)
    assert merge_result.success and merge_result.merge_commit
    assert merge_result.merge_worktree is not None
    base_sha = await git_ops.get_main_sha()
    item = RealMergeItem(
        request=req,
        merge_result=merge_result,
        merge_wt=merge_result.merge_worktree,
        base_sha=base_sha,
        speculative=False,
    )
    return req, item


def _pass_result() -> MagicMock:
    """Return a MagicMock verify-pass result.

    Mirrors test_merge_queue_concurrent_verify.py's _mock_verify_pass()
    (per-file duplication convention) — LocalRunner.run_merge_verify only
    reads .passed off the scoped result on the pass path, so a bare
    MagicMock(passed=True, ...) is sufficient (proven by the existing
    TestRunInflightVerifyHappyPath local-lease tests in that file).
    """
    return MagicMock(passed=True, summary='')


@pytest.mark.asyncio
class TestDeadInflightVerifyAborts:
    """SpeculativeMergeWorker._run_inflight_verify LOCAL no-progress abort
    trigger (task 2420 DEFECT 1, split from 2357; extends #1728).

    The #1728 alpha owner-heartbeat keeps a LIVE worker's merge worktree
    ROOT mtime fresh even while its verify subprocess is dead/hung, so the
    existing 3h root-mtime reaper never fires for this failure mode. This
    class drives _run_inflight_verify directly (bare worker, no run() loop)
    against a REAL local merge worktree with a gated fake
    run_scoped_verification that never writes under merge_wt — simulating a
    dead/hung verify with zero content progress.

    RED until step-4 GREEN adds abort trigger 3 (elapsed-only to start).

    step-5 RED / step-6 GREEN adds two guards that step-4's elapsed-only,
    lease-agnostic trigger fails: a healthy LOCAL verify that keeps writing
    under merge_wt must NOT be aborted (progress resets the clock), and a
    REMOTE lease must NEVER be progress-aborted (scope fence — remote
    verify-hang is owned by task 2362's ssh keepalive).
    """

    async def test_dead_local_verify_is_aborted_and_requeued_within_budget(
        self,
        git_ops: GitOps,
        config: OrchestratorConfig,
        caplog: pytest.LogCaptureFixture,
    ) -> None:
        from orchestrator.merge_queue import InflightStatus, SpeculativeMergeWorker
        from orchestrator.verify_runner import HostLease

        gate_entered = asyncio.Event()
        never_release = asyncio.Event()

        async def _dead_gate(*args: object, **kwargs: object) -> object:
            # Simulates a dead/hung verify subprocess: never returns, and
            # (crucially) never writes anything under merge_wt — zero
            # content progress for the no-progress budget to observe.
            gate_entered.set()
            await never_release.wait()
            raise AssertionError('unreachable — never_release is never set in this test')

        req, item = await _make_merged_item(
            git_ops, config, 'dead-verify-a', 'da.py', 'x=1\n',
        )
        q: asyncio.Queue[MergeRequest] = asyncio.Queue()
        worker = SpeculativeMergeWorker(git_ops, q)
        worker._register_owned_merge_worktree(item.merge_wt)

        # Fast, deterministic tunables (VERIFY_ABANDON_POLL_SECS convention).
        worker.VERIFY_ABANDON_POLL_SECS = 0.02
        worker.INFLIGHT_VERIFY_PROGRESS_PROBE_SECS = 0.02
        worker.INFLIGHT_VERIFY_PROGRESS_BUDGET_SECS = 0.2

        fake_local = MagicMock()
        fake_local.name = 'local'
        fake_local.is_local = True
        lease = HostLease(name='local', runner=fake_local, is_local=True)

        t0 = 1_000_000.0
        worker._request_ledger.on_dequeue(req, now=t0)

        with (
            patch('orchestrator.merge_queue.run_scoped_verification', _dead_gate),
            caplog.at_level(logging.WARNING, logger='orchestrator.merge_queue'),
        ):
            result = await asyncio.wait_for(
                worker._run_inflight_verify(item, lease), timeout=5.0,
            )

        assert result.status == InflightStatus.REQUEUED, (
            f'expected a dead-verify abort to REQUEUE, got status={result.status!r}'
        )
        assert not q.empty(), 'dead verify must re-dispatch the request onto _queue'
        assert result.merge_wt is None, 'merge_wt must be cleaned on dead-verify abort'
        assert req.request_id not in worker._request_ledger.open_request_ids(), (
            'on_requeued must clear the ledger entry so the parked request never ages out'
        )

        warnings = [r for r in caplog.records if r.levelno == logging.WARNING]
        assert any(
            req.task_id in r.message and 'progress' in r.message.lower()
            for r in warnings
        ), f'expected a WARNING naming the task + no-progress budget, got: {caplog.text}'

    async def test_healthy_writing_local_verify_is_not_aborted(
        self,
        git_ops: GitOps,
        config: OrchestratorConfig,
    ) -> None:
        """A LOCAL verify that keeps writing under merge_wt must never be
        progress-aborted — content progress resets the no-progress clock.

        RED (step-5) until step-6 GREEN converts trigger 3 from elapsed-only
        to a genuine no-PROGRESS budget: step-4's elapsed-only trigger aborts
        this healthy, actively-writing verify exactly like a dead one, since
        it never looks at worktree content at all.
        """
        from orchestrator.merge_queue import SpeculativeMergeWorker
        from orchestrator.verify_runner import HostLease

        release_event = asyncio.Event()

        req, item = await _make_merged_item(
            git_ops, config, 'healthy-verify-a', 'ha.py', 'x=1\n',
        )
        q: asyncio.Queue[MergeRequest] = asyncio.Queue()
        worker = SpeculativeMergeWorker(git_ops, q)
        worker._register_owned_merge_worktree(item.merge_wt)

        worker.VERIFY_ABANDON_POLL_SECS = 0.02
        worker.INFLIGHT_VERIFY_PROGRESS_PROBE_SECS = 0.02
        worker.INFLIGHT_VERIFY_PROGRESS_BUDGET_SECS = 0.2

        async def _healthy_writing_gate(*args: Any, **kwargs: object) -> MagicMock:
            merge_wt_arg = Path(args[0])
            target = merge_wt_arg / 'target'
            target.mkdir(exist_ok=True)
            i = 0
            # Keep writing fresh content well past several budget windows so
            # a working no-progress budget's clock is repeatedly reset.
            while not release_event.is_set():
                (target / f'{i}.tmp').write_text('progress')
                i += 1
                await asyncio.sleep(worker.INFLIGHT_VERIFY_PROGRESS_PROBE_SECS)
            return _pass_result()

        fake_local = MagicMock()
        fake_local.name = 'local'
        fake_local.is_local = True
        lease = HostLease(name='local', runner=fake_local, is_local=True)

        with patch('orchestrator.merge_queue.run_scoped_verification', _healthy_writing_gate):
            verify_future = asyncio.ensure_future(worker._run_inflight_verify(item, lease))
            # Let several budget windows elapse while content keeps writing.
            await asyncio.sleep(worker.INFLIGHT_VERIFY_PROGRESS_BUDGET_SECS * 4)
            release_event.set()
            result = await asyncio.wait_for(verify_future, timeout=5.0)

        assert result.status is None, (
            f'a healthy, progressing local verify must NOT be progress-aborted; '
            f'got status={result.status!r}'
        )
        assert q.empty(), 'healthy verify must not be re-dispatched'

    async def test_remote_lease_is_never_progress_aborted(
        self,
        git_ops: GitOps,
        config: OrchestratorConfig,
    ) -> None:
        """A REMOTE lease must never be progress-aborted (scope fence): the
        remote verify-hang facet is owned by task 2362's ssh keepalive, and a
        remote verify writes to the REMOTE host's worktree, not the local
        merge_wt — so a local content-mtime budget would false-abort a
        healthy remote verify.

        RED (step-5) until step-6 GREEN gates trigger 3 on lease.is_local:
        step-4's elapsed-only trigger is lease-agnostic and aborts this dead
        REMOTE verify exactly like a LOCAL one.
        """
        from orchestrator.merge_queue import SpeculativeMergeWorker
        from orchestrator.verify_runner import HostLease

        never_release = asyncio.Event()
        gate_entered = asyncio.Event()

        async def _dead_remote_verify(*args: object, **kwargs: object) -> MagicMock:
            gate_entered.set()
            await never_release.wait()
            return _pass_result()  # pragma: no cover — never reached in this test

        req, item = await _make_merged_item(
            git_ops, config, 'remote-dead-verify-a', 'ra.py', 'x=1\n',
        )
        q: asyncio.Queue[MergeRequest] = asyncio.Queue()
        worker = SpeculativeMergeWorker(git_ops, q)
        worker._register_owned_merge_worktree(item.merge_wt)

        worker.VERIFY_ABANDON_POLL_SECS = 0.02
        worker.INFLIGHT_VERIFY_PROGRESS_PROBE_SECS = 0.02
        worker.INFLIGHT_VERIFY_PROGRESS_BUDGET_SECS = 0.2

        fake_remote = MagicMock()
        fake_remote.name = 'remote-host'
        fake_remote.is_local = False
        fake_remote.run_merge_verify = AsyncMock(side_effect=_dead_remote_verify)
        fake_remote.cancel_verify = AsyncMock(return_value=0)
        lease = HostLease(name='remote-host', runner=fake_remote, is_local=False)

        verify_future = asyncio.ensure_future(worker._run_inflight_verify(item, lease))
        await asyncio.wait_for(gate_entered.wait(), timeout=15.0)

        # Wall-clock comfortably exceeds the (tiny) budget several times over
        # — a LOCAL lease would already have been progress-aborted by now.
        await asyncio.sleep(worker.INFLIGHT_VERIFY_PROGRESS_BUDGET_SECS * 5)

        assert not verify_future.done(), (
            'a REMOTE lease must never be progress-aborted (scope fence: remote '
            'verify-hang is owned by task 2362 ssh keepalive)'
        )
        assert q.empty(), 'remote lease must not be re-dispatched by the progress budget'

        verify_future.cancel()
        with contextlib.suppress(BaseException):
            await verify_future


# ---------------------------------------------------------------------------
# task 2420 step-7 RED / step-8 GREEN: busy-loop cap converts a repeated dead
# LOCAL verify to 'blocked'; a successful verify clears the per-task counter
# ---------------------------------------------------------------------------


async def _dead_gate_never_returns(*args: object, **kwargs: object) -> MagicMock:
    """Simulates a dead/hung LOCAL verify: never returns, never writes.

    Shared by TestRepeatedDeadVerifyBusyLoopCap's multiple dead-verify
    attempts (per-file duplication convention).
    """
    await asyncio.Event().wait()
    raise AssertionError('unreachable — this Event is never set')  # pragma: no cover


@pytest.mark.asyncio
class TestRepeatedDeadVerifyBusyLoopCap:
    """A deterministically-hanging LOCAL verify must not busy-loop the queue
    forever: after MAX_INFLIGHT_DEAD_VERIFY_ABORTS consecutive dead aborts
    for the SAME task, the request resolves terminally as 'blocked' instead
    of being re-queued again — surfacing the infra failure loudly rather
    than silently churning a slot. A subsequent SUCCESSFUL verify for that
    task clears the counter, so a later transient hang starts a fresh count.

    RED until step-8 GREEN adds the per-task _inflight_dead_verify_aborts
    counter — step-6 re-queues an arbitrary number of consecutive dead
    verifies unconditionally.
    """

    async def test_repeated_dead_verify_converts_to_blocked_and_success_resets_counter(
        self,
        git_ops: GitOps,
        config: OrchestratorConfig,
    ) -> None:
        from orchestrator.merge_queue import InflightStatus, SpeculativeMergeWorker
        from orchestrator.verify_runner import HostLease

        task_id = 'dead-repeat-task'

        q: asyncio.Queue[MergeRequest] = asyncio.Queue()
        worker = SpeculativeMergeWorker(git_ops, q)
        worker.VERIFY_ABANDON_POLL_SECS = 0.02
        worker.INFLIGHT_VERIFY_PROGRESS_PROBE_SECS = 0.02
        worker.INFLIGHT_VERIFY_PROGRESS_BUDGET_SECS = 0.2
        worker.MAX_INFLIGHT_DEAD_VERIFY_ABORTS = 2

        fake_local = MagicMock()
        fake_local.name = 'local'
        fake_local.is_local = True

        # ── Attempt 1: dead verify → REQUEUED (counter -> 1, below MAX) ──
        req1, item1 = await _make_merged_item(
            git_ops, config, 'dead-repeat-branch-1', 'dr1.py', 'a=1\n', task_id=task_id,
        )
        worker._register_owned_merge_worktree(item1.merge_wt)
        lease1 = HostLease(name='local', runner=fake_local, is_local=True)
        with patch('orchestrator.merge_queue.run_scoped_verification', _dead_gate_never_returns):
            result1 = await asyncio.wait_for(
                worker._run_inflight_verify(item1, lease1), timeout=5.0,
            )
        assert result1.status == InflightStatus.REQUEUED
        assert not req1.result.done()
        # Drain the re-queued req1 so it doesn't shadow the emptiness checks
        # below — a real merger loop would dequeue it well before the next
        # dispatch attempt lands.
        assert q.get_nowait() is req1

        # ── Attempt 2 (MAX-th, same task_id): dead verify → terminal 'blocked' ──
        req2, item2 = await _make_merged_item(
            git_ops, config, 'dead-repeat-branch-2', 'dr2.py', 'b=2\n', task_id=task_id,
        )
        worker._register_owned_merge_worktree(item2.merge_wt)
        lease2 = HostLease(name='local', runner=fake_local, is_local=True)
        with patch('orchestrator.merge_queue.run_scoped_verification', _dead_gate_never_returns):
            result2 = await asyncio.wait_for(
                worker._run_inflight_verify(item2, lease2), timeout=5.0,
            )

        assert result2.status is None, (
            f'the MAX-th consecutive dead abort must resolve terminally '
            f'(mirroring the except-Exception blocked path), not REQUEUED '
            f'again — got status={result2.status!r}'
        )
        assert result2.outcome is not None and result2.outcome.status == 'blocked'
        reason = result2.outcome.reason.lower()
        assert 'dead' in reason and 'hung' in reason, (
            f"expected the blocked reason to mention 'dead'/'hung' verify, got: "
            f'{result2.outcome.reason!r}'
        )
        assert result2.merge_wt is None

        assert req2.result.done(), 'the MAX-th abort must resolve req2.result directly'
        outcome2 = req2.result.result()
        assert outcome2.status == 'blocked'

        assert q.empty(), 'the MAX-th dead abort must NOT be re-queued (busy-loop guard)'

        # ── Attempt 3: a SUCCESSFUL verify for the SAME task_id clears the counter ──
        req3, item3 = await _make_merged_item(
            git_ops, config, 'dead-repeat-branch-3', 'dr3.py', 'c=3\n', task_id=task_id,
        )
        worker._register_owned_merge_worktree(item3.merge_wt)
        lease3 = HostLease(name='local', runner=fake_local, is_local=True)
        with patch(
            'orchestrator.merge_queue.run_scoped_verification',
            AsyncMock(return_value=_pass_result()),
        ):
            result3 = await asyncio.wait_for(
                worker._run_inflight_verify(item3, lease3), timeout=5.0,
            )
        assert result3.status is None and result3.outcome is None, (
            f'expected a clean pass, got {result3!r}'
        )
        assert worker._inflight_dead_verify_aborts.get(task_id, 0) == 0, (
            'a successful verify must clear the per-task dead-verify-abort counter'
        )

        # ── Attempt 4: dead verify again for the SAME task_id starts a FRESH episode ──
        req4, item4 = await _make_merged_item(
            git_ops, config, 'dead-repeat-branch-4', 'dr4.py', 'd=4\n', task_id=task_id,
        )
        worker._register_owned_merge_worktree(item4.merge_wt)
        lease4 = HostLease(name='local', runner=fake_local, is_local=True)
        with patch('orchestrator.merge_queue.run_scoped_verification', _dead_gate_never_returns):
            result4 = await asyncio.wait_for(
                worker._run_inflight_verify(item4, lease4), timeout=5.0,
            )
        assert result4.status == InflightStatus.REQUEUED, (
            'after a successful verify clears the counter, the next dead verify '
            'must start a fresh episode (REQUEUED), not immediately resolve blocked'
        )

    async def test_failed_verify_also_clears_dead_verify_abort_counter(
        self,
        git_ops: GitOps,
        config: OrchestratorConfig,
    ) -> None:
        """task 2420 amend (reviewer finding #1): a completed-but-FAILED
        verify (a real, non-hung failure) proves the subprocess was not
        hung, so it must clear the per-task dead-verify-abort counter just
        like a pass — not only on the `out is None` pass path. Otherwise a
        hang -> real-failure -> hang sequence would silently accumulate two
        dead-abort counts toward MAX_INFLIGHT_DEAD_VERIFY_ABORTS even though
        a verify genuinely ran to completion (and failed for real reasons)
        in between.
        """
        from orchestrator.merge_queue import InflightStatus, SpeculativeMergeWorker
        from orchestrator.verify_runner import HostLease

        task_id = 'dead-then-real-failure-task'

        q: asyncio.Queue[MergeRequest] = asyncio.Queue()
        worker = SpeculativeMergeWorker(git_ops, q)
        worker.VERIFY_ABANDON_POLL_SECS = 0.02
        worker.INFLIGHT_VERIFY_PROGRESS_PROBE_SECS = 0.02
        worker.INFLIGHT_VERIFY_PROGRESS_BUDGET_SECS = 0.2
        worker.MAX_INFLIGHT_DEAD_VERIFY_ABORTS = 2

        fake_local = MagicMock()
        fake_local.name = 'local'
        fake_local.is_local = True

        # ── Attempt 1: dead verify -> REQUEUED (counter -> 1, below MAX) ──
        req1, item1 = await _make_merged_item(
            git_ops, config, 'dead-then-fail-branch-1', 'dtf1.py', 'a=1\n', task_id=task_id,
        )
        worker._register_owned_merge_worktree(item1.merge_wt)
        lease1 = HostLease(name='local', runner=fake_local, is_local=True)
        with patch('orchestrator.merge_queue.run_scoped_verification', _dead_gate_never_returns):
            result1 = await asyncio.wait_for(
                worker._run_inflight_verify(item1, lease1), timeout=5.0,
            )
        assert result1.status == InflightStatus.REQUEUED
        assert worker._inflight_dead_verify_aborts.get(task_id, 0) == 1

        # ── Attempt 2: a genuine (non-hung) FAILED verify completes
        # promptly. Patches `_run_post_merge_verify` directly (not
        # `run_scoped_verification`) so this test doesn't need to fabricate
        # a scoped VerifyResult that satisfies every internal gate inside
        # `_run_post_merge_verify` (flock-contention/unscoped-gate/ENOSPC/
        # main-health-probe) — it only needs a completed, non-hung failure
        # outcome to reach _run_inflight_verify's fail branch. ──
        async def _fail_fast(*args: object, **kwargs: object) -> MergeOutcome:
            return MergeOutcome('blocked', reason='synthetic real verify failure (not a hang)')

        req2, item2 = await _make_merged_item(
            git_ops, config, 'dead-then-fail-branch-2', 'dtf2.py', 'b=2\n', task_id=task_id,
        )
        worker._register_owned_merge_worktree(item2.merge_wt)
        lease2 = HostLease(name='local', runner=fake_local, is_local=True)
        with patch('orchestrator.merge_queue._run_post_merge_verify', _fail_fast):
            result2 = await asyncio.wait_for(
                worker._run_inflight_verify(item2, lease2), timeout=5.0,
            )
        assert result2.status is None, (
            f'a real (non-hung) failure must resolve via the normal fail '
            f'path, not the busy-loop-capped path — got status={result2.status!r}'
        )
        assert result2.outcome is not None and result2.outcome.status == 'blocked'
        assert worker._inflight_dead_verify_aborts.get(task_id, 0) == 0, (
            'a completed (even failed) verify proves the subprocess was not '
            'hung -- it must clear the counter just like a pass'
        )

        # ── Attempt 3: dead verify again for the SAME task_id starts a
        # FRESH episode (REQUEUED), not resolved as a 2nd consecutive dead
        # abort inherited from attempt 1. ──
        req3, item3 = await _make_merged_item(
            git_ops, config, 'dead-then-fail-branch-3', 'dtf3.py', 'c=3\n', task_id=task_id,
        )
        worker._register_owned_merge_worktree(item3.merge_wt)
        lease3 = HostLease(name='local', runner=fake_local, is_local=True)
        with patch('orchestrator.merge_queue.run_scoped_verification', _dead_gate_never_returns):
            result3 = await asyncio.wait_for(
                worker._run_inflight_verify(item3, lease3), timeout=5.0,
            )
        assert result3.status == InflightStatus.REQUEUED, (
            'after the intervening real failure cleared the counter, this '
            'dead verify must start a fresh episode (REQUEUED), not resolve '
            'blocked as if it were the 2nd consecutive dead abort'
        )

    async def test_blocked_terminal_path_clears_counter_for_fresh_resubmission(
        self,
        git_ops: GitOps,
        config: OrchestratorConfig,
    ) -> None:
        """task 2420 amend (reviewer finding #2): the busy-loop-capped
        'blocked' terminal path must also clear the per-task counter — not
        only a later successful verify. Otherwise a task_id that gets
        re-submitted right after its (human-resolved) 'blocked' outcome,
        with no intervening successful verify, would immediately re-trip
        the cap on its very first fresh dead-verify abort — denying it a
        new set of retry attempts.
        """
        from orchestrator.merge_queue import InflightStatus, SpeculativeMergeWorker
        from orchestrator.verify_runner import HostLease

        task_id = 'dead-repeat-resubmit-task'

        q: asyncio.Queue[MergeRequest] = asyncio.Queue()
        worker = SpeculativeMergeWorker(git_ops, q)
        worker.VERIFY_ABANDON_POLL_SECS = 0.02
        worker.INFLIGHT_VERIFY_PROGRESS_PROBE_SECS = 0.02
        worker.INFLIGHT_VERIFY_PROGRESS_BUDGET_SECS = 0.2
        worker.MAX_INFLIGHT_DEAD_VERIFY_ABORTS = 2

        fake_local = MagicMock()
        fake_local.name = 'local'
        fake_local.is_local = True

        # ── Attempt 1: dead verify -> REQUEUED (counter -> 1, below MAX) ──
        req1, item1 = await _make_merged_item(
            git_ops, config, 'dead-resubmit-branch-1', 'drs1.py', 'a=1\n', task_id=task_id,
        )
        worker._register_owned_merge_worktree(item1.merge_wt)
        lease1 = HostLease(name='local', runner=fake_local, is_local=True)
        with patch('orchestrator.merge_queue.run_scoped_verification', _dead_gate_never_returns):
            result1 = await asyncio.wait_for(
                worker._run_inflight_verify(item1, lease1), timeout=5.0,
            )
        assert result1.status == InflightStatus.REQUEUED

        # ── Attempt 2 (MAX-th, same task_id): dead verify -> terminal 'blocked' ──
        req2, item2 = await _make_merged_item(
            git_ops, config, 'dead-resubmit-branch-2', 'drs2.py', 'b=2\n', task_id=task_id,
        )
        worker._register_owned_merge_worktree(item2.merge_wt)
        lease2 = HostLease(name='local', runner=fake_local, is_local=True)
        with patch('orchestrator.merge_queue.run_scoped_verification', _dead_gate_never_returns):
            result2 = await asyncio.wait_for(
                worker._run_inflight_verify(item2, lease2), timeout=5.0,
            )
        assert result2.outcome is not None and result2.outcome.status == 'blocked'
        assert worker._inflight_dead_verify_aborts.get(task_id, 0) == 0, (
            'the terminal blocked path must pop the counter immediately — a '
            'resubmission of this task_id must not inherit the capped count'
        )

        # ── Attempt 3: SAME task_id, resubmitted immediately (no intervening
        # success) — must start a FRESH episode (REQUEUED), not re-trip the
        # cap on its very first dead abort. ──
        req3, item3 = await _make_merged_item(
            git_ops, config, 'dead-resubmit-branch-3', 'drs3.py', 'c=3\n', task_id=task_id,
        )
        worker._register_owned_merge_worktree(item3.merge_wt)
        lease3 = HostLease(name='local', runner=fake_local, is_local=True)
        with patch('orchestrator.merge_queue.run_scoped_verification', _dead_gate_never_returns):
            result3 = await asyncio.wait_for(
                worker._run_inflight_verify(item3, lease3), timeout=5.0,
            )
        assert result3.status == InflightStatus.REQUEUED, (
            'a task_id resubmitted right after its blocked resolution must '
            'get a fresh dead-verify-abort budget, not immediately re-block'
        )
