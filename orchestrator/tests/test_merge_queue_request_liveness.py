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
import fcntl
import logging
import os
import re
import time
from pathlib import Path
from typing import Any
from unittest.mock import AsyncMock, MagicMock, patch

import pytest

from orchestrator.config import GitConfig, OrchestratorConfig
from orchestrator.git_ops import GitOps, _run
from orchestrator.merge_types import MergeOutcome, MergeRequest, QueuedBranch

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


# ── Warm-lane variants (task 3003, pre-1) ──────────────────────────────────
# The plain `git_config` above omits `persistent_merge_worktree`, which
# defaults to False (config.py:1483).  With the knob off,
# `_acquire_warm_verify_worktree` returns `merge_wt` unchanged
# (merge_liveness.py:712-713) and `reset_persistent_merge_worktree` is never
# reached — the warm-swap reset seam would be untestable from this file.  The
# `warm_*` trio below is byte-identical to the plain trio except for that one
# knob, so a test can drive the LOCAL serial-head warm path for real (the
# reset-contention seam this task fixes) while every pre-existing test keeps
# the cold ephemeral path unchanged.


@pytest.fixture
def warm_git_config() -> GitConfig:
    """`git_config` with the persistent warm merge-verify worktree ENABLED."""
    return GitConfig(
        main_branch='main',
        branch_prefix='task/',
        remote='origin',
        worktree_dir='.worktrees',
        push_after_advance=False,
        persistent_merge_worktree=True,
    )


@pytest.fixture
def warm_git_ops(warm_git_config: GitConfig, git_repo: Path) -> GitOps:
    return GitOps(warm_git_config, git_repo)


@pytest.fixture
def warm_config(git_repo: Path, warm_git_config: GitConfig) -> OrchestratorConfig:
    """Single-host OrchestratorConfig with the warm merge-verify lane ON."""
    return OrchestratorConfig(project_root=git_repo, git=warm_git_config)


def _make_request(
    task_id: str,
    branch: str,
    worktree: Path,
    config: OrchestratorConfig,
) -> MergeRequest:
    """Build a MergeRequest with a fresh Future for the running event loop."""
    return MergeRequest(
        task_id=task_id,
        branch=QueuedBranch.parse(branch, config.git.branch_prefix),
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
        assert req.branch.bare_id in msg
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
            assert req.branch.bare_id in msg

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
        branch=QueuedBranch.parse(branch, config.git.branch_prefix),
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

    task 2921 (load-robustness): the sub-1s budget tunables set by every
    test below (VERIFY_ABANDON_POLL_SECS=0.02,
    INFLIGHT_VERIFY_PROGRESS_PROBE_SECS=0.02,
    INFLIGHT_VERIFY_PROGRESS_BUDGET_SECS=0.2) stay load-safe for the
    must-ABORT (positive) tests: host load only DELAYS a bounded, expected
    abort, it never turns a genuinely dead/coasting verify into a false
    non-abort. The must-NOT-abort test instead removes the real-time
    dependency entirely via deterministic injection rather than relying on
    a widened budget (see test_healthy_writing_local_verify_is_not_aborted's
    newest_content_mtime patch). The one residual full-storm risk is a >5s
    cumulative event-loop stall during a single _run_inflight_verify call
    exceeding the outer asyncio.wait_for — every must-abort call site's
    timeout below is widened 5.0 -> 15.0 to guard against that (still
    comfortably under the 60s pytest default timeout).
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
                worker._run_inflight_verify(item, lease), timeout=15.0,
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

        task 2921: the verify gate itself is now a plain release-event wait
        (no real writer coroutine racing the budget under host load) — the
        progress signal this test relies on instead is a deterministic
        `newest_content_mtime` injection (patched alongside this gate), so
        the no-progress clock resets on every LOCAL probe regardless of
        real-time event-loop scheduling delay under a loaded host.
        newest_content_mtime's real-FS walk behaviour is independently
        covered by test_merge_liveness.py.
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

        async def _gate(*args: object, **kwargs: object) -> MagicMock:
            await release_event.wait()
            return _pass_result()

        # task 2921: strictly-increasing stub so EVERY LOCAL content-mtime
        # probe observes fresh progress and resets _last_progress_at —
        # decouples this must-NOT-abort assertion from real wall-clock
        # file-write timing (which a busy host can starve past the budget).
        _mtime = [1000.0]

        def _always_progress(_root: Path) -> float:
            _mtime[0] += 1.0
            return _mtime[0]

        fake_local = MagicMock()
        fake_local.name = 'local'
        fake_local.is_local = True
        lease = HostLease(name='local', runner=fake_local, is_local=True)

        with (
            patch('orchestrator.merge_queue.run_scoped_verification', _gate),
            patch('orchestrator.merge_queue.newest_content_mtime', _always_progress),
        ):
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

    async def test_remote_lease_held_with_no_live_dispatch_is_aborted_and_requeued_within_budget(
        self,
        git_ops: GitOps,
        config: OrchestratorConfig,
        caplog: pytest.LogCaptureFixture,
    ) -> None:
        """task 2566: a REMOTE lease coasting with NO live ssh dispatch
        (dispatch_in_flight=False) must be progress-aborted + re-dispatched
        within budget — this is the latent gap task 2566 closes (the 7200s
        cold-timeout coast: nothing watches a remote-lease-held window once
        the ssh child has exited).  The ssh dispatch itself stays owned by
        task 2362's keepalive and is never touched here (see
        test_remote_lease_with_live_dispatch_is_not_progress_aborted below).
        run_merge_verify hanging simulates the lease-held stall (e.g. the
        ssh child already exited, or a post-failure local probe hangs) while
        dispatch_in_flight is driven independently on the fake — mirroring
        how the real RemoteRunner derives it from _inflight_request_id, not
        from whether the awaited coroutine has itself returned.

        RED until step-4 GREEN un-gates trigger 3 for remote leases: today
        trigger 3 is `if lease.is_local:`-gated and never runs for a remote
        lease, so this coast is never aborted.
        """
        from orchestrator.merge_queue import InflightStatus, SpeculativeMergeWorker
        from orchestrator.verify_runner import HostLease

        never_release = asyncio.Event()
        gate_entered = asyncio.Event()

        async def _dead_remote_verify(*args: object, **kwargs: object) -> MagicMock:
            gate_entered.set()
            await never_release.wait()
            raise AssertionError('unreachable — never_release is never set in this test')

        req, item = await _make_merged_item(
            git_ops, config, 'remote-coast-verify-a', 'rca.py', 'x=1\n',
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
        fake_remote.dispatch_in_flight = False
        fake_remote.run_merge_verify = AsyncMock(side_effect=_dead_remote_verify)
        fake_remote.cancel_verify = AsyncMock(return_value=0)
        lease = HostLease(name='remote-host', runner=fake_remote, is_local=False)

        with caplog.at_level(logging.WARNING, logger='orchestrator.merge_queue'):
            result = await asyncio.wait_for(
                worker._run_inflight_verify(item, lease), timeout=15.0,
            )

        assert result.status == InflightStatus.REQUEUED, (
            f'expected a remote lease-held coast (no live dispatch) to '
            f'REQUEUE, got status={result.status!r}'
        )
        assert not q.empty(), 'a lease-held coast must re-dispatch the request onto _queue'
        assert result.merge_wt is None, 'merge_wt must be cleaned on a lease-held coast abort'

        warnings = [r for r in caplog.records if r.levelno == logging.WARNING]
        assert any(
            req.task_id in r.message and 'progress' in r.message.lower()
            for r in warnings
        ), f'expected a WARNING naming the task + no-progress budget, got: {caplog.text}'

    async def test_remote_lease_with_live_dispatch_is_not_progress_aborted(
        self,
        git_ops: GitOps,
        config: OrchestratorConfig,
    ) -> None:
        """A REMOTE lease with a LIVE ssh dispatch (dispatch_in_flight=True)
        must never be progress-aborted: the remote verify-hang facet of a
        live dispatch is owned by task 2362's ssh keepalive, and a remote
        verify writes to the REMOTE host's worktree, not the local
        merge_wt — so a local content-mtime budget would false-abort a
        healthy remote verify.  Trigger 3 must treat a live dispatch as
        progress and keep resetting its no-progress clock for as long as
        dispatch_in_flight stays True.

        task 2566 revises this from the former BLANKET invariant "a REMOTE
        lease is NEVER progress-aborted" to this narrower one scoped to a
        LIVE dispatch — the blanket exemption was exactly the coast gap
        closed by test_remote_lease_held_with_no_live_dispatch_is_aborted_and_requeued_within_budget
        above (dispatch_in_flight=False case). This guard pins the
        still-required protection: a live dispatch is progress and must not
        be aborted out from under it.
        """
        from orchestrator.merge_queue import SpeculativeMergeWorker
        from orchestrator.verify_runner import HostLease

        never_release = asyncio.Event()
        gate_entered = asyncio.Event()

        async def _live_dispatch_remote_verify(*args: object, **kwargs: object) -> MagicMock:
            gate_entered.set()
            await never_release.wait()
            return _pass_result()  # pragma: no cover — never reached in this test

        req, item = await _make_merged_item(
            git_ops, config, 'remote-live-dispatch-a', 'rla.py', 'x=1\n',
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
        fake_remote.dispatch_in_flight = True
        fake_remote.run_merge_verify = AsyncMock(side_effect=_live_dispatch_remote_verify)
        fake_remote.cancel_verify = AsyncMock(return_value=0)
        lease = HostLease(name='remote-host', runner=fake_remote, is_local=False)

        verify_future = asyncio.ensure_future(worker._run_inflight_verify(item, lease))
        await asyncio.wait_for(gate_entered.wait(), timeout=15.0)

        # Wall-clock comfortably exceeds the (tiny) budget several times over
        # — a lease with no live dispatch would already have been
        # progress-aborted by now.
        await asyncio.sleep(worker.INFLIGHT_VERIFY_PROGRESS_BUDGET_SECS * 5)

        assert not verify_future.done(), (
            'a REMOTE lease with a live ssh dispatch must never be '
            'progress-aborted (task 2362 owns tearing down a genuinely dead '
            'dispatch)'
        )
        assert q.empty(), 'a live-dispatch remote lease must not be re-dispatched by the progress budget'

        verify_future.cancel()
        with contextlib.suppress(BaseException):
            await verify_future

    async def test_remote_lease_with_runner_missing_dispatch_in_flight_attr_is_not_progress_aborted(
        self,
        git_ops: GitOps,
        config: OrchestratorConfig,
    ) -> None:
        """task 2566 reviewer finding (test-coverage, merge_queue.py:11139):
        pins the `getattr(lease.runner, 'dispatch_in_flight', True)`
        fail-safe DEFAULT, which no other test here exercises. Every other
        remote test sets `fake_remote.dispatch_in_flight` explicitly on a
        MagicMock — but MagicMock auto-vivifies any attribute on first
        access, so even reading an unset `fake_remote.dispatch_in_flight`
        never actually falls through to getattr's third-positional-arg
        default; it returns the mock's freshly auto-created (truthy) child
        attribute instead.

        This test uses a plain stub runner class that genuinely has no
        `dispatch_in_flight` attribute (AttributeError on direct access), so
        getattr's default is the only thing that can make the no-progress
        clock keep resetting. Mirrors
        test_remote_lease_with_live_dispatch_is_not_progress_aborted above:
        the runner must be treated as "dispatch live" and never
        progress-aborted.
        """
        from orchestrator.merge_queue import SpeculativeMergeWorker
        from orchestrator.verify_runner import HostLease

        never_release = asyncio.Event()
        gate_entered = asyncio.Event()

        class _RemoteRunnerMissingDispatchInFlight:
            """No `dispatch_in_flight` attribute at all — unlike MagicMock,
            reading it raises AttributeError, so getattr(..., True) in
            merge_queue's trigger 3 genuinely falls through to its default.
            """

            name = 'remote-host'
            is_local = False

            async def run_merge_verify(self, *args: object, **kwargs: object) -> MagicMock:
                gate_entered.set()
                await never_release.wait()
                return _pass_result()  # pragma: no cover — never reached in this test

            async def cancel_verify(self) -> int:
                return 0

        stub_runner = _RemoteRunnerMissingDispatchInFlight()
        assert not hasattr(stub_runner, 'dispatch_in_flight'), (
            'sanity: this stub must genuinely lack the attribute for the '
            'getattr(..., True) fail-safe default to be exercised'
        )

        req, item = await _make_merged_item(
            git_ops, config, 'remote-no-attr-a', 'rna.py', 'x=1\n',
        )
        q: asyncio.Queue[MergeRequest] = asyncio.Queue()
        worker = SpeculativeMergeWorker(git_ops, q)
        worker._register_owned_merge_worktree(item.merge_wt)

        worker.VERIFY_ABANDON_POLL_SECS = 0.02
        worker.INFLIGHT_VERIFY_PROGRESS_PROBE_SECS = 0.02
        worker.INFLIGHT_VERIFY_PROGRESS_BUDGET_SECS = 0.2

        lease = HostLease(name='remote-host', runner=stub_runner, is_local=False)

        verify_future = asyncio.ensure_future(worker._run_inflight_verify(item, lease))
        await asyncio.wait_for(gate_entered.wait(), timeout=15.0)

        # Wall-clock comfortably exceeds the (tiny) budget several times over
        # — a lease whose progress signal read False would already have been
        # progress-aborted by now.
        await asyncio.sleep(worker.INFLIGHT_VERIFY_PROGRESS_BUDGET_SECS * 5)

        assert not verify_future.done(), (
            'a REMOTE runner exposing no dispatch_in_flight attribute must '
            'fail safe to "dispatch live" (getattr default True) and never '
            'be progress-aborted'
        )
        assert q.empty(), (
            'a runner missing dispatch_in_flight must not be re-dispatched '
            'by the progress budget'
        )

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

    task 2921 (load-robustness): same rationale as
    TestDeadInflightVerifyAborts above — the sub-1s budget tunables each
    test sets are load-safe for the dead/coast (must-eventually-abort-or-
    convert) attempts below, since host load only delays a bounded,
    expected outcome, never suppresses it. The SUCCESS attempt in every
    4-attempt test patches _run_post_merge_verify directly (not
    run_scoped_verification), so it completes synchronously and never
    races the budget with real git work. Every outer asyncio.wait_for(...)
    wrapping a _run_inflight_verify call below is widened from 5.0 to
    15.0, guarding against a >5s cumulative event-loop stall
    under a full-suite storm raising TimeoutError instead of letting the
    (still-bounded) dead-verify machinery resolve — even 4 widened waits
    in one test stays comfortably under the 60s pytest default timeout.
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
                worker._run_inflight_verify(item1, lease1), timeout=15.0,
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
                worker._run_inflight_verify(item2, lease2), timeout=15.0,
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
        # task 2921: patches _run_post_merge_verify directly (not
        # run_scoped_verification) so the real main-SHA/git work
        # run_scoped_verification's caller does under merge_wt never races
        # the tiny 0.2s budget under host load — mirrors
        # test_repeated_remote_coast_converts_to_blocked_and_success_resets_counter's
        # Attempt 3 below.
        req3, item3 = await _make_merged_item(
            git_ops, config, 'dead-repeat-branch-3', 'dr3.py', 'c=3\n', task_id=task_id,
        )
        worker._register_owned_merge_worktree(item3.merge_wt)
        lease3 = HostLease(name='local', runner=fake_local, is_local=True)

        async def _pass_fast(*args: object, **kwargs: object) -> None:
            return None

        with patch('orchestrator.merge_queue._run_post_merge_verify', _pass_fast):
            result3 = await asyncio.wait_for(
                worker._run_inflight_verify(item3, lease3), timeout=15.0,
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
                worker._run_inflight_verify(item4, lease4), timeout=15.0,
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
                worker._run_inflight_verify(item1, lease1), timeout=15.0,
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
                worker._run_inflight_verify(item2, lease2), timeout=15.0,
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
                worker._run_inflight_verify(item3, lease3), timeout=15.0,
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
                worker._run_inflight_verify(item1, lease1), timeout=15.0,
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
                worker._run_inflight_verify(item2, lease2), timeout=15.0,
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
                worker._run_inflight_verify(item3, lease3), timeout=15.0,
            )
        assert result3.status == InflightStatus.REQUEUED, (
            'a task_id resubmitted right after its blocked resolution must '
            'get a fresh dead-verify-abort budget, not immediately re-block'
        )

    async def test_repeated_remote_coast_converts_to_blocked_and_success_resets_counter(
        self,
        git_ops: GitOps,
        config: OrchestratorConfig,
    ) -> None:
        """task 2566: repeated REMOTE-lease-held coasts (dispatch_in_flight
        =False, no live ssh dispatch) must convert to terminal 'blocked'
        after MAX_INFLIGHT_DEAD_VERIFY_ABORTS, exactly mirroring the LOCAL
        busy-loop guard above (test_repeated_dead_verify_converts_to_blocked_and_success_resets_counter)
        — and a subsequent successful remote verify must clear the per-task
        counter.

        RED until step-4 GREEN un-gates trigger 3 for remote leases: today
        a remote coast is re-queued forever (trigger 3 never runs for a
        remote lease), so the busy-loop cap never trips.
        """
        from orchestrator.merge_queue import InflightStatus, SpeculativeMergeWorker
        from orchestrator.verify_runner import HostLease

        task_id = 'remote-coast-repeat-task'

        async def _dead_remote_gate(*args: object, **kwargs: object) -> MagicMock:
            await asyncio.Event().wait()
            raise AssertionError('unreachable — this Event is never set')  # pragma: no cover

        q: asyncio.Queue[MergeRequest] = asyncio.Queue()
        worker = SpeculativeMergeWorker(git_ops, q)
        worker.VERIFY_ABANDON_POLL_SECS = 0.02
        worker.INFLIGHT_VERIFY_PROGRESS_PROBE_SECS = 0.02
        worker.INFLIGHT_VERIFY_PROGRESS_BUDGET_SECS = 0.2
        worker.MAX_INFLIGHT_DEAD_VERIFY_ABORTS = 2

        fake_remote = MagicMock()
        fake_remote.name = 'remote-host'
        fake_remote.is_local = False
        fake_remote.cancel_verify = AsyncMock(return_value=0)

        # ── Attempt 1: remote coast (no live dispatch) -> REQUEUED (counter -> 1, below MAX) ──
        req1, item1 = await _make_merged_item(
            git_ops, config, 'remote-coast-repeat-branch-1', 'rcr1.py', 'a=1\n', task_id=task_id,
        )
        worker._register_owned_merge_worktree(item1.merge_wt)
        fake_remote.dispatch_in_flight = False
        fake_remote.run_merge_verify = AsyncMock(side_effect=_dead_remote_gate)
        lease1 = HostLease(name='remote-host', runner=fake_remote, is_local=False)
        result1 = await asyncio.wait_for(
            worker._run_inflight_verify(item1, lease1), timeout=15.0,
        )
        assert result1.status == InflightStatus.REQUEUED
        assert not req1.result.done()
        # Drain the re-queued req1 so it doesn't shadow the emptiness checks
        # below — mirrors the LOCAL variant above.
        assert q.get_nowait() is req1

        # ── Attempt 2 (MAX-th, same task_id): remote coast -> terminal 'blocked' ──
        req2, item2 = await _make_merged_item(
            git_ops, config, 'remote-coast-repeat-branch-2', 'rcr2.py', 'b=2\n', task_id=task_id,
        )
        worker._register_owned_merge_worktree(item2.merge_wt)
        fake_remote.dispatch_in_flight = False
        fake_remote.run_merge_verify = AsyncMock(side_effect=_dead_remote_gate)
        lease2 = HostLease(name='remote-host', runner=fake_remote, is_local=False)
        result2 = await asyncio.wait_for(
            worker._run_inflight_verify(item2, lease2), timeout=15.0,
        )

        assert result2.status is None, (
            f'the MAX-th consecutive remote coast must resolve terminally, '
            f'not REQUEUED again — got status={result2.status!r}'
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

        assert q.empty(), 'the MAX-th remote coast must NOT be re-queued (busy-loop guard)'

        # ── Attempt 3: a SUCCESSFUL remote verify for the SAME task_id clears the counter ──
        # Patches _run_post_merge_verify directly (not fake_remote.run_merge_verify)
        # so this test doesn't need to fabricate a scoped VerifyResult that
        # satisfies every internal gate inside _run_post_merge_verify — mirrors
        # test_failed_verify_also_clears_dead_verify_abort_counter's Attempt 2
        # above (same file, same rationale, different outcome: pass not fail).
        req3, item3 = await _make_merged_item(
            git_ops, config, 'remote-coast-repeat-branch-3', 'rcr3.py', 'c=3\n', task_id=task_id,
        )
        worker._register_owned_merge_worktree(item3.merge_wt)
        lease3 = HostLease(name='remote-host', runner=fake_remote, is_local=False)

        async def _pass_fast(*args: object, **kwargs: object) -> None:
            return None

        with patch('orchestrator.merge_queue._run_post_merge_verify', _pass_fast):
            result3 = await asyncio.wait_for(
                worker._run_inflight_verify(item3, lease3), timeout=15.0,
            )
        assert result3.status is None and result3.outcome is None, (
            f'expected a clean pass, got {result3!r}'
        )
        assert worker._inflight_dead_verify_aborts.get(task_id, 0) == 0, (
            'a successful remote verify must clear the per-task dead-verify-abort counter'
        )

        # ── Attempt 4: remote coast again for the SAME task_id starts a FRESH episode ──
        req4, item4 = await _make_merged_item(
            git_ops, config, 'remote-coast-repeat-branch-4', 'rcr4.py', 'd=4\n', task_id=task_id,
        )
        worker._register_owned_merge_worktree(item4.merge_wt)
        fake_remote.dispatch_in_flight = False
        fake_remote.run_merge_verify = AsyncMock(side_effect=_dead_remote_gate)
        lease4 = HostLease(name='remote-host', runner=fake_remote, is_local=False)
        result4 = await asyncio.wait_for(
            worker._run_inflight_verify(item4, lease4), timeout=15.0,
        )
        assert result4.status == InflightStatus.REQUEUED, (
            'after a successful verify clears the counter, the next remote '
            'coast must start a fresh episode (REQUEUED), not immediately '
            'resolve blocked'
        )


# ---------------------------------------------------------------------------
# task 3003 amend (reviewer_comprehensive, duplication): the three pieces of
# scaffold every TestContendedLeaseDefers case below needs — a LOCAL host
# lease, a refusing warm-swap reset, and one dispatch attempt through
# _run_inflight_verify.  Factored here (the pattern this file already uses for
# _make_merged_item) so each test carries only its own knobs and assertions:
# the duplicated scaffold was ~150 of the class's lines and hid the real
# per-test differences.
# ---------------------------------------------------------------------------


def _local_lease() -> Any:
    """A single-slot LOCAL ``HostLease`` over a stub runner.

    LOCAL because the whole contended-lane seam lives in
    ``_run_inflight_verify``'s local warm-swap branch.  A fresh instance per
    dispatch is deliberate and safe: ``_run_inflight_verify`` only reads
    ``lease.is_local``/``lease.runner`` and never releases the lease (that is
    ``_finalize_inflight``'s job), so nothing accumulates on it across attempts.
    """
    from orchestrator.verify_runner import HostLease  # noqa: PLC0415

    fake_local = MagicMock()
    fake_local.name = 'local'
    fake_local.is_local = True
    return HostLease(name='local', runner=fake_local, is_local=True)


def _held_lane_reset(warm_path: Path, holder_pgid: int) -> Any:
    """A ``reset_persistent_merge_worktree`` stand-in that refuses IMMEDIATELY.

    Stands in for the FOREIGN-holder pre-check at the top of the real method:
    ``MergeVerifyLeaseHeld``, raised with no bounded wait at all (hence no
    ``wait_secs``), which is what makes the defer's minimum inter-attempt
    period load-bearing.
    """
    from orchestrator.git_ops import MergeVerifyLeaseHeld  # noqa: PLC0415

    async def _reset(*_a: object, **_k: object) -> Path:
        raise MergeVerifyLeaseHeld(warm_path, holder_pgid)

    return _reset


async def _drive_defer(
    worker: Any,
    git_ops: GitOps,
    config: OrchestratorConfig,
    branch: str,
    *,
    task_id: str | None = None,
    lease: Any = None,
    reset: Any = None,
    verify: Any = None,
    timeout: float = 15.0,
) -> tuple[MergeRequest, Any]:
    """Drive ONE dispatch attempt for *task_id* through ``_run_inflight_verify``.

    Fresh merged item on its own *branch* but (when *task_id* is given) the SAME
    task_id, which is what a re-dispatch of one task looks like and what lets
    the per-task streak state accumulate across attempts.

    Patches ``reset_persistent_merge_worktree`` with *reset* and/or
    ``_run_post_merge_verify`` with *verify* for the duration of the attempt
    only.  Returns ``(req, InflightVerifyResult)``.
    """
    req, item = await _make_merged_item(
        git_ops, config, branch, f'{branch}.py', 'x=1\n', task_id=task_id,
    )
    worker._register_owned_merge_worktree(item.merge_wt)
    worker._request_ledger.on_dequeue(req, now=1_000_000.0)
    with contextlib.ExitStack() as stack:
        if reset is not None:
            stack.enter_context(
                patch.object(git_ops, 'reset_persistent_merge_worktree', reset)
            )
        if verify is not None:
            stack.enter_context(
                patch('orchestrator.merge_queue._run_post_merge_verify', verify)
            )
        result = await asyncio.wait_for(
            worker._run_inflight_verify(
                item, lease if lease is not None else _local_lease(),
            ),
            timeout=timeout,
        )
    return req, result


@pytest.mark.asyncio
class TestContendedLeaseDefers:
    """A contended merge-verify lease DEFERS (requeues), never blocks
    (task 2828, limb 2 / step-05/06).

    When ``merge_verify_lease`` raises MergeVerifyLeaseContended (its bounded
    wait timed out rather than yield a verify unprotected), the exception
    surfaces at ``verify_task.result()`` inside ``_run_inflight_verify``.
    Today the generic ``except Exception`` maps that to
    MergeOutcome('blocked'); step-06 adds an ``except MergeVerifyLeaseContended``
    clause BEFORE it that requeues the item exactly like the operator-halt
    abort — req.result left pending, per-task retry counters untouched.

    task 3003 extends the SAME contract to the warm-swap RESET seam, which
    takes the SAME ``<lane_dir>.lock`` one step earlier in
    ``_run_inflight_verify``: ``_acquire_warm_verify_worktree`` (merge_queue.py's
    LOCAL branch) → ``GitOps.reset_persistent_merge_worktree``. Its bounded-wait
    timeout used to raise a BARE RuntimeError, which the generic handler mapped
    to a deterministic-reason MergeOutcome('blocked') — identical signature on
    every attempt, which then fed workflow.py's consecutive_merge_thrash ladder
    into a false-positive L2 (reify 5354/5300/5328, signature 3173b64436423738).
    """

    async def test_reset_persistent_merge_worktree_contended_defers(
        self,
        warm_git_ops: GitOps,
        warm_config: OrchestratorConfig,
        monkeypatch: pytest.MonkeyPatch,
    ) -> None:
        """task 3003, limb (a): a contended lane lock on the warm-swap RESET
        must DEFER (requeue), exactly like a contended lease.

        Mechanism reproduced verbatim from the incident: a SAME-process holder
        of ``lane_lock_path(_merge-verify)`` with NO holder-pgid rendezvous
        written.  ``_merge_verify_lease_active()`` (git_ops.py:2163) is
        fail-OPEN on a missing rendezvous key, so the ``MergeVerifyLeaseHeld``
        pre-check (which also excludes our OWN pgid) never fires and control
        falls straight through to the bounded flock acquire — which times out.

        RED on main: today that timeout is a bare ``RuntimeError``, so the
        generic ``except Exception`` resolves
        ``MergeOutcome('blocked', reason='Verification error: Timed out after
        30s ...')`` with ``req.result`` RESOLVED.
        """
        from orchestrator import git_ops as git_ops_mod  # noqa: PLC0415
        from orchestrator.merge_queue import (  # noqa: PLC0415
            InflightStatus,
            SpeculativeMergeWorker,
        )
        from orchestrator.verify_cancel import lane_lock_path  # noqa: PLC0415

        # Shrink the RESET path's own bounded wait so the test doesn't sit for
        # the real 30s.  STRICT setattr (no raising=False) on purpose: the
        # reset must have its OWN constant, decoupled from the seed's
        # `flock(1) -w` arg (_SEED_WARM_LANE_LOCK_WAIT_SECS) — if it silently
        # kept sharing that one, this line must fail rather than pass by
        # accident.
        monkeypatch.setattr(git_ops_mod, '_RESET_WARM_LANE_LOCK_WAIT_SECS', 1)

        req, item = await _make_merged_item(
            warm_git_ops, warm_config, 'reset-contended-a', 'rca.py', 'x=1\n',
        )
        q: asyncio.Queue[MergeRequest] = asyncio.Queue()
        worker = SpeculativeMergeWorker(warm_git_ops, q)
        worker._register_owned_merge_worktree(item.merge_wt)
        # This test's raiser waits only 1s (shrunk above), so the defer's
        # minimum inter-attempt period would otherwise sleep the remaining 29s.
        # The throttle itself is covered by test_zero_wait_defer_is_throttled /
        # test_self_throttling_raiser_is_not_slept_again — opt out here.
        worker.CONTENDED_LEASE_DEFER_MIN_PERIOD_SECS = 0.0

        lease = _local_lease()

        worker._request_ledger.on_dequeue(req, now=1_000_000.0)

        # Sentinel: the defer must neither increment NOR pop this counter
        # (unlike the generic 'blocked' path, which pops it).
        worker._inflight_dead_verify_aborts[req.task_id] = 2

        verify_started = False

        async def _must_not_run(*_a: object, **_k: object) -> object:
            nonlocal verify_started
            verify_started = True
            raise AssertionError(
                'the warm-swap reset must raise BEFORE any verify is dispatched'
            )

        lock_path = lane_lock_path(warm_git_ops.persistent_merge_worktree_path)
        lock_path.parent.mkdir(parents=True, exist_ok=True)
        lock_fd = os.open(lock_path, os.O_RDWR | os.O_CREAT)
        fcntl.flock(lock_fd, fcntl.LOCK_EX | fcntl.LOCK_NB)
        try:
            with (
                patch(
                    'orchestrator.merge_queue._run_post_merge_verify', _must_not_run,
                ),
                patch.object(
                    worker, '_note_requeue', wraps=worker._note_requeue,
                ) as spy_note,
                patch.object(
                    worker, '_release_or_cleanup', wraps=worker._release_or_cleanup,
                ) as spy_release,
            ):
                result = await asyncio.wait_for(
                    worker._run_inflight_verify(item, lease), timeout=15.0,
                )
        finally:
            fcntl.flock(lock_fd, fcntl.LOCK_UN)
            os.close(lock_fd)

        assert not verify_started, (
            'the reset raised before the verify — _run_post_merge_verify must '
            'never have been reached'
        )

        # DEFER, not block:
        assert result.status == InflightStatus.REQUEUED, (
            f'a contended lane lock on the warm-swap reset must REQUEUE, got '
            f'status={result.status!r} outcome={result.outcome!r}'
        )
        assert result.outcome is None, 'requeue must carry no MergeOutcome'
        assert result.merge_wt is None, 'merge_wt must be released on requeue'
        assert not req.result.done(), (
            'req.result must be left PENDING (deferred), never resolved to '
            'blocked — a resolved blocked outcome here is what fed the '
            'consecutive_merge_thrash false-positive L2'
        )

        # Same requeue mechanics as the contended-lease block:
        assert not q.empty(), 'the request must be re-dispatched onto _queue'
        assert req.request_id not in worker._request_ledger.open_request_ids(), (
            'on_requeued must clear the ledger entry so the parked request '
            'never ages out'
        )
        spy_release.assert_called_once()
        spy_note.assert_called_once()

        # Per-task retry counter untouched (neither incremented nor popped).
        assert worker._inflight_dead_verify_aborts.get(req.task_id) == 2, (
            'the contended-reset requeue must leave the per-task '
            'dead-verify-abort counter untouched'
        )

    async def test_reset_lane_lease_held_defers(
        self,
        warm_git_ops: GitOps,
        warm_config: OrchestratorConfig,
        caplog: pytest.LogCaptureFixture,
    ) -> None:
        """task 3003, limb (c): a DIFFERENT-process lease holder on the SAME
        warm-swap seam must ALSO defer, not block.

        ``MergeVerifyLeaseHeld`` is raised by the fail-CLOSED pre-check at the
        very top of ``reset_persistent_merge_worktree`` (git_ops.py) when a
        FOREIGN live pgid holds the merge-verify lease.  It reaches
        ``_run_inflight_verify`` through exactly the same
        ``_acquire_warm_verify_worktree`` call as the contended timeout, and
        it deserves exactly the same response — the tree was never touched and
        the holder will eventually go away.

        RED on main: the type is not in the requeue ``except`` arm, so it falls
        to the generic ``except Exception`` and resolves
        ``MergeOutcome('blocked', reason='Verification error: Refusing to reset
        persistent merge worktree ...')`` — another deterministic,
        thrash-feeding signature, and a direct contradiction of the REQUEUE /
        ``counts_against_requeue_cap=False`` disposition ALREADY declared for
        this exact type in workflow_types.py.

        Also pins the streak log's honesty: ``MergeVerifyLeaseHeld`` carries no
        ``wait_secs``, so the rising-severity ERROR must NOT print a fabricated
        "~0 min of continuous contention" duration — a zero there would tell an
        operator the opposite of the truth (the holder has been there a while;
        we simply cannot say how long from this exception).
        """
        from orchestrator.merge_queue import (  # noqa: PLC0415
            InflightStatus,
            SpeculativeMergeWorker,
        )

        # A pgid that is neither ours nor live (Linux pid_max is nowhere near
        # 2**31-1) — stands in for the foreign holder the real pre-check finds.
        foreign_pgid = 2**31 - 1
        warm_path = warm_git_ops.persistent_merge_worktree_path
        _lease_held_reset = _held_lane_reset(warm_path, foreign_pgid)

        q: asyncio.Queue[MergeRequest] = asyncio.Queue()
        worker = SpeculativeMergeWorker(warm_git_ops, q)
        # Small threshold so the SECOND attempt deterministically crosses it
        # (mirrors test_consecutive_contended_requeues_raise_log_severity).
        worker.CONTENDED_LEASE_REQUEUE_WARN_STREAK = 2
        # MergeVerifyLeaseHeld carries no wait_secs, so the defer's minimum
        # inter-attempt period would sleep its full 30s on EACH attempt here.
        # The throttle is covered by test_zero_wait_defer_is_throttled — this
        # test is about the defer contract and the streak log, so opt out.
        worker.CONTENDED_LEASE_DEFER_MIN_PERIOD_SECS = 0.0

        lease = _local_lease()

        task_id = 'lease-held-holder'

        async def _drive_one(branch: str) -> tuple[MergeRequest, Any]:
            # Distinct branch, SAME task_id — a re-dispatch of one task, which
            # is what makes the per-task streak state accumulate.
            return await _drive_defer(
                worker, warm_git_ops, warm_config, branch,
                task_id=task_id, lease=lease, reset=_lease_held_reset,
            )

        # ── Attempt 1: the full defer contract, identical to the contended case ──
        worker._inflight_dead_verify_aborts[task_id] = 2
        with (
            patch.object(
                worker, '_note_requeue', wraps=worker._note_requeue,
            ) as spy_note,
            patch.object(
                worker, '_release_or_cleanup', wraps=worker._release_or_cleanup,
            ) as spy_release,
        ):
            req, result = await _drive_one('lhd-0')

        assert result.status == InflightStatus.REQUEUED, (
            f'a foreign-held merge-verify lease on the warm-swap reset must '
            f'REQUEUE, got status={result.status!r} outcome={result.outcome!r}'
        )
        assert result.outcome is None, 'requeue must carry no MergeOutcome'
        assert result.merge_wt is None, 'merge_wt must be released on requeue'
        assert not req.result.done(), (
            'req.result must be left PENDING (deferred), never resolved to blocked'
        )
        assert not q.empty(), 'the request must be re-dispatched onto _queue'
        assert req.request_id not in worker._request_ledger.open_request_ids(), (
            'on_requeued must clear the ledger entry so the parked request '
            'never ages out'
        )
        spy_release.assert_called_once()
        spy_note.assert_called_once()
        assert worker._inflight_dead_verify_aborts.get(task_id) == 2, (
            'the lease-held requeue must leave the per-task dead-verify-abort '
            'counter untouched'
        )

        # ── Attempt 2: crosses the streak threshold — the ERROR must not lie ──
        caplog.clear()
        with caplog.at_level(logging.WARNING, logger='orchestrator.merge_queue'):
            _, result2 = await _drive_one('lhd-1')

        assert result2.status == InflightStatus.REQUEUED
        streak = worker.CONTENDED_LEASE_REQUEUE_WARN_STREAK
        assert worker._contended_lease_requeues[task_id] == streak, (
            'consecutive lease-held defers for one task must accumulate the '
            'same operator-visible streak a contended lease does'
        )
        errors = [r for r in caplog.records if r.levelno >= logging.ERROR]
        assert errors, 'crossing the streak threshold must log an ERROR'
        msg = errors[0].getMessage()
        assert str(streak) in msg, (
            'the rising-severity ERROR must still name the streak length'
        )
        assert str(foreign_pgid) in msg, (
            'the ERROR must name the foreign holder pgid so an operator can '
            'identify who is holding the lane'
        )
        assert 'continuous contention' not in msg, (
            'MergeVerifyLeaseHeld carries no wait_secs, so the streak ERROR '
            'must NOT print a fabricated duration estimate (getattr(..., 0.0) '
            'renders "~0 min of continuous contention", which tells the '
            'operator the exact opposite of the truth). Suppress the duration '
            'clause — or substitute a holder-pgid clause — when wait_secs is '
            f'absent. Got: {msg!r}'
        )

    async def test_zero_wait_defer_is_throttled(
        self,
        warm_git_ops: GitOps,
        warm_config: OrchestratorConfig,
        monkeypatch: pytest.MonkeyPatch,
    ) -> None:
        """task 3003 review fix (1): a ZERO-WAIT defer must be THROTTLED.

        ``MergeVerifyLeaseHeld`` is raised by an IMMEDIATE fail-CLOSED
        pre-check at the top of ``reset_persistent_merge_worktree`` — before
        any bounded wait.  So unlike the two contended raisers (which have
        already burned 30 s / 300 s inside the acquire by the time they
        surface), this one costs nothing to hit.  With a bare
        ``_release_or_cleanup`` + ``put_nowait`` defer the merger would spin
        dequeue → ``git worktree add`` + merge → instant refusal → cleanup →
        requeue at whatever rate git allows, for the WHOLE 1–2 h holder
        window.  The defer must therefore observe a minimum INTER-ATTEMPT
        period.

        RED on main: ``CONTENDED_LEASE_DEFER_MIN_PERIOD_SECS`` does not exist
        (strict ``monkeypatch.setattr`` → AttributeError) and no sleep is
        performed.
        """
        from orchestrator.merge_queue import (  # noqa: PLC0415
            InflightStatus,
            SpeculativeMergeWorker,
        )

        foreign_pgid = 2**31 - 1
        warm_path = warm_git_ops.persistent_merge_worktree_path
        _lease_held_reset = _held_lane_reset(warm_path, foreign_pgid)

        req, item = await _make_merged_item(
            warm_git_ops, warm_config, 'defer-throttle-a', 'dta.py', 'x=1\n',
        )
        q: asyncio.Queue[MergeRequest] = asyncio.Queue()
        worker = SpeculativeMergeWorker(warm_git_ops, q)
        worker._register_owned_merge_worktree(item.merge_wt)
        # STRICT setattr (raising=True): the throttle must be a real class
        # attribute following the MAX_*/WARN_STREAK monkeypatch convention, not
        # something this test invents on the instance.
        monkeypatch.setattr(worker, 'CONTENDED_LEASE_DEFER_MIN_PERIOD_SECS', 0.25)

        lease = _local_lease()

        worker._request_ledger.on_dequeue(req, now=1_000_000.0)

        loop = asyncio.get_running_loop()
        with patch.object(
            warm_git_ops, 'reset_persistent_merge_worktree', _lease_held_reset,
        ):
            t0 = loop.time()
            result = await asyncio.wait_for(
                worker._run_inflight_verify(item, lease), timeout=15.0,
            )
            elapsed = loop.time() - t0

        # A sleep can only ever OVERSHOOT its argument, so a lower bound
        # slightly under the configured period is deterministic (no upper
        # bound is asserted — that would be timing-fragile).
        assert elapsed >= 0.2, (
            f'a zero-wait MergeVerifyLeaseHeld defer must observe the minimum '
            f'inter-attempt period (0.25s here) before re-queuing, else the '
            f'merger hot-spins merge→refuse→cleanup→requeue for the whole '
            f'holder window; returned after {elapsed:.3f}s'
        )

        # …and the throttle must not have cost us any of the defer contract:
        assert result.status == InflightStatus.REQUEUED
        assert result.outcome is None, 'requeue must carry no MergeOutcome'
        assert result.merge_wt is None, 'merge_wt must be released on requeue'
        assert not req.result.done(), (
            'req.result must be left PENDING (deferred), never resolved'
        )
        assert not q.empty(), 'the request must be re-dispatched onto _queue'

    async def test_self_throttling_raiser_is_not_slept_again(
        self,
        warm_git_ops: GitOps,
        warm_config: OrchestratorConfig,
        monkeypatch: pytest.MonkeyPatch,
    ) -> None:
        """task 3003 review fix (1), scope fence: the throttle is a minimum
        inter-attempt PERIOD, not an unconditional extra sleep.

        A raiser that already burned a bounded wait inside the acquire has
        ALREADY paid the period — task 2828's lease waits 300 s
        (``_MERGE_VERIFY_LEASE_WAIT_SECS``) and the reset path waits 30 s
        (``_RESET_WARM_LANE_LOCK_WAIT_SECS``), both ≥ any sane minimum period.
        Sleeping again on top of that would slow the ALREADY-throttled paths
        for no benefit, so the backoff must be ``max(0, PERIOD - wait_secs)``.
        """
        from orchestrator.git_ops import MergeVerifyLeaseContended  # noqa: PLC0415
        from orchestrator.merge_queue import (  # noqa: PLC0415
            InflightStatus,
            SpeculativeMergeWorker,
        )

        async def _lease_contended_reset(*_a: object, **_k: object) -> Path:
            raise MergeVerifyLeaseContended(Path('/x/_merge-verify.lock'), 300.0)

        req, item = await _make_merged_item(
            warm_git_ops, warm_config, 'defer-throttle-b', 'dtb.py', 'x=1\n',
        )
        q: asyncio.Queue[MergeRequest] = asyncio.Queue()
        worker = SpeculativeMergeWorker(warm_git_ops, q)
        worker._register_owned_merge_worktree(item.merge_wt)
        monkeypatch.setattr(worker, 'CONTENDED_LEASE_DEFER_MIN_PERIOD_SECS', 0.25)

        lease = _local_lease()

        worker._request_ledger.on_dequeue(req, now=1_000_000.0)

        # Asserted on the SLEEP, not on wall-clock elapsed.  The sibling test
        # above states the rule — an upper bound is timing-fragile — and it
        # binds here: the timed region also releases the merge worktree (real
        # git work), so on a loaded box (this repo runs its own merge workers
        # concurrently) that cost dwarfs the 0.25s signal and an `elapsed <`
        # bound false-fails.  The backoff duration is load-independent and
        # pins the contract directly.
        _sleeps: list[float] = []
        _real_sleep = asyncio.sleep

        async def _recording_sleep(delay: float, *a: object, **k: object) -> object:
            _sleeps.append(float(delay))
            return await _real_sleep(delay, *a, **k)

        with (
            patch.object(
                warm_git_ops,
                'reset_persistent_merge_worktree',
                _lease_contended_reset,
            ),
            patch.object(asyncio, 'sleep', _recording_sleep),
        ):
            result = await asyncio.wait_for(
                worker._run_inflight_verify(item, lease), timeout=15.0,
            )

        assert not [d for d in _sleeps if d >= 0.2], (
            f'wait_secs=300.0 already exceeds the 0.25s minimum inter-attempt '
            f'period, so the defer must sleep ZERO additional seconds — the '
            f'backoff is max(0, PERIOD - wait_secs), not an unconditional '
            f'extra sleep; slept {_sleeps!r}'
        )
        assert result.status == InflightStatus.REQUEUED
        assert result.outcome is None
        assert not req.result.done()
        assert not q.empty()

    async def test_cancel_during_defer_backoff_leaves_requeue_to_canceller(
        self,
        warm_git_ops: GitOps,
        warm_config: OrchestratorConfig,
        monkeypatch: pytest.MonkeyPatch,
    ) -> None:
        """task 3003 review fix (4): A CANCELLING CALLER OWNS THE REQUEST.

        ``_run_inflight_verify`` runs as a background ``ensure_future`` task, so
        a cancellation can land INSIDE the defer backoff.  It must NOT re-queue
        from a ``finally`` there: every production cancel site already
        discharges the request itself (``stop()`` resolves it with the shutdown
        outcome and retires it BEFORE cancelling; ``_verifier_loop``'s
        head-failure cascade either re-queues the entry or re-dispatches it via
        ``_remerge``), so a ``finally`` would DOUBLE-file it.

        This pins the convention already followed by the operator-halt and
        dead-verify defer branches on this same coroutine: they await before
        ``put_nowait`` too, and a cancellation there simply skips the requeue.
        The sibling test below drives the cascade path that a ``finally`` would
        actually corrupt.
        """
        from orchestrator.merge_queue import SpeculativeMergeWorker  # noqa: PLC0415

        foreign_pgid = 2**31 - 1
        warm_path = warm_git_ops.persistent_merge_worktree_path
        _lease_held_reset = _held_lane_reset(warm_path, foreign_pgid)

        req, item = await _make_merged_item(
            warm_git_ops, warm_config, 'defer-cancel', 'dc.py', 'x=1\n',
        )
        q: asyncio.Queue[MergeRequest] = asyncio.Queue()
        worker = SpeculativeMergeWorker(warm_git_ops, q)
        worker._register_owned_merge_worktree(item.merge_wt)
        # Long enough that the cancellation below lands squarely inside it.
        monkeypatch.setattr(worker, 'CONTENDED_LEASE_DEFER_MIN_PERIOD_SECS', 5.0)

        lease = _local_lease()

        worker._request_ledger.on_dequeue(req, now=1_000_000.0)

        # Deterministic barrier: wait until the ephemeral worktree has actually
        # been released before cancelling, so the cancellation provably lands in
        # the backoff and not in the (unbounded, git-I/O) cleanup ahead of it.
        # This also pins the ORDERING — the backoff must come AFTER the release,
        # never hold a worktree and its disk across the wait.
        released = asyncio.Event()
        _real_release = worker._release_or_cleanup

        async def _release_then_signal(
            merge_wt: Path | None, *, spec_warm: bool
        ) -> None:
            try:
                await _real_release(merge_wt, spec_warm=spec_warm)
            finally:
                released.set()

        with patch.object(worker, '_release_or_cleanup', _release_then_signal), \
                patch.object(
                    warm_git_ops, 'reset_persistent_merge_worktree', _lease_held_reset,
                ):
            verify_fut = asyncio.ensure_future(
                worker._run_inflight_verify(item, lease)
            )
            await asyncio.wait_for(released.wait(), timeout=10.0)
            await asyncio.sleep(0.1)
            assert not verify_fut.done(), (
                'the defer must still be inside its backoff here — if it has '
                'already returned there is no throttle to be cancelled during'
            )
            verify_fut.cancel()
            with contextlib.suppress(asyncio.CancelledError):
                await verify_fut

        assert q.empty(), (
            'cancellation mid-backoff must NOT re-queue: the cancelling caller '
            'owns the request. stop() has already resolved+retired it, and the '
            'head-failure cascade re-queues or re-dispatches it itself — a '
            'requeue from here would put it on _queue a SECOND time'
        )
        assert req.request_id in worker._request_ledger.open_request_ids(), (
            'the ledger entry must survive too: on_requeued belongs to the '
            'cancelling caller, which re-arms (or retires) the request itself'
        )
        assert not req.result.done(), (
            'a cancelled defer must still leave req.result PENDING — resolving '
            'it is likewise the canceller\'s job'
        )

    async def test_defer_streak_caps_terminally_after_max_contention_secs(
        self,
        warm_git_ops: GitOps,
        warm_config: OrchestratorConfig,
        monkeypatch: pytest.MonkeyPatch,
    ) -> None:
        """task 3003 review fix (2): the defer streak must be BOUNDED.

        A contended-lane defer is deliberately fail-safe — it never escalates
        and its disposition is ``counts_against_requeue_cap=False`` — so
        ``_contended_lease_requeues`` only ever raises log severity.  A
        PERMANENTLY wedged lane holder would therefore defer this task
        forever, with no terminal resolution and nothing but a repeating
        ERROR.  Past a generous elapsed budget the defer must convert to a
        terminal 'blocked', mirroring the MAX_INFLIGHT_DEAD_VERIFY_ABORTS
        busy-loop cap.

        RED on main: ``MAX_CONTENDED_LEASE_DEFER_SECS`` does not exist (strict
        monkeypatch → AttributeError) and every defer requeues unconditionally.
        """
        from orchestrator.merge_queue import (  # noqa: PLC0415
            InflightStatus,
            SpeculativeMergeWorker,
        )

        foreign_pgid = 2**31 - 1
        warm_path = warm_git_ops.persistent_merge_worktree_path
        _lease_held_reset = _held_lane_reset(warm_path, foreign_pgid)

        q: asyncio.Queue[MergeRequest] = asyncio.Queue()
        worker = SpeculativeMergeWorker(warm_git_ops, q)
        worker.CONTENDED_LEASE_DEFER_MIN_PERIOD_SECS = 0.0
        # Strict setattr: the cap must be a real class attribute (same
        # convention as MAX_INFLIGHT_DEAD_VERIFY_ABORTS), not test invention.
        monkeypatch.setattr(worker, 'MAX_CONTENDED_LEASE_DEFER_SECS', 0.05)

        lease = _local_lease()

        task_id = 'defer-cap'

        async def _drive_one(branch: str) -> tuple[MergeRequest, Any]:
            # Distinct branch, SAME task_id — a re-dispatch of one task, which
            # is what makes the per-task streak state accumulate.
            return await _drive_defer(
                worker, warm_git_ops, warm_config, branch,
                task_id=task_id, lease=lease, reset=_lease_held_reset,
            )

        # task 3003 amend (reviewer_comprehensive, resource_cleanup): one short
        # of MAX_INFLIGHT_DEAD_VERIFY_ABORTS (3), so if the terminal cap below
        # fails to pop this counter the task's very NEXT dead/hung verify after
        # an operator clears the lane would trip the busy-loop cap and be
        # abandoned without a requeue.
        worker._inflight_dead_verify_aborts[task_id] = 2

        # ── Attempt 1: inside the budget — the 3003 defer behaviour stands ──
        req1, result1 = await _drive_one('cap-0')
        assert result1.status == InflightStatus.REQUEUED, (
            'contention BELOW the elapsed cap must still DEFER — the cap must '
            'not collapse the whole fix back into an immediate block'
        )
        assert not req1.result.done(), 'a below-cap defer leaves req.result PENDING'
        assert q.qsize() == 1
        # The DEFER path is not terminal, so it must leave the dead-verify
        # counter exactly as it found it (the pre-existing contract).
        assert worker._inflight_dead_verify_aborts.get(task_id) == 2, (
            'a defer must leave the per-task dead-verify-abort counter untouched'
        )

        await asyncio.sleep(0.1)  # push the streak past the 0.05s budget

        # ── Attempt 2: past the budget — terminal, not another defer ──
        req2, result2 = await _drive_one('cap-1')
        assert result2.outcome is not None, (
            'past MAX_CONTENDED_LEASE_DEFER_SECS the defer must resolve '
            'TERMINALLY — an unbounded fail-safe requeue with no escalation '
            'and no requeue-cap accounting never converges'
        )
        assert result2.outcome.status == 'blocked'
        assert result2.merge_wt is None, 'the terminal branch must release merge_wt'
        assert req2.result.done(), 'the capped attempt must resolve req.result'
        assert req2.result.result() is result2.outcome
        assert q.qsize() == 1, (
            'the capped attempt must NOT be re-queued (only attempt 1 is on '
            'the queue)'
        )

        reason = result2.outcome.reason or ''
        assert str(foreign_pgid) in reason, (
            f'the terminal reason must name the lane holder so an operator can '
            f'go look at it; got {reason!r}'
        )
        assert re.search(r'\d+s', reason), (
            f'the terminal reason must state how long the lane was unavailable; '
            f'got {reason!r}'
        )

        # Both pieces of per-task state must be popped, so an operator-resolved
        # re-submission of this task gets a FRESH budget instead of capping on
        # its very first defer (same reason MAX_INFLIGHT_DEAD_VERIFY_ABORTS
        # pops its counter on the terminal path).
        assert task_id not in worker._contended_lease_requeues, (
            'the terminal cap must pop the streak counter'
        )
        assert task_id not in worker._contended_lease_first_defer_at, (
            'the terminal cap must pop the first-defer timestamp too, else a '
            'later unrelated contention streak inherits a stale start and caps '
            'out on its very first defer'
        )
        assert task_id not in worker._contended_lease_last_defer_at, (
            'the terminal cap must pop the last-defer timestamp too — the three '
            'streak dicts are only meaningful together'
        )
        # task 3003 amend (reviewer_comprehensive, resource_cleanup): and the
        # dead-verify-abort counter, exactly as the two OTHER terminal 'blocked'
        # exits on this coroutine do (the busy-loop cap and the generic
        # `except Exception`, both for the reasons recorded at task 2420).
        assert task_id not in worker._inflight_dead_verify_aborts, (
            'a TERMINAL cap-out must pop the per-task dead-verify-abort counter '
            'too, else the stale count survives an operator-resolved '
            're-submission and the task\'s first dead verify trips the '
            'busy-loop cap and is abandoned without a requeue — and the dict '
            'grows unboundedly for permanently-blocked task_ids'
        )

    async def test_stale_streak_stamp_does_not_cap_a_fresh_defer(
        self,
        warm_git_ops: GitOps,
        warm_config: OrchestratorConfig,
        monkeypatch: pytest.MonkeyPatch,
    ) -> None:
        """task 3003 amend (reviewer_comprehensive, robustness): the elapsed cap
        must measure CONTINUOUS contention, so a STALE streak stamp must never
        cap out the first defer of a fresh streak.

        ``_contended_lease_first_defer_at`` is only cleared on exits that
        actually re-reach ``_run_inflight_verify``.  A deferred request need not:
        its requeued dispatch can remerge into a CONFLICT (a DecidedItem
        passthrough that never enters this coroutine), be abandoned, or die at
        shutdown — while the per-worker dicts live for the whole orchestrator
        process (~8 h between fleet redeploys).

        The failure this pins: task X defers once at T0 (stamp set), its
        requeued dispatch resolves as a conflict so no verify ever runs, and
        HOURS later the same task_id is re-submitted and hits ONE brief,
        entirely transient lane contention.  With the stamp still in place the
        elapsed span is already past the cap, so the VERY FIRST defer of that
        fresh streak resolves ``MergeOutcome('blocked')`` — reintroducing exactly
        the false-positive blocked-on-transient-lane-busy resolution this task
        was chartered to remove, via its own guard.

        RED before the amend: the first defer of the new streak caps out.
        """
        from orchestrator.merge_queue import (  # noqa: PLC0415
            InflightStatus,
            SpeculativeMergeWorker,
        )

        foreign_pgid = 2**31 - 1
        warm_path = warm_git_ops.persistent_merge_worktree_path
        _lease_held_reset = _held_lane_reset(warm_path, foreign_pgid)

        q: asyncio.Queue[MergeRequest] = asyncio.Queue()
        worker = SpeculativeMergeWorker(warm_git_ops, q)
        worker.CONTENDED_LEASE_DEFER_MIN_PERIOD_SECS = 0.0
        # A cap this task's seeded stamp is FAR past, so the only thing that can
        # keep this defer alive is recognising the streak as broken.
        monkeypatch.setattr(worker, 'MAX_CONTENDED_LEASE_DEFER_SECS', 5.0)
        # Strict setattr: the staleness window must be a real class attribute
        # (same convention as MAX_*/WARN_STREAK), not test invention.
        monkeypatch.setattr(worker, 'CONTENDED_LEASE_STREAK_STALE_FACTOR', 4.0)
        monkeypatch.setattr(worker, 'CONTENDED_LEASE_STREAK_STALE_FLOOR_SECS', 60.0)

        task_id = 'stale-streak'

        # The state an ABANDONED streak leaves behind: one defer, an hour ago,
        # with nothing since — a gap no defer cadence can explain (the raiser
        # here carries no wait at all and the throttle is 0, so the staleness
        # window is its 60s floor).
        _long_ago = time.monotonic() - 3600.0
        worker._contended_lease_requeues[task_id] = 1
        worker._contended_lease_first_defer_at[task_id] = _long_ago
        worker._contended_lease_last_defer_at[task_id] = _long_ago

        _, result = await _drive_defer(
            worker, warm_git_ops, warm_config, 'stale-streak-0',
            task_id=task_id, reset=_lease_held_reset,
        )

        assert result.status == InflightStatus.REQUEUED, (
            f'the FIRST defer of a fresh streak must DEFER even when a stale '
            f'start stamp from an abandoned streak survives — capping here is '
            f'the false-positive blocked-on-transient-lane-busy resolution this '
            f'task removed; got status={result.status!r} '
            f'outcome={result.outcome!r}'
        )
        assert result.outcome is None, 'a fresh defer must carry no MergeOutcome'
        assert q.qsize() == 1, 'the fresh defer must be re-queued'

        # …and the streak state must have been RESTARTED, not merely tolerated:
        # a fresh count of 1 and a stamp measuring ~0s elapsed, so the next real
        # streak gets the full budget it is entitled to.
        assert worker._contended_lease_requeues[task_id] == 1, (
            f'the broken streak must be closed and a NEW one opened at 1, not '
            f'continued; got {worker._contended_lease_requeues.get(task_id)!r}'
        )
        assert (
            time.monotonic() - worker._contended_lease_first_defer_at[task_id]
            < worker.MAX_CONTENDED_LEASE_DEFER_SECS
        ), (
            'the new streak must date from THIS defer — a stamp still inside '
            'the stale window would cap out the very next defer instead'
        )

    async def test_capped_defer_reasons_are_thrash_signature_distinct(
        self,
        warm_git_ops: GitOps,
        warm_config: OrchestratorConfig,
        monkeypatch: pytest.MonkeyPatch,
    ) -> None:
        """task 3003 review fix (2), the guard on the guard: two consecutive
        cap-outs must NOT share a merge-outcome signature.

        The cap re-introduces a 'blocked' resolution on the lane-contention
        path — the exact resolution class 3003 was chartered to remove.  If its
        reason string were invariant, two consecutive cap-outs would hash to
        the same ``merge_outcome_signature`` and walk workflow.py's
        ``consecutive_merge_thrash`` ladder (``max_consecutive_merge_thrash``
        defaults to 2) straight back into the false-positive L2.

        task 3003 amend (reviewer_comprehensive, test_quality): the collision is
        forced rather than hoped against.  Resting the guarantee on the elapsed
        seconds and the streak count would make it PROBABILISTIC, because in
        production both are near-CONSTANT across consecutive cap-outs — each
        streak starts fresh at the cap, so each caps at the first defer past the
        same budget at the same cadence, rendering ~the same '14400s across 481
        consecutive deferred attempts' and differing only in the sub-second
        jitter that survives ``:.0f``.  So both cap-outs here are driven with
        BYTE-IDENTICAL elapsed and streak components (seeded from inside the
        refusing reset, i.e. microseconds before the arm reads them, so no real
        git work can perturb the rendering) and the reasons must STILL differ.
        Only a strictly-increasing per-worker cap-out ordinal can do that.

        ``normalize_cause_hint`` strips only ANSI escapes and file:line tails,
        so standalone digits genuinely survive into the hash — asserted below
        rather than assumed, since the whole argument rests on it.
        """
        from shared.task_metadata import RetryLedger  # noqa: PLC0415

        from orchestrator.merge_queue import (  # noqa: PLC0415
            SpeculativeMergeWorker,
        )

        # The normalizer keeps standalone digits (its file:line rule needs a
        # source-file extension before the colon), so an integer-seconds
        # component is genuinely signature-bearing. Pin that here — the whole
        # anti-thrash argument below rests on it.
        assert (
            RetryLedger.normalize_cause_hint('lane unavailable for 7s')
            != RetryLedger.normalize_cause_hint('lane unavailable for 8s')
        ), 'normalize_cause_hint must preserve standalone digits'

        from orchestrator.git_ops import MergeVerifyLeaseHeld  # noqa: PLC0415

        foreign_pgid = 2**31 - 1
        warm_path = warm_git_ops.persistent_merge_worktree_path

        q: asyncio.Queue[MergeRequest] = asyncio.Queue()
        worker = SpeculativeMergeWorker(warm_git_ops, q)
        worker.CONTENDED_LEASE_DEFER_MIN_PERIOD_SECS = 0.0
        monkeypatch.setattr(worker, 'MAX_CONTENDED_LEASE_DEFER_SECS', 5.0)

        lease = _local_lease()

        task_id = 'defer-cap-sig'
        # 7.0s of elapsed contention and a streak of 6+1 — the SAME numbers for
        # both cap-outs.  Seeded from INSIDE the refusing reset, the last thing
        # that runs before the defer arm, so the microsecond gap to the arm's
        # own time.monotonic() read cannot shift the ':.0f' rendering; seeding
        # before the dispatch would let _make_merged_item's real git work (which
        # can easily take a second) push one cap-out's elapsed to '8s' and pass
        # this test for entirely the wrong reason.
        _seeded_elapsed = 7.0
        _seeded_streak = 6

        async def _seeding_reset(*_a: object, **_k: object) -> Path:
            _now = time.monotonic()
            worker._contended_lease_requeues[task_id] = _seeded_streak
            worker._contended_lease_first_defer_at[task_id] = _now - _seeded_elapsed
            worker._contended_lease_last_defer_at[task_id] = _now
            raise MergeVerifyLeaseHeld(warm_path, foreign_pgid)

        async def _drive_one(branch: str) -> tuple[MergeRequest, Any]:
            # Distinct branch, SAME task_id — a re-dispatch of one task, which
            # is what makes the per-task streak state accumulate.
            return await _drive_defer(
                worker, warm_git_ops, warm_config, branch,
                task_id=task_id, lease=lease, reset=_seeding_reset,
            )

        # ── Two cap-outs of the same wedged lane, with IDENTICAL renderings of
        # every OTHER reason component ──
        _, result_a = await _drive_one('sig-a0')
        assert result_a.outcome is not None and result_a.outcome.status == 'blocked'
        reason_a = result_a.outcome.reason or ''
        # Cap-out A popped all three streak dicts, so cap-out B genuinely
        # re-seeds from scratch — exactly as a re-submission after an operator
        # looked at (but did not fix) the lane would.
        assert task_id not in worker._contended_lease_first_defer_at
        _, result_b = await _drive_one('sig-b0')
        assert result_b.outcome is not None and result_b.outcome.status == 'blocked'
        reason_b = result_b.outcome.reason or ''

        # Pre-condition of the real assertion: the elapsed + streak components
        # genuinely COLLIDE here, so any surviving difference is structural.
        _shape = re.compile(r'unavailable for (\d+)s across (\d+) consecutive')
        _shape_a, _shape_b = _shape.search(reason_a), _shape.search(reason_b)
        assert _shape_a is not None and _shape_b is not None, (
            f'the terminal reason must state the elapsed span and the streak '
            f'length; got {reason_a!r} / {reason_b!r}'
        )
        assert _shape_a.groups() == _shape_b.groups() == (
            f'{_seeded_elapsed:.0f}', str(_seeded_streak + 1),
        ), (
            f'this test only means something if the elapsed/streak components '
            f'collide — seed them harder; got {_shape_a.groups()} vs '
            f'{_shape_b.groups()}'
        )

        assert reason_a != reason_b, (
            f'two cap-outs whose elapsed span and streak length render '
            f'IDENTICALLY produced an IDENTICAL reason — in production those two '
            f'components are near-constant across consecutive cap-outs, so this '
            f'is the invariant signature that fed the consecutive_merge_thrash '
            f'false-positive L2 this task removed. Carry a strictly-increasing '
            f'per-worker cap-out ordinal in the reason. Got {reason_a!r} both '
            f'times.'
        )
        sig_a = RetryLedger.compute_merge_outcome_signature('', '', reason_a)
        sig_b = RetryLedger.compute_merge_outcome_signature('', '', reason_b)
        assert sig_a != sig_b, (
            f'the cap-out reasons differ but normalise to the SAME '
            f'merge_outcome_signature ({sig_a}), so consecutive cap-outs would '
            f'still walk the anti-thrash ladder: {reason_a!r} vs {reason_b!r}'
        )

    async def test_contended_lease_requeues_and_leaves_result_pending(
        self,
        git_ops: GitOps,
        config: OrchestratorConfig,
    ) -> None:
        from orchestrator.git_ops import MergeVerifyLeaseContended
        from orchestrator.merge_queue import InflightStatus, SpeculativeMergeWorker

        async def _lease_contended_verify(*_args: object, **_kwargs: object) -> object:
            # The lease acquire timed out; merge_verify_lease raised before the
            # verify body ran. Surfaces at verify_task.result() in
            # _run_inflight_verify exactly as the real lease would.
            raise MergeVerifyLeaseContended(
                Path('/x/_merge-verify.lock'), 300.0,
            )

        req, item = await _make_merged_item(
            git_ops, config, 'lease-contended-a', 'lca.py', 'x=1\n',
        )
        q: asyncio.Queue[MergeRequest] = asyncio.Queue()
        worker = SpeculativeMergeWorker(git_ops, q)
        worker._register_owned_merge_worktree(item.merge_wt)

        lease = _local_lease()

        worker._request_ledger.on_dequeue(req, now=1_000_000.0)

        # Pre-seed the per-task dead-verify-abort counter with a sentinel so we
        # can prove the contended-lease requeue neither increments NOR pops it
        # (unlike the generic 'blocked' path, which pops it).
        worker._inflight_dead_verify_aborts[req.task_id] = 2

        with (
            patch(
                'orchestrator.merge_queue._run_post_merge_verify',
                _lease_contended_verify,
            ),
            patch.object(
                worker, '_note_requeue', wraps=worker._note_requeue,
            ) as spy_note,
            patch.object(
                worker, '_release_or_cleanup', wraps=worker._release_or_cleanup,
            ) as spy_release,
        ):
            result = await asyncio.wait_for(
                worker._run_inflight_verify(item, lease), timeout=5.0,
            )

        # DEFER, not block:
        assert result.status == InflightStatus.REQUEUED, (
            f'a contended lease must REQUEUE, got status={result.status!r}'
        )
        assert result.outcome is None, 'requeue must carry no MergeOutcome'
        assert result.merge_wt is None, 'merge_wt must be released on requeue'
        assert not req.result.done(), (
            'req.result must be left PENDING (deferred), never resolved to blocked'
        )

        # Same requeue mechanics as the operator-halt block:
        assert not q.empty(), 'the request must be re-dispatched onto _queue'
        assert req.request_id not in worker._request_ledger.open_request_ids(), (
            'on_requeued must clear the ledger entry so the parked request '
            'never ages out'
        )
        spy_release.assert_called_once()
        spy_note.assert_called_once()

        # Per-task retry counter untouched (neither incremented nor popped).
        assert worker._inflight_dead_verify_aborts.get(req.task_id) == 2, (
            'the contended-lease requeue must leave the per-task '
            'dead-verify-abort counter untouched'
        )

    async def test_two_consecutive_contended_resets_produce_no_blocked_outcome(
        self,
        warm_git_ops: GitOps,
        warm_config: OrchestratorConfig,
        monkeypatch: pytest.MonkeyPatch,
    ) -> None:
        """task 3003, the ANTI-THRASH invariant — the incident's exact shape.

        Two consecutive dispatch attempts for the SAME task against a lane
        lock that is held throughout must produce ZERO ``MergeOutcome``s.

        Why this is the right altitude for "consecutive_merge_thrash
        untouched", and not a workflow-level harness: ``workflow.py`` gates its
        whole anti-thrash block on ``self._last_merge_block_reason``, which is
        only ever set from a RESOLVED blocked outcome.  A defer leaves
        ``req.result`` PENDING, so ``compute_merge_outcome_signature`` is never
        computed and ``consecutive_merge_thrash`` is never incremented — the
        counter is STRUCTURALLY unreachable for this failure class rather than
        merely happening to stay at zero.  Asserting "no outcome, future
        pending, twice in a row" is therefore the strongest honest statement
        available here; reading a counter that is never written would be
        theatre.

        Two is the number that matters: ``max_consecutive_merge_thrash``
        defaults to 2, so two identical contended timeouts were precisely what
        tripped the false-positive L2 (reify 5354/5300/5328, signature
        3173b64436423738) — each attempt used to resolve
        ``MergeOutcome('blocked', reason='Verification error: Timed out after
        30s ...')``, a reason string with no varying component, hence the same
        signature every time.
        """
        from orchestrator import git_ops as git_ops_mod  # noqa: PLC0415
        from orchestrator.merge_queue import (  # noqa: PLC0415
            InflightStatus,
            SpeculativeMergeWorker,
        )
        from orchestrator.verify_cancel import lane_lock_path  # noqa: PLC0415

        monkeypatch.setattr(git_ops_mod, '_RESET_WARM_LANE_LOCK_WAIT_SECS', 1)

        q: asyncio.Queue[MergeRequest] = asyncio.Queue()
        worker = SpeculativeMergeWorker(warm_git_ops, q)
        # 1s bounded wait (shrunk above) → the defer's minimum inter-attempt
        # period would sleep the remaining 29s on each of the two attempts.
        # Covered separately by test_zero_wait_defer_is_throttled.
        worker.CONTENDED_LEASE_DEFER_MIN_PERIOD_SECS = 0.0

        lease = _local_lease()

        task_id = 'thrash-shape'

        async def _must_not_run(*_a: object, **_k: object) -> object:
            raise AssertionError(
                'the warm-swap reset must raise BEFORE any verify is dispatched'
            )

        lock_path = lane_lock_path(warm_git_ops.persistent_merge_worktree_path)
        lock_path.parent.mkdir(parents=True, exist_ok=True)
        lock_fd = os.open(lock_path, os.O_RDWR | os.O_CREAT)
        fcntl.flock(lock_fd, fcntl.LOCK_EX | fcntl.LOCK_NB)
        try:
            reqs = []
            for i in range(2):
                # Fresh merged item on a distinct branch, SAME task_id — this
                # is what a re-dispatch of the same task looks like.
                req, result = await _drive_defer(
                    worker, warm_git_ops, warm_config, f'thrash-{i}',
                    task_id=task_id, lease=lease, verify=_must_not_run,
                )
                reqs.append(req)

                assert result.status == InflightStatus.REQUEUED, (
                    f'attempt #{i + 1} against a held lane lock must REQUEUE, '
                    f'got status={result.status!r} outcome={result.outcome!r}'
                )
                assert result.outcome is None, (
                    f'attempt #{i + 1} produced a MergeOutcome — with a '
                    f'deterministic reason this is exactly the thrash food '
                    f'that tripped the false-positive L2: {result.outcome!r}'
                )
        finally:
            fcntl.flock(lock_fd, fcntl.LOCK_UN)
            os.close(lock_fd)

        assert not any(r.result.done() for r in reqs), (
            'ZERO MergeOutcomes across two consecutive contended attempts — '
            'every req.result must still be PENDING, so no merge_outcome_'
            'signature is ever computed and consecutive_merge_thrash stays '
            'structurally unreachable'
        )
        assert worker._contended_lease_requeues[task_id] == 2, (
            'both defers must land on the same per-task streak counter so a '
            'genuinely wedged holder is still operator-visible'
        )

    async def test_genuine_verify_failure_still_blocks_on_warm_path(
        self,
        warm_git_ops: GitOps,
        warm_config: OrchestratorConfig,
    ) -> None:
        """SCOPE FENCE: with the lane lock FREE, a real verify failure on the
        warm path must still surface its failure — the defer must not have
        swallowed the normal blocked route.

        Both failure shapes are covered, and they legitimately differ:

        * the verify RAISES → the generic ``except Exception`` resolves
          ``MergeOutcome('blocked')`` on ``req.result`` immediately (so a
          failed verify never stalls the queue);
        * the verify RETURNS a failing outcome → ``_run_inflight_verify``
          hands it back on ``InflightVerifyResult.outcome`` and leaves
          ``req.result`` for ``_finalize_inflight`` to resolve, which is that
          method's documented contract ("does NOT resolve req.result ...
          except on exception").

        What must hold for BOTH: NOT requeued, and the contended-lane streak
        POPPED — because the lane lock genuinely WAS acquired here, so any
        prior streak is broken.
        """
        from orchestrator.merge_queue import (  # noqa: PLC0415
            InflightStatus,
            SpeculativeMergeWorker,
        )

        q: asyncio.Queue[MergeRequest] = asyncio.Queue()
        worker = SpeculativeMergeWorker(warm_git_ops, q)

        lease = _local_lease()

        failing = MergeOutcome('blocked', reason='verify failed: 3 tests')

        async def _verify_raises(*_a: object, **_k: object) -> object:
            raise RuntimeError('verify boom')

        async def _verify_returns_failure(*_a: object, **_k: object) -> object:
            return failing

        # ── Shape 1: the verify RAISES ──
        task_id_a = 'warm-verify-raises'
        req_a, item_a = await _make_merged_item(
            warm_git_ops, warm_config, 'warm-fail-raise', 'wfr.py', 'x=1\n',
            task_id=task_id_a,
        )
        worker._register_owned_merge_worktree(item_a.merge_wt)
        worker._request_ledger.on_dequeue(req_a, now=1_000_000.0)
        # Pre-seed a streak so we can prove an ACQUIRED lane breaks it.
        worker._contended_lease_requeues[task_id_a] = 1
        with patch('orchestrator.merge_queue._run_post_merge_verify', _verify_raises):
            result_a = await asyncio.wait_for(
                worker._run_inflight_verify(item_a, lease), timeout=15.0,
            )

        assert result_a.status != InflightStatus.REQUEUED, (
            'a genuine verify failure must NOT be deferred — the lane lock was '
            'free and the verify really ran'
        )
        assert result_a.outcome is not None and result_a.outcome.status == 'blocked', (
            f'expected a blocked MergeOutcome, got {result_a.outcome!r}'
        )
        assert 'verify boom' in result_a.outcome.reason
        assert req_a.result.done(), (
            'a raising verify resolves req.result immediately so the failure '
            'never stalls the queue'
        )
        assert q.empty(), 'a genuine verify failure must not be re-queued'
        assert task_id_a not in worker._contended_lease_requeues, (
            'the lane lock WAS acquired, so any prior contended streak is '
            'broken and must be popped'
        )

        # ── Shape 2: the verify RETURNS a failing outcome ──
        task_id_b = 'warm-verify-returns-fail'
        req_b, item_b = await _make_merged_item(
            warm_git_ops, warm_config, 'warm-fail-return', 'wfrt.py', 'y=1\n',
            task_id=task_id_b,
        )
        worker._register_owned_merge_worktree(item_b.merge_wt)
        worker._request_ledger.on_dequeue(req_b, now=1_000_000.0)
        worker._contended_lease_requeues[task_id_b] = 1
        with patch(
            'orchestrator.merge_queue._run_post_merge_verify',
            _verify_returns_failure,
        ):
            result_b = await asyncio.wait_for(
                worker._run_inflight_verify(item_b, lease), timeout=15.0,
            )

        assert result_b.status != InflightStatus.REQUEUED, (
            'a returned verify failure must NOT be deferred'
        )
        assert result_b.outcome is failing, (
            f'the failing outcome must be handed back verbatim, got '
            f'{result_b.outcome!r}'
        )
        assert q.empty(), 'a genuine verify failure must not be re-queued'
        assert task_id_b not in worker._contended_lease_requeues, (
            'the lane lock WAS acquired, so any prior contended streak is '
            'broken and must be popped'
        )

    async def test_reset_git_failure_still_blocks(
        self,
        warm_git_ops: GitOps,
        warm_config: OrchestratorConfig,
    ) -> None:
        """SCOPE FENCE: a non-contention fault from INSIDE the reset's body
        must still map to ``MergeOutcome('blocked')``, never to a defer.

        This is what keeps the typed raise honest.  ``reset_persistent_merge_
        worktree`` raises a plain ``RuntimeError`` for every git fault in its
        body (``git worktree add`` / ``git reset --hard`` / the artifact-
        retaining clean) and ``MergeVerifyLeaseContended`` ONLY at the lock
        acquire.  If the typed raise had been scoped any wider — e.g. wrapping
        the whole method — a genuine, PERMANENT git fault would be re-queued
        forever, converting a loud blocked outcome into a silent infinite
        defer loop.  So this test is the reason the raise is site-local.
        """
        from orchestrator.merge_queue import (  # noqa: PLC0415
            InflightStatus,
            SpeculativeMergeWorker,
        )

        async def _reset_git_fault(*_a: object, **_k: object) -> Path:
            # Verbatim shape of the in-body faults at git_ops.py's
            # reset-in-place branch (a plain RuntimeError, NOT the typed class).
            raise RuntimeError(
                f'Failed to reset persistent merge worktree '
                f'{warm_git_ops.persistent_merge_worktree_path} to deadbeef: '
                f'fatal: Could not reset index file to revision'
            )

        async def _must_not_run(*_a: object, **_k: object) -> object:
            raise AssertionError('the reset faulted — no verify may be dispatched')

        req, item = await _make_merged_item(
            warm_git_ops, warm_config, 'reset-git-fault', 'rgf.py', 'x=1\n',
        )
        q: asyncio.Queue[MergeRequest] = asyncio.Queue()
        worker = SpeculativeMergeWorker(warm_git_ops, q)
        worker._register_owned_merge_worktree(item.merge_wt)

        lease = _local_lease()

        worker._request_ledger.on_dequeue(req, now=1_000_000.0)

        with (
            patch.object(
                warm_git_ops, 'reset_persistent_merge_worktree', _reset_git_fault,
            ),
            patch('orchestrator.merge_queue._run_post_merge_verify', _must_not_run),
        ):
            result = await asyncio.wait_for(
                worker._run_inflight_verify(item, lease), timeout=15.0,
            )

        assert result.status != InflightStatus.REQUEUED, (
            f'a git fault inside the reset is a REAL failure, not a busy lane — '
            f'deferring it would loop forever, got status={result.status!r}'
        )
        assert result.outcome is not None and result.outcome.status == 'blocked', (
            f'expected a blocked MergeOutcome, got {result.outcome!r}'
        )
        assert 'Failed to reset persistent merge worktree' in result.outcome.reason
        assert req.result.done(), 'a genuine git fault must RESOLVE req.result'
        assert q.empty(), 'a genuine git fault must not be re-queued'
        assert req.task_id not in worker._contended_lease_requeues, (
            'a git fault is not lane contention — it must never open a '
            'contended-lane streak'
        )

    async def test_consecutive_contended_requeues_raise_log_severity(
        self,
        git_ops: GitOps,
        config: OrchestratorConfig,
        caplog: pytest.LogCaptureFixture,
    ) -> None:
        """After CONTENDED_LEASE_REQUEUE_WARN_STREAK unbroken contended-lease
        requeues for the SAME task_id, the per-requeue WARNING rises to an
        ERROR naming the streak, so a long-running / wedged lane holder is
        operator-visible instead of looping silently; a subsequent verify that
        actually RUNS resets the streak (task 2828 amend — reviewer_comprehensive
        robustness suggestion #1).
        """
        from orchestrator.git_ops import MergeVerifyLeaseContended
        from orchestrator.merge_queue import (
            InflightStatus,
            SpeculativeMergeWorker,
        )

        async def _lease_contended_verify(*_a: object, **_k: object) -> object:
            raise MergeVerifyLeaseContended(Path('/x/_merge-verify.lock'), 300.0)

        async def _generic_verify_error(*_a: object, **_k: object) -> object:
            # A non-lease verify error: the verify actually ran (lease acquired)
            # and failed — hits the generic 'blocked' path that RESETS the streak.
            raise RuntimeError('verify boom')

        q: asyncio.Queue[MergeRequest] = asyncio.Queue()
        worker = SpeculativeMergeWorker(git_ops, q)
        # Small threshold for a fast, deterministic streak (mirrors the
        # MAX_INFLIGHT_DEAD_VERIFY_ABORTS monkeypatch convention).
        worker.CONTENDED_LEASE_REQUEUE_WARN_STREAK = 3

        lease = _local_lease()

        task_id = 'wedged-holder'

        async def _drive_one(branch: str, verify: object) -> Any:
            # Distinct branch, SAME task_id — a re-dispatch of one task, which
            # is what makes the per-task streak state accumulate.
            _, result = await _drive_defer(
                worker, git_ops, config, branch,
                task_id=task_id, lease=lease, verify=verify, timeout=5.0,
            )
            return result

        # Requeues 1..threshold-1 stay at WARNING (no ERROR yet).
        for i in range(worker.CONTENDED_LEASE_REQUEUE_WARN_STREAK - 1):
            caplog.clear()
            with caplog.at_level(logging.WARNING, logger='orchestrator.merge_queue'):
                result = await _drive_one(f'clc-{i}', _lease_contended_verify)
            assert result.status == InflightStatus.REQUEUED
            assert worker._contended_lease_requeues[task_id] == i + 1
            assert not [r for r in caplog.records if r.levelno >= logging.ERROR], (
                f'requeue #{i + 1} is below the streak threshold — must stay WARNING'
            )

        # The requeue that REACHES the threshold rises to ERROR naming the streak.
        caplog.clear()
        with caplog.at_level(logging.WARNING, logger='orchestrator.merge_queue'):
            result = await _drive_one('clc-cross', _lease_contended_verify)
        assert result.status == InflightStatus.REQUEUED
        streak = worker.CONTENDED_LEASE_REQUEUE_WARN_STREAK
        assert worker._contended_lease_requeues[task_id] == streak
        errors = [r for r in caplog.records if r.levelno >= logging.ERROR]
        assert errors, 'crossing the streak threshold must log an ERROR'
        assert str(streak) in errors[0].getMessage(), (
            'the rising-severity ERROR must name the streak length so an '
            'operator can see how long the holder has blocked this verify'
        )

        # A verify that actually RUNS (here: a generic verify error, i.e. the
        # lease WAS acquired) resets the streak for that task.
        result = await _drive_one('clc-runs', _generic_verify_error)
        assert task_id not in worker._contended_lease_requeues, (
            'a verify that actually ran must reset the contended-lease streak'
        )

    async def test_streak_error_is_logged_once_at_the_crossing(
        self,
        warm_git_ops: GitOps,
        warm_config: OrchestratorConfig,
        caplog: pytest.LogCaptureFixture,
    ) -> None:
        """task 3003 review fix (3): the rising-severity ERROR fires ONCE, at
        the crossing — not on every defer past it.

        The branch tests ``streak >= CONTENDED_LEASE_REQUEUE_WARN_STREAK``, so
        once the streak is crossed EVERY subsequent defer emits an identical
        ERROR.  Against a wedged holder that is thousands of duplicate ERROR
        lines saying nothing new.  The operator still wants a per-defer
        heartbeat, so the WARNING stays on every defer; only the alarm is
        de-duplicated (with step-10's terminal
        ``MAX_CONTENDED_LEASE_DEFER_SECS`` cap closing the streak out loudly at
        the other end).

        RED on main: ``>=`` emits an ERROR on defers 2, 3 and 4.
        """
        from orchestrator.merge_queue import (  # noqa: PLC0415
            InflightStatus,
            SpeculativeMergeWorker,
        )

        foreign_pgid = 2**31 - 1
        warm_path = warm_git_ops.persistent_merge_worktree_path
        _lease_held_reset = _held_lane_reset(warm_path, foreign_pgid)

        q: asyncio.Queue[MergeRequest] = asyncio.Queue()
        worker = SpeculativeMergeWorker(warm_git_ops, q)
        worker.CONTENDED_LEASE_DEFER_MIN_PERIOD_SECS = 0.0
        worker.CONTENDED_LEASE_REQUEUE_WARN_STREAK = 2
        # Comfortably above the whole test's wall-clock, so the terminal cap
        # cannot fire first and truncate the streak we are measuring.
        worker.MAX_CONTENDED_LEASE_DEFER_SECS = 3600.0

        lease = _local_lease()

        task_id = 'streak-log-dedupe'

        async def _drive_one(branch: str) -> InflightStatus | None:
            _, result = await _drive_defer(
                worker, warm_git_ops, warm_config, branch,
                task_id=task_id, lease=lease, reset=_lease_held_reset,
            )
            return result.status

        # Four consecutive defers, NOT clearing caplog between them.
        caplog.clear()
        with caplog.at_level(logging.WARNING, logger='orchestrator.merge_queue'):
            for i in range(4):
                assert await _drive_one(f'sld-{i}') == InflightStatus.REQUEUED

        lane_records = [
            r for r in caplog.records if 'lane unavailable' in r.getMessage()
        ]
        errors = [r for r in lane_records if r.levelno >= logging.ERROR]
        warnings = [r for r in lane_records if r.levelno == logging.WARNING]

        assert len(errors) == 1, (
            f'the streak ERROR must be emitted exactly ONCE, at the crossing — '
            f'a `>=` test re-alarms on every defer past the threshold, which '
            f'against a wedged holder is thousands of identical lines. Got '
            f'{len(errors)}: {[r.getMessage() for r in errors]}'
        )
        assert str(worker.CONTENDED_LEASE_REQUEUE_WARN_STREAK) in (
            errors[0].getMessage()
        ), 'the crossing ERROR must still name the streak length'
        assert len(warnings) == 3, (
            f'defers below the threshold AND every defer past it must still '
            f'emit a per-defer WARNING heartbeat (4 defers - 1 crossing ERROR '
            f'= 3 WARNINGs); got {len(warnings)}'
        )

    async def test_deferred_attempt_does_not_advance_cold_verify_valve(
        self,
        warm_git_ops: GitOps,
        warm_config: OrchestratorConfig,
    ) -> None:
        """task 3003, reviewer's secondary finding: a DEFERRED attempt must not
        advance the periodic cold-verify safety valve.

        ``self._verify_attempt_count += 1`` runs BEFORE
        ``_acquire_warm_verify_worktree``, so an attempt that never reaches a
        verify at all still counts toward
        ``persistent_merge_worktree_safety_valve_every_n``.  A long wedge would
        burn through the valve's period on pure defers and fire a from-scratch
        cold verify for no reason.  The counter must measure attempts that
        actually VERIFIED.

        RED on main: two defers advance it by two.
        """
        from orchestrator.merge_queue import (  # noqa: PLC0415
            InflightStatus,
            SpeculativeMergeWorker,
        )

        foreign_pgid = 2**31 - 1
        warm_path = warm_git_ops.persistent_merge_worktree_path
        _lease_held_reset = _held_lane_reset(warm_path, foreign_pgid)

        async def _verify_returns_failure(*_a: object, **_k: object) -> object:
            return MergeOutcome('blocked', reason='verify failed: 1 test')

        q: asyncio.Queue[MergeRequest] = asyncio.Queue()
        worker = SpeculativeMergeWorker(warm_git_ops, q)
        worker.CONTENDED_LEASE_DEFER_MIN_PERIOD_SECS = 0.0

        lease = _local_lease()

        count_before = worker._verify_attempt_count

        # ── Two defers: the warm acquire never succeeded, no verify ran ──
        for i in range(2):
            _, result = await _drive_defer(
                worker, warm_git_ops, warm_config, f'valve-defer-{i}',
                task_id='valve-defer', lease=lease, reset=_lease_held_reset,
            )
            assert result.status == InflightStatus.REQUEUED

        assert worker._verify_attempt_count == count_before, (
            f'a deferred attempt never ran a verify, so it must not advance '
            f'the cold-verify safety-valve counter — a long wedge would '
            f'otherwise fire a from-scratch cold verify purely from defers. '
            f'Went {count_before} -> {worker._verify_attempt_count}'
        )

        # ── One attempt that really verifies: the counter must advance by 1 ──
        # (the fix DEFERS the commit; it must not DROP the count.)
        req, item = await _make_merged_item(
            warm_git_ops, warm_config, 'valve-real', 'vr.py', 'x=1\n',
            task_id='valve-real',
        )
        worker._register_owned_merge_worktree(item.merge_wt)
        worker._request_ledger.on_dequeue(req, now=1_000_000.0)
        with patch(
            'orchestrator.merge_queue._run_post_merge_verify',
            _verify_returns_failure,
        ):
            result = await asyncio.wait_for(
                worker._run_inflight_verify(item, lease), timeout=60.0,
            )
        assert result.status != InflightStatus.REQUEUED

        assert worker._verify_attempt_count == count_before + 1, (
            f'an attempt whose warm acquire SUCCEEDED must still advance the '
            f'counter exactly once — the fix defers the commit past the '
            f'acquire, it does not drop it. Got '
            f'{worker._verify_attempt_count}, expected {count_before + 1}'
        )

    async def test_safety_valve_still_fires_on_the_nth_real_attempt(
        self,
        warm_git_ops: GitOps,
        warm_config: OrchestratorConfig,
    ) -> None:
        """SCOPE FENCE for the counter move: the valve must keep its documented
        1-BASED contract and must not shift by one.

        ``_safety_valve_due`` documents ``attempt_count`` as "the 1-based count
        of verifying attempts ... incremented before calling this", so the Nth
        real attempt — and only the Nth — must be handed
        ``safety_valve_due=True``.  Moving the commit past the acquire must
        preserve that exactly; an off-by-one here would either never fire the
        valve or fire it a verify early.
        """
        from orchestrator.merge_queue import SpeculativeMergeWorker  # noqa: PLC0415

        every_n = 2
        valve_config = warm_config.model_copy(
            update={
                'git': warm_config.git.model_copy(
                    update={
                        'persistent_merge_worktree_safety_valve_every_n': every_n,
                    },
                ),
            },
        )

        async def _verify_returns_failure(*_a: object, **_k: object) -> object:
            return MergeOutcome('blocked', reason='verify failed: 1 test')

        q: asyncio.Queue[MergeRequest] = asyncio.Queue()
        worker = SpeculativeMergeWorker(warm_git_ops, q)

        lease = _local_lease()

        # Observed at the BEHAVIOURAL seam rather than by spying the
        # safety_valve_due flag: _acquire_warm_verify_worktree's serial-head
        # branch returns the ephemeral merge_wt untouched when the valve is due
        # (`if not persistent_merge_worktree or safety_valve_due: return
        # merge_wt, False`), so it calls reset_persistent_merge_worktree on a
        # NOT-due attempt and skips it on a due one.  Spying that PUBLIC GitOps
        # method pins the same 1-based contract through its observable effect —
        # a cold from-scratch verify on the Nth attempt — and keeps this file
        # clear of a new orchestrator.merge_queue.<private> reach-back patch,
        # which test_merge_queue_reachback_patch_guard.py freezes by name (the
        # private is imported into merge_queue at module load, so patching the
        # defining merge_liveness module would not take effect and an
        # ALLOWLIST entry in the guard is the only alternative).
        reset_calls: list[str] = []
        _real_reset = warm_git_ops.reset_persistent_merge_worktree

        async def _spy_reset(merge_commit: str, *a: object, **k: object) -> Path:
            reset_calls.append(merge_commit)
            return await _real_reset(merge_commit, *a, **k)

        for i in range(every_n):
            req, item = await _make_merged_item(
                warm_git_ops, valve_config, f'valve-real-{i}', f'vrn{i}.py', 'x=1\n',
                task_id=f'valve-nth-{i}',
            )
            worker._register_owned_merge_worktree(item.merge_wt)
            worker._request_ledger.on_dequeue(req, now=1_000_000.0)
            _before = len(reset_calls)
            with (
                patch.object(
                    warm_git_ops, 'reset_persistent_merge_worktree', _spy_reset,
                ),
                patch(
                    'orchestrator.merge_queue._run_post_merge_verify',
                    _verify_returns_failure,
                ),
            ):
                await asyncio.wait_for(
                    worker._run_inflight_verify(item, lease), timeout=60.0,
                )
            _warm_swapped = len(reset_calls) > _before
            # 1-based: attempts 1..N-1 swap to the warm tree; the Nth is the
            # valve's cold from-scratch attempt and must NOT swap.
            _expected_warm = (i + 1) % every_n != 0
            assert _warm_swapped is _expected_warm, (
                f'with every_n={every_n} the valve must be due on the '
                f'{every_n}th real attempt and only then (1-based contract), so '
                f'attempt #{i + 1} must '
                f'{"swap to the warm tree" if _expected_warm else "run cold"}; '
                f'reset_persistent_merge_worktree '
                f'{"was" if _warm_swapped else "was not"} called'
            )

        assert len(reset_calls) == every_n - 1, (
            f'exactly the non-valve attempts may swap to the warm tree; got '
            f'{len(reset_calls)} warm swaps across {every_n} attempts'
        )


# ---------------------------------------------------------------------------
# task 3082 step-5 RED / step-6 GREEN: the dead-verify abort must requeue into
# the LIVE queue — registry included.
#
# Trigger 3 does `_release_or_cleanup + put_nowait + on_requeued` but NOT
# `_note_requeue`, so the registry is left at VERIFYING while the request sits
# on `_queue`. The existing TestDeadInflightVerifyAborts tests never registered
# the request at all, which is exactly why they never exercised the registry
# side of this branch. The one added `_register_item` line below is the whole
# difference.
# ---------------------------------------------------------------------------


@pytest.mark.asyncio
class TestDeadVerifyAbortRequeuesIntoTheLiveQueue:
    """A dead-verify no-progress abort must return the LIFECYCLE REGISTRY to
    QUEUED at the requeue site, so the re-queued request re-enters through the
    normal drain instead of being coalesce-dropped (task 3082 step-5 RED /
    step-6 GREEN).

    Extends TestDeadInflightVerifyAborts::
    test_dead_local_verify_is_aborted_and_requeued_within_budget's driver
    verbatim (real merged item, real wall clock, 0.2s no-progress budget, the
    never-returning gate) rather than inventing a second one.

    RED until step-6 adds the missing ``_note_requeue`` at the requeue site.
    """

    async def _drive_dead_abort(
        self,
        git_ops: GitOps,
        config: OrchestratorConfig,
        branch: str,
        filename: str,
    ):
        """Drive one trigger-3 dead-verify abort on a REGISTERED request.

        Returns ``(worker, q, req, vr, spy_note)``.
        """
        from orchestrator.merge_queue import ItemLifecycleState, SpeculativeMergeWorker
        from orchestrator.verify_runner import HostLease

        fake_eq = _FakeEscalationQueue(open_l1=False)
        req, item = await _make_merged_item(git_ops, config, branch, filename, 'x=1\n')
        q: asyncio.Queue[MergeRequest] = asyncio.Queue()
        worker = SpeculativeMergeWorker(git_ops, q, escalation_queue=fake_eq)
        worker._register_owned_merge_worktree(item.merge_wt)

        worker.VERIFY_ABANDON_POLL_SECS = 0.02
        worker.INFLIGHT_VERIFY_PROGRESS_PROBE_SECS = 0.02
        worker.INFLIGHT_VERIFY_PROGRESS_BUDGET_SECS = 0.2

        fake_local = MagicMock()
        fake_local.name = 'local'
        fake_local.is_local = True
        lease = HostLease(name='local', runner=fake_local, is_local=True)

        worker._request_ledger.on_dequeue(req, now=1_000_000.0)
        # THE one addition vs the existing driver: production always arrives
        # here registered at VERIFYING (via the dispatch chokepoint).
        worker._register_item(req, initial=ItemLifecycleState.VERIFYING)

        with (
            patch(
                'orchestrator.merge_queue.run_scoped_verification',
                _dead_gate_never_returns,
            ),
            patch.object(worker, '_note_requeue', wraps=worker._note_requeue) as spy_note,
        ):
            vr = await asyncio.wait_for(
                worker._run_inflight_verify(item, lease), timeout=15.0,
            )
        return worker, q, req, vr, spy_note, fake_eq

    async def test_trigger3_abort_returns_the_registry_to_queued_at_the_requeue_site(
        self, git_ops: GitOps, config: OrchestratorConfig,
    ) -> None:
        """The requeue SITE itself must bounce the registry to QUEUED — this
        assertion is deliberately independent of ``_finalize_inflight``'s
        chokepoint repair (never called here), so per-branch symmetry is
        pinned on its own.
        """
        from orchestrator.merge_queue import InflightStatus, ItemLifecycleState

        worker, q, req, vr, spy_note, _fake_eq = await self._drive_dead_abort(
            git_ops, config, 'df3082-dead-verify-requeue', 'dvr.py',
        )
        rid = req.request_id

        assert vr.status == InflightStatus.REQUEUED, (
            f'a dead-verify abort must REQUEUE, got status={vr.status!r}'
        )
        current = worker._lifecycle.current(rid)
        assert current == ItemLifecycleState.QUEUED, (
            f'the requeue site must return the registry to QUEUED so the request '
            f'can re-enter through the drain; registry reads {current!r}'
        )
        assert worker._live_items[rid] is req, (
            f'_live_items must hold the MergeRequest after the requeue: '
            f'{worker._live_items.get(rid)!r}'
        )
        # Drain the (inert — no worker loop is running here) queue rather than
        # reaching into asyncio.Queue's undocumented `_queue` deque: this is the
        # test's last use of `q`, and identity membership is the actual claim.
        parked = [q.get_nowait() for _ in range(q.qsize())]
        assert any(p is req for p in parked), (
            f'the request must actually be parked on the live queue: {parked!r}'
        )
        assert not req.result.done(), (
            'a re-queued request must be left PENDING for its re-dispatch'
        )
        assert rid not in worker._request_ledger.open_request_ids(), (
            'on_requeued must clear the ledger entry so the parked request never ages out'
        )
        spy_note.assert_called_once()

    async def test_abort_requeue_is_rebuffered_not_coalesce_dropped(
        self, git_ops: GitOps, config: OrchestratorConfig,
        caplog: pytest.LogCaptureFixture,
    ) -> None:
        """The re-queued request must survive the very next drain: land in a
        lane buffer, keep its future PENDING, and leave no finalize head.
        """
        from orchestrator.merge_queue import ItemLifecycleState

        worker, _q, req, _vr, _spy, fake_eq = await self._drive_dead_abort(
            git_ops, config, 'df3082-dead-verify-redrain', 'dvrd.py',
        )
        rid = req.request_id

        with caplog.at_level(logging.WARNING, logger='orchestrator.merge_queue'):
            worker._drain_queue_into_lanes()

        buffered = [it for buf in worker._lane_buffers.values() for it in buf]
        assert req in buffered, (
            f'the re-queued request must be buffered by the next drain, not '
            f'coalesce-dropped; lane buffers hold {buffered!r}'
        )
        current = worker._lifecycle.current(rid)
        assert current == ItemLifecycleState.LANE_BUFFERED, (
            f'the re-drained request must reach LANE_BUFFERED; registry reads {current!r}'
        )
        assert not req.result.done(), (
            'the real waiter must NOT be handed a fabricated already_merged by the '
            'coalesce path'
        )
        assert worker._finalizing_head_entry() is None, (
            f'no finalize head may survive the abort: {worker._finalizing_head_entry()!r}'
        )
        drop_warnings = [
            r.getMessage() for r in caplog.records
            if 'dropping duplicate/re-entrant merge submission' in r.getMessage()
        ]
        assert drop_warnings == [], (
            f'the re-drain must not be coalesce-dropped: {drop_warnings!r}'
        )
        assert fake_eq.submitted == [], (
            f'a clean abort-and-re-drain must not escalate: {fake_eq.submitted!r}'
        )


# ---------------------------------------------------------------------------
# task 3082 step-9: end-to-end — a dead-verify abort SELF-HEALS, and every
# user-observable surface tells the truth while it does.
#
# This is the consolidated guard against the whole class recurring through a
# DIFFERENT trigger.  Steps 2/4/6/8 fix the four mechanisms; this class pins
# the seven observable surfaces the task enumerates, mapped 1:1 in the
# assertions below.
#
# Deliberately NOT asserted: "landings resume" / queue throughput.  The
# task's corrected OPERATIONAL NOTE measured landings_total advancing 4->7
# while the zombie sat, with other tasks verifying on the same host
# afterwards.  The queue was never wedged — the defect is observability
# integrity plus the LATENT frozen-prefix wedge (surface 7).  Asserting
# throughput would pin a property that never broke.
# ---------------------------------------------------------------------------


def _assert_quiescent_registry(
    worker,
    main_sha: str,
    requests: list[MergeRequest],
) -> None:
    """Assert the registry/ledger/two-layer quiescence surfaces for *worker*.

    Copied and narrowed from test_merge_queue_invariant_integration_gate.py's
    ``_assert_quiescent`` (:510-586) — per-file duplication convention, see
    this module's docstring.  The permit/worktree sub-checks (b)/(c) are
    dropped deliberately: both short-circuit to ``[]`` on a stopped worker,
    and this class asserts AFTER ``worker.stop()``, so including them would
    be vacuous rather than meaningful.

    Retained, because each is meaningful post-stop:
      (a) every request resolved — no dangling in-flight work.
      (d) the request-liveness ledger is empty AFTER ``sweep_resolved()``.
          Resolution is detected PASSIVELY (RequestLedger has no on-resolve
          hook), so sweeping first is required, not optional.
      (e) ``two_layer_invariants(main_sha) == []`` — *main_sha* MUST be a
          REAL sha, never the ``'unknown'`` sentinel: the base-chain and
          verify-base sub-checks are silently SKIPPED for 'unknown', which
          would make this pass vacuously.
      (f) ``set(worker._lifecycle.non_terminal_items()) == set()`` — no
          ItemLifecycle registry leak survives quiescence.  This is the
          surface a phantom finalize head corrupts.
    """
    for req in requests:
        assert req.result.done() or req.result.cancelled(), (
            f'request {req.request_id!r} (task {req.task_id!r}) still pending at quiescence'
        )

    worker._request_ledger.sweep_resolved()
    assert worker._request_ledger.is_empty(), (
        f'request-liveness ledger non-empty at quiescence: '
        f'{worker._request_ledger.open_request_ids()!r}'
    )

    assert main_sha and main_sha != 'unknown', (
        f'this helper requires a REAL main_sha (the "unknown" sentinel silently '
        f'skips the frozen-prefix sub-checks), got {main_sha!r}'
    )
    tli = worker.two_layer_invariants(main_sha)
    assert tli == [], (
        f'two_layer_invariants({main_sha!r}) non-empty at quiescence: {tli!r}'
    )

    registry_ids = set(worker._lifecycle.non_terminal_items())
    assert registry_ids == set(), (
        f'ItemLifecycle registry non-terminal at quiescence: {registry_ids!r}'
    )


class _HangThenPassVerify:
    """Stateful ``run_scoped_verification`` stub: HANGS once, then PASSES.

    Call 1 blocks on a never-set Event (mirrors ``_dead_gate_never_returns``
    above) so trigger 3's no-progress budget fires and the request is
    RE-QUEUED.  Call 2 returns a pass, so the re-dispatched merge actually
    lands.  ``.calls`` is the surface-1 assertion: it must reach 2, proving
    the re-queued request genuinely re-entered the pipeline rather than being
    swallowed by ``_coalesce_reentrant_drain``.
    """

    def __init__(self) -> None:
        self.calls = 0
        self.first_entered = asyncio.Event()

    async def __call__(self, *args: object, **kwargs: object) -> MagicMock:
        self.calls += 1
        if self.calls == 1:
            self.first_entered.set()
            await asyncio.Event().wait()
            raise AssertionError('unreachable — never set')  # pragma: no cover
        return MagicMock(
            passed=True, summary='ok', test_output='ok',
            lint_output='', type_output='', category='',
            timed_out=False, verify_skipped=False,
        )


@pytest.mark.asyncio
class TestDeadVerifyAbortSelfHealsEndToEnd:
    """A dead-verify no-progress abort must SELF-HEAL through the live queue,
    leaving every user-observable surface truthful (task 3082 step-9).

    Modelled on ``TestWedgedVerifyIntegration`` (:412) — the only class in
    this file that runs a real ``asyncio.create_task(worker.run())`` loop and
    feeds it via ``await q.put(req)``.  Unlike that class (which observes a
    verify that stays wedged), this one lets the abort fire and then asserts
    the recovery.

    May already be green after steps 2/4/6/8 — expected and fine.  Its job is
    to be the consolidated cross-surface guard, not a fresh RED signal.
    """

    async def _drive_abort_then_land(
        self,
        git_ops: GitOps,
        config: OrchestratorConfig,
        branch: str,
        filename: str,
        caplog: pytest.LogCaptureFixture,
    ):
        """Run one abort-then-land cycle on a real ``worker.run()`` loop.

        Returns ``(worker, req, outcome, gate, fake_eq, main_sha, snap_after)``.
        """
        from orchestrator.merge_queue import SpeculativeMergeWorker

        wt = await _make_branch_with_file(git_ops, branch, filename, 'x = 1\n')

        fake_eq = _FakeEscalationQueue(open_l1=False)
        q: asyncio.Queue = asyncio.Queue()
        worker = SpeculativeMergeWorker(git_ops, q, escalation_queue=fake_eq)

        # Same small instance-level constants step-5 uses — REAL wall clock,
        # no monkeypatched time.*, and small enough for the 60s per-test
        # timeout (timeout_method='thread').
        worker.VERIFY_ABANDON_POLL_SECS = 0.02
        worker.INFLIGHT_VERIFY_PROGRESS_PROBE_SECS = 0.02
        worker.INFLIGHT_VERIFY_PROGRESS_BUDGET_SECS = 0.2

        req = _make_request(branch, branch, wt, config)
        gate = _HangThenPassVerify()

        with (
            caplog.at_level(logging.WARNING, logger='orchestrator.merge_queue'),
            patch('orchestrator.merge_queue.run_scoped_verification', gate),
        ):
            worker_task = asyncio.create_task(worker.run())
            try:
                await q.put(req)
                # Wait for the DEAD first verify to be entered, so the
                # no-progress budget is genuinely armed before we wait on
                # the recovery.
                await asyncio.wait_for(gate.first_entered.wait(), timeout=20.0)
                outcome = await asyncio.wait_for(req.result, timeout=40.0)
            finally:
                with contextlib.suppress(Exception):
                    await asyncio.wait_for(worker.stop(), timeout=10.0)
                with contextlib.suppress(Exception):
                    await asyncio.wait_for(worker_task, timeout=10.0)

        main_sha = await git_ops.get_main_sha()
        return worker, req, outcome, gate, fake_eq, main_sha, worker.snapshot()

    async def test_abort_then_redispatch_delivers_the_true_outcome_and_a_clean_queue(
        self,
        git_ops: GitOps,
        config: OrchestratorConfig,
        caplog: pytest.LogCaptureFixture,
    ) -> None:
        """The seven surfaces from the task's REGRESSION COVERAGE list."""
        from orchestrator.merge_queue import InflightEntry, ItemLifecycleState

        worker, req, outcome, gate, fake_eq, main_sha, snap = (
            await self._drive_abort_then_land(
                git_ops, config, 'df3082-e2e-selfheal', 'selfheal.py', caplog,
            )
        )
        rid = req.request_id
        messages = [r.getMessage() for r in caplog.records]

        # ── (1) the request ACTUALLY re-enters the live queue and is
        #        re-dispatched — not swallowed by _coalesce_reentrant_drain.
        assert gate.calls == 2, (
            f'expected the abort to be followed by a genuine RE-DISPATCH '
            f'(run_scoped_verification called twice), got {gate.calls} call(s) — '
            f'a single call means the re-queued request was dropped before '
            f're-dispatch'
        )
        drop_warnings = [
            m for m in messages
            if 'dropping duplicate/re-entrant merge submission' in m
        ]
        assert drop_warnings == [], (
            f'the re-queued request must never be coalesce-dropped: {drop_warnings!r}'
        )

        # ── (2) the entry never remains finalizing / position 0 / head_of_line
        #        past its abort point.
        assert worker._finalizing_head_entry() is None, (
            f'a phantom finalize head survived the abort: '
            f'{worker._finalizing_head_entry()!r}'
        )
        finalizing = [e for e in snap['entries'] if e['state'] == 'finalizing']
        assert finalizing == [], (
            f"no snapshot entry may report state='finalizing' once quiescent: "
            f'{finalizing!r}'
        )
        assert snap['depth'] == 0, (
            f"snapshot depth must be 0 once quiescent, got {snap['depth']} "
            f"with entries {snap['entries']!r}"
        )
        assert snap['head_of_line'] is None, (
            f"head_of_line must be None once quiescent, got {snap['head_of_line']!r}"
        )
        assert snap['verify_in_progress'] is None, (
            f"verify_in_progress must be None once quiescent, got "
            f"{snap['verify_in_progress']!r}"
        )
        _live = worker._live_items.get(rid)
        assert not isinstance(_live, InflightEntry), (
            f'a non-TERMINAL InflightEntry survived in _live_items for {rid}: {_live!r}'
        )
        _cur = worker._lifecycle.current(rid)
        assert _cur in (None, ItemLifecycleState.TERMINAL), (
            f'the landed request must end TERMINAL (or be retired), registry '
            f'reads {_cur!r}'
        )

        # ── (3) the waiter's future is NOT resolved to already_merged by the
        #        abort path — it receives the TRUE outcome.
        assert outcome.status == 'done', (
            f'the waiter must receive the TRUE outcome of the re-dispatched '
            f'merge, got {outcome!r}'
        )
        assert outcome.status != 'already_merged', (
            f'the abort path must never fabricate already_merged for the real '
            f'waiter: {outcome!r}'
        )

        # ── (4) occupancy.by_host must NOT report a host busy for the
        #        aborted/requeued entry.  by_host merges the FINALIZE HEAD's
        #        lease in HEAD-FIRST over the _inflight leases, so a phantom
        #        head injects a stale lease for an actually-free host — the
        #        surface that made this class look like a stuck host slot on
        #        three separate days.
        occ = snap['occupancy']
        assert req.task_id not in occ['by_host'].values(), (
            f'the aborted/requeued task must not be reported as occupying a '
            f"host: by_host={occ['by_host']!r}"
        )
        assert occ['hosts_busy'] == 0, (
            f"hosts_busy must be 0 once quiescent, got {occ['hosts_busy']} "
            f"(by_host={occ['by_host']!r})"
        )

        # ── (6) two_layer_invariants + registry/ledger quiescence, against the
        #        REAL post-merge main SHA (never the 'unknown' sentinel).
        _assert_quiescent_registry(worker, main_sha, [req])

        # ── (7) the aborted/requeued entry is ABSENT from the frozen prefix
        #        and never becomes its tip.  _frozen_inflight_entries appends
        #        the finalize head whenever its phase is in
        #        {verifying, gate_reverify, finalizing}, so a phantom head puts
        #        a DEAD merge commit at the frozen-prefix tip and every later
        #        real-verify dispatch mismatches it.  Survivable today only
        #        because _warn_if_verify_base_not_frozen_tip is log-only —
        #        pinning it here is what lets the separate eps=1890
        #        enforcement-flip task proceed without this class re-poisoning it.
        fp = worker.frozen_prefix()
        assert rid not in fp, (
            f'the aborted/requeued request must not sit in the frozen prefix: {fp!r}'
        )
        assert fp == (), (
            f'the frozen prefix must be empty once quiescent, got {fp!r}'
        )
        tip = worker.frozen_prefix_tip(main_sha)
        assert tip == main_sha, (
            f'with an empty frozen prefix the tip must be the REAL main sha '
            f'{main_sha!r}, not a dead merge commit: {tip!r}'
        )

        # ── Plus: no rejected-transition escalation, and no accretion WARNING.
        rejected = [
            e for e in fake_eq.submitted
            if e.category == 'merge_lifecycle_transition_rejected'
        ]
        assert rejected == [], (
            f'a legitimate dead-verify abort must not fire a rejected-transition '
            f'L1: {[(e.category, e.summary) for e in rejected]!r}'
        )
        accretion = [
            m for m in messages
            if 'Invariant violation:' in m or 'extra InflightEntry object(s)' in m
        ]
        assert accretion == [], (
            f'_finalizing_head_entry must not report accretion: {accretion!r}'
        )

    async def test_recovery_is_internal_to_the_abort_path_not_operator_driven(
        self,
        git_ops: GitOps,
        config: OrchestratorConfig,
        caplog: pytest.LogCaptureFixture,
    ) -> None:
        """Recovery must need NO operator action of any kind.

        Pins the task's OPERATIONAL NOTE: a live zombie of this class cannot
        be cleared by ``merge_cancel`` (proved three times in the field),
        because the zombie lives in the ORCHESTRATOR's ``_live_items``, not
        the escalation server's waiter registry — only a process restart
        clears it.  So the recovery MUST be internal to the abort path.

        This test therefore calls NO cancel API, no ``merge_cancel``, no
        ``_cancel_request``, no halt/unhalt, and no ``_retire_item`` — the
        drive is purely: put the request on the queue and wait.
        """
        worker, req, outcome, gate, fake_eq, main_sha, snap = (
            await self._drive_abort_then_land(
                git_ops, config, 'df3082-e2e-nooperator', 'nooperator.py', caplog,
            )
        )

        assert gate.calls == 2, (
            f'the abort path must re-dispatch on its own, with no operator '
            f'intervention; run_scoped_verification saw {gate.calls} call(s)'
        )
        assert outcome.status == 'done', (
            f'the request must land unaided, got {outcome!r}'
        )
        assert not worker._operator_halt.is_set(), (
            'recovery must not depend on (or leave behind) an operator halt'
        )
        assert not worker.is_wip_halted, (
            'recovery must not depend on (or leave behind) a WIP halt'
        )
        assert not req.result.cancelled(), (
            'the waiter must be resolved by the re-dispatch, never cancelled'
        )
        _assert_quiescent_registry(worker, main_sha, [req])
