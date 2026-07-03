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
import contextlib
import logging
import time
from pathlib import Path
from unittest.mock import MagicMock, patch

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
            SpeculativeItem,
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

        item = SpeculativeItem(
            request=req,
            merge_result=MagicMock(),
            merge_wt=tmp_path / 'merge_wt',
            base_sha='deadbeef',
            speculative=False,
            skip_verify=False,
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
