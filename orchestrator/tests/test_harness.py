"""Tests for Harness external-dep and delivered-check escalation sinks
(task 1580; task 2583 adds the delivered-check sink).

Asserts that:
- After Harness construction, scheduler._callbacks.on_external_dep_block is set
  (non-None) — the constructor-injected callback bundle (task 2235).
- Invoking the callback sets the task to 'blocked' via scheduler.set_task_status.
- Invoking the callback submits exactly one L1 Escalation to the escalation queue.
- A second invocation with an open L1 is deduplicated (no duplicate submission).
- Task 2583 (epsilon): the same shape for on_delivered_check_block, but filing
  a BORN-AT-L2 escalation (severity='critical', level=2) instead of an L1.
"""

from __future__ import annotations

import asyncio
import logging
from collections.abc import Awaitable, Callable
from pathlib import Path
from unittest.mock import AsyncMock, MagicMock, patch

import pytest
from _orch_helpers import wait_responsive
from escalation.models import DeliveredCheckOwner, Escalation
from escalation.queue import EscalationQueue

from orchestrator import delivered_check_owner
from orchestrator.config import OrchestratorConfig, VerifyRunnerConfig
from orchestrator.harness import Harness
from orchestrator.merge_lane.liveness import MergeLivenessConfigError

# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------

def _make_harness(tmp_path: Path) -> Harness:
    """Build a Harness via normal constructor (real Scheduler, no MCP server)."""
    config = OrchestratorConfig(project_root=tmp_path)
    return Harness(config)


def _install_mock_escalation_queue(harness: Harness) -> MagicMock:
    """Attach a MagicMock EscalationQueue to the harness and return it."""
    mock_queue = MagicMock(spec=EscalationQueue)
    mock_queue.has_open_l1.return_value = False   # default: no open L1
    mock_queue.get_by_task.return_value = []      # default: no open L2 (task 2583)
    mock_queue.make_id.return_value = 'esc-test-external-1'
    harness._escalation_queue = mock_queue
    return mock_queue


def _delivered_check_callback(harness: Harness) -> Callable[..., Awaitable[None]]:
    """The wired ``on_delivered_check_block`` callback, asserted installed."""
    callback = harness.scheduler._callbacks.on_delivered_check_block
    assert callback is not None
    return callback


# ---------------------------------------------------------------------------
# TestHarnessExternalDepBlockWiring (step-13 RED / step-14 GREEN)
# ---------------------------------------------------------------------------

class TestHarnessExternalDepBlockWiring:
    """Harness must wire on_external_dep_block into the Scheduler's
    SchedulerCallbacks bundle at construction time (task 2235)."""

    @staticmethod
    def _real_queue_harness(tmp_path: Path) -> tuple[Harness, EscalationQueue]:
        harness = _make_harness(tmp_path)
        queue = EscalationQueue(tmp_path / 'esc')
        harness._escalation_queue = queue
        harness.scheduler.set_task_status = AsyncMock(return_value=True)
        return harness, queue

    @staticmethod
    async def _block(harness: Harness, *, category: str, summary: str, detail: str) -> None:
        callback = harness.scheduler._callbacks.on_external_dep_block
        assert callback is not None
        await callback('42', summary=summary, detail=detail, category=category)

    @classmethod
    async def _block_cancelled(cls, harness: Harness) -> None:
        await cls._block(
            harness,
            category='dependency_discovered',
            summary='EXTERNAL_DEP_CANCELLED: task 42 — dep cancelled',
            detail='dep dark_factory:5 is cancelled',
        )

    @classmethod
    async def _block_resolver_degraded(cls, harness: Harness) -> None:
        await cls._block(
            harness,
            category='infra_issue',
            summary='EXTERNAL_DEP_RESOLVER_DEGRADED: task 42 — resolver degraded for 3 ticks',
            detail='the external-dep resolver failed for 3 consecutive ticks',
        )

    def test_callback_installed_after_construction(self, tmp_path: Path) -> None:
        """After Harness construction, scheduler._callbacks.on_external_dep_block is not None."""
        harness = _make_harness(tmp_path)

        assert harness.scheduler._callbacks.on_external_dep_block is not None, (
            'Harness must wire on_external_dep_block into the SchedulerCallbacks '
            'bundle at Scheduler construction time'
        )

    @pytest.mark.asyncio
    async def test_callback_sets_task_blocked(self, tmp_path: Path) -> None:
        """Invoking the callback sets the task to 'blocked' via scheduler.set_task_status."""
        harness = _make_harness(tmp_path)
        _install_mock_escalation_queue(harness)

        # Patch scheduler.set_task_status to avoid real MCP calls.
        harness.scheduler.set_task_status = AsyncMock(return_value=True)

        await self._block_cancelled(harness)

        harness.scheduler.set_task_status.assert_called_once_with('42', 'blocked')

    @pytest.mark.asyncio
    async def test_callback_submits_l1_escalation(self, tmp_path: Path) -> None:
        """Invoking the callback submits exactly one L1 Escalation to the queue."""
        harness = _make_harness(tmp_path)
        mock_queue = _install_mock_escalation_queue(harness)
        harness.scheduler.set_task_status = AsyncMock(return_value=True)

        await self._block_cancelled(harness)

        mock_queue.submit.assert_called_once()
        submitted: Escalation = mock_queue.submit.call_args.args[0]
        assert submitted.task_id == '42', (
            f'Escalation task_id must be "42"; got {submitted.task_id!r}'
        )
        assert submitted.level == 1, (
            f'Must file an L1 escalation; got level={submitted.level}'
        )
        assert 'EXTERNAL_DEP_CANCELLED' in submitted.summary, (
            f'Escalation summary must carry the prefix; got {submitted.summary!r}'
        )

    @pytest.mark.asyncio
    async def test_callback_no_escalation_queue_is_safe(self, tmp_path: Path) -> None:
        """With no escalation queue installed, callback must not raise."""
        harness = _make_harness(tmp_path)
        harness._escalation_queue = None
        harness.scheduler.set_task_status = AsyncMock(return_value=True)

        # Must not raise even without a queue.
        assert harness.scheduler._callbacks.on_external_dep_block is not None
        await harness.scheduler._callbacks.on_external_dep_block(
            '99',
            summary='EXTERNAL_DEP_UNRESOLVED: ...',
            detail='detail',
            category='dependency_discovered',
        )

    @pytest.mark.asyncio
    async def test_callback_files_a_new_root_cause_past_an_open_l1_of_another_category(
        self, tmp_path: Path,
    ) -> None:
        """The scheduler's resolver-degraded infra_issue L1 must not hide a cancelled dep."""
        harness, queue = self._real_queue_harness(tmp_path)

        await self._block_resolver_degraded(harness)
        await self._block_cancelled(harness)

        pending_l1s = queue.get_by_task('42', status='pending', level=1)
        by_category = {e.category: e for e in pending_l1s}
        assert len(pending_l1s) == 2, f'Expected one L1 per cause; got {pending_l1s!r}'
        assert 'EXTERNAL_DEP_RESOLVER_DEGRADED' in by_category['infra_issue'].summary
        assert 'EXTERNAL_DEP_CANCELLED' in by_category['dependency_discovered'].summary

    @pytest.mark.asyncio
    async def test_callback_still_deduplicates_within_its_category(self, tmp_path: Path) -> None:
        """A repeat block for the same category files no second L1 (real queue, real dedupe)."""
        harness, queue = self._real_queue_harness(tmp_path)

        await self._block_cancelled(harness)
        await self._block_cancelled(harness)

        pending_l1s = queue.get_by_task('42', status='pending', level=1)
        assert len(pending_l1s) == 1, f'Expected exactly 1 L1 after dedup; got {pending_l1s!r}'

    @pytest.mark.asyncio
    async def test_failed_status_write_still_files_and_names_the_gate(
        self, tmp_path: Path, caplog: pytest.LogCaptureFixture,
    ) -> None:
        """A raising status write is logged with its category and summary; the L1 still files."""
        harness, queue = self._real_queue_harness(tmp_path)
        harness.scheduler.set_task_status = AsyncMock(side_effect=RuntimeError('fused-memory down'))
        caplog.set_level(logging.WARNING, logger='orchestrator.harness')

        await self._block_cancelled(harness)

        assert len(queue.get_by_task('42', status='pending', level=1)) == 1
        status_warnings = [
            r.getMessage() for r in caplog.records if 'set_task_status raised' in r.getMessage()
        ]
        assert len(status_warnings) == 1, status_warnings
        assert 'dependency_discovered' in status_warnings[0]
        assert 'EXTERNAL_DEP_CANCELLED' in status_warnings[0]


# ---------------------------------------------------------------------------
# TestHarnessDeliveredCheckBlockWiring (task 2583 — step-13 RED / step-14 GREEN)
# ---------------------------------------------------------------------------

class TestHarnessDeliveredCheckBlockWiring:
    """Harness must wire on_delivered_check_block into the Scheduler's
    SchedulerCallbacks bundle at construction time (task 2583, epsilon).

    Mirrors TestHarnessExternalDepBlockWiring, but the filed escalation is
    BORN AT L2 (severity='critical', level=2, agent_role=
    'orchestrator-scheduler' — a harness-sentinel role) instead of L1, and
    dedupe is scoped to level=2 + agent_role rather than has_open_l1,
    which reads only L1s.
    """

    _SUMMARY = (
        "DEP_CAPABILITY_NOT_DELIVERED: task 42 — dep 5 done but check "
        "'cap' fails on main@abc123456789"
    )

    def test_callback_installed_after_construction(self, tmp_path: Path) -> None:
        """After Harness construction, scheduler._callbacks.on_delivered_check_block
        is not None."""
        harness = _make_harness(tmp_path)

        assert harness.scheduler._callbacks.on_delivered_check_block is not None, (
            'Harness must wire on_delivered_check_block into the SchedulerCallbacks '
            'bundle at Scheduler construction time'
        )

    @pytest.mark.asyncio
    async def test_callback_sets_task_blocked(self, tmp_path: Path) -> None:
        """Invoking the callback sets the task to 'blocked' via scheduler.set_task_status."""
        harness = _make_harness(tmp_path)
        _install_mock_escalation_queue(harness)
        harness.scheduler.set_task_status = AsyncMock(return_value=True)

        await _delivered_check_callback(harness)(
            '42',
            summary=self._SUMMARY,
            detail='detail',
            category='dependency_capability',
        )

        harness.scheduler.set_task_status.assert_called_once_with('42', 'blocked')

    @pytest.mark.asyncio
    async def test_callback_submits_l2_escalation(self, tmp_path: Path) -> None:
        """Invoking the callback submits exactly one born-at-L2 Escalation."""
        harness = _make_harness(tmp_path)
        mock_queue = _install_mock_escalation_queue(harness)
        harness.scheduler.set_task_status = AsyncMock(return_value=True)

        await _delivered_check_callback(harness)(
            '42',
            summary=self._SUMMARY,
            detail='detail',
            category='dependency_capability',
        )

        mock_queue.submit.assert_called_once()
        submitted: Escalation = mock_queue.submit.call_args.args[0]
        assert submitted.task_id == '42', (
            f'Escalation task_id must be "42"; got {submitted.task_id!r}'
        )
        assert submitted.level == 2, (
            f'Must file a born-at-L2 escalation; got level={submitted.level}'
        )
        assert submitted.severity == 'critical', (
            f'Born-at-L2 requires a BORN_AT_L2_SEVERITIES severity; got {submitted.severity!r}'
        )
        assert submitted.agent_role == 'orchestrator-scheduler', (
            f'Must file with a harness-sentinel agent_role; got {submitted.agent_role!r}'
        )
        assert submitted.category == 'dependency_capability', (
            f'Escalation category must be dependency_capability; got {submitted.category!r}'
        )
        assert 'DEP_CAPABILITY_NOT_DELIVERED' in submitted.summary, (
            f'Escalation summary must carry the prefix; got {submitted.summary!r}'
        )

    @pytest.mark.asyncio
    async def test_callback_deduplicates_on_open_l2(self, tmp_path: Path) -> None:
        """Second invocation with an open L2 must NOT submit a duplicate escalation,
        and must not pay for the owner's sidecar read either."""
        harness = _make_harness(tmp_path)
        mock_queue = _install_mock_escalation_queue(harness)
        harness.scheduler.set_task_status = AsyncMock(return_value=True)
        owner: DeliveredCheckOwner = {
            'task_id': '5',
            'check_name': 'cap',
            'sidecar': {'path': 'docs/x.capability-manifest.yaml', 'on_main': 'unverified'},
        }
        verified: list[DeliveredCheckOwner] = []

        def _record_verify(owner_arg: DeliveredCheckOwner, **_kwargs) -> DeliveredCheckOwner:
            verified.append(owner_arg)
            return owner_arg

        callback = _delivered_check_callback(harness)
        with patch.object(delivered_check_owner, 'verify_check_owner', _record_verify):
            # First call — no open L2 yet → submits
            mock_queue.get_by_task.return_value = []
            await callback(
                '42',
                summary=self._SUMMARY,
                detail='detail',
                category='dependency_capability',
                owner=owner,
            )

            # Second call — an open L2 now exists → must NOT submit again
            mock_queue.get_by_task.return_value = [MagicMock()]
            await callback(
                '42',
                summary=self._SUMMARY,
                detail='detail',
                category='dependency_capability',
                owner=owner,
            )

        assert mock_queue.submit.call_count == 1, (
            f'Should deduplicate: submit called {mock_queue.submit.call_count} times '
            f'(expected exactly 1)'
        )
        assert verified == [owner], 'the deduped call must not read the owner sidecar'
        # Dedupe must be scoped to level=2 + agent_role — not has_open_l1,
        # which reads only L1s and so cannot see the born-at-L2 record.
        mock_queue.get_by_task.assert_any_call(
            '42', status='pending', level=2, agent_role='orchestrator-scheduler'
        )

    @pytest.mark.asyncio
    async def test_callback_no_escalation_queue_is_safe(self, tmp_path: Path) -> None:
        """With no escalation queue installed, callback must not raise."""
        harness = _make_harness(tmp_path)
        harness._escalation_queue = None
        harness.scheduler.set_task_status = AsyncMock(return_value=True)

        # Must not raise even without a queue.
        await _delivered_check_callback(harness)(
            '99',
            summary=self._SUMMARY,
            detail='detail',
            category='dependency_capability',
        )

    @pytest.mark.asyncio
    async def test_callback_records_and_renders_the_check_owner(self, tmp_path: Path) -> None:
        """Called with the failed descriptor and the owner (sidecar None, so
        no git read), the filed record carries the owner as data and the
        owner block in its detail."""
        harness = _make_harness(tmp_path)
        mock_queue = _install_mock_escalation_queue(harness)
        harness.scheduler.set_task_status = AsyncMock(return_value=True)
        owner: DeliveredCheckOwner = {'task_id': '5', 'check_name': 'cap', 'sidecar': None}

        await _delivered_check_callback(harness)(
            '42',
            summary=self._SUMMARY,
            detail='detail',
            category='dependency_capability',
            check={'name': 'cap', 'kind': 'path', 'expect': 'present', 'paths': ['x.py']},
            owner=owner,
        )

        submitted: Escalation = mock_queue.submit.call_args.args[0]
        assert submitted.delivered_check_owner == owner
        assert 'Check owner: task 5' in submitted.detail
        assert submitted.detail.splitlines()[-1].endswith(
            'set task 42 back to pending once the check passes on main.'
        )

    @pytest.mark.asyncio
    async def test_legacy_call_keeps_the_re_pend_recipe_without_an_owner(
        self, tmp_path: Path
    ) -> None:
        """A caller that predates the owner kwargs still gets Resolved-6's
        manual re-pend recipe, and the record carries no owner."""
        harness = _make_harness(tmp_path)
        mock_queue = _install_mock_escalation_queue(harness)
        harness.scheduler.set_task_status = AsyncMock(return_value=True)

        await _delivered_check_callback(harness)(
            '42',
            summary=self._SUMMARY,
            detail='detail',
            category='dependency_capability',
        )

        submitted: Escalation = mock_queue.submit.call_args.args[0]
        assert submitted.delivered_check_owner is None
        assert submitted.detail == (
            'detail\nBlocked dependents are NOT auto-re-pended — set task 42 back to '
            'pending once the check passes on main.'
        )


# ---------------------------------------------------------------------------
# TestHarnessMergeLivenessGuard (step-7 RED / step-8 GREEN)
# ---------------------------------------------------------------------------

class TestHarnessMergeLivenessGuard:
    """Harness._start_merge_worker must be FAIL-CLOSED on an over-budget liveness config.

    Heartbeat-floor model (task-1729 β):
      worst_case = _HEARTBEAT_POLL_S × TOUCH_MISS_TOLERANCE = 600 s (invariant)
      threshold  = safety_factor × INFLIGHT_MERGE_WORKTREE_LIVENESS_SECS = 8100 s
      safe = 600 < 8100 → True  (cold timeout is irrelevant; K is irrelevant).

    Over-budget lever: monkeypatch TOUCH_MISS_TOLERANCE to a large value so that
    floor = 30 × 1000 = 30000 ≥ 8100 → refuses startup.
    """

    @pytest.mark.asyncio
    async def test_cold_9000_now_constructs_cleanly(self, tmp_path: Path) -> None:
        """cold_timeout=9000 (refused by old model) must construct cleanly under heartbeat model.

        Old model: worst_case = 1 × 9000 = 9000 ≥ 8100 → refused.
        Heartbeat model: worst_case = 600 < 8100 → safe → worker constructed.
        """
        config = OrchestratorConfig(
            project_root=tmp_path,
            merge_verify_cold_command_timeout_secs=9000,
        )
        harness = Harness(config)

        # Must NOT raise MergeLivenessConfigError under heartbeat model
        await harness._start_merge_worker()

        assert harness._merge_worker is not None

    @pytest.mark.asyncio
    async def test_in_budget_config_constructs_worker(self, tmp_path: Path) -> None:
        """Default/in-budget config (cold_timeout=7200 default, bound=1) must NOT raise."""
        config = OrchestratorConfig(project_root=tmp_path)
        harness = Harness(config)

        # Must NOT raise MergeLivenessConfigError
        await harness._start_merge_worker()

        assert harness._merge_worker is not None

    @pytest.mark.asyncio
    async def test_crash_config_constructs_worker(self, tmp_path: Path) -> None:
        """K=2 config (one enabled runner) + cold_timeout=7200 must NOT raise.

        Regression-lock: stays green under BOTH the old per-host model
        (per_host = ceil(2/2) = 1 → worst_case = 7200 < 8100 → safe) and the
        new heartbeat-floor model (floor = _HEARTBEAT_POLL_S × TOUCH_MISS_TOLERANCE
        = 600 < 8100 → safe).  Guards against a future refactor accidentally
        re-coupling K to the liveness check.
        """
        r1 = VerifyRunnerConfig(
            name='r1', ssh_host='h1.local', git_remote='remote1', df_checkout_path='/df',
        )
        config = OrchestratorConfig(
            project_root=tmp_path,
            verify_runners=[r1],                           # → K = 1 + 1 = 2
            merge_verify_cold_command_timeout_secs=7200,   # shipped default; safe under both models
        )
        harness = Harness(config)

        # Must NOT raise MergeLivenessConfigError under either model
        await harness._start_merge_worker()

        assert harness._merge_worker is not None

    @pytest.mark.asyncio
    async def test_over_budget_via_low_tolerance_raises(self, tmp_path: Path) -> None:
        """Monkeypatching TOUCH_MISS_TOLERANCE=1000 → floor=30000 ≥ 8100 → raises.

        Demonstrates that the heartbeat-floor lever is TOUCH_MISS_TOLERANCE (not
        cold_timeout or K).  A test can force an over-budget verdict by raising
        the tolerance constant so floor = _HEARTBEAT_POLL_S × 1000 = 30000 > 8100.
        """
        config = OrchestratorConfig(project_root=tmp_path)
        harness = Harness(config)

        with patch('orchestrator.merge_lane.liveness.TOUCH_MISS_TOLERANCE', 1000), pytest.raises(MergeLivenessConfigError):
            await harness._start_merge_worker()


# ---------------------------------------------------------------------------
# TestHarnessMergeVerifySurvivorBarrierWiring (task 2828, step-09 RED / step-10 GREEN)
# ---------------------------------------------------------------------------


class TestHarnessMergeVerifySurvivorBarrierWiring:
    """Harness._launch_merge_worker_loop must invoke the startup survivor barrier
    (GitOps.reap_merge_verify_survivors) ONCE, BEFORE the merge-worker loop is
    scheduled — so a previous run's orphaned verify subtree under _merge-verify
    is reaped before the worker's first reset_persistent_merge_worktree can
    clobber a live tree (task 2828, limb 1).  The call must be fail-open: a
    raising barrier must NOT crash startup.
    """

    @pytest.mark.asyncio
    async def test_barrier_awaited_once_before_worker_loop(
        self, tmp_path: Path,
    ) -> None:
        """reap_merge_verify_survivors is awaited exactly once and strictly
        before the worker's run() loop begins."""
        config = OrchestratorConfig(project_root=tmp_path)
        harness = Harness(config)

        call_order: list[str] = []
        worker_started = asyncio.Event()

        async def _record_reap(*_a, **_k):
            call_order.append('reap')
            return True

        reap_mock = AsyncMock(side_effect=_record_reap)
        harness.git_ops.reap_merge_verify_survivors = reap_mock  # type: ignore[method-assign]

        async def _noop_run(self_w):  # type: ignore[no-untyped-def]
            call_order.append('worker_loop')
            worker_started.set()

        try:
            with patch(
                'orchestrator.merge_lane.worker.SpeculativeMergeWorker.run', _noop_run,
            ):
                await harness._start_merge_worker()
                await harness._launch_merge_worker_loop()
                await asyncio.wait_for(worker_started.wait(), timeout=5)

            reap_mock.assert_awaited_once()
            assert call_order == ['reap', 'worker_loop'], (
                'the survivor barrier must run to completion BEFORE the merge '
                f'worker loop starts; got order {call_order}'
            )
        finally:
            await harness._stop_merge_worker()

    @pytest.mark.asyncio
    async def test_barrier_failure_is_fail_open(self, tmp_path: Path) -> None:
        """A raising barrier must NOT propagate — the worker loop still launches.

        Mirrors the enforce_merge_liveness_margin generic-except precedent: the
        barrier is best-effort, so a failure is logged and swallowed rather than
        aborting the merge-worker launch.
        """
        config = OrchestratorConfig(project_root=tmp_path)
        harness = Harness(config)

        reap_mock = AsyncMock(side_effect=RuntimeError('scan blew up'))
        harness.git_ops.reap_merge_verify_survivors = reap_mock  # type: ignore[method-assign]
        worker_started = asyncio.Event()

        async def _noop_run(self_w):  # type: ignore[no-untyped-def]
            worker_started.set()

        try:
            with patch(
                'orchestrator.merge_lane.worker.SpeculativeMergeWorker.run', _noop_run,
            ):
                await harness._start_merge_worker()
                # Must NOT raise despite the barrier raising.
                await harness._launch_merge_worker_loop()
                await wait_responsive(
                    worker_started.wait(),
                    label='the merge-worker loop launches despite a failing barrier',
                )

            reap_mock.assert_awaited_once()
        finally:
            await harness._stop_merge_worker()
