"""Tests for the harness's landed-outbox dispatch-gate wiring
(task 2156, W1 δ — PRD merge-queue-reliability §8.2 SD-1, boundary B5).

Covers:
  step-5  (RED)  ``Harness._landed_dispatch_gate`` None-guard + delegation
                 to ``merge_queue.reconcile_landed_task``, plus the
                 ``__init__``-time install of
                 ``scheduler._landed_outbox_gate`` to the bound method.

  3057/step-15   Both harness callers ARM the merge_queue delivered-checks
                 guard from live config, and the withheld count is visible
                 in the operator-facing INFO summary.

Mirrors test_merge_queue_landed_reconciler.py's
``TestHarnessReconcileLandedOutboxWiring`` / ``_build_harness`` exactly —
same bare-harness construction helper, same None-guard-and-delegate
assertion style, same patch-the-module-fn convention.
"""

from __future__ import annotations

import logging
from unittest.mock import AsyncMock, MagicMock, patch

import pytest

from orchestrator.config import DeliveredChecksConfig
from orchestrator.harness import Harness


def _build_harness(mock_orch_config) -> Harness:
    """Construct a Harness with heavy constructors patched out.

    Mirrors test_merge_queue_landed_reconciler._build_harness /
    test_harness_orphan_merge_reaper_wiring._build_harness's bare-harness
    construction helper.
    """
    mock_orch_config.max_concurrent_tasks = 2
    mock_orch_config.fused_memory.project_id = 'test'

    with patch('orchestrator.harness.McpLifecycle'), \
         patch('orchestrator.harness.Scheduler'), \
         patch('orchestrator.harness.BriefingAssembler'):
        return Harness(mock_orch_config)


@pytest.mark.asyncio
class TestHarnessLandedDispatchGateWiring:
    """Harness._landed_dispatch_gate: None-guard + delegation to the module fn."""

    async def test_none_worker_is_noop(self, mock_orch_config) -> None:
        """RED until step-6 adds Harness._landed_dispatch_gate.

        Mirrors _reconcile_landed_outbox's None-guard: no merge worker means
        nothing to consult — must not touch git_ops/scheduler, and must
        fail-open (return False, i.e. do not gate dispatch).
        """
        h = _build_harness(mock_orch_config)
        h._merge_worker = None

        with patch(
            'orchestrator.harness.reconcile_landed_task', new=AsyncMock(),
        ) as mock_reconcile:
            result = await h._landed_dispatch_gate('Z')

        assert result is False
        mock_reconcile.assert_not_called()

    async def test_delegates_with_worker_landed_outbox_git_ops_and_scheduler(
        self, mock_orch_config,
    ) -> None:
        """RED until step-6 adds Harness._landed_dispatch_gate.

        When a merge worker with a bound LandedOutbox is present, the
        harness delegates to the module-level reconcile_landed_task with
        the task_id plus the harness's own git_ops/scheduler and the
        worker's outbox, and returns its result verbatim. Also threads the
        harness's shared ProvenanceConflictSink (task 2677) and, since task
        3057, the three params that ARM the RC-2 delivered-checks guard.
        """
        h = _build_harness(mock_orch_config)
        h.config.delivered_checks = DeliveredChecksConfig(
            enabled=True, check_timeout_secs=11.0,
        )
        worker = MagicMock()
        worker._landed_outbox = MagicMock()
        h._merge_worker = worker

        with patch(
            'orchestrator.harness.reconcile_landed_task',
            new=AsyncMock(return_value=True),
        ) as mock_reconcile:
            result = await h._landed_dispatch_gate('Z')

        assert result is True
        mock_reconcile.assert_awaited_once_with(
            'Z', git_ops=h.git_ops, scheduler=h.scheduler,
            outbox=worker._landed_outbox,
            provenance_conflict_sink=h._provenance_conflict_sink,
            project_root=str(h.config.project_root),
            check_timeout_secs=11.0,
            delivered_checks_enabled=True,
        )


@pytest.mark.asyncio
class TestHarnessLandedDispatchGateInstall:
    """Harness.__init__ wires scheduler._landed_outbox_gate to the bound method."""

    async def test_scheduler_attribute_wired_to_bound_method(
        self, mock_orch_config,
    ) -> None:
        """RED until step-6 installs the wiring in Harness.__init__.

        Same bound-method equality idiom as the warm-base probe wiring test
        (test_harness_warm_lane_wiring.py) — a freshly-accessed bound method
        is a new wrapper object each time, so ``==`` (not ``is``) is the
        correct comparison; MagicMock retains the exact object assigned
        during __init__, so this is a genuine RED (unset/auto-vivified Mock
        != bound method, or AttributeError while the method doesn't exist
        at all) before Harness.__init__ wires the callable.
        """
        h = _build_harness(mock_orch_config)

        assert h.scheduler._landed_outbox_gate == h._landed_dispatch_gate, (
            'Harness must wire scheduler._landed_outbox_gate = '
            'harness._landed_dispatch_gate after construction'
        )


# ---------------------------------------------------------------------------
# Task 3057 step-15 (RED) — seam 8 WIRING.
#
# step-14 gave reconcile_landed_row/_task/_outbox `None` defaults that leave
# the delivered-checks guard UNARMED. That is deliberate (every pre-existing
# caller stays byte-identical) but it also means the merge_queue work lands
# fully DEAD unless the harness explicitly arms it from live config. An
# unarmed guard is exactly the "coverage on paper" failure mode this task
# exists to prevent, so the arming is pinned here as its own contract.
# ---------------------------------------------------------------------------


def _armed_harness(mock_orch_config, *, enabled: bool = True) -> tuple[Harness, MagicMock]:
    """Bare harness with a live ``delivered_checks`` section and a merge worker."""
    h = _build_harness(mock_orch_config)
    h.config.delivered_checks = DeliveredChecksConfig(
        enabled=enabled, check_timeout_secs=11.0,
    )
    worker = MagicMock()
    worker._landed_outbox = MagicMock()
    h._merge_worker = worker
    return h, worker


@pytest.mark.asyncio
class TestHarnessArmsDeliveredChecksGuard:
    """Both harness callers must ARM the merge_queue RC-2 capability guard."""

    async def test_reconcile_landed_outbox_arms_the_guard(
        self, mock_orch_config,
    ) -> None:
        h, worker = _armed_harness(mock_orch_config)

        with patch(
            'orchestrator.harness.reconcile_landed_outbox',
            new=AsyncMock(return_value={}),
        ) as mock_reconcile:
            await h._reconcile_landed_outbox()

        assert mock_reconcile.await_args is not None
        kwargs = mock_reconcile.await_args.kwargs
        assert kwargs['project_root'] == str(h.config.project_root)
        assert kwargs['check_timeout_secs'] == 11.0
        assert kwargs['delivered_checks_enabled'] is True

    async def test_landed_dispatch_gate_arms_the_guard(
        self, mock_orch_config,
    ) -> None:
        h, worker = _armed_harness(mock_orch_config)

        with patch(
            'orchestrator.harness.reconcile_landed_task',
            new=AsyncMock(return_value=False),
        ) as mock_reconcile:
            await h._landed_dispatch_gate('Z')

        assert mock_reconcile.await_args is not None
        kwargs = mock_reconcile.await_args.kwargs
        assert kwargs['project_root'] == str(h.config.project_root)
        assert kwargs['check_timeout_secs'] == 11.0
        assert kwargs['delivered_checks_enabled'] is True

    @pytest.mark.parametrize('method,module_fn', [
        ('_reconcile_landed_outbox', 'reconcile_landed_outbox'),
        ('_landed_dispatch_gate', 'reconcile_landed_task'),
    ])
    async def test_kill_switch_is_forwarded_not_short_circuited(
        self, mock_orch_config, method: str, module_fn: str,
    ) -> None:
        """``enabled=False`` must reach the guard, never be re-implemented here.

        The fleet-wide kill switch lives in exactly ONE place (the shared
        helper). A harness that short-circuited locally would silently
        diverge from the other ten seams on the next hot reload.
        """
        h, _worker = _armed_harness(mock_orch_config, enabled=False)
        args = () if method == '_reconcile_landed_outbox' else ('Z',)

        with patch(
            f'orchestrator.harness.{module_fn}',
            new=AsyncMock(return_value={} if not args else False),
        ) as mock_reconcile:
            await getattr(h, method)(*args)

        assert mock_reconcile.await_args is not None
        assert mock_reconcile.await_args.kwargs['delivered_checks_enabled'] is False
        # ...and still ARMED: the kill switch is the helper's business, so the
        # path must stay wired rather than falling back to the None default.
        assert mock_reconcile.await_args.kwargs['project_root'] is not None
        assert mock_reconcile.await_args.kwargs['check_timeout_secs'] == 11.0


@pytest.mark.asyncio
class TestReconcileLandedOutboxSummaryVisibility:
    """A withheld hollow-done must be VISIBLE, not absorbed into ``skipped``."""

    async def test_summary_includes_the_withheld_count(
        self, mock_orch_config, caplog,
    ) -> None:
        h, _worker = _armed_harness(mock_orch_config)
        report = {
            'pruned_not_landed': 0, 'marked_done': 1, 'already_done_pruned': 0,
            'skipped': 0, 'stale_conflict': 0, 'delivered_checks_withheld': 3,
            'errors': 0,
        }

        with caplog.at_level(logging.INFO, logger='orchestrator.harness'), patch(
            'orchestrator.harness.reconcile_landed_outbox',
            new=AsyncMock(return_value=report),
        ):
            await h._reconcile_landed_outbox()

        assert 'delivered_checks_withheld=3' in caplog.text

    async def test_report_missing_the_new_key_still_logs(
        self, mock_orch_config, caplog,
    ) -> None:
        """An older/foreign reconciler's report dict must not raise.

        Preserves the existing ``report.get(key, 0)`` idiom for the new key —
        a KeyError here would abort startup reconciliation over a telemetry
        field.
        """
        h, _worker = _armed_harness(mock_orch_config)

        with caplog.at_level(logging.INFO, logger='orchestrator.harness'), patch(
            'orchestrator.harness.reconcile_landed_outbox',
            new=AsyncMock(return_value={'marked_done': 1}),
        ):
            await h._reconcile_landed_outbox()

        assert 'delivered_checks_withheld=0' in caplog.text
