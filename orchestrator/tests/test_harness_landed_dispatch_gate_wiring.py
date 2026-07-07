"""Tests for the harness's landed-outbox dispatch-gate wiring
(task 2156, W1 δ — PRD merge-queue-reliability §8.2 SD-1, boundary B5).

Covers:
  step-5  (RED)  ``Harness._landed_dispatch_gate`` None-guard + delegation
                 to ``merge_queue.reconcile_landed_task``, plus the
                 ``__init__``-time install of
                 ``scheduler._landed_outbox_gate`` to the bound method.

Mirrors test_merge_queue_landed_reconciler.py's
``TestHarnessReconcileLandedOutboxWiring`` / ``_build_harness`` exactly —
same bare-harness construction helper, same None-guard-and-delegate
assertion style, same patch-the-module-fn convention.
"""

from __future__ import annotations

from unittest.mock import AsyncMock, MagicMock, patch

import pytest

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
        worker's outbox, and returns its result verbatim.
        """
        h = _build_harness(mock_orch_config)
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
