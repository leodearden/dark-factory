"""``Harness._reconcile_landed_outbox``'s None-guard, on the startup path.

Task 5030 (PRD ``plans/merge-lane-quality-prd.md`` task γ7) deleted
``test_merge_queue_landed_reconciler.py::TestHarnessReconcileLandedOutboxWiring``
as part of migrating that file off the lane's internals.  Its
delegation/arming half survives, in
``test_harness_landed_dispatch_gate_wiring.py::TestHarnessArmsDeliveredChecksGuard``;
its None-guard half did not, and nothing else covered it -- that file's
``test_none_worker_is_noop`` targets the sibling ``_landed_dispatch_gate``.

That guard runs UNCONDITIONALLY at startup (``harness.py::Harness.start``
calls this method with no precondition of its own), so losing it is an
``AttributeError`` at boot for any deployment with the merge worker disabled --
not a degraded feature, a dead orchestrator.

WHY A FILE OF ITS OWN.  The natural home is beside its
``_landed_dispatch_gate`` twin in ``test_harness_landed_dispatch_gate_wiring.py``,
which already defines the ``_build_harness`` helper this imports.  Task 5030
holds no lock on that file, so the test lands here instead and imports the
helper rather than copying it -- the same cross-test-module import
``test_merge_speculation.py`` makes of ``test_merge_queue_concurrent_verify``.
Folding this module into its twin is a one-move edit for whoever next holds
both locks.

Deliberately imports NO merge-lane module: ``merge_lane_metrics``'s test sweep
enrols a file on exactly that condition, and a harness test reaching a
harness-private attribute would enrol at a non-zero ``private_reads`` and push
the ratchet's derived total up for coverage that has nothing to do with the
lane seam.
"""

from __future__ import annotations

from unittest.mock import AsyncMock, patch

import pytest
from test_harness_landed_dispatch_gate_wiring import _build_harness


@pytest.mark.asyncio
class TestHarnessReconcileLandedOutboxNoneGuard:
    """No merge worker (or no bound outbox) means there is nothing to reconcile."""

    async def test_none_worker_is_noop(self, mock_orch_config) -> None:
        """A disabled merge worker must not reach the module-level reconciler.

        Mirrors ``_landed_dispatch_gate``'s None-guard test exactly: nothing
        raised, and the delegate never called -- which is also what proves the
        guard returned rather than merely surviving.
        """
        h = _build_harness(mock_orch_config)
        h._merge_worker = None

        with patch(
            'orchestrator.harness.reconcile_landed_outbox', new=AsyncMock(),
        ) as mock_reconcile:
            assert await h._reconcile_landed_outbox() is None

        mock_reconcile.assert_not_called()

    async def test_worker_without_a_bound_outbox_is_noop(
        self, mock_orch_config,
    ) -> None:
        """A merge worker present but carrying no LandedOutbox is the second
        arm of the same guard, and fails the same way if it is dropped."""
        h = _build_harness(mock_orch_config)
        h._merge_worker = AsyncMock()
        h._merge_worker._landed_outbox = None

        with patch(
            'orchestrator.harness.reconcile_landed_outbox', new=AsyncMock(),
        ) as mock_reconcile:
            assert await h._reconcile_landed_outbox() is None

        mock_reconcile.assert_not_called()
