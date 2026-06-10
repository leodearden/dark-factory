"""Tests for K / num_hosts wiring in harness._start_merge_worker (task 1716).

Verifies that _start_merge_worker derives K from
``1 + len(config.enabled_verify_runners)`` (not from the pinned
``_MERGE_AHEAD_BOUND``), and passes ``num_hosts=K`` to
``enforce_persistent_worktree_serial_lane``.

Modeled on test_harness_merge_registry_wiring.py:154
TestStartMergeWorkerCallsLivenessGuard.

RED until step-12 replaces ``_k = _MERGE_AHEAD_BOUND`` with
``_k = 1 + len(self.config.enabled_verify_runners)`` and passes
``num_hosts=_k`` to ``enforce_persistent_worktree_serial_lane``.
"""

from __future__ import annotations

from unittest.mock import AsyncMock, MagicMock, patch

import pytest

from orchestrator.config import VerifyRunnerConfig
from orchestrator.harness import Harness

# ---------------------------------------------------------------------------
# Harness builder
# ---------------------------------------------------------------------------


def _build_harness(config) -> Harness:
    """Construct a Harness with heavy constructors patched out."""
    with patch('orchestrator.harness.McpLifecycle'), \
         patch('orchestrator.harness.Scheduler'), \
         patch('orchestrator.harness.BriefingAssembler'):
        return Harness(config)


# ---------------------------------------------------------------------------
# Tests
# ---------------------------------------------------------------------------


@pytest.mark.asyncio
class TestHarnessKFromConfig:
    """K and num_hosts are both derived from 1 + len(config.enabled_verify_runners).

    Patching strategy
    -----------------
    _start_merge_worker does a LOCAL import:
        from orchestrator.merge_queue import (
            _MERGE_AHEAD_BOUND, SpeculativeMergeWorker,
            enforce_merge_liveness_margin, enforce_persistent_worktree_serial_lane,
        )
    Patching the SOURCE module attribute (``orchestrator.merge_queue.*``) is
    what the local import picks up at call time; patching ``orchestrator.harness.*``
    would have no effect because no module-level binding exists there.
    """

    async def _run_start_worker(self, config):
        """Helper: build harness, patch, call _start_merge_worker, return mocks."""
        h = _build_harness(config)

        with patch('orchestrator.merge_queue.SpeculativeMergeWorker') as mock_smw_cls, \
             patch.object(h, '_build_service_restart_coordinator',
                          return_value=MagicMock()), \
             patch('asyncio.create_task', return_value=MagicMock()), \
             patch('orchestrator.merge_queue.enforce_merge_liveness_margin') as mock_liveness, \
             patch('orchestrator.merge_queue.enforce_persistent_worktree_serial_lane') as mock_serial:

            mock_smw = MagicMock()
            mock_smw.run = AsyncMock(return_value=None)
            mock_smw_cls.return_value = mock_smw

            await h._start_merge_worker()

        return mock_smw_cls, mock_liveness, mock_serial

    async def test_k3_with_two_enabled_runners(self, mock_orch_config) -> None:
        """Two enabled verify_runners → K=3.

        Asserts:
          (a) SpeculativeMergeWorker(speculation_depth=3)
          (b) enforce_persistent_worktree_serial_lane(..., merge_ahead_bound=3, num_hosts=3)
          (c) enforce_merge_liveness_margin(..., merge_ahead_bound=3)

        RED until step-12 wires K from config.enabled_verify_runners.
        """
        r1 = VerifyRunnerConfig(name='r1', ssh_host='h1.local', git_remote='remote1')
        r2 = VerifyRunnerConfig(name='r2', ssh_host='h2.local', git_remote='remote2')
        mock_orch_config.enabled_verify_runners = [r1, r2]
        mock_orch_config.max_concurrent_tasks = 2
        mock_orch_config.fused_memory.project_id = 'test'

        mock_smw_cls, mock_liveness, mock_serial = await self._run_start_worker(
            mock_orch_config
        )

        # (a) SpeculativeMergeWorker called with speculation_depth=3
        smw_kwargs = mock_smw_cls.call_args[1]
        assert smw_kwargs['speculation_depth'] == 3, (
            f'Expected speculation_depth=3 (K=1+2), got {smw_kwargs.get("speculation_depth")!r}; '
            'step-12 wires _k from config.enabled_verify_runners'
        )

        # (b) enforce_persistent_worktree_serial_lane called with merge_ahead_bound=3
        #     AND num_hosts=3 (currently num_hosts is never passed — step-12 adds it)
        mock_serial.assert_called_once()
        serial_kwargs = mock_serial.call_args[1]
        assert serial_kwargs.get('merge_ahead_bound') == 3, (
            f'Expected merge_ahead_bound=3, got {serial_kwargs!r}'
        )
        assert serial_kwargs.get('num_hosts') == 3, (
            f'Expected num_hosts=3, got {serial_kwargs!r}; '
            'step-12 adds num_hosts=_k to enforce_persistent_worktree_serial_lane'
        )

        # (c) enforce_merge_liveness_margin called with merge_ahead_bound=3
        mock_liveness.assert_called_once()
        liveness_kwargs = mock_liveness.call_args[1]
        assert liveness_kwargs.get('merge_ahead_bound') == 3, (
            f'Expected merge_ahead_bound=3, got {liveness_kwargs!r}'
        )

    async def test_k1_with_zero_runners_byte_identical(self, mock_orch_config) -> None:
        """Zero verify_runners → K=1, num_hosts=1 (byte-identical to pre-task behavior).

        RED until step-12 adds num_hosts=_k to enforce_persistent_worktree_serial_lane.
        """
        mock_orch_config.enabled_verify_runners = []
        mock_orch_config.max_concurrent_tasks = 2
        mock_orch_config.fused_memory.project_id = 'test'

        mock_smw_cls, mock_liveness, mock_serial = await self._run_start_worker(
            mock_orch_config
        )

        # speculation_depth=1
        smw_kwargs = mock_smw_cls.call_args[1]
        assert smw_kwargs['speculation_depth'] == 1, (
            f'Expected speculation_depth=1, got {smw_kwargs.get("speculation_depth")!r}'
        )

        # enforce_persistent_worktree_serial_lane: merge_ahead_bound=1, num_hosts=1
        serial_kwargs = mock_serial.call_args[1]
        assert serial_kwargs.get('merge_ahead_bound') == 1
        assert serial_kwargs.get('num_hosts') == 1, (
            f'Expected num_hosts=1, got {serial_kwargs!r}; '
            'step-12 adds explicit num_hosts=_k (=1 when no runners)'
        )

        # enforce_merge_liveness_margin: merge_ahead_bound=1
        liveness_kwargs = mock_liveness.call_args[1]
        assert liveness_kwargs.get('merge_ahead_bound') == 1
