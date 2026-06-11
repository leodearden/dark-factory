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

import logging
from pathlib import Path
from unittest.mock import AsyncMock, MagicMock, patch

import pytest

from orchestrator.config import GitConfig, OrchestratorConfig, VerifyRunnerConfig
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
          (c) enforce_merge_liveness_margin(config) — config only, no kwargs
              (task-1729 drops raw _k: heartbeat floor makes K irrelevant to the liveness guard)

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

        # (c) enforce_merge_liveness_margin called with self.config ONLY (no bound/hosts kwargs;
        #     task-1729 drops raw _k — heartbeat floor makes K irrelevant to the liveness guard)
        mock_liveness.assert_called_once_with(mock_orch_config)

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

        # (c) enforce_merge_liveness_margin called with self.config ONLY (no bound/hosts kwargs;
        #     task-1729 drops raw _k — heartbeat floor makes K irrelevant to the liveness guard)
        mock_liveness.assert_called_once_with(mock_orch_config)


# ---------------------------------------------------------------------------
# TestStartMergeWorkerK2LivenessRepro — γ integration gate (task 1730)
# ---------------------------------------------------------------------------


@pytest.mark.asyncio
class TestStartMergeWorkerK2LivenessRepro:
    """K=2 startup repro through the REAL liveness guard (γ integration gate).

    Drives _start_merge_worker end-to-end with the actual
    enforce_merge_liveness_margin against the 2026-06-10 crash config
    (one enabled verify_runner → K=2, cold_timeout_secs=7200).

    Patch set: ONLY SpeculativeMergeWorker / asyncio.create_task /
    _build_service_restart_coordinator.  Both guards run REAL.

    GREEN on current main (α+β merged).
    RED against pre-β formula: encoded via config-independence assertion —
    worst_case_secs==600 for cold_timeout 7200 AND 9000, i.e. cold timeout
    no longer drives the verdict (the β property that fixes the crash).
    """

    async def test_k2_startup_passes_real_liveness_guard(
        self,
        tmp_path: Path,
        caplog: pytest.LogCaptureFixture,
    ) -> None:
        """K=2, cold_timeout=7200 passes the real enforce_merge_liveness_margin.

        The exact operator config that crashed pre-β is now safe because the
        heartbeat-floor formula (worst_case = _HEARTBEAT_POLL_S ×
        TOUCH_MISS_TOLERANCE = 600 s) decouples from cold timeout and K.
        """
        from orchestrator.merge_queue import check_merge_liveness_margin

        # Exact 2026-06-10 crash config: one enabled verify_runner → K=2,
        # cold_timeout=7200.  Pre-β the formula scaled with cold_timeout×K and
        # raised MergeLivenessConfigError; post-β the floor is constant at 600 s.
        config = OrchestratorConfig(
            project_root=tmp_path,
            verify_runners=[VerifyRunnerConfig(
                name='laptop',
                ssh_host='leo-laptop',
                git_remote='leo-laptop',
            )],
            merge_verify_cold_command_timeout_secs=7200,
            git=GitConfig(
                persistent_merge_worktree=True,
                main_branch='main',
                branch_prefix='task/',
                remote='origin',
                worktree_dir='.worktrees',
            ),
        )

        h = _build_harness(config)

        with patch('orchestrator.merge_queue.SpeculativeMergeWorker') as mock_smw_cls, \
             patch.object(h, '_build_service_restart_coordinator',
                          return_value=MagicMock()), \
             patch('asyncio.create_task', return_value=MagicMock()), \
             caplog.at_level(logging.INFO, logger='orchestrator.harness'):

            # Use plain MagicMock for run (not AsyncMock): asyncio.create_task is
            # mocked, so run() never needs to produce a real awaitable coroutine.
            # Using AsyncMock would create an unawaited coroutine when create_task
            # is mocked away, triggering the project's "error:coroutine.*was never
            # awaited" filterwarnings rule.
            mock_smw = MagicMock()
            mock_smw_cls.return_value = mock_smw

            # (1) No exception raised — the crash-loop repro now passes.
            # enforce_merge_liveness_margin raises MergeLivenessConfigError when
            # worst_case_secs >= threshold_secs; clean return proves the real guard
            # accepted this config.  (A raised error would propagate and fail the test.)
            await h._start_merge_worker()

        # (2) SpeculativeMergeWorker called once with speculation_depth=2 (K = 1 + 1)
        mock_smw_cls.assert_called_once()
        smw_kwargs = mock_smw_cls.call_args[1]
        assert smw_kwargs['speculation_depth'] == 2, (
            f'Expected speculation_depth=2 (K=1+1 verify_runner), '
            f'got {smw_kwargs.get("speculation_depth")!r}'
        )

        # (3) 'Speculative merge worker started' in caplog
        assert 'Speculative merge worker started' in caplog.text, (
            f'Expected startup log in caplog; got: {caplog.text!r}'
        )

        # (4) Serial-lane guard at K=2/num_hosts=2 → per_host=ceil(2/2)=1 → no raise.
        # Proven implicitly by clean completion above with persistent_merge_worktree=True
        # (the guard early-returns only when the knob is off; here it runs and passes).
        # This regression-holds the 1692 seam: num_hosts=K keeps per-host bound = 1.

        # β property — config-independence of worst_case_secs.
        # Pre-β formula: worst_case = cold_timeout × bound / num_hosts (K-coupled) →
        #   7200 × 2 / ... >> threshold → MergeLivenessConfigError → crash loop.
        # Post-β formula: worst_case = _HEARTBEAT_POLL_S × TOUCH_MISS_TOLERANCE = 600 s
        #   → invariant across cold_timeout and K.
        cfg_9000 = OrchestratorConfig(
            project_root=tmp_path,
            merge_verify_cold_command_timeout_secs=9000,
        )
        result_7200 = check_merge_liveness_margin(config)
        result_9000 = check_merge_liveness_margin(cfg_9000)

        assert result_7200.worst_case_secs == 600.0, (
            f'Expected heartbeat floor=600 s for cold_timeout=7200; '
            f'got {result_7200.worst_case_secs}'
        )
        assert result_9000.worst_case_secs == 600.0, (
            'cold timeout must not feed worst_case_secs (β property that fixes the crash); '
            f'cfg_7200 floor={result_7200.worst_case_secs}, '
            f'cfg_9000 floor={result_9000.worst_case_secs}'
        )
        assert result_7200.safe is True
        assert result_9000.safe is True
