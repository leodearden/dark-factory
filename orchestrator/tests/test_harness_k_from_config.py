"""Tests for K / num_hosts wiring in harness._start_merge_worker.

Task 1716 wired the merge lane's sizing from config.  Task 8526 (merge-lane
policy δ, plans/merge-lane-policy-prd.md D4) split it into two facts:

- the verify-host count, ``1 + len(config.enabled_verify_runners)``, which
  bounds concurrently leased verifies (one allocator slot per host) and so is
  what ``enforce_persistent_worktree_serial_lane`` receives as both
  ``merge_ahead_bound`` and ``num_hosts``;
- the speculation depth K = hosts + 1, which sizes the worker's frontier
  (``speculation_depth``) and the spec warm-lane pool.

Modeled on test_harness_merge_registry_wiring.py::TestStartMergeWorkerCallsLivenessGuard.
"""

from __future__ import annotations

import logging
from pathlib import Path
from unittest.mock import MagicMock, patch

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
    """K = hosts + 1; the serial-lane guard is fed the host count.

    hosts = 1 + len(config.enabled_verify_runners).

    Patching strategy
    -----------------
    _start_merge_worker does a LOCAL import:
        from orchestrator.merge_lane.worker import (
            _MERGE_AHEAD_BOUND, SpeculativeMergeWorker,
            enforce_merge_liveness_margin, enforce_persistent_worktree_serial_lane,
        )
    Patching the SOURCE module attribute (``orchestrator.merge_lane.worker.*``) is
    what the local import picks up at call time; patching ``orchestrator.harness.*``
    would have no effect because no module-level binding exists there.
    """

    async def _run_start_worker(self, h: Harness):
        """Helper: patch, call *h*._start_merge_worker, return mocks."""
        with patch('orchestrator.merge_lane.worker.SpeculativeMergeWorker') as mock_smw_cls, \
             patch.object(h, '_build_service_restart_coordinator',
                          return_value=MagicMock()), \
             patch('asyncio.create_task', return_value=MagicMock()), \
             patch('orchestrator.merge_lane.liveness.enforce_merge_liveness_margin') as mock_liveness, \
             patch('orchestrator.merge_lane.liveness.enforce_persistent_worktree_serial_lane') as mock_serial:

            mock_smw = MagicMock()
            mock_smw_cls.return_value = mock_smw

            await h._start_merge_worker()

        return mock_smw_cls, mock_liveness, mock_serial

    async def test_k_is_hosts_plus_one_with_two_enabled_runners(
        self, mock_orch_config,
    ) -> None:
        """Two enabled verify_runners → hosts=3, K=4.

        Asserts:
          (a) SpeculativeMergeWorker(speculation_depth=4)
          (b) enforce_persistent_worktree_serial_lane(..., merge_ahead_bound=3, num_hosts=3)
              — the HOST count, the bound on concurrently leased verifies, not K
          (c) enforce_merge_liveness_margin(config) — config only, no kwargs
              (task-1729 drops raw _k: heartbeat floor makes K irrelevant to the liveness guard)
        """
        r1 = VerifyRunnerConfig(
            name='r1', ssh_host='h1.local', git_remote='remote1', df_checkout_path='/df',
        )
        r2 = VerifyRunnerConfig(
            name='r2', ssh_host='h2.local', git_remote='remote2', df_checkout_path='/df',
        )
        mock_orch_config.enabled_verify_runners = [r1, r2]
        mock_orch_config.max_concurrent_tasks = 2
        mock_orch_config.fused_memory.project_id = 'test'

        mock_smw_cls, mock_liveness, mock_serial = await self._run_start_worker(
            _build_harness(mock_orch_config),
        )

        smw_kwargs = mock_smw_cls.call_args[1]
        assert smw_kwargs['speculation_depth'] == 4, (
            f'Expected speculation_depth=4 (K = hosts + 1 = 3 + 1), '
            f'got {smw_kwargs.get("speculation_depth")!r}'
        )

        mock_serial.assert_called_once()
        serial_kwargs = mock_serial.call_args[1]
        assert serial_kwargs.get('merge_ahead_bound') == 3, (
            f'Expected merge_ahead_bound=3 (the host count), got {serial_kwargs!r}'
        )
        assert serial_kwargs.get('num_hosts') == 3, (
            f'Expected num_hosts=3, got {serial_kwargs!r}'
        )

        mock_liveness.assert_called_once_with(mock_orch_config)

    async def test_zero_runners_is_one_host_and_k_two(self, mock_orch_config) -> None:
        """Zero verify_runners → hosts=1, K=2; the guard sees bound 1 over 1 host."""
        mock_orch_config.enabled_verify_runners = []
        mock_orch_config.max_concurrent_tasks = 2
        mock_orch_config.fused_memory.project_id = 'test'

        mock_smw_cls, mock_liveness, mock_serial = await self._run_start_worker(
            _build_harness(mock_orch_config),
        )

        smw_kwargs = mock_smw_cls.call_args[1]
        assert smw_kwargs['speculation_depth'] == 2, (
            f'Expected speculation_depth=2, got {smw_kwargs.get("speculation_depth")!r}'
        )

        serial_kwargs = mock_serial.call_args[1]
        assert serial_kwargs.get('merge_ahead_bound') == 1
        assert serial_kwargs.get('num_hosts') == 1, (
            f'Expected num_hosts=1, got {serial_kwargs!r}'
        )

        mock_liveness.assert_called_once_with(mock_orch_config)

    async def test_spec_pool_and_worker_share_one_k(self, mock_orch_config) -> None:
        """The spec warm-lane pool holds one lane per frontier entry: K of them."""
        r1 = VerifyRunnerConfig(
            name='r1', ssh_host='h1.local', git_remote='remote1', df_checkout_path='/df',
        )
        mock_orch_config.enabled_verify_runners = [r1]
        mock_orch_config.max_concurrent_tasks = 2
        mock_orch_config.fused_memory.project_id = 'test'
        mock_orch_config.git = mock_orch_config.git.model_copy(
            update={'merge_spec_warm_lane_pool': True},
        )

        h = _build_harness(mock_orch_config)
        pool = h.git_ops.spec_warm_lane_pool
        assert pool is not None

        mock_smw_cls, _, _ = await self._run_start_worker(h)

        depth = mock_smw_cls.call_args[1]['speculation_depth']
        assert depth == 3
        assert pool.capacity == depth


# ---------------------------------------------------------------------------
# TestStartMergeWorkerK2LivenessRepro — γ integration gate (task 1730)
# ---------------------------------------------------------------------------


@pytest.mark.asyncio
class TestStartMergeWorkerK2LivenessRepro:
    """Two-host startup repro through the REAL guards (γ integration gate).

    Drives _start_merge_worker end-to-end with the actual
    enforce_merge_liveness_margin against the 2026-06-10 crash config
    (one enabled verify_runner → two hosts, cold_timeout_secs=7200).

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
        """Two hosts, cold_timeout=7200 pass the real enforce_merge_liveness_margin.

        The exact operator config that crashed pre-β is now safe because the
        heartbeat-floor formula (worst_case = _HEARTBEAT_POLL_S ×
        TOUCH_MISS_TOLERANCE = 600 s) decouples from cold timeout and K.
        """
        # Exact 2026-06-10 crash config: one enabled verify_runner (two hosts),
        # cold_timeout=7200.  Pre-β the formula scaled with cold_timeout×K and
        # raised MergeLivenessConfigError; post-β the floor is constant at 600 s.
        config = OrchestratorConfig(
            project_root=tmp_path,
            verify_runners=[VerifyRunnerConfig(
                name='laptop',
                ssh_host='leo-laptop',
                git_remote='leo-laptop',
                df_checkout_path='/home/leo/src/dark-factory',
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

        with patch('orchestrator.merge_lane.worker.SpeculativeMergeWorker') as mock_smw_cls, \
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

        # (2) SpeculativeMergeWorker called once with speculation_depth=3
        #     (hosts = 1 + 1 verify_runner = 2, K = hosts + 1)
        mock_smw_cls.assert_called_once()
        smw_kwargs = mock_smw_cls.call_args[1]
        assert smw_kwargs['speculation_depth'] == 3, (
            f'Expected speculation_depth=3 (K = 2 hosts + 1), '
            f'got {smw_kwargs.get("speculation_depth")!r}'
        )

        # (3) 'Speculative merge worker constructed' in caplog
        assert 'Speculative merge worker constructed' in caplog.text, (
            f'Expected startup log in caplog; got: {caplog.text!r}'
        )

        # (4) Serial-lane guard fed the host count (bound 2 over 2 hosts) →
        # per_host = 1 → no raise.  Proven by clean completion above with
        # persistent_merge_worktree=True (the guard early-returns only when the
        # knob is off; here it runs and passes).  Fed K = 3 against 2 hosts it
        # would compute per_host = 2 and refuse startup — reify's live shape.
        # The β config-independence property (worst_case_secs invariant across cold
        # timeouts) is owned by TestMergeLivenessMargin in test_merge_queue.py.
