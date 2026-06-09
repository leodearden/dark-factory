"""Tests for Harness external-dep escalation sink (task 1580).

Asserts that:
- After Harness construction, scheduler._on_external_dep_block is set (non-None).
- Invoking the callback sets the task to 'blocked' via scheduler.set_task_status.
- Invoking the callback submits exactly one L1 Escalation to the escalation queue.
- A second invocation with an open L1 is deduplicated (no duplicate submission).
"""

from __future__ import annotations

from pathlib import Path
from unittest.mock import AsyncMock, MagicMock, patch

import pytest
from escalation.models import Escalation
from escalation.queue import EscalationQueue

from orchestrator.config import OrchestratorConfig
from orchestrator.harness import Harness
from orchestrator.merge_queue import MergeLivenessConfigError

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
    mock_queue.make_id.return_value = 'esc-test-external-1'
    harness._escalation_queue = mock_queue
    return mock_queue


# ---------------------------------------------------------------------------
# TestHarnessExternalDepBlockWiring (step-13 RED / step-14 GREEN)
# ---------------------------------------------------------------------------

class TestHarnessExternalDepBlockWiring:
    """Harness must wire scheduler._on_external_dep_block after Scheduler construction."""

    def test_callback_installed_after_construction(self, tmp_path: Path) -> None:
        """After Harness construction, scheduler._on_external_dep_block is not None."""
        harness = _make_harness(tmp_path)

        assert harness.scheduler._on_external_dep_block is not None, (
            'Harness must install _on_external_dep_block on the scheduler '
            'right after Scheduler construction'
        )

    @pytest.mark.asyncio
    async def test_callback_sets_task_blocked(self, tmp_path: Path) -> None:
        """Invoking the callback sets the task to 'blocked' via scheduler.set_task_status."""
        harness = _make_harness(tmp_path)
        _install_mock_escalation_queue(harness)

        # Patch scheduler.set_task_status to avoid real MCP calls.
        harness.scheduler.set_task_status = AsyncMock(return_value=True)

        assert harness.scheduler._on_external_dep_block is not None
        await harness.scheduler._on_external_dep_block(
            '42',
            summary='EXTERNAL_DEP_CANCELLED: task 42 — dep cancelled',
            detail='dep dark_factory:5 is cancelled',
            category='dependency_discovered',
        )

        harness.scheduler.set_task_status.assert_called_once_with('42', 'blocked')

    @pytest.mark.asyncio
    async def test_callback_submits_l1_escalation(self, tmp_path: Path) -> None:
        """Invoking the callback submits exactly one L1 Escalation to the queue."""
        harness = _make_harness(tmp_path)
        mock_queue = _install_mock_escalation_queue(harness)
        harness.scheduler.set_task_status = AsyncMock(return_value=True)

        assert harness.scheduler._on_external_dep_block is not None
        await harness.scheduler._on_external_dep_block(
            '42',
            summary='EXTERNAL_DEP_CANCELLED: task 42 — dep cancelled',
            detail='dep dark_factory:5 is cancelled',
            category='dependency_discovered',
        )

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
    async def test_callback_deduplicates_on_open_l1(self, tmp_path: Path) -> None:
        """Second invocation with open L1 must NOT submit a duplicate escalation."""
        harness = _make_harness(tmp_path)
        mock_queue = _install_mock_escalation_queue(harness)
        harness.scheduler.set_task_status = AsyncMock(return_value=True)

        assert harness.scheduler._on_external_dep_block is not None

        # First call — no open L1 yet → submits
        mock_queue.has_open_l1.return_value = False
        await harness.scheduler._on_external_dep_block(
            '42',
            summary='EXTERNAL_DEP_CANCELLED: task 42 — dep cancelled',
            detail='detail',
            category='dependency_discovered',
        )

        # Second call — L1 is now open → must NOT submit again
        mock_queue.has_open_l1.return_value = True
        await harness.scheduler._on_external_dep_block(
            '42',
            summary='EXTERNAL_DEP_CANCELLED: task 42 — dep cancelled',
            detail='detail',
            category='dependency_discovered',
        )

        assert mock_queue.submit.call_count == 1, (
            f'Should deduplicate: submit called {mock_queue.submit.call_count} times '
            f'(expected exactly 1)'
        )

    @pytest.mark.asyncio
    async def test_callback_no_escalation_queue_is_safe(self, tmp_path: Path) -> None:
        """With no escalation queue installed, callback must not raise."""
        harness = _make_harness(tmp_path)
        harness._escalation_queue = None
        harness.scheduler.set_task_status = AsyncMock(return_value=True)

        # Must not raise even without a queue.
        assert harness.scheduler._on_external_dep_block is not None
        await harness.scheduler._on_external_dep_block(
            '99',
            summary='EXTERNAL_DEP_UNRESOLVED: ...',
            detail='detail',
            category='dependency_discovered',
        )


# ---------------------------------------------------------------------------
# TestHarnessMergeLivenessGuard (step-7 RED / step-8 GREEN)
# ---------------------------------------------------------------------------

class TestHarnessMergeLivenessGuard:
    """Harness._start_merge_worker must be FAIL-CLOSED on an over-budget config.

    At default bound (K=1) and cold timeout 9000 s:
      worst_case = 1 × 9000 = 9000 ≥ threshold (0.75 × 10800 = 8100) → NOT safe.
    At default bound and default cold timeout (7200 s):
      worst_case = 1 × 7200 = 7200 < 8100 → safe → worker starts.
    """

    @pytest.mark.asyncio
    async def test_over_budget_config_raises_at_startup(self, tmp_path: Path) -> None:
        """Over-budget config (cold_timeout=9000, bound=1) must raise MergeLivenessConfigError."""
        config = OrchestratorConfig(
            project_root=tmp_path,
            merge_verify_cold_command_timeout_secs=9000,
        )
        harness = Harness(config)

        with pytest.raises(MergeLivenessConfigError):
            await harness._start_merge_worker()

    @pytest.mark.asyncio
    async def test_in_budget_config_starts_worker(self, tmp_path: Path) -> None:
        """Default/in-budget config (cold_timeout=7200 default, bound=1) must NOT raise.

        Patch SpeculativeMergeWorker.run to a no-op coroutine so the asyncio
        task doesn't run indefinitely and the test tears down cleanly.
        """
        config = OrchestratorConfig(project_root=tmp_path)
        harness = Harness(config)

        async def _noop_run(self_w):  # type: ignore[no-untyped-def]
            pass

        with patch(
            'orchestrator.merge_queue.SpeculativeMergeWorker.run',
            _noop_run,
        ):
            # Must NOT raise MergeLivenessConfigError
            await harness._start_merge_worker()

        try:
            # Worker task was created and will have completed (no-op)
            assert harness._merge_worker_task is not None
        finally:
            # Explicit teardown: stop the worker and await the task so the test
            # cannot leave dangling asyncio tasks or open resources.
            await harness._stop_merge_worker()
