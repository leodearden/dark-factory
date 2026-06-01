"""Tests for harness wiring of StaleServiceRestartCoordinator (step-11).

Mirrors the mocked-Harness fixture from test_orphan_reaper.py:
patch McpLifecycle/Scheduler/BriefingAssembler; set h.event_store=MagicMock().

Asserts:
  (a) _build_service_restart_coordinator() returns a coordinator whose fields
      reflect the fused_memory_restart_* config values.
  (b) _start_merge_worker constructs SpeculativeMergeWorker with
      on_merge_landed bound to the coordinator's note_merge.
  (c) _maybe_restart_stale_service(agents_idle=True) delegates to
      coordinator.maybe_restart(agents_idle=True); no-op when coordinator
      is None.
"""

from __future__ import annotations

from pathlib import Path
from unittest.mock import AsyncMock, MagicMock, patch

import pytest

from orchestrator.harness import Harness
from orchestrator.service_restart import StaleServiceRestartCoordinator

# ---------------------------------------------------------------------------
# Fixture
# ---------------------------------------------------------------------------


@pytest.fixture
def harness(tmp_path: Path, mock_orch_config):
    """Minimal harness with mocked heavy deps, configured for service restart."""
    mock_orch_config.fused_memory_restart_on_merge_enabled = True
    mock_orch_config.fused_memory_restart_debounce_secs = 60.0
    mock_orch_config.fused_memory_restart_watch_prefixes = ['fused-memory/src/']
    mock_orch_config.fused_memory_restart_script = 'scripts/restart-fused-memory.sh'

    with patch('orchestrator.harness.McpLifecycle'), \
         patch('orchestrator.harness.Scheduler'), \
         patch('orchestrator.harness.BriefingAssembler'):
        h = Harness(mock_orch_config)

    h.scheduler = MagicMock()
    h.scheduler._dispatched = set()
    h.event_store = MagicMock()
    h.git_ops.get_merge_diff_files = AsyncMock(return_value=[])
    return h


# ---------------------------------------------------------------------------
# (a) _build_service_restart_coordinator
# ---------------------------------------------------------------------------


class TestBuildServiceRestartCoordinator:
    """_build_service_restart_coordinator() reads all four fused_memory_restart_* fields."""

    def test_builds_coordinator_with_correct_config_values(self, harness: Harness):
        """Coordinator fields match the fused_memory_restart_* config values."""
        coord = harness._build_service_restart_coordinator()

        assert isinstance(coord, StaleServiceRestartCoordinator)
        assert coord.enabled is True
        assert coord._debounce_secs == 60.0
        assert coord._watch_prefixes == ['fused-memory/src/']
        assert coord._script_path == 'scripts/restart-fused-memory.sh'

    def test_builds_disabled_coordinator_when_config_says_false(
        self, harness: Harness
    ):
        """When config disables restart, coordinator.enabled is False."""
        harness.config.fused_memory_restart_on_merge_enabled = False

        coord = harness._build_service_restart_coordinator()

        assert isinstance(coord, StaleServiceRestartCoordinator)
        assert coord.enabled is False

    def test_coordinator_git_ops_is_harness_git_ops(self, harness: Harness):
        """Coordinator receives the harness's git_ops instance."""
        coord = harness._build_service_restart_coordinator()

        assert coord._git_ops is harness.git_ops

    def test_coordinator_project_root_matches_config(self, harness: Harness):
        """Coordinator project_root matches config.project_root."""
        coord = harness._build_service_restart_coordinator()

        assert coord._project_root == Path(harness.config.project_root)


# ---------------------------------------------------------------------------
# (b) _start_merge_worker wires on_merge_landed → coordinator.note_merge
# ---------------------------------------------------------------------------


@pytest.mark.asyncio
class TestStartMergeWorkerWiresCallback:
    """_start_merge_worker passes on_merge_landed bound to coordinator's note_merge."""

    async def test_on_merge_landed_is_coordinator_note_merge(
        self, harness: Harness
    ):
        """After _start_merge_worker, SpeculativeMergeWorker received
        on_merge_landed=coordinator.note_merge."""
        with patch('orchestrator.merge_queue.SpeculativeMergeWorker') as mock_smw, \
             patch('asyncio.create_task'):
            await harness._start_merge_worker()

        # Coordinator should have been created and stored
        assert harness._service_restart_coordinator is not None
        assert isinstance(harness._service_restart_coordinator, StaleServiceRestartCoordinator)

        # SpeculativeMergeWorker constructor captured the callback
        mock_smw.assert_called_once()
        call_kwargs = mock_smw.call_args.kwargs
        assert 'on_merge_landed' in call_kwargs
        assert call_kwargs['on_merge_landed'] == \
            harness._service_restart_coordinator.note_merge


# ---------------------------------------------------------------------------
# (c) _maybe_restart_stale_service delegates to coordinator
# ---------------------------------------------------------------------------


@pytest.mark.asyncio
class TestMaybeRestartStaleService:
    """_maybe_restart_stale_service delegates to coordinator.maybe_restart."""

    async def test_delegates_agents_idle_true(self, harness: Harness):
        """_maybe_restart_stale_service(agents_idle=True) → coordinator.maybe_restart(agents_idle=True)."""
        coord = MagicMock()
        coord.maybe_restart = AsyncMock(return_value=True)
        harness._service_restart_coordinator = coord

        result = await harness._maybe_restart_stale_service(agents_idle=True)

        coord.maybe_restart.assert_awaited_once_with(agents_idle=True)
        assert result is True

    async def test_delegates_agents_idle_false(self, harness: Harness):
        """_maybe_restart_stale_service(agents_idle=False) → coordinator.maybe_restart(agents_idle=False)."""
        coord = MagicMock()
        coord.maybe_restart = AsyncMock(return_value=False)
        harness._service_restart_coordinator = coord

        result = await harness._maybe_restart_stale_service(agents_idle=False)

        coord.maybe_restart.assert_awaited_once_with(agents_idle=False)
        assert result is False

    async def test_noop_when_coordinator_is_none(self, harness: Harness):
        """_maybe_restart_stale_service returns False and does not crash when coordinator is None."""
        harness._service_restart_coordinator = None

        result = await harness._maybe_restart_stale_service(agents_idle=True)

        assert not result  # False or None — just don't crash or fire
