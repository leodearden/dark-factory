"""Tests for harness wiring of StaleServiceRestartCoordinator (multi-coordinator API).

Two coordinators are built and stored in a list:
  - fused-memory (service_name='fused-memory', require_idle=True)
  - dashboard   (service_name='dashboard', require_idle=False)

Asserts:
  (a) _build_service_restart_coordinator() returns fused-memory coordinator.
  (b) _build_dashboard_restart_coordinator() returns dashboard coordinator.
  (c) After _start_merge_worker, harness._service_restart_coordinators is a
      list of two coordinators.
  (d) _maybe_restart_stale_service(agents_idle=X) calls maybe_restart(agents_idle=X)
      on EVERY coordinator in the list; returns True iff any returned True.
      No-op returning False when list is empty.
"""

from __future__ import annotations

import logging
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
    """Minimal harness with mocked heavy deps, configured for both service restarts."""
    # fused-memory restart config
    mock_orch_config.fused_memory_restart_on_merge_enabled = True
    mock_orch_config.fused_memory_restart_debounce_secs = 60.0
    mock_orch_config.fused_memory_restart_watch_prefixes = ['fused-memory/src/']
    mock_orch_config.fused_memory_restart_script = 'scripts/restart-fused-memory.sh'
    # dashboard restart config
    mock_orch_config.dashboard_restart_on_merge_enabled = True
    mock_orch_config.dashboard_restart_debounce_secs = 20.0
    mock_orch_config.dashboard_restart_watch_prefixes = ['dashboard/src/']
    mock_orch_config.dashboard_restart_script = 'scripts/restart-dashboard.sh'

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
# (a) _build_service_restart_coordinator — fused-memory builder unchanged
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

    def test_fused_memory_coordinator_has_correct_service_name(self, harness: Harness):
        """Fused-memory coordinator has service_name='fused-memory'."""
        coord = harness._build_service_restart_coordinator()

        assert coord._service_name == 'fused-memory'

    def test_fused_memory_coordinator_requires_idle(self, harness: Harness):
        """Fused-memory coordinator has require_idle=True (idle-only restart)."""
        coord = harness._build_service_restart_coordinator()

        assert coord._require_idle is True


# ---------------------------------------------------------------------------
# (b) _build_dashboard_restart_coordinator — new leaf-service builder
# ---------------------------------------------------------------------------


class TestBuildDashboardRestartCoordinator:
    """_build_dashboard_restart_coordinator() reads dashboard_restart_* config fields."""

    def test_builds_coordinator_with_correct_config_values(self, harness: Harness):
        """Dashboard coordinator fields match the dashboard_restart_* config values."""
        coord = harness._build_dashboard_restart_coordinator()

        assert isinstance(coord, StaleServiceRestartCoordinator)
        assert coord.enabled is True
        assert coord._debounce_secs == 20.0
        assert coord._watch_prefixes == ['dashboard/src/']
        assert coord._script_path == 'scripts/restart-dashboard.sh'

    def test_dashboard_coordinator_has_correct_service_name(self, harness: Harness):
        """Dashboard coordinator has service_name='dashboard'."""
        coord = harness._build_dashboard_restart_coordinator()

        assert coord._service_name == 'dashboard'

    def test_dashboard_coordinator_does_not_require_idle(self, harness: Harness):
        """Dashboard coordinator has require_idle=False (leaf — fires while agents dispatch)."""
        coord = harness._build_dashboard_restart_coordinator()

        assert coord._require_idle is False

    def test_dashboard_coordinator_has_empty_script_args(self, harness: Harness):
        """Dashboard coordinator has script_args=[] (no --drain)."""
        coord = harness._build_dashboard_restart_coordinator()

        assert coord._script_args == []

    def test_dashboard_coordinator_git_ops_is_harness_git_ops(self, harness: Harness):
        """Dashboard coordinator receives the harness's git_ops instance."""
        coord = harness._build_dashboard_restart_coordinator()

        assert coord._git_ops is harness.git_ops

    def test_dashboard_coordinator_project_root_matches_config(self, harness: Harness):
        """Dashboard coordinator project_root matches config.project_root."""
        coord = harness._build_dashboard_restart_coordinator()

        assert coord._project_root == Path(harness.config.project_root)

    def test_builds_disabled_coordinator_when_config_says_false(
        self, harness: Harness
    ):
        """When config disables restart, dashboard coordinator.enabled is False."""
        harness.config.dashboard_restart_on_merge_enabled = False

        coord = harness._build_dashboard_restart_coordinator()

        assert isinstance(coord, StaleServiceRestartCoordinator)
        assert coord.enabled is False


# ---------------------------------------------------------------------------
# (c) _start_merge_worker stores list of two coordinators
# ---------------------------------------------------------------------------


@pytest.mark.asyncio
class TestStartMergeWorkerBuildsCoordinatorList:
    """After _start_merge_worker, harness._service_restart_coordinators is a list of two."""

    async def test_coordinators_list_has_two_entries(self, harness: Harness):
        """_start_merge_worker populates _service_restart_coordinators with fused+dashboard."""
        with patch('orchestrator.merge_queue.SpeculativeMergeWorker'), \
             patch('asyncio.create_task'), \
             patch('orchestrator.merge_queue.check_merge_liveness_margin'):
            await harness._start_merge_worker()

        assert isinstance(harness._service_restart_coordinators, list)
        assert len(harness._service_restart_coordinators) == 2
        # First entry: fused-memory (require_idle=True)
        assert harness._service_restart_coordinators[0]._service_name == 'fused-memory'
        assert harness._service_restart_coordinators[0]._require_idle is True
        # Second entry: dashboard (require_idle=False)
        assert harness._service_restart_coordinators[1]._service_name == 'dashboard'
        assert harness._service_restart_coordinators[1]._require_idle is False

    async def test_start_merge_worker_continues_when_liveness_check_raises(
        self, harness: Harness, caplog
    ):
        """_start_merge_worker must not propagate liveness-check exceptions; list still built."""
        with patch('orchestrator.merge_queue.SpeculativeMergeWorker') as mock_smw, \
             patch('asyncio.create_task') as mock_ct, \
             patch(
                 'orchestrator.merge_queue.check_merge_liveness_margin',
                 side_effect=RuntimeError('liveness boom'),
             ), \
             caplog.at_level(logging.WARNING):
            await harness._start_merge_worker()

        mock_smw.assert_called_once()
        assert isinstance(harness._service_restart_coordinators, list)
        assert len(harness._service_restart_coordinators) == 2
        mock_ct.assert_called_once()
        assert 'liveness boom' in caplog.text


# ---------------------------------------------------------------------------
# (d) _maybe_restart_stale_service delegates to ALL coordinators in list
# ---------------------------------------------------------------------------


@pytest.mark.asyncio
class TestMaybeRestartStaleService:
    """_maybe_restart_stale_service iterates every coordinator in the list."""

    async def test_iterates_all_coordinators_agents_idle_true(self, harness: Harness):
        """Calls maybe_restart(agents_idle=True) on every coordinator; returns True if any fired."""
        coord_a = MagicMock()
        coord_a.maybe_restart = AsyncMock(return_value=True)
        coord_b = MagicMock()
        coord_b.maybe_restart = AsyncMock(return_value=False)
        harness._service_restart_coordinators = [coord_a, coord_b]

        result = await harness._maybe_restart_stale_service(agents_idle=True)

        coord_a.maybe_restart.assert_awaited_once_with(agents_idle=True)
        coord_b.maybe_restart.assert_awaited_once_with(agents_idle=True)
        assert result is True

    async def test_returns_false_when_no_coordinator_fires(self, harness: Harness):
        """Returns False when all coordinators return False."""
        coord_a = MagicMock()
        coord_a.maybe_restart = AsyncMock(return_value=False)
        coord_b = MagicMock()
        coord_b.maybe_restart = AsyncMock(return_value=False)
        harness._service_restart_coordinators = [coord_a, coord_b]

        result = await harness._maybe_restart_stale_service(agents_idle=True)

        coord_a.maybe_restart.assert_awaited_once_with(agents_idle=True)
        coord_b.maybe_restart.assert_awaited_once_with(agents_idle=True)
        assert result is False

    async def test_delegates_agents_idle_false(self, harness: Harness):
        """Passes agents_idle=False to every coordinator."""
        coord_a = MagicMock()
        coord_a.maybe_restart = AsyncMock(return_value=False)
        coord_b = MagicMock()
        coord_b.maybe_restart = AsyncMock(return_value=True)
        harness._service_restart_coordinators = [coord_a, coord_b]

        result = await harness._maybe_restart_stale_service(agents_idle=False)

        coord_a.maybe_restart.assert_awaited_once_with(agents_idle=False)
        coord_b.maybe_restart.assert_awaited_once_with(agents_idle=False)
        assert result is True

    async def test_noop_when_list_is_empty(self, harness: Harness):
        """Returns False and does not crash when list is empty."""
        harness._service_restart_coordinators = []

        result = await harness._maybe_restart_stale_service(agents_idle=True)

        assert result is False


# ---------------------------------------------------------------------------
# (e) _note_merge_all fan-out + fail-open tests
# ---------------------------------------------------------------------------


@pytest.mark.asyncio
class TestNoteMergeAll:
    """_note_merge_all fans out note_merge to every coordinator, fail-open per coordinator."""

    async def test_fans_out_to_all_coordinators(self, harness: Harness):
        """_note_merge_all awaits note_merge on every coordinator with identical args."""
        coord_a = MagicMock()
        coord_a.note_merge = AsyncMock()
        coord_b = MagicMock()
        coord_b.note_merge = AsyncMock()
        harness._service_restart_coordinators = [coord_a, coord_b]

        await harness._note_merge_all('task-1', 'base-sha', 'head-sha')

        coord_a.note_merge.assert_awaited_once_with('task-1', 'base-sha', 'head-sha')
        coord_b.note_merge.assert_awaited_once_with('task-1', 'base-sha', 'head-sha')

    async def test_fail_open_when_first_coordinator_raises(
        self, harness: Harness, caplog
    ):
        """When the first coordinator's note_merge raises, the second is still awaited."""
        coord_a = MagicMock()
        coord_a.note_merge = AsyncMock(side_effect=RuntimeError('note boom'))
        coord_b = MagicMock()
        coord_b.note_merge = AsyncMock()
        harness._service_restart_coordinators = [coord_a, coord_b]

        with caplog.at_level(logging.WARNING):
            # Must NOT raise
            await harness._note_merge_all('task-2', 'base2', 'head2')

        coord_b.note_merge.assert_awaited_once_with('task-2', 'base2', 'head2')
        assert 'note boom' in caplog.text

    async def test_noop_when_list_is_empty(self, harness: Harness):
        """_note_merge_all does not crash when list is empty."""
        harness._service_restart_coordinators = []

        # Must NOT raise
        await harness._note_merge_all('task-3', 'base3', 'head3')


# ---------------------------------------------------------------------------
# (f) _start_merge_worker wires on_merge_landed to _note_merge_all
# ---------------------------------------------------------------------------


@pytest.mark.asyncio
class TestStartMergeWorkerOnMergeLandedWiring:
    """After _start_merge_worker, SpeculativeMergeWorker receives on_merge_landed=harness._note_merge_all."""

    async def test_worker_wired_with_note_merge_all(self, harness: Harness):
        """_start_merge_worker passes harness._note_merge_all as on_merge_landed."""
        with patch('orchestrator.merge_queue.SpeculativeMergeWorker') as mock_smw, \
             patch('asyncio.create_task'), \
             patch('orchestrator.merge_queue.check_merge_liveness_margin'):
            await harness._start_merge_worker()

        call_kwargs = mock_smw.call_args.kwargs
        assert call_kwargs['on_merge_landed'] is harness._note_merge_all
