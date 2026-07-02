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
    h.git_ops.get_merge_diff_files = AsyncMock(return_value=([], None))
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
        """_note_merge_all pre-fetches diff once then passes it to every coordinator's note_merge."""
        coord_a = MagicMock()
        coord_a.note_merge = AsyncMock()
        coord_b = MagicMock()
        coord_b.note_merge = AsyncMock()
        harness._service_restart_coordinators = [coord_a, coord_b]

        await harness._note_merge_all('task-1', 'base-sha', 'head-sha')

        # Pre-fetched diff comes from fixture's AsyncMock(return_value=[])
        coord_a.note_merge.assert_awaited_once_with(
            'task-1', 'base-sha', 'head-sha', prefetched_diff=[]
        )
        coord_b.note_merge.assert_awaited_once_with(
            'task-1', 'base-sha', 'head-sha', prefetched_diff=[]
        )

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

        coord_b.note_merge.assert_awaited_once_with(
            'task-2', 'base2', 'head2', prefetched_diff=[]
        )
        assert 'note boom' in caplog.text

    async def test_git_ops_failure_skips_all_coordinators(
        self, harness: Harness, caplog
    ):
        """When the shared git diff fetch returns an error, no coordinator's note_merge is called.

        get_merge_diff_files is now total (never raises) — errors come via the
        error slot.  _note_merge_all must check err is not None and skip all
        coordinators, logging 'skipping all coordinators'.
        """
        harness.git_ops.get_merge_diff_files = AsyncMock(
            return_value=([], RuntimeError('git boom'))
        )
        coord_a = MagicMock()
        coord_a.note_merge = AsyncMock()
        coord_b = MagicMock()
        coord_b.note_merge = AsyncMock()
        harness._service_restart_coordinators = [coord_a, coord_b]

        with caplog.at_level(logging.WARNING):
            await harness._note_merge_all('task-3', 'base3', 'head3')

        coord_a.note_merge.assert_not_awaited()
        coord_b.note_merge.assert_not_awaited()
        assert 'skipping all coordinators' in caplog.text

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
        """_start_merge_worker passes harness._note_merge_all as on_merge_landed.

        Bound-method equality uses __self__ identity + __func__ identity; that
        is what == checks (is would fail because each attribute access creates a
        fresh wrapper object).
        """
        with patch('orchestrator.merge_queue.SpeculativeMergeWorker') as mock_smw, \
             patch('asyncio.create_task'), \
             patch('orchestrator.merge_queue.check_merge_liveness_margin'):
            await harness._start_merge_worker()

        call_kwargs = mock_smw.call_args.kwargs
        assert call_kwargs['on_merge_landed'] == harness._note_merge_all


# ---------------------------------------------------------------------------
# (g) Busy-branch no-double-fire: _maybe_restart_stale_service(agents_idle=False)
# ---------------------------------------------------------------------------


@pytest.mark.asyncio
class TestMaybeRestartStaleServiceBusyPath:
    """Verify busy-path (agents_idle=False) behavior with real coordinator instances.

    The run() busy-wait branch calls _maybe_restart_stale_service(agents_idle=False)
    on every tick.  These tests confirm:
    - A dashboard coordinator (require_idle=False) fires exactly once after being
      armed; the second call is a no-op (pending was cleared on first fire).
    - A fused-memory coordinator (require_idle=True) never fires on the busy path.
    """

    async def test_dashboard_coordinator_fires_at_most_once_across_busy_ticks(
        self, harness: Harness
    ):
        """Calling _maybe_restart_stale_service(agents_idle=False) twice fires at most once.

        This models two consecutive busy-wait ticks, confirming the no-double-fire
        invariant documented in the busy-branch comment.
        """
        executor = AsyncMock()
        current_time: list[float] = [0.0]

        dashboard_coord = StaleServiceRestartCoordinator(
            git_ops=harness.git_ops,
            event_store=harness.event_store,
            watch_prefixes=['dashboard/src/'],
            debounce_secs=0.0,  # no debounce — fire immediately after arm
            enabled=True,
            restart_executor=executor,
            clock=lambda: current_time[0],
            service_name='dashboard',
            require_idle=False,
        )
        harness._service_restart_coordinators = [dashboard_coord]

        # Arm via pre-fetched diff (skip git_ops)
        await dashboard_coord.note_merge('task-leaf', 'base', 'head', prefetched_diff=['dashboard/src/app.py'])
        assert dashboard_coord.is_pending is True

        # First busy tick: coordinator fires and clears pending
        current_time[0] = 1.0
        first = await harness._maybe_restart_stale_service(agents_idle=False)
        assert first is True
        executor.assert_awaited_once()
        assert dashboard_coord.is_pending is False

        # Second busy tick (consecutive): pending already cleared — must NOT double-fire
        second = await harness._maybe_restart_stale_service(agents_idle=False)
        assert second is False
        executor.assert_awaited_once()  # still exactly one call total

    async def test_fused_memory_coordinator_never_fires_on_busy_path(
        self, harness: Harness
    ):
        """require_idle=True coordinator is a no-op when agents_idle=False.

        Confirms that fused-memory stays reserved for the idle branch even when
        _maybe_restart_stale_service is called with agents_idle=False on the
        busy-wait path.
        """
        executor = AsyncMock()
        current_time: list[float] = [0.0]

        fused_coord = StaleServiceRestartCoordinator(
            git_ops=harness.git_ops,
            event_store=harness.event_store,
            watch_prefixes=['fused-memory/src/'],
            debounce_secs=0.0,
            enabled=True,
            restart_executor=executor,
            clock=lambda: current_time[0],
            service_name='fused-memory',
            require_idle=True,  # idle-only — must not fire on busy path
        )
        harness._service_restart_coordinators = [fused_coord]

        await fused_coord.note_merge('task-1', 'base', 'head', prefetched_diff=['fused-memory/src/server.py'])
        assert fused_coord.is_pending is True

        # Multiple busy ticks: fused-memory must never fire
        current_time[0] = 1.0
        for _ in range(3):
            result = await harness._maybe_restart_stale_service(agents_idle=False)
            assert result is False

        executor.assert_not_awaited()
        assert fused_coord.is_pending is True  # still pending — waiting for the idle branch


# ---------------------------------------------------------------------------
# U2 (task 1973): Harness._merge_pipeline_idle
# ---------------------------------------------------------------------------


class TestMergePipelineIdle:
    """_merge_pipeline_idle() reports whether the merge queue + worker pipeline is drained.

    Used as the orchestrator restart coordinator's restart_precondition — a
    genuine quiet window requires BOTH the merge queue empty AND the merge
    worker's in-flight pipeline (snapshot()['depth']) at zero.
    """

    def test_true_when_merge_worker_is_none(self, harness: Harness):
        """(a) _merge_worker=None → True (bare/unit-test harness)."""
        harness._merge_worker = None

        assert harness._merge_pipeline_idle() is True

    def test_true_when_depth_zero_and_queue_empty(self, harness: Harness):
        """(b) snapshot depth==0 and _merge_queue empty → True."""
        worker = MagicMock()
        worker.snapshot.return_value = {'depth': 0}
        harness._merge_worker = worker
        assert harness._merge_queue.empty()

        assert harness._merge_pipeline_idle() is True

    def test_false_when_depth_nonzero(self, harness: Harness):
        """(c) snapshot depth==2 → False."""
        worker = MagicMock()
        worker.snapshot.return_value = {'depth': 2}
        harness._merge_worker = worker

        assert harness._merge_pipeline_idle() is False

    def test_false_when_queue_has_item_even_if_depth_zero(self, harness: Harness):
        """(d) snapshot depth==0 but _merge_queue non-empty → False."""
        worker = MagicMock()
        worker.snapshot.return_value = {'depth': 0}
        harness._merge_worker = worker
        harness._merge_queue.put_nowait({'task_id': 'queued-task'})

        assert harness._merge_pipeline_idle() is False

    def test_false_when_snapshot_raises(self, harness: Harness):
        """(e) snapshot() raising → False (fail-safe), no exception propagates."""
        worker = MagicMock()
        worker.snapshot.side_effect = RuntimeError('snapshot boom')
        harness._merge_worker = worker

        result = harness._merge_pipeline_idle()  # must not raise

        assert result is False
