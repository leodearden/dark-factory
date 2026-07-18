"""Tests for harness wiring of StaleServiceRestartCoordinator (multi-coordinator API).

Three coordinators are built and stored in a list:
  - fused-memory (service_name='fused-memory', require_idle=True)
  - dashboard   (service_name='dashboard', require_idle=False)
  - orchestrator (service_name='orchestrator', require_idle=True, U2 — task 1973)

Asserts:
  (a) _build_service_restart_coordinator() returns fused-memory coordinator.
  (b) _build_dashboard_restart_coordinator() returns dashboard coordinator.
  (c) After _start_merge_worker, harness._service_restart_coordinators is a
      list of three coordinators.
  (d) _maybe_restart_stale_service(agents_idle=X) calls maybe_restart(agents_idle=X)
      on EVERY coordinator in the list; returns True iff any returned True.
      No-op returning False when list is empty.
"""

from __future__ import annotations

import logging
from pathlib import Path
from unittest.mock import AsyncMock, MagicMock, patch

import pytest
import yaml
from escalation.queue import EscalationQueue

from orchestrator.harness import Harness
from orchestrator.service_restart import (
    FLEET_DEPLOY_CLOCK_RELPATH,
    StaleServiceRestartCoordinator,
)

# Repo root + committed dark-factory orchestrator config, for the
# stamp_clock_on_fire/orchestrator_restart_script drift guard below. Mirrors
# tests/scripts/test_orchestrator_restart_config_drift.py's REPO_ROOT/
# DF_CONFIG_PATH pattern (that file is outside this task's locked module
# scope, so the guard lives here instead).
REPO_ROOT = Path(__file__).resolve().parents[2]
DF_CONFIG_PATH = REPO_ROOT / "dark-factory-orchestrator.yaml"

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
    # orchestrator restart config (U2 — task 1973); mirrors the real Config
    # defaults (disabled-by-default operator gate — see config.py).
    mock_orch_config.orchestrator_restart_on_merge_enabled = False
    mock_orch_config.orchestrator_restart_debounce_secs = 300.0
    mock_orch_config.orchestrator_restart_watch_prefixes = ['orchestrator/src/']
    mock_orch_config.orchestrator_restart_script = 'scripts/restart-orchestrator.sh'
    mock_orch_config.orchestrator_restart_on_active_secs = 10
    # Restart-safe self-redeploy rate cap (task 2371). Mirrors the real Config
    # default (8h). Only the orchestrator's own coordinator receives this — the
    # fused-memory/dashboard builders keep the 0.0 default (no gating).
    mock_orch_config.orchestrator_restart_min_interval_secs = 28800.0
    # Force-fire escape (fleet-redeploy PRD task delta — task 2398). Mirrors
    # the real Config default (75 min). Only the orchestrator's own
    # coordinator receives this — the fused-memory/dashboard builders keep
    # the 0.0 default (force-fire disabled, byte-identical behaviour).
    mock_orch_config.orchestrator_restart_force_fire_after_secs = 4500.0
    # Bounded pre-enqueue MERGE-phase grace (task 2753). Mirrors the real
    # Config default (10 min). Only the orchestrator's own coordinator receives
    # this (as merge_phase_grace_secs) — the fused-memory/dashboard builders
    # keep the 0.0 default (no hold, byte-identical behaviour).
    mock_orch_config.orchestrator_restart_merge_phase_grace_secs = 600.0

    with patch('orchestrator.harness.McpLifecycle'), \
         patch('orchestrator.harness.Scheduler'), \
         patch('orchestrator.harness.BriefingAssembler'):
        h = Harness(mock_orch_config)

    h.scheduler = MagicMock()
    h.scheduler._dispatched = set()
    # Default: no pre-enqueue MERGE-phase workflow in the grace window, so the
    # merge pipeline reads idle unless a test opts in (task 2753).
    h.scheduler.merge_phase_grace_active.return_value = False
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
# (c) _start_merge_worker stores list of three coordinators
# ---------------------------------------------------------------------------


@pytest.mark.asyncio
class TestStartMergeWorkerBuildsCoordinatorList:
    """After _start_merge_worker, harness._service_restart_coordinators is a list of three."""

    async def test_coordinators_list_has_three_entries(self, harness: Harness):
        """_start_merge_worker populates _service_restart_coordinators with fused+dashboard+orchestrator."""
        with patch('orchestrator.merge_queue.SpeculativeMergeWorker'), \
             patch('asyncio.create_task'), \
             patch('orchestrator.merge_queue.check_merge_liveness_margin'):
            await harness._start_merge_worker()

        assert isinstance(harness._service_restart_coordinators, list)
        assert len(harness._service_restart_coordinators) == 3
        # First entry: fused-memory (require_idle=True)
        assert harness._service_restart_coordinators[0]._service_name == 'fused-memory'
        assert harness._service_restart_coordinators[0]._require_idle is True
        # Second entry: dashboard (require_idle=False)
        assert harness._service_restart_coordinators[1]._service_name == 'dashboard'
        assert harness._service_restart_coordinators[1]._require_idle is False
        # Third entry: orchestrator (require_idle=True, U2 — task 1973)
        assert harness._service_restart_coordinators[2]._service_name == 'orchestrator'
        assert harness._service_restart_coordinators[2]._require_idle is True

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
        assert len(harness._service_restart_coordinators) == 3
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

    def test_false_when_merge_phase_grace_active(self, harness: Harness):
        """(f, task 2753) depth==0 + queue empty but a pre-enqueue MERGE-phase
        workflow is inside the grace window → NOT idle (the polite redeploy
        must defer until the workflow reaches the durable merge journal)."""
        worker = MagicMock()
        worker.snapshot.return_value = {'depth': 0}
        harness._merge_worker = worker
        assert harness._merge_queue.empty()
        harness.scheduler.merge_phase_grace_active.return_value = True

        assert harness._merge_pipeline_idle() is False

    def test_true_when_merge_phase_grace_inactive(self, harness: Harness):
        """(g, task 2753) depth==0 + queue empty + no active merge-phase grace
        → idle (the new gate does not spuriously withhold when no workflow is
        in its pre-enqueue window)."""
        worker = MagicMock()
        worker.snapshot.return_value = {'depth': 0}
        harness._merge_worker = worker
        harness.scheduler.merge_phase_grace_active.return_value = False

        assert harness._merge_pipeline_idle() is True


# ---------------------------------------------------------------------------
# task 2753: Harness._merge_phase_grace_active — fail-toward-FALSE add-on gate
# ---------------------------------------------------------------------------


class TestMergePhaseGraceActive:
    """_merge_phase_grace_active() reads the config grace and delegates to the
    Scheduler, but fails toward FALSE (no grace) — unlike _merge_pipeline_idle's
    snapshot fail-safe which defers.  A bug in this protective add-on must
    degrade to exact pre-task-2753 behaviour (never a NEW indefinite veto).
    """

    def test_delegates_to_scheduler_with_config_grace(self, harness: Harness):
        """Passes the config grace to scheduler.merge_phase_grace_active and
        returns its result."""
        harness.config.orchestrator_restart_merge_phase_grace_secs = 600.0
        harness.scheduler.merge_phase_grace_active.return_value = True

        assert harness._merge_phase_grace_active() is True
        harness.scheduler.merge_phase_grace_active.assert_called_once_with(600.0)

    def test_false_when_config_grace_zero_without_calling_scheduler(
        self, harness: Harness
    ):
        """grace_secs <= 0 short-circuits to False (kill switch) — the scheduler
        is never consulted."""
        harness.config.orchestrator_restart_merge_phase_grace_secs = 0
        harness.scheduler.merge_phase_grace_active.side_effect = AssertionError(
            'scheduler must not be consulted when the config grace is disabled'
        )

        assert harness._merge_phase_grace_active() is False

    def test_false_and_warns_when_scheduler_raises(
        self, harness: Harness, caplog: pytest.LogCaptureFixture
    ):
        """A raising scheduler call degrades to False (no grace) + a WARNING —
        NOT a defer, which would reintroduce the indefinite-veto failure mode
        this task fixes."""
        harness.config.orchestrator_restart_merge_phase_grace_secs = 600.0
        harness.scheduler.merge_phase_grace_active.side_effect = RuntimeError('boom')

        with caplog.at_level(logging.WARNING, logger='orchestrator.harness'):
            assert harness._merge_phase_grace_active() is False

        assert any(
            'merge_phase' in r.message.lower() for r in caplog.records
        ), 'a raising merge-phase grace check must log a WARNING'


# ---------------------------------------------------------------------------
# U2 (task 1973): Harness._build_orchestrator_restart_coordinator
# ---------------------------------------------------------------------------


class TestBuildOrchestratorRestartCoordinator:
    """_build_orchestrator_restart_coordinator() reads orchestrator_restart_* config.

    Unlike the fused-memory/dashboard builders, this coordinator is wired
    with a CUSTOM restart_executor (cgroup-escaping systemd-run, not the
    coordinator's default create_subprocess_exec path) and a
    restart_precondition (the merge-drain gate, Harness._merge_pipeline_idle).
    """

    def test_builds_coordinator_with_correct_config_values(self, harness: Harness):
        """Coordinator fields match the orchestrator_restart_* config values."""
        coord = harness._build_orchestrator_restart_coordinator()

        assert isinstance(coord, StaleServiceRestartCoordinator)
        assert coord._debounce_secs == 300.0
        assert coord._watch_prefixes == ['orchestrator/src/']
        assert coord._script_path == 'scripts/restart-orchestrator.sh'

    def test_service_name_is_orchestrator(self, harness: Harness):
        """Coordinator has service_name='orchestrator'."""
        coord = harness._build_orchestrator_restart_coordinator()

        assert coord._service_name == 'orchestrator'

    def test_requires_idle(self, harness: Harness):
        """Coordinator has require_idle=True — never fires while agents dispatch."""
        coord = harness._build_orchestrator_restart_coordinator()

        assert coord._require_idle is True

    def test_disabled_by_default(self, harness: Harness):
        """Disabled under the default config (operator gate — soak + sign-off required)."""
        coord = harness._build_orchestrator_restart_coordinator()

        assert coord.enabled is False

    def test_enabled_when_config_flag_flipped(self, harness: Harness):
        """Enabled once the operator flips orchestrator_restart_on_merge_enabled."""
        harness.config.orchestrator_restart_on_merge_enabled = True

        coord = harness._build_orchestrator_restart_coordinator()

        assert coord.enabled is True

    def test_restart_precondition_is_merge_pipeline_idle(self, harness: Harness):
        """restart_precondition is harness._merge_pipeline_idle (the drain gate)."""
        coord = harness._build_orchestrator_restart_coordinator()

        # Bound-method equality: __self__ identity + __func__ identity.
        assert coord._restart_precondition == harness._merge_pipeline_idle

    def test_merge_phase_hold_is_grace_active(self, harness: Harness):
        """(task 2753) merge_phase_hold is harness._merge_phase_grace_active —
        the bounded pre-enqueue MERGE-phase hold on the force-fire path."""
        coord = harness._build_orchestrator_restart_coordinator()

        # Bound-method equality: __self__ identity + __func__ identity.
        assert coord._merge_phase_hold == harness._merge_phase_grace_active

    def test_merge_phase_grace_secs_matches_config(self, harness: Harness):
        """(task 2753) merge_phase_grace_secs is wired from
        config.orchestrator_restart_merge_phase_grace_secs."""
        coord = harness._build_orchestrator_restart_coordinator()

        assert (
            coord._merge_phase_grace_secs
            == harness.config.orchestrator_restart_merge_phase_grace_secs
        )

    def test_restart_executor_is_custom_injected(self, harness: Harness):
        """restart_executor is a custom closure — NOT the coordinator's default path.

        The default path (create_subprocess_exec + start_new_session=True) does
        NOT cgroup-escape, so it must never be used for the orchestrator's own
        self-restart (see module design note in service_restart.py).
        """
        coord = harness._build_orchestrator_restart_coordinator()

        assert coord._restart_executor is not None

    def test_git_ops_is_harness_git_ops(self, harness: Harness):
        """Coordinator receives the harness's git_ops instance."""
        coord = harness._build_orchestrator_restart_coordinator()

        assert coord._git_ops is harness.git_ops

    def test_project_root_matches_config(self, harness: Harness):
        """Coordinator project_root matches config.project_root."""
        coord = harness._build_orchestrator_restart_coordinator()

        assert coord._project_root == Path(harness.config.project_root)

    def test_min_interval_secs_matches_config_default(self, harness: Harness):
        """The orchestrator coordinator gets the config's min-interval cap (28800.0)."""
        coord = harness._build_orchestrator_restart_coordinator()

        assert coord._min_interval_secs == 28800.0

    def test_force_fire_after_secs_matches_config(self, harness: Harness):
        """The orchestrator coordinator gets the config's force-fire bound (4500.0).

        Fleet-redeploy PRD task delta (task 2398): only the orchestrator's
        own coordinator receives a non-zero force_fire_after_secs — the
        fused-memory/dashboard builders keep the 0.0 (disabled) default,
        see test_fused_memory_coordinator_has_no_force_fire and
        test_dashboard_coordinator_has_no_force_fire below.
        """
        coord = harness._build_orchestrator_restart_coordinator()

        assert coord._force_fire_after_secs == 4500.0

    def test_state_path_under_data_orchestrator(self, harness: Harness):
        """The rate-cap state file lives beside the merge-queue journal.

        Restart-safety of the cap depends on a persisted last-fire timestamp;
        it must be anchored under <project_root>/data/orchestrator/ (durable),
        NOT inside a worktree or a scratch dir.
        """
        coord = harness._build_orchestrator_restart_coordinator()

        expected = (
            Path(harness.config.project_root)
            / 'data'
            / 'orchestrator'
            / 'last_redeploy_orchestrator.json'
        )
        assert coord._state_path == expected

    def test_fused_memory_coordinator_has_no_rate_cap(self, harness: Harness):
        """fused-memory keeps the 0.0 default → its behaviour is unchanged."""
        coord = harness._build_service_restart_coordinator()

        assert coord._min_interval_secs == 0.0
        assert coord._state_path is None

    def test_dashboard_coordinator_has_no_rate_cap(self, harness: Harness):
        """dashboard keeps the 0.0 default → its behaviour is unchanged."""
        coord = harness._build_dashboard_restart_coordinator()

        assert coord._min_interval_secs == 0.0
        assert coord._state_path is None

    def test_fused_memory_coordinator_has_no_force_fire(self, harness: Harness):
        """fused-memory keeps the 0.0 default → force-fire stays disabled (byte-identical)."""
        coord = harness._build_service_restart_coordinator()

        assert coord._force_fire_after_secs == 0.0

    def test_dashboard_coordinator_has_no_force_fire(self, harness: Harness):
        """dashboard keeps the 0.0 default → force-fire stays disabled (byte-identical)."""
        coord = harness._build_dashboard_restart_coordinator()

        assert coord._force_fire_after_secs == 0.0

    def test_orchestrator_coordinator_stamp_clock_on_fire_is_false(self, harness: Harness):
        """The orchestrator's own coordinator never persists the clock on fire (task 2396).

        restart-all-orchestrators.sh becomes the SOLE on-disk clock writer,
        stamping only on its verified-fresh exit-0 path — closing the
        "failed-detached-deploy silences the backstop for 8h" hole (I2).
        """
        coord = harness._build_orchestrator_restart_coordinator()

        assert coord._stamp_clock_on_fire is False

    def test_fused_memory_coordinator_stamp_clock_on_fire_is_true(self, harness: Harness):
        """fused-memory keeps the stamp_clock_on_fire default (True) — byte-identical."""
        coord = harness._build_service_restart_coordinator()

        assert coord._stamp_clock_on_fire is True

    def test_dashboard_coordinator_stamp_clock_on_fire_is_true(self, harness: Harness):
        """dashboard keeps the stamp_clock_on_fire default (True) — byte-identical."""
        coord = harness._build_dashboard_restart_coordinator()

        assert coord._stamp_clock_on_fire is True

    def test_state_path_value_preserved_via_fleet_deploy_clock_relpath(self, harness: Harness):
        """The state_path VALUE survives the FLEET_DEPLOY_CLOCK_RELPATH refactor.

        test_state_path_under_data_orchestrator (above) already pins the
        literal path; this guards that the same value is produced once
        harness.py builds it from the single-source-of-truth constant
        (FLEET_DEPLOY_CLOCK_RELPATH) instead of an inline literal.
        """
        coord = harness._build_orchestrator_restart_coordinator()

        expected = Path(harness.config.project_root) / FLEET_DEPLOY_CLOCK_RELPATH
        assert coord._state_path == expected


# ---------------------------------------------------------------------------
# reviewer_comprehensive amendment (task 2396): stamp_clock_on_fire=False /
# orchestrator_restart_script coupling drift guard
# ---------------------------------------------------------------------------


def test_orchestrator_restart_script_is_clock_stamping_script_in_committed_config() -> None:
    """orchestrator/config.yaml's orchestrator_restart_script must be the
    clock-stamping script (scripts/restart-all-orchestrators.sh) as long as
    the orchestrator's own coordinator sets stamp_clock_on_fire=False (task
    2396) — see test_orchestrator_coordinator_stamp_clock_on_fire_is_false
    above, which pins that premise.

    _build_orchestrator_restart_coordinator hardcodes stamp_clock_on_fire=
    False for the orchestrator's own coordinator: restart-all-orchestrators.sh
    becomes the SOLE on-disk writer of the shared fleet-deploy clock.
    OrchestratorConfig's own field default ('scripts/restart-orchestrator.sh',
    see config.py) does NOT stamp that clock, and this file's ``harness``
    fixture deliberately mirrors that non-stamping default for its own,
    unrelated coverage (orchestrator_restart_script =
    'scripts/restart-orchestrator.sh') — so nothing else in this file would
    catch a real deployment drifting away from the stamping script. If that
    ever happened (a config.yaml typo, or a revert to the field default),
    BOTH the coordinator's own min_interval_secs cap and the watchdog's
    shared-clock backstop gate would silently stop advancing on
    self-redeploys, reintroducing the "backstop unaware of fleet deploys"
    hole (I1/I2) this task closes — with every test in this file still
    green. Pins the COMMITTED production config directly, mirroring
    tests/scripts/test_orchestrator_restart_config_drift.py's pattern.
    """
    cfg = yaml.safe_load(DF_CONFIG_PATH.read_text(encoding="utf-8"))
    assert cfg.get("orchestrator_restart_script") == "scripts/restart-all-orchestrators.sh", (
        "orchestrator/config.yaml's orchestrator_restart_script must stay "
        "pointed at the clock-stamping script "
        "(scripts/restart-all-orchestrators.sh): the orchestrator's own "
        "coordinator is built with stamp_clock_on_fire=False (task 2396) and "
        "relies entirely on this script to persist the shared fleet-deploy "
        "clock. Any other value silently degrades both the coordinator's own "
        "restart-safe rate cap and the watchdog's shared-clock backstop gate "
        "to a same-process-lifetime-only cap that never survives a restart."
    )


# ---------------------------------------------------------------------------
# U2 (task 1973): end-to-end — armed -> drain-gated -> systemd-run fire
# ---------------------------------------------------------------------------


@pytest.mark.asyncio
class TestOrchestratorCoordinatorEndToEnd:
    """Full wiring through the real list built by _start_merge_worker.

    Exercises the actual production path: note_merge fan-out
    (_note_merge_all) arms the orchestrator coordinator on a watched-path
    diff, and _maybe_restart_stale_service fires it via systemd-run only once
    the merge pipeline is drained (restart_precondition).
    """

    async def test_fires_systemd_run_once_pipeline_drained(self, harness: Harness):
        """note_merge arms the coordinator; a drained pipeline lets it fire via systemd-run.

        Also confirms the systemd-run argv actually threads config's
        on_active_secs/project_root/script through (not just that SOME
        non-None executor ran), and that a second fire's transient unit
        name advances via the closure's itertools counter.
        """
        harness.config.orchestrator_restart_on_merge_enabled = True
        harness.config.orchestrator_restart_debounce_secs = 0.0
        # Disable the self-redeploy rate cap (task 2371) for this test: it
        # deliberately fires TWICE in one process lifetime to prove the
        # transient-unit counter advances — a concern orthogonal to the cap,
        # which has its own dedicated coverage. With the 8h default active the
        # second fire would be (correctly) throttled, masking the counter check.
        harness.config.orchestrator_restart_min_interval_secs = 0.0

        with patch('orchestrator.merge_queue.SpeculativeMergeWorker') as mock_smw, \
             patch('asyncio.create_task'), \
             patch('orchestrator.merge_queue.check_merge_liveness_margin'):
            await harness._start_merge_worker()

        # Drained pipeline: no in-flight/verifying merge, empty queue.
        mock_smw.return_value.snapshot.return_value = {'depth': 0}
        assert harness._merge_queue.empty()

        harness.git_ops.get_merge_diff_files = AsyncMock(
            return_value=(['orchestrator/src/orchestrator/git_ops.py'], None)
        )
        orch_coord = harness._service_restart_coordinators[2]
        expected_script = str(
            Path(harness.config.project_root) / 'scripts/restart-orchestrator.sh'
        )

        await harness._note_merge_all('task-1', 'base', 'head')
        assert orch_coord.is_pending is True

        fake_proc = MagicMock()
        fake_proc.communicate = AsyncMock(return_value=(b'', None))
        fake_proc.returncode = 0
        with patch(
            'orchestrator.service_restart.asyncio.create_subprocess_exec',
            new_callable=AsyncMock,
            return_value=fake_proc,
        ) as mock_exec:
            fired = await harness._maybe_restart_stale_service(agents_idle=True)

        assert fired is True
        mock_exec.assert_awaited_once()
        pos_args = mock_exec.call_args.args
        assert pos_args[0] == 'systemd-run'
        assert '--on-active=10' in pos_args
        assert '--unit=orch-selfrestart-on-merge-0.service' in pos_args
        assert expected_script in pos_args
        assert orch_coord.is_pending is False

        # Second merge + fire: the executor closure's itertools counter must
        # advance the unit name — a fixed/hardcoded name would collide with
        # the (already --collect'd) prior unit and is the exact regression
        # the counter exists to prevent.
        await harness._note_merge_all('task-2', 'base2', 'head2')
        assert orch_coord.is_pending is True

        with patch(
            'orchestrator.service_restart.asyncio.create_subprocess_exec',
            new_callable=AsyncMock,
            return_value=fake_proc,
        ) as mock_exec_2:
            fired_2 = await harness._maybe_restart_stale_service(agents_idle=True)

        assert fired_2 is True
        pos_args_2 = mock_exec_2.call_args.args
        assert '--unit=orch-selfrestart-on-merge-1.service' in pos_args_2

    async def test_precondition_blocks_fire_when_merge_in_flight(self, harness: Harness):
        """When the merge pipeline is NOT drained (depth>0), the fire is deferred."""
        harness.config.orchestrator_restart_on_merge_enabled = True
        harness.config.orchestrator_restart_debounce_secs = 0.0

        with patch('orchestrator.merge_queue.SpeculativeMergeWorker') as mock_smw, \
             patch('asyncio.create_task'), \
             patch('orchestrator.merge_queue.check_merge_liveness_margin'):
            await harness._start_merge_worker()

        mock_smw.return_value.snapshot.return_value = {'depth': 1}  # in-flight — NOT drained

        harness.git_ops.get_merge_diff_files = AsyncMock(
            return_value=(['orchestrator/src/orchestrator/git_ops.py'], None)
        )
        orch_coord = harness._service_restart_coordinators[2]

        await harness._note_merge_all('task-1', 'base', 'head')
        assert orch_coord.is_pending is True

        with patch(
            'orchestrator.service_restart.asyncio.create_subprocess_exec',
            new_callable=AsyncMock,
        ) as mock_exec:
            fired = await harness._maybe_restart_stale_service(agents_idle=True)

        assert fired is False
        mock_exec.assert_not_awaited()
        assert orch_coord.is_pending is True  # still armed, waiting for drain

    async def test_disabled_config_never_arms(self, harness: Harness):
        """With orchestrator_restart_on_merge_enabled=False (fixture default), it never arms."""
        with patch('orchestrator.merge_queue.SpeculativeMergeWorker'), \
             patch('asyncio.create_task'), \
             patch('orchestrator.merge_queue.check_merge_liveness_margin'):
            await harness._start_merge_worker()

        orch_coord = harness._service_restart_coordinators[2]
        assert orch_coord.enabled is False

        # Re-bind (same fixture default) so pyright narrows the attribute to
        # AsyncMock in this scope — it can't see the fixture's assignment
        # from a different function — and .assert_not_awaited() is valid.
        harness.git_ops.get_merge_diff_files = AsyncMock(return_value=([], None))
        result = await orch_coord.note_merge('task-1', 'base', 'head')

        assert result is False
        assert orch_coord.is_pending is False
        harness.git_ops.get_merge_diff_files.assert_not_awaited()


# ---------------------------------------------------------------------------
# task 2237 step-21: harness escalation wiring for the self-restart executor
#
# _build_orchestrator_restart_coordinator's _systemd_run_restart_executor
# closure must build an on_failure_escalation EscalationSpec from the
# harness's live _escalation_queue and thread it through to
# schedule_detached_systemd_restart — closing the RP-4 on-failure-escalation
# gap for the REAL named consumer (proc_supervision.py itself only makes the
# gap-closure possible; nothing in production called it with a non-None
# on_failure_escalation until this wiring).
# ---------------------------------------------------------------------------


@pytest.mark.asyncio
class TestOrchestratorRestartExecutorEscalationWiring:
    """The self-restart executor closure passes an on_failure_escalation
    EscalationSpec (built from harness._escalation_queue) through to
    schedule_detached_systemd_restart — or None when no queue is wired yet."""

    async def test_passes_escalation_spec_built_from_harness_queue(
        self, harness: Harness, tmp_path: Path,
    ):
        """With a real EscalationQueue wired, the spec's queue_dir/agent_role
        match the harness's queue and summary/task_id are non-empty."""
        harness.config.orchestrator_restart_on_merge_enabled = True
        harness._escalation_queue = EscalationQueue(tmp_path / 'escalations')

        coord = harness._build_orchestrator_restart_coordinator()
        assert coord._restart_executor is not None

        spy = AsyncMock()
        with patch('orchestrator.harness.schedule_detached_systemd_restart', spy):
            await coord._restart_executor()

        spy.assert_awaited_once()
        call_kwargs = spy.call_args.kwargs
        assert 'on_failure_escalation' in call_kwargs
        spec = call_kwargs['on_failure_escalation']
        assert spec is not None
        assert spec.queue_dir == str(harness._escalation_queue.queue_dir)
        assert spec.agent_role == 'orchestrator-deterministic'
        assert spec.summary
        assert spec.task_id

    async def test_passes_none_when_escalation_queue_is_none(
        self, harness: Harness,
    ):
        """With _escalation_queue=None (bare harness / before
        _start_escalation_server runs), the closure passes
        on_failure_escalation=None without raising."""
        harness.config.orchestrator_restart_on_merge_enabled = True
        harness._escalation_queue = None

        coord = harness._build_orchestrator_restart_coordinator()
        assert coord._restart_executor is not None

        spy = AsyncMock()
        with patch('orchestrator.harness.schedule_detached_systemd_restart', spy):
            await coord._restart_executor()  # must not raise

        spy.assert_awaited_once()
        call_kwargs = spy.call_args.kwargs
        assert 'on_failure_escalation' in call_kwargs
        assert call_kwargs['on_failure_escalation'] is None


# ---------------------------------------------------------------------------
# task 2237 step-21: anti-orphan guard
#
# proc_supervision.py must be a real, importable, referenced consumer of
# service_restart.py — not an orphaned module — and the pre-2237 "accepted
# gap" docstring (which described the on-failure-escalation gap this task
# closes) must be gone.
# ---------------------------------------------------------------------------


class TestProcSupervisionAntiOrphanGuard:
    """Regression lock: service_restart.py actually delegates to
    proc_supervision.RestartPlan, and the old "accepted gap" docstring
    paragraph it used to carry is gone."""

    def test_proc_supervision_imports_cleanly(self) -> None:
        import orchestrator.proc_supervision  # noqa: F401
