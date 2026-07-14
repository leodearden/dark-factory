"""Tests for Harness's LifecycleRegistry wiring (task 2241, W10-η).

PRD ``plans/harness-supervision-prd.md`` §5.3 (LR-1/2/3): the eleven
background-loop/service lifecycles scattered across harness.py collapse
behind ONE reusable seam — ``orchestrator.background_service``'s
``LifecycleRegistry`` — so the recurring shutdown-hang class (survey 2.3;
tasks 108/161/162/169/875/1080) becomes structurally impossible.

Covers:
  step-15 RED — (a) LR-3/S2 registration-list literal: with all eleven
                enable-gates at their (True) default, harness._lifecycle's
                registered service names equal the canonical START order;
                a disabled sweep is conditionally absent.
            (b) Wiring: run() awaits start_all() during startup (before the
                recovery block) and stop_all() in the finally.

RED until step-16 GREEN adds ``Harness._build_lifecycle_registry`` /
``Harness._lifecycle``. This module accesses both defensively — inside test
bodies, never at module/class scope — so collection stays clean while they
do not yet exist.
"""
from __future__ import annotations

from pathlib import Path
from unittest.mock import AsyncMock, MagicMock, patch

import pytest

from orchestrator.config import OrchestratorConfig
from orchestrator.harness import Harness

# Canonical start-order registration list — task 2241 plan.json "HARNESS
# MIGRATION": escalation-server and merge-worker first (the recovery block
# depends on both being live); the remaining bespoke adapter (offline-lane)
# and the seven BackgroundService sweeps follow.
_CANONICAL_ORDER = [
    'escalation-server',
    'merge-worker',
    'offline-lane',
    'orphan-l0-reaper',
    'terminal-status-watcher',
    'watcher-supervisor',
    'stranded-reconcile',
    'main-tip-sweep',
    'no-landings-breaker',
    'deterministic-recon-sweep',
    'warm-lane-gc',
]


# ---------------------------------------------------------------------------
# step-15a: LR-3/S2 registration-list literal
# ---------------------------------------------------------------------------


class TestLifecycleRegistrationOrder:
    """harness._build_lifecycle_registry() wires the canonical eleven-name order."""

    def test_all_eleven_enabled_matches_canonical_order(self, tmp_path: Path) -> None:
        """A real OrchestratorConfig defaults every one of the eleven
        enable-gates to True, so the freshly built registry's services are
        exactly the canonical START order (LR-3)."""
        config = OrchestratorConfig(project_root=tmp_path)
        harness = Harness(config)

        harness._build_lifecycle_registry()

        assert harness._lifecycle is not None
        assert [s.name for s in harness._lifecycle.services] == _CANONICAL_ORDER

    def test_disabled_sweep_is_conditionally_absent(self, tmp_path: Path) -> None:
        """A disabled BackgroundService-shaped sweep is absent from the
        registry, while the rest of the canonical order is undisturbed —
        conditional registration, not a new ``enabled`` field on
        BackgroundService itself."""
        config = OrchestratorConfig(project_root=tmp_path)
        config.warm_lane_gc_enabled = False
        harness = Harness(config)

        harness._build_lifecycle_registry()

        assert harness._lifecycle is not None
        names = [s.name for s in harness._lifecycle.services]
        assert 'warm-lane-gc' not in names
        assert names == [n for n in _CANONICAL_ORDER if n != 'warm-lane-gc']


# ---------------------------------------------------------------------------
# step-15b: run() wiring — start_all() at startup, stop_all() in the finally
# ---------------------------------------------------------------------------


def _make_run_wired_harness(tmp_path: Path) -> tuple[Harness, MagicMock]:
    """Real Harness with run()'s non-lifecycle side-effects stubbed.

    Mirrors test_harness_watcher_supervisor._make_run_wired_harness /
    test_singleton_lock._build_dirty_tree_startup_harness: patches
    McpLifecycle/Scheduler/BriefingAssembler at construction so no real
    servers start, and short-circuits the dispatch loop via a sentinel
    RuntimeError from scheduler.acquire_next so run() returns (via its
    finally:) almost immediately.
    """
    config = OrchestratorConfig(project_root=tmp_path)

    with patch('orchestrator.harness.McpLifecycle') as mock_mcp_cls, \
         patch('orchestrator.harness.Scheduler'), \
         patch('orchestrator.harness.BriefingAssembler'):
        h = Harness(config)

    mock_mcp = mock_mcp_cls.return_value
    mock_mcp.start = AsyncMock()
    mock_mcp.stop = AsyncMock()

    h.git_ops = MagicMock()
    h.git_ops.has_dirty_working_tree = AsyncMock(return_value=None)
    h.git_ops.worktree_base = tmp_path / '.worktrees'

    h._dismiss_stale_escalations = AsyncMock()
    h._tag_task_modules = AsyncMock()
    h._recover_crashed_tasks = AsyncMock()
    h._reconcile_lane_checkouts = AsyncMock()
    h._reconcile_stranded_in_progress = AsyncMock()
    h._tag_prd_metadata = AsyncMock()

    h.scheduler = MagicMock()
    h.scheduler.get_tasks = AsyncMock(return_value=[{'id': '1', 'status': 'pending'}])
    h.scheduler.get_statuses = AsyncMock(return_value=({'1': 'pending'}, None))
    h.scheduler.set_task_status = AsyncMock()
    h.scheduler.acquire_next = AsyncMock(side_effect=RuntimeError('__stop_sentinel__'))

    return h, mock_mcp


class TestLifecycleRegistryRunWiring:
    """run() awaits start_all() during startup and stop_all() in the finally.

    Pre-builds harness._lifecycle and monkeypatches its start_all/stop_all to
    fresh AsyncMocks BEFORE calling run(). _build_lifecycle_registry() is
    idempotent (a no-op once harness._lifecycle is already set), so run()'s
    own internal call does not clobber the pre-built (patched) registry —
    the SAME instance, with the patched methods, is what run() awaits.
    """

    @pytest.mark.asyncio
    async def test_start_all_awaited_during_startup(self, tmp_path: Path) -> None:
        h, _mock_mcp = _make_run_wired_harness(tmp_path)
        h._build_lifecycle_registry()
        assert h._lifecycle is not None
        h._lifecycle.start_all = AsyncMock()
        h._lifecycle.stop_all = AsyncMock()

        with pytest.raises(RuntimeError, match='__stop_sentinel__'):
            await h.run(prd_path=None)

        h._lifecycle.start_all.assert_awaited_once()

    @pytest.mark.asyncio
    async def test_stop_all_awaited_in_finally(self, tmp_path: Path) -> None:
        h, _mock_mcp = _make_run_wired_harness(tmp_path)
        h._build_lifecycle_registry()
        assert h._lifecycle is not None
        h._lifecycle.start_all = AsyncMock()
        h._lifecycle.stop_all = AsyncMock()

        with pytest.raises(RuntimeError, match='__stop_sentinel__'):
            await h.run(prd_path=None)

        h._lifecycle.stop_all.assert_awaited_once()

    @pytest.mark.asyncio
    async def test_start_all_awaited_before_dismiss_stale_escalations(
        self, tmp_path: Path,
    ) -> None:
        """start_all() sits at the former escalation-server startup
        position — BEFORE the recovery block, whose first step is
        _dismiss_stale_escalations (task 2241 design decision: the
        startup-ritual collapse is safe because escalation-server +
        merge-worker are registered first-in-order and the seven sweeps are
        sleep-first)."""
        h, _mock_mcp = _make_run_wired_harness(tmp_path)
        h._build_lifecycle_registry()
        assert h._lifecycle is not None
        h._lifecycle.start_all = AsyncMock()
        h._lifecycle.stop_all = AsyncMock()

        parent = MagicMock(name='order')
        dismiss_mock = AsyncMock()
        h._dismiss_stale_escalations = dismiss_mock
        parent.attach_mock(h._lifecycle.start_all, 'start_all')
        parent.attach_mock(dismiss_mock, '_dismiss_stale_escalations')

        with pytest.raises(RuntimeError, match='__stop_sentinel__'):
            await h.run(prd_path=None)

        call_strs = [str(c) for c in parent.mock_calls]
        start_idx = next(i for i, s in enumerate(call_strs) if 'start_all' in s)
        dismiss_idx = next(
            i for i, s in enumerate(call_strs) if '_dismiss_stale_escalations' in s
        )
        assert start_idx < dismiss_idx, (
            f'start_all() must be awaited before _dismiss_stale_escalations(); '
            f'calls: {call_strs}'
        )
