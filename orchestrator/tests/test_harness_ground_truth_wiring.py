"""Tests for Harness._get_ground_truth() — the lazy TaskGroundTruth wiring
seam (task 2243, W10-θ2).

``TaskGroundTruth`` (θ1, orchestrator/src/orchestrator/task_ground_truth.py)
captures its ``escalation_queue`` collaborator at CONSTRUCTION time.
``Harness._escalation_queue`` is ``None`` at ``__init__`` and is only
populated later, during ``run()`` startup (harness.py:7115). A resolver
built eagerly in ``__init__`` — or memoized the first time it's asked for,
with no refresh — would freeze that startup-time ``None`` forever, silently
breaking every open-escalation-aware recovery row (`_RECOVERY` rows f/g/h)
for the rest of the process lifetime. ``_get_ground_truth()`` must therefore
build (or rebuild) lazily, always capturing the LIVE queue. This module pins
that "frozen-None trap".
"""

from __future__ import annotations

from unittest.mock import ANY, AsyncMock, MagicMock, patch

import pytest

from orchestrator.harness import _RECONCILE_HEARTBEAT_TTL, Harness
from orchestrator.task_ground_truth import (
    BranchState,
    BranchStateKind,
    RecoveryAction,
    TaskGroundTruth,
    TruthReport,
)

# ---------------------------------------------------------------------------
# Harness fixture (Pattern A — mirrors test_reconcile_stranded.py's `harness`
# fixture, minus the git_ops/scheduler behavioural rewiring that only the
# reconcile-sweep tests need).
# ---------------------------------------------------------------------------


@pytest.fixture
def harness(mock_orch_config):
    """Create a bare Harness with mocked internals for unit testing wiring."""
    with patch('orchestrator.harness.McpLifecycle'), \
         patch('orchestrator.harness.Scheduler'), \
         patch('orchestrator.harness.BriefingAssembler'):
        h = Harness(mock_orch_config)
    return h


class TestGetGroundTruthWiring:
    def test_escalation_queue_is_none_at_construction(self, harness: Harness):
        """Precondition this whole module pins against: __init__ leaves
        _escalation_queue at None; it is only set later, during run()."""
        assert harness._escalation_queue is None

    def test_returns_a_task_ground_truth_instance(self, harness: Harness):
        harness._escalation_queue = MagicMock(name='live_escalation_queue')

        resolver = harness._get_ground_truth()

        assert isinstance(resolver, TaskGroundTruth)

    def test_captures_live_escalation_queue_not_init_time_none(self, harness: Harness):
        """The frozen-None trap, direct pin: a resolver built while
        _escalation_queue is still None (e.g. a call that races startup)
        must NOT permanently freeze that None. Once run() populates the
        live queue, the next call must pick it up."""
        stale = harness._get_ground_truth()
        assert stale.escalation_queue is None  # built before startup populated it

        live_queue = MagicMock(name='live_escalation_queue')
        harness._escalation_queue = live_queue

        fresh = harness._get_ground_truth()

        assert fresh.escalation_queue is live_queue

    def test_captures_git_ops_scheduler_and_worktree_resolver(self, harness: Harness):
        harness._escalation_queue = MagicMock(name='live_escalation_queue')

        resolver = harness._get_ground_truth()

        assert resolver.git_ops is harness.git_ops
        assert resolver.scheduler is harness.scheduler
        assert resolver.worktree_resolver == harness._resolve_task_worktree

    def test_heartbeat_ttl_matches_reconcile_constant(self, harness: Harness):
        """No dedicated OrchestratorConfig field exists for heartbeat_ttl (see
        harness.py's _RECONCILE_HEARTBEAT_TTL docstring) — the resolver is
        bound to that hardcoded module constant, not sourced from config.
        Pin against the imported constant (not a literal timedelta) so a
        future change to the constant can't silently drift from this test."""
        harness._escalation_queue = MagicMock(name='live_escalation_queue')

        resolver = harness._get_ground_truth()

        assert resolver.heartbeat_ttl == _RECONCILE_HEARTBEAT_TTL

    def test_memoizes_across_calls_once_queue_is_stable(self, harness: Harness):
        """Once _escalation_queue is stable (the steady-state post-startup
        case), repeated calls must return the SAME resolver instance rather
        than rebuilding on every reconcile check."""
        harness._escalation_queue = MagicMock(name='live_escalation_queue')

        first = harness._get_ground_truth()
        second = harness._get_ground_truth()

        assert first is second


# ---------------------------------------------------------------------------
# task 2243 (W10-θ2) step-17 — delegation + registration parity.
#
# (a) Delegation: the migrated sweep must route its ENTIRE mark-done
#     decision through _get_ground_truth().recovery_for(tid) — it must never
#     independently re-derive branch state via git_ops.is_ancestor /
#     git_ops.find_merge_marker itself.  Proven here with a STUBBED resolver
#     (unlike test_reconcile_stranded.py's G1/G2, which drive the REAL
#     resolver against a bound MergeProvenance journal row / git fallback):
#     even when the resolver reports MARK_DONE_WITH_PROVENANCE on an
#     on-main/journal-hit shape, the sweep code path itself never touches
#     git_ops.is_ancestor/find_merge_marker directly — this is a structural
#     delegation pin, independent of the real resolver's own journal-first
#     short-circuit (which is what G1/G2 pin instead).
# ---------------------------------------------------------------------------


class TestReconcileSweepDelegatesToGroundTruth:
    @pytest.mark.asyncio
    async def test_mark_done_dispatch_never_awaits_git_archaeology_directly(
        self, harness: Harness,
    ):
        tid = '9101'
        sha = 'c3' * 20
        harness._escalation_queue = MagicMock(name='live_escalation_queue')

        harness.scheduler.get_statuses = AsyncMock(  # type: ignore[attr-defined]
            return_value=({tid: 'in-progress'}, None),
        )
        harness.scheduler.get_task = AsyncMock(  # type: ignore[attr-defined]
            return_value={'status': 'in-progress', 'metadata': {}},
        )
        harness.scheduler.mark_done = AsyncMock()  # type: ignore[attr-defined]

        # Never configured with a return value that would let the sweep
        # "accidentally" produce the right answer by calling these itself —
        # any await against either proves a direct (non-delegated) call.
        harness.git_ops.is_ancestor = AsyncMock()  # type: ignore[attr-defined]
        harness.git_ops.find_merge_marker = AsyncMock()  # type: ignore[attr-defined]
        harness.git_ops.warm_lane_ref_is_degenerate = AsyncMock(  # type: ignore[attr-defined]
            return_value=False,
        )
        # task 2500 FIX 1 — commit_effect_present_in_main is a legitimate
        # applier-side sibling check to warm_lane_ref_is_degenerate above
        # (same "sweep-side refinement" pattern), so it IS expected to be
        # called on the ON_MAIN mark-done path; give it a real return value
        # so the flip is reached, mirroring warm_lane_ref_is_degenerate.
        harness.git_ops.commit_effect_present_in_main = AsyncMock(  # type: ignore[attr-defined]
            return_value=True,
        )

        report = TruthReport(
            db_status='in-progress',
            live_claimant=None,
            branch_state=BranchState(BranchStateKind.ON_MAIN, sha),
            worktree_present=False,
            open_escalations=[],
            deploy_phase=None,
        )
        fake_resolver = MagicMock(name='fake_ground_truth')
        fake_resolver.recovery_for = AsyncMock(
            return_value=(report, RecoveryAction.MARK_DONE_WITH_PROVENANCE),
        )
        harness._get_ground_truth = MagicMock(return_value=fake_resolver)  # type: ignore[method-assign]

        await harness._reconcile_stranded_in_progress()

        fake_resolver.recovery_for.assert_awaited_once_with(tid)
        harness.scheduler.mark_done.assert_awaited_once_with(  # type: ignore[attr-defined]
            tid, kind='found_on_main', sha=sha, note=ANY,
        )
        harness.git_ops.is_ancestor.assert_not_awaited()  # type: ignore[attr-defined]
        harness.git_ops.find_merge_marker.assert_not_awaited()  # type: ignore[attr-defined]


# ---------------------------------------------------------------------------
# task 2243 (W10-θ2) step-17(b) — LifecycleRegistry registration parity.
#
# η (task 2241) already registers the three loop sweeps this task migrates
# ('stranded-reconcile', 'main-tip-sweep', 'deterministic-recon-sweep') as
# BackgroundServices in _build_lifecycle_registry (harness.py, commit
# d4c714d900) — test_harness_lifecycle_registry.py's
# test_all_eleven_enabled_matches_canonical_order already covers the full
# eleven-name canonical order as part of η's own suite.  This is θ2's own
# narrower confirmation, scoped to just the three sweeps this task's
# migration concerns (the analysis's "confirm LifecycleRegistry (η)
# registration" — a parity assertion, not new registration).
# ---------------------------------------------------------------------------


class TestLifecycleRegistryReconcileSweepsRegistered:
    def test_three_migrated_sweeps_registered_in_lifecycle_registry(
        self, mock_orch_config,
    ):
        mock_orch_config.stranded_reconcile_enabled = True
        mock_orch_config.main_tip_sweep_enabled = True
        mock_orch_config.deterministic_recon_sweep_enabled = True
        mock_orch_config.deterministic_recon_sweep_interval_secs = 900.0

        with patch('orchestrator.harness.McpLifecycle'), \
             patch('orchestrator.harness.Scheduler'), \
             patch('orchestrator.harness.BriefingAssembler'):
            h = Harness(mock_orch_config)

        h._build_lifecycle_registry()

        assert h._lifecycle is not None
        names = {s.name for s in h._lifecycle.services}
        assert {
            'stranded-reconcile', 'main-tip-sweep', 'deterministic-recon-sweep',
        } <= names
