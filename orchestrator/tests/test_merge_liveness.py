"""Tests for orchestrator.merge_liveness: extracted startup liveness-margin
guard, verify-host-unreachable alarms, and persistent-worktree operational
guards (MQ-refactor task γ).

These tests encode the behavior-preserving contracts of the module split,
mirroring task β's test_merge_gates.py:

1. Module-existence — ``orchestrator.merge_liveness`` exists and exports the
   full closure of moved symbols (liveness-margin types/functions,
   verify-host-unreachable alarm helpers, and persistent-worktree guards).
2. Logger-name — the module logs under the ``orchestrator.merge_queue``
   logger name (not ``orchestrator.merge_liveness``) so existing ``caplog``
   assertions filtered to the merge_queue logger keep capturing the moved
   guards' WARNING-level messages.
3. Reach-back / string-path monkeypatch routing — the existing test suite
   monkeypatches liveness-guard dependencies (including two module-level
   CONSTANTS that stay behind in merge_queue.py) by STRING PATH
   ``orchestrator.merge_queue.<name>``.  A moved function must resolve a
   monkeypatched-or-staying sibling via a function-local deferred import so
   those patches stay effective even though the function body now lives in
   this module.  Each ``TestReachBackRouting`` test below patches BOTH
   namespaces (merge_liveness-local naive vs. merge_queue reach-back
   target) with CONTRASTING behaviour.  Every target in this module is a
   SYNC function, so — unlike merge_shadow / merge_drift — this class
   carries no ``@pytest.mark.asyncio`` marker.  Also covers the
   engine-constant default-argument hazard's OTHER half:
   ``enforce_persistent_worktree_serial_lane``'s ``_MERGE_AHEAD_BOUND``
   None-default (design_decisions #3 in plan.json names both
   ``liveness_secs``/``INFLIGHT_MERGE_WORKTREE_LIVENESS_SECS`` and
   ``merge_ahead_bound``/``_MERGE_AHEAD_BOUND``; the step-3 plan prose only
   spelled out the former by name).
4. Shim re-export identity (added in a later step, once merge_queue.py's
   shim swap lands).
"""

from __future__ import annotations

from pathlib import Path
from unittest.mock import MagicMock, patch

import pytest

from orchestrator.config import GitConfig, OrchestratorConfig


def test_merge_liveness_exports_moved_public_symbols() -> None:
    from orchestrator.merge_liveness import (
        _MERGE_WORKER_LOOP_DIED_SENTINEL,
        _VERIFY_HOST_RECOVERED_SENTINEL_PREFIX,
        _VERIFY_HOST_UNREACHABLE_SENTINEL_PREFIX,
        MergeLivenessAssessment,
        MergeLivenessConfigError,
        PersistentWorktreeConfigError,
        _acquire_warm_verify_worktree,
        _alarm_verify_host_unreachable,
        _clear_verify_host_unreachable,
        _safety_valve_due,
        _verify_host_unreachable_sentinel,
        check_merge_liveness_margin,
        enforce_merge_liveness_margin,
        enforce_persistent_worktree_serial_lane,
    )

    for name, obj in {
        'MergeLivenessAssessment': MergeLivenessAssessment,
        'check_merge_liveness_margin': check_merge_liveness_margin,
        'MergeLivenessConfigError': MergeLivenessConfigError,
        'enforce_merge_liveness_margin': enforce_merge_liveness_margin,
        'PersistentWorktreeConfigError': PersistentWorktreeConfigError,
        '_safety_valve_due': _safety_valve_due,
        '_VERIFY_HOST_UNREACHABLE_SENTINEL_PREFIX': _VERIFY_HOST_UNREACHABLE_SENTINEL_PREFIX,
        '_VERIFY_HOST_RECOVERED_SENTINEL_PREFIX': _VERIFY_HOST_RECOVERED_SENTINEL_PREFIX,
        '_MERGE_WORKER_LOOP_DIED_SENTINEL': _MERGE_WORKER_LOOP_DIED_SENTINEL,
        '_verify_host_unreachable_sentinel': _verify_host_unreachable_sentinel,
        '_alarm_verify_host_unreachable': _alarm_verify_host_unreachable,
        '_clear_verify_host_unreachable': _clear_verify_host_unreachable,
        '_acquire_warm_verify_worktree': _acquire_warm_verify_worktree,
        'enforce_persistent_worktree_serial_lane': enforce_persistent_worktree_serial_lane,
    }.items():
        assert obj is not None, f'{name} must not be None'


def test_merge_liveness_logger_name_is_merge_queue() -> None:
    """merge_liveness emits under the 'orchestrator.merge_queue' logger name.

    RED (pre-module): ``orchestrator.merge_liveness`` does not exist yet.

    Required so existing ``caplog.at_level(..., logger='orchestrator.merge_queue')``
    assertions in test_merge_queue.py keep capturing the moved guards'
    WARNING-level messages after relocation.
    """
    import orchestrator.merge_liveness as merge_liveness

    assert merge_liveness.logger.name == 'orchestrator.merge_queue'


class TestReachBackRouting:
    """Reach-back / string-path monkeypatch routing contract.

    Every target below is a SYNC function; see the module docstring for why
    this class carries no ``@pytest.mark.asyncio`` marker.
    """

    def test_enforce_merge_liveness_margin_reachback_to_check_merge_liveness_margin(
        self, tmp_path: Path,
    ) -> None:
        """(6) enforce_merge_liveness_margin must resolve check_merge_liveness_margin
        via orchestrator.merge_queue, not the co-located merge_liveness copy."""
        from orchestrator.merge_liveness import (
            MergeLivenessAssessment,
            MergeLivenessConfigError,
            enforce_merge_liveness_margin,
        )

        cfg = OrchestratorConfig(project_root=tmp_path)

        # Naive-resolution target: reports safe → would NOT raise.
        naive_safe = MergeLivenessAssessment(
            worst_case_secs=1.0, threshold_secs=100.0, liveness_secs=10800.0,
            heartbeat_poll_secs=30.0, touch_miss_tolerance=20,
            safety_factor=0.75, safe=True,
        )
        # Reach-back target: reports UNSAFE → must raise MergeLivenessConfigError.
        reachback_unsafe = MergeLivenessAssessment(
            worst_case_secs=999999.0, threshold_secs=100.0, liveness_secs=10800.0,
            heartbeat_poll_secs=30.0, touch_miss_tolerance=20,
            safety_factor=0.75, safe=False,
        )

        with (
            patch(
                'orchestrator.merge_liveness.check_merge_liveness_margin',
                MagicMock(return_value=naive_safe),
            ),
            patch(
                'orchestrator.merge_queue.check_merge_liveness_margin',
                MagicMock(return_value=reachback_unsafe),
            ),
            pytest.raises(MergeLivenessConfigError),
        ):
            enforce_merge_liveness_margin(cfg)

    def test_check_merge_liveness_margin_reachback_to_heartbeat_constants(
        self, tmp_path: Path,
    ) -> None:
        """(7a) check_merge_liveness_margin must read _HEARTBEAT_POLL_S and
        TOUCH_MISS_TOLERANCE from orchestrator.merge_queue at call time."""
        from orchestrator.merge_liveness import check_merge_liveness_margin

        cfg = OrchestratorConfig(project_root=tmp_path)

        with (
            patch('orchestrator.merge_queue._HEARTBEAT_POLL_S', 1000.0),
            patch('orchestrator.merge_queue.TOUCH_MISS_TOLERANCE', 1000),
        ):
            result = check_merge_liveness_margin(cfg, liveness_secs=10800.0)

        assert result.worst_case_secs == 1000.0 * 1000, (
            f'expected worst_case_secs to reflect the orchestrator.merge_queue-patched '
            f'_HEARTBEAT_POLL_S x TOUCH_MISS_TOLERANCE (1,000,000), got '
            f'{result.worst_case_secs!r}'
        )
        assert result.safe is False, (
            f'floor of 1,000,000s must blow well past the threshold; got safe={result.safe!r}'
        )

    def test_check_merge_liveness_margin_reachback_to_inflight_liveness_default(
        self, tmp_path: Path,
    ) -> None:
        """(7b) check_merge_liveness_margin's None-default liveness_secs must
        resolve INFLIGHT_MERGE_WORKTREE_LIVENESS_SECS from orchestrator.merge_queue."""
        from orchestrator.merge_liveness import check_merge_liveness_margin

        cfg = OrchestratorConfig(project_root=tmp_path)

        with patch('orchestrator.merge_queue.INFLIGHT_MERGE_WORKTREE_LIVENESS_SECS', 600.0):
            result = check_merge_liveness_margin(cfg)  # liveness_secs omitted → None-default path

        assert result.liveness_secs == 600.0, (
            f'expected the None-default liveness_secs to resolve via the '
            f'orchestrator.merge_queue-patched INFLIGHT_MERGE_WORKTREE_LIVENESS_SECS, '
            f'got {result.liveness_secs!r}'
        )

    def test_enforce_persistent_worktree_serial_lane_reachback_to_merge_ahead_bound_default(
        self, tmp_path: Path,
    ) -> None:
        """enforce_persistent_worktree_serial_lane's None-default merge_ahead_bound
        must resolve _MERGE_AHEAD_BOUND from orchestrator.merge_queue (same
        def-time-constant hazard as liveness_secs above; design_decisions #3)."""
        from orchestrator.merge_liveness import (
            PersistentWorktreeConfigError,
            enforce_persistent_worktree_serial_lane,
        )

        cfg = OrchestratorConfig(
            project_root=tmp_path,
            git=GitConfig(persistent_merge_worktree=True),
        )

        with (
            patch('orchestrator.merge_queue._MERGE_AHEAD_BOUND', 5),
            pytest.raises(PersistentWorktreeConfigError),
        ):
            # merge_ahead_bound omitted → None-default path; num_hosts=1 →
            # per_host=ceil(5/1)=5 > 1 → raises.
            enforce_persistent_worktree_serial_lane(cfg)
