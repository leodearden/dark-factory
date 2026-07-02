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
   this module.  (added in a later step)
4. Shim re-export identity (added in a later step, once merge_queue.py's
   shim swap lands).
"""

from __future__ import annotations


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
