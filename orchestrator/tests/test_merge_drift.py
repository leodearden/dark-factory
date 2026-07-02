"""Tests for orchestrator.merge_drift: extracted drift-check detective
subsystem (MQ-refactor task γ).

These tests encode the behavior-preserving contracts of the module split,
mirroring task β's test_merge_gates.py:

1. Module-existence — ``orchestrator.merge_drift`` exists and exports the
   full closure of moved symbols (the drift-check runner and its cadence
   gate).
2. Logger-name — the module logs under the ``orchestrator.merge_queue``
   logger name (not ``orchestrator.merge_drift``) so existing ``caplog``
   assertions filtered to the merge_queue logger keep capturing the moved
   drift-check's WARNING-level fail-open messages.
3. Reach-back / string-path monkeypatch routing — the existing test suite
   monkeypatches drift-check dependencies by STRING PATH
   ``orchestrator.merge_queue.<name>``.  A moved function must resolve a
   monkeypatched-or-staying sibling via a function-local deferred import so
   those patches stay effective even though the function body now lives in
   this module.  (added in a later step)
4. Shim re-export identity (added in a later step, once merge_queue.py's
   shim swap lands).
"""

from __future__ import annotations


def test_merge_drift_exports_moved_public_symbols() -> None:
    from orchestrator.merge_drift import _maybe_run_drift_check, _run_drift_check

    for name, obj in {
        '_run_drift_check': _run_drift_check,
        '_maybe_run_drift_check': _maybe_run_drift_check,
    }.items():
        assert obj is not None, f'{name} must not be None'


def test_merge_drift_logger_name_is_merge_queue() -> None:
    """merge_drift emits under the 'orchestrator.merge_queue' logger name.

    RED (pre-module): ``orchestrator.merge_drift`` does not exist yet.

    Required so existing ``caplog.at_level(..., logger='orchestrator.merge_queue')``
    assertions in test_merge_queue_multihost_wiring.py keep capturing the
    moved drift-check's WARNING-level fail-open messages after relocation.
    """
    import orchestrator.merge_drift as merge_drift

    assert merge_drift.logger.name == 'orchestrator.merge_queue'
