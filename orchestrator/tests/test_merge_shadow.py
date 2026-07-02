"""Tests for orchestrator.merge_shadow: extracted warm-vs-cold shadow-compare
subsystem (MQ-refactor task γ).

These tests encode the behavior-preserving contracts of the module split,
mirroring task β's test_merge_gates.py:

1. Module-existence — ``orchestrator.merge_shadow`` exists and exports the
   full closure of moved symbols (state/diff types, per-test parsers,
   sentinels, and the shadow-compare functions).
2. Logger-name — the module logs under the ``orchestrator.merge_queue``
   logger name (not ``orchestrator.merge_shadow``) so existing ``caplog``
   assertions filtered to the merge_queue logger keep capturing the moved
   shadow-compare's WARNING/INFO lines.
3. Reach-back / string-path monkeypatch routing — the existing test suite
   monkeypatches shadow-compare dependencies by STRING PATH
   ``orchestrator.merge_queue.<name>``.  A moved function must resolve a
   monkeypatched-or-staying sibling via a function-local deferred import so
   those patches stay effective even though the function body now lives in
   this module.  (added in a later step)
4. Shim re-export identity (added in a later step, once merge_queue.py's
   shim swap lands).
"""

from __future__ import annotations


def test_merge_shadow_exports_moved_public_symbols() -> None:
    from orchestrator.merge_shadow import (
        _LIBTEST_TEST_LINE_RE,
        _NEXTEST_SUMMARY_LINE_RE,
        _NEXTEST_TEST_LINE_RE,
        _WARM_COLD_SHADOW_SENTINEL,
        _WARM_COLD_SHADOW_UNPARSEABLE_SENTINEL,
        ShadowCompareDiff,
        ShadowCompareState,
        _alarm_warm_shadow_unparseable,
        _classify_test_status,
        _load_shadow_compare_state,
        _maybe_schedule_shadow_compare,
        _nextest_reported_test_count,
        _persistent_alarm_tests,
        _run_cold_shadow_verify,
        _run_shadow_compare,
        _save_shadow_compare_state,
        _shadow_compare_due,
        _submit_shadow_divergence_escalation,
        diff_per_test_results,
        parse_per_test_results,
    )

    for name, obj in {
        'ShadowCompareState': ShadowCompareState,
        'ShadowCompareDiff': ShadowCompareDiff,
        'parse_per_test_results': parse_per_test_results,
        '_classify_test_status': _classify_test_status,
        '_nextest_reported_test_count': _nextest_reported_test_count,
        '_NEXTEST_TEST_LINE_RE': _NEXTEST_TEST_LINE_RE,
        '_LIBTEST_TEST_LINE_RE': _LIBTEST_TEST_LINE_RE,
        '_NEXTEST_SUMMARY_LINE_RE': _NEXTEST_SUMMARY_LINE_RE,
        'diff_per_test_results': diff_per_test_results,
        '_persistent_alarm_tests': _persistent_alarm_tests,
        '_load_shadow_compare_state': _load_shadow_compare_state,
        '_save_shadow_compare_state': _save_shadow_compare_state,
        '_shadow_compare_due': _shadow_compare_due,
        '_WARM_COLD_SHADOW_SENTINEL': _WARM_COLD_SHADOW_SENTINEL,
        '_WARM_COLD_SHADOW_UNPARSEABLE_SENTINEL': _WARM_COLD_SHADOW_UNPARSEABLE_SENTINEL,
        '_submit_shadow_divergence_escalation': _submit_shadow_divergence_escalation,
        '_alarm_warm_shadow_unparseable': _alarm_warm_shadow_unparseable,
        '_run_cold_shadow_verify': _run_cold_shadow_verify,
        '_run_shadow_compare': _run_shadow_compare,
        '_maybe_schedule_shadow_compare': _maybe_schedule_shadow_compare,
    }.items():
        assert obj is not None, f'{name} must not be None'


def test_merge_shadow_logger_name_is_merge_queue() -> None:
    """merge_shadow emits under the 'orchestrator.merge_queue' logger name.

    RED (pre-module): ``orchestrator.merge_shadow`` does not exist yet.

    Required so existing ``caplog.at_level(..., logger='orchestrator.merge_queue')``
    assertions in the warm/cold shadow-compare test files keep capturing the
    moved functions' WARNING/INFO-level messages after relocation.
    """
    import orchestrator.merge_shadow as merge_shadow

    assert merge_shadow.logger.name == 'orchestrator.merge_queue'
