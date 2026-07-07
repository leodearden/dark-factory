"""Pure-function unit tests for the RetryLedger evaluators (task 2172 step-3).

These evaluators are the pure decision cores extracted from
``_handle_no_plan_failure``, ``_check_infra_resume_thrash`` and
``_check_merge_outcome_thrash``: a :class:`RetryLedger` plus runtime inputs
in, a verdict (updated ledger + escalate decision) out. No I/O, no mocks —
config thresholds and the no-plan 2/3 constants are passed in explicitly by
the caller so these stay pure and unit-testable in isolation.
"""

from __future__ import annotations

import dataclasses

import pytest
from shared.task_metadata import RetryLedger

from orchestrator.workflow import (  # type: ignore[attr-defined]
    _evaluate_infra_resume,
    _evaluate_merge_thrash,
    _evaluate_no_plan,
    _LedgerVerdict,
)

# ---------------------------------------------------------------------------
# _evaluate_no_plan
# ---------------------------------------------------------------------------


def test_no_plan_first_failure_sets_counter_to_one_no_escalate():
    ledger = RetryLedger()
    verdict = _evaluate_no_plan(ledger, 'SHA-A')

    assert isinstance(verdict, _LedgerVerdict)
    assert verdict.ledger.consecutive_no_plan_failures == 1
    assert verdict.ledger.total_no_plan_failures == 1
    assert verdict.ledger.last_no_plan_main_sha == 'SHA-A'
    assert verdict.escalate is False


def test_no_plan_same_sha_increments_consecutive_and_escalates_at_two():
    ledger = RetryLedger(
        last_no_plan_main_sha='SHA-A',
        consecutive_no_plan_failures=1,
        total_no_plan_failures=1,
    )
    verdict = _evaluate_no_plan(ledger, 'SHA-A')

    assert verdict.ledger.consecutive_no_plan_failures == 2
    assert verdict.ledger.total_no_plan_failures == 2
    assert verdict.escalate is True
    assert verdict.trigger == 'same-SHA counter'


def test_no_plan_different_sha_resets_consecutive_to_one():
    ledger = RetryLedger(
        last_no_plan_main_sha='SHA-OLD',
        consecutive_no_plan_failures=5,
        total_no_plan_failures=1,
    )
    verdict = _evaluate_no_plan(ledger, 'SHA-NEW')

    assert verdict.ledger.consecutive_no_plan_failures == 1
    assert verdict.ledger.last_no_plan_main_sha == 'SHA-NEW'
    assert verdict.ledger.total_no_plan_failures == 2
    assert verdict.escalate is False


def test_no_plan_empty_current_sha_resets_to_one():
    ledger = RetryLedger(
        last_no_plan_main_sha='',
        consecutive_no_plan_failures=5,
        total_no_plan_failures=1,
    )
    verdict = _evaluate_no_plan(ledger, '')

    assert verdict.ledger.consecutive_no_plan_failures == 1
    assert verdict.ledger.last_no_plan_main_sha == ''


def test_no_plan_total_counter_backstops_across_changing_shas():
    """total=2 already, main moved again (new SHA) → total=3 escalates
    even though the per-SHA counter just reset to 1."""
    ledger = RetryLedger(
        last_no_plan_main_sha='SHA-B',
        consecutive_no_plan_failures=1,
        total_no_plan_failures=2,
    )
    verdict = _evaluate_no_plan(ledger, 'SHA-C')

    assert verdict.ledger.consecutive_no_plan_failures == 1
    assert verdict.ledger.total_no_plan_failures == 3
    assert verdict.escalate is True
    assert verdict.trigger == 'total counter'


def test_no_plan_same_sha_trigger_takes_precedence_over_total():
    """When both thresholds are hit simultaneously, the same-SHA label wins."""
    ledger = RetryLedger(
        last_no_plan_main_sha='SHA-A',
        consecutive_no_plan_failures=1,
        total_no_plan_failures=2,
    )
    verdict = _evaluate_no_plan(ledger, 'SHA-A')

    assert verdict.ledger.consecutive_no_plan_failures == 2
    assert verdict.ledger.total_no_plan_failures == 3
    assert verdict.escalate is True
    assert verdict.trigger == 'same-SHA counter'


# ---------------------------------------------------------------------------
# _evaluate_infra_resume
# ---------------------------------------------------------------------------


def test_infra_resume_infra_issue_no_growth_increments():
    ledger = RetryLedger(
        consecutive_infra_resume_failures=1,
        last_infra_resume_iteration_count=4,
    )
    verdict = _evaluate_infra_resume(ledger, 4, 'infra_issue', 3)

    assert verdict.ledger.consecutive_infra_resume_failures == 2
    assert verdict.ledger.last_infra_resume_iteration_count == 4
    assert verdict.escalate is False


def test_infra_resume_infra_issue_with_growth_resets_to_one():
    ledger = RetryLedger(
        consecutive_infra_resume_failures=2,
        last_infra_resume_iteration_count=4,
    )
    verdict = _evaluate_infra_resume(ledger, 7, 'infra_issue', 3)

    assert verdict.ledger.consecutive_infra_resume_failures == 1
    assert verdict.ledger.last_infra_resume_iteration_count == 7
    assert verdict.escalate is False


def test_infra_resume_non_infra_category_resets_to_zero():
    ledger = RetryLedger(
        consecutive_infra_resume_failures=2,
        last_infra_resume_iteration_count=4,
    )
    verdict = _evaluate_infra_resume(ledger, 4, 'task_failure', 3)

    assert verdict.ledger.consecutive_infra_resume_failures == 0
    # Iteration count is always refreshed, regardless of category.
    assert verdict.ledger.last_infra_resume_iteration_count == 4
    assert verdict.escalate is False


def test_infra_resume_none_category_resets_to_zero():
    ledger = RetryLedger(consecutive_infra_resume_failures=1)
    verdict = _evaluate_infra_resume(ledger, 9, None, 3)

    assert verdict.ledger.consecutive_infra_resume_failures == 0
    assert verdict.ledger.last_infra_resume_iteration_count == 9


def test_infra_resume_escalates_at_threshold():
    ledger = RetryLedger(
        consecutive_infra_resume_failures=2,
        last_infra_resume_iteration_count=4,
    )
    verdict = _evaluate_infra_resume(ledger, 4, 'infra_issue', 3)

    assert verdict.ledger.consecutive_infra_resume_failures == 3
    assert verdict.escalate is True


def test_infra_resume_threshold_is_configurable():
    ledger = RetryLedger(
        consecutive_infra_resume_failures=1,
        last_infra_resume_iteration_count=4,
    )
    verdict = _evaluate_infra_resume(ledger, 4, 'infra_issue', 2)

    assert verdict.ledger.consecutive_infra_resume_failures == 2
    assert verdict.escalate is True


def test_infra_resume_verdict_exposes_trigger_field():
    ledger = RetryLedger()
    verdict = _evaluate_infra_resume(ledger, 0, None, 3)

    assert isinstance(verdict.trigger, str)


# ---------------------------------------------------------------------------
# _evaluate_merge_thrash
# ---------------------------------------------------------------------------


def test_merge_thrash_matching_signature_increments():
    ledger = RetryLedger(
        consecutive_merge_thrash=1,
        last_merge_outcome_signature='sig-abc',
    )
    verdict = _evaluate_merge_thrash(ledger, 'sig-abc', 'sig-abc', 2)

    assert verdict.ledger.consecutive_merge_thrash == 2
    assert verdict.ledger.last_merge_outcome_signature == 'sig-abc'
    assert verdict.escalate is True


def test_merge_thrash_differing_signature_resets_to_one():
    ledger = RetryLedger(
        consecutive_merge_thrash=1,
        last_merge_outcome_signature='sig-abc',
    )
    verdict = _evaluate_merge_thrash(ledger, 'sig-abc', 'sig-xyz', 2)

    assert verdict.ledger.consecutive_merge_thrash == 1
    assert verdict.ledger.last_merge_outcome_signature == 'sig-xyz'
    assert verdict.escalate is False


def test_merge_thrash_none_prev_signature_resets_to_one():
    ledger = RetryLedger()
    verdict = _evaluate_merge_thrash(ledger, None, 'sig-1', 2)

    assert verdict.ledger.consecutive_merge_thrash == 1
    assert verdict.ledger.last_merge_outcome_signature == 'sig-1'
    assert verdict.escalate is False


def test_merge_thrash_threshold_is_configurable():
    ledger = RetryLedger(
        consecutive_merge_thrash=1,
        last_merge_outcome_signature='sig-abc',
    )
    verdict = _evaluate_merge_thrash(ledger, 'sig-abc', 'sig-abc', 3)

    assert verdict.ledger.consecutive_merge_thrash == 2
    assert verdict.escalate is False


def test_merge_thrash_verdict_exposes_trigger_field():
    ledger = RetryLedger()
    verdict = _evaluate_merge_thrash(ledger, None, 'sig', 2)

    assert isinstance(verdict.trigger, str)


# ---------------------------------------------------------------------------
# Verdict shape
# ---------------------------------------------------------------------------


def test_verdict_ledger_is_a_retry_ledger_instance_and_input_not_mutated():
    ledger = RetryLedger(consecutive_no_plan_failures=1)
    verdict = _evaluate_no_plan(ledger, 'SHA-A')

    assert isinstance(verdict.ledger, RetryLedger)
    assert ledger.consecutive_no_plan_failures == 1, 'input ledger must not be mutated'


def test_ledger_verdict_is_frozen():
    ledger = RetryLedger()
    verdict = _evaluate_no_plan(ledger, 'SHA-A')

    with pytest.raises(dataclasses.FrozenInstanceError):
        verdict.escalate = False  # type: ignore[misc]
