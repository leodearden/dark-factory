"""The record of one Abort-trigger-3 firing: its event payload and its blocked reason.

Every test drives ``NoProgressAbort`` and ``emit_no_progress_abort`` through
their public API alone -- no git, no worker -- so the contract both durable
surfaces depend on is pinned where it is defined.
"""
from __future__ import annotations

import dataclasses
import json

import pytest
from _recording_event_store import _RecordingEventStore

from orchestrator.event_store import EventType
from orchestrator.merge_lane.no_progress_abort import (
    NoProgressAbort,
    emit_no_progress_abort,
)

_EVENT_KEYS = {
    'request_id',
    'lease_kind',
    'runner',
    'no_progress_secs',
    'budget_secs',
    'strike',
    'max_strikes',
    'capped',
    'dispatch_returned',
}


def _abort(**overrides) -> NoProgressAbort:
    fields = {
        'task_id': 't1',
        'request_id': 'mr-1',
        'runner': 'local',
        'is_local': True,
        'no_progress_secs': 5407.5,
        'budget_secs': 5400.0,
        'strike': 1,
        'max_strikes': 3,
        'dispatch_seen_live': False,
    }
    fields.update(overrides)
    return NoProgressAbort(**fields)


def _remote(**overrides) -> NoProgressAbort:
    return _abort(runner='laptop', is_local=False, **overrides)


def _every_lease_variant(**overrides) -> list[NoProgressAbort]:
    return [
        _abort(**overrides),
        _remote(dispatch_seen_live=True, **overrides),
        _remote(dispatch_seen_live=False, **overrides),
    ]


class TestEventData:
    def test_local_abort_carries_every_key_with_local_values(self):
        data = _abort().event_data()

        assert set(data) == _EVENT_KEYS
        assert data['request_id'] == 'mr-1'
        assert data['lease_kind'] == 'local'
        assert data['runner'] == 'local'
        assert data['no_progress_secs'] == 5407.5
        assert data['budget_secs'] == 5400.0
        assert data['strike'] == 1
        assert data['max_strikes'] == 3
        assert data['capped'] is False
        assert data['dispatch_returned'] is None
        assert json.loads(json.dumps(data)) == data

    def test_remote_abort_reports_whether_a_dispatch_was_seen_in_flight(self):
        returned = _remote(dispatch_seen_live=True).event_data()
        never_seen = _remote(dispatch_seen_live=False).event_data()

        assert returned['lease_kind'] == 'remote'
        assert returned['runner'] == 'laptop'
        assert returned['dispatch_returned'] is True
        assert never_seen['lease_kind'] == 'remote'
        assert never_seen['dispatch_returned'] is False

    def test_local_abort_has_no_dispatch_to_report(self):
        abort = _abort(dispatch_seen_live=True)

        assert abort.dispatch_returned is None
        assert abort.event_data()['dispatch_returned'] is None


class TestCapped:
    def test_a_strike_below_the_cap_is_not_capped(self):
        assert _abort(strike=1, max_strikes=3).capped is False

    def test_the_strike_that_reaches_the_cap_is_capped(self):
        assert _abort(strike=3, max_strikes=3).capped is True

    def test_a_strike_past_a_lowered_cap_is_capped(self):
        assert _abort(strike=4, max_strikes=2).capped is True


class TestTerminalReason:
    def test_reason_states_the_budget_the_strike_count_and_the_lease(self):
        local = _abort(strike=3).terminal_reason()
        remote = _remote(strike=3).terminal_reason()

        assert '5400s' in local
        assert '3 consecutive' in local
        assert 'local' in local
        assert 'remote' in remote

    def test_a_fractional_budget_renders_without_trailing_zeros(self):
        reason = _abort(budget_secs=0.2, no_progress_secs=0.25).terminal_reason()

        assert '0.2s' in reason

    def test_each_lease_variant_renders_a_distinct_reason(self):
        reasons = [a.terminal_reason() for a in _every_lease_variant(strike=3)]

        assert len(set(reasons)) == 3

    def test_reason_makes_no_dead_or_hung_diagnosis(self):
        for abort in _every_lease_variant(strike=3):
            reason = abort.terminal_reason().lower()
            assert 'dead' not in reason
            assert 'hung' not in reason

    def test_reason_is_identical_across_measured_durations(self):
        """RetryLedger.compute_merge_outcome_signature hashes this reason.

        shared/src/shared/task_metadata.py::RetryLedger.compute_merge_outcome_signature
        falls back to the normalised reason when the outcome carries no
        failure_category or cause_hint, which a no-progress cap-out never
        does. A jittery measurement in the reason would give every cap-out a
        fresh signature and defeat consecutive_merge_thrash.
        """
        early = _abort(strike=3, no_progress_secs=5401.0)
        late = _abort(strike=3, no_progress_secs=5409.9)

        assert early.terminal_reason() == late.terminal_reason()

    def test_reason_avoids_the_review_category_substrings(self):
        """orchestrator/src/orchestrator/workflow.py::TaskWorkflow infers the review category by substring.

        Its blocked fall-through files 'verification failed' as
        post_merge_verify and 'ff' or 'advanced' as merge_ff_failed; anything
        else is merge_error, which is what a no-progress cap-out is.
        """
        for abort in _every_lease_variant(strike=3):
            reason = abort.terminal_reason().lower()
            assert 'ff' not in reason
            assert 'advanced' not in reason
            assert 'verification failed' not in reason


class TestInvariant:
    def test_a_zero_strike_is_rejected(self):
        with pytest.raises(ValueError, match='strike'):
            _abort(strike=0)

    def test_an_abort_before_the_budget_elapsed_is_rejected_naming_both_values(self):
        with pytest.raises(ValueError) as excinfo:
            _abort(no_progress_secs=5399.5, budget_secs=5400.0)

        message = str(excinfo.value)
        assert '5399.5' in message
        assert '5400' in message

    def test_the_record_is_frozen(self):
        abort = _abort()

        with pytest.raises(dataclasses.FrozenInstanceError):
            abort.strike = 2  # type: ignore[misc]


class TestEmit:
    def test_no_event_store_is_a_silent_no_op(self):
        assert emit_no_progress_abort(None, _abort()) is None

    def test_emits_one_event_keyed_on_the_task(self):
        store = _RecordingEventStore()
        abort = _abort()

        emit_no_progress_abort(store, abort)  # type: ignore[arg-type]

        assert store.events == [
            ('merge_verify_progress_abort', {'task_id': 't1', 'data': abort.event_data()}),
        ]

    def test_event_type_value_is_its_name(self):
        assert str(EventType.merge_verify_progress_abort) == 'merge_verify_progress_abort'
