"""The C1 re-arm contract as pure functions (plans/completion-driven-triggers-prd.md C1).

Hand-built metadata mappings and fixed aware datetimes; no I/O.
"""

from __future__ import annotations

import logging
from datetime import UTC, datetime, timedelta

import pytest

from orchestrator.predicate_recheck import (
    PREDICATE_RECHECK_KEY,
    PREDICATE_VERDICT_KEY,
    PredicateRecheck,
    RecheckDeclaration,
    rearmed,
    recheck_declaration,
    render_verdict,
    reset,
    stall_detail,
    verdict_stamps,
)

NOW = datetime(2026, 10, 25, 12, 0, tzinfo=UTC)
EARLIER = NOW - timedelta(days=3)
DATED = {'mode': 'dated', 'at': '2026-10-25T12:00:00+00:00'}
DECLARED = {'kind': 'predicate', 'script': 's', 'timeout_secs': 60, 'recheck_secs': 86400}

STORED = {
    'consecutive': 3,
    'first_rearmed_at': EARLIER.isoformat(),
    'last_checked_at': NOW.isoformat(),
    'next_check_at': (NOW + timedelta(seconds=600)).isoformat(),
}


def _record(consecutive: int = 3, first_rearmed_at: datetime | None = EARLIER) -> PredicateRecheck:
    return PredicateRecheck(
        consecutive=consecutive,
        first_rearmed_at=first_rearmed_at,
        last_checked_at=EARLIER,
        next_check_at=None,
    )


class TestStoredRecord:
    def test_absent_record_reads_as_none(self):
        assert PredicateRecheck.from_metadata({}) is None

    def test_well_formed_record_round_trips(self):
        record = PredicateRecheck.from_metadata({PREDICATE_RECHECK_KEY: STORED})

        assert record is not None
        assert record.to_metadata() == STORED
        assert set(record.to_metadata()) == {
            'consecutive', 'first_rearmed_at', 'last_checked_at', 'next_check_at',
        }

    @pytest.mark.parametrize(
        'raw',
        [
            {**STORED, 'consecutive': '3'},
            {**STORED, 'consecutive': True},
            {**STORED, 'consecutive': -1},
            'not a record',
        ],
        ids=['str-count', 'bool-count', 'negative-count', 'non-dict'],
    )
    def test_malformed_record_reads_as_none_with_a_warning(self, raw, caplog):
        with caplog.at_level(logging.WARNING, logger='orchestrator.predicate_recheck'):
            assert PredicateRecheck.from_metadata({PREDICATE_RECHECK_KEY: raw}) is None

        assert len(caplog.records) == 1
        assert repr(raw) in caplog.records[0].getMessage()


class TestTransitions:
    def test_first_rearm_starts_the_streak(self):
        record = rearmed(None, NOW, 86400)

        assert record.consecutive == 1
        assert record.first_rearmed_at == NOW
        assert record.last_checked_at == NOW
        assert record.next_check_at == NOW + timedelta(seconds=86400)
        assert record.to_metadata() == {
            'consecutive': 1,
            'first_rearmed_at': NOW.isoformat(),
            'last_checked_at': NOW.isoformat(),
            'next_check_at': (NOW + timedelta(seconds=86400)).isoformat(),
        }

    def test_later_rearm_extends_the_streak_and_keeps_the_first_time(self):
        record = rearmed(_record(consecutive=3), NOW, 600)

        assert record.consecutive == 4
        assert record.first_rearmed_at == EARLIER
        assert record.next_check_at == NOW + timedelta(seconds=600)

    def test_rearm_after_a_never_rearmed_reset_stamps_the_first_time(self):
        record = rearmed(reset(None, EARLIER), NOW, 600)

        assert record.consecutive == 1
        assert record.first_rearmed_at == NOW

    def test_reset_ends_the_streak_and_keeps_the_first_time(self):
        record = reset(_record(consecutive=3), NOW)

        assert record.consecutive == 0
        assert record.first_rearmed_at == EARLIER
        assert record.last_checked_at == NOW
        assert record.next_check_at is None

    def test_reset_without_a_record(self):
        record = reset(None, NOW)

        assert record.consecutive == 0
        assert record.first_rearmed_at is None
        assert record.last_checked_at == NOW


class TestRecheckDeclaration:
    def test_declared_on_a_dated_milestone(self):
        declaration = recheck_declaration(DECLARED, DATED)

        assert declaration == RecheckDeclaration(recheck_secs=86400, gap=None)

    def test_undeclared_names_the_missing_field(self):
        before_done = {k: v for k, v in DECLARED.items() if k != 'recheck_secs'}
        declaration = recheck_declaration(before_done, DATED)

        assert declaration.recheck_secs is None
        assert declaration.gap is not None
        assert 'before_done.recheck_secs' in declaration.gap

    def test_declared_on_a_delayed_milestone_names_the_dated_rule(self):
        declaration = recheck_declaration(DECLARED, {'mode': 'delayed', 'after_secs': 60})

        assert declaration.recheck_secs is None
        assert declaration.gap is not None
        assert "mode='dated'" in declaration.gap

    def test_invalid_before_done_names_the_validation_failure(self):
        declaration = recheck_declaration({**DECLARED, 'recheck_secs': 10}, DATED)

        assert declaration.recheck_secs is None
        assert declaration.gap is not None
        assert 'before_done.recheck_secs: ' in declaration.gap
        assert 'greater than or equal to 300' in declaration.gap
        assert '\n' not in declaration.gap

    @pytest.mark.parametrize(
        ('recheck_secs', 'gap'), [(86400, 'a gap'), (None, None)], ids=['both', 'neither'],
    )
    def test_exactly_one_field_is_set(self, recheck_secs, gap):
        with pytest.raises(ValueError, match='exactly one'):
            RecheckDeclaration(recheck_secs=recheck_secs, gap=gap)


class TestVerdictStamps:
    def test_nothing_to_write_without_a_verdict_a_prior_stamp_or_a_record(self):
        assert verdict_stamps({}, None, None) == {}

    def test_a_verdict_is_stamped(self):
        verdict = {'verdict': 'pass', 'share': 0.8}

        assert verdict_stamps({}, verdict, None) == {PREDICATE_VERDICT_KEY: verdict}

    def test_a_prior_stamp_is_cleared_when_the_check_gave_no_verdict(self):
        metadata = {PREDICATE_VERDICT_KEY: {'verdict': 'stale'}}

        assert verdict_stamps(metadata, None, None) == {PREDICATE_VERDICT_KEY: None}

    def test_a_record_rides_along(self):
        record = reset(_record(), NOW)

        assert verdict_stamps({}, None, record) == {PREDICATE_RECHECK_KEY: record.to_metadata()}

    def test_a_not_yet_exit_always_stamps_its_verdict(self):
        assert verdict_stamps({}, None, None, always_stamp_verdict=True) == {
            PREDICATE_VERDICT_KEY: None,
        }


class TestStallDetail:
    def _detail(self, first_rearmed_at: datetime | None, verdict: dict | None) -> list[str]:
        return stall_detail(
            'the gate',
            streak=30,
            stall_cap=30,
            first_rearmed_at=first_rearmed_at,
            now=NOW,
            recheck_secs=86400,
            verdict=verdict,
        ).splitlines()

    def test_names_the_streak_the_elapsed_time_and_the_last_verdict(self):
        lines = self._detail(EARLIER, {'verdict': 'not_yet'})

        assert lines[0] == 'the gate'
        assert 'streak 30/30' in lines[1]
        assert lines[2] == f'first_rearmed_at: {EARLIER.isoformat()} (3 days, 0:00:00 ago)'
        assert lines[3] == 'recheck_secs: 86400'
        assert lines[4] == 'Last verdict: {"verdict":"not_yet"}'

    def test_a_gate_that_never_re_armed_says_so(self):
        lines = self._detail(None, None)

        assert lines[2].startswith('first_rearmed_at: none')
        assert lines[4] == f'Last verdict: {render_verdict(None)}'
        assert render_verdict(None) == 'no structured verdict'
