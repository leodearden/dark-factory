"""Tests for orchestrator.pause_policy: the pause-class enum and its policy table."""
from __future__ import annotations

import pytest

from orchestrator.pause_policy import (
    STORED_EWA_PREDICATE_CLASSES,
    Bound,
    HaltNotPermittedError,
    LiveRecovery,
    PauseClass,
    PausePolicy,
    Rehydration,
    Retestability,
    policy_for,
    stored_pause_class,
)


class TestPauseClass:
    def test_has_exactly_the_six_classes(self):
        assert {c.value for c in PauseClass} == {
            'ewa_rate',
            'park_stop',
            'cost_ceiling',
            'no_landings',
            'watcher_guard',
            'operator',
        }

    def test_watcher_guard_is_an_enum_member(self):
        assert PauseClass.WATCHER_GUARD.value == 'watcher_guard'


class TestPolicyFor:
    @pytest.mark.parametrize('pause_class', list(PauseClass))
    def test_table_is_total_and_self_keyed(self, pause_class):
        assert policy_for(pause_class).pause_class is pause_class

    def test_accepts_a_plain_string_equal_to_a_member_value(self):
        assert policy_for('ewa_rate').pause_class is PauseClass.EWA_RATE

    def test_rejects_a_non_member_naming_it_and_the_valid_values(self):
        with pytest.raises(ValueError) as excinfo:
            policy_for('bogus')
        message = str(excinfo.value)
        assert 'bogus' in message
        for pause_class in PauseClass:
            assert pause_class.value in message


def _policy(
    pause_class: PauseClass,
    predicate_retestable: Retestability,
    *,
    rehydration: Rehydration = Rehydration.REASSERT,
    live_recovery: LiveRecovery = LiveRecovery.NEVER,
    may_halt: bool = True,
) -> PausePolicy:
    return PausePolicy(
        pause_class=pause_class,
        may_halt=may_halt,
        predicate_retestable=predicate_retestable,
        persisted=True,
        rehydration=rehydration,
        live_recovery=live_recovery,
        bound=Bound.UNBOUNDED,
    )


class TestPolicyTable:
    @pytest.mark.parametrize(
        'expected',
        [
            _policy(
                PauseClass.EWA_RATE,
                Retestability.NO,
                rehydration=Rehydration.RETEST_STORED_EWA,
            ),
            _policy(PauseClass.PARK_STOP, Retestability.NO),
            _policy(PauseClass.COST_CEILING, Retestability.YES),
            _policy(
                PauseClass.NO_LANDINGS,
                Retestability.PARTIAL,
                live_recovery=LiveRecovery.OWN_EVIDENCE,
            ),
            _policy(PauseClass.WATCHER_GUARD, Retestability.NO),
            _policy(PauseClass.OPERATOR, Retestability.NOT_APPLICABLE),
        ],
        ids=lambda policy: policy.pause_class.value,
    )
    def test_row_matches_current_behaviour(self, expected):
        assert policy_for(expected.pause_class) == expected

    def test_halting_classes_without_a_retestable_predicate(self):
        """The load-bearing rule's ratchet (PRD §1): only these may still halt blind."""
        blind_halters = {
            c
            for c in PauseClass
            if policy_for(c).may_halt
            and policy_for(c).predicate_retestable is Retestability.NO
        }
        assert blind_halters == {
            PauseClass.EWA_RATE,
            PauseClass.PARK_STOP,
            PauseClass.WATCHER_GUARD,
        }


class TestStoredEwaIsPredicate:
    def test_follows_the_rehydration_column(self):
        assert _policy(
            PauseClass.PARK_STOP,
            Retestability.NO,
            rehydration=Rehydration.RETEST_STORED_EWA,
        ).stored_ewa_is_predicate
        assert not _policy(
            PauseClass.EWA_RATE, Retestability.NO,
        ).stored_ewa_is_predicate

    def test_class_set_is_the_tables_predicate_rows(self):
        assert {
            c for c in PauseClass if policy_for(c).stored_ewa_is_predicate
        } == STORED_EWA_PREDICATE_CLASSES

    def test_only_ewa_rate_holds_a_live_stored_ewa_today(self):
        assert {PauseClass.EWA_RATE} == STORED_EWA_PREDICATE_CLASSES


class TestRequireMayHalt:
    def test_raises_for_a_class_that_may_not_halt(self):
        policy = _policy(PauseClass.EWA_RATE, Retestability.NO, may_halt=False)
        with pytest.raises(HaltNotPermittedError) as excinfo:
            policy.require_may_halt()
        message = str(excinfo.value)
        assert 'ewa_rate' in message
        assert 'invariant 1' in message

    def test_returns_none_for_a_class_that_may_halt(self):
        policy = _policy(PauseClass.COST_CEILING, Retestability.YES)
        assert policy.require_may_halt() is None


class TestStoredPauseClass:
    def test_null_reads_as_operator(self):
        assert stored_pause_class(None) is PauseClass.OPERATOR

    def test_member_value_reads_as_its_class(self):
        assert stored_pause_class('cost_ceiling') is PauseClass.COST_CEILING

    def test_unrecognised_value_reads_as_operator(self):
        assert stored_pause_class('not-a-class') is PauseClass.OPERATOR
