"""Tests for shared.invocation_outcome — InvocationOutcome sum type + classify_invocation."""

from __future__ import annotations

import dataclasses
from datetime import UTC, datetime

import pytest

from shared.invocation_outcome import (
    OK,
    AuthFailed,
    CapHit,
    CliLocalError,
    Failure,
    InvocationOutcome,
    NearCap,
    ZeroOutputWedge,
)


class TestInvocationOutcomeSumType:
    """Contract test: base type + 7 variants, frozen, structural equality."""

    def test_ok_is_invocation_outcome(self):
        assert isinstance(OK(), InvocationOutcome)

    def test_cap_hit_is_invocation_outcome_and_round_trips_none(self):
        outcome = CapHit(resets_at=None, reason='cap')
        assert isinstance(outcome, InvocationOutcome)
        assert outcome.resets_at is None
        assert outcome.reason == 'cap'

    def test_cap_hit_round_trips_datetime(self):
        now = datetime(2026, 1, 1, tzinfo=UTC)
        outcome = CapHit(resets_at=now, reason='cap')
        assert outcome.resets_at == now

    def test_near_cap_is_invocation_outcome(self):
        outcome = NearCap(reason='near')
        assert isinstance(outcome, InvocationOutcome)
        assert outcome.reason == 'near'

    def test_auth_failed_round_trips_status(self):
        outcome = AuthFailed(status=401)
        assert isinstance(outcome, InvocationOutcome)
        assert outcome.status == 401

    def test_cli_local_error_round_trips_marker(self):
        outcome = CliLocalError(marker='is already in use')
        assert isinstance(outcome, InvocationOutcome)
        assert outcome.marker == 'is already in use'

    def test_zero_output_wedge_is_invocation_outcome(self):
        assert isinstance(ZeroOutputWedge(), InvocationOutcome)

    def test_failure_round_trips_kind(self):
        outcome = Failure(kind='unknown')
        assert isinstance(outcome, InvocationOutcome)
        assert outcome.kind == 'unknown'

    @pytest.mark.parametrize(
        'instance',
        [
            OK(),
            CapHit(resets_at=None, reason='r'),
            NearCap(reason='r'),
            AuthFailed(status=401),
            CliLocalError(marker='m'),
            ZeroOutputWedge(),
            Failure(kind='k'),
        ],
        ids=['OK', 'CapHit', 'NearCap', 'AuthFailed', 'CliLocalError', 'ZeroOutputWedge', 'Failure'],
    )
    def test_frozen(self, instance):
        with pytest.raises(dataclasses.FrozenInstanceError):
            instance.some_attr = 'x'  # type: ignore[attr-defined]

    def test_equal_field_variants_compare_equal(self):
        assert CapHit(resets_at=None, reason='r') == CapHit(resets_at=None, reason='r')
        assert AuthFailed(status=401) == AuthFailed(status=401)
        assert OK() == OK()

    def test_different_variants_are_unequal(self):
        assert CapHit(resets_at=None, reason='r') != NearCap(reason='r')
        assert OK() != ZeroOutputWedge()
        assert AuthFailed(status=401) != AuthFailed(status=403)
