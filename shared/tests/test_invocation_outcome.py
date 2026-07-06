"""Tests for shared.invocation_outcome — InvocationOutcome sum type + classify_invocation."""

from __future__ import annotations

import dataclasses
from datetime import UTC, datetime, timedelta

import pytest

from shared.cli_invoke import AgentResult
from shared.invocation_outcome import (
    OK,
    AuthFailed,
    CapHit,
    CliLocalError,
    Failure,
    InvocationOutcome,
    NearCap,
    ZeroOutputWedge,
    _extract_cap_message,
    _parse_resets_at,
    classify_invocation,
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


class TestParseResetsAt:
    """_parse_resets_at: 7.1.a — None on parse failure, never a fabricated now+1h."""

    FIXED_NOW = datetime(2026, 1, 1, tzinfo=UTC)

    def test_unparseable_text_returns_none(self):
        text = "You've used all available credits. Upgrade your plan for more capacity."
        assert _parse_resets_at(text, now=self.FIXED_NOW) is None

    def test_no_reset_info_at_all_returns_none(self):
        assert _parse_resets_at('no reset info here', now=self.FIXED_NOW) is None

    def test_relative_hours(self):
        result = _parse_resets_at('resets in 3h', now=self.FIXED_NOW)
        assert result == self.FIXED_NOW + timedelta(hours=3)

    def test_relative_minutes(self):
        result = _parse_resets_at('resets in 45m', now=self.FIXED_NOW)
        assert result == self.FIXED_NOW + timedelta(minutes=45)

    def test_relative_days(self):
        result = _parse_resets_at('resets in 2d', now=self.FIXED_NOW)
        assert result == self.FIXED_NOW + timedelta(days=2)

    def test_relative_hours_word_form(self):
        """'resets in 3 hours' — the unit char is read from the first letter of 'hours'."""
        result = _parse_resets_at('resets in 3 hours', now=self.FIXED_NOW)
        assert result == self.FIXED_NOW + timedelta(hours=3)

    def test_absolute_date_returns_tz_aware_datetime(self):
        text = (
            "You've hit your usage limit for Claude Pro. "
            'Your plan resets on Apr 10, 9pm (UTC).'
        )
        result = _parse_resets_at(text, now=self.FIXED_NOW)
        assert result is not None
        assert result.tzinfo is not None
        assert (result.year, result.month, result.day, result.hour) == (2026, 4, 10, 21)

    def test_default_now_reads_wall_clock_when_not_injected(self):
        """classify_invocation calls this without injecting now — must not raise."""
        result = _parse_resets_at('resets in 1h')
        assert result is not None


class TestExtractCapMessage:
    """_extract_cap_message: returns the sentence containing the matched prefix."""

    def test_extracts_sentence_containing_prefix(self):
        text = "You've hit your usage limit. Your plan resets in 3h."
        message = _extract_cap_message(text, "You've hit your")
        assert "You've hit your usage limit" in message

    def test_returns_empty_string_when_prefix_absent(self):
        assert _extract_cap_message('nothing relevant here', "You've hit your") == ''

    def test_case_insensitive_match(self):
        text = "YOU'VE HIT YOUR usage limit. resets in 3h."
        message = _extract_cap_message(text, "you've hit your")
        assert message != ''


class TestClassifyInvocationAuthFailedPrecedence:
    """AuthFailed is highest precedence; narrow to {401, 403} — 429 falls through."""

    def test_401_returns_auth_failed(self):
        result = AgentResult(success=False, output='', api_error_status=401)
        outcome = classify_invocation(result, strict_confirm=True)
        assert outcome == AuthFailed(status=401)

    def test_403_returns_auth_failed(self):
        result = AgentResult(success=False, output='', api_error_status=403)
        outcome = classify_invocation(result, strict_confirm=True)
        assert outcome == AuthFailed(status=403)

    def test_401_outranks_cap_prefix_in_same_text(self):
        """AuthFailed must win even when the same result also carries a cap message."""
        result = AgentResult(
            success=False,
            output="You've hit your usage limit. Your plan resets in 3h.",
            api_error_status=401,
        )
        outcome = classify_invocation(result, strict_confirm=True)
        assert outcome == AuthFailed(status=401)

    def test_429_with_cap_text_is_cap_hit_not_auth_failed(self):
        """429 is deliberately excluded from AuthFailed — it carries a real cap body."""
        result = AgentResult(
            success=False,
            output="You're out of extra usage for this billing period. Your plan resets in 2h.",
            api_error_status=429,
        )
        outcome = classify_invocation(result, strict_confirm=True)
        assert isinstance(outcome, CapHit)
        assert not isinstance(outcome, AuthFailed)

    def test_429_alone_with_no_cap_text_is_not_auth_failed(self):
        result = AgentResult(success=False, output='some unrelated text', api_error_status=429)
        outcome = classify_invocation(result, strict_confirm=True)
        assert not isinstance(outcome, AuthFailed)
        assert isinstance(outcome, Failure)
