"""Tests for shared.invocation_outcome — InvocationOutcome sum type + classify_invocation."""

from __future__ import annotations

import dataclasses
import json
from datetime import UTC, datetime, timedelta
from pathlib import Path

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


class TestClassifyInvocationCliLocalError:
    """Local CLI/usage errors must never be misclassified as a cap hit (reify-3604)."""

    def test_session_id_in_use_is_cli_local_error(self):
        result = AgentResult(
            success=False,
            output='',
            stderr='Error: Session ID abc-123 is already in use.',
        )
        outcome = classify_invocation(result, strict_confirm=True)
        assert outcome == CliLocalError(marker='is already in use')

    def test_unrecognized_arguments_is_cli_local_error(self):
        result = AgentResult(success=False, output='', stderr='error: unrecognized arguments: --foo')
        outcome = classify_invocation(result, strict_confirm=True)
        assert outcome == CliLocalError(marker='unrecognized arguments')

    def test_unknown_option_is_cli_local_error(self):
        result = AgentResult(success=False, output='', stderr="unknown option '--bar'")
        outcome = classify_invocation(result, strict_confirm=True)
        assert outcome == CliLocalError(marker='unknown option')

    def test_invalid_value_is_cli_local_error(self):
        result = AgentResult(success=False, output='', stderr="invalid value for '--model'")
        outcome = classify_invocation(result, strict_confirm=True)
        assert outcome == CliLocalError(marker='invalid value')

    def test_no_such_file_or_directory_is_cli_local_error(self):
        result = AgentResult(success=False, output='', stderr='no such file or directory: /tmp/x')
        outcome = classify_invocation(result, strict_confirm=True)
        assert outcome == CliLocalError(marker='no such file or directory')

    def test_permission_denied_is_cli_local_error(self):
        result = AgentResult(success=False, output='', stderr='permission denied: /tmp/x')
        outcome = classify_invocation(result, strict_confirm=True)
        assert outcome == CliLocalError(marker='permission denied')

    def test_marker_in_output_not_just_stderr(self):
        result = AgentResult(success=False, output='unrecognized arguments: --foo', stderr='')
        outcome = classify_invocation(result, strict_confirm=True)
        assert isinstance(outcome, CliLocalError)

    def test_case_insensitive_match(self):
        result = AgentResult(success=False, output='', stderr='PERMISSION DENIED: /tmp/x')
        outcome = classify_invocation(result, strict_confirm=True)
        assert isinstance(outcome, CliLocalError)

    def test_cli_local_error_outranks_cap_message(self):
        """reify-3604: a --session-id collision must never be treated as a cap hit."""
        result = AgentResult(
            success=False,
            stderr='Error: Session ID abc-123 is already in use.',
            output="You've hit your usage limit. Your plan resets in 3h.",
        )
        outcome = classify_invocation(result, strict_confirm=True)
        assert outcome == CliLocalError(marker='is already in use')
        assert not isinstance(outcome, CapHit)

    def test_auth_failed_still_outranks_cli_local_error(self):
        result = AgentResult(
            success=False,
            output='',
            api_error_status=401,
            stderr='Error: Session ID abc-123 is already in use.',
        )
        outcome = classify_invocation(result, strict_confirm=True)
        assert outcome == AuthFailed(status=401)


class TestClassifyInvocationClaudeCap:
    """Claude CapHit/NearCap detection and the strict_confirm (DD-2) toggle."""

    def test_relative_hours_cap_with_confirm_keyword(self):
        result = AgentResult(
            success=False,
            output="You've hit your usage limit for Claude Pro. Your plan resets in 3 hours.",
        )
        outcome = classify_invocation(result, strict_confirm=True)
        assert isinstance(outcome, CapHit)
        assert outcome.resets_at is not None

    def test_cap_prefix_with_no_parseable_reset_time_yields_resets_at_none(self):
        result = AgentResult(
            success=False,
            output="You've used all available credits. Upgrade your plan for more capacity.",
        )
        outcome = classify_invocation(result, strict_confirm=True)
        assert isinstance(outcome, CapHit)
        assert outcome.resets_at is None

    def test_out_of_extra_usage_prefix_is_cap_hit(self):
        result = AgentResult(
            success=False,
            output="You're out of extra usage for this billing period. Your plan resets in 2h.",
        )
        outcome = classify_invocation(result, strict_confirm=True)
        assert isinstance(outcome, CapHit)
        assert outcome.resets_at is not None

    def test_now_using_extra_prefix_is_cap_hit(self):
        result = AgentResult(
            success=False,
            output="You're now using extra compute credits. Your plan resets in 1h.",
        )
        outcome = classify_invocation(result, strict_confirm=True)
        assert isinstance(outcome, CapHit)

    def test_absolute_date_reset_is_cap_hit_with_resets_at(self):
        result = AgentResult(
            success=False,
            output=(
                "You've hit your usage limit for Claude Pro. "
                'Your plan resets on Apr 10, 9pm (UTC).'
            ),
        )
        outcome = classify_invocation(result, strict_confirm=True)
        assert isinstance(outcome, CapHit)
        assert outcome.resets_at is not None

    def test_near_cap_close_to_usage_limit(self):
        result = AgentResult(
            success=False,
            output="You're close to reaching your usage limit. Your plan resets in 1h.",
        )
        outcome = classify_invocation(result, strict_confirm=True)
        assert isinstance(outcome, NearCap)

    def test_near_cap_close_to_plan_limit(self):
        result = AgentResult(
            success=False,
            output="You're close to reaching your plan limit. Your plan resets in 5h.",
        )
        outcome = classify_invocation(result, strict_confirm=True)
        assert isinstance(outcome, NearCap)

    def test_strict_confirm_true_prefix_without_confirm_keyword_is_failure(self):
        """Prefix present ('You're out of extra') but no CAP_CONFIRM_KEYWORDS hit."""
        result = AgentResult(
            success=False,
            output="You're out of extra usage for this billing period.",
        )
        outcome = classify_invocation(result, strict_confirm=True)
        assert isinstance(outcome, Failure)

    def test_strict_confirm_false_accepts_prefix_only(self):
        """DD-2: the probe regime (strict_confirm=False) skips the confirm-keyword guard."""
        result = AgentResult(
            success=False,
            output="You're out of extra usage for this billing period.",
        )
        outcome = classify_invocation(result, strict_confirm=False)
        assert isinstance(outcome, CapHit)
        assert outcome.resets_at is None

    def test_negative_hit_the_nail_no_prefix_match(self):
        result = AgentResult(success=False, output="You've hit the nail on the head")
        outcome = classify_invocation(result, strict_confirm=True)
        assert isinstance(outcome, Failure)

    def test_negative_bare_upgrade_without_plan_or_subscription(self):
        result = AgentResult(
            success=False,
            output="You've used all available credits. Upgrade for more capacity.",
        )
        outcome = classify_invocation(result, strict_confirm=True)
        assert isinstance(outcome, Failure)

    def test_negative_close_to_finish_line_without_confirm_keyword(self):
        result = AgentResult(success=False, output="You're close to the finish line")
        outcome = classify_invocation(result, strict_confirm=True)
        assert isinstance(outcome, Failure)


class TestClassifyInvocationBackends:
    """Codex/Gemini backend cap-pattern switch; claude's default path is unaffected."""

    @pytest.mark.parametrize(
        ('text', 'pattern'),
        [
            ('Error: usage limit reached', 'usage limit reached'),
            ('Error: rate limit', 'rate limit'),
            ('Error: quota exceeded', 'quota exceeded'),
            ('Error: insufficient_quota', 'insufficient_quota'),
            ('Error: rate_limit_exceeded', 'rate_limit_exceeded'),
        ],
    )
    def test_codex_cap_patterns(self, text, pattern):
        result = AgentResult(success=False, output=text)
        outcome = classify_invocation(result, strict_confirm=True, backend='codex')
        assert outcome == CapHit(resets_at=None, reason=f'Codex cap hit: {pattern}')

    @pytest.mark.parametrize(
        ('text', 'pattern'),
        [
            ('Error: quota exceeded', 'quota exceeded'),
            ('Error: rate limit', 'rate limit'),
            ('Error: resource exhausted', 'resource exhausted'),
            ('Error: RESOURCE_EXHAUSTED', 'RESOURCE_EXHAUSTED'),
            ('Error: quota_exceeded', 'quota_exceeded'),
        ],
    )
    def test_gemini_cap_patterns(self, text, pattern):
        result = AgentResult(success=False, output=text)
        outcome = classify_invocation(result, strict_confirm=True, backend='gemini')
        assert outcome == CapHit(resets_at=None, reason=f'Gemini cap hit: {pattern}')

    def test_matching_is_case_insensitive(self):
        result = AgentResult(success=False, output='ERROR: USAGE LIMIT REACHED')
        outcome = classify_invocation(result, strict_confirm=True, backend='codex')
        assert isinstance(outcome, CapHit)

    def test_claude_default_backend_unaffected_by_codex_patterns(self):
        """A codex-only pattern must not trip the cap tier for backend='claude'."""
        result = AgentResult(success=False, output='Error: insufficient_quota')
        outcome = classify_invocation(result, strict_confirm=True)
        assert not isinstance(outcome, CapHit)

    def test_backend_patterns_checked_against_stderr_too(self):
        result = AgentResult(success=False, output='', stderr='Error: quota exceeded')
        outcome = classify_invocation(result, strict_confirm=True, backend='codex')
        assert isinstance(outcome, CapHit)


class TestClassifyInvocationWedgeOkFailure:
    """ZeroOutputWedge + OK + Failure tail, and the full precedence chain."""

    def test_timed_out_zero_transcript_turns_is_wedge(self):
        """Transcript-authoritative: transcript_turns == 0 -> ZeroOutputWedge."""
        result = AgentResult(success=False, output='', timed_out=True, transcript_turns=0)
        outcome = classify_invocation(result, strict_confirm=True)
        assert outcome == ZeroOutputWedge()

    def test_timed_out_legacy_fallback_zero_turns_and_cost_is_wedge(self):
        """transcript_turns=None -> legacy turns==0 and cost_usd==0.0 fallback."""
        result = AgentResult(
            success=False,
            output='',
            timed_out=True,
            turns=0,
            cost_usd=0.0,
            transcript_turns=None,
        )
        outcome = classify_invocation(result, strict_confirm=True)
        assert outcome == ZeroOutputWedge()

    def test_timed_out_with_transcript_progress_is_not_a_wedge(self):
        """reify-4415: transcript_turns > 0 means real work happened -> not a wedge."""
        result = AgentResult(success=False, output='', timed_out=True, transcript_turns=5)
        outcome = classify_invocation(result, strict_confirm=True)
        assert not isinstance(outcome, ZeroOutputWedge)
        assert isinstance(outcome, Failure)

    def test_cap_hit_outranks_zero_output_wedge(self):
        """A cap message on a zero-turn timeout must classify as CapHit, not a wedge."""
        result = AgentResult(
            success=False,
            output="You're out of extra usage for this billing period. Your plan resets in 2h.",
            timed_out=True,
            transcript_turns=0,
        )
        outcome = classify_invocation(result, strict_confirm=False)
        assert isinstance(outcome, CapHit)
        assert not isinstance(outcome, ZeroOutputWedge)

    def test_success_true_is_ok(self):
        result = AgentResult(success=True, output='done')
        outcome = classify_invocation(result, strict_confirm=True)
        assert outcome == OK()

    def test_no_signal_at_all_falls_to_failure_tail(self):
        result = AgentResult(success=False, output='generic failure, no signal')
        outcome = classify_invocation(result, strict_confirm=True)
        assert isinstance(outcome, Failure)


# --- B3: golden-corpus acceptance test -------------------------------------
#
# corpus.json is the checked-in source of truth (see its README for the
# record schema and provenance). New CLI wordings are pinned by appending a
# row there rather than editing this test.

_CORPUS_PATH = Path(__file__).parent / 'fixtures' / 'cap_strings' / 'corpus.json'

_VARIANT_CLASSES: dict[str, type[InvocationOutcome]] = {
    'OK': OK,
    'CapHit': CapHit,
    'NearCap': NearCap,
    'AuthFailed': AuthFailed,
    'CliLocalError': CliLocalError,
    'ZeroOutputWedge': ZeroOutputWedge,
    'Failure': Failure,
}


def _load_corpus() -> list[dict]:
    with _CORPUS_PATH.open() as f:
        return json.load(f)


def _agent_result_from_record(record: dict) -> AgentResult:
    """Build an AgentResult from a corpus record.

    ``success`` and ``output`` are required (no-default) positional fields on
    AgentResult itself, so this helper supplies the README-documented
    defaults (``success=False``, ``output=''``) explicitly; every other field
    defaults exactly as AgentResult itself defaults.
    """
    return AgentResult(
        success=record.get('success', False),
        output=record.get('output', ''),
        stderr=record.get('stderr', ''),
        turns=record.get('turns', 0),
        cost_usd=record.get('cost_usd', 0.0),
        timed_out=record.get('timed_out', False),
        transcript_turns=record.get('transcript_turns'),
        api_error_status=record.get('api_error_status'),
    )


_CORPUS = _load_corpus()


class TestGoldenCorpus:
    """B3 — the user-observable signal.

    Every historical cap/error string in corpus.json maps to its recorded
    expected InvocationOutcome variant, per backend, with strict_confirm
    toggling the prefix-only vs confirm-guarded regime.
    """

    @pytest.mark.parametrize('record', _CORPUS, ids=[r['id'] for r in _CORPUS])
    def test_corpus_record_classifies_as_expected(self, record):
        result = _agent_result_from_record(record)
        outcome = classify_invocation(
            result,
            strict_confirm=record['strict_confirm'],
            backend=record.get('backend', 'claude'),
        )

        expected_cls = _VARIANT_CLASSES[record['expected']]
        assert isinstance(outcome, expected_cls), (
            f'{record["id"]}: expected {record["expected"]}, '
            f'got {type(outcome).__name__} ({outcome!r})'
        )

        resets_at_expectation = record.get('resets_at')
        if record['expected'] == 'CapHit' and resets_at_expectation == 'set':
            assert outcome.resets_at is not None, (
                f'{record["id"]}: expected resets_at to be set, got None'
            )
        elif record['expected'] == 'CapHit' and resets_at_expectation == 'none':
            assert outcome.resets_at is None, (
                f'{record["id"]}: expected resets_at to be None, got {outcome.resets_at!r}'
            )
