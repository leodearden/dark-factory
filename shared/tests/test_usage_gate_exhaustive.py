"""Exhaustive unit tests for UsageGate.

Covers: cap detection patterns, reset time parsing, cap message extraction,
handle_cap_detected, refresh_capped_accounts, before_invoke, confirm_account_ok,
on_agent_complete, shutdown, and all properties.
"""

from __future__ import annotations

import asyncio
import json
import os
from datetime import UTC, datetime, timedelta
from unittest.mock import AsyncMock, patch

import pytest

from shared.config_models import AccountConfig, UsageCapConfig
from shared.usage_gate import (
    CAP_HIT_PREFIXES,
    CODEX_CAP_PATTERNS,
    GEMINI_CAP_PATTERNS,
    NEAR_CAP_PREFIXES,
    SessionBudgetExhausted,
    UsageGate,
    _extract_cap_message,
    _parse_resets_at,
    _read_oauth_token,
)

# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------


def make_gate(
    account_names: list[str],
    *,
    cost_store=None,
    wait_for_reset: bool = False,
    session_budget_usd: float | None = None,
    probe_interval_secs: int = 300,
    max_probe_interval_secs: int = 1800,
) -> UsageGate:
    acct_cfgs = []
    env_vars: dict[str, str] = {}
    for name in account_names:
        env_key = f'TEST_TOKEN_{name.upper().replace("-", "_")}'
        env_vars[env_key] = f'fake-token-{name}'
        acct_cfgs.append(AccountConfig(name=name, oauth_token_env=env_key))
    config = UsageCapConfig(
        accounts=acct_cfgs,
        wait_for_reset=wait_for_reset,
        session_budget_usd=session_budget_usd,
        probe_interval_secs=probe_interval_secs,
        max_probe_interval_secs=max_probe_interval_secs,
    )
    with patch.dict(os.environ, env_vars):
        gate = UsageGate(config, cost_store=cost_store)
    # Mock _run_probe to prevent real subprocess spawning
    gate._run_probe = AsyncMock(return_value=True)
    return gate


def make_mock_cost_store() -> AsyncMock:
    store = AsyncMock()
    store.save_account_event = AsyncMock(return_value=None)
    return store


# =========================================================================
# TestCapDetectionPatterns
# =========================================================================


class TestCapDetectionPatterns:
    """detect_cap_hit pattern matching — all prefixes, backends, and edge cases."""

    # --- CAP_HIT_PREFIXES (3 tests) ---

    @pytest.mark.parametrize('prefix', CAP_HIT_PREFIXES, ids=lambda p: p[:30])
    def test_cap_hit_prefix_detected(self, prefix):
        gate = make_gate(['a'])
        text = f'{prefix} usage limit. Your plan resets in 5h.'
        assert gate.detect_cap_hit('', text) is True

    # --- NEAR_CAP_PREFIXES (2 tests) ---

    @pytest.mark.parametrize('prefix', NEAR_CAP_PREFIXES, ids=lambda p: p[:30])
    def test_near_cap_prefix_detected(self, prefix):
        gate = make_gate(['a'])
        text = f'{prefix} your usage limit. Your plan resets in 2h.'
        assert gate.detect_cap_hit('', text) is True

    # --- CODEX_CAP_PATTERNS with backend='codex' (5 tests) ---

    @pytest.mark.parametrize('pattern', CODEX_CAP_PATTERNS)
    def test_codex_cap_pattern_detected(self, pattern):
        gate = make_gate(['a'])
        text = f'Error: {pattern}'
        assert gate.detect_cap_hit('', text, backend='codex') is True

    # --- GEMINI_CAP_PATTERNS with backend='gemini' (5 tests) ---

    @pytest.mark.parametrize('pattern', GEMINI_CAP_PATTERNS)
    def test_gemini_cap_pattern_detected(self, pattern):
        gate = make_gate(['a'])
        text = f'Error: {pattern}'
        assert gate.detect_cap_hit('', text, backend='gemini') is True

    # --- Case insensitivity for Claude patterns ---

    def test_cap_hit_prefix_uppercase(self):
        gate = make_gate(['a'])
        text = "YOU'VE HIT YOUR usage limit resets in 3h"
        assert gate.detect_cap_hit('', text) is True

    def test_cap_hit_prefix_lowercase(self):
        gate = make_gate(['a'])
        text = "you've hit your usage limit resets in 3h"
        assert gate.detect_cap_hit('', text) is True

    def test_cap_hit_prefix_mixed_case(self):
        gate = make_gate(['a'])
        text = "You'Ve HiT yOuR usage limit resets in 3h"
        assert gate.detect_cap_hit('', text) is True

    # --- Pattern location: stderr only, result_text only, both ---

    def test_cap_in_stderr_only(self):
        gate = make_gate(['a'])
        stderr = "You've hit your usage limit. resets in 3h"
        assert gate.detect_cap_hit(stderr, '') is True

    def test_cap_in_result_text_only(self):
        gate = make_gate(['a'])
        result = "You've hit your usage limit. resets in 3h"
        assert gate.detect_cap_hit('', result) is True

    def test_cap_in_both(self):
        gate = make_gate(['a'])
        text = "You've hit your usage limit. resets in 3h"
        assert gate.detect_cap_hit(text, text) is True

    # --- No false positives ---

    def test_no_false_positive_normal_output(self):
        gate = make_gate(['a'])
        assert gate.detect_cap_hit('', 'Task completed successfully') is False

    def test_no_false_positive_compliment(self):
        gate = make_gate(['a'])
        assert gate.detect_cap_hit('', 'Your code is great') is False

    def test_no_false_positive_partial_match(self):
        """'You've hit the nail on the head' contains 'You've hit' but not 'your'."""
        gate = make_gate(['a'])
        # The prefix is "You've hit your" — 'You've hit the nail' should NOT match
        assert gate.detect_cap_hit('', "You've hit the nail on the head") is False

    # --- Backend routing: codex patterns don't trigger for claude ---

    def test_codex_pattern_not_detected_for_claude_backend(self):
        gate = make_gate(['a'])
        # 'insufficient_quota' is codex-only
        assert gate.detect_cap_hit('', 'insufficient_quota', backend='claude') is False

    def test_claude_patterns_checked_for_all_backends(self):
        """Claude CAP_HIT_PREFIXES are always checked regardless of backend."""
        gate = make_gate(['a'])
        text = "You've hit your usage limit. resets in 3h"
        assert gate.detect_cap_hit('', text, backend='codex') is True
        gate2 = make_gate(['b'])
        assert gate2.detect_cap_hit('', text, backend='gemini') is True

    # --- Empty strings ---

    def test_empty_strings(self):
        gate = make_gate(['a'])
        assert gate.detect_cap_hit('', '') is False

    # --- Pattern marks correct account as capped ---

    def test_marks_correct_account_by_token(self):
        gate = make_gate(['a', 'b'])
        token_b = gate._accounts[1].token
        gate.detect_cap_hit('', "You've hit your usage limit resets in 3h", oauth_token=token_b)
        assert gate._accounts[0].capped is False
        assert gate._accounts[1].capped is True

    # --- Unknown token caps first uncapped ---

    def test_unknown_token_caps_first_uncapped(self):
        gate = make_gate(['a', 'b'])
        gate.detect_cap_hit(
            '', "You've hit your usage limit resets in 3h", oauth_token='unknown-token'
        )
        assert gate._accounts[0].capped is True
        assert gate._accounts[1].capped is False

    # --- No oauth_token and all uncapped ---

    def test_no_oauth_token_caps_first_uncapped(self):
        gate = make_gate(['a', 'b'])
        gate.detect_cap_hit('', "You've hit your usage limit resets in 3h", oauth_token=None)
        assert gate._accounts[0].capped is True

    # --- Gemini RESOURCE_EXHAUSTED exact case ---

    def test_gemini_resource_exhausted_exact_case(self):
        gate = make_gate(['a'])
        assert gate.detect_cap_hit('', 'RESOURCE_EXHAUSTED', backend='gemini') is True

    # --- Realistic cap-hit smoke tests (verbatim Claude UI strings) ---

    def test_realistic_cap_hit_out_of_extra_usage(self):
        """Verbatim: 'You're out of extra usage for this billing period.'"""
        gate = make_gate(['a'])
        msg = "You're out of extra usage for this billing period. Your plan resets in 3 hours."
        assert gate.detect_cap_hit('', msg) is True

    def test_realistic_cap_hit_pro_plan_date_format(self):
        """Verbatim: 'You've hit your usage limit for Claude Pro. Your plan resets on Apr 10, 9pm (UTC).'"""
        gate = make_gate(['a'])
        msg = "You've hit your usage limit for Claude Pro. Your plan resets on Apr 10, 9pm (UTC)."
        assert gate.detect_cap_hit('', msg) is True

    # --- Realistic near-cap smoke tests (verbatim Claude UI strings) ---

    def test_realistic_near_cap_close_to_limit(self):
        """Verbatim: 'You're close to reaching your usage limit. Your plan resets in 1h.'"""
        gate = make_gate(['a'])
        msg = "You're close to reaching your usage limit. Your plan resets in 1h."
        assert gate.detect_cap_hit('', msg) is True

    # --- Parametrized realistic cap messages (one per prefix) ---

    @pytest.mark.parametrize('message,expected', [
        # CAP_HIT_PREFIXES
        (
            "You've hit your usage limit for Claude Pro. Your plan resets in 3 hours.",
            True,
        ),
        (
            "You've used all available credits. Upgrade for more capacity.",
            True,
        ),
        (
            "You're out of extra usage for this billing period. Your plan resets in 2h.",
            True,
        ),
        (
            "You're now using extra compute credits. Your plan resets in 1h.",
            True,
        ),
        # NEAR_CAP_PREFIXES
        (
            "You're close to reaching your plan limit. Your plan resets in 5h.",
            True,
        ),
    ], ids=[
        'cap_hit_prefix_hit_your',
        'cap_hit_prefix_used',
        'cap_hit_prefix_out_of_extra',
        'cap_hit_prefix_now_using_extra',
        'near_cap_prefix_close_to',
    ])
    def test_realistic_cap_messages(self, message, expected):
        gate = make_gate(['a'])
        assert gate.detect_cap_hit('', message) is expected


# =========================================================================
# TestNearCapStateDistinction
# =========================================================================


class TestNearCapStateDistinction:
    """Behavioral tests that distinguish NEAR_CAP from CAP_HIT state transitions.

    Step 4 (spec-first): test_near_cap_does_not_set_capped_true FAILS until
    step-5 implementation adds _handle_near_cap_warning and near_cap field.
    """

    def test_near_cap_does_not_set_capped_true(self):
        """NEAR_CAP message must NOT set acct.capped=True; must set acct.near_cap=True."""
        gate = make_gate(['a'])
        msg = "You're close to reaching your usage limit. Your plan resets in 4h."
        result = gate.detect_cap_hit('', msg)
        acct = gate._accounts[0]
        assert result is True               # detection must still return True
        assert acct.capped is False         # account is NOT blocked
        assert acct.near_cap is True        # but the near-cap warning flag is set

    def test_cap_hit_still_sets_capped_true(self):
        """CAP_HIT message must still set acct.capped=True (existing behavior preserved)."""
        gate = make_gate(['a'])
        msg = "You've hit your usage limit. Your plan resets in 3h."
        result = gate.detect_cap_hit('', msg)
        acct = gate._accounts[0]
        assert result is True
        assert acct.capped is True

    def test_near_cap_then_cap_hit_sets_capped(self):
        """Near-cap followed by cap-hit on the same account sets capped=True, near_cap=False."""
        gate = make_gate(['a'])
        acct = gate._accounts[0]

        gate.detect_cap_hit('', "You're close to reaching your usage limit. Your plan resets in 4h.")
        assert acct.near_cap is True
        assert acct.capped is False

        gate.detect_cap_hit('', "You've hit your usage limit. Your plan resets in 3h.")
        assert acct.capped is True
        assert acct.near_cap is False

    def test_near_cap_does_not_close_gate(self):
        """A single-account gate must remain open after a NEAR_CAP message."""
        gate = make_gate(['a'])
        gate.detect_cap_hit('', "You're close to reaching your usage limit. Your plan resets in 1h.")
        assert gate._open.is_set() is True

    def test_near_cap_does_not_start_resume_probe(self):
        """NEAR_CAP must NOT launch a resume probe task."""
        gate = make_gate(['a'], wait_for_reset=True)
        gate.detect_cap_hit('', "You're close to reaching your usage limit. Your plan resets in 4h.")
        acct = gate._accounts[0]
        assert acct.resume_task is None

    def test_near_cap_fires_cost_event_with_cost_store(self):
        """_handle_near_cap_warning must fire a cost event when cost_store is set."""
        cost_store = make_mock_cost_store()
        gate = make_gate(['a'], cost_store=cost_store)
        msg = "You're close to reaching your usage limit. Your plan resets in 4h."
        with patch.object(gate, '_fire_cost_event') as mock_fire:
            gate.detect_cap_hit('', msg)
        mock_fire.assert_called_once()

    def test_near_cap_no_cost_event_without_cost_store(self):
        """_handle_near_cap_warning must NOT fire a cost event when cost_store is None."""
        gate = make_gate(['a'], cost_store=None)
        msg = "You're close to reaching your usage limit. Your plan resets in 4h."
        with patch.object(gate, '_fire_cost_event') as mock_fire:
            gate.detect_cap_hit('', msg)
        mock_fire.assert_not_called()


# =========================================================================
# TestCapHitNowUsingExtraSemantics
# =========================================================================


class TestCapHitNowUsingExtraSemantics:
    """Behavioral tests asserting 'You're now using extra' triggers CAP_HIT, not NEAR_CAP.

    'You're now using extra compute credits' means the account has crossed the
    base plan's hard cap and is billing overage — semantically equivalent to a
    cap hit, not merely a warning. These tests FAIL until step-3 moves the
    prefix from NEAR_CAP_PREFIXES to CAP_HIT_PREFIXES.
    """

    # All messages include 'resets' as a secondary keyword so they remain valid
    # through the step-6 CAP_CONFIRM_KEYWORDS enforcement.

    _MSG = "You're now using extra compute credits. Your plan resets in 4h."

    def test_now_using_extra_sets_capped_true(self):
        """'You're now using extra' must set acct.capped=True (CAP_HIT routing)."""
        gate = make_gate(['a'])
        gate.detect_cap_hit('', self._MSG)
        acct = gate._accounts[0]
        assert acct.capped is True

    def test_now_using_extra_does_not_set_near_cap(self):
        """'You're now using extra' must leave acct.near_cap=False (not NEAR_CAP routing)."""
        gate = make_gate(['a'])
        gate.detect_cap_hit('', self._MSG)
        acct = gate._accounts[0]
        assert acct.near_cap is False

    def test_now_using_extra_routes_to_handle_cap_detected(self):
        """Only _handle_cap_detected must be called, not _handle_near_cap_warning."""
        gate = make_gate(['a'])
        with patch.object(gate, '_handle_cap_detected') as mock_cap, \
                patch.object(gate, '_handle_near_cap_warning') as mock_near:
            gate.detect_cap_hit('', self._MSG)
        mock_cap.assert_called_once()
        mock_near.assert_not_called()

    def test_now_using_extra_closes_gate_when_single_account(self):
        """A single-account gate must close (_open cleared) after a cap hit."""
        gate = make_gate(['a'])
        gate.detect_cap_hit('', self._MSG)
        assert gate._open.is_set() is False

    def test_now_using_extra_parses_resets_at_from_message(self):
        """_handle_cap_detected must parse 'resets in 4h' and set acct.resets_at ~4h ahead."""
        gate = make_gate(['a'])
        before = datetime.now(UTC)
        gate.detect_cap_hit('', self._MSG)
        acct = gate._accounts[0]
        assert acct.resets_at is not None
        expected = before + timedelta(hours=4)
        assert abs((acct.resets_at - expected).total_seconds()) < 5


# =========================================================================
# TestResetTimeParsing
# =========================================================================


class TestResetTimeParsing:
    """_parse_resets_at: relative, absolute, fallback, edge cases."""

    def test_relative_hours(self):
        before = datetime.now(UTC)
        result = _parse_resets_at('resets in 3h')
        after = datetime.now(UTC)
        assert before + timedelta(hours=3) <= result <= after + timedelta(hours=3)

    def test_relative_minutes(self):
        before = datetime.now(UTC)
        result = _parse_resets_at('resets in 45m')
        after = datetime.now(UTC)
        assert before + timedelta(minutes=45) <= result <= after + timedelta(minutes=45)

    def test_relative_days(self):
        before = datetime.now(UTC)
        result = _parse_resets_at('resets in 2d')
        after = datetime.now(UTC)
        assert before + timedelta(days=2) <= result <= after + timedelta(days=2)

    def test_relative_zero_hours(self):
        before = datetime.now(UTC)
        result = _parse_resets_at('resets in 0h')
        after = datetime.now(UTC)
        assert before <= result <= after + timedelta(seconds=1)

    def test_absolute_with_date(self):
        # "resets Mar 30, 6pm (UTC)" — should parse to a valid datetime
        result = _parse_resets_at('resets Mar 30, 6pm (UTC)')
        assert result.minute == 0
        assert result.hour == 18
        assert result.month == 3
        assert result.day == 30

    def test_absolute_time_only(self):
        result = _parse_resets_at('resets 9pm (UTC)')
        assert result.hour == 21
        assert result.minute == 0

    def test_absolute_time_with_minutes(self):
        result = _parse_resets_at('resets 3:00 AM (UTC)')
        assert result.hour == 3
        assert result.minute == 0

    def test_absolute_non_utc_timezone(self):
        result = _parse_resets_at('resets 6pm (Europe/London)')
        # Should return a valid UTC datetime — just confirm it parses
        assert result.tzinfo is not None or result.tzinfo == UTC

    def test_fallback_no_reset_info(self):
        before = datetime.now(UTC)
        result = _parse_resets_at('no reset info whatsoever')
        after = datetime.now(UTC)
        # Fallback: 1 hour from now
        assert before + timedelta(hours=1) - timedelta(seconds=1) <= result
        assert result <= after + timedelta(hours=1) + timedelta(seconds=1)

    def test_embedded_in_longer_text(self):
        text = "You've hit your limit. Your usage resets in 5h. Please wait."
        before = datetime.now(UTC)
        result = _parse_resets_at(text)
        after = datetime.now(UTC)
        assert before + timedelta(hours=5) <= result <= after + timedelta(hours=5)

    def test_case_insensitivity(self):
        before = datetime.now(UTC)
        result = _parse_resets_at('RESETS IN 3H')
        after = datetime.now(UTC)
        assert before + timedelta(hours=3) <= result <= after + timedelta(hours=3)

    def test_year_rollover(self):
        """resets Jan 1, 12am (UTC) when current month is not January
        or when it is in the past should still parse."""
        result = _parse_resets_at('resets Jan 1, 12am (UTC)')
        assert result.month == 1
        assert result.day == 1
        # If Jan 1 is in the past, parser bumps to next year
        now = datetime.now(UTC)
        if now.month > 1 or (now.month == 1 and now.day > 1):
            assert result.year >= now.year


# =========================================================================
# TestExtractCapMessage
# =========================================================================


class TestExtractCapMessage:
    """_extract_cap_message extraction and truncation."""

    def test_extracts_full_line_with_prefix(self):
        text = "Some preamble\nYou've hit your usage limit. Resets in 3h.\nMore text"
        result = _extract_cap_message(text, "You've hit your")
        assert result == "You've hit your usage limit. Resets in 3h."

    def test_returns_empty_when_no_match(self):
        assert _extract_cap_message('nothing here', "You've hit your") == ''

    def test_long_text_truncated_at_newline(self):
        line = "You've hit your " + 'x' * 300
        text = line + '\nNext line'
        result = _extract_cap_message(text, "You've hit your")
        assert result == line.strip()

    def test_no_newline_truncated_at_200_chars(self):
        text = "You've hit your " + 'x' * 300
        result = _extract_cap_message(text, "You've hit your")
        # The function takes from idx to idx+200 when no newline
        assert len(result) == 200


# =========================================================================
# TestHandleCapDetected
# =========================================================================


class TestHandleCapDetected:
    """_handle_cap_detected: account marking, gate state, cost events."""

    def test_marks_correct_account_by_token(self):
        gate = make_gate(['a', 'b'])
        token_b = gate._accounts[1].token
        gate._handle_cap_detected('reason', None, token_b)
        assert gate._accounts[0].capped is False
        assert gate._accounts[1].capped is True

    def test_unknown_token_caps_first_uncapped(self):
        gate = make_gate(['a', 'b'])
        gate._handle_cap_detected('reason', None, 'unknown-token')
        assert gate._accounts[0].capped is True
        assert gate._accounts[1].capped is False

    def test_all_capped_unknown_token_logs_warning(self, caplog):
        gate = make_gate(['a'])
        gate._accounts[0].capped = True
        import logging
        with caplog.at_level(logging.WARNING, logger='shared.usage_gate'):
            gate._handle_cap_detected('reason', None, 'unknown-token')
        assert any('no matching account' in r.message.lower() for r in caplog.records)

    def test_no_matching_account_returns_early(self):
        """All accounts capped + unknown token: should not crash."""
        gate = make_gate(['a'])
        gate._accounts[0].capped = True
        gate._handle_cap_detected('reason', None, 'unknown-token')
        # Just verifying no exception

    def test_sets_capped_true(self):
        gate = make_gate(['a'])
        token = gate._accounts[0].token
        gate._handle_cap_detected('reason', None, token)
        assert gate._accounts[0].capped is True

    def test_sets_probing_false_and_probe_in_flight_false(self):
        gate = make_gate(['a'])
        gate._accounts[0].probing = True
        gate._accounts[0].probe_in_flight = True
        token = gate._accounts[0].token
        gate._handle_cap_detected('reason', None, token)
        assert gate._accounts[0].probing is False
        assert gate._accounts[0].probe_in_flight is False

    def test_sets_resets_at(self):
        gate = make_gate(['a'])
        token = gate._accounts[0].token
        target = datetime.now(UTC) + timedelta(hours=2)
        gate._handle_cap_detected('reason', target, token)
        assert gate._accounts[0].resets_at == target

    def test_pause_started_at_set_only_if_none(self):
        gate = make_gate(['a'])
        token = gate._accounts[0].token
        original_time = datetime(2025, 1, 1, tzinfo=UTC)
        gate._accounts[0].pause_started_at = original_time
        gate._handle_cap_detected('reason', None, token)
        assert gate._accounts[0].pause_started_at == original_time

    def test_pause_started_at_set_when_none(self):
        gate = make_gate(['a'])
        token = gate._accounts[0].token
        before = datetime.now(UTC)
        gate._handle_cap_detected('reason', None, token)
        after = datetime.now(UTC)
        assert gate._accounts[0].pause_started_at is not None
        assert before <= gate._accounts[0].pause_started_at <= after

    @pytest.mark.asyncio
    async def test_starts_probe_task_when_wait_for_reset(self):
        # Must be async so asyncio.get_running_loop() succeeds inside
        # _start_account_resume_probe.
        gate = make_gate(['a'], wait_for_reset=True)
        token = gate._accounts[0].token
        gate._handle_cap_detected('reason', datetime.now(UTC) + timedelta(hours=1), token)
        acct = gate._accounts[0]
        assert acct.resume_task is not None
        # Clean up
        acct.resume_task.cancel()
        await asyncio.sleep(0)

    def test_no_probe_task_when_not_wait_for_reset(self):
        gate = make_gate(['a'], wait_for_reset=False)
        token = gate._accounts[0].token
        gate._handle_cap_detected('reason', None, token)
        assert gate._accounts[0].resume_task is None

    def test_closes_gate_when_all_capped(self):
        gate = make_gate(['a', 'b'])
        gate._accounts[0].capped = True
        token_b = gate._accounts[1].token
        gate._handle_cap_detected('reason', None, token_b)
        assert gate._open.is_set() is False

    def test_does_not_close_gate_when_others_uncapped(self):
        gate = make_gate(['a', 'b'])
        token_a = gate._accounts[0].token
        gate._handle_cap_detected('reason', None, token_a)
        assert gate._open.is_set() is True

    def test_fires_cost_event_with_cost_store(self):
        cost_store = make_mock_cost_store()
        gate = make_gate(['a'], cost_store=cost_store)
        token = gate._accounts[0].token
        with patch.object(gate, '_fire_cost_event') as mock_fire:
            gate._handle_cap_detected('reason', None, token)
        mock_fire.assert_called_once()

    def test_no_cost_event_without_cost_store(self):
        gate = make_gate(['a'], cost_store=None)
        token = gate._accounts[0].token
        with patch.object(gate, '_fire_cost_event') as mock_fire:
            gate._handle_cap_detected('reason', None, token)
        mock_fire.assert_not_called()

    def test_double_cap_pause_started_at_not_overwritten(self):
        gate = make_gate(['a'])
        token = gate._accounts[0].token
        gate._handle_cap_detected('first', None, token)
        first_pause = gate._accounts[0].pause_started_at
        gate._accounts[0].capped = False  # uncap to allow re-detection
        gate._handle_cap_detected('second', None, token)
        assert gate._accounts[0].pause_started_at == first_pause


# =========================================================================
# TestRefreshCappedAccounts
# =========================================================================


@pytest.mark.asyncio
class TestRefreshCappedAccounts:
    """_refresh_capped_accounts: uncapping logic based on resets_at."""

    async def test_past_resets_at_uncaps(self):
        gate = make_gate(['a'])
        gate._accounts[0].capped = True
        gate._accounts[0].resets_at = datetime.now(UTC) - timedelta(minutes=5)
        gate._accounts[0].pause_started_at = datetime.now(UTC) - timedelta(minutes=10)
        result = await gate._refresh_capped_accounts()
        assert result is True
        assert gate._accounts[0].capped is False

    async def test_future_resets_at_stays_capped(self):
        gate = make_gate(['a'])
        gate._accounts[0].capped = True
        gate._accounts[0].resets_at = datetime.now(UTC) + timedelta(hours=2)
        result = await gate._refresh_capped_accounts()
        assert result is False
        assert gate._accounts[0].capped is True

    async def test_none_resets_at_stays_capped(self):
        gate = make_gate(['a'])
        gate._accounts[0].capped = True
        gate._accounts[0].resets_at = None
        result = await gate._refresh_capped_accounts()
        assert result is False
        assert gate._accounts[0].capped is True

    async def test_mixed_past_future_none(self):
        gate = make_gate(['a', 'b', 'c'])
        for acct in gate._accounts:
            acct.capped = True
        gate._accounts[0].resets_at = datetime.now(UTC) - timedelta(minutes=1)
        gate._accounts[0].pause_started_at = datetime.now(UTC) - timedelta(minutes=5)
        gate._accounts[1].resets_at = datetime.now(UTC) + timedelta(hours=2)
        gate._accounts[2].resets_at = None
        result = await gate._refresh_capped_accounts()
        assert result is True
        assert gate._accounts[0].capped is False
        assert gate._accounts[1].capped is True
        assert gate._accounts[2].capped is True

    async def test_already_uncapped_no_change(self):
        gate = make_gate(['a'])
        # Not capped, no changes expected
        result = await gate._refresh_capped_accounts()
        assert result is False
        assert gate._accounts[0].capped is False

    async def test_gate_reopens_when_any_uncapped(self):
        gate = make_gate(['a'])
        gate._accounts[0].capped = True
        gate._accounts[0].resets_at = datetime.now(UTC) - timedelta(minutes=1)
        gate._accounts[0].pause_started_at = datetime.now(UTC) - timedelta(minutes=5)
        gate._open.clear()
        await gate._refresh_capped_accounts()
        assert gate._open.is_set() is True

    async def test_gate_stays_closed_when_none_uncapped(self):
        gate = make_gate(['a'])
        gate._accounts[0].capped = True
        gate._accounts[0].resets_at = datetime.now(UTC) + timedelta(hours=2)
        gate._open.clear()
        await gate._refresh_capped_accounts()
        assert gate._open.is_set() is False

    async def test_pause_duration_tracked(self):
        gate = make_gate(['a'])
        gate._accounts[0].capped = True
        gate._accounts[0].resets_at = datetime.now(UTC) - timedelta(minutes=1)
        gate._accounts[0].pause_started_at = datetime.now(UTC) - timedelta(seconds=60)
        await gate._refresh_capped_accounts()
        assert gate._total_pause_secs == pytest.approx(60, abs=2)

    async def test_pause_started_at_cleared_on_uncap(self):
        gate = make_gate(['a'])
        gate._accounts[0].capped = True
        gate._accounts[0].resets_at = datetime.now(UTC) - timedelta(minutes=1)
        gate._accounts[0].pause_started_at = datetime.now(UTC) - timedelta(minutes=5)
        await gate._refresh_capped_accounts()
        assert gate._accounts[0].pause_started_at is None

    async def test_multiple_accounts_uncap_simultaneously(self):
        gate = make_gate(['a', 'b'])
        for acct in gate._accounts:
            acct.capped = True
            acct.resets_at = datetime.now(UTC) - timedelta(minutes=1)
            acct.pause_started_at = datetime.now(UTC) - timedelta(minutes=5)
        result = await gate._refresh_capped_accounts()
        assert result is True
        assert all(not a.capped for a in gate._accounts)
        assert gate._open.is_set() is True

    # --- near_cap clearing on uncap ---

    async def test_clears_near_cap_on_uncap(self):
        # near_cap AND capped: when reset time passes, uncap should also clear near_cap.
        gate = make_gate(['a'])
        gate._accounts[0].capped = True
        gate._accounts[0].near_cap = True
        gate._accounts[0].resets_at = datetime.now(UTC) - timedelta(minutes=1)
        gate._accounts[0].pause_started_at = datetime.now(UTC) - timedelta(minutes=5)
        result = await gate._refresh_capped_accounts()
        assert result is True
        assert gate._accounts[0].capped is False
        assert gate._accounts[0].near_cap is False

    async def test_does_not_clear_near_cap_when_still_capped(self):
        # Regression guard: near_cap must NOT be touched when account stays capped.
        gate = make_gate(['a'])
        gate._accounts[0].capped = True
        gate._accounts[0].near_cap = True
        gate._accounts[0].resets_at = datetime.now(UTC) + timedelta(hours=2)
        result = await gate._refresh_capped_accounts()
        assert result is False
        assert gate._accounts[0].capped is True
        assert gate._accounts[0].near_cap is True


# =========================================================================
# TestBeforeInvokeCore
# =========================================================================


@pytest.mark.asyncio
class TestBeforeInvokeCore:
    """before_invoke: account selection, probing, session budget, blocking."""

    async def test_returns_first_available_token(self):
        gate = make_gate(['a', 'b'])
        token = await gate.before_invoke()
        assert token == 'fake-token-a'

    async def test_skips_capped_accounts(self):
        gate = make_gate(['a', 'b'])
        gate._accounts[0].capped = True
        token = await gate.before_invoke()
        assert token == 'fake-token-b'

    async def test_skips_probe_in_flight_accounts(self):
        gate = make_gate(['a', 'b'])
        gate._accounts[0].probe_in_flight = True
        token = await gate.before_invoke()
        assert token == 'fake-token-b'

    async def test_claims_probe_slot(self):
        gate = make_gate(['a'])
        gate._accounts[0].probing = True
        token = await gate.before_invoke()
        assert token == 'fake-token-a'
        assert gate._accounts[0].probing is False
        assert gate._accounts[0].probe_in_flight is True
        # Gate should be cleared (not set) to block other tasks
        assert gate._open.is_set() is False

    async def test_session_budget_below_limit(self):
        gate = make_gate(['a'], session_budget_usd=10.0)
        gate._cumulative_cost = 5.0
        token = await gate.before_invoke()
        assert token == 'fake-token-a'

    async def test_session_budget_at_exact_limit_raises(self):
        gate = make_gate(['a'], session_budget_usd=10.0)
        gate._cumulative_cost = 10.0
        with pytest.raises(SessionBudgetExhausted):
            await gate.before_invoke()

    async def test_session_budget_above_limit_raises(self):
        gate = make_gate(['a'], session_budget_usd=10.0)
        gate._cumulative_cost = 15.0
        with pytest.raises(SessionBudgetExhausted) as exc_info:
            await gate.before_invoke()
        assert exc_info.value.cumulative_cost == 15.0

    async def test_session_budget_not_configured(self):
        gate = make_gate(['a'], session_budget_usd=None)
        gate._cumulative_cost = 999999.0
        token = await gate.before_invoke()
        assert token == 'fake-token-a'

    async def test_session_budget_float_edge(self):
        gate = make_gate(['a'], session_budget_usd=10.0)
        gate._cumulative_cost = 9.999999999
        # 9.999999999 < 10.0, so should not raise
        token = await gate.before_invoke()
        assert token == 'fake-token-a'

    async def test_no_accounts_raises_runtime_error(self):
        gate = make_gate(['a'])
        gate._accounts.clear()
        with pytest.raises(RuntimeError, match='No OAuth accounts'):
            await gate.before_invoke()

    async def test_blocks_when_all_capped(self):
        gate = make_gate(['a'])
        gate._accounts[0].capped = True
        gate._accounts[0].resets_at = datetime.now(UTC) + timedelta(hours=5)
        with pytest.raises(asyncio.TimeoutError):
            await asyncio.wait_for(gate.before_invoke(), timeout=0.1)

    async def test_unblocks_when_gate_opened(self):
        gate = make_gate(['a'])
        gate._accounts[0].capped = True
        gate._accounts[0].resets_at = datetime.now(UTC) + timedelta(hours=5)

        async def open_gate_soon():
            await asyncio.sleep(0.05)
            gate._accounts[0].capped = False
            gate._open.set()

        task = asyncio.create_task(open_gate_soon())
        token = await asyncio.wait_for(gate.before_invoke(), timeout=1.0)
        assert token == 'fake-token-a'
        await task

    async def test_failover_event_on_account_change(self):
        cost_store = make_mock_cost_store()
        gate = make_gate(['a', 'b'], cost_store=cost_store)
        gate._accounts[0].capped = True
        gate._last_account_name = 'a'
        with patch.object(gate, '_fire_cost_event') as mock_fire:
            await gate.before_invoke()
        mock_fire.assert_called_once()
        args = mock_fire.call_args[0]
        assert args[0] == 'b'
        assert args[1] == 'failover'

    async def test_no_failover_event_on_first_invoke(self):
        cost_store = make_mock_cost_store()
        gate = make_gate(['a'], cost_store=cost_store)
        assert gate._last_account_name is None
        with patch.object(gate, '_fire_cost_event') as mock_fire:
            await gate.before_invoke()
        mock_fire.assert_not_called()

    async def test_no_failover_event_when_same_account(self):
        cost_store = make_mock_cost_store()
        gate = make_gate(['a'], cost_store=cost_store)
        gate._last_account_name = 'a'
        with patch.object(gate, '_fire_cost_event') as mock_fire:
            await gate.before_invoke()
        mock_fire.assert_not_called()

    async def test_no_cost_event_when_no_cost_store(self):
        gate = make_gate(['a', 'b'], cost_store=None)
        gate._accounts[0].capped = True
        gate._last_account_name = 'a'
        with patch.object(gate, '_fire_cost_event') as mock_fire:
            await gate.before_invoke()
        mock_fire.assert_not_called()


# =========================================================================
# TestConfirmAccountOk
# =========================================================================


class TestConfirmAccountOk:
    """confirm_account_ok: clearing probing state."""

    def test_clears_probe_in_flight(self):
        gate = make_gate(['a'])
        gate._accounts[0].probe_in_flight = True
        gate.confirm_account_ok(gate._accounts[0].token)
        assert gate._accounts[0].probe_in_flight is False

    def test_resets_probe_count(self):
        gate = make_gate(['a'])
        gate._accounts[0].probe_in_flight = True
        gate._accounts[0].probe_count = 5
        gate.confirm_account_ok(gate._accounts[0].token)
        assert gate._accounts[0].probe_count == 0

    def test_opens_gate(self):
        gate = make_gate(['a'])
        gate._accounts[0].probe_in_flight = True
        gate._open.clear()
        gate.confirm_account_ok(gate._accounts[0].token)
        assert gate._open.is_set() is True

    def test_noop_when_probe_in_flight_false(self):
        gate = make_gate(['a'])
        gate._accounts[0].probe_in_flight = False
        gate._accounts[0].probe_count = 3
        gate.confirm_account_ok(gate._accounts[0].token)
        # probe_count should NOT be reset when probe_in_flight was already False
        assert gate._accounts[0].probe_count == 3

    def test_noop_with_none_token(self):
        gate = make_gate(['a'])
        gate.confirm_account_ok(None)
        # Just verifying no exception

    def test_noop_with_unknown_token(self):
        gate = make_gate(['a'])
        gate.confirm_account_ok('completely-unknown-token')
        # Just verifying no exception

    def test_no_crash_with_empty_accounts(self):
        gate = make_gate(['a'])
        gate._accounts.clear()
        gate.confirm_account_ok('any-token')
        # No crash

    # --- near_cap clearing ---

    def test_clears_near_cap_with_probe_in_flight(self):
        gate = make_gate(['a'])
        gate._accounts[0].near_cap = True
        gate._accounts[0].probe_in_flight = True
        gate.confirm_account_ok(gate._accounts[0].token)
        assert gate._accounts[0].near_cap is False

    def test_clears_near_cap_without_probe_in_flight(self):
        # Happy-path bug: near_cap set, billing reset, no full cap hit, probe never ran.
        gate = make_gate(['a'])
        gate._accounts[0].near_cap = True
        gate._accounts[0].probe_in_flight = False
        gate._accounts[0].probe_count = 3
        gate._open.clear()
        gate.confirm_account_ok(gate._accounts[0].token)
        assert gate._accounts[0].near_cap is False
        # probe_count and gate should NOT change (only near_cap is cleared in this branch)
        assert gate._accounts[0].probe_count == 3
        assert gate._open.is_set() is False

    def test_does_not_clear_near_cap_with_none_token(self):
        gate = make_gate(['a'])
        gate._accounts[0].near_cap = True
        gate.confirm_account_ok(None)
        assert gate._accounts[0].near_cap is True

    def test_does_not_clear_near_cap_with_unknown_token(self):
        gate = make_gate(['a'])
        gate._accounts[0].near_cap = True
        gate.confirm_account_ok('completely-unknown-token')
        assert gate._accounts[0].near_cap is True


# =========================================================================
# TestProperties
# =========================================================================


class TestProperties:
    """Property accessors: is_paused, paused_reason, cumulative_cost, etc."""

    # --- is_paused ---

    def test_is_paused_all_capped(self):
        gate = make_gate(['a', 'b'])
        gate._accounts[0].capped = True
        gate._accounts[1].capped = True
        assert gate.is_paused is True

    def test_is_paused_some_capped(self):
        gate = make_gate(['a', 'b'])
        gate._accounts[0].capped = True
        assert gate.is_paused is False

    def test_is_paused_none_capped(self):
        gate = make_gate(['a', 'b'])
        assert gate.is_paused is False

    def test_is_paused_no_accounts(self):
        gate = make_gate(['a'])
        gate._accounts.clear()
        assert gate.is_paused is False

    # --- paused_reason ---

    def test_paused_reason_returns_last_set(self):
        gate = make_gate(['a'])
        gate._paused_reason = 'All accounts capped'
        assert gate.paused_reason == 'All accounts capped'

    # --- cumulative_cost ---

    def test_cumulative_cost_starts_at_zero(self):
        gate = make_gate(['a'])
        assert gate.cumulative_cost == 0.0

    def test_cumulative_cost_accumulates(self):
        gate = make_gate(['a'])
        gate.on_agent_complete(1.5)
        gate.on_agent_complete(2.3)
        assert gate.cumulative_cost == pytest.approx(3.8)

    # --- total_pause_secs ---

    def test_total_pause_secs_zero_when_no_pause(self):
        gate = make_gate(['a'])
        assert gate.total_pause_secs == 0.0

    def test_total_pause_secs_includes_current_active_pause(self):
        gate = make_gate(['a'])
        gate._pause_started_at = datetime.now(UTC) - timedelta(seconds=30)
        assert gate.total_pause_secs == pytest.approx(30, abs=2)

    def test_total_pause_secs_sum_of_completed_pauses(self):
        gate = make_gate(['a'])
        gate._total_pause_secs = 120.0
        assert gate.total_pause_secs == pytest.approx(120.0)

    # --- account_count ---

    def test_account_count(self):
        gate = make_gate(['a', 'b', 'c'])
        assert gate.account_count == 3

    # --- active_account_name ---

    def test_active_account_name_first_uncapped(self):
        gate = make_gate(['a', 'b'])
        assert gate.active_account_name == 'a'

    def test_active_account_name_skips_capped(self):
        gate = make_gate(['a', 'b'])
        gate._accounts[0].capped = True
        assert gate.active_account_name == 'b'

    def test_active_account_name_none_when_all_capped(self):
        gate = make_gate(['a', 'b'])
        gate._accounts[0].capped = True
        gate._accounts[1].capped = True
        assert gate.active_account_name is None


# =========================================================================
# TestShutdownExhaustive
# =========================================================================


@pytest.mark.asyncio
class TestShutdownExhaustive:
    """shutdown: task cancellation, draining, cleanup, idempotency."""

    async def test_cancels_resume_tasks(self):
        gate = make_gate(['a', 'b'], wait_for_reset=True)
        # Manually create resume tasks
        for acct in gate._accounts:
            acct.capped = True
            acct.resets_at = datetime.now(UTC) + timedelta(hours=5)
            gate._start_account_resume_probe(acct)
            assert acct.resume_task is not None

        await gate.shutdown()
        for acct in gate._accounts:
            assert acct.resume_task is None

    async def test_drains_background_tasks(self):
        cost_store = make_mock_cost_store()
        gate = make_gate(['a'], cost_store=cost_store)

        block_event = asyncio.Event()

        async def blocking_save(*args, **kwargs):
            await block_event.wait()

        cost_store.save_account_event.side_effect = blocking_save
        gate._fire_cost_event('a', 'cap_hit', '{}')
        await asyncio.sleep(0)
        assert len(gate._background_tasks) >= 1

        await gate.shutdown()
        assert len(gate._background_tasks) == 0

    async def test_empty_state_no_crash(self):
        gate = make_gate(['a'])
        gate._accounts.clear()
        gate._background_tasks.clear()
        await gate.shutdown()

    async def test_probe_config_dir_cleaned_up(self):
        gate = make_gate(['a'])
        with patch.object(gate._probe_config_dir, 'cleanup') as mock_cleanup:
            await gate.shutdown()
        mock_cleanup.assert_called_once()

    async def test_double_shutdown_no_crash(self):
        gate = make_gate(['a'])
        await gate.shutdown()
        await gate.shutdown()

    async def test_resume_task_already_done_no_crash(self):
        gate = make_gate(['a'], wait_for_reset=True)
        acct = gate._accounts[0]
        # Create a task that finishes immediately
        acct.resume_task = asyncio.create_task(asyncio.sleep(0))
        await asyncio.sleep(0.01)
        assert acct.resume_task.done()
        await gate.shutdown()


# ---------------------------------------------------------------------------
# Coverage gap: _init_accounts fallback paths
# ---------------------------------------------------------------------------


@pytest.mark.asyncio
class TestInitAccountsFallbacks:
    """Cover lines 130-143: missing env vars, default credential fallback."""

    async def test_missing_env_var_skips_account(self):
        """Account with missing env var is skipped (lines 130-134)."""
        config = UsageCapConfig(
            accounts=[AccountConfig(name='missing', oauth_token_env='NONEXISTENT_TOKEN_VAR')],
            wait_for_reset=False,
        )
        gate = UsageGate(config)
        assert len(gate._accounts) == 0 or gate._accounts[0].name == 'default'

    async def test_no_accounts_no_default_credential(self):
        """No configured accounts + no default cred → empty (lines 138-143)."""
        config = UsageCapConfig(accounts=[], wait_for_reset=False)
        with patch('shared.usage_gate._read_oauth_token', return_value=None):
            gate = UsageGate(config)
        assert gate._accounts == []

    async def test_no_accounts_falls_back_to_default_credential(self):
        """No configured accounts → falls back to default credential (lines 138-141)."""
        config = UsageCapConfig(accounts=[], wait_for_reset=False)
        with patch('shared.usage_gate._read_oauth_token', return_value='default-tok'):
            gate = UsageGate(config)
        assert len(gate._accounts) == 1
        assert gate._accounts[0].name == 'default'
        assert gate._accounts[0].token == 'default-tok'

    async def test_partial_env_vars_skips_missing(self):
        """Mix of valid and missing env vars (lines 130-134)."""
        config = UsageCapConfig(
            accounts=[
                AccountConfig(name='valid', oauth_token_env='TEST_VALID_TOKEN'),
                AccountConfig(name='missing', oauth_token_env='TEST_MISSING_TOKEN'),
            ],
            wait_for_reset=False,
        )
        with patch.dict(os.environ, {'TEST_VALID_TOKEN': 'tok-valid'}):
            gate = UsageGate(config)
        assert len(gate._accounts) == 1
        assert gate._accounts[0].name == 'valid'


# ---------------------------------------------------------------------------
# Coverage gap: _read_oauth_token
# ---------------------------------------------------------------------------


class TestReadOauthToken:
    """Cover lines 618-638: reading default credential from disk."""

    def test_direct_access_token(self, tmp_path):
        cred_file = tmp_path / '.credentials.json'
        cred_file.write_text(json.dumps({'accessToken': 'tok-direct'}))
        with patch('shared.usage_gate.CREDENTIALS_PATH', cred_file):
            assert _read_oauth_token() == 'tok-direct'

    def test_direct_access_token_snake_case(self, tmp_path):
        cred_file = tmp_path / '.credentials.json'
        cred_file.write_text(json.dumps({'access_token': 'tok-snake'}))
        with patch('shared.usage_gate.CREDENTIALS_PATH', cred_file):
            assert _read_oauth_token() == 'tok-snake'

    def test_nested_access_token(self, tmp_path):
        cred_file = tmp_path / '.credentials.json'
        cred_file.write_text(json.dumps({'claudeAiOauth': {'accessToken': 'tok-nested'}}))
        with patch('shared.usage_gate.CREDENTIALS_PATH', cred_file):
            assert _read_oauth_token() == 'tok-nested'

    def test_nested_access_token_snake_case(self, tmp_path):
        cred_file = tmp_path / '.credentials.json'
        cred_file.write_text(json.dumps({'provider': {'access_token': 'tok-nested-snake'}}))
        with patch('shared.usage_gate.CREDENTIALS_PATH', cred_file):
            assert _read_oauth_token() == 'tok-nested-snake'

    def test_file_not_found(self, tmp_path):
        cred_file = tmp_path / '.credentials.json'
        with patch('shared.usage_gate.CREDENTIALS_PATH', cred_file):
            assert _read_oauth_token() is None

    def test_invalid_json(self, tmp_path):
        cred_file = tmp_path / '.credentials.json'
        cred_file.write_text('not json')
        with patch('shared.usage_gate.CREDENTIALS_PATH', cred_file):
            assert _read_oauth_token() is None

    def test_empty_dict(self, tmp_path):
        cred_file = tmp_path / '.credentials.json'
        cred_file.write_text('{}')
        with patch('shared.usage_gate.CREDENTIALS_PATH', cred_file):
            assert _read_oauth_token() is None

    def test_non_dict_json(self, tmp_path):
        cred_file = tmp_path / '.credentials.json'
        cred_file.write_text('"just a string"')
        with patch('shared.usage_gate.CREDENTIALS_PATH', cred_file):
            assert _read_oauth_token() is None


# ---------------------------------------------------------------------------
# Coverage gap: property setters (lines 597-610)
# ---------------------------------------------------------------------------


class TestPropertySetters:
    """Cover project_id and run_id getter/setter pairs."""

    def test_project_id_round_trip(self):
        gate = make_gate(['a'])
        assert gate.project_id is None
        gate.project_id = 'proj-123'
        assert gate.project_id == 'proj-123'
        gate.project_id = None
        assert gate.project_id is None

    def test_run_id_round_trip(self):
        gate = make_gate(['a'])
        assert gate.run_id is None
        gate.run_id = 'run-456'
        assert gate.run_id == 'run-456'
        gate.run_id = None
        assert gate.run_id is None


# ---------------------------------------------------------------------------
# Coverage gap: _start_account_resume_probe RuntimeError (lines 321-322)
# ---------------------------------------------------------------------------


class TestStartAccountResumeProbeNoLoop:
    """Cover lines 321-322: no running event loop."""

    def test_no_event_loop_does_not_crash(self):
        """_start_account_resume_probe must not crash when no event loop is running."""
        gate = make_gate(['a'], wait_for_reset=True)
        acct = gate._accounts[0]
        # Patch get_running_loop to raise RuntimeError (no loop)
        with patch('shared.usage_gate.asyncio.get_running_loop', side_effect=RuntimeError):
            gate._start_account_resume_probe(acct)
        assert acct.resume_task is None


# ---------------------------------------------------------------------------
# Coverage gap: _write_cost_event error path (line 332, 343)
# ---------------------------------------------------------------------------


@pytest.mark.asyncio
class TestWriteCostEventErrors:
    """Cover _write_cost_event error handling."""

    async def test_write_cost_event_swallows_exception(self, caplog):
        """Exception in save_account_event is caught and logged (line 343)."""
        store = make_mock_cost_store()
        store.save_account_event.side_effect = RuntimeError('db error')
        gate = make_gate(['a'], cost_store=store)
        # Should not raise
        import logging
        with caplog.at_level(logging.WARNING, logger='shared.usage_gate'):
            await gate._write_cost_event('a', 'test', '{}')
        assert any('CostStore write failed' in r.message for r in caplog.records)

    async def test_write_cost_event_noop_without_cost_store(self):
        """_write_cost_event returns immediately when cost_store is None (line 332)."""
        gate = make_gate(['a'], cost_store=None)
        # Should not raise
        await gate._write_cost_event('a', 'test', '{}')


# ---------------------------------------------------------------------------
# Coverage gap: _fire_cost_event no-event-loop path (lines 353-360)
# ---------------------------------------------------------------------------


@pytest.mark.asyncio
class TestFireCostEventNoLoop:

    async def test_fire_cost_event_noop_without_cost_store(self):
        """_fire_cost_event returns immediately when cost_store is None (line 353)."""
        gate = make_gate(['a'], cost_store=None)
        gate._fire_cost_event('a', 'test', '{}')
        assert len(gate._background_tasks) == 0


# ---------------------------------------------------------------------------
# Coverage gap: before_invoke "All accounts capped" log (line 219)
# ---------------------------------------------------------------------------


@pytest.mark.asyncio
class TestBeforeInvokeAllCappedLog:

    async def test_all_capped_log_reached(self):
        """When all accounts are capped with future resets_at, the 'All accounts capped'
        log line (219) is reached before blocking on _open.wait()."""
        gate = make_gate(['a'])
        gate._accounts[0].capped = True
        gate._accounts[0].resets_at = datetime.now(UTC) + timedelta(hours=1)

        # before_invoke will call _refresh_capped_accounts (nothing to uncap),
        # then reach line 219 and block on _open.wait()
        with pytest.raises(asyncio.TimeoutError):
            await asyncio.wait_for(gate.before_invoke(), timeout=0.1)


# ---------------------------------------------------------------------------
# Coverage gap: reset time parsing edge cases (lines 684, 692, 704-705, 725, 736-737)
# ---------------------------------------------------------------------------


class TestResetTimeParsingEdgeCases:
    """Cover remaining branches in _parse_resets_at."""

    def test_absolute_date_with_unknown_month(self):
        """Unknown month abbreviation should fall through to time-only or fallback."""
        dt = _parse_resets_at('resets Xyz 30, 6pm (UTC)')
        # Should fall back to 1h from now since month is invalid
        expected = datetime.now(UTC) + timedelta(hours=1)
        assert abs((dt - expected).total_seconds()) < 5

    def test_absolute_date_unparseable_time_format(self):
        """Time format that doesn't match any pattern (line 692)."""
        dt = _parse_resets_at('resets Mar 30, 99:99pm (UTC)')
        # Should fall through to time-only or fallback
        expected = datetime.now(UTC) + timedelta(hours=1)
        assert abs((dt - expected).total_seconds()) < 5

    def test_absolute_date_with_bad_timezone(self):
        """Invalid timezone (lines 704-705 exception catch)."""
        dt = _parse_resets_at('resets Mar 30, 6pm (Fake/Zone)')
        # Falls through to time-only match or 1h fallback
        expected = datetime.now(UTC) + timedelta(hours=1)
        assert abs((dt - expected).total_seconds()) < 5

    def test_time_only_unparseable_format(self):
        """Time-only with unparseable format (line 725 fallback)."""
        dt = _parse_resets_at('resets 99:99pm (UTC)')
        # Falls back to 1h
        expected = datetime.now(UTC) + timedelta(hours=1)
        assert abs((dt - expected).total_seconds()) < 5

    def test_time_only_bad_timezone(self):
        """Time-only with invalid timezone (lines 736-737 exception catch)."""
        dt = _parse_resets_at('resets 6pm (Fake/Zone)')
        # Falls back to 1h
        expected = datetime.now(UTC) + timedelta(hours=1)
        assert abs((dt - expected).total_seconds()) < 5

    def test_absolute_date_past_year_wraps(self):
        """Absolute date in past → next year (line 701-702)."""
        # Use a month that's definitely in the past
        now = datetime.now(UTC)
        # If we're in April, January 1 is in the past
        dt = _parse_resets_at('resets Jan 1, 12am (UTC)')
        assert dt.year >= now.year  # either this year (if Jan is future) or next
