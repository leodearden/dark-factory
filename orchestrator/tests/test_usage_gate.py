"""Tests for usage cap detection, gate lifecycle, and reset time parsing."""

from __future__ import annotations

import asyncio
from datetime import UTC, datetime, timedelta
from unittest.mock import AsyncMock

import pytest
from shared.config_models import AccountConfig, UsageCapConfig
from shared.usage_gate import (
    AccountState,
    SessionBudgetExhausted,
    UsageGate,
    _extract_cap_message,
    _parse_resets_at,
)

# --- Helpers ---


def _make_gate(
    num_accounts: int = 2,
    wait_for_reset: bool = False,
    session_budget_usd: float | None = None,
    probe_interval_secs: int = 300,
    max_probe_interval_secs: int = 1800,
) -> UsageGate:
    """Create a UsageGate with mock accounts (tokens pre-injected)."""
    acct_cfgs = [
        AccountConfig(name='max-a', oauth_token_env='CLAUDE_OAUTH_A'),
        AccountConfig(name='max-b', oauth_token_env='CLAUDE_OAUTH_B'),
        AccountConfig(name='max-c', oauth_token_env='CLAUDE_OAUTH_C'),
    ][:num_accounts]
    config = UsageCapConfig(
        wait_for_reset=wait_for_reset,
        session_budget_usd=session_budget_usd,
        probe_interval_secs=probe_interval_secs,
        max_probe_interval_secs=max_probe_interval_secs,
        accounts=acct_cfgs,
    )
    gate = UsageGate.__new__(UsageGate)
    gate._config = config
    gate._open = asyncio.Event()
    gate._open.set()
    gate._lock = asyncio.Lock()
    gate._cumulative_cost = 0.0
    gate._paused_reason = ''
    gate._pause_started_at = None
    gate._total_pause_secs = 0.0
    gate._cost_store = None
    gate._project_id = None
    gate._run_id = None
    gate._last_account_name = None
    tokens = ['token-a', 'token-b', 'token-c']
    gate._accounts = [
        AccountState(name=f'max-{chr(97+i)}', token=tokens[i])
        for i in range(num_accounts)
    ]
    return gate


# --- Cap hit detection ---


class TestDetectCapHit:
    def test_detects_hit_your_limit(self):
        gate = _make_gate(num_accounts=1)
        stderr = "You've hit your usage limit for Claude. Your usage resets in 3h."
        assert gate.detect_cap_hit(stderr, '', oauth_token='token-a') is True

    def test_detects_youve_used(self):
        gate = _make_gate(num_accounts=1)
        result = "You've used all available tokens. Usage resets in 45m."
        assert gate.detect_cap_hit('', result, oauth_token='token-a') is True

    def test_detects_out_of_extra_usage(self):
        gate = _make_gate(num_accounts=1)
        stderr = "You're out of extra usage for this period."
        assert gate.detect_cap_hit(stderr, '', oauth_token='token-a') is True

    def test_detects_near_cap_close_to(self):
        gate = _make_gate(num_accounts=1)
        stderr = "You're close to your usage limit."
        assert gate.detect_cap_hit(stderr, '', oauth_token='token-a') is True

    def test_detects_near_cap_extra_usage(self):
        gate = _make_gate(num_accounts=1)
        result = "You're now using extra usage credits."
        assert gate.detect_cap_hit('', result, oauth_token='token-a') is True

    def test_no_false_positive_normal_output(self):
        gate = _make_gate(num_accounts=1)
        assert gate.detect_cap_hit('', 'Task completed successfully') is False
        assert gate.detect_cap_hit('some warning', 'all good') is False

    def test_case_insensitive(self):
        gate = _make_gate(num_accounts=1)
        assert gate.detect_cap_hit("YOU'VE HIT YOUR limit", '', oauth_token='token-a') is True


# --- Reset time parsing ---


class TestParseResetsAt:
    def test_relative_hours(self):
        dt = _parse_resets_at('resets in 3h')
        assert dt is not None
        expected = datetime.now(UTC) + timedelta(hours=3)
        assert abs((dt - expected).total_seconds()) < 2

    def test_relative_minutes(self):
        dt = _parse_resets_at('resets in 45m')
        assert dt is not None
        expected = datetime.now(UTC) + timedelta(minutes=45)
        assert abs((dt - expected).total_seconds()) < 2

    def test_relative_days(self):
        dt = _parse_resets_at('resets in 2d')
        assert dt is not None
        expected = datetime.now(UTC) + timedelta(days=2)
        assert abs((dt - expected).total_seconds()) < 2

    def test_absolute_time_with_timezone(self):
        dt = _parse_resets_at('resets 9pm (UTC)')
        assert dt is not None
        # Should be in the future (today or tomorrow at 9pm UTC)
        assert dt > datetime.now(UTC) - timedelta(hours=1)

    def test_absolute_time_with_minutes(self):
        dt = _parse_resets_at('resets 3:00 AM (UTC)')
        assert dt is not None

    def test_fallback_to_1_hour(self):
        dt = _parse_resets_at('no reset info here')
        assert dt is not None
        expected = datetime.now(UTC) + timedelta(hours=1)
        assert abs((dt - expected).total_seconds()) < 2

    def test_embedded_in_longer_text(self):
        text = "You've hit your limit. Your usage resets in 5h. Please wait."
        dt = _parse_resets_at(text)
        assert dt is not None
        expected = datetime.now(UTC) + timedelta(hours=5)
        assert abs((dt - expected).total_seconds()) < 2


# --- Extract cap message ---


class TestExtractCapMessage:
    def test_extracts_full_line(self):
        text = "Some preamble\nYou've hit your usage limit for Claude.\nMore text"
        msg = _extract_cap_message(text, "You've hit your")
        assert msg == "You've hit your usage limit for Claude."

    def test_returns_empty_on_no_match(self):
        assert _extract_cap_message('no match here', "You've hit your") == ''


# --- Session budget ---


class TestSessionBudget:
    @pytest.mark.asyncio
    async def test_budget_not_exceeded(self):
        gate = _make_gate(num_accounts=1, session_budget_usd=10.0)
        gate.on_agent_complete(5.0)
        # Should not raise
        await gate.before_invoke()

    @pytest.mark.asyncio
    async def test_budget_exceeded_raises(self):
        gate = _make_gate(num_accounts=1, session_budget_usd=10.0)
        gate.on_agent_complete(10.0)
        with pytest.raises(SessionBudgetExhausted) as exc_info:
            await gate.before_invoke()
        assert exc_info.value.cumulative_cost == 10.0

    @pytest.mark.asyncio
    async def test_no_budget_configured(self):
        gate = _make_gate(num_accounts=1)
        gate.on_agent_complete(999.0)
        # Should not raise when no budget is set
        await gate.before_invoke()


# --- Gate lifecycle ---


class TestGateLifecycle:
    @pytest.mark.asyncio
    async def test_starts_open(self):
        gate = _make_gate(num_accounts=1)
        assert not gate.is_paused
        # before_invoke should return immediately
        token = await asyncio.wait_for(gate.before_invoke(), timeout=0.1)
        assert token == 'token-a'

    @pytest.mark.asyncio
    async def test_capped_account_means_paused(self):
        gate = _make_gate(num_accounts=1)
        gate._accounts[0].capped = True
        assert gate.is_paused

    @pytest.mark.asyncio
    async def test_uncap_unpauses(self):
        gate = _make_gate(num_accounts=1)
        gate._accounts[0].capped = True
        assert gate.is_paused
        gate._accounts[0].capped = False
        assert not gate.is_paused

    @pytest.mark.asyncio
    async def test_before_invoke_blocks_when_all_capped(self):
        gate = _make_gate(num_accounts=1)
        gate._accounts[0].capped = True
        # resets_at in future — refresh won't uncap
        gate._accounts[0].resets_at = datetime.now(UTC) + timedelta(hours=1)

        with pytest.raises(asyncio.TimeoutError):
            await asyncio.wait_for(gate.before_invoke(), timeout=0.1)

    @pytest.mark.asyncio
    async def test_before_invoke_unblocks_on_uncap(self):
        gate = _make_gate(num_accounts=1)
        gate._accounts[0].capped = True
        gate._accounts[0].resets_at = datetime.now(UTC) + timedelta(hours=1)

        async def uncap_after_delay():
            await asyncio.sleep(0.05)
            gate._accounts[0].capped = False
            gate._open.set()

        asyncio.create_task(uncap_after_delay())
        token = await asyncio.wait_for(gate.before_invoke(), timeout=0.5)
        assert token == 'token-a'

    @pytest.mark.asyncio
    async def test_pause_tracks_duration(self):
        gate = _make_gate(num_accounts=1)
        assert gate.total_pause_secs == 0.0
        gate._pause_started_at = datetime.now(UTC)
        await asyncio.sleep(0.05)
        assert gate.total_pause_secs > 0

    @pytest.mark.asyncio
    async def test_cumulative_cost(self):
        gate = _make_gate(num_accounts=1)
        assert gate.cumulative_cost == 0.0
        gate.on_agent_complete(1.5)
        gate.on_agent_complete(2.3)
        assert gate.cumulative_cost == pytest.approx(3.8)

    @pytest.mark.asyncio
    async def test_no_accounts_raises(self):
        gate = _make_gate(num_accounts=1)
        gate._accounts = []
        with pytest.raises(RuntimeError, match='No OAuth accounts'):
            await gate.before_invoke()


# --- Startup check ---


class TestStartupCheck:
    @pytest.mark.asyncio
    async def test_startup_is_noop(self):
        """Startup check no longer queries any API — all accounts remain open."""
        gate = _make_gate(num_accounts=2)
        await gate.check_at_startup()
        assert not gate.is_paused
        assert gate._accounts[0].capped is False
        assert gate._accounts[1].capped is False


# --- Multi-account before_invoke ---


class TestBeforeInvoke:
    @pytest.mark.asyncio
    async def test_returns_first_available_token(self):
        gate = _make_gate()
        token = await gate.before_invoke()
        assert token == 'token-a'

    @pytest.mark.asyncio
    async def test_returns_second_when_first_capped(self):
        gate = _make_gate()
        gate._accounts[0].capped = True
        token = await gate.before_invoke()
        assert token == 'token-b'

    @pytest.mark.asyncio
    async def test_blocks_when_all_capped_then_resumes(self):
        gate = _make_gate()
        gate._accounts[0].capped = True
        gate._accounts[0].resets_at = datetime.now(UTC) + timedelta(hours=1)
        gate._accounts[1].capped = True
        gate._accounts[1].resets_at = datetime.now(UTC) + timedelta(hours=1)

        async def uncap_after_delay():
            await asyncio.sleep(0.05)
            gate._accounts[1].capped = False
            gate._open.set()

        asyncio.create_task(uncap_after_delay())
        token = await asyncio.wait_for(gate.before_invoke(), timeout=0.5)
        assert token == 'token-b'

    @pytest.mark.asyncio
    async def test_refresh_uncaps_account_before_blocking(self):
        """When all accounts are flagged as capped but one's reset time has
        passed, before_invoke should uncap it via the time-based refresh."""
        gate = _make_gate()
        gate._accounts[0].capped = True
        gate._accounts[0].pause_started_at = datetime.now(UTC)
        gate._accounts[0].resets_at = datetime.now(UTC) - timedelta(hours=1)  # past
        gate._accounts[1].capped = True
        gate._accounts[1].resets_at = datetime.now(UTC) + timedelta(hours=1)  # future

        token = await asyncio.wait_for(gate.before_invoke(), timeout=0.5)

        assert token == 'token-a'
        assert gate._accounts[0].capped is False
        assert gate._accounts[1].capped is True


# --- Cap hit detection (multi-account) ---


class TestDetectCapHitMultiAccount:
    def test_marks_correct_account_capped(self):
        gate = _make_gate()
        hit = gate.detect_cap_hit(
            "You've hit your usage limit", '', 'claude', oauth_token='token-a',
        )
        assert hit is True
        assert gate._accounts[0].capped is True
        assert gate._accounts[1].capped is False
        # Global gate still open (account B available)
        assert gate._open.is_set()

    def test_closes_global_gate_when_all_capped(self):
        gate = _make_gate()
        gate._accounts[1].capped = True  # pre-cap account B
        hit = gate.detect_cap_hit(
            "You've hit your usage limit", '', 'claude', oauth_token='token-a',
        )
        assert hit is True
        assert gate._accounts[0].capped is True
        # Global gate closed — all capped
        assert not gate._open.is_set()

    def test_unknown_token_caps_first_uncapped(self):
        gate = _make_gate()
        hit = gate.detect_cap_hit(
            "You've hit your usage limit", '', 'claude', oauth_token='unknown-token',
        )
        assert hit is True
        assert gate._accounts[0].capped is True
        # B still available
        assert gate._open.is_set()


# --- is_paused ---


class TestIsPaused:
    def test_not_paused_when_one_account_available(self):
        gate = _make_gate()
        gate._accounts[0].capped = True
        assert not gate.is_paused

    def test_paused_when_all_accounts_capped(self):
        gate = _make_gate()
        gate._accounts[0].capped = True
        gate._accounts[1].capped = True
        assert gate.is_paused

    def test_not_paused_with_no_accounts(self):
        gate = _make_gate(num_accounts=1)
        gate._accounts = []
        assert not gate.is_paused


# --- active_account_name ---


class TestActiveAccountName:
    def test_returns_first_uncapped(self):
        gate = _make_gate()
        assert gate.active_account_name == 'max-a'

    def test_returns_second_when_first_capped(self):
        gate = _make_gate()
        gate._accounts[0].capped = True
        assert gate.active_account_name == 'max-b'

    def test_returns_none_when_all_capped(self):
        gate = _make_gate()
        gate._accounts[0].capped = True
        gate._accounts[1].capped = True
        assert gate.active_account_name is None


# --- Shutdown ---


class TestShutdown:
    @pytest.mark.asyncio
    async def test_cancels_all_account_resume_tasks(self):
        gate = _make_gate()

        async def never_return():
            await asyncio.sleep(3600)

        for acct in gate._accounts:
            acct.resume_task = asyncio.create_task(never_return())

        await gate.shutdown()

        for acct in gate._accounts:
            assert acct.resume_task is None


# --- Three-account failover ---


class TestThreeAccountFailover:
    @pytest.mark.asyncio
    async def test_rotates_through_all_three(self):
        """Cap A → picks B, cap B → picks C."""
        gate = _make_gate(num_accounts=3)

        token = await gate.before_invoke()
        assert token == 'token-a'

        gate.detect_cap_hit("You've hit your limit", '', 'claude', 'token-a')
        token = await gate.before_invoke()
        assert token == 'token-b'

        gate.detect_cap_hit("You've hit your limit", '', 'claude', 'token-b')
        token = await gate.before_invoke()
        assert token == 'token-c'

    @pytest.mark.asyncio
    async def test_skips_middle_capped_account(self):
        gate = _make_gate(num_accounts=3)
        gate._accounts[0].capped = True
        gate._accounts[1].capped = True
        token = await gate.before_invoke()
        assert token == 'token-c'


# --- Refresh (time-based) ---


class TestRefreshCappedAccounts:
    @pytest.mark.asyncio
    async def test_uncaps_account_when_resets_at_passed(self):
        gate = _make_gate(num_accounts=1)
        acct = gate._accounts[0]
        acct.capped = True
        acct.resets_at = datetime.now(UTC) - timedelta(hours=1)
        acct.pause_started_at = datetime.now(UTC) - timedelta(hours=2)

        any_uncapped = await gate._refresh_capped_accounts()

        assert any_uncapped is True
        assert acct.capped is False

    @pytest.mark.asyncio
    async def test_keeps_account_capped_when_resets_at_in_future(self):
        gate = _make_gate(num_accounts=1)
        acct = gate._accounts[0]
        acct.capped = True
        acct.resets_at = datetime.now(UTC) + timedelta(hours=1)

        any_uncapped = await gate._refresh_capped_accounts()

        assert any_uncapped is False
        assert acct.capped is True

    @pytest.mark.asyncio
    async def test_keeps_account_capped_when_resets_at_is_none(self):
        gate = _make_gate(num_accounts=1)
        acct = gate._accounts[0]
        acct.capped = True
        acct.resets_at = None

        any_uncapped = await gate._refresh_capped_accounts()

        assert any_uncapped is False
        assert acct.capped is True

    @pytest.mark.asyncio
    async def test_uncaps_multiple_past_reset_accounts(self):
        gate = _make_gate(num_accounts=2)
        for acct in gate._accounts:
            acct.capped = True
            acct.resets_at = datetime.now(UTC) - timedelta(hours=1)
            acct.pause_started_at = datetime.now(UTC) - timedelta(hours=2)

        any_uncapped = await gate._refresh_capped_accounts()

        assert any_uncapped is True
        assert not gate._accounts[0].capped
        assert not gate._accounts[1].capped

    @pytest.mark.asyncio
    async def test_mixed_reset_times(self):
        """One account past reset, one still in future."""
        gate = _make_gate(num_accounts=3)
        for acct in gate._accounts:
            acct.capped = True
            acct.pause_started_at = datetime.now(UTC)
        gate._accounts[0].resets_at = datetime.now(UTC) + timedelta(hours=1)  # future
        gate._accounts[1].resets_at = datetime.now(UTC) - timedelta(hours=1)  # past
        gate._accounts[2].resets_at = datetime.now(UTC) + timedelta(hours=2)  # future

        any_uncapped = await gate._refresh_capped_accounts()

        assert any_uncapped is True
        assert gate._accounts[0].capped is True
        assert gate._accounts[1].capped is False
        assert gate._accounts[2].capped is True

    @pytest.mark.asyncio
    async def test_gate_reopens_when_any_uncapped(self):
        gate = _make_gate(num_accounts=2)
        gate._accounts[0].capped = True
        gate._accounts[0].resets_at = datetime.now(UTC) + timedelta(hours=1)
        gate._accounts[1].capped = True
        gate._accounts[1].resets_at = datetime.now(UTC) - timedelta(hours=1)
        gate._accounts[1].pause_started_at = datetime.now(UTC)
        gate._open.clear()

        await gate._refresh_capped_accounts()

        assert gate._open.is_set()

    @pytest.mark.asyncio
    async def test_tracks_pause_duration_on_uncap(self):
        gate = _make_gate(num_accounts=1)
        acct = gate._accounts[0]
        acct.capped = True
        acct.resets_at = datetime.now(UTC) - timedelta(seconds=1)
        acct.pause_started_at = datetime.now(UTC) - timedelta(seconds=10)

        await gate._refresh_capped_accounts()

        assert gate._total_pause_secs > 9.0

    @pytest.mark.asyncio
    async def test_before_invoke_blocks_when_resets_at_in_future(self):
        """When all capped with resets_at in future, before_invoke should block."""
        gate = _make_gate(num_accounts=2)
        for acct in gate._accounts:
            acct.capped = True
            acct.resets_at = datetime.now(UTC) + timedelta(hours=1)

        with pytest.raises(asyncio.TimeoutError):
            await asyncio.wait_for(gate.before_invoke(), timeout=0.2)

    @pytest.mark.asyncio
    async def test_before_invoke_finds_reset_via_refresh(self):
        """All flagged capped, but B's reset time has passed — should pick B."""
        gate = _make_gate(num_accounts=3)
        for acct in gate._accounts:
            acct.capped = True
            acct.pause_started_at = datetime.now(UTC)
            acct.resets_at = datetime.now(UTC) + timedelta(hours=1)
        # B has already reset
        gate._accounts[1].resets_at = datetime.now(UTC) - timedelta(hours=1)

        token = await asyncio.wait_for(gate.before_invoke(), timeout=0.5)

        assert token == 'token-b'
        assert gate._accounts[1].capped is False


# --- Resume probe (timer-based) ---


class TestResumeProbe:
    @pytest.mark.asyncio
    async def test_sleeps_until_resets_at_then_uncaps(self):
        gate = _make_gate()
        acct = gate._accounts[0]
        acct.capped = True
        acct.pause_started_at = datetime.now(UTC)
        acct.resets_at = datetime.now(UTC) + timedelta(milliseconds=50)

        await asyncio.wait_for(
            gate._account_resume_probe_loop(acct), timeout=1.0,
        )

        assert acct.capped is False
        assert gate._open.is_set()

    @pytest.mark.asyncio
    async def test_reopens_global_gate(self):
        gate = _make_gate()
        acct = gate._accounts[0]
        acct.capped = True
        acct.pause_started_at = datetime.now(UTC)
        acct.resets_at = datetime.now(UTC) - timedelta(seconds=1)  # already past
        gate._open.clear()

        await gate._account_resume_probe_loop(acct)

        assert gate._open.is_set()

    @pytest.mark.asyncio
    async def test_noop_if_already_uncapped(self):
        """If _refresh_capped_accounts already uncapped the account, probe skips."""
        gate = _make_gate()
        acct = gate._accounts[0]
        acct.capped = False  # already uncapped
        acct.pause_started_at = None
        acct.resets_at = datetime.now(UTC) - timedelta(seconds=1)

        old_pause_secs = gate._total_pause_secs
        await gate._account_resume_probe_loop(acct)

        # No double-counting of pause duration
        assert gate._total_pause_secs == old_pause_secs

    @pytest.mark.asyncio
    async def test_defaults_to_1h_when_no_resets_at(self):
        """When resets_at is None, probe defaults to 1h — verify it doesn't crash."""
        gate = _make_gate()
        acct = gate._accounts[0]
        acct.capped = True
        acct.pause_started_at = datetime.now(UTC)
        acct.resets_at = None

        # Start probe, then cancel after a short delay (don't wait 1h)
        task = asyncio.create_task(gate._account_resume_probe_loop(acct))
        await asyncio.sleep(0.05)
        # Probe should still be sleeping (1h default)
        assert not task.done()
        assert acct.capped is True
        task.cancel()
        await asyncio.wait_for(task, timeout=1.0)

    @pytest.mark.asyncio
    async def test_probe_cancellation(self):
        """Probe should be cancellable while sleeping and leave account capped."""
        gate = _make_gate()
        acct = gate._accounts[0]
        acct.capped = True
        acct.resets_at = datetime.now(UTC) + timedelta(hours=1)

        task = asyncio.create_task(gate._account_resume_probe_loop(acct))
        await asyncio.sleep(0.05)
        task.cancel()
        await asyncio.wait_for(task, timeout=1.0)

        assert acct.capped is True

    @pytest.mark.asyncio
    async def test_tracks_pause_duration(self):
        gate = _make_gate()
        acct = gate._accounts[0]
        acct.capped = True
        acct.pause_started_at = datetime.now(UTC) - timedelta(seconds=10)
        acct.resets_at = datetime.now(UTC) - timedelta(seconds=1)

        await gate._account_resume_probe_loop(acct)

        assert gate._total_pause_secs > 9.0
        assert acct.pause_started_at is None


# --- Cap retry rotation ---


class TestCapRetryRotation:
    @pytest.mark.asyncio
    async def test_cap_retry_rotation_order(self):
        """Cap A → pick B → cap B → pick C."""
        gate = _make_gate(num_accounts=3)

        token1 = await gate.before_invoke()
        assert token1 == 'token-a'

        gate.detect_cap_hit("You've hit your limit · resets 5pm",
                            '', 'claude', oauth_token='token-a')
        assert gate._accounts[0].capped is True
        assert gate._open.is_set()

        token2 = await gate.before_invoke()
        assert token2 == 'token-b'

        gate.detect_cap_hit("You've hit your limit · resets 6am",
                            '', 'claude', oauth_token='token-b')
        assert gate._accounts[1].capped is True
        assert gate._open.is_set()

        token3 = await gate.before_invoke()
        assert token3 == 'token-c'

    @pytest.mark.asyncio
    async def test_all_capped_then_one_resets_via_timer(self):
        """All three capped, then B's reset time arrives."""
        gate = _make_gate(num_accounts=3)
        for acct in gate._accounts:
            acct.capped = True
            acct.resets_at = datetime.now(UTC) + timedelta(hours=1)
        gate._open.clear()

        async def uncap_b_after_delay():
            await asyncio.sleep(0.05)
            gate._accounts[1].capped = False
            gate._accounts[1].pause_started_at = None
            gate._open.set()

        asyncio.create_task(uncap_b_after_delay())
        token = await asyncio.wait_for(gate.before_invoke(), timeout=0.5)

        assert token == 'token-b'


# --- Single account via unified path ---


class TestSingleAccountUnifiedPath:
    """Verify that 1 account in the list works correctly through the
    unified (formerly multi-account) code path.
    """

    @pytest.mark.asyncio
    async def test_returns_single_token(self):
        gate = _make_gate(num_accounts=1)
        token = await gate.before_invoke()
        assert token == 'token-a'

    @pytest.mark.asyncio
    async def test_blocks_when_single_account_capped(self):
        gate = _make_gate(num_accounts=1)
        gate._accounts[0].capped = True
        gate._accounts[0].resets_at = datetime.now(UTC) + timedelta(hours=1)

        with pytest.raises(asyncio.TimeoutError):
            await asyncio.wait_for(gate.before_invoke(), timeout=0.1)

    @pytest.mark.asyncio
    async def test_cap_hit_marks_single_account(self):
        gate = _make_gate(num_accounts=1)
        gate.detect_cap_hit("You've hit your limit", '', 'claude', 'token-a')
        assert gate._accounts[0].capped is True
        assert gate.is_paused
        assert not gate._open.is_set()

    @pytest.mark.asyncio
    async def test_probe_uncaps_single_account(self):
        gate = _make_gate(num_accounts=1)
        acct = gate._accounts[0]
        acct.capped = True
        acct.pause_started_at = datetime.now(UTC)
        acct.resets_at = datetime.now(UTC) + timedelta(milliseconds=50)

        await asyncio.wait_for(
            gate._account_resume_probe_loop(acct), timeout=1.0,
        )

        assert acct.capped is False
        assert gate._open.is_set()

    @pytest.mark.asyncio
    async def test_startup_check_is_noop(self):
        gate = _make_gate(num_accounts=1)
        await gate.check_at_startup()
        assert gate._accounts[0].capped is False
        assert not gate.is_paused


# --- Refresh vs probe interaction ---


class TestRefreshVsProbeInteraction:
    @pytest.mark.asyncio
    async def test_refresh_detects_reset_even_if_probe_sleeping(self):
        """Refresh should uncap an account whose reset time has passed,
        even though its probe task is still sleeping."""
        gate = _make_gate(num_accounts=3)
        for acct in gate._accounts:
            acct.capped = True
            acct.pause_started_at = datetime.now(UTC)
            acct.resets_at = datetime.now(UTC) + timedelta(hours=1)
        # B has already reset
        gate._accounts[1].resets_at = datetime.now(UTC) - timedelta(hours=1)

        refreshed = await gate._refresh_capped_accounts()

        assert refreshed is True
        assert gate._accounts[0].capped is True
        assert gate._accounts[1].capped is False
        assert gate._accounts[2].capped is True

    @pytest.mark.asyncio
    async def test_gate_reopens_when_refresh_finds_uncapped(self):
        gate = _make_gate(num_accounts=2)
        gate._accounts[0].capped = True
        gate._accounts[0].resets_at = datetime.now(UTC) + timedelta(hours=1)
        gate._accounts[1].capped = True
        gate._accounts[1].pause_started_at = datetime.now(UTC)
        gate._accounts[1].resets_at = datetime.now(UTC) - timedelta(hours=1)
        gate._open.clear()

        await gate._refresh_capped_accounts()

        assert gate._open.is_set()


# --- Probe interval / backoff ---


class TestProbeInterval:
    @pytest.mark.asyncio
    async def test_probe_uncaps_after_interval_not_resets_at(self):
        """Probe should uncap after probe_interval_secs, not wait for resets_at."""
        gate = _make_gate(probe_interval_secs=1, max_probe_interval_secs=10)
        acct = gate._accounts[0]
        acct.capped = True
        acct.pause_started_at = datetime.now(UTC)
        # resets_at is far in the future
        acct.resets_at = datetime.now(UTC) + timedelta(hours=5)

        await asyncio.wait_for(
            gate._account_resume_probe_loop(acct), timeout=3.0,
        )

        # Should have uncapped optimistically before resets_at
        assert acct.capped is False
        assert gate._open.is_set()
        # probe_count incremented (optimistic uncap, not past reset)
        assert acct.probe_count == 1

    @pytest.mark.asyncio
    async def test_probe_backoff(self):
        """Higher probe_count should mean longer sleep via exponential backoff."""
        gate = _make_gate(probe_interval_secs=1, max_probe_interval_secs=100)
        acct = gate._accounts[0]
        acct.capped = True
        acct.pause_started_at = datetime.now(UTC)
        acct.resets_at = datetime.now(UTC) + timedelta(hours=5)
        acct.probe_count = 3  # 2^3 = 8 second interval

        # Should NOT complete in 0.5 seconds (interval is 8s)
        task = asyncio.create_task(gate._account_resume_probe_loop(acct))
        await asyncio.sleep(0.5)
        assert not task.done()
        assert acct.capped is True
        task.cancel()
        await asyncio.wait_for(task, timeout=1.0)

    @pytest.mark.asyncio
    async def test_probe_resets_count_after_resets_at(self):
        """probe_count should reset to 0 when past resets_at."""
        gate = _make_gate(probe_interval_secs=1, max_probe_interval_secs=10)
        acct = gate._accounts[0]
        acct.capped = True
        acct.pause_started_at = datetime.now(UTC)
        acct.resets_at = datetime.now(UTC) - timedelta(seconds=1)  # already past
        acct.probe_count = 5

        await gate._account_resume_probe_loop(acct)

        assert acct.capped is False
        assert acct.probe_count == 0

    @pytest.mark.asyncio
    async def test_probe_respects_ceiling(self):
        """Backoff should be capped at max_probe_interval_secs."""
        gate = _make_gate(probe_interval_secs=1, max_probe_interval_secs=2)
        acct = gate._accounts[0]
        acct.capped = True
        acct.pause_started_at = datetime.now(UTC)
        acct.resets_at = datetime.now(UTC) + timedelta(hours=5)
        acct.probe_count = 10  # 2^10 = 1024, but capped at 2

        # Should complete within 3 seconds (ceiling = 2s, not 1024s)
        await asyncio.wait_for(
            gate._account_resume_probe_loop(acct), timeout=4.0,
        )
        assert acct.capped is False

    @pytest.mark.asyncio
    async def test_probe_sleeps_remaining_when_close_to_reset(self):
        """When resets_at is closer than probe interval, sleep only until resets_at."""
        gate = _make_gate(probe_interval_secs=100, max_probe_interval_secs=1000)
        acct = gate._accounts[0]
        acct.capped = True
        acct.pause_started_at = datetime.now(UTC)
        # 50ms until reset — much less than 100s probe interval
        acct.resets_at = datetime.now(UTC) + timedelta(milliseconds=50)

        await asyncio.wait_for(
            gate._account_resume_probe_loop(acct), timeout=1.0,
        )

        assert acct.capped is False
        assert acct.probe_count == 0  # past reset → count reset


# --- Parse resets_at with date ---


class TestParseResetsAtWithDate:
    def test_date_time_timezone(self):
        """Parse 'resets Mar 30, 6pm (Europe/London)'."""
        dt = _parse_resets_at('resets Mar 30, 6pm (Europe/London)')
        assert dt is not None
        # Should be a real datetime, not the 1h fallback
        fallback = datetime.now(UTC) + timedelta(hours=1)
        assert abs((dt - fallback).total_seconds()) > 60

    def test_date_time_with_comma(self):
        dt = _parse_resets_at('resets Mar 30, 6pm (UTC)')
        assert dt is not None
        assert dt.month == 3 or dt.month == 3  # March (possibly next year)
        assert dt.hour == 18
        assert dt.minute == 0

    def test_date_time_no_comma(self):
        dt = _parse_resets_at('resets Mar 30 6pm (UTC)')
        assert dt is not None
        assert dt.hour == 18

    def test_date_time_with_minutes(self):
        dt = _parse_resets_at('resets Mar 31, 2:30pm (Europe/London)')
        assert dt is not None

    def test_embedded_in_cap_message(self):
        text = "You've hit your limit · resets Mar 30, 6pm (Europe/London)"
        dt = _parse_resets_at(text)
        assert dt is not None
        fallback = datetime.now(UTC) + timedelta(hours=1)
        assert abs((dt - fallback).total_seconds()) > 60


# --- CostStore init / properties ---


class TestCostStoreInit:
    def test_init_accepts_cost_store_param(self):
        """UsageGate.__init__ must accept an optional cost_store keyword arg."""
        mock_cs = AsyncMock()
        config = UsageCapConfig(wait_for_reset=False, accounts=[])
        gate = UsageGate(config, cost_store=mock_cs)
        assert gate._cost_store is mock_cs

    def test_init_cost_store_defaults_to_none(self):
        """cost_store defaults to None when not supplied."""
        config = UsageCapConfig(wait_for_reset=False, accounts=[])
        gate = UsageGate(config)
        assert gate._cost_store is None

    def test_project_id_defaults_to_none(self):
        gate = _make_gate()
        assert gate.project_id is None

    def test_run_id_defaults_to_none(self):
        gate = _make_gate()
        assert gate.run_id is None

    def test_project_id_setter(self):
        gate = _make_gate()
        gate.project_id = 'my-project'
        assert gate.project_id == 'my-project'

    def test_run_id_setter(self):
        gate = _make_gate()
        gate.run_id = 'run-abc-123'
        assert gate.run_id == 'run-abc-123'

    def test_last_account_name_defaults_to_none(self):
        """_last_account_name is initialised to None."""
        gate = _make_gate()
        assert gate._last_account_name is None


# --- CostStore cap_hit event ---


class TestCostStoreCapHitEvent:
    @pytest.mark.asyncio
    async def test_cap_hit_calls_save_account_event(self):
        """detect_cap_hit fires save_account_event('cap_hit') when cost_store is set."""
        mock_cs = AsyncMock()
        gate = _make_gate(num_accounts=1)
        gate._cost_store = mock_cs
        gate._project_id = 'proj-1'
        gate._run_id = 'run-42'

        gate.detect_cap_hit(
            "You've hit your usage limit resets in 3h", '', 'claude', 'token-a',
        )
        # Drain the fire-and-forget task
        await asyncio.sleep(0)

        mock_cs.save_account_event.assert_called_once()
        call_kwargs = mock_cs.save_account_event.call_args
        assert call_kwargs.kwargs.get('account_name', call_kwargs.args[0] if call_kwargs.args else None) == 'max-a' or \
               call_kwargs.args[0] == 'max-a'
        # event_type must be 'cap_hit'
        event_type = call_kwargs.kwargs.get('event_type') or call_kwargs.args[1]
        assert event_type == 'cap_hit'

    @pytest.mark.asyncio
    async def test_cap_hit_event_carries_project_and_run_ids(self):
        mock_cs = AsyncMock()
        gate = _make_gate(num_accounts=1)
        gate._cost_store = mock_cs
        gate._project_id = 'dark_factory'
        gate._run_id = 'run-99'

        gate.detect_cap_hit(
            "You've hit your usage limit resets in 1h", '', 'claude', 'token-a',
        )
        await asyncio.sleep(0)

        call = mock_cs.save_account_event.call_args
        # project_id and run_id must be passed
        all_args = list(call.args) + list(call.kwargs.values())
        assert 'dark_factory' in all_args
        assert 'run-99' in all_args

    @pytest.mark.asyncio
    async def test_cap_hit_event_details_contains_reason(self):
        """details field should be a JSON string containing the cap reason."""
        mock_cs = AsyncMock()
        gate = _make_gate(num_accounts=1)
        gate._cost_store = mock_cs

        gate.detect_cap_hit(
            "You've hit your usage limit", '', 'claude', 'token-a',
        )
        await asyncio.sleep(0)

        call = mock_cs.save_account_event.call_args
        # Find the details arg (should be a JSON string with 'reason' key)
        all_args = list(call.args) + list(call.kwargs.values())
        details_str = next((a for a in all_args if isinstance(a, str) and 'reason' in a), None)
        assert details_str is not None
        import json as _json
        details = _json.loads(details_str)
        assert 'reason' in details

    @pytest.mark.asyncio
    async def test_no_cap_hit_event_without_cost_store(self):
        """No event emitted and no error raised when cost_store is None."""
        gate = _make_gate(num_accounts=1)
        assert gate._cost_store is None
        # Should complete without error
        gate.detect_cap_hit("You've hit your usage limit", '', 'claude', 'token-a')
        await asyncio.sleep(0)  # no tasks to drain, but no error either


# --- CostStore resumed event ---


class TestCostStoreResumedEvent:
    @pytest.mark.asyncio
    async def test_resumed_event_after_past_reset(self):
        """'resumed' event is emitted when probe uncaps after resets_at has passed."""
        mock_cs = AsyncMock()
        gate = _make_gate(num_accounts=1)
        gate._cost_store = mock_cs
        gate._project_id = 'proj-x'
        gate._run_id = 'run-r1'

        acct = gate._accounts[0]
        acct.capped = True
        acct.pause_started_at = datetime.now(UTC)
        acct.resets_at = datetime.now(UTC) - timedelta(seconds=1)  # past

        await gate._account_resume_probe_loop(acct)

        mock_cs.save_account_event.assert_called_once()
        call = mock_cs.save_account_event.call_args
        event_type = call.kwargs.get('event_type') or call.args[1]
        assert event_type == 'resumed'
        account_name = call.kwargs.get('account_name') or call.args[0]
        assert account_name == 'max-a'

    @pytest.mark.asyncio
    async def test_resumed_event_optimistic_probe(self):
        """'resumed' event is emitted on optimistic probe (resets_at in future)."""
        mock_cs = AsyncMock()
        gate = _make_gate(probe_interval_secs=1, max_probe_interval_secs=10,
                          num_accounts=1)
        gate._cost_store = mock_cs

        acct = gate._accounts[0]
        acct.capped = True
        acct.pause_started_at = datetime.now(UTC)
        acct.resets_at = datetime.now(UTC) + timedelta(hours=5)  # future

        await asyncio.wait_for(
            gate._account_resume_probe_loop(acct), timeout=3.0,
        )

        mock_cs.save_account_event.assert_called_once()
        call = mock_cs.save_account_event.call_args
        event_type = call.kwargs.get('event_type') or call.args[1]
        assert event_type == 'resumed'

    @pytest.mark.asyncio
    async def test_resumed_event_details_has_label(self):
        """details JSON should include 'label' key describing the resume type."""
        mock_cs = AsyncMock()
        gate = _make_gate(num_accounts=1)
        gate._cost_store = mock_cs

        acct = gate._accounts[0]
        acct.capped = True
        acct.pause_started_at = datetime.now(UTC)
        acct.resets_at = datetime.now(UTC) - timedelta(seconds=1)

        await gate._account_resume_probe_loop(acct)

        call = mock_cs.save_account_event.call_args
        all_args = list(call.args) + list(call.kwargs.values())
        import json as _json
        details_str = next((a for a in all_args if isinstance(a, str) and 'label' in a), None)
        assert details_str is not None
        details = _json.loads(details_str)
        assert 'label' in details

    @pytest.mark.asyncio
    async def test_no_resumed_event_without_cost_store(self):
        """No error when cost_store is None and probe loop completes."""
        gate = _make_gate(num_accounts=1)
        assert gate._cost_store is None

        acct = gate._accounts[0]
        acct.capped = True
        acct.pause_started_at = datetime.now(UTC)
        acct.resets_at = datetime.now(UTC) - timedelta(seconds=1)

        # Should complete cleanly with no AttributeError
        await gate._account_resume_probe_loop(acct)
        assert acct.capped is False


# --- CostStore failover event ---


class TestCostStoreFailoverEvent:
    @pytest.mark.asyncio
    async def test_failover_event_emitted_on_account_switch(self):
        """When before_invoke switches accounts, a 'failover' event is emitted."""
        mock_cs = AsyncMock()
        gate = _make_gate(num_accounts=2)
        gate._cost_store = mock_cs

        # First call — establishes max-a as last_account_name, no failover
        token1 = await gate.before_invoke()
        assert token1 == 'token-a'
        assert gate._last_account_name == 'max-a'
        mock_cs.save_account_event.assert_not_called()

        # Cap max-a, second call should pick max-b and emit failover
        gate._accounts[0].capped = True
        token2 = await gate.before_invoke()
        assert token2 == 'token-b'
        assert gate._last_account_name == 'max-b'

        mock_cs.save_account_event.assert_called_once()
        call = mock_cs.save_account_event.call_args
        event_type = call.kwargs.get('event_type') or call.args[1]
        assert event_type == 'failover'

    @pytest.mark.asyncio
    async def test_no_failover_on_first_invoke(self):
        """First call to before_invoke sets last_account_name but emits no event."""
        mock_cs = AsyncMock()
        gate = _make_gate(num_accounts=2)
        gate._cost_store = mock_cs

        await gate.before_invoke()
        mock_cs.save_account_event.assert_not_called()

    @pytest.mark.asyncio
    async def test_no_failover_when_same_account(self):
        """No failover event if same account is returned on consecutive calls."""
        mock_cs = AsyncMock()
        gate = _make_gate(num_accounts=2)
        gate._cost_store = mock_cs

        await gate.before_invoke()
        mock_cs.save_account_event.assert_not_called()

        await gate.before_invoke()
        mock_cs.save_account_event.assert_not_called()

    @pytest.mark.asyncio
    async def test_failover_details_has_from_and_to(self):
        """details JSON should include 'from' and 'to' account names."""
        mock_cs = AsyncMock()
        gate = _make_gate(num_accounts=2)
        gate._cost_store = mock_cs

        await gate.before_invoke()
        gate._accounts[0].capped = True
        await gate.before_invoke()

        call = mock_cs.save_account_event.call_args
        all_args = list(call.args) + list(call.kwargs.values())
        import json as _json
        details_str = next((a for a in all_args if isinstance(a, str) and 'from' in a), None)
        assert details_str is not None
        details = _json.loads(details_str)
        assert details.get('from') == 'max-a'
        assert details.get('to') == 'max-b'

    @pytest.mark.asyncio
    async def test_no_failover_event_without_cost_store(self):
        """No error on failover when cost_store is None."""
        gate = _make_gate(num_accounts=2)
        assert gate._cost_store is None

        await gate.before_invoke()
        gate._accounts[0].capped = True
        token = await gate.before_invoke()
        assert token == 'token-b'  # failover happened silently
