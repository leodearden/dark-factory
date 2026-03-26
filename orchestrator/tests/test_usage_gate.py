"""Tests for usage cap detection, gate lifecycle, and reset time parsing."""

from __future__ import annotations

import asyncio
from datetime import UTC, datetime, timedelta
from unittest.mock import AsyncMock, patch

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
    probe_interval_secs: int = 300,
    session_budget_usd: float | None = None,
) -> UsageGate:
    """Create a UsageGate with mock accounts (tokens pre-injected)."""
    acct_cfgs = [
        AccountConfig(name='max-a', oauth_token_env='CLAUDE_OAUTH_A'),
        AccountConfig(name='max-b', oauth_token_env='CLAUDE_OAUTH_B'),
        AccountConfig(name='max-c', oauth_token_env='CLAUDE_OAUTH_C'),
    ][:num_accounts]
    config = UsageCapConfig(
        wait_for_reset=wait_for_reset,
        probe_interval_secs=probe_interval_secs,
        session_budget_usd=session_budget_usd,
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

        async def api_capped(token=None):
            return {'five_hour': {'utilization': 0.99}}

        with patch.object(gate, '_query_usage_api', side_effect=api_capped), \
                pytest.raises(asyncio.TimeoutError):
            await asyncio.wait_for(gate.before_invoke(), timeout=0.1)

    @pytest.mark.asyncio
    async def test_before_invoke_unblocks_on_uncap(self):
        gate = _make_gate(num_accounts=1)
        gate._accounts[0].capped = True

        async def uncap_after_delay():
            await asyncio.sleep(0.05)
            gate._accounts[0].capped = False
            gate._open.set()

        async def api_capped(token=None):
            return {'five_hour': {'utilization': 0.99}}

        with patch.object(gate, '_query_usage_api', side_effect=api_capped):
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
    async def test_pauses_when_over_threshold(self):
        gate = _make_gate(num_accounts=1)
        mock_usage = {
            'five_hour': {
                'utilization': 0.98,
                'resets_at': (datetime.now(UTC) + timedelta(hours=1)).isoformat(),
            }
        }
        with patch.object(gate, '_query_usage_api', new_callable=AsyncMock,
                          return_value=mock_usage):
            await gate.check_at_startup()
            assert gate.is_paused
            assert gate._accounts[0].capped is True

    @pytest.mark.asyncio
    async def test_stays_open_when_under_threshold(self):
        gate = _make_gate(num_accounts=1)
        mock_usage = {
            'five_hour': {'utilization': 0.5, 'resets_at': None}
        }
        with patch.object(gate, '_query_usage_api', new_callable=AsyncMock,
                          return_value=mock_usage):
            await gate.check_at_startup()
            assert not gate.is_paused

    @pytest.mark.asyncio
    async def test_stays_open_when_api_unavailable(self):
        gate = _make_gate(num_accounts=1)
        with patch.object(gate, '_query_usage_api', new_callable=AsyncMock,
                          return_value=None):
            await gate.check_at_startup()
            assert not gate.is_paused

    @pytest.mark.asyncio
    async def test_marks_capped_account_at_startup(self):
        gate = _make_gate(num_accounts=2)

        async def mock_query(token=None):
            if token == 'token-a':
                return {'five_hour': {'utilization': 0.99, 'resets_at': None}}
            return {'five_hour': {'utilization': 0.3}}

        with patch.object(gate, '_query_usage_api', side_effect=mock_query):
            await gate.check_at_startup()

        assert gate._accounts[0].capped is True
        assert gate._accounts[1].capped is False
        assert not gate.is_paused  # account B still available

    @pytest.mark.asyncio
    async def test_pauses_when_all_capped_at_startup(self):
        gate = _make_gate(num_accounts=2)

        async def mock_query(token=None):
            return {'five_hour': {'utilization': 0.99, 'resets_at': None}}

        with patch.object(gate, '_query_usage_api', side_effect=mock_query):
            await gate.check_at_startup()

        assert gate._accounts[0].capped is True
        assert gate._accounts[1].capped is True
        assert gate.is_paused


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
        gate._accounts[1].capped = True

        async def mock_query(token=None):
            return {'five_hour': {'utilization': 0.99}}

        async def uncap_after_delay():
            await asyncio.sleep(0.05)
            gate._accounts[1].capped = False
            gate._open.set()

        with patch.object(gate, '_query_usage_api', side_effect=mock_query):
            asyncio.create_task(uncap_after_delay())
            token = await asyncio.wait_for(gate.before_invoke(), timeout=0.5)
        assert token == 'token-b'

    @pytest.mark.asyncio
    async def test_refresh_uncaps_account_before_blocking(self):
        """When all accounts are flagged as capped but one has actually reset,
        before_invoke should detect the reset via a fresh API check."""
        gate = _make_gate()
        gate._accounts[0].capped = True
        gate._accounts[0].pause_started_at = datetime.now(UTC)
        gate._accounts[1].capped = True

        async def mock_query(token=None):
            if token == 'token-a':
                return {'five_hour': {'utilization': 0.3}}  # A has reset
            return {'five_hour': {'utilization': 0.99}}     # B still capped

        with patch.object(gate, '_query_usage_api', side_effect=mock_query):
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


# --- Refresh on API failure ---


class TestRefreshOnApiFailure:
    @pytest.mark.asyncio
    async def test_api_failure_keeps_accounts_capped_without_resets_at(self):
        """When resets_at is None, API failure should keep accounts capped."""
        gate = _make_gate(num_accounts=3)
        for acct in gate._accounts:
            acct.capped = True

        async def api_down(token=None):
            return None

        with patch.object(gate, '_query_usage_api', side_effect=api_down):
            any_uncapped = await gate._refresh_capped_accounts()

        assert any_uncapped is False
        assert all(a.capped for a in gate._accounts)

    @pytest.mark.asyncio
    async def test_tentative_uncap_after_resets_at_passed(self):
        """After resets_at has passed and 2+ API failures, tentatively uncap."""
        gate = _make_gate(num_accounts=2)
        for acct in gate._accounts:
            acct.capped = True
            acct.resets_at = datetime.now(UTC) - timedelta(hours=1)  # past
            acct.api_fail_count = 1  # one prior failure

        async def api_down(token=None):
            return None

        with patch.object(gate, '_query_usage_api', side_effect=api_down):
            any_uncapped = await gate._refresh_capped_accounts()

        assert any_uncapped is True
        # Both should be tentatively uncapped (both past reset + >=2 failures)
        assert not gate._accounts[0].capped
        assert not gate._accounts[1].capped

    @pytest.mark.asyncio
    async def test_no_tentative_uncap_before_resets_at(self):
        """Before resets_at, API failure should not tentatively uncap."""
        gate = _make_gate(num_accounts=1)
        acct = gate._accounts[0]
        acct.capped = True
        acct.resets_at = datetime.now(UTC) + timedelta(hours=1)  # future
        acct.api_fail_count = 10  # many failures

        async def api_down(token=None):
            return None

        with patch.object(gate, '_query_usage_api', side_effect=api_down):
            any_uncapped = await gate._refresh_capped_accounts()

        assert any_uncapped is False
        assert acct.capped is True

    @pytest.mark.asyncio
    async def test_no_tentative_uncap_on_first_failure(self):
        """First API failure after resets_at should not uncap (needs >=2)."""
        gate = _make_gate(num_accounts=1)
        acct = gate._accounts[0]
        acct.capped = True
        acct.resets_at = datetime.now(UTC) - timedelta(hours=1)
        acct.api_fail_count = 0  # first failure

        async def api_down(token=None):
            return None

        with patch.object(gate, '_query_usage_api', side_effect=api_down):
            any_uncapped = await gate._refresh_capped_accounts()

        assert any_uncapped is False
        assert acct.capped is True
        assert acct.api_fail_count == 1

    @pytest.mark.asyncio
    async def test_api_failure_for_some_accounts(self):
        """API returns None for A (down) but valid data for B (reset)."""
        gate = _make_gate(num_accounts=3)
        for acct in gate._accounts:
            acct.capped = True
            acct.pause_started_at = datetime.now(UTC)

        async def mixed_api(token=None):
            if token == 'token-a':
                return None  # API failure
            if token == 'token-b':
                return {'five_hour': {'utilization': 0.3}}  # reset
            return {'five_hour': {'utilization': 0.99}}  # still capped

        with patch.object(gate, '_query_usage_api', side_effect=mixed_api):
            any_uncapped = await gate._refresh_capped_accounts()

        assert any_uncapped is True
        assert gate._accounts[0].capped is True   # A: API failed, stays capped
        assert gate._accounts[1].capped is False   # B: confirmed reset
        assert gate._accounts[2].capped is True    # C: still over threshold

    @pytest.mark.asyncio
    async def test_api_success_resets_fail_count(self):
        """Successful API call should reset api_fail_count."""
        gate = _make_gate(num_accounts=1)
        acct = gate._accounts[0]
        acct.capped = True
        acct.api_fail_count = 5

        async def api_capped(token=None):
            return {'five_hour': {'utilization': 0.99}}

        with patch.object(gate, '_query_usage_api', side_effect=api_capped):
            await gate._refresh_capped_accounts()

        assert acct.api_fail_count == 0

    @pytest.mark.asyncio
    async def test_before_invoke_blocks_when_all_api_fail(self):
        """When all capped and API returns 403 with resets_at in future,
        before_invoke should block."""
        gate = _make_gate(num_accounts=2)
        for acct in gate._accounts:
            acct.capped = True
            acct.resets_at = datetime.now(UTC) + timedelta(hours=1)

        async def api_down(token=None):
            return None

        with patch.object(gate, '_query_usage_api', side_effect=api_down), \
                pytest.raises(asyncio.TimeoutError):
            await asyncio.wait_for(gate.before_invoke(), timeout=0.2)

    @pytest.mark.asyncio
    async def test_before_invoke_finds_reset_via_refresh(self):
        """All flagged capped, but B has actually reset — should pick B."""
        gate = _make_gate(num_accounts=3)
        for acct in gate._accounts:
            acct.capped = True
            acct.pause_started_at = datetime.now(UTC)

        async def mock_query(token=None):
            if token == 'token-b':
                return {'five_hour': {'utilization': 0.3}}
            return {'five_hour': {'utilization': 0.99}}

        with patch.object(gate, '_query_usage_api', side_effect=mock_query):
            token = await asyncio.wait_for(gate.before_invoke(), timeout=0.5)

        assert token == 'token-b'
        assert gate._accounts[1].capped is False


# --- Account probe ---


class TestAccountProbe:
    @pytest.mark.asyncio
    async def test_resume_reopens_gate(self):
        gate = _make_gate()
        acct = gate._accounts[0]
        acct.capped = True
        acct.pause_started_at = datetime.now(UTC)

        async def mock_query(token=None):
            return {'five_hour': {'utilization': 0.3}}

        with patch.object(gate, '_query_usage_api', side_effect=mock_query):
            await gate._account_resume_probe_loop(acct)

        assert acct.capped is False
        assert gate._open.is_set()

    @pytest.mark.asyncio
    async def test_stays_capped_on_api_failure_then_succeeds(self):
        """API failure must NOT uncap — probe retries until API succeeds."""
        gate = _make_gate(probe_interval_secs=1)
        acct = gate._accounts[0]
        acct.capped = True
        acct.pause_started_at = datetime.now(UTC)
        acct.resets_at = None  # no reset time → skips initial sleep

        call_count = 0

        async def mock_query(token=None):
            nonlocal call_count
            call_count += 1
            if call_count < 3:
                return None  # API failure
            return {'five_hour': {'utilization': 0.3}}

        with patch.object(gate, '_query_usage_api', side_effect=mock_query):
            await gate._account_resume_probe_loop(acct)

        assert acct.capped is False
        assert call_count == 3

    @pytest.mark.asyncio
    async def test_tentative_uncap_after_resets_at_in_probe(self):
        """Probe tentatively uncaps after resets_at + 2 API failures."""
        gate = _make_gate(probe_interval_secs=1)
        acct = gate._accounts[0]
        acct.capped = True
        acct.pause_started_at = datetime.now(UTC)
        acct.resets_at = datetime.now(UTC) - timedelta(hours=1)  # past

        call_count = 0

        async def always_fail(token=None):
            nonlocal call_count
            call_count += 1
            return None

        with patch.object(gate, '_query_usage_api', side_effect=always_fail):
            await gate._account_resume_probe_loop(acct)

        # Should tentatively uncap after 2 failures (since past reset)
        assert acct.capped is False
        assert call_count == 2
        assert gate._open.is_set()

    @pytest.mark.asyncio
    async def test_no_tentative_uncap_without_resets_at_in_probe(self):
        """Probe should NOT tentatively uncap when resets_at is None."""
        gate = _make_gate(probe_interval_secs=1)
        acct = gate._accounts[0]
        acct.capped = True
        acct.resets_at = None  # no reset time known

        call_count = 0

        async def fail_then_succeed(token=None):
            nonlocal call_count
            call_count += 1
            if call_count <= 3:
                return None
            return {'five_hour': {'utilization': 0.3}}

        with patch.object(gate, '_query_usage_api', side_effect=fail_then_succeed):
            await gate._account_resume_probe_loop(acct)

        assert acct.capped is False
        assert call_count == 4  # 3 failures, then success

    @pytest.mark.asyncio
    async def test_probe_cancellation(self):
        """Probe should be cancellable while sleeping and leave account capped."""
        gate = _make_gate(probe_interval_secs=3600)
        acct = gate._accounts[0]
        acct.capped = True
        acct.resets_at = datetime.now(UTC) + timedelta(hours=1)

        task = asyncio.create_task(gate._account_resume_probe_loop(acct))
        await asyncio.sleep(0.05)
        task.cancel()
        await asyncio.wait_for(task, timeout=1.0)

        assert acct.capped is True

    @pytest.mark.asyncio
    async def test_api_success_resets_fail_count_in_probe(self):
        """Successful API call in probe should reset api_fail_count."""
        gate = _make_gate(probe_interval_secs=1)
        acct = gate._accounts[0]
        acct.capped = True
        acct.pause_started_at = datetime.now(UTC)
        acct.api_fail_count = 5

        async def api_low(token=None):
            return {'five_hour': {'utilization': 0.3}}

        with patch.object(gate, '_query_usage_api', side_effect=api_low):
            await gate._account_resume_probe_loop(acct)

        assert acct.api_fail_count == 0


# --- Reproduce original bug ---


class TestReproduceOptimisticUncapBug:
    """Verify the original bug is fixed: API 403 no longer causes a
    tight loop on the first account.
    """

    @pytest.mark.asyncio
    async def test_api_403_does_not_immediately_uncap(self):
        """All capped, API returns 403 — no resets_at so no tentative uncap."""
        gate = _make_gate(num_accounts=3)
        for acct in gate._accounts:
            acct.capped = True
            # No resets_at set → tentative uncap not eligible

        refresh_calls = []

        async def api_403(token=None):
            refresh_calls.append(token)
            return None

        with patch.object(gate, '_query_usage_api', side_effect=api_403):
            any_uncapped = await gate._refresh_capped_accounts()

        assert any_uncapped is False
        assert set(refresh_calls) == {'token-a', 'token-b', 'token-c'}
        assert all(a.capped for a in gate._accounts)

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
    async def test_all_capped_then_one_resets_via_probe(self):
        """All three capped, then B resets via its background probe."""
        gate = _make_gate(num_accounts=3)
        for acct in gate._accounts:
            acct.capped = True
        gate._open.clear()

        async def uncap_b_after_delay():
            await asyncio.sleep(0.05)
            gate._accounts[1].capped = False
            gate._accounts[1].pause_started_at = None
            gate._open.set()

        async def api_capped(token=None):
            return {'five_hour': {'utilization': 0.99}}

        with patch.object(gate, '_query_usage_api', side_effect=api_capped):
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

        async def api_capped(token=None):
            return {'five_hour': {'utilization': 0.99}}

        with patch.object(gate, '_query_usage_api', side_effect=api_capped), \
                pytest.raises(asyncio.TimeoutError):
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
        gate = _make_gate(num_accounts=1, probe_interval_secs=1)
        acct = gate._accounts[0]
        acct.capped = True
        acct.pause_started_at = datetime.now(UTC)

        async def api_low(token=None):
            return {'five_hour': {'utilization': 0.3}}

        with patch.object(gate, '_query_usage_api', side_effect=api_low):
            await gate._account_resume_probe_loop(acct)

        assert acct.capped is False
        assert gate._open.is_set()

    @pytest.mark.asyncio
    async def test_startup_check_single_account(self):
        gate = _make_gate(num_accounts=1)
        mock_usage = {'five_hour': {'utilization': 0.99, 'resets_at': None}}
        with patch.object(gate, '_query_usage_api', new_callable=AsyncMock,
                          return_value=mock_usage):
            await gate.check_at_startup()
        assert gate._accounts[0].capped is True
        assert gate.is_paused


# --- Refresh vs probe interaction ---


class TestRefreshVsProbeInteraction:
    @pytest.mark.asyncio
    async def test_refresh_detects_reset_even_if_probe_sleeping(self):
        gate = _make_gate(num_accounts=3)
        for acct in gate._accounts:
            acct.capped = True
            acct.pause_started_at = datetime.now(UTC)

        async def api_b_reset(token=None):
            if token == 'token-b':
                return {'five_hour': {'utilization': 0.2}}
            return {'five_hour': {'utilization': 0.99}}

        with patch.object(gate, '_query_usage_api', side_effect=api_b_reset):
            refreshed = await gate._refresh_capped_accounts()

        assert refreshed is True
        assert gate._accounts[0].capped is True
        assert gate._accounts[1].capped is False
        assert gate._accounts[2].capped is True

    @pytest.mark.asyncio
    async def test_gate_reopens_when_refresh_finds_uncapped(self):
        gate = _make_gate(num_accounts=2)
        gate._accounts[0].capped = True
        gate._accounts[1].capped = True
        gate._accounts[1].pause_started_at = datetime.now(UTC)
        gate._open.clear()

        async def api_b_reset(token=None):
            if token == 'token-b':
                return {'five_hour': {'utilization': 0.2}}
            return {'five_hour': {'utilization': 0.99}}

        with patch.object(gate, '_query_usage_api', side_effect=api_b_reset):
            await gate._refresh_capped_accounts()

        assert gate._open.is_set()
