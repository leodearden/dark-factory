"""Tests for usage cap detection, gate lifecycle, and reset time parsing."""

from __future__ import annotations

import asyncio
from datetime import UTC, datetime, timedelta
from unittest.mock import AsyncMock, patch

import pytest

from orchestrator.config import UsageCapConfig
from orchestrator.usage_gate import (
    SessionBudgetExhausted,
    UsageGate,
    _extract_cap_message,
    _parse_resets_at,
)


@pytest.fixture
def default_config():
    return UsageCapConfig(wait_for_reset=False)


@pytest.fixture
def gate(default_config):
    return UsageGate(default_config)


# --- Cap hit detection ---


class TestDetectCapHit:
    def test_detects_hit_your_limit(self, gate):
        stderr = "You've hit your usage limit for Claude. Your usage resets in 3h."
        assert gate.detect_cap_hit(stderr, '') is True
        # Gate should be pausing (async task scheduled)

    def test_detects_youve_used(self, gate):
        stderr = ''
        result = "You've used all available tokens. Usage resets in 45m."
        assert gate.detect_cap_hit(stderr, result) is True

    def test_detects_out_of_extra_usage(self, gate):
        stderr = "You're out of extra usage for this period."
        assert gate.detect_cap_hit(stderr, '') is True

    def test_detects_near_cap_close_to(self, gate):
        stderr = "You're close to your usage limit."
        assert gate.detect_cap_hit(stderr, '') is True

    def test_detects_near_cap_extra_usage(self, gate):
        result = "You're now using extra usage credits."
        assert gate.detect_cap_hit('', result) is True

    def test_no_false_positive_normal_output(self, gate):
        assert gate.detect_cap_hit('', 'Task completed successfully') is False
        assert gate.detect_cap_hit('some warning', 'all good') is False

    def test_case_insensitive(self, gate):
        assert gate.detect_cap_hit("YOU'VE HIT YOUR limit", '') is True


# --- Reset time parsing ---


class TestParseResetsAt:
    def test_relative_hours(self):
        dt = _parse_resets_at('resets in 3h')
        expected = datetime.now(UTC) + timedelta(hours=3)
        assert abs((dt - expected).total_seconds()) < 2

    def test_relative_minutes(self):
        dt = _parse_resets_at('resets in 45m')
        expected = datetime.now(UTC) + timedelta(minutes=45)
        assert abs((dt - expected).total_seconds()) < 2

    def test_relative_days(self):
        dt = _parse_resets_at('resets in 2d')
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
        expected = datetime.now(UTC) + timedelta(hours=1)
        assert abs((dt - expected).total_seconds()) < 2

    def test_embedded_in_longer_text(self):
        text = "You've hit your limit. Your usage resets in 5h. Please wait."
        dt = _parse_resets_at(text)
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
        config = UsageCapConfig(session_budget_usd=10.0)
        gate = UsageGate(config)
        gate.on_agent_complete(5.0)
        # Should not raise
        await gate.before_invoke()

    @pytest.mark.asyncio
    async def test_budget_exceeded_raises(self):
        config = UsageCapConfig(session_budget_usd=10.0)
        gate = UsageGate(config)
        gate.on_agent_complete(10.0)
        with pytest.raises(SessionBudgetExhausted) as exc_info:
            await gate.before_invoke()
        assert exc_info.value.cumulative_cost == 10.0

    @pytest.mark.asyncio
    async def test_no_budget_configured(self, gate):
        gate.on_agent_complete(999.0)
        # Should not raise when no budget is set
        await gate.before_invoke()


# --- Gate lifecycle ---


class TestGateLifecycle:
    @pytest.mark.asyncio
    async def test_starts_open(self, gate):
        assert not gate.is_paused
        # before_invoke should return immediately
        await asyncio.wait_for(gate.before_invoke(), timeout=0.1)

    @pytest.mark.asyncio
    async def test_pause_closes_gate(self, gate):
        await gate._pause('test reason', None)
        assert gate.is_paused
        assert gate.paused_reason == 'test reason'

    @pytest.mark.asyncio
    async def test_resume_opens_gate(self, gate):
        await gate._pause('test', None)
        assert gate.is_paused
        await gate._resume()
        assert not gate.is_paused
        assert gate.paused_reason == ''

    @pytest.mark.asyncio
    async def test_before_invoke_blocks_when_paused(self, gate):
        await gate._pause('test', None)

        # before_invoke should block
        with pytest.raises(asyncio.TimeoutError):
            await asyncio.wait_for(gate.before_invoke(), timeout=0.1)

    @pytest.mark.asyncio
    async def test_before_invoke_unblocks_on_resume(self, gate):
        await gate._pause('test', None)

        async def resume_after_delay():
            await asyncio.sleep(0.05)
            await gate._resume()

        asyncio.create_task(resume_after_delay())
        # Should unblock within 0.2s
        await asyncio.wait_for(gate.before_invoke(), timeout=0.5)
        assert not gate.is_paused

    @pytest.mark.asyncio
    async def test_pause_is_idempotent(self, gate):
        await gate._pause('first', None)
        await gate._pause('second', None)  # should not error
        assert gate.is_paused
        # Reason stays as first since gate was already paused
        assert gate.paused_reason == 'first'

    @pytest.mark.asyncio
    async def test_pause_tracks_duration(self, gate):
        assert gate.total_pause_secs == 0.0
        await gate._pause('test', None)
        await asyncio.sleep(0.05)
        assert gate.total_pause_secs > 0
        await gate._resume()
        assert gate.total_pause_secs > 0

    @pytest.mark.asyncio
    async def test_cumulative_cost(self, gate):
        assert gate.cumulative_cost == 0.0
        gate.on_agent_complete(1.5)
        gate.on_agent_complete(2.3)
        assert gate.cumulative_cost == pytest.approx(3.8)

    @pytest.mark.asyncio
    async def test_shutdown_cancels_resume_task(self, gate):
        gate._config = UsageCapConfig(wait_for_reset=True, probe_interval_secs=1)
        await gate._pause('test', datetime.now(UTC) + timedelta(hours=1))
        assert gate._resume_task is not None
        await gate.shutdown()
        assert gate._resume_task is None


# --- Startup check ---


class TestStartupCheck:
    @pytest.mark.asyncio
    async def test_pauses_when_over_threshold(self, gate):
        mock_usage = {
            'five_hour': {
                'utilization': 0.98,
                'resets_at': (datetime.now(UTC) + timedelta(hours=1)).isoformat(),
            }
        }
        with patch.object(gate, '_query_usage_api', new_callable=AsyncMock, return_value=mock_usage):
            gate._config = UsageCapConfig(wait_for_reset=False)
            await gate.check_at_startup()
            assert gate.is_paused
            assert 'five_hour' in gate.paused_reason

    @pytest.mark.asyncio
    async def test_stays_open_when_under_threshold(self, gate):
        mock_usage = {
            'five_hour': {'utilization': 0.5, 'resets_at': None}
        }
        with patch.object(gate, '_query_usage_api', new_callable=AsyncMock, return_value=mock_usage):
            await gate.check_at_startup()
            assert not gate.is_paused

    @pytest.mark.asyncio
    async def test_stays_open_when_api_unavailable(self, gate):
        with patch.object(gate, '_query_usage_api', new_callable=AsyncMock, return_value=None):
            await gate.check_at_startup()
            assert not gate.is_paused


# --- Resume probe loop ---


class TestResumeProbeLoop:
    @pytest.mark.asyncio
    async def test_resumes_when_utilization_drops(self, gate):
        gate._config = UsageCapConfig(
            wait_for_reset=True,
            probe_interval_secs=1,
        )

        # Start paused
        await gate._pause('test cap', None)
        assert gate.is_paused

        # Mock API to return low utilization
        mock_usage = {'five_hour': {'utilization': 0.3}}
        with patch.object(gate, '_query_usage_api', new_callable=AsyncMock, return_value=mock_usage):
            await gate._resume_probe_loop()

        assert not gate.is_paused

    @pytest.mark.asyncio
    async def test_resumes_optimistically_on_api_failure(self, gate):
        gate._config = UsageCapConfig(wait_for_reset=True, probe_interval_secs=1)
        await gate._pause('test cap', None)

        with patch.object(gate, '_query_usage_api', new_callable=AsyncMock, return_value=None):
            await gate._resume_probe_loop()

        assert not gate.is_paused
