"""Tests for cli_invoke cap-hit retry backoff."""

from __future__ import annotations

from unittest.mock import AsyncMock, MagicMock, patch

import pytest

from shared.cli_invoke import _CAP_HIT_COOLDOWN_SECS, AgentResult, invoke_with_cap_retry


def _make_result(**overrides) -> AgentResult:
    defaults = dict(success=True, output='ok', cost_usd=0.5, stderr='')
    defaults.update(overrides)
    return AgentResult(**defaults)


class TestAgentResultAccountNameField:

    def test_account_name_field_defaults_empty(self):
        """AgentResult has account_name field that defaults to empty string."""
        result = AgentResult(success=True, output='ok')
        assert result.account_name == ''


@pytest.mark.asyncio
class TestAccountNameThreading:

    async def test_account_name_set_from_usage_gate(self):
        """account_name is stamped from usage_gate.active_account_name on success."""
        gate = MagicMock()
        gate.before_invoke = AsyncMock(return_value='token-a')
        gate.detect_cap_hit = MagicMock(return_value=False)
        gate.active_account_name = 'acct-a'
        gate.on_agent_complete = MagicMock()

        result = _make_result()

        with patch(
            'shared.cli_invoke.invoke_claude_agent',
            new_callable=AsyncMock,
            return_value=result,
        ):
            got = await invoke_with_cap_retry(gate, 'test-label', prompt='hi')

        assert got.account_name == 'acct-a'

    async def test_account_name_none_coerced_to_empty(self):
        """When active_account_name is None, result.account_name is ''."""
        gate = MagicMock()
        gate.before_invoke = AsyncMock(return_value='token-a')
        gate.detect_cap_hit = MagicMock(return_value=False)
        gate.active_account_name = None
        gate.on_agent_complete = MagicMock()

        result = _make_result()

        with patch(
            'shared.cli_invoke.invoke_claude_agent',
            new_callable=AsyncMock,
            return_value=result,
        ):
            got = await invoke_with_cap_retry(gate, 'test-label', prompt='hi')

        assert got.account_name == ''

    async def test_account_name_reflects_failover_account(self):
        """After cap hit + failover, account_name reflects the retry account."""
        from unittest.mock import PropertyMock

        gate = MagicMock()
        gate.before_invoke = AsyncMock(side_effect=['token-a', 'token-b'])
        gate.detect_cap_hit = MagicMock(side_effect=[True, False])
        # active_account_name is read 3 times:
        #   loop 1 capture, loop 1 logging (cap hit), loop 2 capture
        type(gate).active_account_name = PropertyMock(
            side_effect=['acct-a', 'acct-b', 'acct-b'],
        )
        gate.on_agent_complete = MagicMock()

        result = _make_result()

        with (
            patch(
                'shared.cli_invoke.invoke_claude_agent',
                new_callable=AsyncMock,
                return_value=result,
            ),
            patch('shared.cli_invoke.asyncio') as mock_asyncio,
        ):
            mock_asyncio.sleep = AsyncMock()
            got = await invoke_with_cap_retry(gate, 'test-label', prompt='hi')

        assert got.account_name == 'acct-b'


@pytest.mark.asyncio
class TestCapHitBackoff:

    async def test_sleeps_before_retry_on_cap_hit(self):
        """invoke_with_cap_retry sleeps _CAP_HIT_COOLDOWN_SECS on cap hit."""
        gate = MagicMock()
        gate.before_invoke = AsyncMock(side_effect=['token-a', 'token-b'])
        gate.detect_cap_hit = MagicMock(side_effect=[True, False])
        gate.active_account_name = 'acct-b'
        gate.on_agent_complete = MagicMock()

        result = _make_result()

        with (
            patch(
                'shared.cli_invoke.invoke_claude_agent',
                new_callable=AsyncMock,
                return_value=result,
            ) as mock_invoke,
            patch('shared.cli_invoke.asyncio') as mock_asyncio,
        ):
            mock_asyncio.sleep = AsyncMock()
            got = await invoke_with_cap_retry(gate, 'test-label', prompt='hi')

            mock_asyncio.sleep.assert_called_once_with(_CAP_HIT_COOLDOWN_SECS)
            assert mock_invoke.call_count == 2
            assert got.success is True

    async def test_no_sleep_when_no_cap_hit(self):
        """No sleep when invocation succeeds on first try."""
        gate = MagicMock()
        gate.before_invoke = AsyncMock(return_value='token-a')
        gate.detect_cap_hit = MagicMock(return_value=False)
        gate.on_agent_complete = MagicMock()

        result = _make_result()

        with (
            patch(
                'shared.cli_invoke.invoke_claude_agent',
                new_callable=AsyncMock,
                return_value=result,
            ),
            patch('shared.cli_invoke.asyncio') as mock_asyncio,
        ):
            mock_asyncio.sleep = AsyncMock()
            got = await invoke_with_cap_retry(gate, 'test-label', prompt='hi')

            mock_asyncio.sleep.assert_not_called()
            assert got.success is True

    async def test_multiple_cap_hits_sleep_each_time(self):
        """Each cap hit triggers a separate sleep."""
        gate = MagicMock()
        gate.before_invoke = AsyncMock(
            side_effect=['token-a', 'token-b', 'token-c'],
        )
        gate.detect_cap_hit = MagicMock(side_effect=[True, True, False])
        gate.active_account_name = 'next-acct'
        gate.on_agent_complete = MagicMock()

        result = _make_result()

        with (
            patch(
                'shared.cli_invoke.invoke_claude_agent',
                new_callable=AsyncMock,
                return_value=result,
            ),
            patch('shared.cli_invoke.asyncio') as mock_asyncio,
        ):
            mock_asyncio.sleep = AsyncMock()
            await invoke_with_cap_retry(gate, 'test-label', prompt='hi')

            assert mock_asyncio.sleep.call_count == 2
            for call in mock_asyncio.sleep.call_args_list:
                assert call.args == (_CAP_HIT_COOLDOWN_SECS,)
