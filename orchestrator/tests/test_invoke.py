"""Tests for invoke_with_cap_retry account_name stamping in orchestrator/agents/invoke.py."""

from __future__ import annotations

from unittest.mock import AsyncMock, MagicMock, patch

import pytest

from shared.cli_invoke import AgentResult

from orchestrator.agents.invoke import invoke_with_cap_retry


def _make_result(**overrides) -> AgentResult:
    defaults = dict(success=True, output='ok', cost_usd=0.5, stderr='')
    defaults.update(overrides)
    return AgentResult(**defaults)


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
            'orchestrator.agents.invoke.invoke_agent',
            new_callable=AsyncMock,
            return_value=result,
        ):
            got = await invoke_with_cap_retry(gate, 'test-label', prompt='hi',
                                              system_prompt='sys', cwd='/tmp')

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
            'orchestrator.agents.invoke.invoke_agent',
            new_callable=AsyncMock,
            return_value=result,
        ):
            got = await invoke_with_cap_retry(gate, 'test-label', prompt='hi',
                                              system_prompt='sys', cwd='/tmp')

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
                'orchestrator.agents.invoke.invoke_agent',
                new_callable=AsyncMock,
                return_value=result,
            ),
            patch('orchestrator.agents.invoke.asyncio') as mock_asyncio,
        ):
            mock_asyncio.sleep = AsyncMock()
            got = await invoke_with_cap_retry(gate, 'test-label', prompt='hi',
                                              system_prompt='sys', cwd='/tmp')

        assert got.account_name == 'acct-b'


@pytest.mark.asyncio
class TestAccountNameNoGate:

    async def test_account_name_empty_without_gate(self):
        """When usage_gate=None, result.account_name is ''."""
        result = _make_result()

        with patch(
            'orchestrator.agents.invoke.invoke_agent',
            new_callable=AsyncMock,
            return_value=result,
        ):
            got = await invoke_with_cap_retry(None, 'test-label', prompt='hi',
                                              system_prompt='sys', cwd='/tmp')

        assert got.account_name == ''
