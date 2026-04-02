"""Tests for invoke_with_cap_retry in orchestrator/agents/invoke.py."""

from __future__ import annotations

from unittest.mock import AsyncMock, MagicMock, patch

import pytest
from shared.cli_invoke import CAP_HIT_RESUME_PROMPT, AgentResult

from orchestrator.agents.invoke import invoke_with_cap_retry


def _make_result(
    success: bool = True,
    output: str = 'ok',
    cost_usd: float = 0.5,
    stderr: str = '',
    session_id: str = '',
) -> AgentResult:
    return AgentResult(
        success=success, output=output, cost_usd=cost_usd,
        stderr=stderr, session_id=session_id,
    )


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


@pytest.mark.asyncio
class TestCapHitResume:

    async def test_resume_on_cap_hit_claude_backend(self):
        """Claude backend cap hit with session_id → resume on retry."""
        gate = MagicMock()
        gate.before_invoke = AsyncMock(side_effect=['token-a', 'token-b'])
        gate.detect_cap_hit = MagicMock(side_effect=[True, False])
        gate.active_account_name = 'acct-b'
        gate.on_agent_complete = MagicMock()

        capped_result = _make_result(session_id='sess-abc')
        ok_result = _make_result()

        with (
            patch(
                'orchestrator.agents.invoke.invoke_agent',
                new_callable=AsyncMock,
                side_effect=[capped_result, ok_result],
            ) as mock_invoke,
            patch('orchestrator.agents.invoke.asyncio') as mock_asyncio,
        ):
            mock_asyncio.sleep = AsyncMock()
            await invoke_with_cap_retry(
                gate, 'test-label',
                prompt='hi', system_prompt='sys', cwd='/tmp', backend='claude',
            )

            second_call = mock_invoke.call_args_list[1]
            assert second_call.kwargs.get('resume_session_id') == 'sess-abc'
            assert second_call.kwargs.get('prompt') == CAP_HIT_RESUME_PROMPT

    async def test_no_resume_on_cap_hit_codex_backend(self):
        """Codex backend cap hit → no resume_session_id (not supported)."""
        gate = MagicMock()
        gate.before_invoke = AsyncMock(side_effect=['token-a', 'token-b'])
        gate.detect_cap_hit = MagicMock(side_effect=[True, False])
        gate.active_account_name = 'acct-b'
        gate.on_agent_complete = MagicMock()

        capped_result = _make_result(session_id='sess-abc')
        ok_result = _make_result()

        with (
            patch(
                'orchestrator.agents.invoke.invoke_agent',
                new_callable=AsyncMock,
                side_effect=[capped_result, ok_result],
            ) as mock_invoke,
            patch('orchestrator.agents.invoke.asyncio') as mock_asyncio,
        ):
            mock_asyncio.sleep = AsyncMock()
            await invoke_with_cap_retry(
                gate, 'test-label',
                prompt='hi', system_prompt='sys', cwd='/tmp', backend='codex',
            )

            second_call = mock_invoke.call_args_list[1]
            assert 'resume_session_id' not in second_call.kwargs
            assert second_call.kwargs.get('prompt') == 'hi'

    async def test_resume_failure_falls_back_to_fresh(self):
        """Resume returns success=False → retry with original prompt."""
        gate = MagicMock()
        gate.before_invoke = AsyncMock(side_effect=['token-a', 'token-b', 'token-c'])
        gate.detect_cap_hit = MagicMock(side_effect=[True, False, False])
        gate.active_account_name = 'acct-b'
        gate.on_agent_complete = MagicMock()

        capped_result = _make_result(session_id='sess-abc')
        failed_resume = _make_result(success=False)
        ok_result = _make_result()

        with (
            patch(
                'orchestrator.agents.invoke.invoke_agent',
                new_callable=AsyncMock,
                side_effect=[capped_result, failed_resume, ok_result],
            ) as mock_invoke,
            patch('orchestrator.agents.invoke.asyncio') as mock_asyncio,
        ):
            mock_asyncio.sleep = AsyncMock()
            got = await invoke_with_cap_retry(
                gate, 'test-label',
                prompt='original', system_prompt='sys', cwd='/tmp', backend='claude',
            )

            assert mock_invoke.call_count == 3
            third_call = mock_invoke.call_args_list[2]
            assert 'resume_session_id' not in third_call.kwargs
            assert third_call.kwargs.get('prompt') == 'original'
            assert got.success is True
