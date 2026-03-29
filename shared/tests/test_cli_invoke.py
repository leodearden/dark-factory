"""Tests for cli_invoke cap-hit retry backoff."""

from __future__ import annotations

import json
from unittest.mock import AsyncMock, MagicMock, patch

import pytest

from shared.cli_invoke import (
    _CAP_HIT_COOLDOWN_SECS,
    AgentResult,
    _parse_claude_output,
    _SubprocessResult,
    invoke_with_cap_retry,
)


def _make_result(**overrides) -> AgentResult:
    defaults = dict(success=True, output='ok', cost_usd=0.5, stderr='')
    defaults.update(overrides)
    return AgentResult(**defaults)


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


class TestAgentResultTokenFields:
    """AgentResult exposes four optional token-count fields."""

    def test_token_fields_default_to_none(self):
        """All four token fields default to None when not supplied."""
        result = AgentResult(success=True, output='ok')
        assert result.input_tokens is None
        assert result.output_tokens is None
        assert result.cache_read_tokens is None
        assert result.cache_create_tokens is None

    def test_token_fields_accept_values(self):
        """AgentResult accepts explicit values for all four token fields."""
        result = AgentResult(
            success=True,
            output='ok',
            input_tokens=1500,
            output_tokens=800,
            cache_read_tokens=300,
            cache_create_tokens=50,
        )
        assert result.input_tokens == 1500
        assert result.output_tokens == 800
        assert result.cache_read_tokens == 300
        assert result.cache_create_tokens == 50

    def test_make_result_helper_backward_compat(self):
        """Existing _make_result helper still works without token fields."""
        result = _make_result()
        assert result.success is True
        assert result.output == 'ok'
        # Token fields should default to None
        assert result.input_tokens is None
        assert result.output_tokens is None
        assert result.cache_read_tokens is None
        assert result.cache_create_tokens is None


class TestParseClaudeTokens:
    """_parse_claude_output populates token fields from the usage block."""

    def _make_subprocess_result(self, data: dict, returncode: int = 0) -> _SubprocessResult:
        return _SubprocessResult(
            stdout=json.dumps(data),
            stderr='',
            returncode=returncode,
            duration_ms=100,
        )

    def test_parses_all_token_fields(self):
        """All four token fields are extracted from usage block."""
        data = {
            'subtype': 'success',
            'result': 'done',
            'cost_usd': 0.05,
            'duration_ms': 1000,
            'num_turns': 3,
            'session_id': 'sess-1',
            'usage': {
                'input_tokens': 1500,
                'output_tokens': 800,
                'cache_read_input_tokens': 300,
                'cache_creation_input_tokens': 50,
            },
        }
        result = _parse_claude_output(self._make_subprocess_result(data))
        assert result.input_tokens == 1500
        assert result.output_tokens == 800
        assert result.cache_read_tokens == 300
        assert result.cache_create_tokens == 50

    def test_no_usage_block_gives_none(self):
        """When no usage key is present, all token fields are None."""
        data = {
            'subtype': 'success',
            'result': 'done',
            'cost_usd': 0.05,
            'duration_ms': 1000,
            'num_turns': 3,
            'session_id': 'sess-1',
        }
        result = _parse_claude_output(self._make_subprocess_result(data))
        assert result.input_tokens is None
        assert result.output_tokens is None
        assert result.cache_read_tokens is None
        assert result.cache_create_tokens is None

    def test_partial_usage_block_gives_none_for_missing_keys(self):
        """Missing keys within usage block default to None."""
        data = {
            'subtype': 'success',
            'result': 'done',
            'usage': {
                'input_tokens': 500,
                'output_tokens': 200,
                # no cache keys
            },
        }
        result = _parse_claude_output(self._make_subprocess_result(data))
        assert result.input_tokens == 500
        assert result.output_tokens == 200
        assert result.cache_read_tokens is None
        assert result.cache_create_tokens is None
