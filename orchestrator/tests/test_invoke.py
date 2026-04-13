"""Tests for invoke_with_cap_retry in orchestrator/agents/invoke.py."""

from __future__ import annotations

import asyncio
import json
from unittest.mock import AsyncMock, MagicMock, patch

import pytest
from shared.cli_invoke import CAP_HIT_RESUME_PROMPT, AgentResult

from orchestrator.agents.invoke import (
    _invoke_claude_with_sandbox,
    _invoke_codex,
    _invoke_gemini,
    _parse_codex_output,
    _parse_gemini_output,
    _run_subprocess_local,
    _SubprocessResult,
    invoke_with_cap_retry,
)


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


# ── _run_subprocess_local timed_out, _parse_codex_output, _parse_gemini_output ─


@pytest.mark.asyncio
class TestRunSubprocessLocalTimedOut:

    async def test_run_subprocess_local_sets_timed_out_on_timeout(self, tmp_path):
        """_run_subprocess_local sets timed_out=True when TimeoutError fires."""
        proc = MagicMock()
        proc.communicate = AsyncMock(side_effect=TimeoutError)
        proc.kill = MagicMock()
        proc.wait = AsyncMock()
        proc.returncode = None

        async def fake_exec(*args, **kwargs):
            return proc

        with patch('orchestrator.agents.invoke.asyncio.create_subprocess_exec',
                   side_effect=fake_exec):
            result = await _run_subprocess_local(
                ['fake'], cwd=tmp_path, env={}, backend='codex', model='gpt-5.4',
                max_budget_usd=1.0, timeout_seconds=0.1,
            )

        assert result.timed_out is True
        assert 'Process killed after' in result.stderr and 'timeout' in result.stderr
        assert result.returncode == 1


class TestParseCodexOutputTimedOutDefault:
    """Parser does not propagate timed_out (callers handle it via replace())."""

    def test_timed_out_true_input_yields_false_on_empty_stdout(self):
        """_parse_codex_output returns timed_out=False regardless of input — empty stdout."""
        sub = _SubprocessResult(stdout='', stderr='timeout', returncode=1,
                                duration_ms=100, timed_out=True)
        agent = _parse_codex_output(sub, 'gpt-5.4')
        assert agent.timed_out is False

    def test_timed_out_true_input_yields_false_on_json_decode_error(self):
        """_parse_codex_output returns timed_out=False regardless of input — parse error."""
        sub = _SubprocessResult(stdout='not json at all', stderr='', returncode=1,
                                duration_ms=100, timed_out=True)
        agent = _parse_codex_output(sub, 'gpt-5.4')
        assert agent.timed_out is False

    def test_timed_out_true_input_yields_false_on_normal_parse(self):
        """_parse_codex_output returns timed_out=False regardless of input — valid JSONL."""
        jsonl = json.dumps({'type': 'thread.started', 'thread_id': 'tid-1'}) + '\n'
        jsonl += json.dumps({
            'type': 'item.completed',
            'item': {'type': 'agent_message', 'text': 'hello'},
        }) + '\n'
        jsonl += json.dumps({'type': 'turn.completed', 'usage': {'input_tokens': 10, 'output_tokens': 5}}) + '\n'
        sub = _SubprocessResult(stdout=jsonl, stderr='', returncode=0,
                                duration_ms=100, timed_out=True)
        agent = _parse_codex_output(sub, 'gpt-5.4')
        assert agent.timed_out is False

    def test_timed_out_false_input_yields_false(self):
        """_parse_codex_output returns timed_out=False when input is also False."""
        sub = _SubprocessResult(stdout='', stderr='err', returncode=1,
                                duration_ms=100, timed_out=False)
        agent = _parse_codex_output(sub, 'gpt-5.4')
        assert agent.timed_out is False


class TestParseGeminiOutputTimedOutDefault:
    """Parser does not propagate timed_out (callers handle it via replace())."""

    def test_timed_out_true_input_yields_false_on_empty_stdout(self):
        """_parse_gemini_output returns timed_out=False regardless of input — empty stdout."""
        sub = _SubprocessResult(stdout='', stderr='timeout', returncode=1,
                                duration_ms=100, timed_out=True)
        agent = _parse_gemini_output(sub, 'gemini-3.1-pro-preview')
        assert agent.timed_out is False

    def test_timed_out_true_input_yields_false_on_json_decode_error(self):
        """_parse_gemini_output returns timed_out=False regardless of input — parse error."""
        sub = _SubprocessResult(stdout='not json', stderr='', returncode=1,
                                duration_ms=100, timed_out=True)
        agent = _parse_gemini_output(sub, 'gemini-3.1-pro-preview')
        assert agent.timed_out is False

    def test_timed_out_true_input_yields_false_on_normal_parse(self):
        """_parse_gemini_output returns timed_out=False regardless of input — valid JSON."""
        data = json.dumps({'response': 'hi', 'stats': {'input_tokens': 10, 'output_tokens': 5}})
        sub = _SubprocessResult(stdout=data, stderr='', returncode=0,
                                duration_ms=100, timed_out=True)
        agent = _parse_gemini_output(sub, 'gemini-3.1-pro-preview')
        assert agent.timed_out is False

    def test_timed_out_false_input_yields_false(self):
        """_parse_gemini_output returns timed_out=False when input is also False."""
        sub = _SubprocessResult(stdout='', stderr='err', returncode=1,
                                duration_ms=100, timed_out=False)
        agent = _parse_gemini_output(sub, 'gemini-3.1-pro-preview')
        assert agent.timed_out is False


class TestParseCodexGeminiOutputDocstringContract:
    """Parser docstrings must document the timed_out=False contract."""

    def test_parse_codex_docstring_contains_does_not_set_timed_out(self):
        """_parse_codex_output.__doc__ contains 'does not set timed_out' contract note."""
        assert _parse_codex_output.__doc__ is not None
        assert 'does not set timed_out' in _parse_codex_output.__doc__

    def test_parse_gemini_docstring_contains_does_not_set_timed_out(self):
        """_parse_gemini_output.__doc__ contains 'does not set timed_out' contract note."""
        assert _parse_gemini_output.__doc__ is not None
        assert 'does not set timed_out' in _parse_gemini_output.__doc__


# ── caller-level timed_out propagation (characterization tests) ───────────────


@pytest.mark.asyncio
class TestCodexCallerPropagatesTimedOut:
    """_invoke_codex must propagate timed_out=True from subprocess result."""

    async def test_codex_caller_propagates_timed_out(self, tmp_path):
        """_invoke_codex returns AgentResult with timed_out=True when subprocess timed out."""
        timed_result = _SubprocessResult(stdout='', stderr='timeout', returncode=1,
                                         duration_ms=100, timed_out=True)
        with patch('orchestrator.agents.invoke._run_subprocess_local',
                   new_callable=AsyncMock, return_value=timed_result):
            agent = await _invoke_codex(
                prompt='hello', system_prompt='sys', cwd=tmp_path,
                model='gpt-5.4', max_budget_usd=1.0,
                mcp_config=None, sandbox_modules=None, effort=None,
                timeout_seconds=30.0,
            )
        assert agent.timed_out is True

    async def test_codex_caller_propagates_timed_out_false(self, tmp_path):
        """_invoke_codex returns AgentResult with timed_out=False when subprocess did not time out."""
        not_timed_result = _SubprocessResult(stdout='', stderr='err', returncode=1,
                                             duration_ms=100, timed_out=False)
        with patch('orchestrator.agents.invoke._run_subprocess_local',
                   new_callable=AsyncMock, return_value=not_timed_result):
            agent = await _invoke_codex(
                prompt='hello', system_prompt='sys', cwd=tmp_path,
                model='gpt-5.4', max_budget_usd=1.0,
                mcp_config=None, sandbox_modules=None, effort=None,
                timeout_seconds=30.0,
            )
        assert agent.timed_out is False


@pytest.mark.asyncio
class TestGeminiCallerPropagatesTimedOut:
    """_invoke_gemini must propagate timed_out=True from subprocess result."""

    async def test_gemini_caller_propagates_timed_out(self, tmp_path):
        """_invoke_gemini returns AgentResult with timed_out=True when subprocess timed out."""
        timed_result = _SubprocessResult(stdout='', stderr='timeout', returncode=1,
                                         duration_ms=100, timed_out=True)
        with patch('orchestrator.agents.invoke._run_subprocess_local',
                   new_callable=AsyncMock, return_value=timed_result):
            agent = await _invoke_gemini(
                prompt='hello', system_prompt='sys', cwd=tmp_path,
                model='gemini-3.1-pro-preview', max_budget_usd=1.0,
                mcp_config=None, sandbox_modules=None, effort=None,
                timeout_seconds=30.0,
            )
        assert agent.timed_out is True


@pytest.mark.asyncio
class TestSandboxCallerPropagatesTimedOut:
    """_invoke_claude_with_sandbox must propagate timed_out=True from subprocess result."""

    async def test_sandbox_caller_propagates_timed_out(self, tmp_path):
        """_invoke_claude_with_sandbox returns timed_out=True when subprocess timed out."""
        timed_result = _SubprocessResult(stdout='', stderr='timeout', returncode=1,
                                         duration_ms=100, timed_out=True)
        with (
            patch('orchestrator.agents.invoke._run_subprocess',
                  new_callable=AsyncMock, return_value=timed_result),
            patch('orchestrator.agents.sandbox.is_bwrap_available', return_value=True),
            patch('orchestrator.agents.sandbox.build_bwrap_command', side_effect=lambda cmd, *a, **k: cmd),
        ):
            agent = await _invoke_claude_with_sandbox(
                prompt='hello', system_prompt='sys', cwd=tmp_path,
                model='claude-sonnet-4-5', max_turns=5, max_budget_usd=1.0,
                allowed_tools=None, disallowed_tools=None,
                mcp_config=None, output_schema=None,
                permission_mode='bypassPermissions',
                sandbox_modules=['src'],
                effort=None, timeout_seconds=30.0,
            )
        assert agent.timed_out is True


# ===================================================================
# TestReleaseProbeSlotOnException (orchestrator invoke_with_cap_retry)
# ===================================================================


@pytest.mark.asyncio
class TestReleaseProbeSlotOnException:
    """invoke_with_cap_retry (orchestrator) calls release_probe_slot() when invoke raises."""

    async def test_release_probe_slot_called_on_runtime_error(self):
        """release_probe_slot is called with oauth_token when invoke_agent raises."""
        gate = MagicMock()
        gate.before_invoke = AsyncMock(return_value='tok-a')
        gate.detect_cap_hit = MagicMock(return_value=False)
        gate.active_account_name = 'acct-a'
        gate.on_agent_complete = MagicMock()
        gate.confirm_account_ok = MagicMock()
        gate.release_probe_slot = MagicMock()

        with (
            patch(
                'orchestrator.agents.invoke.invoke_agent',
                new_callable=AsyncMock,
                side_effect=RuntimeError('subprocess failed'),
            ),
            pytest.raises(RuntimeError, match='subprocess failed'),
        ):
            await invoke_with_cap_retry(gate, 'lbl', prompt='hi',
                                        system_prompt='sys', cwd='/tmp')

        gate.release_probe_slot.assert_called_once_with('tok-a')

    async def test_runtime_error_propagates(self):
        """RuntimeError raised by invoke_agent propagates with its message intact."""
        gate = MagicMock()
        gate.before_invoke = AsyncMock(return_value='tok-a')
        gate.active_account_name = 'acct-a'
        gate.release_probe_slot = MagicMock()

        with (
            patch(
                'orchestrator.agents.invoke.invoke_agent',
                new_callable=AsyncMock,
                side_effect=RuntimeError('crash'),
            ),
            pytest.raises(RuntimeError) as exc_info,
        ):
            await invoke_with_cap_retry(gate, 'lbl', prompt='hi',
                                        system_prompt='sys', cwd='/tmp')

        assert str(exc_info.value) == 'crash'  # error message preserved verbatim

    async def test_confirm_account_ok_not_called_when_invoke_raises(self):
        """confirm_account_ok is NOT called when invoke_agent raises."""
        gate = MagicMock()
        gate.before_invoke = AsyncMock(return_value='tok-a')
        gate.active_account_name = 'acct-a'
        gate.confirm_account_ok = MagicMock()
        gate.release_probe_slot = MagicMock()

        with (
            patch(
                'orchestrator.agents.invoke.invoke_agent',
                new_callable=AsyncMock,
                side_effect=RuntimeError('crash'),
            ),
            pytest.raises(RuntimeError),
        ):
            await invoke_with_cap_retry(gate, 'lbl', prompt='hi',
                                        system_prompt='sys', cwd='/tmp')

        gate.confirm_account_ok.assert_not_called()

    async def test_cancelled_error_release_probe_slot(self):
        """CancelledError (BaseException, not Exception) triggers release_probe_slot."""
        gate = MagicMock()
        gate.before_invoke = AsyncMock(return_value='tok-a')
        gate.active_account_name = 'acct-a'
        gate.release_probe_slot = MagicMock()

        with (
            patch(
                'orchestrator.agents.invoke.invoke_agent',
                new_callable=AsyncMock,
                side_effect=asyncio.CancelledError(),
            ),
            pytest.raises(asyncio.CancelledError),
        ):
            await invoke_with_cap_retry(gate, 'lbl', prompt='hi',
                                        system_prompt='sys', cwd='/tmp')

        gate.release_probe_slot.assert_called_once_with('tok-a')
