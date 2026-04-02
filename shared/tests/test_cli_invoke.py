"""Tests for cli_invoke cap-hit retry backoff and CostStore instrumentation."""

from __future__ import annotations

import asyncio
import json
import logging
from pathlib import Path
from unittest.mock import AsyncMock, MagicMock, patch

import pytest

from shared.cli_invoke import (
    _CAP_HIT_COOLDOWN_SECS,
    _MAX_CAP_COOLDOWN_SECS,
    CAP_HIT_RESUME_PROMPT,
    AgentResult,
    _SubprocessResult,
    _to_token_count,
    invoke_claude_agent,
    invoke_with_cap_retry,
)


class TestToTokenCount:

    def test_zero_returns_none(self):
        """_to_token_count(0) returns None — zero means provider did not report."""
        assert _to_token_count(0) is None

    def test_none_returns_none(self):
        """_to_token_count(None) returns None — provider did not report."""
        assert _to_token_count(None) is None

    def test_positive_int_returned_unchanged(self):
        """_to_token_count(42) returns 42 — real token count passes through."""
        assert _to_token_count(42) == 42


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
        gate.account_count = 2
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
        gate.account_count = 2
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
        gate.account_count = 2
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
class TestAccountNameNoGate:

    async def test_account_name_empty_without_gate(self):
        """When usage_gate=None, result.account_name is ''."""
        result = _make_result()

        with patch(
            'shared.cli_invoke.invoke_claude_agent',
            new_callable=AsyncMock,
            return_value=result,
        ):
            got = await invoke_with_cap_retry(None, 'test-label', prompt='hi')

        assert got.account_name == ''


@pytest.mark.asyncio
class TestCapHitBackoff:

    async def test_sleeps_before_retry_on_cap_hit(self):
        """invoke_with_cap_retry sleeps _CAP_HIT_COOLDOWN_SECS on cap hit."""
        gate = MagicMock()
        gate.account_count = 2
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
        gate.account_count = 2
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

    async def test_multiple_cap_hits_within_first_cycle_use_base_cooldown(self):
        """Cap hits within the first cycle through accounts use base cooldown."""
        gate = MagicMock()
        gate.account_count = 3  # 3 accounts → first 3 hits are cycle 0
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

    async def test_backoff_escalates_after_full_account_cycle(self):
        """Cooldown doubles after cycling through all accounts once."""
        gate = MagicMock()
        gate.account_count = 2  # 2 accounts → cycle boundary at hit 2
        gate.before_invoke = AsyncMock(
            side_effect=['token-a', 'token-b', 'token-a', 'token-b'],
        )
        # 3 cap hits then success: hits 1-2 are cycle 0 (5s), hit 3 is cycle 1 (10s)
        gate.detect_cap_hit = MagicMock(side_effect=[True, True, True, False])
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

            assert mock_asyncio.sleep.call_count == 3
            sleeps = [call.args[0] for call in mock_asyncio.sleep.call_args_list]
            # Hits 1,2 → cycle 0 → 5s; hit 3 → cycle 1 → 10s
            assert sleeps == [5.0, 5.0, 10.0]

    async def test_backoff_caps_at_max(self):
        """Cooldown never exceeds _MAX_CAP_COOLDOWN_SECS."""
        gate = MagicMock()
        gate.account_count = 1  # 1 account → every hit starts a new cycle
        # Need enough hits to exceed max: 5 * 2^6 = 320 > 300
        num_hits = 8
        gate.before_invoke = AsyncMock(
            side_effect=['token-a'] * (num_hits + 1),
        )
        gate.detect_cap_hit = MagicMock(
            side_effect=[True] * num_hits + [False],
        )
        gate.active_account_name = 'acct-a'
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

            sleeps = [call.args[0] for call in mock_asyncio.sleep.call_args_list]
            assert all(s <= _MAX_CAP_COOLDOWN_SECS for s in sleeps)
            # Last few should be capped at 300
            assert sleeps[-1] == _MAX_CAP_COOLDOWN_SECS


@pytest.mark.asyncio
class TestInvokeWithCapRetryCostStore:

    async def test_save_invocation_called_on_success(self):
        """save_invocation is awaited with correct args after successful invoke."""
        gate = MagicMock()
        gate.account_count = 2
        gate.before_invoke = AsyncMock(return_value='token-a')
        gate.detect_cap_hit = MagicMock(return_value=False)
        gate.active_account_name = 'acct-a'
        gate.on_agent_complete = MagicMock()

        cost_store = MagicMock()
        cost_store.save_invocation = AsyncMock()

        result = _make_result(
            cost_usd=1.23, duration_ms=5000,
            input_tokens=100, output_tokens=200,
            cache_read_tokens=50, cache_create_tokens=10,
        )

        with patch(
            'shared.cli_invoke.invoke_claude_agent',
            new_callable=AsyncMock,
            return_value=result,
        ):
            await invoke_with_cap_retry(
                gate, 'test-label',
                cost_store=cost_store, run_id='run-1', task_id='t-1',
                project_id='proj-1', role='implementer',
                prompt='hi', model='sonnet',
            )

        cost_store.save_invocation.assert_awaited_once()
        call_kwargs = cost_store.save_invocation.call_args.kwargs
        assert call_kwargs['run_id'] == 'run-1'
        assert call_kwargs['task_id'] == 't-1'
        assert call_kwargs['project_id'] == 'proj-1'
        assert call_kwargs['role'] == 'implementer'
        assert call_kwargs['model'] == 'sonnet'
        assert call_kwargs['account_name'] == 'acct-a'
        assert call_kwargs['cost_usd'] == 1.23
        assert call_kwargs['input_tokens'] == 100
        assert call_kwargs['output_tokens'] == 200
        assert call_kwargs['cache_read_tokens'] == 50
        assert call_kwargs['cache_create_tokens'] == 10
        assert call_kwargs['duration_ms'] == 5000
        assert call_kwargs['capped'] is False
        assert 'started_at' in call_kwargs
        assert 'completed_at' in call_kwargs

    async def test_save_account_event_on_cap_hit(self):
        """save_account_event is awaited with cap_hit on cap detection."""
        gate = MagicMock()
        gate.account_count = 2
        gate.before_invoke = AsyncMock(side_effect=['token-a', 'token-b'])
        gate.detect_cap_hit = MagicMock(side_effect=[True, False])
        gate.active_account_name = 'acct-b'
        gate.on_agent_complete = MagicMock()

        cost_store = MagicMock()
        cost_store.save_invocation = AsyncMock()
        cost_store.save_account_event = AsyncMock()

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
            await invoke_with_cap_retry(
                gate, 'test-label',
                cost_store=cost_store, run_id='run-1', project_id='proj-1',
                prompt='hi',
            )

        cost_store.save_account_event.assert_awaited_once()
        call_kwargs = cost_store.save_account_event.call_args.kwargs
        assert call_kwargs['event_type'] == 'cap_hit'
        assert call_kwargs['details'] == 'test-label'
        assert 'created_at' in call_kwargs

    async def test_no_error_when_cost_store_none(self):
        """No CostStore-related errors when cost_store=None (default)."""
        gate = MagicMock()
        gate.account_count = 2
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

        assert got.success is True

    async def test_save_invocation_error_swallowed(self, caplog):
        """save_invocation failure is logged but does not break the return."""
        gate = MagicMock()
        gate.account_count = 2
        gate.before_invoke = AsyncMock(return_value='token-a')
        gate.detect_cap_hit = MagicMock(return_value=False)
        gate.active_account_name = 'acct-a'
        gate.on_agent_complete = MagicMock()

        cost_store = MagicMock()
        cost_store.save_invocation = AsyncMock(side_effect=RuntimeError('db error'))

        result = _make_result()

        with patch(
            'shared.cli_invoke.invoke_claude_agent',
            new_callable=AsyncMock,
            return_value=result,
        ), caplog.at_level(logging.WARNING):
            got = await invoke_with_cap_retry(
                gate, 'test-label',
                cost_store=cost_store, prompt='hi',
            )

        assert got.success is True
        assert 'Failed to save invocation cost' in caplog.text

    async def test_save_account_event_error_swallowed(self, caplog):
        """save_account_event failure is logged but retry still happens."""
        gate = MagicMock()
        gate.account_count = 2
        gate.before_invoke = AsyncMock(side_effect=['token-a', 'token-b'])
        gate.detect_cap_hit = MagicMock(side_effect=[True, False])
        gate.active_account_name = 'acct-b'
        gate.on_agent_complete = MagicMock()

        cost_store = MagicMock()
        cost_store.save_invocation = AsyncMock()
        cost_store.save_account_event = AsyncMock(side_effect=RuntimeError('db error'))

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
            with caplog.at_level(logging.WARNING):
                got = await invoke_with_cap_retry(
                    gate, 'test-label',
                    cost_store=cost_store, prompt='hi',
                )

        assert got.success is True
        assert 'Failed to save cap_hit event' in caplog.text
        # save_invocation still called on the successful retry
        cost_store.save_invocation.assert_awaited_once()

    async def test_capped_false_on_successful_invocation(self):
        """capped=False on save_invocation even after prior cap hits."""
        gate = MagicMock()
        gate.account_count = 2
        gate.before_invoke = AsyncMock(side_effect=['token-a', 'token-b'])
        gate.detect_cap_hit = MagicMock(side_effect=[True, False])
        gate.active_account_name = 'acct-b'
        gate.on_agent_complete = MagicMock()

        cost_store = MagicMock()
        cost_store.save_invocation = AsyncMock()
        cost_store.save_account_event = AsyncMock()

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
            await invoke_with_cap_retry(
                gate, 'test-label',
                cost_store=cost_store, prompt='hi',
            )

        assert cost_store.save_invocation.call_args.kwargs['capped'] is False


@pytest.mark.asyncio
class TestCapHitResume:

    async def test_resume_session_id_passed_on_cap_hit(self):
        """Cap hit with session_id in result → second invoke uses --resume."""
        gate = MagicMock()
        gate.account_count = 2
        gate.before_invoke = AsyncMock(side_effect=['token-a', 'token-b'])
        gate.detect_cap_hit = MagicMock(side_effect=[True, False])
        gate.active_account_name = 'acct-b'
        gate.on_agent_complete = MagicMock()

        capped_result = _make_result(session_id='sess-123')
        ok_result = _make_result()

        with (
            patch(
                'shared.cli_invoke.invoke_claude_agent',
                new_callable=AsyncMock,
                side_effect=[capped_result, ok_result],
            ) as mock_invoke,
            patch('shared.cli_invoke.asyncio') as mock_asyncio,
        ):
            mock_asyncio.sleep = AsyncMock()
            await invoke_with_cap_retry(gate, 'test-label', prompt='do stuff')

            assert mock_invoke.call_count == 2
            second_call = mock_invoke.call_args_list[1]
            assert second_call.kwargs.get('resume_session_id') == 'sess-123'
            assert second_call.kwargs.get('prompt') == CAP_HIT_RESUME_PROMPT

    async def test_fresh_start_when_no_session_id(self):
        """Cap hit with empty session_id → second invoke uses original prompt."""
        gate = MagicMock()
        gate.account_count = 2
        gate.before_invoke = AsyncMock(side_effect=['token-a', 'token-b'])
        gate.detect_cap_hit = MagicMock(side_effect=[True, False])
        gate.active_account_name = 'acct-b'
        gate.on_agent_complete = MagicMock()

        capped_result = _make_result(session_id='')
        ok_result = _make_result()

        with (
            patch(
                'shared.cli_invoke.invoke_claude_agent',
                new_callable=AsyncMock,
                side_effect=[capped_result, ok_result],
            ) as mock_invoke,
            patch('shared.cli_invoke.asyncio') as mock_asyncio,
        ):
            mock_asyncio.sleep = AsyncMock()
            await invoke_with_cap_retry(gate, 'test-label', prompt='do stuff')

            second_call = mock_invoke.call_args_list[1]
            assert 'resume_session_id' not in second_call.kwargs
            assert second_call.kwargs.get('prompt') == 'do stuff'

    async def test_resume_failure_falls_back_to_fresh(self):
        """Resume returns success=False (not cap hit) → retry with original prompt."""
        gate = MagicMock()
        gate.account_count = 2
        gate.before_invoke = AsyncMock(side_effect=['token-a', 'token-b', 'token-a'])
        gate.detect_cap_hit = MagicMock(side_effect=[True, False, False])
        gate.active_account_name = 'acct-b'
        gate.on_agent_complete = MagicMock()

        capped_result = _make_result(session_id='sess-123')
        failed_resume = _make_result(success=False, output='resume error')
        ok_result = _make_result()

        with (
            patch(
                'shared.cli_invoke.invoke_claude_agent',
                new_callable=AsyncMock,
                side_effect=[capped_result, failed_resume, ok_result],
            ) as mock_invoke,
            patch('shared.cli_invoke.asyncio') as mock_asyncio,
        ):
            mock_asyncio.sleep = AsyncMock()
            got = await invoke_with_cap_retry(gate, 'test-label', prompt='do stuff')

            assert mock_invoke.call_count == 3
            # Second call: resume attempt
            assert mock_invoke.call_args_list[1].kwargs.get('resume_session_id') == 'sess-123'
            # Third call: fresh fallback
            third_call = mock_invoke.call_args_list[2]
            assert 'resume_session_id' not in third_call.kwargs
            assert third_call.kwargs.get('prompt') == 'do stuff'
            assert got.success is True

    async def test_original_prompt_preserved_across_multiple_retries(self):
        """Multiple cap hits → fallback always uses original prompt, never mutated."""
        gate = MagicMock()
        gate.account_count = 3
        gate.before_invoke = AsyncMock(side_effect=['t-a', 't-b', 't-c'])
        # Two cap hits (with session), then success
        gate.detect_cap_hit = MagicMock(side_effect=[True, True, False])
        gate.active_account_name = 'next'
        gate.on_agent_complete = MagicMock()

        r1 = _make_result(session_id='sess-1')
        r2 = _make_result(session_id='sess-2')
        r3 = _make_result()

        with (
            patch(
                'shared.cli_invoke.invoke_claude_agent',
                new_callable=AsyncMock,
                side_effect=[r1, r2, r3],
            ) as mock_invoke,
            patch('shared.cli_invoke.asyncio') as mock_asyncio,
        ):
            mock_asyncio.sleep = AsyncMock()
            await invoke_with_cap_retry(
                gate, 'test-label', prompt='original prompt here',
            )

            # First call: original prompt
            assert mock_invoke.call_args_list[0].kwargs.get('prompt') == 'original prompt here'
            # Second call: resume sess-1
            assert mock_invoke.call_args_list[1].kwargs.get('resume_session_id') == 'sess-1'
            assert mock_invoke.call_args_list[1].kwargs.get('prompt') == CAP_HIT_RESUME_PROMPT
            # Third call: resume sess-2
            assert mock_invoke.call_args_list[2].kwargs.get('resume_session_id') == 'sess-2'
            assert mock_invoke.call_args_list[2].kwargs.get('prompt') == CAP_HIT_RESUME_PROMPT


# ── ARG_MAX protection ─────────────────────────────────────────────────


def _make_subprocess_result(stdout='', stderr='', returncode=0, duration_ms=100):
    return _SubprocessResult(
        stdout=stdout, stderr=stderr,
        returncode=returncode, duration_ms=duration_ms,
    )


def _successful_json_output(**overrides):
    data = {
        'result': 'ok',
        'subtype': 'success',
        'cost_usd': 0.01,
        'duration_ms': 100,
        'num_turns': 1,
        'session_id': 'sess-test',
    }
    data.update(overrides)
    return json.dumps(data)


@pytest.mark.asyncio
class TestLargePayloadHandling:
    """Verify system prompt and user prompt bypass CLI args to avoid ARG_MAX."""

    async def test_system_prompt_uses_temp_file(self, tmp_path):
        """System prompt is written to a temp file via --system-prompt-file, not inline."""
        captured_cmd = []

        async def fake_exec(*args, **kwargs):
            captured_cmd.extend(args)
            proc = MagicMock()
            proc.communicate = AsyncMock(return_value=(
                _successful_json_output().encode(),
                b'',
            ))
            proc.returncode = 0
            proc.terminate = MagicMock()
            proc.kill = MagicMock()
            proc.wait = AsyncMock()
            return proc

        with patch('shared.cli_invoke.asyncio.create_subprocess_exec', side_effect=fake_exec):
            await invoke_claude_agent(
                prompt='hello',
                system_prompt='You are a test assistant.',
                cwd=tmp_path,
            )

        # --system-prompt-file should appear, --system-prompt (with inline value) should not
        assert '--system-prompt-file' in captured_cmd
        assert '--system-prompt' not in captured_cmd

        # The file path argument should follow --system-prompt-file
        idx = captured_cmd.index('--system-prompt-file')
        file_path = captured_cmd[idx + 1]
        # File should be cleaned up after invocation
        assert not Path(file_path).exists(), 'Temp system prompt file was not cleaned up'

    async def test_prompt_sent_via_stdin_not_args(self, tmp_path):
        """User prompt is piped via stdin, not passed as a CLI argument."""
        captured_cmd = []
        captured_kwargs = {}

        async def fake_exec(*args, **kwargs):
            captured_cmd.extend(args)
            captured_kwargs.update(kwargs)
            proc = MagicMock()
            proc.communicate = AsyncMock(return_value=(
                _successful_json_output().encode(),
                b'',
            ))
            proc.returncode = 0
            proc.terminate = MagicMock()
            proc.kill = MagicMock()
            proc.wait = AsyncMock()
            return proc

        prompt_text = 'This is the user prompt for testing'
        with patch('shared.cli_invoke.asyncio.create_subprocess_exec', side_effect=fake_exec):
            await invoke_claude_agent(
                prompt=prompt_text,
                system_prompt='sys',
                cwd=tmp_path,
            )

        # Prompt text must NOT appear in any command argument
        for arg in captured_cmd:
            assert prompt_text not in str(arg), (
                f'Prompt text found in cmd arg: {arg!r}'
            )

        # stdin must be PIPE (for piping prompt data)
        assert captured_kwargs.get('stdin') == asyncio.subprocess.PIPE

    async def test_temp_files_cleaned_up_on_error(self, tmp_path):
        """Temp files are cleaned up even when subprocess raises."""
        created_files = []
        original_mkstemp = __import__('tempfile').mkstemp

        def tracking_mkstemp(**kwargs):
            fd, path = original_mkstemp(**kwargs)
            created_files.append(path)
            return fd, path

        async def failing_exec(*args, **kwargs):
            raise RuntimeError('Simulated subprocess failure')

        with (
            patch('shared.cli_invoke.tempfile.mkstemp', side_effect=tracking_mkstemp),
            patch('shared.cli_invoke.asyncio.create_subprocess_exec', side_effect=failing_exec),
            pytest.raises(RuntimeError, match='Simulated subprocess failure'),
        ):
            await invoke_claude_agent(
                prompt='hello',
                system_prompt='sys',
                cwd=tmp_path,
            )

        # All temp files should be cleaned up
        assert len(created_files) >= 1, 'Expected at least 1 temp file (system prompt)'
        for f in created_files:
            assert not Path(f).exists(), f'Temp file not cleaned up: {f}'

    async def test_large_payload_no_arg_exceeds_max_strlen(self, tmp_path):
        """260KB system prompt + 260KB user prompt: no CLI arg exceeds MAX_ARG_STRLEN."""
        MAX_ARG_STRLEN = 131072  # 128KB, Linux per-argument limit

        captured_cmd = []
        captured_communicate_input = []

        async def fake_exec(*args, **kwargs):
            captured_cmd.extend(args)
            proc = MagicMock()

            async def fake_communicate(input=None):
                captured_communicate_input.append(input)
                return (_successful_json_output().encode(), b'')

            proc.communicate = fake_communicate
            proc.returncode = 0
            proc.terminate = MagicMock()
            proc.kill = MagicMock()
            proc.wait = AsyncMock()
            return proc

        large_system = 'S' * 260_000
        large_prompt = 'P' * 260_000

        with patch('shared.cli_invoke.asyncio.create_subprocess_exec', side_effect=fake_exec):
            await invoke_claude_agent(
                prompt=large_prompt,
                system_prompt=large_system,
                cwd=tmp_path,
            )

        # No individual CLI argument should exceed MAX_ARG_STRLEN
        for arg in captured_cmd:
            assert len(str(arg).encode()) <= MAX_ARG_STRLEN, (
                f'CLI arg exceeds MAX_ARG_STRLEN ({len(str(arg).encode())} bytes): {str(arg)[:100]}...'
            )

        # The large prompt should arrive via stdin, not args
        assert len(captured_communicate_input) == 1
        assert captured_communicate_input[0] == large_prompt.encode()

    async def test_resume_skips_system_prompt_file(self, tmp_path):
        """When resuming a session, --system-prompt-file is not used."""
        captured_cmd = []

        async def fake_exec(*args, **kwargs):
            captured_cmd.extend(args)
            proc = MagicMock()
            proc.communicate = AsyncMock(return_value=(
                _successful_json_output().encode(),
                b'',
            ))
            proc.returncode = 0
            proc.terminate = MagicMock()
            proc.kill = MagicMock()
            proc.wait = AsyncMock()
            return proc

        with patch('shared.cli_invoke.asyncio.create_subprocess_exec', side_effect=fake_exec):
            await invoke_claude_agent(
                prompt='continue',
                system_prompt='ignored on resume',
                cwd=tmp_path,
                resume_session_id='sess-abc',
            )

        assert '--resume' in captured_cmd
        assert '--system-prompt-file' not in captured_cmd
        assert '--system-prompt' not in captured_cmd
