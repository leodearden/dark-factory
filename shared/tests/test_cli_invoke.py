"""Tests for cli_invoke cap-hit retry backoff and CostStore instrumentation."""

from __future__ import annotations

import asyncio
import itertools
import json
import logging
import os
import uuid
from datetime import UTC, datetime
from pathlib import Path
from typing import Any
from unittest.mock import AsyncMock, MagicMock, patch

import pytest

from shared.cli_invoke import (
    _CAP_HIT_COOLDOWN_SECS,
    _MAX_CAP_COOLDOWN_SECS,
    CAP_HIT_RESUME_PROMPT,
    AgentFailureKind,
    AgentResult,
    _cpu_priority_prefix,
    _parse_claude_output,
    _run_subprocess,
    _SubprocessResult,
    _to_token_count,
    build_failure_message,
    classify_agent_failure,
    count_transcript_turns,
    invoke_claude_agent,
    invoke_with_cap_retry,
    is_timed_out_with_progress,
    is_zero_output_timeout,
    read_transcript_records,
)
from shared.testing import make_gate_mock


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


def _make_result(**overrides: Any) -> AgentResult:
    # dict[str, Any] is intentional: AgentResult fields have heterogeneous
    # types and spreading a concrete dict[str, <union>] defeats pyright's
    # per-parameter type checking at the ** call site.
    defaults: dict[str, Any] = {
        'success': True, 'output': 'ok', 'cost_usd': 0.5, 'stderr': '',
    }
    defaults.update(overrides)
    return AgentResult(**defaults)


class TestAgentResultAccountNameField:

    def test_account_name_field_defaults_empty(self):
        """AgentResult has account_name field that defaults to empty string."""
        result = AgentResult(success=True, output='ok')
        assert result.account_name == ''


class TestAgentResultTimedOutField:

    def test_agent_result_has_timed_out_field_defaults_false(self):
        """AgentResult has timed_out field that defaults to False."""
        result = AgentResult(success=True, output='ok')
        assert result.timed_out is False


class TestSubprocessResultTimedOutField:

    def test_subprocess_result_has_timed_out_field_defaults_false(self):
        """_SubprocessResult has timed_out field that defaults to False."""
        result = _SubprocessResult(stdout='', stderr='', returncode=0, duration_ms=0)
        assert result.timed_out is False


@pytest.mark.asyncio
class TestAccountNameThreading:

    async def test_account_name_set_from_usage_gate(self):
        """account_name is stamped from usage_gate.active_account_name on success."""
        gate = make_gate_mock(
            account_count=2,
            before_invoke=AsyncMock(return_value='token-a'),
            active_account_name='acct-a',
        )

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
        gate = make_gate_mock(
            account_count=2,
            before_invoke=AsyncMock(return_value='token-a'),
            active_account_name=None,
        )

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

        gate = make_gate_mock(
            account_count=2,
            before_invoke=AsyncMock(side_effect=['token-a', 'token-b']),
            detect_cap_hit=MagicMock(side_effect=[True, False]),
        )
        # active_account_name is read on each loop iteration (slot
        # __aenter__ captures it, and the cap-hit path logs it).
        type(gate).active_account_name = PropertyMock(
            side_effect=['acct-a', 'acct-b', 'acct-b', 'acct-b'],
        )

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
        gate = make_gate_mock(
            account_count=2,
            before_invoke=AsyncMock(side_effect=['token-a', 'token-b']),
            detect_cap_hit=MagicMock(side_effect=[True, False]),
            active_account_name='acct-b',
        )

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
        gate = make_gate_mock(
            account_count=2,
            before_invoke=AsyncMock(return_value='token-a'),
        )

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
        gate = make_gate_mock(
            account_count=3,  # 3 accounts → first 3 hits are cycle 0
            before_invoke=AsyncMock(
                side_effect=['token-a', 'token-b', 'token-c'],
            ),
            detect_cap_hit=MagicMock(side_effect=[True, True, False]),
            active_account_name='next-acct',
        )

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
        gate = make_gate_mock(
            account_count=2,  # 2 accounts → cycle boundary at hit 2
            before_invoke=AsyncMock(
                side_effect=['token-a', 'token-b', 'token-a', 'token-b'],
            ),
            # 3 cap hits then success: hits 1-2 are cycle 0 (5s), hit 3 is cycle 1 (10s)
            detect_cap_hit=MagicMock(side_effect=[True, True, True, False]),
            active_account_name='next-acct',
        )

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
        # Need enough hits to exceed max: 5 * 2^6 = 320 > 300
        num_hits = 8
        gate = make_gate_mock(
            account_count=1,  # 1 account → every hit starts a new cycle
            before_invoke=AsyncMock(
                side_effect=['token-a'] * (num_hits + 1),
            ),
            detect_cap_hit=MagicMock(
                side_effect=[True] * num_hits + [False],
            ),
            active_account_name='acct-a',
        )

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
        gate = make_gate_mock(
            account_count=2,
            before_invoke=AsyncMock(return_value='token-a'),
            active_account_name='acct-a',
        )

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
        gate = make_gate_mock(
            account_count=2,
            before_invoke=AsyncMock(side_effect=['token-a', 'token-b']),
            detect_cap_hit=MagicMock(side_effect=[True, False]),
            active_account_name='acct-b',
        )

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
        gate = make_gate_mock(
            account_count=2,
            before_invoke=AsyncMock(return_value='token-a'),
            active_account_name='acct-a',
        )

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
        gate = make_gate_mock(
            account_count=2,
            before_invoke=AsyncMock(return_value='token-a'),
            active_account_name='acct-a',
        )

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
        gate = make_gate_mock(
            account_count=2,
            before_invoke=AsyncMock(side_effect=['token-a', 'token-b']),
            detect_cap_hit=MagicMock(side_effect=[True, False]),
            active_account_name='acct-b',
        )

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
        gate = make_gate_mock(
            account_count=2,
            before_invoke=AsyncMock(side_effect=['token-a', 'token-b']),
            detect_cap_hit=MagicMock(side_effect=[True, False]),
            active_account_name='acct-b',
        )

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
        gate = make_gate_mock(
            account_count=2,
            before_invoke=AsyncMock(side_effect=['token-a', 'token-b']),
            detect_cap_hit=MagicMock(side_effect=[True, False]),
            active_account_name='acct-b',
        )

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
        gate = make_gate_mock(
            account_count=2,
            before_invoke=AsyncMock(side_effect=['token-a', 'token-b']),
            detect_cap_hit=MagicMock(side_effect=[True, False]),
            active_account_name='acct-b',
        )

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
        gate = make_gate_mock(
            account_count=2,
            before_invoke=AsyncMock(side_effect=['token-a', 'token-b', 'token-a']),
            detect_cap_hit=MagicMock(side_effect=[True, False, False]),
            active_account_name='acct-b',
        )

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
        gate = make_gate_mock(
            account_count=3,
            before_invoke=AsyncMock(side_effect=['t-a', 't-b', 't-c']),
            # Two cap hits (with session), then success
            detect_cap_hit=MagicMock(side_effect=[True, True, False]),
            active_account_name='next',
        )

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

    async def test_fresh_fallback_regenerates_session_id(self):
        """Failed resume → fresh fallback hands the CLI a NEW pre-allocated UUID.

        Regression for reify-3604: the resume attempt may already have committed
        the caller's pre-allocated UUID to disk, so reusing it on the fresh
        ``--session-id`` retry makes the CLI exit instantly with 'Session ID …
        is already in use'.  The fallback must drop resume_session_id and
        regenerate session_id.
        """
        gate = make_gate_mock(
            account_count=1,
            before_invoke=AsyncMock(side_effect=['token-a', 'token-a']),
            detect_cap_hit=MagicMock(return_value=False),
            active_account_name='acct',
        )

        # cost_usd defaults to 0.5 in _make_result, so the failure does not trip
        # the zero-cost heuristic — it routes through the resume-fallback branch.
        failed_resume = _make_result(success=False, output='resume error')
        ok_result = _make_result()

        with (
            patch(
                'shared.cli_invoke.invoke_claude_agent',
                new_callable=AsyncMock,
                side_effect=[failed_resume, ok_result],
            ) as mock_invoke,
            patch('shared.cli_invoke.asyncio') as mock_asyncio,
        ):
            mock_asyncio.sleep = AsyncMock()
            await invoke_with_cap_retry(
                gate, 'test-label',
                prompt='do stuff',
                session_id='sess-orig',
                resume_session_id='sess-orig',
            )

        assert mock_invoke.call_count == 2
        # First call: resume attempt with the caller's session id.
        assert mock_invoke.call_args_list[0].kwargs.get('resume_session_id') == 'sess-orig'
        # Second call: fresh fallback — resume dropped, session_id regenerated.
        second_call = mock_invoke.call_args_list[1]
        assert 'resume_session_id' not in second_call.kwargs
        new_sid = second_call.kwargs.get('session_id')
        assert new_sid and new_sid != 'sess-orig'
        assert str(uuid.UUID(new_sid)) == new_sid


# ── Caller-initiated resume (crash recovery) ───────────────────────────


@pytest.mark.asyncio
class TestCallerInitiatedResume:
    """Regression for crash-recovery prompt-loss at workflow ↔ cli_invoke boundary.

    When the orchestrator resumes a crashed agent, it now passes the real task
    prompt + resume_session_id to invoke_with_cap_retry.  cli_invoke must
    substitute CRASH_RECOVERY_RESUME_PROMPT for the first subprocess call and
    restore the real task prompt on any non-cap resume failure fallback.
    """

    async def test_caller_resume_substitutes_continuation_prompt_and_restores_real_prompt_on_fresh_fallback(self):
        """cli_invoke swaps in CRASH_RECOVERY_RESUME_PROMPT then restores real prompt on fallback."""
        from shared.cli_invoke import CRASH_RECOVERY_RESUME_PROMPT  # noqa: PLC0415

        gate = make_gate_mock(
            account_count=1,
            before_invoke=AsyncMock(side_effect=['tok-1', 'tok-2']),
            detect_cap_hit=MagicMock(return_value=False),
            active_account_name='acct',
        )

        failed_resume = _make_result(success=False, output='resume error')
        ok_result = _make_result()

        with patch(
            'shared.cli_invoke.invoke_claude_agent',
            new_callable=AsyncMock,
            side_effect=[failed_resume, ok_result],
        ) as mock_invoke:
            got = await invoke_with_cap_retry(
                gate, 'test-label',
                prompt='real task prompt',
                resume_session_id='sess-xyz',
            )

        assert got.success is True
        assert mock_invoke.call_count == 2

        # First call: resume attempt — cli_invoke must substitute the continuation prompt
        first_call = mock_invoke.call_args_list[0]
        assert first_call.kwargs.get('prompt') == CRASH_RECOVERY_RESUME_PROMPT
        assert first_call.kwargs.get('resume_session_id') == 'sess-xyz'

        # Second call: fresh fallback — real task prompt restored, no resume_session_id
        second_call = mock_invoke.call_args_list[1]
        assert 'resume_session_id' not in second_call.kwargs
        assert second_call.kwargs.get('prompt') == 'real task prompt'


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


# ── session_id preallocation (crash-recovery resume) ──────────────────


@pytest.mark.asyncio
class TestSessionIdPreallocation:
    """Pre-allocated session UUIDs let crash recovery resume the same session."""

    async def _run_with_capture(self, tmp_path, **kwargs):
        captured_cmd: list = []

        async def fake_exec(*args, **kw):
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
                prompt='hi', system_prompt='sys', cwd=tmp_path, **kwargs,
            )
        return captured_cmd

    async def test_session_id_emits_flag(self, tmp_path):
        """When session_id is set and resume_session_id is not, --session-id <uuid> appears."""
        cmd = await self._run_with_capture(tmp_path, session_id='uuid-fresh')
        assert '--session-id' in cmd
        idx = cmd.index('--session-id')
        assert cmd[idx + 1] == 'uuid-fresh'
        assert '--resume' not in cmd

    async def test_resume_wins_over_session_id(self, tmp_path):
        """When both resume_session_id and session_id are set, --resume wins."""
        cmd = await self._run_with_capture(
            tmp_path, session_id='uuid-fresh', resume_session_id='uuid-resume',
        )
        assert '--resume' in cmd
        idx = cmd.index('--resume')
        assert cmd[idx + 1] == 'uuid-resume'
        # --session-id and --resume are mutually exclusive at the CLI level
        assert '--session-id' not in cmd

    async def test_no_flag_when_unset(self, tmp_path):
        """Without session_id, neither --session-id nor --resume appears."""
        cmd = await self._run_with_capture(tmp_path)
        assert '--session-id' not in cmd
        assert '--resume' not in cmd


# ── env_overrides plumbing ────────────────────────────────────────────


@pytest.mark.asyncio
class TestEnvOverrides:
    """Verify env_overrides are merged into the subprocess env without mutating os.environ."""

    async def test_env_overrides_merged_into_subprocess_env(self, tmp_path):
        """env_overrides keys appear in the env dict passed to create_subprocess_exec.

        When ANTHROPIC_BASE_URL is present, the bridge is started and the URL in
        the subprocess env is the bridge's local URL (not the raw upstream value).
        Other overrides are merged verbatim.
        """
        captured_kwargs = {}

        async def fake_exec(*args, **kwargs):
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

        overrides = {
            'ANTHROPIC_BASE_URL': 'http://vllm:8000/v1',
            'ANTHROPIC_API_KEY': 'dummy',
            'ANTHROPIC_DEFAULT_SONNET_MODEL': 'Qwen/Qwen3-Coder-Next',
        }

        MockVllmBridge, mock_bridge = _make_mock_bridge('http://127.0.0.1:54321')

        with (
            patch('shared.cli_invoke.asyncio.create_subprocess_exec', side_effect=fake_exec),
            patch('shared.cli_invoke.VllmBridge', MockVllmBridge),
        ):
            await invoke_claude_agent(
                prompt='hello',
                system_prompt='sys',
                cwd=tmp_path,
                env_overrides=overrides,
            )

        env = captured_kwargs['env']
        # ANTHROPIC_BASE_URL is rewritten to the bridge's local URL
        assert env['ANTHROPIC_BASE_URL'] == 'http://127.0.0.1:54321'
        assert env['ANTHROPIC_API_KEY'] == 'dummy'
        assert env['ANTHROPIC_DEFAULT_SONNET_MODEL'] == 'Qwen/Qwen3-Coder-Next'

    async def test_env_overrides_do_not_mutate_os_environ(self, tmp_path):
        """Passing env_overrides must not modify the calling process's os.environ."""
        captured_kwargs = {}

        async def fake_exec(*args, **kwargs):
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

        sentinel_key = '_TEST_ENV_OVERRIDE_SENTINEL'
        assert sentinel_key not in os.environ, 'Sentinel already in os.environ — test precondition violated'

        overrides = {sentinel_key: 'should-not-leak'}

        with patch('shared.cli_invoke.asyncio.create_subprocess_exec', side_effect=fake_exec):
            await invoke_claude_agent(
                prompt='hello',
                system_prompt='sys',
                cwd=tmp_path,
                env_overrides=overrides,
            )

        # The override must reach the subprocess env
        assert captured_kwargs['env'][sentinel_key] == 'should-not-leak'
        # But must NOT leak into os.environ
        assert sentinel_key not in os.environ

    async def test_env_overrides_none_is_harmless(self, tmp_path):
        """env_overrides=None (default) produces a valid subprocess env."""
        captured_kwargs = {}

        async def fake_exec(*args, **kwargs):
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

        with patch('shared.cli_invoke.asyncio.create_subprocess_exec', side_effect=fake_exec):
            await invoke_claude_agent(
                prompt='hello',
                system_prompt='sys',
                cwd=tmp_path,
                env_overrides=None,
            )

        # Should still have an env dict (base os.environ minus ANTHROPIC_API_KEY)
        assert isinstance(captured_kwargs['env'], dict)
        assert len(captured_kwargs['env']) > 0


# ── VllmBridge activation tests ──────────────────────────────────────────────


def _make_mock_bridge(url: str = 'http://127.0.0.1:54321'):
    """Return a (MockClass, mock_instance) pair for patching VllmBridge."""
    mock_instance = MagicMock()
    mock_instance.start = AsyncMock()
    mock_instance.stop = AsyncMock()
    mock_instance.url = url
    MockClass = MagicMock(return_value=mock_instance)
    return MockClass, mock_instance


def _make_fake_exec(captured_kwargs: dict):
    """Return a fake create_subprocess_exec that records env kwargs."""
    async def fake_exec(*args, **kwargs):
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
    return fake_exec


@pytest.mark.asyncio
class TestVllmBridgeActivation:
    """VllmBridge is started transparently when env_overrides contains ANTHROPIC_BASE_URL."""

    async def test_starts_bridge_when_base_url_present(self, tmp_path):
        """Bridge is constructed with upstream_url, started, and env is rewritten to bridge URL."""
        captured_kwargs: dict = {}
        MockVllmBridge, mock_bridge = _make_mock_bridge('http://127.0.0.1:54321')

        with (
            patch('shared.cli_invoke.asyncio.create_subprocess_exec',
                  side_effect=_make_fake_exec(captured_kwargs)),
            patch('shared.cli_invoke.VllmBridge', MockVllmBridge, create=True),
        ):
            await invoke_claude_agent(
                prompt='hello',
                system_prompt='sys',
                cwd=tmp_path,
                env_overrides={'ANTHROPIC_BASE_URL': 'http://upstream:8000'},
            )

        # Bridge constructed with upstream URL
        MockVllmBridge.assert_called_once_with(upstream_url='http://upstream:8000')
        # start() awaited exactly once
        mock_bridge.start.assert_awaited_once()
        # subprocess env has bridge URL, not original upstream URL
        assert captured_kwargs['env']['ANTHROPIC_BASE_URL'] == 'http://127.0.0.1:54321'

    async def test_does_not_start_bridge_when_base_url_absent(self, tmp_path):
        """Bridge is NOT instantiated when env_overrides lacks ANTHROPIC_BASE_URL."""
        MockVllmBridge, _ = _make_mock_bridge()
        captured_kwargs: dict = {}

        with (
            patch('shared.cli_invoke.asyncio.create_subprocess_exec',
                  side_effect=_make_fake_exec(captured_kwargs)),
            patch('shared.cli_invoke.VllmBridge', MockVllmBridge, create=True),
        ):
            # No ANTHROPIC_BASE_URL in overrides
            await invoke_claude_agent(
                prompt='hello',
                system_prompt='sys',
                cwd=tmp_path,
                env_overrides={'FOO': 'bar'},
            )
            # No env_overrides at all
            await invoke_claude_agent(
                prompt='hello',
                system_prompt='sys',
                cwd=tmp_path,
                env_overrides=None,
            )

        # Bridge was never instantiated
        MockVllmBridge.assert_not_called()

    async def test_stops_bridge_on_success(self, tmp_path):
        """bridge.stop() is awaited after a successful subprocess invocation."""
        call_order: list[str] = []
        captured_kwargs: dict = {}

        mock_instance = MagicMock()

        async def mock_start():
            call_order.append('start')

        async def mock_stop():
            call_order.append('stop')

        mock_instance.start = mock_start
        mock_instance.stop = mock_stop
        mock_instance.url = 'http://127.0.0.1:54321'
        MockVllmBridge = MagicMock(return_value=mock_instance)

        async def fake_exec_recording(*args, **kwargs):
            captured_kwargs.update(kwargs)
            call_order.append('exec')
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

        with (
            patch('shared.cli_invoke.asyncio.create_subprocess_exec',
                  side_effect=fake_exec_recording),
            patch('shared.cli_invoke.VllmBridge', MockVllmBridge, create=True),
        ):
            await invoke_claude_agent(
                prompt='hello',
                system_prompt='sys',
                cwd=tmp_path,
                env_overrides={'ANTHROPIC_BASE_URL': 'http://upstream:8000'},
            )

        assert 'start' in call_order
        assert 'stop' in call_order
        # start before exec, stop after exec
        assert call_order.index('start') < call_order.index('exec')
        assert call_order.index('exec') < call_order.index('stop')

    async def test_stops_bridge_on_subprocess_exception(self, tmp_path):
        """bridge.stop() is awaited even when the subprocess raises."""
        MockVllmBridge, mock_bridge = _make_mock_bridge()

        async def fake_exec_raises(*args, **kwargs):
            raise RuntimeError('subprocess failed')

        with (
            patch('shared.cli_invoke.asyncio.create_subprocess_exec',
                  side_effect=fake_exec_raises),
            patch('shared.cli_invoke.VllmBridge', MockVllmBridge, create=True),
            pytest.raises(RuntimeError, match='subprocess failed'),
        ):
            await invoke_claude_agent(
                prompt='hello',
                system_prompt='sys',
                cwd=tmp_path,
                env_overrides={'ANTHROPIC_BASE_URL': 'http://upstream:8000'},
            )

        # stop() awaited exactly once despite the exception
        mock_bridge.stop.assert_awaited_once()

    async def test_stops_bridge_when_start_raises(self, tmp_path):
        """bridge.stop() is awaited in the finally clause even when bridge.start() raises."""
        # Construct a bridge mock whose start() raises mid-way through initialisation
        mock_instance = MagicMock()
        mock_instance.start = AsyncMock(side_effect=RuntimeError('partial init failure'))
        mock_instance.stop = AsyncMock()
        mock_instance.url = 'http://127.0.0.1:54321'
        MockVllmBridge = MagicMock(return_value=mock_instance)

        # fake_exec must NEVER be reached because start() raises before the subprocess call
        captured_kwargs: dict = {}

        with (
            patch('shared.cli_invoke.asyncio.create_subprocess_exec',
                  side_effect=_make_fake_exec(captured_kwargs)),
            patch('shared.cli_invoke.VllmBridge', MockVllmBridge, create=True),
            pytest.raises(RuntimeError, match='partial init failure'),
        ):
            await invoke_claude_agent(
                prompt='hello',
                system_prompt='sys',
                cwd=tmp_path,
                env_overrides={'ANTHROPIC_BASE_URL': 'http://upstream:8000'},
            )

        # Bridge WAS constructed with the upstream URL
        MockVllmBridge.assert_called_once_with(upstream_url='http://upstream:8000')
        # start() was attempted exactly once
        mock_instance.start.assert_awaited_once()
        # stop() was called by the finally clause despite the start failure
        mock_instance.stop.assert_awaited_once()
        # subprocess was never reached (start raised before _run_subprocess)
        assert not captured_kwargs


# ── _run_subprocess timed_out flag ─────────────────────────────────────────


@pytest.mark.asyncio
class TestRunSubprocessTimedOut:

    async def test_run_subprocess_sets_timed_out_on_sigkill_branch(self, tmp_path):
        """_run_subprocess sets timed_out=True on the SIGTERM+SIGKILL branch."""
        proc = MagicMock()
        # Both communicate() calls raise TimeoutError → SIGTERM+SIGKILL path
        proc.communicate = AsyncMock(side_effect=TimeoutError)
        proc.terminate = MagicMock()
        proc.kill = MagicMock()
        proc.wait = AsyncMock()
        proc.returncode = None
        proc.pid = 12345

        async def fake_exec(*args, **kwargs):
            return proc

        with (
            patch('shared.cli_invoke.asyncio.create_subprocess_exec', side_effect=fake_exec),
            patch('shared.cli_invoke.terminate_process_group', new_callable=AsyncMock),
        ):
            result = await _run_subprocess(
                ['fake'], cwd=tmp_path, env={}, model='opus', timeout_seconds=0.1,
            )

        assert result.timed_out is True
        assert result.returncode != 0
        assert 'SIGTERM+SIGKILL' in result.stderr

    async def test_run_subprocess_sets_timed_out_on_sigterm_grace_branch(self, tmp_path):
        """_run_subprocess sets timed_out=True on the SIGTERM-grace branch."""
        valid_json = json.dumps({
            'result': 'ok',
            'subtype': 'success',
            'cost_usd': 0.01,
            'duration_ms': 100,
            'num_turns': 1,
            'session_id': 'sess-grace',
        }).encode()

        call_count = 0

        async def communicate_side_effect(input=None):  # noqa: A002
            nonlocal call_count
            call_count += 1
            if call_count == 1:
                raise TimeoutError
            # Second call (post-SIGTERM grace) returns normally
            return (valid_json, b'')

        proc = MagicMock()
        proc.communicate = AsyncMock(side_effect=communicate_side_effect)
        proc.terminate = MagicMock()
        proc.kill = MagicMock()
        proc.wait = AsyncMock()
        proc.returncode = 0

        async def fake_exec(*args, **kwargs):
            return proc

        with patch('shared.cli_invoke.asyncio.create_subprocess_exec', side_effect=fake_exec):
            result = await _run_subprocess(
                ['fake'], cwd=tmp_path, env={}, model='opus', timeout_seconds=0.1,
            )

        assert result.timed_out is True
        assert 'Process terminated after' in result.stderr
        assert result.returncode == 0  # grace path preserves returncode

    async def test_run_subprocess_stamps_transcript_turns_on_sigkill(self, tmp_path):
        """SIGKILL path stamps transcript_turns from the on-disk transcript."""
        sid = str(uuid.uuid4())
        cfg_dir = tmp_path / 'cfg'
        transcript_dir = cfg_dir / 'projects' / 'slug-abc'
        transcript_dir.mkdir(parents=True)
        # Write a transcript with 3 assistant records (interleaved with non-assistant)
        records = [
            {'type': 'system', 'content': 'init'},
            {'type': 'assistant', 'content': 'turn 1'},
            {'type': 'user', 'content': 'reply'},
            {'type': 'assistant', 'content': 'turn 2'},
            {'type': 'user', 'content': 'reply 2'},
            {'type': 'assistant', 'content': 'turn 3'},
        ]
        (transcript_dir / f'{sid}.jsonl').write_text(
            '\n'.join(json.dumps(r) for r in records) + '\n'
        )

        proc = MagicMock()
        proc.communicate = AsyncMock(side_effect=TimeoutError)
        proc.terminate = MagicMock()
        proc.kill = MagicMock()
        proc.wait = AsyncMock()
        proc.returncode = None
        proc.pid = 12345

        async def fake_exec(*args, **kwargs):
            return proc

        with (
            patch('shared.cli_invoke.asyncio.create_subprocess_exec', side_effect=fake_exec),
            patch('shared.cli_invoke.terminate_process_group', new_callable=AsyncMock),
        ):
            result = await _run_subprocess(
                ['fake'], cwd=tmp_path, env={}, model='opus', timeout_seconds=0.1,
                session_id=sid, config_dir=cfg_dir,
            )

        assert result.timed_out is True
        assert result.transcript_turns == 3

    async def test_run_subprocess_transcript_turns_none_when_no_session_id(self, tmp_path):
        """When session_id is None, transcript_turns remains None on timeout path."""
        proc = MagicMock()
        proc.communicate = AsyncMock(side_effect=TimeoutError)
        proc.terminate = MagicMock()
        proc.kill = MagicMock()
        proc.wait = AsyncMock()
        proc.returncode = None
        proc.pid = 12345

        async def fake_exec(*args, **kwargs):
            return proc

        with (
            patch('shared.cli_invoke.asyncio.create_subprocess_exec', side_effect=fake_exec),
            patch('shared.cli_invoke.terminate_process_group', new_callable=AsyncMock),
        ):
            result = await _run_subprocess(
                ['fake'], cwd=tmp_path, env={}, model='opus', timeout_seconds=0.1,
                session_id=None, config_dir=tmp_path,
            )

        assert result.timed_out is True
        assert result.transcript_turns is None


# ── _parse_claude_output timed_out propagation ───────────────────────────────

_CLAUDE_VALID_JSON_STDOUT = json.dumps({
    'result': 'ok',
    'subtype': 'success',
    'cost_usd': 0.01,
    'duration_ms': 100,
    'num_turns': 1,
    'session_id': 'sess-test',
})


class TestParseClaudeOutputPropagatesTimedOut:
    """Parser always sets timed_out — callers no longer need to patch it post-hoc."""

    @pytest.mark.parametrize('input_timed_out', [True, False])
    @pytest.mark.parametrize(
        'stdout,stderr,returncode',
        [
            ('', 'timeout stderr', 1),
            ('not valid json', '', 1),
            (_CLAUDE_VALID_JSON_STDOUT, '', 0),
        ],
        ids=['empty_stdout', 'json_decode_error', 'normal_parse'],
    )
    def test_propagates_timed_out(self, stdout, stderr, returncode, input_timed_out):
        """_parse_claude_output propagates timed_out from the subprocess result."""
        sub = _SubprocessResult(stdout=stdout, stderr=stderr, returncode=returncode,
                                duration_ms=100, timed_out=input_timed_out)
        agent = _parse_claude_output(sub)
        assert agent.timed_out is input_timed_out

    def test_empty_output_preserves_duration_ms(self):
        """A subprocess that produced no stdout (e.g. SIGTERM'd on timeout)
        must forward its real duration to AgentResult so downstream heuristics
        can distinguish a true zero-cost instant exit from a long timeout.
        """
        sub = _SubprocessResult(
            stdout='', stderr='terminated', returncode=-15,
            duration_ms=240_003, timed_out=True,
        )
        agent = _parse_claude_output(sub)
        assert agent.subtype == 'error_empty_output'
        assert agent.timed_out is True
        assert agent.duration_ms == 240_003


# ── transcript_turns field + propagation ─────────────────────────────────────


class TestTranscriptTurnsFieldAndPropagation:
    """Tests for transcript_turns field on _SubprocessResult + AgentResult,
    and propagation through _parse_claude_output on all three return paths.

    transcript_turns defaults to None and is propagated from the subprocess
    result to the AgentResult on every code path in _parse_claude_output.
    """

    def test_subprocess_result_defaults_transcript_turns_none(self):
        """_SubprocessResult.transcript_turns defaults to None."""
        r = _SubprocessResult(stdout='', stderr='', returncode=0, duration_ms=10)
        assert r.transcript_turns is None

    def test_agent_result_defaults_transcript_turns_none(self):
        """AgentResult.transcript_turns defaults to None."""
        r = AgentResult(success=True, output='ok')
        assert r.transcript_turns is None

    def test_propagation_empty_stdout_path(self):
        """Empty stdout path: _SubprocessResult(transcript_turns=7) → AgentResult.transcript_turns==7."""
        sub = _SubprocessResult(
            stdout='', stderr='', returncode=1,
            duration_ms=100, timed_out=True, transcript_turns=7,
        )
        agent = _parse_claude_output(sub)
        assert agent.subtype == 'error_empty_output'
        assert agent.transcript_turns == 7

    def test_propagation_json_decode_error_path(self):
        """JSON decode error path: transcript_turns=4 → AgentResult.transcript_turns==4."""
        sub = _SubprocessResult(
            stdout='not json at all', stderr='', returncode=1,
            duration_ms=100, timed_out=True, transcript_turns=4,
        )
        agent = _parse_claude_output(sub)
        assert agent.subtype == 'text_output'
        assert agent.transcript_turns == 4

    def test_propagation_valid_json_path(self):
        """Valid JSON parsed path: transcript_turns=2 → AgentResult.transcript_turns==2."""
        sub = _SubprocessResult(
            stdout=_CLAUDE_VALID_JSON_STDOUT, stderr='', returncode=0,
            duration_ms=100, transcript_turns=2,
        )
        agent = _parse_claude_output(sub)
        assert agent.transcript_turns == 2

    def test_propagation_default_none(self):
        """Default (transcript_turns not set) → AgentResult.transcript_turns is None."""
        sub = _SubprocessResult(
            stdout='', stderr='', returncode=1, duration_ms=100,
        )
        agent = _parse_claude_output(sub)
        assert agent.transcript_turns is None


# ── schema salvage (R1) ────────────────────────────────────────────────────────


class TestParseClaudeOutputSchemaSalvage:
    """When the CLI reports is_error=True but a structured_output dict is
    attached, the parser salvages it: success=True and schema_salvaged=True.

    This covers the error_max_turns + valid --json-schema payload case that
    previously blocked the curator's drop/combine decisions (see task #1922
    investigation notes in plans/floating-snuggling-pebble.md).
    """

    def test_is_error_with_structured_output_is_salvaged(self):
        stdout = json.dumps({
            'subtype': 'error_max_turns',
            'is_error': True,
            'cost_usd': 0.01,
            'duration_ms': 500,
            'num_turns': 2,
            'session_id': 'sess-x',
            'structured_output': {'action': 'drop', 'justification': 'dup'},
            'result': 'boom',
        })
        sub = _SubprocessResult(stdout=stdout, stderr='', returncode=1,
                                duration_ms=500, timed_out=False)
        agent = _parse_claude_output(sub)
        assert agent.success is True
        assert agent.schema_salvaged is True
        assert agent.structured_output == {
            'action': 'drop', 'justification': 'dup',
        }
        # Raw error text preserved for diagnostics
        assert agent.output == 'boom'

    def test_is_error_without_structured_output_not_salvaged(self):
        stdout = json.dumps({
            'subtype': 'error_max_turns',
            'is_error': True,
            'cost_usd': 0.0,
            'duration_ms': 100,
            'num_turns': 1,
            'result': 'just an error',
        })
        sub = _SubprocessResult(stdout=stdout, stderr='', returncode=1,
                                duration_ms=100, timed_out=False)
        agent = _parse_claude_output(sub)
        assert agent.success is False
        assert agent.schema_salvaged is False

    def test_clean_success_sets_schema_salvaged_false(self):
        stdout = _CLAUDE_VALID_JSON_STDOUT
        sub = _SubprocessResult(stdout=stdout, stderr='', returncode=0,
                                duration_ms=100, timed_out=False)
        agent = _parse_claude_output(sub)
        assert agent.success is True
        assert agent.schema_salvaged is False

    def test_is_error_with_non_dict_structured_output_not_salvaged(self):
        # structured_output is a string, not a dict — no salvage.
        stdout = json.dumps({
            'subtype': 'error_max_turns',
            'is_error': True,
            'cost_usd': 0.0,
            'duration_ms': 100,
            'num_turns': 1,
            'structured_output': 'not-a-dict',
            'result': 'err',
        })
        sub = _SubprocessResult(stdout=stdout, stderr='', returncode=1,
                                duration_ms=100, timed_out=False)
        agent = _parse_claude_output(sub)
        assert agent.success is False
        assert agent.schema_salvaged is False


# ── api_error_status surfacing ────────────────────────────────────────────────


class TestAgentResultApiErrorStatusField:
    """AgentResult surfaces HTTP error status for auth/permission failures."""

    def test_api_error_status_defaults_none(self):
        """AgentResult.api_error_status defaults to None when not set."""
        result = AgentResult(success=True, output='ok')
        assert result.api_error_status is None

    def test_api_error_status_accepts_int(self):
        """api_error_status can be set to an HTTP status code."""
        result = AgentResult(success=False, output='forbidden', api_error_status=403)
        assert result.api_error_status == 403


class TestParseClaudeOutputApiErrorStatus:
    """_parse_claude_output populates api_error_status when present in CLI JSON."""

    def test_populated_when_present(self):
        """api_error_status passes through from CLI JSON."""
        stdout = json.dumps({
            'subtype': 'error',
            'is_error': True,
            'cost_usd': 0.0,
            'duration_ms': 100,
            'num_turns': 0,
            'api_error_status': 403,
            'result': 'Your organization does not have access to Claude',
        })
        sub = _SubprocessResult(stdout=stdout, stderr='', returncode=1,
                                duration_ms=100, timed_out=False)
        agent = _parse_claude_output(sub)
        assert agent.api_error_status == 403
        assert agent.success is False

    def test_none_when_absent(self):
        """api_error_status is None when CLI JSON omits the field."""
        sub = _SubprocessResult(stdout=_CLAUDE_VALID_JSON_STDOUT, stderr='',
                                returncode=0, duration_ms=100, timed_out=False)
        agent = _parse_claude_output(sub)
        assert agent.api_error_status is None

    def test_none_on_empty_output(self):
        """api_error_status is None for the empty-output early return."""
        sub = _SubprocessResult(stdout='', stderr='', returncode=1,
                                duration_ms=10, timed_out=False)
        agent = _parse_claude_output(sub)
        assert agent.api_error_status is None

    def test_none_on_json_decode_error(self):
        """api_error_status is None for the text-output fallback path."""
        sub = _SubprocessResult(stdout='not json', stderr='', returncode=0,
                                duration_ms=10, timed_out=False)
        agent = _parse_claude_output(sub)
        assert agent.api_error_status is None


# ── caller-level timed_out propagation (characterization tests) ───────────────


@pytest.mark.asyncio
class TestClaudeCallerPropagatesTimedOut:
    """invoke_claude_agent must propagate timed_out=True from subprocess result."""

    async def test_claude_caller_propagates_timed_out(self, tmp_path):
        """invoke_claude_agent returns AgentResult with timed_out=True when subprocess timed out."""
        timed_result = _SubprocessResult(stdout='', stderr='timeout', returncode=1,
                                         duration_ms=100, timed_out=True)
        with patch('shared.cli_invoke._run_subprocess',
                   new_callable=AsyncMock, return_value=timed_result):
            agent = await invoke_claude_agent(
                prompt='hello', system_prompt='sys', cwd=tmp_path,
                model='claude-sonnet-4-5', timeout_seconds=30.0,
            )
        assert agent.timed_out is True


class TestInvokeClaudeAgentForwardsStartupGraceSecs:
    """invoke_claude_agent must forward startup_grace_secs down to _run_subprocess."""

    async def test_invoke_claude_agent_forwards_startup_grace_secs(self, tmp_path):
        """startup_grace_secs passed to invoke_claude_agent reaches _run_subprocess.

        Fails today: invoke_claude_agent has no startup_grace_secs param → TypeError.
        After step-10 it is forwarded via _invoke_claude to _run_subprocess.
        """
        captured: dict = {}
        minimal_result = _SubprocessResult(
            stdout='', stderr='', returncode=0, duration_ms=0,
        )

        async def capturing_run_subprocess(*args, **kwargs):
            captured.update(kwargs)
            # positional args: cmd, cwd, env, model, timeout_seconds
            return minimal_result

        with patch('shared.cli_invoke._run_subprocess', side_effect=capturing_run_subprocess):
            await invoke_claude_agent(
                prompt='x', system_prompt='s', cwd=tmp_path,
                startup_grace_secs=33.0,
            )

        assert captured.get('startup_grace_secs') == 33.0, (
            f'startup_grace_secs not forwarded to _run_subprocess; captured={captured!r}'
        )


def _make_gate(
    *,
    account_count: int = 2,
    before_invoke_tokens: str | list[str] = 'token-a',
    handle_cap_detected: bool = False,
    active_account_name: str = 'acct-a',
) -> MagicMock:
    """Factory for a gate mock used in TestHeuristicCapGating.

    Layers the heuristic-specific ``_handle_cap_detected`` attribute onto the
    canonical ``make_gate_mock()`` so the heuristic cap path can be exercised.
    """
    if isinstance(before_invoke_tokens, list):
        before_invoke = AsyncMock(side_effect=before_invoke_tokens)
    else:
        before_invoke = AsyncMock(return_value=before_invoke_tokens)
    gate = make_gate_mock(
        account_count=account_count,
        before_invoke=before_invoke,
        active_account_name=active_account_name,
    )
    gate._handle_cap_detected = MagicMock(return_value=handle_cap_detected)
    return gate


@pytest.mark.asyncio
class TestHeuristicCapGating:
    """Tests that the heuristic cap-detection path gates retry on
    _handle_cap_detected's return value."""

    async def test_heuristic_cap_no_retry_when_handle_returns_false(self):
        """Heuristic fires but _handle_cap_detected returns False → no retry."""
        gate = _make_gate()

        # Craft a result that triggers the heuristic: error, zero cost, instant
        heuristic_result = _make_result(success=False, cost_usd=0, turns=1, duration_ms=100)

        with (
            patch(
                'shared.cli_invoke.invoke_claude_agent',
                new_callable=AsyncMock,
                return_value=heuristic_result,
            ) as mock_invoke,
            patch('shared.cli_invoke.asyncio') as mock_asyncio,
        ):
            mock_asyncio.sleep = AsyncMock()
            # Previously tested with max_cap_retries=1; param removed in task-1401.
            # The fix should return without raising.
            got = await invoke_with_cap_retry(
                gate, 'test-label',
                prompt='hi',
            )

        assert got.success is False
        mock_invoke.assert_called_once()
        mock_asyncio.sleep.assert_not_called()
        # confirm_account_ok is skipped when cap_marked=False (token unresolvable)
        gate.confirm_account_ok.assert_not_called()
        # on_agent_complete is also skipped — unattributed cap hits must not be
        # miscounted as legitimate zero-cost completions by downstream consumers.
        gate.on_agent_complete.assert_not_called()

    async def test_heuristic_cap_retries_when_handle_returns_true(self):
        """Heuristic fires and _handle_cap_detected returns True → retry happens."""
        gate = _make_gate(
            before_invoke_tokens=['token-a', 'token-b'],
            handle_cap_detected=True,
            active_account_name='acct-b',
        )

        heuristic_result = _make_result(success=False, cost_usd=0, turns=1, duration_ms=100)
        ok_result = _make_result(success=True, cost_usd=0.5)

        with (
            patch(
                'shared.cli_invoke.invoke_claude_agent',
                new_callable=AsyncMock,
                side_effect=[heuristic_result, ok_result],
            ) as mock_invoke,
            patch('shared.cli_invoke.asyncio') as mock_asyncio,
        ):
            mock_asyncio.sleep = AsyncMock()
            got = await invoke_with_cap_retry(gate, 'test-label', prompt='hi')

        assert got.success is True
        assert mock_invoke.call_count == 2
        mock_asyncio.sleep.assert_called_once_with(_CAP_HIT_COOLDOWN_SECS)

    async def test_heuristic_cap_unresolved_token_logs_warning(self, caplog):
        """When _handle_cap_detected returns False, warning is logged."""
        gate = _make_gate()
        heuristic_result = _make_result(success=False, cost_usd=0, turns=1, duration_ms=100)

        with (
            patch(
                'shared.cli_invoke.invoke_claude_agent',
                new_callable=AsyncMock,
                return_value=heuristic_result,
            ),
            patch('shared.cli_invoke.asyncio') as mock_asyncio,
            caplog.at_level(logging.WARNING, logger='shared.cli_invoke'),
        ):
            mock_asyncio.sleep = AsyncMock()
            await invoke_with_cap_retry(gate, 'test-label', prompt='hi')

        assert 'suspicious zero-cost instant exit' in caplog.text
        assert 'heuristic cap suspected but no account could be marked' in caplog.text
        assert 'token unresolved' in caplog.text

    async def test_heuristic_cap_false_does_not_increment_consecutive_hits(self):
        """_handle_cap_detected returning False never increments consecutive_cap_hits."""
        gate = _make_gate()
        heuristic_result = _make_result(success=False, cost_usd=0, turns=1, duration_ms=100)

        with (
            patch(
                'shared.cli_invoke.invoke_claude_agent',
                new_callable=AsyncMock,
                side_effect=[heuristic_result],
            ) as mock_invoke,
            patch('shared.cli_invoke.asyncio') as mock_asyncio,
        ):
            mock_asyncio.sleep = AsyncMock()
            # Unlimited retries: if consecutive_cap_hits were incremented the loop
            # would never terminate (or exhaust side_effect).  The fix must exit
            # immediately without incrementing the counter.  Using side_effect
            # (finite list) means a spurious retry raises StopIteration instantly
            # rather than hanging — faster and more descriptive failure.
            got = await invoke_with_cap_retry(
                gate, 'test-label',
                prompt='hi',
            )

        # invoke_claude_agent called exactly once — no retry loop entered
        mock_invoke.assert_called_once()
        mock_asyncio.sleep.assert_not_called()
        assert got.success is False

    async def test_heuristic_cap_false_falls_through_to_resume_fallback(self):
        """cap_marked=False + resume_session_id → resume-fallback retries fresh, no sleep.

        This exercises the secondary path: heuristic fires but _handle_cap_detected
        returns False (token unresolvable), so the code falls through to the
        resume-session fallback (line ~372) which pops resume_session_id and
        retries from scratch — exactly one fresh retry, zero sleeps.
        """
        gate = _make_gate(before_invoke_tokens=['token-a', 'token-b'])

        # First call: heuristic-triggering failure with a prior resume_session_id
        heuristic_result = _make_result(success=False, cost_usd=0, turns=1, duration_ms=100)
        ok_result = _make_result(success=True, cost_usd=0.5)

        with (
            patch(
                'shared.cli_invoke.invoke_claude_agent',
                new_callable=AsyncMock,
                side_effect=[heuristic_result, ok_result],
            ) as mock_invoke,
            patch('shared.cli_invoke.asyncio') as mock_asyncio,
        ):
            mock_asyncio.sleep = AsyncMock()
            # Simulate a prior pattern-match cap hit that left resume_session_id set
            got = await invoke_with_cap_retry(
                gate, 'test-label',
                prompt='hi', resume_session_id='sess-123',
            )

        assert got.success is True
        # Two calls: first (heuristic+resume-fallback triggers retry), second succeeds
        assert mock_invoke.call_count == 2
        # No cap-hit cooldown sleep — resume-fallback retries immediately
        mock_asyncio.sleep.assert_not_called()

    async def test_heuristic_cap_exponential_backoff_on_consecutive_hits(self):
        """Two consecutive heuristic-True hits produce exponentially increasing cooldowns.

        With account_count=1, each hit increments full_cycles:
          hit 1 → full_cycles=0 → cooldown = _CAP_HIT_COOLDOWN_SECS * 2^0
          hit 2 → full_cycles=1 → cooldown = _CAP_HIT_COOLDOWN_SECS * 2^1
        This pins the counter-increment and backoff formula for the heuristic path.
        """
        gate = _make_gate(account_count=1, handle_cap_detected=True)

        heuristic_result = _make_result(success=False, cost_usd=0, turns=1, duration_ms=100)
        ok_result = _make_result(success=True, cost_usd=0.5)

        with (
            patch(
                'shared.cli_invoke.invoke_claude_agent',
                new_callable=AsyncMock,
                side_effect=[heuristic_result, heuristic_result, ok_result],
            ) as mock_invoke,
            patch('shared.cli_invoke.asyncio') as mock_asyncio,
        ):
            mock_asyncio.sleep = AsyncMock()
            got = await invoke_with_cap_retry(gate, 'test-label', prompt='hi')

        assert got.success is True
        assert mock_invoke.call_count == 3
        assert mock_asyncio.sleep.call_count == 2

        # Cooldown should double on the second hit
        first_cooldown = mock_asyncio.sleep.call_args_list[0][0][0]
        second_cooldown = mock_asyncio.sleep.call_args_list[1][0][0]
        assert first_cooldown == _CAP_HIT_COOLDOWN_SECS
        assert second_cooldown == _CAP_HIT_COOLDOWN_SECS * 2


# ── _run_subprocess process-group fix ────────────────────────────────────────


@pytest.mark.asyncio
class TestRunSubprocessProcessGroup:
    """_run_subprocess must spawn subprocesses in their own process group."""

    async def test_run_subprocess_passes_start_new_session_true(self, tmp_path):
        """create_subprocess_exec is called with start_new_session=True.

        Failing test — _run_subprocess does not pass that kwarg yet.
        """
        captured_kwargs: dict = {}

        async def fake_exec(*args, **kwargs):
            captured_kwargs.update(kwargs)
            proc = MagicMock()
            proc.communicate = AsyncMock(return_value=(b'', b''))
            proc.returncode = 0
            proc.terminate = MagicMock()
            proc.kill = MagicMock()
            proc.wait = AsyncMock()
            return proc

        with patch('shared.cli_invoke.asyncio.create_subprocess_exec', side_effect=fake_exec):
            await _run_subprocess(['echo', 'hi'], tmp_path, env={}, model='test')

        assert captured_kwargs.get('start_new_session') is True


class TestClassifyAgentFailure:
    """Regression coverage for the classifier added to stop empty-output
    escalations from being indistinguishable from real tool crashes to the
    steward.  Ordering of rules matters: TIMED_OUT is checked before MAX_TURNS,
    and API_ERROR before EMPTY_OUTPUT."""

    def test_classify_agent_failure_success(self):
        """A successful result is classified as SUCCESS regardless of other signals."""
        result = AgentResult(success=True, output='done', subtype='success', turns=3)
        cls = classify_agent_failure(result)
        assert cls.kind is AgentFailureKind.SUCCESS
        # diagnostic still populated so upstream logging never loses signal
        assert 'turns=3' in cls.diagnostic_detail

    def test_classify_agent_failure_max_turns(self):
        """subtype=error_max_turns with empty output → MAX_TURNS."""
        result = AgentResult(
            success=False, output='', subtype='error_max_turns',
            turns=75, output_tokens=12345,
        )
        cls = classify_agent_failure(result)
        assert cls.kind is AgentFailureKind.MAX_TURNS
        assert '75 turns' in cls.summary
        assert 'output_tokens=12345' in cls.summary

    def test_classify_agent_failure_empty_output(self):
        """subtype=error_empty_output → EMPTY_OUTPUT (distinct from MAX_TURNS)."""
        result = AgentResult(
            success=False, output='', subtype='error_empty_output', turns=1,
        )
        cls = classify_agent_failure(result)
        assert cls.kind is AgentFailureKind.EMPTY_OUTPUT

    def test_classify_agent_failure_api_error(self):
        """api_error_status populated → API_ERROR with status in summary."""
        result = AgentResult(
            success=False, output='Overloaded', subtype='',
            api_error_status=529,
        )
        cls = classify_agent_failure(result)
        assert cls.kind is AgentFailureKind.API_ERROR
        assert '529' in cls.summary

    def test_classify_agent_failure_timed_out(self):
        """timed_out=True beats error subtypes — wall-clock kill dominates."""
        result = AgentResult(
            success=False, output='', subtype='error_max_turns',
            turns=50, duration_ms=1_800_000, timed_out=True,
        )
        cls = classify_agent_failure(result)
        assert cls.kind is AgentFailureKind.TIMED_OUT
        assert '1800000ms' in cls.summary


class TestBuildFailureMessage:
    """Tests for the build_failure_message formatting helper."""

    def test_build_failure_message_delegates_to_classifier(self):
        """Wrapper prepends '{label} failed: ' and joins classifier summary + diagnostic_detail
        with '\\n' — the classifier's literal output is tested in TestClassifyAgentFailure."""
        result = AgentResult(
            success=False, output='', subtype='error_max_turns', turns=75,
        )
        cls = classify_agent_failure(result)
        msg = build_failure_message('Claude CLI agent', result)
        assert msg == f'Claude CLI agent failed: {cls.summary}\n{cls.diagnostic_detail}'

    def test_build_failure_message_label_is_prefix(self):
        """The label argument is preserved verbatim before ' failed: '."""
        result = AgentResult(success=False, output='', subtype='error_unexpected')
        for label in ('Reconciliation agent (claude-opus-4)', 'arbitrary-label-123'):
            msg = build_failure_message(label, result)
            assert msg.startswith(f'{label} failed: '), (
                f'Expected message to start with {label!r} + " failed: ", got {msg[:100]!r}'
            )


# ---------------------------------------------------------------------------
# Periodic cap_wait log
# ---------------------------------------------------------------------------

_PERIODIC_SLEEP_PATCH = 'shared.cli_invoke.asyncio.sleep'
_PERIODIC_INVOKE_PATCH = 'shared.cli_invoke.invoke_claude_agent'


@pytest.mark.asyncio
class TestCapWaitPeriodicLog:
    """Periodic structured cap_wait JSON line emitted during exact-detect cap-hit waits.

    Verifies: JSON structure and keys, task_id field, soonest_open_at (from
    UsageGate.soonest_resets_at), and ~10-min throttling (600 s window).
    RED until step-8 adds the logging logic.
    """

    def _exact_hit_gate(self, hits: int, *, soonest_resets_at=None) -> MagicMock:
        """Gate mock: detect_cap_hit=True for *hits* calls, then False (success)."""
        gate = make_gate_mock(
            detect_cap_hit=MagicMock(side_effect=[True] * hits + [False]),
        )
        gate.soonest_resets_at = soonest_resets_at
        return gate

    async def test_cap_wait_json_line_emitted_on_first_hit(self, caplog):
        """First exact-detect cap hit emits a cap_wait JSON line at WARNING level."""
        gate = self._exact_hit_gate(hits=1)

        with (
            patch(_PERIODIC_INVOKE_PATCH, new_callable=AsyncMock, return_value=_make_result()),
            patch(_PERIODIC_SLEEP_PATCH, new_callable=AsyncMock),
            patch('shared.cli_invoke.time.monotonic', return_value=0.0),
            caplog.at_level(logging.WARNING, logger='shared.cli_invoke'),
        ):
            await invoke_with_cap_retry(gate, 'my-task', prompt='hi', cap_wait_sanity_secs=None)

        cap_wait_lines = [msg for msg in caplog.messages if '"event"' in msg and 'cap_wait' in msg]
        assert cap_wait_lines, f'Expected cap_wait log line; got messages: {caplog.messages}'
        data = json.loads(cap_wait_lines[0])
        assert data['event'] == 'cap_wait'
        assert 'label' in data
        assert 'elapsed_s' in data
        assert 'soonest_open_at' in data
        assert 'next_probe_in_s' in data

    async def test_cap_wait_label_matches_label_arg(self, caplog):
        """'label' in the cap_wait line equals the label argument to invoke_with_cap_retry."""
        gate = self._exact_hit_gate(hits=1)
        label = 'task-99-architect'

        with (
            patch(_PERIODIC_INVOKE_PATCH, new_callable=AsyncMock, return_value=_make_result()),
            patch(_PERIODIC_SLEEP_PATCH, new_callable=AsyncMock),
            patch('shared.cli_invoke.time.monotonic', return_value=0.0),
            caplog.at_level(logging.WARNING, logger='shared.cli_invoke'),
        ):
            await invoke_with_cap_retry(gate, label, prompt='hi', cap_wait_sanity_secs=None)

        lines = [msg for msg in caplog.messages if '"event"' in msg and 'cap_wait' in msg]
        assert lines
        assert json.loads(lines[0])['label'] == label

    async def test_cap_wait_soonest_open_at_is_iso_of_gate_resets_at(self, caplog):
        """soonest_open_at is the ISO-8601 string of gate.soonest_resets_at when known."""
        reset_dt = datetime(2025, 6, 1, 8, 0, 0, tzinfo=UTC)
        gate = self._exact_hit_gate(hits=1, soonest_resets_at=reset_dt)

        with (
            patch(_PERIODIC_INVOKE_PATCH, new_callable=AsyncMock, return_value=_make_result()),
            patch(_PERIODIC_SLEEP_PATCH, new_callable=AsyncMock),
            patch('shared.cli_invoke.time.monotonic', return_value=0.0),
            caplog.at_level(logging.WARNING, logger='shared.cli_invoke'),
        ):
            await invoke_with_cap_retry(gate, 'task-x', prompt='hi', cap_wait_sanity_secs=None)

        lines = [msg for msg in caplog.messages if '"event"' in msg and 'cap_wait' in msg]
        assert lines
        assert json.loads(lines[0])['soonest_open_at'] == reset_dt.isoformat()

    async def test_cap_wait_soonest_open_at_null_when_unknown(self, caplog):
        """soonest_open_at is null in the JSON when gate.soonest_resets_at is None."""
        gate = self._exact_hit_gate(hits=1, soonest_resets_at=None)

        with (
            patch(_PERIODIC_INVOKE_PATCH, new_callable=AsyncMock, return_value=_make_result()),
            patch(_PERIODIC_SLEEP_PATCH, new_callable=AsyncMock),
            patch('shared.cli_invoke.time.monotonic', return_value=0.0),
            caplog.at_level(logging.WARNING, logger='shared.cli_invoke'),
        ):
            await invoke_with_cap_retry(gate, 'task-x', prompt='hi', cap_wait_sanity_secs=None)

        lines = [msg for msg in caplog.messages if '"event"' in msg and 'cap_wait' in msg]
        assert lines
        assert json.loads(lines[0])['soonest_open_at'] is None

    async def test_cap_wait_throttled_single_log_within_600s(self, caplog):
        """Three consecutive cap hits within a <600 s window emit exactly ONE log line."""
        gate = self._exact_hit_gate(hits=3)

        with (
            patch(_PERIODIC_INVOKE_PATCH, new_callable=AsyncMock, return_value=_make_result()),
            patch(_PERIODIC_SLEEP_PATCH, new_callable=AsyncMock),
            # All monotonic calls return 0.0 — time never advances, throttle stays active
            patch('shared.cli_invoke.time.monotonic', return_value=0.0),
            caplog.at_level(logging.WARNING, logger='shared.cli_invoke'),
        ):
            await invoke_with_cap_retry(gate, 'task', prompt='hi', cap_wait_sanity_secs=None)

        lines = [msg for msg in caplog.messages if '"event"' in msg and 'cap_wait' in msg]
        assert len(lines) == 1, (
            f'Expected exactly 1 cap_wait log within 600 s window, got {len(lines)}: {lines}'
        )

    async def test_cap_wait_second_log_after_600s(self, caplog):
        """A second cap_wait log is emitted after >=600 s have elapsed since the first."""
        gate = self._exact_hit_gate(hits=3)

        # Monotonic sequence: one time.monotonic() call per cap-hit iteration
        #   call 1: retry_start = 0.0
        #   call 2: hit-1 now = 0.0   → log emitted (last=0.0)
        #   call 3: hit-2 now = 300.0 → 300−0=300 < 600 → throttled
        #   call 4: hit-3 now = 700.0 → 700−0=700 >= 600 → log emitted
        monotonic_vals = itertools.chain(
            [0.0, 0.0, 300.0, 700.0],
            itertools.repeat(700.0),
        )

        with (
            patch(_PERIODIC_INVOKE_PATCH, new_callable=AsyncMock, return_value=_make_result()),
            patch(_PERIODIC_SLEEP_PATCH, new_callable=AsyncMock),
            patch('shared.cli_invoke.time.monotonic', side_effect=monotonic_vals),
            caplog.at_level(logging.WARNING, logger='shared.cli_invoke'),
        ):
            await invoke_with_cap_retry(gate, 'task', prompt='hi', cap_wait_sanity_secs=None)

        lines = [msg for msg in caplog.messages if '"event"' in msg and 'cap_wait' in msg]
        assert len(lines) == 2, (
            f'Expected 2 cap_wait logs (first at t=0, second after t=700), '
            f'got {len(lines)}: {lines}'
        )


# ── StructuredOutput schema tool must never be blocked (CLI 2.1.168) ───────
#
# CLI 2.1.168 delivers --json-schema structured output through a synthetic tool
# named ``StructuredOutput``.  A ``disallowed_tools=['*']`` wildcard (used by the
# pure-classifier curator/recon callers) now also denies that tool, so every
# structured answer is permission-denied → error_max_structured_output_retries.
# ``_invoke_claude`` must expand the ``'*'`` (only when an ``output_schema`` is
# set) into an explicit deny-list of real built-ins that OMITS StructuredOutput.


def _capture_cmd_exec(captured_cmd):
    """Fake ``create_subprocess_exec`` that records argv and returns a success."""

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

    return fake_exec


def _disallowed_segment(cmd):
    """Return the value list rendered after ``--disallowed-tools`` (up to the
    next ``--flag``)."""
    assert '--disallowed-tools' in cmd, f'no --disallowed-tools in {cmd}'
    idx = cmd.index('--disallowed-tools')
    seg = []
    for arg in cmd[idx + 1:]:
        if isinstance(arg, str) and arg.startswith('--'):
            break
        seg.append(arg)
    return seg


@pytest.mark.asyncio
class TestSchemaToolNotDisallowed:
    """The ``'*'`` deny wildcard must be expanded (excluding StructuredOutput)
    only when an output schema is requested; otherwise it is preserved verbatim.
    """

    _SCHEMA = {
        'type': 'object',
        'properties': {'answer': {'type': 'string'}},
        'required': ['answer'],
        'additionalProperties': False,
    }

    async def test_wildcard_with_schema_expands_excluding_structuredoutput(self, tmp_path):
        captured_cmd = []
        with patch('shared.cli_invoke.asyncio.create_subprocess_exec',
                   side_effect=_capture_cmd_exec(captured_cmd)):
            await invoke_claude_agent(
                prompt='hi', system_prompt='sys', cwd=tmp_path,
                disallowed_tools=['*'], output_schema=self._SCHEMA,
            )
        seg = _disallowed_segment(captured_cmd)
        # Real built-ins are denied explicitly...
        assert 'Bash' in seg
        assert 'Glob' in seg
        # ...but the wildcard and the schema tool are NOT in the deny-list,
        # and StructuredOutput must not appear anywhere on the command line.
        assert '*' not in seg
        assert 'StructuredOutput' not in captured_cmd
        # --json-schema is still rendered.
        assert '--json-schema' in captured_cmd

    async def test_wildcard_without_schema_is_preserved(self, tmp_path):
        """judge.py case: ['*'] with no output_schema must keep blocking all tools."""
        captured_cmd = []
        with patch('shared.cli_invoke.asyncio.create_subprocess_exec',
                   side_effect=_capture_cmd_exec(captured_cmd)):
            await invoke_claude_agent(
                prompt='hi', system_prompt='sys', cwd=tmp_path,
                disallowed_tools=['*'], output_schema=None,
            )
        seg = _disallowed_segment(captured_cmd)
        assert seg == ['*']
        assert '--json-schema' not in captured_cmd

    async def test_specific_list_with_schema_passes_through(self, tmp_path):
        """A specific deny-list (no '*') is rendered unchanged even with a schema."""
        captured_cmd = []
        with patch('shared.cli_invoke.asyncio.create_subprocess_exec',
                   side_effect=_capture_cmd_exec(captured_cmd)):
            await invoke_claude_agent(
                prompt='hi', system_prompt='sys', cwd=tmp_path,
                disallowed_tools=['Bash', 'Read'], output_schema=self._SCHEMA,
            )
        seg = _disallowed_segment(captured_cmd)
        assert seg == ['Bash', 'Read']


class TestSchemaToolDenied:
    """Detect the schema-tool-denied signature (CLI tool-exclusion semantics
    changed again): ``is_error`` with no structured payload and a
    ``StructuredOutput`` permission denial.  This is NOT salvaged to success —
    it sets ``schema_tool_denied`` so the curator can raise a loud escalation.
    """

    def _denied_result(self, denials):
        data = {
            'subtype': 'error_max_structured_output_retries',
            'is_error': True,
            'structured_output': None,
            'permission_denials': denials,
            'result': 'tool denied',
            'cost_usd': 0.0,
            'num_turns': 4,
        }
        return _make_subprocess_result(stdout=json.dumps(data), returncode=1)

    def test_structuredoutput_denial_sets_flag_not_salvaged(self):
        result = self._denied_result([
            {'tool_name': 'StructuredOutput',
             'tool_input': {'action': 'drop', 'justification': 'x'}},
        ])
        parsed = _parse_claude_output(result)
        assert parsed.success is False  # explicitly NOT salvaged to success
        assert parsed.schema_tool_denied is True

    def test_non_schema_denial_does_not_set_flag(self):
        result = self._denied_result([
            {'tool_name': 'Bash', 'tool_input': {'command': 'ls'}},
        ])
        parsed = _parse_claude_output(result)
        assert parsed.success is False
        assert parsed.schema_tool_denied is False


# ── _run_subprocess proc_tree capture on timeout ──────────────────────────────


@pytest.mark.asyncio
class TestRunSubprocessProcTree:
    """_run_subprocess captures a process-group snapshot in proc_tree on timeout.

    Fails RED because _SubprocessResult has no proc_tree field and
    snapshot_process_group is not yet called in the timeout handler.
    """

    @pytest.mark.timeout(15)
    async def test_proc_tree_populated_on_real_timeout(self, tmp_path):
        """A real hanging child's process group is snapshotted in proc_tree on timeout.

        Drives _run_subprocess with 'sleep 30' and a short timeout_seconds.
        Asserts result.timed_out=True AND result.proc_tree is non-empty and
        references the sleep child (by comm name 'sleep' appearing in the snapshot).
        """
        result = await _run_subprocess(
            ['sleep', '30'],
            cwd=tmp_path,
            env=dict(os.environ),
            model='opus',
            timeout_seconds=0.3,
        )
        assert result.timed_out is True
        # proc_tree must exist as a field and be non-empty
        assert result.proc_tree, (
            f'Expected non-empty proc_tree in _SubprocessResult, got: {result.proc_tree!r}'
        )
        # The snapshot should reference the sleep child
        assert 'sleep' in result.proc_tree, (
            f'Expected "sleep" in proc_tree, got: {result.proc_tree!r}'
        )


# ── _parse_claude_output proc_tree propagation ────────────────────────────────


class TestParseClaudeOutputProcTree:
    """_parse_claude_output propagates proc_tree from _SubprocessResult onto AgentResult.

    Fails RED because neither _SubprocessResult nor AgentResult has a proc_tree
    field yet.
    """

    @pytest.mark.parametrize('stdout,returncode', [
        ('', 1),
        ('not-json', 1),
        (_CLAUDE_VALID_JSON_STDOUT, 0),
    ], ids=['empty_stdout', 'json_decode_error', 'normal_parse'])
    def test_proc_tree_propagated_on_all_parse_paths(self, stdout, returncode):
        """proc_tree is copied to AgentResult on every _parse_claude_output return path."""
        tree = 'snapshot_process_group(55555): 1 process(es) in group:\n  pid=55555 ppid=1 state=S wchan=hrtimer_nanosleep comm=sleep\n'
        sub = _SubprocessResult(
            stdout=stdout, stderr='', returncode=returncode,
            duration_ms=100, timed_out=True, proc_tree=tree,
        )
        agent = _parse_claude_output(sub)
        assert agent.proc_tree == tree, (
            f'Expected proc_tree to be propagated, got: {agent.proc_tree!r}'
        )

    def test_proc_tree_defaults_empty_when_not_set(self):
        """proc_tree is empty string by default (no proc_tree on _SubprocessResult)."""
        sub = _SubprocessResult(stdout='', stderr='', returncode=1,
                                duration_ms=10, timed_out=False)
        agent = _parse_claude_output(sub)
        assert agent.proc_tree == ''


class TestIsZeroOutputTimeout:
    """Unit tests for the is_zero_output_timeout() predicate.

    This predicate captures the fresh-invocation wedge condition first observed
    in reify-4429 (2026-06-11): the CLI subprocess hangs for the full
    invocation_timeout producing zero output.  The same three-field condition
    is used by the task-1532 resume-variant wedge guard
    (invoke_with_cap_retry:644-648).
    """

    def _zero_output_result(self) -> AgentResult:
        """Build a canonical zero-output timed-out AgentResult."""
        return AgentResult(
            success=False,
            output='Agent produced no output',
            timed_out=True,
            turns=0,
            cost_usd=0.0,
            duration_ms=1_200_000,
        )

    def test_true_for_canonical_zero_output_timeout(self):
        """timed_out=True, turns=0, cost_usd=0.0 → True."""
        result = self._zero_output_result()
        assert is_zero_output_timeout(result) is True

    def test_false_when_not_timed_out(self):
        """timed_out=False makes it a normal (non-wedged) result."""
        result = self._zero_output_result()
        result.timed_out = False
        assert is_zero_output_timeout(result) is False

    def test_false_when_turns_nonzero(self):
        """turns>0 means the CLI did real agentic work — not a zero-output wedge."""
        result = self._zero_output_result()
        result.turns = 1
        assert is_zero_output_timeout(result) is False

    def test_false_when_cost_nonzero(self):
        """cost_usd>0.0 means tokens were consumed — not a zero-output wedge."""
        result = self._zero_output_result()
        result.cost_usd = 0.01
        assert is_zero_output_timeout(result) is False

    def test_false_for_successful_result(self):
        """A successful result is definitionally not a zero-output wedge."""
        result = AgentResult(
            success=True,
            output='Done!',
            timed_out=False,
            turns=5,
            cost_usd=0.25,
        )
        assert is_zero_output_timeout(result) is False

    # ── transcript_turns-driven cases (task 1778) ─────────────────────────────

    def test_transcript_turns_zero_is_true(self):
        """timed_out=True, transcript_turns=0 → True (transcript says no work)."""
        result = AgentResult(
            success=False, output='', timed_out=True,
            turns=0, cost_usd=0.0, transcript_turns=0,
        )
        assert is_zero_output_timeout(result) is True

    def test_transcript_turns_nonzero_beats_legacy_zero_defaults(self):
        """timed_out=True, transcript_turns=5, turns=0, cost_usd=0.0 → False.

        transcript_turns authoritative: work was done even though legacy
        fields show zero (reify-4415 case: 43 assistant turns, 0 JSON output).
        """
        result = AgentResult(
            success=False, output='', timed_out=True,
            turns=0, cost_usd=0.0, transcript_turns=5,
        )
        assert is_zero_output_timeout(result) is False

    def test_transcript_turns_none_legacy_fallback_zero(self):
        """timed_out=True, transcript_turns=None, turns=0, cost_usd=0.0 → True (legacy fallback)."""
        result = AgentResult(
            success=False, output='', timed_out=True,
            turns=0, cost_usd=0.0, transcript_turns=None,
        )
        assert is_zero_output_timeout(result) is True

    def test_transcript_turns_none_legacy_fallback_nonzero_turns(self):
        """timed_out=True, transcript_turns=None, turns=3 → False (legacy: turns>0)."""
        result = AgentResult(
            success=False, output='', timed_out=True,
            turns=3, cost_usd=0.0, transcript_turns=None,
        )
        assert is_zero_output_timeout(result) is False

    def test_transcript_turns_none_legacy_fallback_nonzero_cost(self):
        """timed_out=True, transcript_turns=None, cost_usd=0.01 → False (legacy: cost>0)."""
        result = AgentResult(
            success=False, output='', timed_out=True,
            turns=0, cost_usd=0.01, transcript_turns=None,
        )
        assert is_zero_output_timeout(result) is False

    def test_not_timed_out_transcript_zero(self):
        """timed_out=False, transcript_turns=0 → False (not a timeout at all)."""
        result = AgentResult(
            success=True, output='done', timed_out=False,
            turns=0, cost_usd=0.0, transcript_turns=0,
        )
        assert is_zero_output_timeout(result) is False


class TestIsTimedOutWithProgress:
    """Tests for is_timed_out_with_progress() and mutual-exclusivity invariant.

    Mutual-exclusivity invariant: when timed_out=True and transcript_turns is
    not None, exactly one of {is_zero_output_timeout, is_timed_out_with_progress}
    is True.
    """

    def test_timed_out_with_nonzero_turns_is_true(self):
        """timed_out=True, transcript_turns=5 → True."""
        result = AgentResult(
            success=False, output='', timed_out=True, transcript_turns=5,
        )
        assert is_timed_out_with_progress(result) is True

    def test_timed_out_with_zero_turns_is_false(self):
        """timed_out=True, transcript_turns=0 → False (no progress)."""
        result = AgentResult(
            success=False, output='', timed_out=True, transcript_turns=0,
        )
        assert is_timed_out_with_progress(result) is False

    def test_timed_out_with_none_turns_is_false(self):
        """timed_out=True, transcript_turns=None → False (transcript unknown)."""
        result = AgentResult(
            success=False, output='', timed_out=True, transcript_turns=None,
        )
        assert is_timed_out_with_progress(result) is False

    def test_not_timed_out_with_nonzero_turns_is_false(self):
        """timed_out=False, transcript_turns=5 → False (not a timeout)."""
        result = AgentResult(
            success=True, output='done', timed_out=False, transcript_turns=5,
        )
        assert is_timed_out_with_progress(result) is False

    def test_mutual_exclusivity_zero_turns(self):
        """timed_out=True, transcript_turns=0: exactly zero_output=True, progress=False."""
        result = AgentResult(
            success=False, output='', timed_out=True, transcript_turns=0,
        )
        zero = is_zero_output_timeout(result)
        progress = is_timed_out_with_progress(result)
        assert zero is True and progress is False
        assert zero != progress  # mutually exclusive

    def test_mutual_exclusivity_nonzero_turns(self):
        """timed_out=True, transcript_turns=5: exactly zero_output=False, progress=True."""
        result = AgentResult(
            success=False, output='', timed_out=True, transcript_turns=5,
        )
        zero = is_zero_output_timeout(result)
        progress = is_timed_out_with_progress(result)
        assert zero is False and progress is True
        assert zero != progress  # mutually exclusive


# ── _cpu_priority_prefix helper ───────────────────────────────────────────────


class TestCpuPriorityPrefix:
    """Unit tests for _cpu_priority_prefix(env) helper in cli_invoke."""

    def test_returns_nice_prefix_for_valid_nice_10(self):
        """env with DF_AGENT_CPU_NICE='10' returns ['nice', '-n', '10']."""
        env = {'DF_AGENT_CPU_NICE': '10'}
        assert _cpu_priority_prefix(env) == ['nice', '-n', '10']

    def test_returns_nice_prefix_for_nice_1(self):
        """DF_AGENT_CPU_NICE='1' (minimum valid) returns ['nice', '-n', '1']."""
        env = {'DF_AGENT_CPU_NICE': '1'}
        assert _cpu_priority_prefix(env) == ['nice', '-n', '1']

    def test_returns_nice_prefix_for_nice_19(self):
        """DF_AGENT_CPU_NICE='19' (maximum valid) returns ['nice', '-n', '19']."""
        env = {'DF_AGENT_CPU_NICE': '19'}
        assert _cpu_priority_prefix(env) == ['nice', '-n', '19']

    def test_returns_empty_for_absent_key(self):
        """Empty dict (key absent) → []."""
        assert _cpu_priority_prefix({}) == []

    def test_returns_empty_for_empty_string(self):
        """DF_AGENT_CPU_NICE='' (empty string) → []."""
        env = {'DF_AGENT_CPU_NICE': ''}
        assert _cpu_priority_prefix(env) == []

    def test_returns_empty_for_zero(self):
        """DF_AGENT_CPU_NICE='0' (not de-prioritizing) → []."""
        env = {'DF_AGENT_CPU_NICE': '0'}
        assert _cpu_priority_prefix(env) == []

    def test_returns_empty_for_negative(self):
        """DF_AGENT_CPU_NICE='-3' (needs privilege) → []."""
        env = {'DF_AGENT_CPU_NICE': '-3'}
        assert _cpu_priority_prefix(env) == []

    def test_returns_empty_for_garbage(self):
        """DF_AGENT_CPU_NICE='garbage' (malformed) → []."""
        env = {'DF_AGENT_CPU_NICE': 'garbage'}
        assert _cpu_priority_prefix(env) == []

    def test_pops_df_agent_cpu_nice_from_env(self):
        """_cpu_priority_prefix pops DF_AGENT_CPU_NICE from the env dict."""
        env = {'DF_AGENT_CPU_NICE': '10', 'OTHER': 'val'}
        _cpu_priority_prefix(env)
        assert 'DF_AGENT_CPU_NICE' not in env
        assert env.get('OTHER') == 'val'  # other keys untouched

    def test_pops_key_even_when_returning_empty(self):
        """DF_AGENT_CPU_NICE='0' is popped (returns []) — key is still removed."""
        env = {'DF_AGENT_CPU_NICE': '0'}
        _cpu_priority_prefix(env)
        assert 'DF_AGENT_CPU_NICE' not in env

    def test_nice_binary_absent_returns_empty(self):
        """Returns [] when nice is not on PATH — degrades to no-renice, never fails spawn."""
        env = {'DF_AGENT_CPU_NICE': '10'}
        with patch('shared.cli_invoke.shutil.which', return_value=None):
            result = _cpu_priority_prefix(env)
        assert result == []
        # Key is still popped so a nested invocation won't try again.
        assert 'DF_AGENT_CPU_NICE' not in env


@pytest.mark.asyncio
class TestRunSubprocessCpuPriorityPrefix:
    """Integration: _run_subprocess prepends nice prefix when DF_AGENT_CPU_NICE is set."""

    async def test_nice_prefix_prepended_to_argv(self, tmp_path):
        """_run_subprocess spawns ['nice', '-n', '10', 'claude', '--x'] when DF_AGENT_CPU_NICE='10'."""
        captured_args = []
        captured_kwargs: dict = {}

        async def fake_exec(*args, **kwargs):
            captured_args.extend(args)
            captured_kwargs.update(kwargs)
            proc = MagicMock()
            proc.communicate = AsyncMock(return_value=(b'', b''))
            proc.returncode = 0
            proc.terminate = MagicMock()
            proc.kill = MagicMock()
            proc.wait = AsyncMock()
            return proc

        with patch('shared.cli_invoke.asyncio.create_subprocess_exec', side_effect=fake_exec):
            await _run_subprocess(
                ['claude', '--x'], tmp_path, env={'DF_AGENT_CPU_NICE': '10'}, model='test',
            )

        assert captured_args[:3] == ['nice', '-n', '10'], f'Expected nice prefix; got {captured_args[:3]}'
        assert captured_args[3:5] == ['claude', '--x']
        # DF_AGENT_CPU_NICE must be stripped from the env so it does not leak
        # to the child and cannot double-renice nested invocations.
        assert 'DF_AGENT_CPU_NICE' not in captured_kwargs.get('env', {}), (
            'DF_AGENT_CPU_NICE leaked into the subprocess env'
        )

    async def test_no_prefix_without_df_agent_cpu_nice(self, tmp_path):
        """_run_subprocess spawns ['claude', '--x'] exactly when env has no DF_AGENT_CPU_NICE."""
        captured_args = []

        async def fake_exec(*args, **kwargs):
            captured_args.extend(args)
            proc = MagicMock()
            proc.communicate = AsyncMock(return_value=(b'', b''))
            proc.returncode = 0
            proc.terminate = MagicMock()
            proc.kill = MagicMock()
            proc.wait = AsyncMock()
            return proc

        with patch('shared.cli_invoke.asyncio.create_subprocess_exec', side_effect=fake_exec):
            await _run_subprocess(
                ['claude', '--x'], tmp_path, env={}, model='test',
            )

        assert captured_args == ['claude', '--x'], f'Expected bare argv; got {captured_args}'


# ── Transcript readers ────────────────────────────────────────────────────────


class TestCountTranscriptTurns:
    """Unit tests for count_transcript_turns(config_dir, session_id).

    The function reads a JSONL transcript file named <session_id>.jsonl under
    <config_dir>/projects/*/ and counts records with type=='assistant'.
    It is tolerant: truncated/unparseable lines are skipped (not None).
    None is returned only when the file cannot be located or the whole read
    raises catastrophically.
    """

    def _write_transcript(self, base: Path, session_id: str, lines: list[str]) -> Path:
        """Write a JSONL transcript under base/projects/<slug>/<session_id>.jsonl."""
        slug_dir = base / 'projects' / 'myproject'
        slug_dir.mkdir(parents=True, exist_ok=True)
        transcript = slug_dir / f'{session_id}.jsonl'
        transcript.write_text('\n'.join(lines) + '\n')
        return transcript

    def test_counts_assistant_records(self, tmp_path):
        """3 assistant records interleaved with user/system → returns 3."""
        sid = 'session-abc-001'
        lines = [
            json.dumps({'type': 'system', 'content': 'hello'}),
            json.dumps({'type': 'assistant', 'content': 'turn 1'}),
            json.dumps({'type': 'user', 'content': 'prompt 2'}),
            json.dumps({'type': 'assistant', 'content': 'turn 2'}),
            json.dumps({'type': 'user', 'content': 'prompt 3'}),
            json.dumps({'type': 'assistant', 'content': 'turn 3'}),
        ]
        self._write_transcript(tmp_path, sid, lines)
        result = count_transcript_turns(
            config_dir=tmp_path, session_id=sid
        )
        assert result == 3

    def test_truncated_final_line_skipped_tolerantly(self, tmp_path):
        """2 complete assistant records + truncated final line → returns 2 (not None)."""
        sid = 'session-abc-002'
        lines = [
            json.dumps({'type': 'assistant', 'content': 'turn 1'}),
            json.dumps({'type': 'user', 'content': 'prompt'}),
            json.dumps({'type': 'assistant', 'content': 'turn 2'}),
            '{"type": "assistant", "content": "trunc',  # truncated / unparseable
        ]
        self._write_transcript(tmp_path, sid, lines)
        result = count_transcript_turns(
            config_dir=tmp_path, session_id=sid
        )
        assert result == 2

    def test_absent_session_returns_none(self, tmp_path):
        """No matching transcript file for the session id → None."""
        sid = 'session-does-not-exist'
        # Create projects dir but no matching file
        (tmp_path / 'projects' / 'myproject').mkdir(parents=True, exist_ok=True)
        result = count_transcript_turns(
            config_dir=tmp_path, session_id=sid
        )
        assert result is None


class TestReadTranscriptRecords:
    """Unit tests for read_transcript_records(config_dir, session_id).

    The function reads a JSONL transcript file and returns ALL parsed records as
    a list of dicts, preserving order.  Tolerant: unparseable lines are skipped,
    not None.  None is returned only when the file cannot be located or the
    whole read raises.
    """

    def _write_transcript(self, base: Path, session_id: str, lines: list[str]) -> Path:
        slug_dir = base / 'projects' / 'myproject'
        slug_dir.mkdir(parents=True, exist_ok=True)
        transcript = slug_dir / f'{session_id}.jsonl'
        transcript.write_text('\n'.join(lines) + '\n')
        return transcript

    def test_returns_all_records_in_order(self, tmp_path):
        """4 well-formed records (mixed types) → list of 4 dicts in order."""
        sid = 'sess-read-001'
        records_in = [
            {'type': 'system', 'content': 'init'},
            {'type': 'user', 'content': 'prompt'},
            {'type': 'assistant', 'content': [{'type': 'tool_use', 'name': 'Bash', 'input': {}}]},
            {'type': 'tool', 'content': 'result'},
        ]
        lines = [json.dumps(r) for r in records_in]
        self._write_transcript(tmp_path, sid, lines)
        result = read_transcript_records(
            config_dir=tmp_path, session_id=sid
        )
        assert isinstance(result, list)
        assert len(result) == 4
        # Verify order and content
        for i, rec in enumerate(records_in):
            assert result[i] == rec

    def test_truncated_final_line_skipped(self, tmp_path):
        """Truncated final line skipped; complete records returned."""
        sid = 'sess-read-002'
        complete = [
            {'type': 'user', 'content': 'hi'},
            {'type': 'assistant', 'content': 'hello'},
        ]
        lines = [json.dumps(r) for r in complete]
        lines.append('{"type": "assistant", "content": "trunc')  # truncated
        self._write_transcript(tmp_path, sid, lines)
        result = read_transcript_records(
            config_dir=tmp_path, session_id=sid
        )
        assert result == complete

    def test_absent_file_returns_none(self, tmp_path):
        """No matching transcript file → None."""
        sid = 'sess-absent-xyz'
        (tmp_path / 'projects' / 'proj').mkdir(parents=True, exist_ok=True)
        result = read_transcript_records(
            config_dir=tmp_path, session_id=sid
        )
        assert result is None


# ── Two-regime liveness watchdog tests ──────────────────────────────────────

@pytest.mark.asyncio
class TestRunSubprocessWatchdog:
    """Two-regime liveness watchdog tests for _run_subprocess.

    These tests verify that the new watchdog param ``startup_grace_secs``
    enables a fast pre-turn-1 kill when a zero-turn startup wedge is detected,
    distinct from the per-role post-turn-1 ceiling.

    The key fake communicate pattern used throughout:
      - First call: hangs via ``asyncio.Event().wait()`` until cancelled by the
        watchdog kill path (raises CancelledError on cancellation).
      - Second call (post-SIGTERM): raises TimeoutError immediately to trigger
        the SIGKILL branch so ``terminate_process_group`` is called.
    """

    @staticmethod
    def _make_hanging_proc():
        """Return (proc, call_count_ref) where communicate hangs on call 1, raises TimeoutError on call 2."""
        call_count = [0]

        async def communicate_side_effect(input=None):  # noqa: A002
            call_count[0] += 1
            if call_count[0] == 1:
                # Hang until the watchdog cancels comm_task — CancelledError is raised here.
                await asyncio.Event().wait()
            # Second call (inside SIGTERM grace window): raise immediately → SIGKILL path.
            raise TimeoutError

        proc = MagicMock()
        proc.communicate = communicate_side_effect
        proc.terminate = MagicMock()
        proc.kill = MagicMock()
        proc.wait = AsyncMock()
        proc.returncode = None
        proc.pid = 12345
        return proc, call_count

    async def test_startup_wedge_killed_at_grace_not_ceiling(self, tmp_path):
        """Startup wedge (0 turns) is killed at startup_grace_secs, not the 5s ceiling.

        With startup_grace_secs=0.05 and timeout_seconds=5.0, the watchdog
        should detect 0 turns after ~0.05s and kill fast.  Wall-clock must be
        well under the 5s ceiling, proving the kill happened at the grace bound.
        """
        import time as _time

        sid = str(uuid.uuid4())
        cfg_dir = tmp_path / 'cfg'
        cfg_dir.mkdir()

        proc, _ = self._make_hanging_proc()
        terminate_pg_mock = AsyncMock()

        async def fake_exec(*args, **kwargs):
            return proc

        with (
            patch('shared.cli_invoke.asyncio.create_subprocess_exec', side_effect=fake_exec),
            patch('shared.cli_invoke.terminate_process_group', terminate_pg_mock),
            patch('shared.cli_invoke.count_transcript_turns', return_value=0),
        ):
            t0 = _time.monotonic()
            result = await _run_subprocess(
                ['fake'], cwd=tmp_path, env={}, model='opus',
                timeout_seconds=5.0, startup_grace_secs=0.05,
                session_id=sid, config_dir=cfg_dir,
            )
            wall = _time.monotonic() - t0

        assert result.timed_out is True, 'Expected timed_out=True for startup wedge kill'
        terminate_pg_mock.assert_called_once()
        assert wall < 1.0, f'Expected fast kill (<1s), got {wall:.3f}s — wedge not killed at grace bound'

    async def test_none_transcript_degrades_to_ceiling_not_grace(self, tmp_path):
        """B7 conservative degrade: unreadable/absent transcript (None) must NOT trigger
        the startup-grace fast kill.

        When count_transcript_turns returns None, the watchdog cannot prove a wedge.
        The run must survive past startup_grace_secs and be killed only at the ceiling
        (timeout_seconds).  Wall-clock must be >= ~0.25s (past the 0.05s grace).

        This is the step-5 RED case: the step-4 first-cut guard `not seen_turn` is
        too broad — it also early-kills on None (unreadable transcript), which is
        wrong.
        """
        import time as _time

        sid = str(uuid.uuid4())
        cfg_dir = tmp_path / 'cfg'
        cfg_dir.mkdir()

        proc, _ = self._make_hanging_proc()
        terminate_pg_mock = AsyncMock()

        async def fake_exec(*args, **kwargs):
            return proc

        with (
            patch('shared.cli_invoke.asyncio.create_subprocess_exec', side_effect=fake_exec),
            patch('shared.cli_invoke.terminate_process_group', terminate_pg_mock),
            patch('shared.cli_invoke.count_transcript_turns', return_value=None),
        ):
            t0 = _time.monotonic()
            result = await _run_subprocess(
                ['fake'], cwd=tmp_path, env={}, model='opus',
                timeout_seconds=0.3, startup_grace_secs=0.05,
                session_id=sid, config_dir=cfg_dir,
            )
            wall = _time.monotonic() - t0

        assert result.timed_out is True
        assert result.transcript_turns is None
        # Must NOT have been killed at the 0.05s grace bound — must reach the 0.3s ceiling.
        assert wall >= 0.2, (
            f'Expected kill at ceiling (~0.3s), but killed early at {wall:.3f}s '
            f'— None transcript should degrade to ceiling, not trigger fast kill'
        )

    async def test_working_regime_survives_grace_killed_at_ceiling(self, tmp_path):
        """B6 long-synchronous-tool survival: ≥1 turn seen → no fast kill at grace, only ceiling.

        A proc that has made progress (count_transcript_turns=5) must NOT be killed
        at the startup_grace_secs bound.  Liveness is proven (seen_turn=True); the
        working regime applies and only the absolute ceiling triggers the kill.
        Wall-clock must be >= ~0.25s (past the 0.05s grace) and result.transcript_turns==5.
        """
        import time as _time

        sid = str(uuid.uuid4())
        cfg_dir = tmp_path / 'cfg'
        cfg_dir.mkdir()

        proc, _ = self._make_hanging_proc()
        terminate_pg_mock = AsyncMock()

        async def fake_exec(*args, **kwargs):
            return proc

        with (
            patch('shared.cli_invoke.asyncio.create_subprocess_exec', side_effect=fake_exec),
            patch('shared.cli_invoke.terminate_process_group', terminate_pg_mock),
            patch('shared.cli_invoke.count_transcript_turns', return_value=5),
        ):
            t0 = _time.monotonic()
            result = await _run_subprocess(
                ['fake'], cwd=tmp_path, env={}, model='opus',
                timeout_seconds=0.3, startup_grace_secs=0.05,
                session_id=sid, config_dir=cfg_dir,
            )
            wall = _time.monotonic() - t0

        assert result.timed_out is True
        assert result.transcript_turns == 5
        # Must NOT have been killed at the 0.05s grace bound — must reach the 0.3s ceiling.
        assert wall >= 0.2, (
            f'Expected kill at ceiling (~0.3s), but killed early at {wall:.3f}s '
            f'— seen_turn=True (5 turns) should prevent startup-grace fast kill'
        )

    async def test_none_transcript_post_grace_does_not_busy_loop(self, tmp_path):
        """Regression: post-grace tight-spin when transcript is unreadable (None).

        After startup_grace_secs expires with live_turns=None (unreadable transcript),
        the poll must NOT degenerate to 0.0 (causing a tight-spin that hammers
        count_transcript_turns hundreds/thousands of times).

        Asserts that count_transcript_turns is called at most 20 times across the
        full 0.4s window (~8 calls expected at the 0.05s poll cadence).

        Fails before step-8: once elapsed >= startup_grace_secs with live_turns=None,
        time_to_grace = max(0.0, grace - elapsed) = 0.0, so
        poll = min(_WATCHDOG_POLL_SECS=0.05, 0.0, time_to_ceiling) = 0.0;
        asyncio.wait(timeout=0.0) returns immediately and the loop tight-spins,
        calling count_transcript_turns hundreds/thousands of times in the 0.4s window.
        """
        sid = str(uuid.uuid4())
        cfg_dir = tmp_path / 'cfg'
        cfg_dir.mkdir()

        proc, _ = self._make_hanging_proc()
        terminate_pg_mock = AsyncMock()

        async def fake_exec(*args, **kwargs):
            return proc

        call_counter = [0]

        def counting_count_turns(config_dir, session_id):
            call_counter[0] += 1
            return None

        with (
            patch('shared.cli_invoke.asyncio.create_subprocess_exec', side_effect=fake_exec),
            patch('shared.cli_invoke.terminate_process_group', terminate_pg_mock),
            patch('shared.cli_invoke.count_transcript_turns', side_effect=counting_count_turns),
            patch('shared.cli_invoke._WATCHDOG_POLL_SECS', 0.05),
        ):
            result = await _run_subprocess(
                ['fake'], cwd=tmp_path, env={}, model='opus',
                timeout_seconds=0.4, startup_grace_secs=0.05,
                session_id=sid, config_dir=cfg_dir,
            )

        assert result.timed_out is True, 'Expected timed_out=True (killed at ceiling)'
        assert call_counter[0] <= 20, (
            f'count_transcript_turns called {call_counter[0]} times — '
            f'busy-loop detected; expected ≤20 (bounded by ~0.05s poll cadence)'
        )
