"""Tests for invoke_with_cap_retry in orchestrator/agents/invoke.py."""

from __future__ import annotations

import asyncio
import contextlib
import json
from unittest.mock import AsyncMock, MagicMock, patch

import pytest
from shared.cli_invoke import CAP_HIT_RESUME_PROMPT, AgentResult
from shared.usage_gate import InvokeSlot

from orchestrator.agents.invoke import (
    _invoke_claude_with_sandbox,
    _invoke_codex,
    _invoke_gemini,
    _parse_codex_output,
    _parse_gemini_output,
    _run_subprocess_local,
    _SubprocessResult,
    invoke_agent,
    invoke_with_cap_retry,
)

# Cap-retry path is now the unified shared loop; orchestrator callers
# pass ``invoke_fn=invoke_agent`` to route through multi-backend dispatch.
# Tests below patch that same function or pass an AsyncMock as ``invoke_fn``.
_INVOKE_AGENT_PATCH = 'orchestrator.agents.invoke.invoke_agent'
_SHARED_ASYNCIO_PATCH = 'shared.cli_invoke.asyncio'


def _attach_invoke_slot(gate: MagicMock) -> MagicMock:
    """Install a real async-CM on gate.invoke_slot() that wires a real InvokeSlot.

    Mirrors ``UsageGate.invoke_slot`` so tests that assert gate-level
    mock calls (``gate.release_probe_slot(token)`` via ``__aexit__`` safety
    net, ``gate.confirm_account_ok(token)``/``gate.on_agent_complete(cost)``
    via ``slot.confirm``) exercise the real delegation.  Complements
    ``_make_slot``/``_make_gate_yielding`` which wrap fully-mocked slots
    for iteration-level control.
    """
    @contextlib.asynccontextmanager
    async def _cm():
        token = await gate.before_invoke()
        slot = InvokeSlot(gate, token)
        try:
            yield slot
        finally:
            if not slot._settled:
                gate.release_probe_slot(token)

    gate.invoke_slot = _cm
    return gate


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


def _make_slot(*, token='token-a', account_name='acct-a', cap_hit=False):
    """MagicMock shaped like shared.usage_gate.InvokeSlot.

    Production in invoke_with_cap_retry reads slot.token / slot.account_name /
    slot.detect_cap_hit / slot.confirm — all mockable via this helper.  Each
    slot represents ONE iteration of the cap-retry loop.
    """
    slot = MagicMock()
    slot.token = token
    slot.account_name = account_name
    slot.detect_cap_hit = MagicMock(return_value=cap_hit)
    slot.confirm = MagicMock()
    slot.settle = MagicMock()
    return slot


def _make_gate_yielding(slots, *, active_account_name=None):
    """UsageGate mock whose successive invoke_slot() calls yield the given slots.

    Without this helper, `gate = MagicMock()` yields an unconstrained slot whose
    detect_cap_hit returns a truthy coroutine — production's `while` loop never
    exits and the test hangs until pytest-timeout fires.
    """
    slot_iter = iter(slots)

    def _new_cm(*args, **kwargs):
        slot = next(slot_iter)
        cm = MagicMock()
        cm.__aenter__ = AsyncMock(return_value=slot)
        cm.__aexit__ = AsyncMock(return_value=False)
        return cm

    gate = MagicMock()
    gate.invoke_slot = MagicMock(side_effect=_new_cm)
    gate.account_count = len(slots)
    gate.active_account_name = (
        active_account_name if active_account_name is not None
        else slots[0].account_name
    )
    gate.before_invoke = AsyncMock(return_value=slots[0].token)
    gate.on_agent_complete = MagicMock()
    gate.confirm_account_ok = MagicMock()
    gate.release_probe_slot = MagicMock()
    return gate


@pytest.mark.asyncio
class TestAccountNameThreading:

    async def test_account_name_set_from_usage_gate(self):
        """account_name is stamped from slot.account_name on success."""
        gate = _make_gate_yielding([_make_slot(account_name='acct-a', cap_hit=False)])
        fake_invoke = AsyncMock(return_value=_make_result())

        got = await invoke_with_cap_retry(
            gate, 'test-label', invoke_fn=fake_invoke,
            prompt='hi', system_prompt='sys', cwd='/tmp',
        )

        assert got.account_name == 'acct-a'

    async def test_account_name_none_coerced_to_empty(self):
        """When slot.account_name is '', result.account_name is ''."""
        gate = _make_gate_yielding([_make_slot(account_name='', cap_hit=False)])
        fake_invoke = AsyncMock(return_value=_make_result())

        got = await invoke_with_cap_retry(
            gate, 'test-label', invoke_fn=fake_invoke,
            prompt='hi', system_prompt='sys', cwd='/tmp',
        )

        assert got.account_name == ''

    async def test_account_name_reflects_failover_account(self):
        """After cap hit + failover, account_name reflects the retry account.

        Production reads slot.account_name (per iteration) — not
        gate.active_account_name — so per-iteration names come from the
        per-slot configuration.
        """
        gate = _make_gate_yielding([
            _make_slot(account_name='acct-a', cap_hit=True),
            _make_slot(account_name='acct-b', cap_hit=False),
        ])
        fake_invoke = AsyncMock(return_value=_make_result())

        with patch(_SHARED_ASYNCIO_PATCH) as mock_asyncio:
            mock_asyncio.sleep = AsyncMock()
            got = await invoke_with_cap_retry(
                gate, 'test-label', invoke_fn=fake_invoke,
                prompt='hi', system_prompt='sys', cwd='/tmp',
            )

        assert got.account_name == 'acct-b'


@pytest.mark.asyncio
class TestAccountNameNoGate:

    async def test_account_name_empty_without_gate(self):
        """When usage_gate=None, result.account_name is ''."""
        fake_invoke = AsyncMock(return_value=_make_result())
        got = await invoke_with_cap_retry(
            None, 'test-label', invoke_fn=fake_invoke,
            prompt='hi', system_prompt='sys', cwd='/tmp',
        )

        assert got.account_name == ''


@pytest.mark.asyncio
class TestCapHitResume:

    async def test_resume_on_cap_hit_claude_backend(self):
        """Claude backend cap hit with session_id → resume on retry."""
        gate = _make_gate_yielding([
            _make_slot(token='token-a', account_name='acct-a', cap_hit=True),
            _make_slot(token='token-b', account_name='acct-b', cap_hit=False),
        ])

        capped_result = _make_result(
            success=True, cost_usd=0.5, session_id='sess-abc',
        )
        capped_result.duration_ms = 6000
        capped_result.turns = 2
        ok_result = _make_result(success=True, cost_usd=0.5)
        ok_result.duration_ms = 6000
        ok_result.turns = 2

        fake_invoke = AsyncMock(side_effect=[capped_result, ok_result])

        with patch(_SHARED_ASYNCIO_PATCH) as mock_asyncio:
            mock_asyncio.sleep = AsyncMock()
            await invoke_with_cap_retry(
                gate, 'test-label', invoke_fn=fake_invoke,
                prompt='hi', system_prompt='sys', cwd='/tmp', backend='claude',
            )

            second_call = fake_invoke.call_args_list[1]
            assert second_call.kwargs.get('resume_session_id') == 'sess-abc'
            assert second_call.kwargs.get('prompt') == CAP_HIT_RESUME_PROMPT

    async def test_no_resume_on_cap_hit_codex_backend(self):
        """Codex backend cap hit → no resume_session_id (not supported).

        The unified loop resumes by session_id when the result has one, regardless
        of backend.  Codex backends produce no session_id, so no resume happens.
        """
        gate = _make_gate_yielding([
            _make_slot(token='token-a', account_name='acct-a', cap_hit=True),
            _make_slot(token='token-b', account_name='acct-b', cap_hit=False),
        ])

        capped_result = _make_result(session_id='')  # codex: no session
        ok_result = _make_result()
        ok_result.duration_ms = 6000
        ok_result.turns = 2
        fake_invoke = AsyncMock(side_effect=[capped_result, ok_result])

        with patch(_SHARED_ASYNCIO_PATCH) as mock_asyncio:
            mock_asyncio.sleep = AsyncMock()
            await invoke_with_cap_retry(
                gate, 'test-label', invoke_fn=fake_invoke,
                prompt='hi', system_prompt='sys', cwd='/tmp', backend='codex',
            )

            second_call = fake_invoke.call_args_list[1]
            assert 'resume_session_id' not in second_call.kwargs
            assert second_call.kwargs.get('prompt') == 'hi'

    async def test_resume_failure_falls_back_to_fresh(self):
        """Resume returns success=False → retry with original prompt."""
        gate = _make_gate_yielding([
            _make_slot(token='token-a', account_name='acct-a', cap_hit=True),
            _make_slot(token='token-b', account_name='acct-b', cap_hit=False),
            _make_slot(token='token-c', account_name='acct-b', cap_hit=False),
        ])

        capped_result = _make_result(session_id='sess-abc')
        failed_resume = _make_result(success=False)
        failed_resume.duration_ms = 6000
        failed_resume.turns = 2
        ok_result = _make_result(success=True, cost_usd=0.5)
        ok_result.duration_ms = 6000
        ok_result.turns = 2
        fake_invoke = AsyncMock(side_effect=[capped_result, failed_resume, ok_result])

        with patch(_SHARED_ASYNCIO_PATCH) as mock_asyncio:
            mock_asyncio.sleep = AsyncMock()
            got = await invoke_with_cap_retry(
                gate, 'test-label', invoke_fn=fake_invoke,
                prompt='original', system_prompt='sys', cwd='/tmp', backend='claude',
            )

            assert fake_invoke.call_count == 3
            third_call = fake_invoke.call_args_list[2]
            assert 'resume_session_id' not in third_call.kwargs
            assert third_call.kwargs.get('prompt') == 'original'
            assert got.success is True


# ── _run_subprocess_local timed_out, _parse_codex_output, _parse_gemini_output ─


@pytest.mark.asyncio
class TestRunSubprocessLocalTimedOut:

    async def test_run_subprocess_local_sets_timed_out_on_timeout(self, tmp_path):
        """_run_subprocess_local propagates timed_out=True when TimeoutError fires.

        This test only verifies that the timed_out flag and error message are
        set correctly on the returned SubprocessResult.  It does not assert
        that terminate_process_group is called — that is covered by a separate
        test that patches the helper directly.
        """
        proc = MagicMock()
        proc.pid = 12345  # int required so pgid safety-check doesn't TypeError
        proc.communicate = AsyncMock(side_effect=TimeoutError)
        proc.wait = AsyncMock()
        proc.returncode = None

        async def fake_exec(*args, **kwargs):
            return proc

        with (
            patch('orchestrator.agents.invoke.asyncio.create_subprocess_exec',
                  side_effect=fake_exec),
            patch('orchestrator.agents.invoke.terminate_process_group',
                  new_callable=AsyncMock),
        ):
            result = await _run_subprocess_local(
                ['fake'], cwd=tmp_path, env={}, backend='codex', model='gpt-5.4',
                max_budget_usd=1.0, timeout_seconds=0.1,
            )

        assert result.timed_out is True
        assert 'Process killed after' in result.stderr and 'timeout' in result.stderr
        assert result.returncode == 1

    async def test_run_subprocess_local_timeout_calls_terminate_process_group(self, tmp_path):
        """terminate_process_group is awaited (not proc.kill) when TimeoutError fires."""
        proc = MagicMock()
        proc.pid = 12345  # int so pgid capture and safety-check pass
        proc.communicate = AsyncMock(side_effect=TimeoutError)
        proc.wait = AsyncMock()
        proc.returncode = None

        async def fake_exec(*args, **kwargs):
            return proc

        with (
            patch('orchestrator.agents.invoke.asyncio.create_subprocess_exec',
                  side_effect=fake_exec),
            patch('orchestrator.agents.invoke.terminate_process_group',
                  new_callable=AsyncMock) as mock_tpg,
        ):
            result = await _run_subprocess_local(
                ['fake'], cwd=tmp_path, env={}, backend='codex', model='gpt-5.4',
                max_budget_usd=1.0, timeout_seconds=0.1,
            )

        mock_tpg.assert_awaited_once()
        # First positional arg must be the proc, second must be the captured pgid
        call_args = mock_tpg.call_args
        assert call_args.args[0] is proc
        assert call_args.args[1] == 12345
        assert result.timed_out is True


_CODEX_VALID_JSONL_STDOUT = (
    json.dumps({'type': 'thread.started', 'thread_id': 'tid-1'}) + '\n'
    + json.dumps({'type': 'item.completed', 'item': {'type': 'agent_message', 'text': 'hello'}}) + '\n'
    + json.dumps({'type': 'turn.completed', 'usage': {'input_tokens': 10, 'output_tokens': 5}}) + '\n'
)


class TestParseCodexOutputPropagatesTimedOut:
    """Parser always sets timed_out — callers no longer need to patch it post-hoc."""

    @pytest.mark.parametrize('input_timed_out', [True, False])
    @pytest.mark.parametrize(
        'stdout,stderr,returncode',
        [
            ('', 'timeout', 1),
            ('not json at all', '', 1),
            (_CODEX_VALID_JSONL_STDOUT, '', 0),
        ],
        ids=['empty_stdout', 'json_decode_error', 'normal_parse'],
    )
    def test_propagates_timed_out(self, stdout, stderr, returncode, input_timed_out):
        """_parse_codex_output propagates timed_out from the subprocess result."""
        sub = _SubprocessResult(stdout=stdout, stderr=stderr, returncode=returncode,
                                duration_ms=100, timed_out=input_timed_out)
        agent = _parse_codex_output(sub, 'gpt-5.4')
        assert agent.timed_out is input_timed_out


_GEMINI_VALID_JSON_STDOUT = json.dumps({'response': 'hi', 'stats': {'input_tokens': 10, 'output_tokens': 5}})


class TestParseGeminiOutputPropagatesTimedOut:
    """Parser always sets timed_out — callers no longer need to patch it post-hoc."""

    @pytest.mark.parametrize('input_timed_out', [True, False])
    @pytest.mark.parametrize(
        'stdout,stderr,returncode',
        [
            ('', 'timeout', 1),
            ('not json', '', 1),
            (_GEMINI_VALID_JSON_STDOUT, '', 0),
        ],
        ids=['empty_stdout', 'json_decode_error', 'normal_parse'],
    )
    def test_propagates_timed_out(self, stdout, stderr, returncode, input_timed_out):
        """_parse_gemini_output propagates timed_out from the subprocess result."""
        sub = _SubprocessResult(stdout=stdout, stderr=stderr, returncode=returncode,
                                duration_ms=100, timed_out=input_timed_out)
        agent = _parse_gemini_output(sub, 'gemini-3.1-pro-preview')
        assert agent.timed_out is input_timed_out


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
        """release_probe_slot is called with oauth_token when invoke raises."""
        gate = MagicMock()
        gate.account_count = 1
        gate.before_invoke = AsyncMock(return_value='tok-a')
        gate.detect_cap_hit = MagicMock(return_value=False)
        gate.active_account_name = 'acct-a'
        gate.on_agent_complete = MagicMock()
        gate.confirm_account_ok = MagicMock()
        gate.release_probe_slot = MagicMock()
        _attach_invoke_slot(gate)

        fake_invoke = AsyncMock(side_effect=RuntimeError('subprocess failed'))

        with pytest.raises(RuntimeError, match='subprocess failed'):
            await invoke_with_cap_retry(
                gate, 'lbl', invoke_fn=fake_invoke,
                prompt='hi', system_prompt='sys', cwd='/tmp',
            )

        gate.release_probe_slot.assert_called_once_with('tok-a')

    async def test_runtime_error_propagates(self):
        """RuntimeError raised by invoke propagates with its message intact."""
        gate = MagicMock()
        gate.account_count = 1
        gate.before_invoke = AsyncMock(return_value='tok-a')
        gate.active_account_name = 'acct-a'
        gate.release_probe_slot = MagicMock()
        _attach_invoke_slot(gate)

        fake_invoke = AsyncMock(side_effect=RuntimeError('crash'))
        with pytest.raises(RuntimeError) as exc_info:
            await invoke_with_cap_retry(
                gate, 'lbl', invoke_fn=fake_invoke,
                prompt='hi', system_prompt='sys', cwd='/tmp',
            )

        assert str(exc_info.value) == 'crash'  # error message preserved verbatim

    async def test_confirm_account_ok_not_called_when_invoke_raises(self):
        """confirm_account_ok is NOT called when invoke raises."""
        gate = MagicMock()
        gate.account_count = 1
        gate.before_invoke = AsyncMock(return_value='tok-a')
        gate.active_account_name = 'acct-a'
        gate.confirm_account_ok = MagicMock()
        gate.release_probe_slot = MagicMock()
        _attach_invoke_slot(gate)

        fake_invoke = AsyncMock(side_effect=RuntimeError('crash'))
        with pytest.raises(RuntimeError):
            await invoke_with_cap_retry(
                gate, 'lbl', invoke_fn=fake_invoke,
                prompt='hi', system_prompt='sys', cwd='/tmp',
            )

        gate.confirm_account_ok.assert_not_called()

    async def test_cancelled_error_release_probe_slot(self):
        """CancelledError (BaseException, not Exception) triggers release_probe_slot."""
        gate = MagicMock()
        gate.account_count = 1
        gate.before_invoke = AsyncMock(return_value='tok-a')
        gate.active_account_name = 'acct-a'
        gate.release_probe_slot = MagicMock()
        _attach_invoke_slot(gate)

        fake_invoke = AsyncMock(side_effect=asyncio.CancelledError())
        with pytest.raises(asyncio.CancelledError):
            await invoke_with_cap_retry(
                gate, 'lbl', invoke_fn=fake_invoke,
                prompt='hi', system_prompt='sys', cwd='/tmp',
            )

        gate.release_probe_slot.assert_called_once_with('tok-a')


# ── _run_subprocess_local process-group fix ──────────────────────────────────


@pytest.mark.asyncio
class TestRunSubprocessLocalProcessGroup:
    """_run_subprocess_local must spawn subprocesses in their own process group."""

    async def test_run_subprocess_local_passes_start_new_session_true(self, tmp_path):
        """create_subprocess_exec is called with start_new_session=True.

        Failing test -- _run_subprocess_local does not pass that kwarg yet.
        """
        captured_kwargs: dict = {}

        async def fake_exec(*args, **kwargs):
            captured_kwargs.update(kwargs)
            proc = MagicMock()
            proc.communicate = AsyncMock(return_value=(b'', b''))
            proc.returncode = 0
            proc.kill = MagicMock()
            proc.wait = AsyncMock()
            return proc

        with patch('orchestrator.agents.invoke.asyncio.create_subprocess_exec',
                   side_effect=fake_exec):
            await _run_subprocess_local(
                ['echo', 'hi'], cwd=tmp_path, env={},
                backend='codex', model='gpt-5.4', max_budget_usd=1.0,
            )

        assert captured_kwargs.get('start_new_session') is True
