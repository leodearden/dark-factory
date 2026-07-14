"""Tests for invoke_with_cap_retry in orchestrator/agents/invoke.py."""

from __future__ import annotations

import asyncio
import contextlib
import json
import logging
from unittest.mock import AsyncMock, MagicMock, patch

import pytest
from _orch_helpers import make_gate_yielding as _make_gate_yielding  # centralized (task 1458)
from _orch_helpers import make_mock_gate as _make_gate  # centralized factory (task 1458)
from shared.cli_invoke import CAP_HIT_RESUME_PROMPT, AgentResult
from shared.usage_gate import AccountLease, InvokeSlot

from orchestrator.agents.invoke import (
    _invoke_claude_with_sandbox,
    _invoke_codex,
    _invoke_gemini,
    _invoke_pi,
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
        # gate.before_invoke() is mocked to return a bare token (task W4-δ
        # changed the real UsageGate.before_invoke to return an AccountLease
        # instead) — wrap it the same way the real invoke_slot() would have
        # built it, using the mock's active_account_name for the lease name,
        # so the real InvokeSlot's lease-derived token/account_name
        # properties see the same values this helper's callers configured.
        lease = token if isinstance(token, AccountLease) else AccountLease(
            name=gate.active_account_name or '', token=token, generation=0,
        )
        slot = InvokeSlot(gate, lease)
        try:
            yield slot
        finally:
            if not slot._settled:
                gate.release_probe_slot(lease.token if lease is not None else None)

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


@pytest.mark.asyncio
class TestPricesThreadedToParser:
    """`prices` must reach both _parse_codex_output and _parse_gemini_output,
    from the backend-specific _invoke_* function directly and via
    invoke_agent(backend=...) (task 2459).

    Uses a deliberately non-seed rate so a passing cost proves the passed-in
    map — not the packaged default_price_table() — was used.
    """

    async def test_invoke_codex_forwards_prices_to_parser(self, tmp_path):
        """_invoke_codex(prices=...) reaches _parse_codex_output's cost calc."""
        codex_result = _SubprocessResult(
            stdout=_CODEX_VALID_JSONL_STDOUT, stderr='', returncode=0, duration_ms=100,
        )
        with patch('orchestrator.agents.invoke._run_subprocess_local',
                   new_callable=AsyncMock, return_value=codex_result):
            agent = await _invoke_codex(
                prompt='hello', system_prompt='sys', cwd=tmp_path,
                model='o4-mini', max_budget_usd=1.0,
                mcp_config=None, sandbox_modules=None, effort=None,
                timeout_seconds=30.0,
                prices={'o4-mini': {'input_per_1m': 100.0, 'output_per_1m': 100.0}},
            )
        assert agent.cost_usd == pytest.approx((10 * 100.0 + 5 * 100.0) / 1_000_000)

    async def test_invoke_agent_forwards_prices_to_invoke_codex(self, tmp_path):
        """invoke_agent(backend='codex', prices=...) forwards through to _invoke_codex."""
        codex_result = _SubprocessResult(
            stdout=_CODEX_VALID_JSONL_STDOUT, stderr='', returncode=0, duration_ms=100,
        )
        with patch('orchestrator.agents.invoke._run_subprocess_local',
                   new_callable=AsyncMock, return_value=codex_result):
            agent = await invoke_agent(
                prompt='hello', system_prompt='sys', cwd=tmp_path,
                backend='codex', model='o4-mini', max_budget_usd=1.0,
                sandbox_modules=None, effort=None, timeout_seconds=30.0,
                prices={'o4-mini': {'input_per_1m': 100.0, 'output_per_1m': 100.0}},
            )
        assert agent.cost_usd == pytest.approx((10 * 100.0 + 5 * 100.0) / 1_000_000)

    async def test_invoke_gemini_forwards_prices_to_parser(self, tmp_path):
        """_invoke_gemini(prices=...) reaches _parse_gemini_output's cost calc."""
        gemini_result = _SubprocessResult(
            stdout=_GEMINI_VALID_JSON_STDOUT, stderr='', returncode=0, duration_ms=100,
        )
        with patch('orchestrator.agents.invoke._run_subprocess_local',
                   new_callable=AsyncMock, return_value=gemini_result):
            agent = await _invoke_gemini(
                prompt='hello', system_prompt='sys', cwd=tmp_path,
                model='gemini-3-flash', max_budget_usd=1.0,
                mcp_config=None, sandbox_modules=None, effort=None,
                timeout_seconds=30.0,
                prices={'gemini-3-flash': {'input_per_1m': 50.0, 'output_per_1m': 50.0}},
            )
        assert agent.cost_usd == pytest.approx((10 * 50.0 + 5 * 50.0) / 1_000_000)

    async def test_invoke_agent_forwards_prices_to_invoke_gemini(self, tmp_path):
        """invoke_agent(backend='gemini', prices=...) forwards through to _invoke_gemini."""
        gemini_result = _SubprocessResult(
            stdout=_GEMINI_VALID_JSON_STDOUT, stderr='', returncode=0, duration_ms=100,
        )
        with patch('orchestrator.agents.invoke._run_subprocess_local',
                   new_callable=AsyncMock, return_value=gemini_result):
            agent = await invoke_agent(
                prompt='hello', system_prompt='sys', cwd=tmp_path,
                backend='gemini', model='gemini-3-flash', max_budget_usd=1.0,
                sandbox_modules=None, effort=None, timeout_seconds=30.0,
                prices={'gemini-3-flash': {'input_per_1m': 50.0, 'output_per_1m': 50.0}},
            )
        assert agent.cost_usd == pytest.approx((10 * 50.0 + 5 * 50.0) / 1_000_000)


_PI_VALID_JSONL_STDOUT = '\n'.join(json.dumps(e) for e in [
    {'type': 'session', 'id': 'sess-invoke-1'},
    {'type': 'turn_end', 'message': {
        'role': 'assistant', 'stopReason': 'stop',
        'usage': {'input': 100, 'output': 20, 'totalTokens': 120, 'cost': {'total': 0.0011}},
        'content': [{'type': 'text', 'text': 'done'}],
    }},
    {'type': 'agent_end', 'messages': [{
        'role': 'assistant', 'stopReason': 'stop',
        'usage': {'input': 100, 'output': 20, 'totalTokens': 120, 'cost': {'total': 0.0011}},
        'content': [{'type': 'text', 'text': 'done'}],
    }], 'willRetry': False},
])


@pytest.mark.asyncio
class TestInvokePiCore:
    """`_invoke_pi` core invocation (deliverable #2): builds the spike-template
    argv, runs it via the UNCHANGED `_run_subprocess_local`, and parses the
    result via `_parse_pi_output`."""

    async def test_returns_parsed_success_result(self, tmp_path):
        pi_result = _SubprocessResult(
            stdout=_PI_VALID_JSONL_STDOUT, stderr='', returncode=0, duration_ms=100,
        )
        with patch('orchestrator.agents.invoke._run_subprocess_local',
                   new_callable=AsyncMock, return_value=pi_result):
            agent = await _invoke_pi(
                'hello', 'sys', cwd=tmp_path, model='anthropic/claude-haiku-4-5',
                max_budget_usd=1.0,
                allowed_tools=['Bash', 'mcp__fused-memory__add_memory'],
                disallowed_tools=None, mcp_config=None, sandbox_modules=None,
                effort=None, oauth_token=None, resume_session_id=None,
                session_id=None, timeout_seconds=30.0, prices=None,
            )
        assert agent.success is True
        assert agent.cost_usd > 0.0
        assert agent.session_id == 'sess-invoke-1'
        assert agent.turns > 0

    async def test_argv_matches_spike_template(self, tmp_path):
        pi_result = _SubprocessResult(
            stdout=_PI_VALID_JSONL_STDOUT, stderr='', returncode=0, duration_ms=100,
        )
        with patch('orchestrator.agents.invoke._run_subprocess_local',
                   new_callable=AsyncMock, return_value=pi_result) as mock_run:
            await _invoke_pi(
                'hello', 'sys', cwd=tmp_path, model='anthropic/claude-haiku-4-5',
                max_budget_usd=1.0,
                allowed_tools=['Bash', 'mcp__fused-memory__add_memory'],
                disallowed_tools=None, mcp_config=None, sandbox_modules=None,
                effort=None, oauth_token=None, resume_session_id=None,
                session_id=None, timeout_seconds=30.0, prices=None,
            )
        cmd = mock_run.call_args.args[0]
        assert cmd[0] == 'pi'
        for token in ('--mode', 'json', '-p', '--model', 'anthropic/claude-haiku-4-5',
                      '--session-dir', '--system-prompt'):
            assert token in cmd, f'{token!r} missing from argv: {cmd!r}'
        tools_csv = cmd[cmd.index('--tools') + 1]
        assert set(tools_csv.split(',')) == {'bash', 'fused_memory_add_memory'}

    async def test_timed_out_propagates(self, tmp_path):
        timed_result = _SubprocessResult(
            stdout='', stderr='timeout', returncode=1, duration_ms=100, timed_out=True,
        )
        with patch('orchestrator.agents.invoke._run_subprocess_local',
                   new_callable=AsyncMock, return_value=timed_result):
            agent = await _invoke_pi(
                'hello', 'sys', cwd=tmp_path, model='anthropic/claude-haiku-4-5',
                max_budget_usd=1.0, allowed_tools=None, disallowed_tools=None,
                mcp_config=None, sandbox_modules=None, effort=None,
                oauth_token=None, resume_session_id=None, session_id=None,
                timeout_seconds=30.0, prices=None,
            )
        assert agent.timed_out is True

    async def test_mcp_config_written_via_write_pi_mcp_config(self, tmp_path):
        """When mcp_config is a dict, _write_pi_mcp_config is invoked (patched
        here) so the on-disk config carries directTools — see the dedicated
        TestWritePiMcpConfig unit tests for the file CONTENT shape."""
        pi_result = _SubprocessResult(
            stdout=_PI_VALID_JSONL_STDOUT, stderr='', returncode=0, duration_ms=100,
        )
        mcp_cfg = {'mcpServers': {'fused-memory': {'command': 'x', 'args': []}}}
        with (
            patch('orchestrator.agents.invoke._run_subprocess_local',
                  new_callable=AsyncMock, return_value=pi_result),
            patch('orchestrator.agents.invoke._write_pi_mcp_config') as mock_write,
        ):
            await _invoke_pi(
                'hello', 'sys', cwd=tmp_path, model='anthropic/claude-haiku-4-5',
                max_budget_usd=1.0, allowed_tools=None, disallowed_tools=None,
                mcp_config=mcp_cfg, sandbox_modules=None, effort=None,
                oauth_token=None, resume_session_id=None, session_id=None,
                timeout_seconds=30.0, prices=None,
            )
        mock_write.assert_called_once()
        assert mcp_cfg in mock_write.call_args.args


@pytest.mark.asyncio
class TestInvokePiFlags:
    """`_invoke_pi` flag construction beyond the core template (deliverable
    #3): --thinking (effort), ANTHROPIC_OAUTH_TOKEN env, --session/
    --session-id, --exclude-tools (disallowed_tools), --no-context-files,
    and the no-silent-cap warning when a spec is unmappable (wildcard)."""

    async def _invoke(self, tmp_path, **overrides):
        """Invoke _invoke_pi with sensible defaults, patched _run_subprocess_local,
        returning the mock so callers can inspect argv/env via call_args."""
        pi_result = _SubprocessResult(
            stdout=_PI_VALID_JSONL_STDOUT, stderr='', returncode=0, duration_ms=100,
        )
        kwargs: dict = dict(
            prompt='hello', system_prompt='sys', cwd=tmp_path,
            model='anthropic/claude-haiku-4-5', max_budget_usd=1.0,
            allowed_tools=None, disallowed_tools=None, mcp_config=None,
            sandbox_modules=None, effort=None, oauth_token=None,
            resume_session_id=None, session_id=None, timeout_seconds=30.0,
            prices=None,
        )
        kwargs.update(overrides)
        prompt = kwargs.pop('prompt')
        system_prompt = kwargs.pop('system_prompt')
        with patch('orchestrator.agents.invoke._run_subprocess_local',
                   new_callable=AsyncMock, return_value=pi_result) as mock_run:
            await _invoke_pi(prompt, system_prompt, **kwargs)
        return mock_run

    async def test_effort_maps_to_thinking_flag(self, tmp_path):
        mock_run = await self._invoke(tmp_path, effort='high')
        cmd = mock_run.call_args.args[0]
        assert '--thinking' in cmd
        assert cmd[cmd.index('--thinking') + 1] == 'high'

    async def test_no_effort_omits_thinking_flag(self, tmp_path):
        mock_run = await self._invoke(tmp_path, effort=None)
        cmd = mock_run.call_args.args[0]
        assert '--thinking' not in cmd

    async def test_oauth_token_sets_anthropic_oauth_token_env(self, tmp_path):
        mock_run = await self._invoke(tmp_path, oauth_token='tok')
        env = mock_run.call_args.args[2]
        assert env.get('ANTHROPIC_OAUTH_TOKEN') == 'tok'

    async def test_resume_session_id_uses_session_flag(self, tmp_path):
        mock_run = await self._invoke(tmp_path, resume_session_id='r1')
        cmd = mock_run.call_args.args[0]
        assert '--session' in cmd
        assert cmd[cmd.index('--session') + 1] == 'r1'
        assert '--session-id' not in cmd

    async def test_session_id_without_resume_uses_session_id_flag(self, tmp_path):
        mock_run = await self._invoke(tmp_path, session_id='s1')
        cmd = mock_run.call_args.args[0]
        assert '--session-id' in cmd
        assert cmd[cmd.index('--session-id') + 1] == 's1'

    async def test_disallowed_tools_maps_to_exclude_tools_flag(self, tmp_path):
        mock_run = await self._invoke(
            tmp_path, disallowed_tools=['mcp__fused-memory__set_task_status'],
        )
        cmd = mock_run.call_args.args[0]
        assert '--exclude-tools' in cmd
        csv = cmd[cmd.index('--exclude-tools') + 1]
        assert 'fused_memory_set_task_status' in csv.split(',')

    async def test_argv_contains_no_context_files(self, tmp_path):
        mock_run = await self._invoke(tmp_path)
        cmd = mock_run.call_args.args[0]
        assert '--no-context-files' in cmd

    async def test_wildcard_spec_is_dropped_and_logged(self, tmp_path, caplog):
        with caplog.at_level(logging.WARNING):
            mock_run = await self._invoke(
                tmp_path, allowed_tools=['mcp__jcodemunch__*', 'Read'],
            )
        cmd = mock_run.call_args.args[0]
        tools_csv = cmd[cmd.index('--tools') + 1]
        assert 'read' in tools_csv.split(',')
        assert 'jcodemunch' not in tools_csv
        warning_records = [r for r in caplog.records if r.levelno >= logging.WARNING]
        assert any('mcp__jcodemunch__*' in r.getMessage() for r in warning_records), (
            f'expected a WARNING naming the dropped spec; got: {[r.getMessage() for r in warning_records]}'
        )


@pytest.mark.asyncio
class TestInvokeAgentPiDispatch:
    """invoke_agent(backend='pi', ...) dispatches to _invoke_pi, forwarding
    the backend-relevant fields (deliverable #5); the Unknown-backend guard
    is unchanged."""

    async def test_dispatches_to_invoke_pi_with_forwarded_kwargs(self, tmp_path):
        sentinel = AgentResult(success=True, output='ok')
        prices = {'x': {'input_per_1m': 1.0, 'output_per_1m': 1.0}}
        mcp_cfg = {'mcpServers': {}}
        with patch(
            'orchestrator.agents.invoke._invoke_pi',
            new_callable=AsyncMock, return_value=sentinel,
        ) as mock_invoke_pi:
            result = await invoke_agent(
                prompt='p', system_prompt='s', cwd=tmp_path,
                backend='pi', model='anthropic/claude-haiku-4-5',
                allowed_tools=['Bash'], disallowed_tools=None,
                mcp_config=mcp_cfg, oauth_token='tok', session_id='sid',
                prices=prices, timeout_seconds=30.0,
            )

        assert result is sentinel
        mock_invoke_pi.assert_awaited_once()
        kwargs = mock_invoke_pi.call_args.kwargs
        assert kwargs.get('allowed_tools') == ['Bash']
        assert kwargs.get('mcp_config') is mcp_cfg
        assert kwargs.get('oauth_token') == 'tok'
        assert kwargs.get('session_id') == 'sid'
        assert kwargs.get('prices') is prices
        assert kwargs.get('model') == 'anthropic/claude-haiku-4-5'
        assert kwargs.get('system_prompt') == 's'
        assert kwargs.get('cwd') == tmp_path

    async def test_unknown_backend_still_raises_value_error(self, tmp_path):
        with pytest.raises(ValueError):
            await invoke_agent(
                prompt='p', system_prompt='s', cwd=tmp_path,
                backend='totally-unknown',
            )


@pytest.mark.asyncio
class TestSandboxPathForwardsSessionConfig:
    """_invoke_claude_with_sandbox sandbox path must forward session_id + config_dir to _run_subprocess.

    Step-15 RED: fails until invoke.py line 212 is updated to thread session_id and config_dir.
    """

    async def test_sandbox_path_passes_session_id_and_config_dir(self, tmp_path):
        """The sandbox-active branch passes session_id and config_dir to _run_subprocess.

        Patches resolve_active_backend → 'bwrap' and wrap_command → identity so the
        sandbox branch is taken.  Captures kwargs on the _run_subprocess mock and
        asserts that session_id and config_dir are forwarded.
        """
        cfg_dir = tmp_path / 'cfg'
        cfg_dir.mkdir()

        captured_kwargs: dict = {}

        async def mock_run_subprocess(cmd, cwd, env, model, timeout_seconds, **kwargs):
            captured_kwargs.update(kwargs)
            return _SubprocessResult(
                stdout='', stderr='', returncode=0, duration_ms=50, timed_out=False,
            )

        with (
            patch(
                'orchestrator.agents.sandbox_dispatch.resolve_active_backend',
                return_value='bwrap',
            ),
            patch(
                'orchestrator.agents.sandbox_dispatch.wrap_command',
                side_effect=lambda cmd, cwd, mods: cmd,
            ),
            patch(
                'orchestrator.agents.invoke._run_subprocess',
                side_effect=mock_run_subprocess,
            ),
        ):
            await _invoke_claude_with_sandbox(
                prompt='hello', system_prompt='sys', cwd=tmp_path,
                model='claude-sonnet-4-5', max_turns=5, max_budget_usd=1.0,
                allowed_tools=None, disallowed_tools=None,
                mcp_config=None, output_schema=None,
                permission_mode='bypassPermissions',
                sandbox_modules=['m'],
                effort=None, timeout_seconds=30.0,
                session_id='sid-xyz',
                config_dir=cfg_dir,
            )

        assert captured_kwargs.get('session_id') == 'sid-xyz', (
            f'Expected session_id="sid-xyz" forwarded to _run_subprocess; got {captured_kwargs!r}'
        )
        assert captured_kwargs.get('config_dir') == cfg_dir, (
            f'Expected config_dir={cfg_dir!r} forwarded to _run_subprocess; got {captured_kwargs!r}'
        )


@pytest.mark.asyncio
class TestStartupGraceSecsForwarding:
    """invoke_agent and _invoke_claude_with_sandbox must forward startup_grace_secs.

    Step-11 RED: fails until invoke.py is updated to thread startup_grace_secs
    through both the non-sandbox (invoke_claude_agent) and sandbox (_run_subprocess)
    paths.
    """

    async def test_non_sandbox_path_forwards_startup_grace_secs(self, tmp_path):
        """Non-sandbox claude path: startup_grace_secs reaches invoke_claude_agent.

        invoke_agent(backend='claude', sandbox_modules=None) falls through to
        invoke_claude_agent().  We patch that and assert startup_grace_secs=33.0 is
        forwarded.

        Fails today: invoke_agent has no startup_grace_secs param → TypeError.
        """
        captured_kwargs: dict = {}

        async def capturing_invoke_claude_agent(*args, **kwargs):
            captured_kwargs.update(kwargs)
            return AgentResult(success=True, output='')

        with patch(
            'orchestrator.agents.invoke.invoke_claude_agent',
            side_effect=capturing_invoke_claude_agent,
        ):
            await invoke_agent(
                prompt='hello', system_prompt='sys', cwd=tmp_path,
                backend='claude', sandbox_modules=None,
                startup_grace_secs=33.0,
            )

        assert captured_kwargs.get('startup_grace_secs') == 33.0, (
            f'startup_grace_secs not forwarded to invoke_claude_agent; captured={captured_kwargs!r}'
        )

    async def test_sandbox_path_forwards_startup_grace_secs(self, tmp_path):
        """Sandbox-active branch: startup_grace_secs reaches _run_subprocess.

        Mirrors TestSandboxPathForwardsSessionConfig: patches resolve_active_backend
        → 'bwrap' and wrap_command → identity so the sandbox branch is taken.
        Captures kwargs on the _run_subprocess mock and asserts startup_grace_secs
        is forwarded.

        Fails today: _invoke_claude_with_sandbox has no startup_grace_secs param →
        TypeError.
        """
        captured_kwargs: dict = {}

        async def mock_run_subprocess(cmd, cwd, env, model, timeout_seconds, **kwargs):
            captured_kwargs.update(kwargs)
            return _SubprocessResult(
                stdout='', stderr='', returncode=0, duration_ms=50, timed_out=False,
            )

        with (
            patch(
                'orchestrator.agents.sandbox_dispatch.resolve_active_backend',
                return_value='bwrap',
            ),
            patch(
                'orchestrator.agents.sandbox_dispatch.wrap_command',
                side_effect=lambda cmd, cwd, mods: cmd,
            ),
            patch(
                'orchestrator.agents.invoke._run_subprocess',
                side_effect=mock_run_subprocess,
            ),
        ):
            await _invoke_claude_with_sandbox(
                prompt='hello', system_prompt='sys', cwd=tmp_path,
                model='claude-sonnet-4-5', max_turns=5, max_budget_usd=1.0,
                allowed_tools=None, disallowed_tools=None,
                mcp_config=None, output_schema=None,
                permission_mode='bypassPermissions',
                sandbox_modules=['m'],
                effort=None, timeout_seconds=30.0,
                startup_grace_secs=33.0,
            )

        assert captured_kwargs.get('startup_grace_secs') == 33.0, (
            f'startup_grace_secs not forwarded to _run_subprocess; captured={captured_kwargs!r}'
        )


@pytest.mark.asyncio
class TestProgressExtensionParamsForwarding:
    """invoke_agent and _invoke_claude_with_sandbox must forward
    working_idle_secs/absolute_cap_secs (task 2360 step-3).

    RED: fails until invoke.py is updated to thread both params through the
    non-sandbox (invoke_claude_agent) and sandbox (_run_subprocess) paths,
    mirroring TestStartupGraceSecsForwarding.
    """

    async def test_non_sandbox_path_forwards_progress_extension_params(self, tmp_path):
        """Non-sandbox claude path: working_idle_secs/absolute_cap_secs reach invoke_claude_agent.

        Fails today: invoke_agent has no such params → TypeError.
        """
        captured_kwargs: dict = {}

        async def capturing_invoke_claude_agent(*args, **kwargs):
            captured_kwargs.update(kwargs)
            return AgentResult(success=True, output='')

        with patch(
            'orchestrator.agents.invoke.invoke_claude_agent',
            side_effect=capturing_invoke_claude_agent,
        ):
            await invoke_agent(
                prompt='hello', system_prompt='sys', cwd=tmp_path,
                backend='claude', sandbox_modules=None,
                working_idle_secs=1800.0,
                absolute_cap_secs=7200.0,
            )

        assert captured_kwargs.get('working_idle_secs') == 1800.0, (
            f'working_idle_secs not forwarded to invoke_claude_agent; captured={captured_kwargs!r}'
        )
        assert captured_kwargs.get('absolute_cap_secs') == 7200.0, (
            f'absolute_cap_secs not forwarded to invoke_claude_agent; captured={captured_kwargs!r}'
        )

    async def test_sandbox_path_forwards_progress_extension_params(self, tmp_path):
        """Sandbox-active branch: working_idle_secs/absolute_cap_secs reach _run_subprocess.

        Mirrors TestStartupGraceSecsForwarding.test_sandbox_path_forwards_startup_grace_secs.

        Fails today: _invoke_claude_with_sandbox has no such params → TypeError.
        """
        captured_kwargs: dict = {}

        async def mock_run_subprocess(cmd, cwd, env, model, timeout_seconds, **kwargs):
            captured_kwargs.update(kwargs)
            return _SubprocessResult(
                stdout='', stderr='', returncode=0, duration_ms=50, timed_out=False,
            )

        with (
            patch(
                'orchestrator.agents.sandbox_dispatch.resolve_active_backend',
                return_value='bwrap',
            ),
            patch(
                'orchestrator.agents.sandbox_dispatch.wrap_command',
                side_effect=lambda cmd, cwd, mods: cmd,
            ),
            patch(
                'orchestrator.agents.invoke._run_subprocess',
                side_effect=mock_run_subprocess,
            ),
        ):
            await _invoke_claude_with_sandbox(
                prompt='hello', system_prompt='sys', cwd=tmp_path,
                model='claude-sonnet-4-5', max_turns=5, max_budget_usd=1.0,
                allowed_tools=None, disallowed_tools=None,
                mcp_config=None, output_schema=None,
                permission_mode='bypassPermissions',
                sandbox_modules=['m'],
                effort=None, timeout_seconds=30.0,
                working_idle_secs=1800.0,
                absolute_cap_secs=7200.0,
            )

        assert captured_kwargs.get('working_idle_secs') == 1800.0, (
            f'working_idle_secs not forwarded to _run_subprocess; captured={captured_kwargs!r}'
        )
        assert captured_kwargs.get('absolute_cap_secs') == 7200.0, (
            f'absolute_cap_secs not forwarded to _run_subprocess; captured={captured_kwargs!r}'
        )


# ===================================================================
# TestReleaseProbeSlotOnException (orchestrator invoke_with_cap_retry)
# ===================================================================


@pytest.mark.asyncio
class TestReleaseProbeSlotOnException:
    """invoke_with_cap_retry (orchestrator) calls release_probe_slot() when invoke raises."""

    async def test_release_probe_slot_called_on_runtime_error(self):
        """release_probe_slot is called with oauth_token when invoke raises."""
        gate = _make_gate(before_invoke=AsyncMock(return_value='tok-a'))
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
        gate = _make_gate(before_invoke=AsyncMock(return_value='tok-a'))
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
        gate = _make_gate(before_invoke=AsyncMock(return_value='tok-a'))
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
        gate = _make_gate(before_invoke=AsyncMock(return_value='tok-a'))
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


# ── _make_gate factory contract ───────────────────────────────────────────────


class TestMakeGateFactory:
    """Regression: _make_gate must set every UsageGate attribute in one place.

    Prior bare-MagicMock() drift caused a 122-error cascade (tasks 1313/1339)
    when a new attribute (soonest_resets_at) was added to UsageGate but not
    propagated to every construction site.  This test pins the factory contract
    so a new attribute addition only requires one edit here.
    """

    def test_make_gate_covers_usage_gate_public_property_surface(self):
        """_make_gate() sets a default for every known UsageGate @property.

        Uses a hardcoded checklist so that adding a new @property to UsageGate
        without also updating ``_GATE_PROPERTY_DEFAULTS`` in ``_orch_helpers.py``
        causes a genuine test failure (a self-referential inspect call cannot catch
        this — it passes by construction).  Regression for the 122-error cascade
        (tasks 1313/1339) where soonest_resets_at was missed at construction sites.

        When UsageGate gains a new @property, update the set below AND
        ``_GATE_PROPERTY_DEFAULTS`` in ``orchestrator/tests/_orch_helpers.py``.
        """
        expected_props = {
            'account_count',
            'active_account_name',
            'cumulative_cost',
            'is_paused',
            'paused_reason',
            'project_id',
            'run_id',
            'soonest_resets_at',
            'total_pause_secs',
        }
        gate = _make_gate()
        assert isinstance(gate, MagicMock)
        gate_vars = vars(gate)
        missing = expected_props - gate_vars.keys()
        assert not missing, (
            f'_make_gate() is missing defaults for UsageGate @property members: {missing!r}. '
            'Add them to _GATE_PROPERTY_DEFAULTS in _orch_helpers.py so every '
            'construction site stays in sync.'
        )

    def test_make_gate_override_and_passthrough(self):
        """Named overrides are applied; arbitrary kwargs are set via setattr."""
        gate = _make_gate(soonest_resets_at=123, account_count=3, custom_attr='x',
                          paused_reason='X')
        assert gate.soonest_resets_at == 123
        assert gate.account_count == 3
        assert gate.custom_attr == 'x'
        assert gate.paused_reason == 'X'
        # Non-overridden defaults remain
        assert gate.active_account_name == 'acct-a'

    def test_make_gate_yielding_propagates_factory_defaults(self):
        """_make_gate_yielding routes through _make_gate: factory defaults present.

        Every key that make_mock_gate() normally sets must also appear on a gate
        produced by _make_gate_yielding, proving that _make_gate_yielding uses
        the factory rather than a bare MagicMock().
        """
        factory_keys = set(vars(_make_gate()).keys())
        yielding_keys = set(vars(_make_gate_yielding([_make_slot()])).keys())
        assert factory_keys <= yielding_keys, (
            f'_make_gate_yielding is missing factory keys: {factory_keys - yielding_keys!r}'
        )

