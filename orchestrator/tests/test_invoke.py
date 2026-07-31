"""Tests for invoke_with_cap_retry in orchestrator/agents/invoke.py."""

from __future__ import annotations

import asyncio
import contextlib
import json
import logging
import os
import signal
import subprocess
from pathlib import Path
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
    async def _cm(scope=None):
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


# ── _run_subprocess_local cancellation (task 3224) ──────────────────────────
#
# shared/tests is NOT on this suite's sys.path (conftest.py adds
# orchestrator/tests, orchestrator/src, shared/src, escalation/src only), so
# these helpers mirror shared/tests/test_proc_group.py's _pgid_gone_within /
# _kill_group rather than importing them.


async def _pgid_gone_within(pgid: int, timeout: float = 5.0, step: float = 0.1) -> bool:
    """Poll until a process group is fully reaped by the kernel.

    A one-shot ``os.killpg(pgid, 0)`` right after a kill is racy: reaping is
    asynchronous (grandchildren can be reparented to a subreaper and linger
    as zombies briefly), so this asserts the real contract — eventual group
    death — via a bounded poll instead of a single check.

    PermissionError (EPERM) is also treated as "gone": it can only fire once
    the kernel has recycled *pgid* to a process owned by another user, which
    means the group this test spawned is definitively no longer around.
    """
    iterations = max(1, int(timeout / step))
    for _ in range(iterations):
        try:
            os.killpg(pgid, 0)
        except (ProcessLookupError, PermissionError):
            return True
        await asyncio.sleep(step)
    return False


def _kill_group(pgid: int) -> None:
    """Best-effort SIGKILL of an entire process group (test cleanup only).

    Precondition: *pgid* must belong to a process known to still be ALIVE.
    Calling this after the process has already been reaped risks landing on
    a recycled, unrelated process group.
    """
    with contextlib.suppress(ProcessLookupError, OSError):
        os.killpg(pgid, signal.SIGKILL)


@pytest.mark.asyncio
class TestRunSubprocessLocalCancellation:

    @pytest.mark.timeout(20)
    async def test_cancellation_reaps_real_process_group(self, tmp_path):
        """Cancelling the awaiting task must reap the real OS process group.

        Regression test for task 3224: _run_subprocess_local spawns with
        start_new_session=True but (pre-fix) has no except asyncio.CancelledError
        handler, so asyncio.wait_for(proc.communicate(...)) is cancelled while
        the spawned process group is never signalled — the agent survives as
        an orphan still editing/committing in its worktree.

        Drives a REAL ``sleep 30`` subprocess (not a mock) and asserts the OS
        process group is actually gone after cancellation, via a bounded poll
        on os.killpg(pgid, 0). A mock-only "terminate_process_group was
        called" assertion would pass even with the wrong pgid and would not
        have caught the original orphan.
        """
        # Capture the real spawner BEFORE patching -- orchestrator.agents.invoke.asyncio
        # IS the global asyncio module, so a self-delegating spy captured after
        # patching would recurse into itself.
        real_exec = asyncio.create_subprocess_exec

        spawned: list[asyncio.subprocess.Process] = []

        async def spy_exec(*args, **kwargs):
            proc = await real_exec(*args, **kwargs)
            # This patches the *global* asyncio module attribute, so for the
            # duration of the `with` block below EVERY subprocess spawned
            # anywhere in this worker process (e.g. this suite's background
            # git-subprocess workers, see conftest.py's
            # _reap_leaked_merge_workers) flows through this spy too. Only
            # record the ``sleep 30`` this test actually launched, so an
            # unrelated concurrent spawn landing first can't be mistaken for
            # it and have its process group polled/killed below.
            if args[:2] == ('sleep', '30'):
                spawned.append(proc)
            return proc

        pgid = None
        try:
            with patch('orchestrator.agents.invoke.asyncio.create_subprocess_exec',
                        side_effect=spy_exec):
                task = asyncio.ensure_future(_run_subprocess_local(
                    ['sleep', '30'],
                    cwd=tmp_path,
                    env=dict(os.environ),
                    backend='codex',
                    model='gpt-5.4',
                    max_budget_usd=1.0,
                    timeout_seconds=30.0,
                ))

                # Bounded poll (~2s max) until the spy has recorded the spawned proc.
                deadline = asyncio.get_running_loop().time() + 2.0
                while not spawned and asyncio.get_running_loop().time() < deadline:
                    await asyncio.sleep(0.05)
                assert len(spawned) == 1, (
                    f'expected exactly one matching "sleep 30" spawn within 2s, '
                    f'got {len(spawned)}: {spawned}'
                )

                proc = spawned[0]
                pgid = proc.pid  # start_new_session ⇒ pgid == pid

                task.cancel()
                with pytest.raises(asyncio.CancelledError):
                    await task

            assert await _pgid_gone_within(pgid), (
                f'Process group {pgid} was not reaped within 5s after '
                f'cancellation — orphaned agent process group left running.'
            )
        finally:
            if pgid is not None and spawned and spawned[0].returncode is None:
                _kill_group(pgid)

    async def test_cancel_during_timeout_kill_still_reaps_group(self, tmp_path):
        """A cancel landing inside the TimeoutError handler's kill must still reap.

        If the awaiting task is cancelled while terminate_process_group is
        already running (SIGTERM sent, SIGKILL escalation pending), a handler
        that is only a SIBLING of `except TimeoutError:` cannot catch it --
        an exception raised inside an except block is not caught by a sibling
        handler of the same try. That would abandon the escalation with a
        SIGTERM-ignoring child left alive. This asserts the cancellation is
        instead caught by something that also wraps the timeout-kill path, so
        terminate_process_group is retried.

        Fully mocked -- no real process, no sleep-based timing. The fake
        terminate_process_group hangs deterministically (on an unset Event)
        on its first call so the test can synchronize on "the task is now
        suspended inside the kill" without guessing at a sleep duration.
        """
        proc = MagicMock()
        proc.pid = 12345  # int so pgid capture and safety-check pass
        proc.communicate = AsyncMock(side_effect=TimeoutError)
        proc.wait = AsyncMock()
        proc.returncode = None

        async def fake_exec(*args, **kwargs):
            return proc

        calls: list[tuple] = []
        entered = asyncio.Event()
        hang_forever = asyncio.Event()  # never set -- first call hangs until cancelled

        async def fake_terminate_process_group(*args, **kwargs):
            # *args/**kwargs (rather than a `grace_secs=5.0`-defaulted
            # parameter) so the recorded call distinguishes "grace_secs was
            # passed as a keyword" from "grace_secs was omitted and a local
            # default filled in" -- the real
            # shared.proc_group.terminate_process_group declares grace_secs
            # keyword-only, and a defaulted fake parameter would silently
            # accept (and record 5.0 for) a positional-arg regression too.
            calls.append((args, kwargs))
            if len(calls) == 1:
                entered.set()
                await hang_forever.wait()

        with (
            patch('orchestrator.agents.invoke.asyncio.create_subprocess_exec',
                  side_effect=fake_exec),
            patch('orchestrator.agents.invoke.terminate_process_group',
                  side_effect=fake_terminate_process_group),
        ):
            task = asyncio.ensure_future(_run_subprocess_local(
                ['fake'], cwd=tmp_path, env={}, backend='codex', model='gpt-5.4',
                max_budget_usd=1.0, timeout_seconds=0.01,
            ))

            # Wait until the task is provably suspended inside the
            # TimeoutError handler's terminate_process_group await.
            await asyncio.wait_for(entered.wait(), timeout=5)

            task.cancel()
            with contextlib.suppress(asyncio.CancelledError):
                await task

        assert len(calls) == 2, (
            f'expected terminate_process_group to be retried once the cancel '
            f'landed during the abandoned timeout-kill, got {len(calls)} '
            f'call(s): {calls}'
        )
        second_args, second_kwargs = calls[1]
        assert second_args[1] == 12345, f'pgid mismatch on retry: {calls[1]}'
        # Must be a keyword, equal to 5.0 -- catches both "kwarg dropped"
        # (kwargs would be {}) and "passed positionally" (it would show up
        # in second_args instead, not second_kwargs).
        assert second_kwargs.get('grace_secs') == 5.0, (
            f'expected grace_secs=5.0 passed as a keyword on retry (the real '
            f'shared.proc_group.terminate_process_group declares grace_secs '
            f'keyword-only), got {calls[1]}'
        )


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
class TestCodexNoAgentsMdWorktreeLeak:
    """Defect (a): _invoke_codex must not leave an AGENTS.md file
    stage-eligible in the worktree — the file must never be written at all.
    Instructions (system_prompt + prompt) must instead be delivered via
    stdin, mirroring the claude backend's ARG_MAX-avoidance stdin piping.
    """

    async def test_no_agents_md_and_instructions_delivered_via_stdin(self, tmp_path):
        """No AGENTS.md is ever written (so none is ever stage-eligible for
        the single-implementer-commit path's `git add -A -- . :!.claude`,
        git_ops.py:5320), and both system_prompt and prompt reach the
        subprocess exclusively via stdin_data.
        """
        subprocess.run(['git', 'init'], cwd=tmp_path, check=True, capture_output=True)

        captured: dict = {}

        async def fake_run_subprocess_local(
            cmd, cwd, env, backend, model, max_budget_usd, timeout_seconds,
            stdin_data=None,
        ):
            # AT subprocess-call time (before any finally-block cleanup runs):
            # record whether AGENTS.md exists, and mirror GitOps.commit's
            # real staging command to see what a commit right now would pick up.
            captured['agents_md_exists'] = (cwd / 'AGENTS.md').exists()
            subprocess.run(
                ['git', 'add', '-A', '--', '.', ':!.claude'],
                cwd=cwd, check=True, capture_output=True,
            )
            staged = subprocess.run(
                ['git', 'diff', '--cached', '--name-only'],
                cwd=cwd, check=True, capture_output=True, text=True,
            )
            captured['staged_names'] = staged.stdout.splitlines()
            captured['stdin_data'] = stdin_data
            return _SubprocessResult(stdout='', stderr='', returncode=0, duration_ms=1)

        with patch('orchestrator.agents.invoke._run_subprocess_local',
                   side_effect=fake_run_subprocess_local):
            await _invoke_codex(
                prompt='S3NT_USER', system_prompt='S3NT_SYS', cwd=tmp_path,
                model='gpt-5.4', max_budget_usd=1.0,
                mcp_config=None, sandbox_modules=None, effort=None,
            )

        assert captured['agents_md_exists'] is False
        assert 'AGENTS.md' not in captured['staged_names']
        # Exact composed payload — pins ordering (system_prompt before
        # prompt) and the '\n\n' separator, not just that both substrings
        # appear somewhere in the blob.
        assert captured['stdin_data'] == b'S3NT_SYS\n\nS3NT_USER'


@pytest.mark.asyncio
class TestInvokeAgentForwardsMaxTurnsToCodex:
    """Defect (b): invoke_agent must forward max_turns to the codex path for
    dispatcher-signature uniformity/observability, even though codex-cli has
    no native turn cap (the wall-clock watchdog via timeout_seconds is the
    sole enforced ceiling — see _invoke_codex's docstring).
    """

    async def test_max_turns_and_timeout_seconds_forwarded(self, tmp_path):
        dummy_result = AgentResult(success=True, output='')
        with patch('orchestrator.agents.invoke._invoke_codex',
                   new_callable=AsyncMock, return_value=dummy_result) as mock_invoke_codex:
            await invoke_agent(
                prompt='p', system_prompt='s', cwd=tmp_path,
                backend='codex', model='gpt-5.4', max_turns=42,
                max_budget_usd=3.0, timeout_seconds=99.0,
            )

        mock_invoke_codex.assert_awaited_once()
        assert mock_invoke_codex.call_args.kwargs.get('max_turns') == 42
        assert mock_invoke_codex.call_args.kwargs.get('timeout_seconds') == 99.0


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

    async def test_system_prompt_passed_as_file_reference(self, tmp_path):
        """The system prompt is written to a git-excluded file under
        --session-dir and passed as `--system-prompt @<path>` rather than
        embedded on argv — pi's `--system-prompt <text-or-@file>` form
        documents the @file syntax (spike template,
        plans/pi-spike-findings.md:201). Architect/deep-reviewer system
        prompts can be large enough that embedding them (plus a large user
        prompt) on argv risks exceeding the OS ARG_MAX and failing exec()
        with E2BIG."""
        pi_result = _SubprocessResult(
            stdout=_PI_VALID_JSONL_STDOUT, stderr='', returncode=0, duration_ms=100,
        )
        big_system_prompt = 'you are an architect. ' * 50
        captured: dict = {}

        async def fake_run(cmd, cwd, env, backend, model, max_budget_usd, timeout_seconds):
            captured['value'] = cmd[cmd.index('--system-prompt') + 1]
            captured['path'] = Path(captured['value'].removeprefix('@'))
            captured['content_during_call'] = captured['path'].read_text()
            return pi_result

        with patch('orchestrator.agents.invoke._run_subprocess_local', side_effect=fake_run):
            await _invoke_pi(
                'hello', big_system_prompt, cwd=tmp_path, model='anthropic/claude-haiku-4-5',
                max_budget_usd=1.0, allowed_tools=None, disallowed_tools=None,
                mcp_config=None, sandbox_modules=None, effort=None,
                oauth_token=None, resume_session_id=None, session_id=None,
                timeout_seconds=30.0, prices=None,
            )

        assert captured['value'].startswith('@'), captured['value']
        assert big_system_prompt not in captured['value']
        assert captured['content_during_call'] == big_system_prompt
        # Cleaned up afterwards (mirrors codex's AGENTS.md temp-file cleanup).
        assert not captured['path'].exists()


@pytest.mark.asyncio
class TestInvokePiMcpConfigBackupRestore:
    """The robustness-critical `.mcp.json` backup/restore in `_invoke_pi`:
    placement into cwd, backing up a pre-existing repo-committed
    `.mcp.json`, and restoring it in `finally`. Exercised with the REAL
    (unpatched) `_write_pi_mcp_config`, unlike TestInvokePiCore's
    mcp-config test, which patches it out and so never touches disk."""

    async def test_preexisting_mcp_json_is_backed_up_and_restored(self, tmp_path):
        original_content = '{"mcpServers": {"committed-server": {"command": "y"}}}'
        mcp_json_path = tmp_path / '.mcp.json'
        mcp_json_path.write_text(original_content)

        pi_result = _SubprocessResult(
            stdout=_PI_VALID_JSONL_STDOUT, stderr='', returncode=0, duration_ms=100,
        )
        captured: dict = {}

        async def fake_run(cmd, cwd, env, backend, model, max_budget_usd, timeout_seconds):
            # While _invoke_pi is "running" (subprocess in flight), the
            # repo's own .mcp.json has been swapped out for pi's config.
            captured['content_during_call'] = mcp_json_path.read_text()
            return pi_result

        mcp_cfg = {'mcpServers': {'fused-memory': {'command': 'x', 'args': []}}}
        with patch('orchestrator.agents.invoke._run_subprocess_local', side_effect=fake_run):
            await _invoke_pi(
                'hello', 'sys', cwd=tmp_path, model='anthropic/claude-haiku-4-5',
                max_budget_usd=1.0, allowed_tools=None, disallowed_tools=None,
                mcp_config=mcp_cfg, sandbox_modules=None, effort=None,
                oauth_token=None, resume_session_id=None, session_id=None,
                timeout_seconds=30.0, prices=None,
            )

        written_during_call = json.loads(captured['content_during_call'])
        assert written_during_call['mcpServers']['fused-memory']['directTools'] is True
        # The original, pre-existing committed config is restored afterward.
        assert mcp_json_path.read_text() == original_content
        # No leftover backup file.
        assert not (tmp_path / '.mcp.json.pi-invoke-backup').exists()

    async def test_no_preexisting_mcp_json_leaves_none_behind(self, tmp_path):
        mcp_json_path = tmp_path / '.mcp.json'
        assert not mcp_json_path.exists()

        pi_result = _SubprocessResult(
            stdout=_PI_VALID_JSONL_STDOUT, stderr='', returncode=0, duration_ms=100,
        )
        mcp_cfg = {'mcpServers': {'fused-memory': {'command': 'x', 'args': []}}}
        with patch('orchestrator.agents.invoke._run_subprocess_local',
                   new_callable=AsyncMock, return_value=pi_result):
            await _invoke_pi(
                'hello', 'sys', cwd=tmp_path, model='anthropic/claude-haiku-4-5',
                max_budget_usd=1.0, allowed_tools=None, disallowed_tools=None,
                mcp_config=mcp_cfg, sandbox_modules=None, effort=None,
                oauth_token=None, resume_session_id=None, session_id=None,
                timeout_seconds=30.0, prices=None,
            )

        assert not mcp_json_path.exists()
        assert not (tmp_path / '.mcp.json.pi-invoke-backup').exists()


@pytest.mark.asyncio
class TestInvokePiMcpCacheWarmWarning:
    """`_invoke_pi` warns when direct-tool MCP names are requested for
    servers whose pi-mcp-adapter metadata cache (spike Q3 CRITICAL
    GOTCHA — see _write_pi_mcp_config's docstring) hasn't been warmed
    yet, so the silent proxy-`mcp`-tool fallback is observable instead of
    surfacing later as mysterious missing-tool behavior."""

    async def test_warns_when_cache_file_absent(self, tmp_path, caplog):
        pi_result = _SubprocessResult(
            stdout=_PI_VALID_JSONL_STDOUT, stderr='', returncode=0, duration_ms=100,
        )
        missing_cache = tmp_path / 'not-warmed' / 'mcp-cache.json'
        mcp_cfg = {'mcpServers': {'fused-memory': {'command': 'x', 'args': []}}}
        with (
            patch('orchestrator.agents.invoke._run_subprocess_local',
                  new_callable=AsyncMock, return_value=pi_result),
            patch('orchestrator.agents.invoke._pi_mcp_cache_path', return_value=missing_cache),
            caplog.at_level(logging.WARNING),
        ):
            await _invoke_pi(
                'hello', 'sys', cwd=tmp_path, model='anthropic/claude-haiku-4-5',
                max_budget_usd=1.0, allowed_tools=None, disallowed_tools=None,
                mcp_config=mcp_cfg, sandbox_modules=None, effort=None,
                oauth_token=None, resume_session_id=None, session_id=None,
                timeout_seconds=30.0, prices=None,
            )
        warnings = [r for r in caplog.records if r.levelno >= logging.WARNING]
        assert any(
            'fused-memory' in r.getMessage() and 'cache' in r.getMessage().lower()
            for r in warnings
        ), [r.getMessage() for r in warnings]

    async def test_no_warning_when_cache_file_present(self, tmp_path, caplog):
        pi_result = _SubprocessResult(
            stdout=_PI_VALID_JSONL_STDOUT, stderr='', returncode=0, duration_ms=100,
        )
        warmed_cache = tmp_path / 'warmed' / 'mcp-cache.json'
        warmed_cache.parent.mkdir()
        warmed_cache.write_text('{}')
        mcp_cfg = {'mcpServers': {'fused-memory': {'command': 'x', 'args': []}}}
        with (
            patch('orchestrator.agents.invoke._run_subprocess_local',
                  new_callable=AsyncMock, return_value=pi_result),
            patch('orchestrator.agents.invoke._pi_mcp_cache_path', return_value=warmed_cache),
            caplog.at_level(logging.WARNING),
        ):
            await _invoke_pi(
                'hello', 'sys', cwd=tmp_path, model='anthropic/claude-haiku-4-5',
                max_budget_usd=1.0, allowed_tools=None, disallowed_tools=None,
                mcp_config=mcp_cfg, sandbox_modules=None, effort=None,
                oauth_token=None, resume_session_id=None, session_id=None,
                timeout_seconds=30.0, prices=None,
            )
        warnings = [r for r in caplog.records if 'direct-tools cache' in r.getMessage()]
        assert not warnings


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

    async def test_oauth_token_strips_lingering_anthropic_api_key(self, tmp_path, monkeypatch):
        """A lingering ANTHROPIC_API_KEY must not shadow the OAuth token pi
        is being asked to use — otherwise the multi-account OAuth failover
        oauth_token exists for is silently defeated (mirrors
        _invoke_claude_with_sandbox's credential-precedence guard)."""
        monkeypatch.setenv('ANTHROPIC_API_KEY', 'sk-should-not-leak')
        mock_run = await self._invoke(tmp_path, oauth_token='tok')
        env = mock_run.call_args.args[2]
        assert env.get('ANTHROPIC_OAUTH_TOKEN') == 'tok'
        assert 'ANTHROPIC_API_KEY' not in env

    async def test_no_oauth_token_leaves_anthropic_api_key_untouched(self, tmp_path, monkeypatch):
        """Without oauth_token, a directly-configured ANTHROPIC_API_KEY
        (pi's non-OAuth Anthropic auth path) must keep working — the strip
        is scoped to the OAuth-token case, not unconditional."""
        monkeypatch.setenv('ANTHROPIC_API_KEY', 'sk-direct-key')
        mock_run = await self._invoke(tmp_path, oauth_token=None)
        env = mock_run.call_args.args[2]
        assert env.get('ANTHROPIC_API_KEY') == 'sk-direct-key'
        assert 'ANTHROPIC_OAUTH_TOKEN' not in env

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
                side_effect=lambda cmd, cwd, mods, writable_extras=None: cmd,
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
                side_effect=lambda cmd, cwd, mods, writable_extras=None: cmd,
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
class TestSpawnEnvForwarding:
    """invoke_agent and _invoke_claude_with_sandbox must thread spawn_env
    (CLAUDE_SPAWN_* identity vars) through to the Claude subprocess env.

    Step-3 RED (task 2512): fails until invoke.py is updated to accept and
    forward spawn_env on both the non-sandbox (invoke_claude_agent) and
    sandbox (_run_subprocess) paths.
    """

    async def test_non_sandbox_path_forwards_spawn_env(self, tmp_path):
        """Non-sandbox claude path: spawn_env reaches invoke_claude_agent.

        invoke_agent(backend='claude', sandbox_modules=None) falls through to
        invoke_claude_agent().  We patch that and assert spawn_env is
        forwarded unchanged.

        Fails today: invoke_agent has no spawn_env param → TypeError.
        """
        captured_kwargs: dict = {}

        async def capturing_invoke_claude_agent(*args, **kwargs):
            captured_kwargs.update(kwargs)
            return AgentResult(success=True, output='')

        spawn_env = {
            'CLAUDE_SPAWN_ROLE': 'implementer',
            'CLAUDE_SPAWN_PROJECT': 'dark_factory',
            'CLAUDE_SPAWN_TASK_ID': '2512',
            'CLAUDE_SPAWN_PARENT_ID': '2512-abcd1234',
        }

        with patch(
            'orchestrator.agents.invoke.invoke_claude_agent',
            side_effect=capturing_invoke_claude_agent,
        ):
            await invoke_agent(
                prompt='hello', system_prompt='sys', cwd=tmp_path,
                backend='claude', sandbox_modules=None,
                spawn_env=spawn_env,
            )

        assert captured_kwargs.get('spawn_env') == spawn_env, (
            f'spawn_env not forwarded to invoke_claude_agent; captured={captured_kwargs!r}'
        )

    async def test_sandbox_path_forwards_spawn_env(self, tmp_path):
        """Sandbox-active branch: spawn_env's CLAUDE_SPAWN_* keys land in the
        subprocess env passed to _run_subprocess.

        Mirrors TestStartupGraceSecsForwarding.test_sandbox_path_forwards_startup_grace_secs:
        patches resolve_active_backend → 'bwrap' and wrap_command → identity so
        the sandbox branch is taken. Captures the *env* positional arg on the
        _run_subprocess mock and asserts the CLAUDE_SPAWN_* keys appear in it.

        Fails today: _invoke_claude_with_sandbox has no spawn_env param → TypeError.
        """
        captured_env: dict = {}

        async def mock_run_subprocess(cmd, cwd, env, model, timeout_seconds, **kwargs):
            captured_env.update(env)
            return _SubprocessResult(
                stdout='', stderr='', returncode=0, duration_ms=50, timed_out=False,
            )

        spawn_env = {
            'CLAUDE_SPAWN_ROLE': 'implementer',
            'CLAUDE_SPAWN_PROJECT': 'dark_factory',
            'CLAUDE_SPAWN_TASK_ID': '2512',
            'CLAUDE_SPAWN_PARENT_ID': '2512-abcd1234',
        }

        with (
            patch(
                'orchestrator.agents.sandbox_dispatch.resolve_active_backend',
                return_value='bwrap',
            ),
            patch(
                'orchestrator.agents.sandbox_dispatch.wrap_command',
                side_effect=lambda cmd, cwd, mods, writable_extras=None: cmd,
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
                spawn_env=spawn_env,
            )

        for key, value in spawn_env.items():
            assert captured_env.get(key) == value, (
                f'{key} not forwarded to _run_subprocess env; captured={captured_env!r}'
            )

    async def test_sandbox_path_scrubs_inherited_session_id_and_launcher_pid(self, tmp_path):
        """Sandbox-active branch: an inherited CLAUDE_SPAWN_SESSION_ID/
        CLAUDE_SPAWN_LAUNCHER_PID (this process's own os.environ) must not
        leak into the subprocess env once spawn_env is provided -- mirrors
        the non-sandbox scrub in shared.cli_invoke._invoke_claude; see that
        site's comment for the full rationale (session_hooks.hook_session_slug
        prefers an inherited CLAUDE_SPAWN_SESSION_ID outright, which would
        otherwise collapse every spawned agent onto one registry record).
        """
        captured_env: dict = {}

        async def mock_run_subprocess(cmd, cwd, env, model, timeout_seconds, **kwargs):
            captured_env.update(env)
            return _SubprocessResult(
                stdout='', stderr='', returncode=0, duration_ms=50, timed_out=False,
            )

        spawn_env = {
            'CLAUDE_SPAWN_ROLE': 'implementer',
            'CLAUDE_SPAWN_PARENT_ID': '2512-abcd1234',
        }
        inherited = {
            'CLAUDE_SPAWN_SESSION_ID': 'inherited-launching-slug',
            'CLAUDE_SPAWN_LAUNCHER_PID': '99999',
        }

        with (
            patch.dict(os.environ, inherited),
            patch(
                'orchestrator.agents.sandbox_dispatch.resolve_active_backend',
                return_value='bwrap',
            ),
            patch(
                'orchestrator.agents.sandbox_dispatch.wrap_command',
                side_effect=lambda cmd, cwd, mods, writable_extras=None: cmd,
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
                spawn_env=spawn_env,
            )

        assert 'CLAUDE_SPAWN_SESSION_ID' not in captured_env
        assert 'CLAUDE_SPAWN_LAUNCHER_PID' not in captured_env
        assert captured_env.get('CLAUDE_SPAWN_ROLE') == 'implementer'


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
                side_effect=lambda cmd, cwd, mods, writable_extras=None: cmd,
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


@pytest.mark.asyncio
class TestSandboxExtrasForwarding:
    """invoke_agent and _invoke_claude_with_sandbox must thread sandbox_extras
    → wrap_command(writable_extras=...) (task 2905 step-3).

    sandbox_extras is the carve-out vehicle carrying compute_write_set()'s
    absolute contract paths (worktree root + carve-outs) into the sandbox —
    distinct from sandbox_modules/writable_modules (relative, join-to-worktree,
    makedirs). RED: fails until invoke.py threads sandbox_extras through
    invoke_agent and the four sub-invokers to wrap_command's writable_extras.
    """

    async def test_sandbox_path_forwards_sandbox_extras_to_wrap_command(self, tmp_path):
        """_invoke_claude_with_sandbox passes sandbox_extras to wrap_command as
        writable_extras=, with sandbox_modules=[] the positional writable_modules.

        Fails today: _invoke_claude_with_sandbox has no sandbox_extras param →
        TypeError.
        """
        captured: dict = {}

        def capturing_wrap_command(cmd, cwd, mods, writable_extras=None):
            captured['writable_modules'] = mods
            captured['writable_extras'] = writable_extras
            return cmd

        async def mock_run_subprocess(cmd, cwd, env, model, timeout_seconds, **kwargs):
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
                side_effect=capturing_wrap_command,
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
                sandbox_modules=[],
                effort=None, timeout_seconds=30.0,
                sandbox_extras=['/abs/carveout'],
            )

        assert captured.get('writable_extras') == ['/abs/carveout'], (
            f"Expected wrap_command called with writable_extras=['/abs/carveout']; "
            f"got {captured.get('writable_extras')!r}"
        )
        assert captured.get('writable_modules') == [], (
            f'Expected positional writable_modules=[] (sandbox_modules=[]); '
            f"got {captured.get('writable_modules')!r}"
        )

    async def test_invoke_agent_forwards_sandbox_extras_to_claude_subinvoker(self, tmp_path):
        """invoke_agent(backend='claude') forwards sandbox_extras to
        _invoke_claude_with_sandbox.

        Fails today: invoke_agent has no sandbox_extras param → TypeError.
        """
        with patch(
            'orchestrator.agents.invoke._invoke_claude_with_sandbox',
            new_callable=AsyncMock,
            return_value=AgentResult(success=True, output=''),
        ) as mock_claude:
            await invoke_agent(
                prompt='hello', system_prompt='sys', cwd=tmp_path,
                backend='claude', sandbox_modules=[],
                sandbox_extras=['/abs/carveout'],
            )

        assert mock_claude.await_args is not None, 'await_args must be set after one await'
        assert mock_claude.await_args.kwargs.get('sandbox_extras') == ['/abs/carveout'], (
            f'sandbox_extras not forwarded to _invoke_claude_with_sandbox; '
            f'captured={mock_claude.await_args.kwargs!r}'
        )

    @pytest.mark.parametrize(
        ('backend', 'subinvoker'),
        [
            ('codex', '_invoke_codex'),
            ('gemini', '_invoke_gemini'),
            ('pi', '_invoke_pi'),
        ],
    )
    async def test_invoke_agent_forwards_sandbox_extras_to_nonclaude_subinvoker(
        self, tmp_path, backend, subinvoker,
    ):
        """invoke_agent(backend=codex|gemini|pi) forwards sandbox_extras to the
        matching _invoke_<backend> sub-invoker.

        The claude path is covered by
        test_invoke_agent_forwards_sandbox_extras_to_claude_subinvoker; this
        parametrized guard exercises the identical sandbox_extras →
        wrap_command(writable_extras=...) forwarding threaded through the other
        three sub-invokers (task 2905 step-4), so a future edit that drops the
        kwarg from codex/gemini/pi is caught here too.
        """
        with patch(
            f'orchestrator.agents.invoke.{subinvoker}',
            new_callable=AsyncMock,
            return_value=AgentResult(success=True, output=''),
        ) as mock_subinvoker:
            await invoke_agent(
                prompt='hello', system_prompt='sys', cwd=tmp_path,
                backend=backend, sandbox_modules=[],
                sandbox_extras=['/abs/carveout'],
            )

        assert mock_subinvoker.await_args is not None, 'await_args must be set after one await'
        assert mock_subinvoker.await_args.kwargs.get('sandbox_extras') == ['/abs/carveout'], (
            f'sandbox_extras not forwarded to {subinvoker}; '
            f'captured={mock_subinvoker.await_args.kwargs!r}'
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

