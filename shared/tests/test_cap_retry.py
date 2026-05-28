"""Exhaustive tests for invoke_with_cap_retry — cap detection, failover, resume,
cooldown backoff, budget enforcement, cost-store integration, and edge cases.

Covers every branch in shared.cli_invoke.invoke_with_cap_retry (lines 136-274).
"""

from __future__ import annotations

import inspect
import itertools
import json
import logging
import os
import re
import uuid
from unittest.mock import AsyncMock, MagicMock, PropertyMock, call, patch

import pytest

from shared.cli_invoke import (
    _CAP_HIT_COOLDOWN_SECS,
    _MAX_CAP_COOLDOWN_SECS,
    CAP_HIT_RESUME_PROMPT,
    CRASH_RECOVERY_RESUME_PROMPT,
    AgentResult,
    AllAccountsCappedException,
    invoke_with_cap_retry,
)
from shared.config_models import AccountConfig, UsageCapConfig
from shared.usage_gate import SessionBudgetExhausted, UsageGate

# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------

def make_gate(account_names: list[str], **kwargs) -> UsageGate:
    """Create a UsageGate with fake accounts, probe disabled."""
    acct_cfgs = []
    env_vars = {}
    for name in account_names:
        env_key = f'TEST_TOKEN_{name.upper().replace("-", "_")}'
        env_vars[env_key] = f'fake-token-{name}'
        acct_cfgs.append(AccountConfig(name=name, oauth_token_env=env_key))
    config = UsageCapConfig(accounts=acct_cfgs, **kwargs)
    with patch.dict(os.environ, env_vars):
        gate = UsageGate(config)
    gate._run_probe = AsyncMock(return_value=True)
    return gate


def make_result(
    success: bool = True,
    output: str = 'done',
    session_id: str = '',
    stderr: str = '',
    cost_usd: float = 0.5,
    **kw,
) -> AgentResult:
    return AgentResult(
        success=success,
        output=output,
        session_id=session_id,
        stderr=stderr,
        cost_usd=cost_usd,
        **kw,
    )


def _mock_gate(**overrides) -> MagicMock:
    """Build a MagicMock UsageGate with sensible defaults.

    Since commit 065e95a4c9 the cap-retry path in ``invoke_with_cap_retry``
    goes through ``async with usage_gate.invoke_slot() as slot:``, where
    ``slot`` is an :class:`~shared.usage_gate.InvokeSlot` instance that
    proxies to the gate.  This helper wires ``gate.invoke_slot()`` to an
    async-context-manager mock whose ``__aenter__`` yields a slot whose
    ``detect_cap_hit`` / ``confirm`` / ``settle`` methods proxy to the
    corresponding gate attributes — so tests can still assert on
    ``gate.detect_cap_hit.call_args``, ``gate.confirm_account_ok``, etc.

    ``__aexit__`` calls ``gate.release_probe_slot(slot.token)`` unless the
    slot was settled by ``detect_cap_hit(...)==True``, ``confirm(...)``,
    or ``settle()``.  This matches the production ``UsageGate.invoke_slot``
    asynccontextmanager behaviour and keeps the ``release_probe_slot``
    assertions in ``TestReleaseProbeSlotOnException`` /
    ``TestCancelledErrorReleaseProbeSlot`` meaningful.

    Sister helper: ``orchestrator/tests/_orch_helpers.py::make_mock_gate`` —
    same shape but without the ``invoke_slot()`` async-CM wiring (orchestrator
    tests use ``_attach_invoke_slot`` separately for that layer).  Cannot be
    unified here: ``shared`` cannot import from ``orchestrator/tests``
    (that would invert the package layering direction).
    """
    gate = MagicMock()
    gate.account_count = overrides.pop('account_count', 1)
    gate.before_invoke = overrides.pop('before_invoke', AsyncMock(return_value='tok'))
    gate.detect_cap_hit = overrides.pop('detect_cap_hit', MagicMock(return_value=False))
    gate.active_account_name = overrides.pop('active_account_name', 'acct')
    gate.on_agent_complete = overrides.pop('on_agent_complete', MagicMock())
    gate.confirm_account_ok = overrides.pop('confirm_account_ok', MagicMock())
    gate.release_probe_slot = overrides.pop('release_probe_slot', MagicMock())
    gate.soonest_resets_at = overrides.pop('soonest_resets_at', None)
    for k, v in overrides.items():
        setattr(gate, k, v)

    # Wire gate.invoke_slot() to yield an InvokeSlot-shaped proxy.
    # A fresh slot is built on each call to invoke_slot() so that
    # per-iteration side_effects on before_invoke / detect_cap_hit /
    # active_account_name PropertyMock fire in the expected order.
    def _make_invoke_slot_cm():
        holder: dict = {'slot': None}

        async def _aenter_impl(*_args, **_kw):
            token = await gate.before_invoke()
            slot = MagicMock()
            slot.token = token
            slot.account_name = gate.active_account_name
            slot._settled = False

            def _slot_detect_cap_hit(stderr, output, backend='claude'):
                hit = gate.detect_cap_hit(
                    stderr, output, backend, oauth_token=slot.token,
                )
                if hit:
                    slot._settled = True
                return hit

            def _slot_confirm(cost_usd=0.0):
                gate.confirm_account_ok(slot.token)
                gate.on_agent_complete(cost_usd)
                slot._settled = True

            def _slot_settle():
                slot._settled = True

            # Plain synchronous MagicMocks — InvokeSlot.detect_cap_hit /
            # confirm / settle are plain methods, not coroutines.  Using
            # MagicMock (not AsyncMock) is load-bearing: prod does
            # ``if slot.detect_cap_hit(...):`` without ``await``.
            slot.detect_cap_hit = MagicMock(side_effect=_slot_detect_cap_hit)
            slot.confirm = MagicMock(side_effect=_slot_confirm)
            slot.settle = MagicMock(side_effect=_slot_settle)
            holder['slot'] = slot
            return slot

        async def _aexit_impl(*_args, **_kw):
            slot = holder['slot']
            if slot is not None and not slot._settled:
                gate.release_probe_slot(slot.token)
            return None

        cm = MagicMock()
        cm.__aenter__ = AsyncMock(side_effect=_aenter_impl)
        cm.__aexit__ = AsyncMock(side_effect=_aexit_impl)
        return cm

    gate.invoke_slot = MagicMock(side_effect=_make_invoke_slot_cm)
    return gate


def test_mock_gate_defaults_include_release_probe_slot():
    """_mock_gate() should explicitly set release_probe_slot in its defaults.

    Checks vars(gate) rather than hasattr(gate, ...) so that MagicMock's
    silent auto-attribute creation doesn't produce a false positive.
    """
    gate = _mock_gate()
    assert 'release_probe_slot' in vars(gate), (
        "_mock_gate() must explicitly set release_probe_slot so the "
        "exception-cleanup contract is self-documented in the helper."
    )


# Shared patch targets
_INVOKE_PATCH = 'shared.cli_invoke.invoke_claude_agent'
_SLEEP_PATCH = 'shared.cli_invoke.asyncio.sleep'
_LOGGER_WARN_PATCH = 'shared.cli_invoke.logger.warning'


# ===================================================================
# TestCapRetryNormal
# ===================================================================


@pytest.mark.asyncio
class TestCapRetryNormal:
    """Happy-path: no cap hits, single invocation."""

    async def test_no_cap_hit_returns_immediately(self):
        """No cap hit -> returns result immediately, invoke called once."""
        gate = _mock_gate()
        result = make_result()
        with (
            patch(_INVOKE_PATCH, new_callable=AsyncMock, return_value=result) as mock_inv,
            patch(_SLEEP_PATCH, new_callable=AsyncMock),
        ):
            got = await invoke_with_cap_retry(gate, 'lbl', prompt='hi')
        assert got is result
        mock_inv.assert_awaited_once()

    async def test_no_gate_passthrough(self):
        """usage_gate=None -> passthrough, invoke called once, no gate methods."""
        result = make_result()
        with patch(_INVOKE_PATCH, new_callable=AsyncMock, return_value=result) as mock_inv:
            got = await invoke_with_cap_retry(None, 'lbl', prompt='hi')
        assert got is result
        mock_inv.assert_awaited_once()

    async def test_confirm_account_ok_called_on_success(self):
        """confirm_account_ok is called with the oauth_token on success."""
        gate = _mock_gate(before_invoke=AsyncMock(return_value='tok-x'))
        result = make_result()
        with patch(_INVOKE_PATCH, new_callable=AsyncMock, return_value=result):
            await invoke_with_cap_retry(gate, 'lbl', prompt='hi')
        gate.confirm_account_ok.assert_called_once_with('tok-x')

    async def test_on_agent_complete_called_with_cost(self):
        """on_agent_complete is called with result.cost_usd."""
        gate = _mock_gate()
        result = make_result(cost_usd=1.23)
        with patch(_INVOKE_PATCH, new_callable=AsyncMock, return_value=result):
            await invoke_with_cap_retry(gate, 'lbl', prompt='hi')
        gate.on_agent_complete.assert_called_once_with(1.23)

    async def test_account_name_set_on_result(self):
        """result.account_name is set from active_account_name."""
        gate = _mock_gate(active_account_name='my-acct')
        result = make_result()
        with patch(_INVOKE_PATCH, new_callable=AsyncMock, return_value=result):
            got = await invoke_with_cap_retry(gate, 'lbl', prompt='hi')
        assert got.account_name == 'my-acct'

    async def test_cost_invocation_saved(self):
        """save_invocation is awaited with correct params on success."""
        gate = _mock_gate(active_account_name='acct-a')
        cost_store = MagicMock()
        cost_store.save_invocation = AsyncMock()
        result = make_result(
            cost_usd=2.50, duration_ms=7000,
            input_tokens=500, output_tokens=300,
            cache_read_tokens=100, cache_create_tokens=20,
        )
        with patch(_INVOKE_PATCH, new_callable=AsyncMock, return_value=result):
            await invoke_with_cap_retry(
                gate, 'lbl',
                cost_store=cost_store, run_id='r1', task_id='t1',
                project_id='p1', role='impl',
                prompt='hi', model='sonnet',
            )
        cost_store.save_invocation.assert_awaited_once()
        kw = cost_store.save_invocation.call_args.kwargs
        assert kw['run_id'] == 'r1'
        assert kw['task_id'] == 't1'
        assert kw['project_id'] == 'p1'
        assert kw['account_name'] == 'acct-a'
        assert kw['model'] == 'sonnet'
        assert kw['role'] == 'impl'
        assert kw['cost_usd'] == 2.50
        assert kw['input_tokens'] == 500
        assert kw['output_tokens'] == 300
        assert kw['cache_read_tokens'] == 100
        assert kw['cache_create_tokens'] == 20
        assert kw['duration_ms'] == 7000
        assert kw['capped'] is False
        assert 'started_at' in kw
        assert 'completed_at' in kw


# ===================================================================
# TestCapRetryFailover
# ===================================================================


@pytest.mark.asyncio
class TestCapRetryFailover:
    """Cap hit -> retry on next account."""

    async def test_cap_hit_then_success(self):
        """First call caps, second succeeds. Two invoke calls total."""
        gate = _mock_gate(
            account_count=2,
            before_invoke=AsyncMock(side_effect=['tok-a', 'tok-b']),
            detect_cap_hit=MagicMock(side_effect=[True, False]),
            active_account_name='acct-b',
        )
        result = make_result()
        with (
            patch(_INVOKE_PATCH, new_callable=AsyncMock, return_value=result) as mock_inv,
            patch(_SLEEP_PATCH, new_callable=AsyncMock),
        ):
            got = await invoke_with_cap_retry(gate, 'lbl', prompt='hi')
        assert mock_inv.await_count == 2
        assert got.success is True

    async def test_token_written_to_config_dir_on_each_retry(self):
        """config_dir.write_credentials called with each token on retry."""
        gate = _mock_gate(
            account_count=2,
            before_invoke=AsyncMock(side_effect=['tok-a', 'tok-b']),
            detect_cap_hit=MagicMock(side_effect=[True, False]),
            active_account_name='acct-b',
        )
        config_dir = MagicMock()
        config_dir.path = '/tmp/test-config'
        result = make_result()
        with (
            patch(_INVOKE_PATCH, new_callable=AsyncMock, return_value=result),
            patch(_SLEEP_PATCH, new_callable=AsyncMock),
        ):
            await invoke_with_cap_retry(gate, 'lbl', config_dir=config_dir, prompt='hi')
        assert config_dir.write_credentials.call_count == 2
        config_dir.write_credentials.assert_any_call('tok-a')
        config_dir.write_credentials.assert_any_call('tok-b')

    async def test_account_name_changes_between_retries(self):
        """After failover, account_name reflects the new account."""
        gate = _mock_gate(account_count=2)
        gate.before_invoke = AsyncMock(side_effect=['tok-a', 'tok-b'])
        gate.detect_cap_hit = MagicMock(side_effect=[True, False])
        # active_account_name is read:
        #   1st iteration capture, 1st iteration cap-hit logging
        #   2nd iteration capture
        type(gate).active_account_name = PropertyMock(
            side_effect=['acct-a', 'acct-b', 'acct-b'],
        )
        result = make_result()
        with (
            patch(_INVOKE_PATCH, new_callable=AsyncMock, return_value=result),
            patch(_SLEEP_PATCH, new_callable=AsyncMock),
        ):
            got = await invoke_with_cap_retry(gate, 'lbl', prompt='hi')
        assert got.account_name == 'acct-b'

    async def test_three_cap_hits_across_three_accounts(self):
        """3 consecutive cap hits then 4th call succeeds -> 4 invocations."""
        gate = _mock_gate(
            account_count=3,
            before_invoke=AsyncMock(side_effect=['t-a', 't-b', 't-c', 't-a']),
            detect_cap_hit=MagicMock(side_effect=[True, True, True, False]),
            active_account_name='acct-a',
        )
        result = make_result()
        with (
            patch(_INVOKE_PATCH, new_callable=AsyncMock, return_value=result) as mock_inv,
            patch(_SLEEP_PATCH, new_callable=AsyncMock),
        ):
            got = await invoke_with_cap_retry(gate, 'lbl', prompt='hi')
        assert mock_inv.await_count == 4
        assert got.success is True

    async def test_detect_cap_hit_called_with_correct_args(self):
        """detect_cap_hit receives stderr, output, 'claude', and oauth_token from each invocation."""
        gate = _mock_gate(
            account_count=2,
            before_invoke=AsyncMock(side_effect=['tok-a', 'tok-b']),
            detect_cap_hit=MagicMock(side_effect=[True, False]),
            active_account_name='acct-b',
        )
        r1 = make_result(stderr='err1', output='out1')
        r2 = make_result(stderr='err2', output='out2')
        with (
            patch(_INVOKE_PATCH, new_callable=AsyncMock, side_effect=[r1, r2]),
            patch(_SLEEP_PATCH, new_callable=AsyncMock),
        ):
            await invoke_with_cap_retry(gate, 'lbl', prompt='hi')
        calls = gate.detect_cap_hit.call_args_list
        assert calls[0] == call('err1', 'out1', 'claude', oauth_token='tok-a')
        assert calls[1] == call('err2', 'out2', 'claude', oauth_token='tok-b')


# ===================================================================
# TestCapRetryResume
# ===================================================================


@pytest.mark.asyncio
class TestCapRetryResume:
    """Resume logic: session preservation across failover."""

    async def test_cap_hit_with_session_id_resumes(self):
        """Cap hit with session_id -> next invoke gets resume_session_id + CAP_HIT_RESUME_PROMPT."""
        gate = _mock_gate(
            account_count=2,
            before_invoke=AsyncMock(side_effect=['tok-a', 'tok-b']),
            detect_cap_hit=MagicMock(side_effect=[True, False]),
            active_account_name='acct-b',
        )
        capped = make_result(session_id='sess-42')
        ok = make_result()
        with (
            patch(_INVOKE_PATCH, new_callable=AsyncMock, side_effect=[capped, ok]) as mock_inv,
            patch(_SLEEP_PATCH, new_callable=AsyncMock),
        ):
            await invoke_with_cap_retry(gate, 'lbl', prompt='do stuff')
        second = mock_inv.call_args_list[1]
        assert second.kwargs.get('resume_session_id') == 'sess-42'
        assert second.kwargs.get('prompt') == CAP_HIT_RESUME_PROMPT

    async def test_cap_hit_without_session_id_fresh(self):
        """Cap hit without session_id -> next invoke gets original prompt, no resume."""
        gate = _mock_gate(
            account_count=2,
            before_invoke=AsyncMock(side_effect=['tok-a', 'tok-b']),
            detect_cap_hit=MagicMock(side_effect=[True, False]),
            active_account_name='acct-b',
        )
        capped = make_result(session_id='')
        ok = make_result()
        with (
            patch(_INVOKE_PATCH, new_callable=AsyncMock, side_effect=[capped, ok]) as mock_inv,
            patch(_SLEEP_PATCH, new_callable=AsyncMock),
        ):
            await invoke_with_cap_retry(gate, 'lbl', prompt='original')
        second = mock_inv.call_args_list[1]
        assert 'resume_session_id' not in second.kwargs
        assert second.kwargs.get('prompt') == 'original'

    async def test_resume_failure_falls_back_to_fresh(self):
        """Resume fails (success=False, not cap hit) -> falls back to fresh with original prompt."""
        gate = _mock_gate(
            account_count=2,
            before_invoke=AsyncMock(side_effect=['tok-a', 'tok-b', 'tok-a']),
            detect_cap_hit=MagicMock(side_effect=[True, False, False]),
            active_account_name='acct-a',
        )
        capped = make_result(session_id='sess-1')
        resume_fail = make_result(success=False, output='resume broke')
        ok = make_result()
        with (
            patch(_INVOKE_PATCH, new_callable=AsyncMock, side_effect=[capped, resume_fail, ok]) as mock_inv,
            patch(_SLEEP_PATCH, new_callable=AsyncMock),
        ):
            got = await invoke_with_cap_retry(gate, 'lbl', prompt='original')
        assert mock_inv.await_count == 3
        # Second call: resume attempt
        assert mock_inv.call_args_list[1].kwargs.get('resume_session_id') == 'sess-1'
        # Third call: fresh fallback
        third = mock_inv.call_args_list[2]
        assert 'resume_session_id' not in third.kwargs
        assert third.kwargs.get('prompt') == 'original'
        assert got.success is True

    async def test_resume_succeeds_on_second_account(self):
        """Cap hit with session_id, resume succeeds on next account."""
        gate = _mock_gate(
            account_count=2,
            before_invoke=AsyncMock(side_effect=['tok-a', 'tok-b']),
            detect_cap_hit=MagicMock(side_effect=[True, False]),
            active_account_name='acct-b',
        )
        capped = make_result(session_id='sess-x')
        ok = make_result(output='resumed ok')
        with (
            patch(_INVOKE_PATCH, new_callable=AsyncMock, side_effect=[capped, ok]),
            patch(_SLEEP_PATCH, new_callable=AsyncMock),
        ):
            got = await invoke_with_cap_retry(gate, 'lbl', prompt='original')
        assert got.output == 'resumed ok'

    async def test_session_id_preserved_across_multiple_failovers(self):
        """A caps with session, B caps (updates session), C gets resume with B's session."""
        gate = _mock_gate(
            account_count=3,
            before_invoke=AsyncMock(side_effect=['tok-a', 'tok-b', 'tok-c']),
            detect_cap_hit=MagicMock(side_effect=[True, True, False]),
            active_account_name='acct-c',
        )
        r1 = make_result(session_id='sess-A')
        r2 = make_result(session_id='sess-B')  # resume on B produces updated session
        r3 = make_result()
        with (
            patch(_INVOKE_PATCH, new_callable=AsyncMock, side_effect=[r1, r2, r3]) as mock_inv,
            patch(_SLEEP_PATCH, new_callable=AsyncMock),
        ):
            await invoke_with_cap_retry(gate, 'lbl', prompt='original')
        # 2nd call: resume with A's session
        assert mock_inv.call_args_list[1].kwargs.get('resume_session_id') == 'sess-A'
        # 3rd call: resume with B's session (updated)
        assert mock_inv.call_args_list[2].kwargs.get('resume_session_id') == 'sess-B'

    async def test_original_prompt_restored_after_resume_fallback(self):
        """After resume + fallback cycle, original prompt is correctly restored."""
        gate = _mock_gate(
            account_count=2,
            before_invoke=AsyncMock(side_effect=['tok-a', 'tok-b', 'tok-a', 'tok-b']),
            detect_cap_hit=MagicMock(side_effect=[True, False, True, False]),
            active_account_name='acct-b',
        )
        capped1 = make_result(session_id='sess-1')
        resume_fail = make_result(success=False)
        # Second cycle: cap hit again, this time without session_id
        capped2 = make_result(session_id='')
        ok = make_result()
        with (
            patch(_INVOKE_PATCH, new_callable=AsyncMock, side_effect=[capped1, resume_fail, capped2, ok]) as mock_inv,
            patch(_SLEEP_PATCH, new_callable=AsyncMock),
        ):
            await invoke_with_cap_retry(gate, 'lbl', prompt='my original prompt')
        # Call 1: original
        assert mock_inv.call_args_list[0].kwargs.get('prompt') == 'my original prompt'
        # Call 2: resume
        assert mock_inv.call_args_list[1].kwargs.get('prompt') == CAP_HIT_RESUME_PROMPT
        # Call 3: fallback to fresh -> original prompt
        assert mock_inv.call_args_list[2].kwargs.get('prompt') == 'my original prompt'
        # Call 4: fresh again (no session_id on capped2) -> original
        assert mock_inv.call_args_list[3].kwargs.get('prompt') == 'my original prompt'

    async def test_cap_wait_log_survives_non_serializable_soonest_resets_at(self):
        """invoke_with_cap_retry completes when soonest_resets_at is non-JSON-serializable.

        Regression guard for the json.dumps call inside _check_cap_wait
        (called from invoke_with_cap_retry on the cap-hit retry path).
        With soonest_resets_at=MagicMock() the conditional
        ``usage_gate.soonest_resets_at.isoformat()`` branch is taken;
        before the default=str hardening this raised:
          TypeError: Object of type MagicMock is not JSON serializable
        and aborted invoke_with_cap_retry.  After the fix the log call
        must be a no-op for control flow.

        Additionally asserts that the structured cap_wait JSON log is actually
        emitted (verified via json.loads) so a future change that skips the
        _check_cap_wait JSON branch fails loudly instead of silently dropping
        coverage.  Note: logger.warning is called TWICE in this path — once at
        the plain f-string cap-hit message and once at the JSON cap_wait emit —
        so mock_warn.assert_called_once() is intentionally NOT used; instead we
        filter by JSON-decodability and event=='cap_wait'.
        """
        gate = _mock_gate(
            account_count=2,
            before_invoke=AsyncMock(side_effect=['tok-a', 'tok-b']),
            detect_cap_hit=MagicMock(side_effect=[True, False]),
            active_account_name='acct-b',
            soonest_resets_at=MagicMock(),  # truthy + non-serializable
        )
        capped = make_result(session_id='')
        ok = make_result()
        with (
            patch(_INVOKE_PATCH, new_callable=AsyncMock, side_effect=[capped, ok]) as mock_inv,
            patch(_SLEEP_PATCH, new_callable=AsyncMock),
            patch(_LOGGER_WARN_PATCH) as mock_warn,
        ):
            got = await invoke_with_cap_retry(gate, 'lbl', prompt='x')
        assert mock_inv.await_count == 2
        assert got.success

        # Collect all warning calls that decode to a JSON dict with event=='cap_wait'.
        # The co-occurring plain-string cap-hit warning (non-JSON) is expected and
        # will not decode as JSON, so it is filtered out naturally.
        cap_wait_logs = []
        for c in mock_warn.call_args_list:
            arg = c.args[0] if c.args else None
            if not isinstance(arg, str):
                continue
            try:
                payload = json.loads(arg)
            except (TypeError, ValueError):
                continue
            if isinstance(payload, dict) and payload.get('event') == 'cap_wait':
                cap_wait_logs.append(payload)

        assert len(cap_wait_logs) == 1, (
            "Expected exactly ONE JSON cap_wait log from _check_cap_wait; "
            f"got {len(cap_wait_logs)}.  A co-occurring plain-string warning is "
            "expected and filtered out.  If count is 0, the _check_cap_wait JSON "
            "branch was skipped."
        )
        payload = cap_wait_logs[0]
        assert payload['event'] == 'cap_wait'
        assert payload['label'] == 'lbl'
        # soonest_open_at must be a non-None string: proves the
        # usage_gate.soonest_resets_at.isoformat() conditional branch executed AND
        # that json.dumps(..., default=str) stringified the non-serializable MagicMock.
        assert isinstance(payload['soonest_open_at'], str), (
            "soonest_open_at must be a non-None str (MagicMock stringified via "
            "default=str); the .isoformat() branch was not taken or default=str missing"
        )
        assert payload['soonest_open_at'], (
            "soonest_open_at must be non-empty; MagicMock stringified via default=str "
            "cannot produce an empty string"
        )
        # Verify the remaining structured fields are present and numeric.
        assert isinstance(payload.get('elapsed_s'), (int, float)), (
            "elapsed_s must be a numeric value (round(elapsed, 1)) in the cap_wait payload"
        )
        assert isinstance(payload.get('next_probe_in_s'), (int, float)), (
            "next_probe_in_s must be a numeric value (round(cooldown, 1)) in the cap_wait payload"
        )

    async def test_fresh_fallback_regenerates_session_id(self):
        """Fresh fallback after a failed resume regenerates a fresh session_id.

        Regression for the reify-3604 wedge: the resume attempt may have already
        committed the pre-allocated UUID to disk, so reusing it on the fresh
        ``--session-id`` retry would make the CLI exit instantly with
        'Session ID … is already in use'.  The fallback must hand a NEW UUID to
        the fresh invocation (and drop resume_session_id).
        """
        gate = _mock_gate(
            account_count=1,
            before_invoke=AsyncMock(side_effect=['tok', 'tok']),
            detect_cap_hit=MagicMock(side_effect=[False, False]),
            active_account_name='acct',
        )
        # cost_usd defaults to 0.5 in make_result, so the resume failure does NOT
        # trip the zero-cost heuristic — it routes through the resume-fallback branch.
        resume_fail = make_result(success=False, output='resume broke')
        ok = make_result()
        with (
            patch(_INVOKE_PATCH, new_callable=AsyncMock, side_effect=[resume_fail, ok]) as mock_inv,
            patch(_SLEEP_PATCH, new_callable=AsyncMock),
        ):
            await invoke_with_cap_retry(
                gate, 'lbl',
                prompt='p', session_id='sess-orig', resume_session_id='sess-orig',
            )
        assert mock_inv.await_count == 2
        # First call: the resume attempt uses the caller-supplied session id.
        assert mock_inv.call_args_list[0].kwargs.get('resume_session_id') == 'sess-orig'
        # Second call: fresh fallback — resume dropped, session_id regenerated.
        second = mock_inv.call_args_list[1]
        assert 'resume_session_id' not in second.kwargs
        new_sid = second.kwargs.get('session_id')
        assert new_sid, 'fresh fallback must keep a session_id (caller passed one)'
        assert new_sid != 'sess-orig', 'session_id must be regenerated, not reused'
        # Must be a valid UUID (str(uuid.uuid4()) round-trips through UUID()).
        assert str(uuid.UUID(new_sid)) == new_sid


# ===================================================================
# TestCapRetryWedgeGuard
# ===================================================================


@pytest.mark.asyncio
class TestCapRetryWedgeGuard:
    """Wedge guard: a zero-output timed-out call clears resume_session_id
    before the cap-hit branch can re-arm it.

    Covers the esc-task-curator-3/-5/-6 wedge (session c5d446f5-..., 2026-05-27)
    where cap-hit detection + wedge result perpetuated the same broken session.
    """

    async def test_wedge_breaks_cap_hit_perpetuation_of_resume_session_id(self):
        """Cap-hit + wedge result must NOT perpetuate resume_session_id.

        Without the wedge guard, iter 2's cap-hit branch re-sets
        ``invoke_kwargs['resume_session_id'] = result.session_id = 'wedged-X'``
        (the wedge result still carries a session_id from SIGTERM-grace partial
        JSON).  Each subsequent --resume against that orphaned session repeats
        the same hang.

        With the guard, the guard fires *before* the cap-hit branch on iter 2,
        calls ``_reset_for_fresh_retry`` to clear resume, and ``continue``s
        so iter 3 starts completely fresh.

        Iteration trace (with the guard in place):
        - Iter 1: no resume. wedge result. guard skips (no resume yet).
          cap-hit fires → sets resume='wedged-X'. sleep. continue.
        - Iter 2: resume='wedged-X'. wedge result. guard fires →
          _reset_for_fresh_retry clears resume. continue (cap-hit never runs).
        - Iter 3: no resume. ok result. confirm. break.

        Note: with the guard in place, detect_cap_hit is called only on iters 1
        and 3 (not 2), so iter 3 consumes the 2nd side_effect value (True) and
        triggers an additional cap-hit+fresh-retry cycle; iter 4 is the final
        success.  The 4-element side_effects accommodate this correctly.
        The key invariant is resume_ids[2] is None — the wedge-guard-cleared
        iteration must not pass the wedged session forward.
        """
        gate = _mock_gate(
            account_count=1,
            before_invoke=AsyncMock(side_effect=['tok'] * 5),
            detect_cap_hit=MagicMock(side_effect=[True, True, False, False]),
            active_account_name='acct',
        )
        wedge = AgentResult(
            success=False, output='', cost_usd=0.0,
            duration_ms=300_000, turns=0, session_id='wedged-X',
            timed_out=True,
        )
        ok = make_result()
        with (
            patch(_INVOKE_PATCH, new_callable=AsyncMock,
                  side_effect=[wedge, wedge, ok, ok]) as mock_inv,
            patch(_SLEEP_PATCH, new_callable=AsyncMock),
        ):
            await invoke_with_cap_retry(gate, 'lbl', prompt='real-prompt')

        resume_ids = [c.kwargs.get('resume_session_id') for c in mock_inv.call_args_list]
        assert resume_ids[0] is None          # iter 1: no resume before
        assert resume_ids[1] == 'wedged-X'    # iter 2: cap-hit branch set resume from iter 1
        # THE KEY ASSERTION: iter 3 must not resume the wedged session.
        # Without the guard, iter 2's cap-hit branch re-sets resume='wedged-X'
        # (from wedge.session_id), so iter 3 has resume='wedged-X'.
        # With the guard, iter 2 clears resume, so iter 3 starts fresh.
        assert resume_ids[2] is None, (
            f'iter 3 must not resume the wedged session; got {resume_ids[2]!r} '
            f'(full sequence: {resume_ids})'
        )
        # Original prompt must be restored when the guard fires.
        assert mock_inv.call_args_list[2].kwargs['prompt'] == 'real-prompt'

    async def test_wedge_with_caller_set_resume_clears_session_on_next_iteration(
        self, caplog,
    ):
        """Caller-set resume_session_id + zero-output wedge → wedge guard fires.

        The guard's diagnostic log ('zero-output timed-out invocation ...') is
        distinctive from the generic resume-fallback log ('resume failed ...').
        Without the guard, the existing resume-fallback branch (line 681) fires
        and clears resume, but emits the wrong log.  After step-2's wedge-guard
        impl, the guard fires *before* the resume-fallback branch and emits the
        correct log — verified via caplog.
        """
        gate = _mock_gate(
            account_count=1,
            before_invoke=AsyncMock(side_effect=['tok', 'tok']),
            detect_cap_hit=MagicMock(side_effect=[False, False]),
            active_account_name='acct',
        )
        wedge = AgentResult(
            success=False, output='', cost_usd=0.0,
            duration_ms=300_000, turns=0, session_id='wedged-Y',
            timed_out=True,
        )
        ok = make_result()
        with (
            patch(_INVOKE_PATCH, new_callable=AsyncMock,
                  side_effect=[wedge, ok]) as mock_inv,
            patch(_SLEEP_PATCH, new_callable=AsyncMock),
            caplog.at_level(logging.WARNING, logger='shared.cli_invoke'),
        ):
            await invoke_with_cap_retry(
                gate, 'lbl',
                prompt='real-task-prompt',
                resume_session_id='wedged-Y',
            )

        assert mock_inv.await_count == 2
        # Iter 1: caller-set resume; prompt overwritten to CRASH_RECOVERY_RESUME_PROMPT.
        first = mock_inv.call_args_list[0]
        assert first.kwargs.get('resume_session_id') == 'wedged-Y'
        assert first.kwargs.get('prompt') == CRASH_RECOVERY_RESUME_PROMPT
        # Iter 2: guard (or fallback) cleared resume + restored original prompt.
        second = mock_inv.call_args_list[1]
        assert second.kwargs.get('resume_session_id') is None
        assert second.kwargs.get('prompt') == 'real-task-prompt'
        # The wedge guard's distinctive log must be present.
        # Without the guard the generic resume-fallback fires instead, emitting
        # "resume failed (session_id=...)" — which does NOT contain the substring
        # 'zero-output timed-out', so this assertion fails.
        assert any(
            'zero-output timed-out' in record.message
            for record in caplog.records
        ), (
            "Expected the wedge guard's diagnostic log ('zero-output timed-out'); "
            "the generic resume-fallback log does not mention this."
        )

    async def test_happy_path_and_partial_timeout_do_not_trigger_wedge_guard(
        self, caplog,
    ):
        """Negative regression: the wedge guard must not fire on non-wedge results.

        Sub-scenario (a): a successful call with a caller-set resume must not
        be disrupted by the guard.  Sub-scenario (b): a timed-out result that
        did real work (turns>0, cost>0) must not emit the guard's 'zero-output
        timed-out' log.  Both would fail if the guard's condition were
        broadened to fire on any timed_out result.
        """
        # ---- sub-scenario (a): happy-path success ----
        gate_a = _mock_gate(
            account_count=1,
            before_invoke=AsyncMock(return_value='tok'),
            detect_cap_hit=MagicMock(return_value=False),
            active_account_name='acct',
        )
        ok = make_result(success=True, cost_usd=0.5)
        ok._make_result_turns = 3  # not used by guard, just for clarity
        with (
            patch(_INVOKE_PATCH, new_callable=AsyncMock, return_value=ok) as mock_inv_a,
            patch(_SLEEP_PATCH, new_callable=AsyncMock),
        ):
            got_a = await invoke_with_cap_retry(
                gate_a, 'lbl', prompt='p', resume_session_id='sess-ok',
            )
        assert mock_inv_a.await_count == 1       # no retry — succeeded first try
        assert got_a.success is True
        assert mock_inv_a.call_args_list[0].kwargs.get('resume_session_id') == 'sess-ok'

        # ---- sub-scenario (b): timed-out with real work (NOT a wedge) ----
        gate_b = _mock_gate(
            account_count=1,
            before_invoke=AsyncMock(return_value='tok'),
            detect_cap_hit=MagicMock(return_value=False),
            active_account_name='acct',
        )
        partial = AgentResult(
            success=False, output='partial work', cost_usd=0.8,
            duration_ms=300_000, turns=4, session_id='sess-partial',
            timed_out=True,
        )
        partial_ok = make_result()
        with (
            patch(_INVOKE_PATCH, new_callable=AsyncMock,
                  side_effect=[partial, partial_ok]),
            patch(_SLEEP_PATCH, new_callable=AsyncMock),
            caplog.at_level(logging.WARNING, logger='shared.cli_invoke'),
        ):
            await invoke_with_cap_retry(
                gate_b, 'lbl', prompt='p', resume_session_id='sess-partial',
            )
        # The wedge guard must NOT have fired: turns=4 fails the turns==0 check.
        assert not any(
            'zero-output timed-out' in r.message for r in caplog.records
        ), (
            "wedge guard fired on a partial-timeout result (turns=4, cost=0.8); "
            "the guard condition must require turns==0 AND cost_usd==0.0"
        )


@pytest.mark.asyncio
class TestCapRetryCooldown:
    """Exponential backoff cooldown with full-cycle accounting."""

    async def test_first_cap_hit_cooldown_5s(self):
        """First cap hit: cooldown = _CAP_HIT_COOLDOWN_SECS (5.0)."""
        gate = _mock_gate(
            account_count=1,
            before_invoke=AsyncMock(side_effect=['tok', 'tok']),
            detect_cap_hit=MagicMock(side_effect=[True, False]),
            active_account_name='acct',
        )
        result = make_result()
        with (
            patch(_INVOKE_PATCH, new_callable=AsyncMock, return_value=result),
            patch(_SLEEP_PATCH, new_callable=AsyncMock) as mock_sleep,
        ):
            await invoke_with_cap_retry(gate, 'lbl', prompt='hi')
        mock_sleep.assert_awaited_once_with(_CAP_HIT_COOLDOWN_SECS)

    async def test_two_accounts_two_hits_one_full_cycle_10s(self):
        """With 2 accounts, after 2 cap hits (1 full cycle): third hit cooldown = 10s.

        Formula: full_cycles = (consecutive-1) // num_accounts
        Hit 1: (1-1)//2 = 0 -> 5 * 2^0 = 5
        Hit 2: (2-1)//2 = 0 -> 5 * 2^0 = 5
        Hit 3: (3-1)//2 = 1 -> 5 * 2^1 = 10
        """
        gate = _mock_gate(
            account_count=2,
            before_invoke=AsyncMock(side_effect=['t'] * 4),
            detect_cap_hit=MagicMock(side_effect=[True, True, True, False]),
            active_account_name='acct',
        )
        result = make_result()
        with (
            patch(_INVOKE_PATCH, new_callable=AsyncMock, return_value=result),
            patch(_SLEEP_PATCH, new_callable=AsyncMock) as mock_sleep,
        ):
            await invoke_with_cap_retry(gate, 'lbl', prompt='hi')
        sleeps = [c.args[0] for c in mock_sleep.call_args_list]
        assert sleeps == [5.0, 5.0, 10.0]

    async def test_two_accounts_four_hits_two_full_cycles_20s(self):
        """With 2 accounts, after 4 cap hits (2 full cycles): 5th hit cooldown = 20s.

        Hit 1: 5, Hit 2: 5, Hit 3: 10, Hit 4: 10, Hit 5: 20
        """
        gate = _mock_gate(
            account_count=2,
            before_invoke=AsyncMock(side_effect=['t'] * 6),
            detect_cap_hit=MagicMock(side_effect=[True, True, True, True, True, False]),
            active_account_name='acct',
        )
        result = make_result()
        with (
            patch(_INVOKE_PATCH, new_callable=AsyncMock, return_value=result),
            patch(_SLEEP_PATCH, new_callable=AsyncMock) as mock_sleep,
        ):
            await invoke_with_cap_retry(gate, 'lbl', prompt='hi')
        sleeps = [c.args[0] for c in mock_sleep.call_args_list]
        assert sleeps == [5.0, 5.0, 10.0, 10.0, 20.0]

    async def test_cooldown_capped_at_300s(self):
        """Cooldown never exceeds _MAX_CAP_COOLDOWN_SECS (300)."""
        gate = _mock_gate(
            account_count=1,
            before_invoke=AsyncMock(side_effect=['t'] * 20),
            detect_cap_hit=MagicMock(side_effect=[True] * 19 + [False]),
            active_account_name='acct',
        )
        result = make_result()
        with (
            patch(_INVOKE_PATCH, new_callable=AsyncMock, return_value=result),
            patch(_SLEEP_PATCH, new_callable=AsyncMock) as mock_sleep,
        ):
            await invoke_with_cap_retry(gate, 'lbl', prompt='hi')
        sleeps = [c.args[0] for c in mock_sleep.call_args_list]
        assert all(s <= _MAX_CAP_COOLDOWN_SECS for s in sleeps)
        # With 1 account, hit 7: 5*2^6=320 -> capped at 300
        assert 300.0 in sleeps

    async def test_single_account_doubles_each_hit(self):
        """With 1 account, each cap hit is a full cycle -> cooldown doubles each time."""
        gate = _mock_gate(
            account_count=1,
            before_invoke=AsyncMock(side_effect=['t'] * 6),
            detect_cap_hit=MagicMock(side_effect=[True] * 5 + [False]),
            active_account_name='acct',
        )
        result = make_result()
        with (
            patch(_INVOKE_PATCH, new_callable=AsyncMock, return_value=result),
            patch(_SLEEP_PATCH, new_callable=AsyncMock) as mock_sleep,
        ):
            await invoke_with_cap_retry(gate, 'lbl', prompt='hi')
        sleeps = [c.args[0] for c in mock_sleep.call_args_list]
        # Hit 1: (0)//1=0 -> 5, Hit 2: (1)//1=1 -> 10, Hit 3: 20, Hit 4: 40, Hit 5: 80
        assert sleeps == [5.0, 10.0, 20.0, 40.0, 80.0]

    async def test_formula_matches(self):
        """Verify the exact formula: min(5 * 2^((consecutive-1)//num_accounts), 300)."""
        num_accounts = 3
        gate = _mock_gate(
            account_count=num_accounts,
            before_invoke=AsyncMock(side_effect=['t'] * 10),
            detect_cap_hit=MagicMock(side_effect=[True] * 9 + [False]),
            active_account_name='acct',
        )
        result = make_result()
        with (
            patch(_INVOKE_PATCH, new_callable=AsyncMock, return_value=result),
            patch(_SLEEP_PATCH, new_callable=AsyncMock) as mock_sleep,
        ):
            await invoke_with_cap_retry(gate, 'lbl', prompt='hi')
        sleeps = [c.args[0] for c in mock_sleep.call_args_list]
        for i, s in enumerate(sleeps):
            consecutive = i + 1
            expected = min(
                _CAP_HIT_COOLDOWN_SECS * (2 ** ((consecutive - 1) // num_accounts)),
                _MAX_CAP_COOLDOWN_SECS,
            )
            assert s == expected, f'Hit {consecutive}: expected {expected}, got {s}'


# ===================================================================
# TestCapRetryBudget
# ===================================================================


@pytest.mark.asyncio
class TestCapRetryBudget:
    """Session budget enforcement via UsageGate."""

    async def test_budget_exceeded_before_first_invoke(self):
        """SessionBudgetExhausted raised when cumulative cost >= budget at invoke time."""
        gate = make_gate(['a'], session_budget_usd=1.0)
        # Simulate prior cost accumulation
        gate._cumulative_cost = 1.5
        with (
            patch(_INVOKE_PATCH, new_callable=AsyncMock) as mock_inv,
            pytest.raises(SessionBudgetExhausted),
        ):
            await invoke_with_cap_retry(gate, 'lbl', prompt='hi')
        mock_inv.assert_not_awaited()

    async def test_budget_not_exceeded_completes(self):
        """Under budget -> completes normally."""
        gate = make_gate(['a'], session_budget_usd=10.0)
        result = make_result(cost_usd=1.0)
        with patch(_INVOKE_PATCH, new_callable=AsyncMock, return_value=result):
            got = await invoke_with_cap_retry(gate, 'lbl', prompt='hi')
        assert got.success is True

    async def test_budget_exceeded_during_retry_loop(self):
        """Budget exceeded when gate.before_invoke raises during retry -> propagates."""
        gate = _mock_gate(account_count=2)
        gate.before_invoke = AsyncMock(
            side_effect=['tok-a', SessionBudgetExhausted(5.0)],
        )
        gate.detect_cap_hit = MagicMock(side_effect=[True])
        capped = make_result(cost_usd=0.1)
        with (
            patch(_INVOKE_PATCH, new_callable=AsyncMock, return_value=capped),
            patch(_SLEEP_PATCH, new_callable=AsyncMock),
            pytest.raises(SessionBudgetExhausted),
        ):
            await invoke_with_cap_retry(gate, 'lbl', prompt='hi')


# ===================================================================
# TestCapRetryCostStore
# ===================================================================


@pytest.mark.asyncio
class TestCapRetryCostStore:
    """CostStore integration: invocations and events."""

    async def test_save_invocation_correct_params(self):
        """save_invocation called with all correct parameters on success."""
        gate = _mock_gate(active_account_name='acct-a')
        cost_store = MagicMock()
        cost_store.save_invocation = AsyncMock()
        result = make_result(
            cost_usd=3.0, duration_ms=4000,
            input_tokens=1000, output_tokens=500,
            cache_read_tokens=200, cache_create_tokens=50,
        )
        with patch(_INVOKE_PATCH, new_callable=AsyncMock, return_value=result):
            await invoke_with_cap_retry(
                gate, 'lbl',
                cost_store=cost_store, run_id='r', task_id='t',
                project_id='p', role='reviewer',
                prompt='hi', model='haiku',
            )
        cost_store.save_invocation.assert_awaited_once()
        kw = cost_store.save_invocation.call_args.kwargs
        assert kw['model'] == 'haiku'
        assert kw['role'] == 'reviewer'
        assert kw['cost_usd'] == 3.0
        assert kw['capped'] is False
        assert kw['account_name'] == 'acct-a'

    async def test_save_account_event_on_cap_hit(self):
        """save_account_event called with cap_hit on cap detection."""
        gate = _mock_gate(
            account_count=2,
            before_invoke=AsyncMock(side_effect=['tok-a', 'tok-b']),
            detect_cap_hit=MagicMock(side_effect=[True, False]),
            active_account_name='acct-b',
        )
        cost_store = MagicMock()
        cost_store.save_invocation = AsyncMock()
        cost_store.save_account_event = AsyncMock()
        result = make_result()
        with (
            patch(_INVOKE_PATCH, new_callable=AsyncMock, return_value=result),
            patch(_SLEEP_PATCH, new_callable=AsyncMock),
        ):
            await invoke_with_cap_retry(
                gate, 'my-label',
                cost_store=cost_store, run_id='r1', project_id='p1',
                prompt='hi',
            )
        cost_store.save_account_event.assert_awaited_once()
        kw = cost_store.save_account_event.call_args.kwargs
        assert kw['event_type'] == 'cap_hit'
        assert kw['details'] == 'my-label'
        assert 'created_at' in kw

    async def test_save_invocation_exception_swallowed(self, caplog):
        """save_invocation exception is swallowed (logged as warning)."""
        gate = _mock_gate()
        cost_store = MagicMock()
        cost_store.save_invocation = AsyncMock(side_effect=RuntimeError('db boom'))
        result = make_result()
        with (
            patch(_INVOKE_PATCH, new_callable=AsyncMock, return_value=result),
            caplog.at_level(logging.WARNING),
        ):
            got = await invoke_with_cap_retry(
                gate, 'lbl', cost_store=cost_store, prompt='hi',
            )
        assert got.success is True
        assert 'Failed to save invocation cost' in caplog.text

    async def test_save_account_event_exception_swallowed(self, caplog):
        """save_account_event exception is swallowed; retry still happens."""
        gate = _mock_gate(
            account_count=2,
            before_invoke=AsyncMock(side_effect=['tok-a', 'tok-b']),
            detect_cap_hit=MagicMock(side_effect=[True, False]),
            active_account_name='acct-b',
        )
        cost_store = MagicMock()
        cost_store.save_invocation = AsyncMock()
        cost_store.save_account_event = AsyncMock(side_effect=RuntimeError('db error'))
        result = make_result()
        with (
            patch(_INVOKE_PATCH, new_callable=AsyncMock, return_value=result),
            patch(_SLEEP_PATCH, new_callable=AsyncMock),
            caplog.at_level(logging.WARNING),
        ):
            got = await invoke_with_cap_retry(
                gate, 'lbl', cost_store=cost_store, prompt='hi',
            )
        assert got.success is True
        assert 'Failed to save cap_hit event' in caplog.text
        # save_invocation still called on the success path
        cost_store.save_invocation.assert_awaited_once()

    async def test_no_error_when_cost_store_none(self):
        """No crash when cost_store is None (default)."""
        gate = _mock_gate()
        result = make_result()
        with patch(_INVOKE_PATCH, new_callable=AsyncMock, return_value=result):
            got = await invoke_with_cap_retry(gate, 'lbl', prompt='hi')
        assert got.success is True

    async def test_model_defaults_to_opus(self):
        """model defaults to 'opus' when not in invoke_kwargs."""
        gate = _mock_gate()
        cost_store = MagicMock()
        cost_store.save_invocation = AsyncMock()
        result = make_result()
        with patch(_INVOKE_PATCH, new_callable=AsyncMock, return_value=result):
            await invoke_with_cap_retry(
                gate, 'lbl', cost_store=cost_store, prompt='hi',
            )
        kw = cost_store.save_invocation.call_args.kwargs
        assert kw['model'] == 'opus'


# ===================================================================
# TestCapRetryEdgeCases
# ===================================================================


@pytest.mark.asyncio
class TestCapRetryEdgeCases:
    """Edge cases and boundary conditions."""

    async def test_empty_stderr_and_output_no_cap_hit(self):
        """Empty stderr and output -> no cap hit detected."""
        gate = _mock_gate()
        result = make_result(stderr='', output='')
        with patch(_INVOKE_PATCH, new_callable=AsyncMock, return_value=result):
            got = await invoke_with_cap_retry(gate, 'lbl', prompt='hi')
        gate.detect_cap_hit.assert_called_once()
        assert got.success is True

    async def test_cap_hit_on_very_first_invocation(self):
        """Cap hit on the very first invocation (no prior success)."""
        gate = _mock_gate(
            account_count=1,
            before_invoke=AsyncMock(side_effect=['tok', 'tok']),
            detect_cap_hit=MagicMock(side_effect=[True, False]),
            active_account_name='acct',
        )
        result = make_result()
        with (
            patch(_INVOKE_PATCH, new_callable=AsyncMock, return_value=result) as mock_inv,
            patch(_SLEEP_PATCH, new_callable=AsyncMock),
        ):
            got = await invoke_with_cap_retry(gate, 'lbl', prompt='hi')
        assert mock_inv.await_count == 2
        assert got.success is True

    async def test_multiple_cap_patterns_still_one_cap_hit(self):
        """Multiple cap patterns in same output -> detect_cap_hit returns True once per call."""
        gate = _mock_gate(
            account_count=1,
            before_invoke=AsyncMock(side_effect=['tok', 'tok']),
            detect_cap_hit=MagicMock(side_effect=[True, False]),
            active_account_name='acct',
        )
        # Result has multiple cap patterns, but detect_cap_hit is only called once per loop
        result = make_result(
            stderr="You've hit your limit. You've used all.",
            output='usage limit reset',
        )
        with (
            patch(_INVOKE_PATCH, new_callable=AsyncMock, return_value=result),
            patch(_SLEEP_PATCH, new_callable=AsyncMock),
        ):
            await invoke_with_cap_retry(gate, 'lbl', prompt='hi')
        # detect_cap_hit called exactly twice (once per loop iteration)
        assert gate.detect_cap_hit.call_count == 2

    async def test_config_dir_none_no_write_credentials(self):
        """config_dir=None -> write_credentials never called."""
        gate = _mock_gate(before_invoke=AsyncMock(return_value='tok'))
        result = make_result()
        with patch(_INVOKE_PATCH, new_callable=AsyncMock, return_value=result):
            await invoke_with_cap_retry(gate, 'lbl', prompt='hi')
        # No config_dir => no crash, no write_credentials call

    async def test_no_gate_no_config_dir_write(self):
        """oauth_token is None (no gate) -> config_dir.write_credentials not called."""
        config_dir = MagicMock()
        config_dir.path = '/tmp/test'
        result = make_result()
        with patch(_INVOKE_PATCH, new_callable=AsyncMock, return_value=result):
            await invoke_with_cap_retry(
                None, 'lbl', config_dir=config_dir, prompt='hi',
            )
        config_dir.write_credentials.assert_not_called()

    async def test_invoke_kwargs_mutated_correctly(self):
        """invoke_kwargs: resume_session_id added/removed, prompt swapped."""
        gate = _mock_gate(
            account_count=2,
            before_invoke=AsyncMock(side_effect=['t-a', 't-b', 't-a']),
            detect_cap_hit=MagicMock(side_effect=[True, False, False]),
            active_account_name='acct',
        )
        capped = make_result(session_id='sess-1')
        resume_fail = make_result(success=False)
        ok = make_result()
        with (
            patch(_INVOKE_PATCH, new_callable=AsyncMock, side_effect=[capped, resume_fail, ok]) as mock_inv,
            patch(_SLEEP_PATCH, new_callable=AsyncMock),
        ):
            await invoke_with_cap_retry(gate, 'lbl', prompt='original')
        calls = mock_inv.call_args_list
        # Call 1: original prompt, no resume
        assert calls[0].kwargs.get('prompt') == 'original'
        assert 'resume_session_id' not in calls[0].kwargs
        # Call 2: resume with session
        assert calls[1].kwargs.get('resume_session_id') == 'sess-1'
        assert calls[1].kwargs.get('prompt') == CAP_HIT_RESUME_PROMPT
        # Call 3: fresh fallback, resume cleared
        assert 'resume_session_id' not in calls[2].kwargs
        assert calls[2].kwargs.get('prompt') == 'original'

    async def test_original_prompt_preserved_after_multiple_cycles(self):
        """Original prompt preserved even after multiple resume/fresh cycles."""
        gate = _mock_gate(
            account_count=1,
            before_invoke=AsyncMock(side_effect=['t'] * 5),
            detect_cap_hit=MagicMock(side_effect=[True, False, True, False, False]),
            active_account_name='acct',
        )
        # Cycle 1: cap with session -> resume fails -> fresh
        # Cycle 2: cap without session -> fresh succeeds
        r1 = make_result(session_id='s1')
        r2 = make_result(success=False)  # resume fail
        r3 = make_result(session_id='')  # cap hit, no session
        r4 = make_result()  # success
        with (
            patch(_INVOKE_PATCH, new_callable=AsyncMock, side_effect=[r1, r2, r3, r4]) as mock_inv,
            patch(_SLEEP_PATCH, new_callable=AsyncMock),
        ):
            await invoke_with_cap_retry(gate, 'lbl', prompt='precious prompt')
        # Fresh invocations always get the original prompt
        assert mock_inv.call_args_list[0].kwargs['prompt'] == 'precious prompt'
        assert mock_inv.call_args_list[1].kwargs['prompt'] == CAP_HIT_RESUME_PROMPT
        assert mock_inv.call_args_list[2].kwargs['prompt'] == 'precious prompt'
        assert mock_inv.call_args_list[3].kwargs['prompt'] == 'precious prompt'

    async def test_zero_cost_cli_error_not_treated_as_cap(self):
        """A recognised CLI error (zero-cost instant exit) is NOT counted as a cap.

        Regression for reify-3604: ``claude --session-id <X>`` on a reused UUID
        exits in ~300ms with empty cost and 'Session ID … is already in use'.
        The zero-cost heuristic must recognise that as a concrete CLI error and
        fall through (no cap mark, no sleep, no unbounded retry) so the caller
        gets the failed result for normal verify/steward handling.
        """
        gate = _mock_gate(
            account_count=1,
            detect_cap_hit=MagicMock(return_value=False),
        )
        gate._handle_cap_detected = MagicMock(return_value=True)
        result = make_result(
            success=False, cost_usd=0, turns=0, duration_ms=300,
            stderr='Error: Session ID x is already in use.',
        )
        with (
            patch(_INVOKE_PATCH, new_callable=AsyncMock, return_value=result) as mock_inv,
            patch(_SLEEP_PATCH, new_callable=AsyncMock) as mock_sleep,
        ):
            got = await invoke_with_cap_retry(gate, 'lbl', prompt='hi')
        # Single invocation: no cap-driven retry.
        mock_inv.assert_awaited_once()
        # The cap path was not taken.
        gate._handle_cap_detected.assert_not_called()
        mock_sleep.assert_not_awaited()
        # The failed result is returned to the caller verbatim.
        assert got is result
        assert got.success is False

    async def test_zero_cost_unknown_message_still_cap(self):
        """A zero-cost instant exit with an UNRECOGNISED message is still a cap.

        Guards against over-narrowing the new CLI-error carve-out: when stderr is
        empty (and output carries no known CLI-error marker), the heuristic must
        still treat the result as a cap hit and fail over.
        """
        gate = _mock_gate(
            account_count=1,
            before_invoke=AsyncMock(side_effect=['tok', 'tok']),
            detect_cap_hit=MagicMock(side_effect=[False, False]),
            active_account_name='acct',
        )
        gate._handle_cap_detected = MagicMock(return_value=True)
        capped = make_result(
            success=False, cost_usd=0, turns=0, duration_ms=300, stderr='',
        )
        ok = make_result()
        with (
            patch(_INVOKE_PATCH, new_callable=AsyncMock, side_effect=[capped, ok]) as mock_inv,
            patch(_SLEEP_PATCH, new_callable=AsyncMock) as mock_sleep,
        ):
            got = await invoke_with_cap_retry(gate, 'lbl', prompt='hi')
        assert mock_inv.await_count == 2
        gate._handle_cap_detected.assert_called_once()
        mock_sleep.assert_awaited_once()
        assert got.success is True


# ===================================================================
# TestCapRetryTimingAndSequence
# ===================================================================


@pytest.mark.asyncio
class TestCapRetryTimingAndSequence:
    """Verify exact ordering: invoke -> detect -> cost_event -> sleep -> invoke."""

    async def test_invoke_detect_sleep_invoke_sequence(self):
        """Verify invoke -> detect_cap_hit -> sleep -> invoke on cap-hit retry."""
        call_order = []

        gate = _mock_gate(
            account_count=1,
            before_invoke=AsyncMock(side_effect=['tok', 'tok']),
            active_account_name='acct',
        )
        gate.detect_cap_hit = MagicMock(
            side_effect=lambda *a, **kw: (call_order.append('detect'), [True, False][len([x for x in call_order if x == 'detect']) - 1])[1],
        )

        r1 = make_result()
        r2 = make_result()

        async def invoke_side_effect(**kwargs):
            call_order.append('invoke')
            return [r1, r2][len([x for x in call_order if x == 'invoke']) - 1]

        async def sleep_side_effect(duration):
            call_order.append('sleep')

        with (
            patch(_INVOKE_PATCH, new_callable=AsyncMock, side_effect=invoke_side_effect),
            patch(_SLEEP_PATCH, new_callable=AsyncMock, side_effect=sleep_side_effect),
        ):
            await invoke_with_cap_retry(gate, 'lbl', prompt='hi')

        assert call_order == ['invoke', 'detect', 'sleep', 'invoke', 'detect']

    async def test_no_sleep_on_success(self):
        """No sleep when invocation succeeds on first try."""
        gate = _mock_gate()
        result = make_result()
        with (
            patch(_INVOKE_PATCH, new_callable=AsyncMock, return_value=result),
            patch(_SLEEP_PATCH, new_callable=AsyncMock) as mock_sleep,
        ):
            await invoke_with_cap_retry(gate, 'lbl', prompt='hi')
        mock_sleep.assert_not_awaited()

    async def test_sleep_after_cost_event_before_next_invoke(self):
        """Sleep happens AFTER save_account_event, BEFORE next invoke."""
        call_order = []

        gate = _mock_gate(
            account_count=1,
            before_invoke=AsyncMock(side_effect=['tok', 'tok']),
            detect_cap_hit=MagicMock(side_effect=[True, False]),
            active_account_name='acct',
        )

        cost_store = MagicMock()

        async def save_event(**kwargs):
            call_order.append('save_event')

        cost_store.save_account_event = AsyncMock(side_effect=save_event)
        cost_store.save_invocation = AsyncMock()

        r1 = make_result()
        r2 = make_result()
        invoke_count = 0

        async def invoke_side_effect(**kwargs):
            nonlocal invoke_count
            invoke_count += 1
            call_order.append(f'invoke_{invoke_count}')
            return [r1, r2][invoke_count - 1]

        async def sleep_side_effect(duration):
            call_order.append('sleep')

        with (
            patch(_INVOKE_PATCH, new_callable=AsyncMock, side_effect=invoke_side_effect),
            patch(_SLEEP_PATCH, new_callable=AsyncMock, side_effect=sleep_side_effect),
        ):
            await invoke_with_cap_retry(
                gate, 'lbl', cost_store=cost_store, prompt='hi',
            )

        # Order: invoke_1 -> save_event -> sleep -> invoke_2
        assert call_order.index('save_event') < call_order.index('sleep')
        assert call_order.index('sleep') < call_order.index('invoke_2')


# ===================================================================
# TestCapRetryWithRealGate
# ===================================================================


@pytest.mark.asyncio
class TestCapRetryWithRealGate:
    """Tests using a real UsageGate (from make_gate) instead of MagicMock."""

    async def test_real_gate_success_path(self):
        """Real gate: single account, no cap -> success."""
        gate = make_gate(['alpha'])
        result = make_result(cost_usd=0.75)
        with patch(_INVOKE_PATCH, new_callable=AsyncMock, return_value=result):
            got = await invoke_with_cap_retry(gate, 'lbl', prompt='hi')
        assert got.success is True
        assert got.account_name == 'alpha'
        assert gate.cumulative_cost == 0.75

    async def test_real_gate_failover(self):
        """Real gate: first account caps, second account succeeds."""
        gate = make_gate(['alpha', 'beta'])
        capped = make_result(
            success=True,
            stderr="You've hit your usage limit. resets in 1h",
            output='partial',
        )
        ok = make_result(output='complete')

        call_count = 0

        async def invoke_side_effect(**kwargs):
            nonlocal call_count
            call_count += 1
            if call_count == 1:
                return capped
            return ok

        with (
            patch(_INVOKE_PATCH, new_callable=AsyncMock, side_effect=invoke_side_effect),
            patch(_SLEEP_PATCH, new_callable=AsyncMock),
        ):
            got = await invoke_with_cap_retry(gate, 'lbl', prompt='hi')
        assert got.output == 'complete'
        assert got.account_name == 'beta'

    async def test_real_gate_budget_enforcement(self):
        """Real gate with budget: raises when exceeded."""
        gate = make_gate(['alpha'], session_budget_usd=0.50)
        # First call consumes 0.40
        r1 = make_result(cost_usd=0.40)
        with patch(_INVOKE_PATCH, new_callable=AsyncMock, return_value=r1):
            await invoke_with_cap_retry(gate, 'lbl', prompt='hi')
        # Cumulative = 0.40; next call costs 0.20 -> cumulative = 0.60 > 0.50
        # But budget check happens BEFORE invoke based on cumulative_cost
        # 0.40 < 0.50 so the second invoke_with_cap_retry will still proceed
        r2 = make_result(cost_usd=0.20)
        with patch(_INVOKE_PATCH, new_callable=AsyncMock, return_value=r2):
            await invoke_with_cap_retry(gate, 'lbl', prompt='hi')
        # Now cumulative = 0.60 >= 0.50, so the third should raise
        with pytest.raises(SessionBudgetExhausted), patch(_INVOKE_PATCH, new_callable=AsyncMock):
            await invoke_with_cap_retry(gate, 'lbl', prompt='hi')


# ===================================================================
# TestAllAccountsCappedException
# ===================================================================


class TestAllAccountsCappedException:
    """AllAccountsCappedException: attributes and message format."""

    def test_attributes_accessible(self):
        """Exception stores retries, elapsed_secs, label as attributes."""
        exc = AllAccountsCappedException(retries=5, elapsed_secs=120.5, label='my-task')
        assert exc.retries == 5
        assert exc.elapsed_secs == 120.5
        assert exc.label == 'my-task'

    def test_message_includes_all_three(self):
        """Exception message includes retries, elapsed_secs, and label."""
        exc = AllAccountsCappedException(retries=20, elapsed_secs=3601.0, label='Task 7')
        msg = str(exc)
        assert '20' in msg
        assert '3601' in msg
        assert 'Task 7' in msg

    def test_is_exception(self):
        """AllAccountsCappedException is an Exception subclass."""
        exc = AllAccountsCappedException(retries=1, elapsed_secs=0.0, label='x')
        assert isinstance(exc, Exception)

    def test_default_constants_accessible(self):
        """Module-level defaults are accessible from cli_invoke."""
        from shared.cli_invoke import _DEFAULT_CAP_WAIT_SANITY_SECS  # noqa: PLC0415
        assert _DEFAULT_CAP_WAIT_SANITY_SECS == 14 * 86400  # 1_209_600


# ===================================================================
# TestCapRetrySanityBound
# ===================================================================


@pytest.mark.asyncio
class TestCapRetrySanityBound:
    """cap_wait_sanity_secs guard: raise AllAccountsCappedException when elapsed exceeds limit."""

    async def test_raises_when_sanity_exceeded(self):
        """When elapsed > cap_wait_sanity_secs, raises AllAccountsCappedException."""
        gate = _mock_gate(
            account_count=1,
            before_invoke=AsyncMock(side_effect=['tok'] * 5),
            detect_cap_hit=MagicMock(return_value=True),
            active_account_name='acct',
        )
        result = make_result()
        # First call → 0.0 (retry_start), all subsequent → 4000.0
        # elapsed = 4000.0 > cap_wait_sanity_secs=3600.0 → raise
        monotonic_values = itertools.chain([0.0], itertools.repeat(4000.0))

        with (
            patch(_INVOKE_PATCH, new_callable=AsyncMock, return_value=result),
            patch(_SLEEP_PATCH, new_callable=AsyncMock),
            patch('shared.cli_invoke.time.monotonic', side_effect=monotonic_values),
            pytest.raises(AllAccountsCappedException) as exc_info,
        ):
            await invoke_with_cap_retry(
                gate, 'deadline-task',
                cap_wait_sanity_secs=3600.0,
                prompt='hi',
            )
        exc = exc_info.value
        assert exc.elapsed_secs > 3600.0
        assert exc.label == 'deadline-task'
        assert exc.retries == 1

    async def test_sanity_bound_fires_after_many_hits(self):
        """Sanity bound fires regardless of number of cap hits."""
        gate = _mock_gate(
            account_count=1,
            before_invoke=AsyncMock(side_effect=['tok'] * 10),
            detect_cap_hit=MagicMock(return_value=True),
            active_account_name='acct',
        )
        result = make_result()
        # First call → 0.0 (retry_start), all subsequent → 15.0
        # elapsed = 15.0 > cap_wait_sanity_secs=10.0 after first hit
        monotonic_values = itertools.chain([0.0], itertools.repeat(15.0))

        with (
            patch(_INVOKE_PATCH, new_callable=AsyncMock, return_value=result),
            patch(_SLEEP_PATCH, new_callable=AsyncMock),
            patch('shared.cli_invoke.time.monotonic', side_effect=monotonic_values),
            pytest.raises(AllAccountsCappedException) as exc_info,
        ):
            await invoke_with_cap_retry(
                gate, 'deadline-first-task',
                cap_wait_sanity_secs=10.0,
                prompt='hi',
            )
        exc = exc_info.value
        assert exc.retries == 1, f'Expected 1 retry (sanity bound), got {exc.retries}'
        assert exc.elapsed_secs > 10.0, f'elapsed_secs should exceed sanity_secs, got {exc.elapsed_secs}'
        assert exc.label == 'deadline-first-task'

    async def test_no_exception_when_within_sanity_bound(self):
        """When elapsed is well under cap_wait_sanity_secs, no exception is raised."""
        gate = _mock_gate(
            account_count=1,
            before_invoke=AsyncMock(side_effect=['tok'] * 3),
            detect_cap_hit=MagicMock(side_effect=[True, True, False]),
            active_account_name='acct',
        )
        result = make_result()
        # All monotonic calls return 0.0 so elapsed is always 0.0 < sanity_secs=3600.0
        with (
            patch(_INVOKE_PATCH, new_callable=AsyncMock, return_value=result),
            patch(_SLEEP_PATCH, new_callable=AsyncMock),
            patch('shared.cli_invoke.time.monotonic', return_value=0.0),
        ):
            got = await invoke_with_cap_retry(
                gate, 'lbl',
                cap_wait_sanity_secs=3600.0,
                prompt='hi',
            )
        assert got.success is True


# ===================================================================
# TestCapWaitSanitySecs
# ===================================================================


@pytest.mark.asyncio
class TestCapWaitSanitySecs:
    """cap_wait_sanity_secs: the 14-day patient-wait sanity bound for cap-hit retries."""

    async def test_exact_detect_raises_when_sanity_exceeded(self):
        """Exact-detect branch raises AllAccountsCappedException when elapsed > cap_wait_sanity_secs."""
        gate = _mock_gate(
            account_count=1,
            before_invoke=AsyncMock(side_effect=['tok'] * 5),
            detect_cap_hit=MagicMock(return_value=True),
            active_account_name='acct',
        )
        result = make_result()
        # First call → 0.0 (retry_start), all subsequent → 15.0
        # → elapsed = 15.0 > cap_wait_sanity_secs=10.0 after first cap hit
        monotonic_values = itertools.chain([0.0], itertools.repeat(15.0))
        with (
            patch(_INVOKE_PATCH, new_callable=AsyncMock, return_value=result),
            patch(_SLEEP_PATCH, new_callable=AsyncMock),
            patch('shared.cli_invoke.time.monotonic', side_effect=monotonic_values),
            pytest.raises(AllAccountsCappedException) as exc_info,
        ):
            await invoke_with_cap_retry(
                gate, 'sanity-task',
                cap_wait_sanity_secs=10.0,
                prompt='hi',
            )
        exc = exc_info.value
        assert exc.elapsed_secs > 10.0
        assert exc.label == 'sanity-task'
        assert exc.retries == 1

    async def test_heuristic_raises_when_sanity_exceeded(self):
        """Heuristic branch raises AllAccountsCappedException when elapsed > cap_wait_sanity_secs."""
        gate = _mock_gate(
            account_count=1,
            before_invoke=AsyncMock(side_effect=['tok'] * 5),
            detect_cap_hit=MagicMock(return_value=False),
            active_account_name='acct',
        )
        gate._handle_cap_detected = MagicMock(return_value=True)
        heuristic_result = AgentResult(
            success=False, output='Usage limit reached',
            cost_usd=0.0, turns=1, duration_ms=100,
        )
        monotonic_values = itertools.chain([0.0], itertools.repeat(15.0))
        with (
            patch(_INVOKE_PATCH, new_callable=AsyncMock, return_value=heuristic_result),
            patch(_SLEEP_PATCH, new_callable=AsyncMock),
            patch('shared.cli_invoke.time.monotonic', side_effect=monotonic_values),
            pytest.raises(AllAccountsCappedException) as exc_info,
        ):
            await invoke_with_cap_retry(
                gate, 'heuristic-sanity-task',
                cap_wait_sanity_secs=10.0,
                prompt='hi',
            )
        exc = exc_info.value
        assert exc.elapsed_secs > 10.0
        assert exc.label == 'heuristic-sanity-task'
        assert exc.retries == 1


# ===================================================================
# TestCapRetryHeuristicBranch
# ===================================================================


@pytest.mark.asyncio
class TestCapRetryHeuristicBranch:
    """Heuristic cap-hit branch (zero-cost instant exit).

    The only raising bound is cap_wait_sanity_secs.
    """

    def _make_heuristic_result(self) -> AgentResult:
        """Zero-cost instant exit result that triggers the heuristic branch."""
        return AgentResult(
            success=False,
            output='Usage limit reached',
            cost_usd=0.0,
            turns=1,
            duration_ms=100,
        )

    async def test_heuristic_succeeds_count_independent(self):
        """1 heuristic hit then success does not raise."""
        gate = _mock_gate(
            account_count=1,
            before_invoke=AsyncMock(side_effect=['tok'] * 2),
            detect_cap_hit=MagicMock(return_value=False),
            active_account_name='acct',
        )
        gate._handle_cap_detected = MagicMock()
        heuristic_result = self._make_heuristic_result()
        ok_result = make_result()
        with (
            patch(_INVOKE_PATCH, new_callable=AsyncMock, side_effect=[heuristic_result, ok_result]) as mock_inv,
            patch(_SLEEP_PATCH, new_callable=AsyncMock),
        ):
            got = await invoke_with_cap_retry(
                gate, 'lbl', prompt='hi',
            )
        assert got.success is True
        assert mock_inv.await_count == 2

    async def test_heuristic_sanity_exceeded(self):
        """Heuristic branch raises when elapsed > cap_wait_sanity_secs."""
        gate = _mock_gate(
            account_count=1,
            before_invoke=AsyncMock(side_effect=['tok'] * 10),
            detect_cap_hit=MagicMock(return_value=False),  # pattern branch skipped
            active_account_name='acct',
        )
        gate._handle_cap_detected = MagicMock(return_value=True)
        heuristic_result = self._make_heuristic_result()
        # First call → 0.0 (retry_start), subsequent calls → 4000.0
        # elapsed = 4000.0 > cap_wait_sanity_secs=3600.0 → raise
        monotonic_values = itertools.chain([0.0], itertools.repeat(4000.0))

        with (
            patch(_INVOKE_PATCH, new_callable=AsyncMock, return_value=heuristic_result),
            patch(_SLEEP_PATCH, new_callable=AsyncMock),
            patch('shared.cli_invoke.time.monotonic', side_effect=monotonic_values),
            pytest.raises(AllAccountsCappedException) as exc_info,
        ):
            await invoke_with_cap_retry(
                gate, 'heuristic-deadline-task',
                cap_wait_sanity_secs=3600.0,
                prompt='hi',
            )
        exc = exc_info.value
        assert exc.retries == 1, f'Expected 1 retry (sanity bound), got {exc.retries}'
        assert exc.elapsed_secs > 3600.0, (
            f'elapsed_secs should exceed 3600.0 sanity_secs, got {exc.elapsed_secs}'
        )
        assert exc.label == 'heuristic-deadline-task'


# ===================================================================
# TestCapRetrySanityGuardLogging
# ===================================================================


@pytest.mark.asyncio
class TestCapRetrySanityGuardLogging:
    """Verify logger.error is emitted with diagnostic info before raising."""

    async def test_error_logged_before_sanity_raise(self, caplog):
        """logger.error includes label and elapsed time before AllAccountsCappedException."""
        gate = _mock_gate(
            account_count=2,
            before_invoke=AsyncMock(side_effect=['tok'] * 5),
            detect_cap_hit=MagicMock(return_value=True),
            active_account_name='acct',
        )
        result = make_result()
        # First call → 0.0 (retry_start), all subsequent → 4000.0
        # → elapsed = 4000.0 > cap_wait_sanity_secs=3600.0 after just 1 cap hit
        monotonic_values = itertools.chain([0.0], itertools.repeat(4000.0))
        with (
            patch(_INVOKE_PATCH, new_callable=AsyncMock, return_value=result),
            patch(_SLEEP_PATCH, new_callable=AsyncMock),
            patch('shared.cli_invoke.time.monotonic', side_effect=monotonic_values),
            caplog.at_level(logging.ERROR, logger='shared.cli_invoke'),
            pytest.raises(AllAccountsCappedException),
        ):
            await invoke_with_cap_retry(
                gate, 'my-label',
                cap_wait_sanity_secs=3600.0,
                prompt='hi',
            )
        assert any(
            'my-label' in record.message and record.levelno == logging.ERROR
            for record in caplog.records
        ), f'Expected error log with label. Got: {[r.message for r in caplog.records]}'
        error_msgs = [r.message for r in caplog.records if r.levelno == logging.ERROR]
        assert len(error_msgs) >= 1

    async def test_error_log_includes_elapsed_time_on_sanity_raise(self, caplog):
        """logger.error includes elapsed-time diagnostic info on sanity-bound raise.

        Distinct from test_error_logged_before_sanity_raise (which checks label +
        level): this test verifies that the error message body contains the
        elapsed-time diagnostic (e.g. '4000.0s') produced by
        ``_check_cap_wait``'s format string
        ``'{label}: cap-wait sanity bound exceeded after {elapsed:.1f}s ...'``.
        """
        gate = _mock_gate(
            account_count=1,
            before_invoke=AsyncMock(side_effect=['tok'] * 5),
            detect_cap_hit=MagicMock(return_value=True),
            active_account_name='acct',
        )
        result = make_result()
        # First call → 0.0, all subsequent → 4000.0 → elapsed = 4000.0s
        monotonic_values = itertools.chain([0.0], itertools.repeat(4000.0))
        with (
            patch(_INVOKE_PATCH, new_callable=AsyncMock, return_value=result),
            patch(_SLEEP_PATCH, new_callable=AsyncMock),
            patch('shared.cli_invoke.time.monotonic', side_effect=monotonic_values),
            caplog.at_level(logging.ERROR, logger='shared.cli_invoke'),
            pytest.raises(AllAccountsCappedException),
        ):
            await invoke_with_cap_retry(
                gate, 'deadline-label',
                cap_wait_sanity_secs=3600.0,
                prompt='hi',
            )
        error_msgs = [r.message for r in caplog.records if r.levelno == logging.ERROR]
        assert len(error_msgs) >= 1
        # Verify the diagnostic elapsed-time content (not just the label).
        # Use a regex to pin the stable phrase from _check_cap_wait without
        # coupling to the exact float-format precision (e.g. .1f vs .2f).
        assert any(re.search(r'cap-wait sanity bound exceeded after \d+\.\d+s', m) for m in error_msgs), (
            f'Error log should include sanity-bound diagnostic phrase with float-formatted elapsed time. Got: {error_msgs}'
        )


# ===================================================================
# TestReleaseProbeSlotOnException
# ===================================================================


@pytest.mark.asyncio
class TestReleaseProbeSlotOnException:
    """invoke_with_cap_retry calls release_probe_slot() when invoke raises."""

    async def test_release_probe_slot_called_on_runtime_error(self):
        """release_probe_slot is called with oauth_token when invoke_claude_agent raises."""
        gate = _mock_gate(
            before_invoke=AsyncMock(return_value='tok-a'),
            release_probe_slot=MagicMock(),
        )
        with (
            patch(_INVOKE_PATCH, new_callable=AsyncMock, side_effect=RuntimeError('boom')),
            patch(_SLEEP_PATCH, new_callable=AsyncMock),
            pytest.raises(RuntimeError, match='boom'),
        ):
            await invoke_with_cap_retry(gate, 'lbl', prompt='hi')

        gate.release_probe_slot.assert_called_once_with('tok-a')

    async def test_runtime_error_propagates(self):
        """RuntimeError raised by invoke_claude_agent propagates to the caller."""
        gate = _mock_gate(
            before_invoke=AsyncMock(return_value='tok-a'),
            release_probe_slot=MagicMock(),
        )
        with (
            patch(_INVOKE_PATCH, new_callable=AsyncMock, side_effect=RuntimeError('subprocess failed')),
            patch(_SLEEP_PATCH, new_callable=AsyncMock),
            pytest.raises(RuntimeError, match='subprocess failed'),
        ):
            await invoke_with_cap_retry(gate, 'lbl', prompt='hi')

    async def test_confirm_account_ok_not_called_when_invoke_raises(self):
        """confirm_account_ok is NOT called when invoke_claude_agent raises."""
        gate = _mock_gate(
            before_invoke=AsyncMock(return_value='tok-a'),
            release_probe_slot=MagicMock(),
        )
        with (
            patch(_INVOKE_PATCH, new_callable=AsyncMock, side_effect=RuntimeError('boom')),
            patch(_SLEEP_PATCH, new_callable=AsyncMock),
            pytest.raises(RuntimeError),
        ):
            await invoke_with_cap_retry(gate, 'lbl', prompt='hi')

        gate.confirm_account_ok.assert_not_called()


# ===================================================================
# TestProbeSlotLifecycleIntegration
# ===================================================================


@pytest.mark.asyncio
class TestProbeSlotLifecycleIntegration:
    """End-to-end probe slot lifecycle using a real UsageGate (not mock)."""

    async def test_probe_slot_released_after_invoke_raises(self):
        """Full lifecycle: probe slot claimed then released on exception.

        Scenario: gate with one account, probing=True → before_invoke claims
        the probe slot (probe_in_flight=True, _open cleared) → invoke raises →
        release_probe_slot clears state → gate is not deadlocked.
        """
        gate = make_gate(['a'])
        acct = gate._accounts[0]
        # Simulate account just recovered from a cap hit (probe loop succeeded)
        acct.probing = True

        with (
            patch(_INVOKE_PATCH, new_callable=AsyncMock, side_effect=RuntimeError('subprocess failed')),
            patch(_SLEEP_PATCH, new_callable=AsyncMock),
            pytest.raises(RuntimeError, match='subprocess failed'),
        ):
            await invoke_with_cap_retry(gate, 'lbl', prompt='hi')

        # Probe slot must be fully released
        assert acct.probe_in_flight is False, 'probe_in_flight must be cleared'
        assert acct.probe_count == 0, 'probe_count must be reset to 0'
        assert gate._open.is_set(), '_open event must be re-opened (gate not deadlocked)'


# ===================================================================
# TestCancelledErrorReleaseProbeSlot
# ===================================================================


@pytest.mark.asyncio
class TestCancelledErrorReleaseProbeSlot:
    """CancelledError (BaseException, not Exception) tests the catch-all path."""

    async def test_cancelled_error_triggers_release_probe_slot(self):
        """asyncio.CancelledError is a BaseException — must be caught and release_probe_slot called."""
        import asyncio as _asyncio

        gate = _mock_gate(
            before_invoke=AsyncMock(return_value='tok-a'),
            release_probe_slot=MagicMock(),
        )
        with (
            patch(_INVOKE_PATCH, new_callable=AsyncMock, side_effect=_asyncio.CancelledError()),
            patch(_SLEEP_PATCH, new_callable=AsyncMock),
            pytest.raises(_asyncio.CancelledError),
        ):
            await invoke_with_cap_retry(gate, 'lbl', prompt='hi')

        gate.release_probe_slot.assert_called_once_with('tok-a')

    async def test_cancelled_error_propagates(self):
        """CancelledError must propagate (not be swallowed by the except handler)."""
        import asyncio as _asyncio

        gate = _mock_gate(
            before_invoke=AsyncMock(return_value='tok-a'),
            release_probe_slot=MagicMock(),
        )
        with (
            patch(_INVOKE_PATCH, new_callable=AsyncMock, side_effect=_asyncio.CancelledError()),
            patch(_SLEEP_PATCH, new_callable=AsyncMock),
            pytest.raises(_asyncio.CancelledError),
        ):
            await invoke_with_cap_retry(gate, 'lbl', prompt='hi')


# ===================================================================
# TestCapRetryUnattributedCapHit
# ===================================================================


@pytest.mark.asyncio
class TestCapRetryUnattributedCapHit:
    """Heuristic cap fires but _handle_cap_detected returns False (token unresolvable).

    In this scenario, unattributed_cap is set True (skip_confirm was renamed to
    unattributed_cap to better reflect its broader semantics). on_agent_complete
    must NOT be called and cost_store.save_invocation must record capped=True.
    """

    def _make_heuristic_result(self) -> AgentResult:
        """Zero-cost instant-exit result that triggers the heuristic branch."""
        return AgentResult(
            success=False,
            output='Usage limit reached',
            cost_usd=0.0,
            turns=1,
            duration_ms=100,
        )

    async def test_on_agent_complete_not_called_when_unattributed(self):
        """on_agent_complete is NOT called when heuristic cap fires but
        _handle_cap_detected returns False (unattributed cap hit).

        Rationale: cost_usd=0 means the call is a no-op for budget math, but
        any invocation-counting logic built on on_agent_complete would miscount
        this as a legitimate zero-cost completion.
        """
        gate = _mock_gate(
            account_count=1,
            detect_cap_hit=MagicMock(return_value=False),
        )
        gate._handle_cap_detected = MagicMock(return_value=False)
        heuristic_result = self._make_heuristic_result()

        with (
            patch(_INVOKE_PATCH, new_callable=AsyncMock, return_value=heuristic_result),
            patch(_SLEEP_PATCH, new_callable=AsyncMock),
        ):
            await invoke_with_cap_retry(gate, 'unattributed-task', prompt='hi')

        gate.on_agent_complete.assert_not_called()
        gate.confirm_account_ok.assert_not_called()

    async def test_on_agent_complete_called_after_retry_succeeds(self):
        """on_agent_complete IS called once after a heuristic cap-hit retry succeeds.

        Guards against over-gating: we must not suppress on_agent_complete for
        legitimate completions that follow a retry.
        """
        gate = _mock_gate(
            account_count=1,
            before_invoke=AsyncMock(side_effect=['tok'] * 2),
            detect_cap_hit=MagicMock(return_value=False),
            active_account_name='acct',
        )
        gate._handle_cap_detected = MagicMock(return_value=True)  # cap marked → retry

        heuristic_result = self._make_heuristic_result()
        ok_result = make_result(cost_usd=1.23)

        with (
            patch(_INVOKE_PATCH, new_callable=AsyncMock,
                  side_effect=[heuristic_result, ok_result]),
            patch(_SLEEP_PATCH, new_callable=AsyncMock),
        ):
            await invoke_with_cap_retry(gate, 'retry-task', prompt='hi')

        gate.on_agent_complete.assert_called_once_with(1.23)

    async def test_save_invocation_capped_true_when_unattributed(self):
        """cost_store.save_invocation is called with capped=True when heuristic
        fires and _handle_cap_detected returns False.

        The capped column was previously hardcoded to False. Unattributed cap
        hits are the one case where a cap-hit result reaches save_invocation —
        they should be recorded accurately for dashboard queries.
        """
        gate = _mock_gate(
            account_count=1,
            detect_cap_hit=MagicMock(return_value=False),
        )
        gate._handle_cap_detected = MagicMock(return_value=False)
        heuristic_result = self._make_heuristic_result()

        cost_store = MagicMock()
        cost_store.save_invocation = AsyncMock()

        with (
            patch(_INVOKE_PATCH, new_callable=AsyncMock, return_value=heuristic_result),
            patch(_SLEEP_PATCH, new_callable=AsyncMock),
        ):
            await invoke_with_cap_retry(
                gate, 'unattributed-capped',
                cost_store=cost_store, prompt='hi',
            )

        cost_store.save_invocation.assert_awaited_once()
        kw = cost_store.save_invocation.call_args.kwargs
        assert kw['capped'] is True, (
            f'Expected capped=True for unattributed cap hit, got capped={kw["capped"]!r}'
        )

    async def test_save_invocation_capped_false_for_normal_success(self):
        """cost_store.save_invocation uses capped=False for a normal successful result.

        Regression guard: the capped flag must default to False when no
        unattributed cap hit occurred.
        """
        gate = _mock_gate(
            account_count=1,
            detect_cap_hit=MagicMock(return_value=False),
        )
        ok_result = make_result(cost_usd=0.5)

        cost_store = MagicMock()
        cost_store.save_invocation = AsyncMock()

        with (
            patch(_INVOKE_PATCH, new_callable=AsyncMock, return_value=ok_result),
            patch(_SLEEP_PATCH, new_callable=AsyncMock),
        ):
            await invoke_with_cap_retry(
                gate, 'normal-task',
                cost_store=cost_store, prompt='hi',
            )

        cost_store.save_invocation.assert_awaited_once()
        kw = cost_store.save_invocation.call_args.kwargs
        assert kw['capped'] is False, (
            f'Expected capped=False for normal success, got capped={kw["capped"]!r}'
        )


# ===================================================================
# TestAuthFailure403Detection
# ===================================================================


@pytest.mark.asyncio
class TestAuthFailure403Detection:
    """invoke_with_cap_retry routes 4xx api_error_status to _handle_auth_failure."""

    async def test_403_marks_account_auth_failed_and_fails_over(self):
        """A 403 result triggers _handle_auth_failure, then retries on next account."""
        gate = make_gate(['a', 'b'])

        results = iter([
            make_result(
                success=False,
                output='Your organization does not have access to Claude',
                api_error_status=403,
                cost_usd=0.0,
            ),
            make_result(success=True, cost_usd=0.5),
        ])

        async def fake_invoke(**kwargs):
            return next(results)

        with (
            patch(_INVOKE_PATCH, side_effect=fake_invoke) as mock_inv,
            patch(_SLEEP_PATCH, new_callable=AsyncMock),
        ):
            got = await invoke_with_cap_retry(gate, 'lbl', prompt='hi')

        assert got.success is True
        assert mock_inv.await_count == 2
        # First account must now be auth_failed.
        assert gate._accounts[0].auth_failed is True
        # Second account remains healthy.
        assert gate._accounts[1].auth_failed is False

    async def test_401_also_treated_as_auth_failure(self):
        """401 api_error_status is treated as auth failure (alongside 403)."""
        gate = make_gate(['a', 'b'])
        results = iter([
            make_result(success=False, output='Unauthorized',
                         api_error_status=401, cost_usd=0.0),
            make_result(success=True, cost_usd=0.5),
        ])

        async def fake_invoke(**kwargs):
            return next(results)

        with (
            patch(_INVOKE_PATCH, side_effect=fake_invoke),
            patch(_SLEEP_PATCH, new_callable=AsyncMock),
        ):
            got = await invoke_with_cap_retry(gate, 'lbl', prompt='hi')
        assert got.success is True
        assert gate._accounts[0].auth_failed is True

    async def test_429_routes_to_cap_hit_not_auth_failed(self):
        """HTTP 429 with a cap-message body must mark the account capped, not
        auth_failed.  The 4xx-broadcast routing introduced before the fix sent
        429 to ``_handle_auth_failure`` so ``AllAccountsCappedException`` never
        fired and the curator worker's cap-defer / wait-for-open machinery
        never engaged — yielding a real 2026-05-08 incident where 10 reify
        tickets dropped during a cap storm.
        """
        gate = make_gate(['a', 'b'])

        cap_body = (
            "You're out of extra usage · resets May 13, 2pm "
            "(Europe/London)"
        )
        results = iter([
            make_result(
                success=False, output=cap_body,
                api_error_status=429, cost_usd=0.0,
            ),
            make_result(success=True, cost_usd=0.5),
        ])

        async def fake_invoke(**kwargs):
            return next(results)

        with (
            patch(_INVOKE_PATCH, side_effect=fake_invoke),
            patch(_SLEEP_PATCH, new_callable=AsyncMock),
            patch.object(
                UsageGate, '_handle_auth_failure',
                autospec=True, side_effect=AssertionError(
                    '_handle_auth_failure must not be called for 429',
                ),
            ),
            patch.object(
                UsageGate, '_handle_cap_detected', autospec=True,
                wraps=UsageGate._handle_cap_detected,
            ) as mock_cap_detected,
        ):
            got = await invoke_with_cap_retry(gate, 'lbl', prompt='hi')

        assert got.success is True
        # The first invocation must have flagged account 'a' capped.
        assert gate._accounts[0].capped is True, (
            f'expected capped, got {gate._accounts[0]!r}'
        )
        assert gate._accounts[0].auth_failed is False
        # _handle_cap_detected must have been called with a parsed resets_at.
        assert mock_cap_detected.called
        call_kwargs = mock_cap_detected.call_args.kwargs
        call_args = mock_cap_detected.call_args.args
        # autospec passes self as the first positional; resets_at is the
        # third arg in the signature (reason, resets_at, oauth_token).
        resets_at = call_kwargs.get('resets_at')
        if resets_at is None and len(call_args) >= 3:
            resets_at = call_args[2]
        assert resets_at is not None, (
            f'expected resets_at to be parsed; call_args={call_args!r} '
            f'kwargs={call_kwargs!r}'
        )

    async def test_429_all_accounts_raises_all_accounts_capped(self):
        """When every account answers HTTP 429 with a cap-message body,
        ``invoke_with_cap_retry`` must raise ``AllAccountsCappedException``.

        Uses a deadline-based raise (not count-based) so the test remains correct
        after the count guard is removed in step-2.  Monotonic is patched to simulate
        a large elapsed time so the deadline fires after the first cap hit.
        """
        gate = make_gate(['a', 'b'])
        cap_body = "You're out of extra usage · resets in 30m"

        async def fake_invoke(**_kw):
            return make_result(
                success=False, output=cap_body,
                api_error_status=429, cost_usd=0.0,
            )

        # First call → 0.0 (retry_start), all subsequent → 4000.0
        # → elapsed > cap_wait_sanity_secs=3600.0 after just one cap hit
        monotonic_values = itertools.chain([0.0], itertools.repeat(4000.0))
        with (
            patch(_INVOKE_PATCH, side_effect=fake_invoke),
            patch(_SLEEP_PATCH, new_callable=AsyncMock),
            patch('shared.cli_invoke.time.monotonic', side_effect=monotonic_values),
            pytest.raises(AllAccountsCappedException),
        ):
            await invoke_with_cap_retry(
                gate, 'lbl', prompt='hi',
                cap_wait_sanity_secs=3600.0,
            )

    async def test_500_not_treated_as_auth_failure(self):
        """5xx api_error_status is NOT treated as auth failure.

        Use non-zero cost / turns so the zero-cost-instant-exit heuristic
        doesn't mask the assertion by treating the 500 as a cap hit.
        """
        gate = make_gate(['a'])
        result = make_result(
            success=True, output='server error',
            api_error_status=500, cost_usd=0.1, duration_ms=6000, turns=2,
        )
        with (
            patch(_INVOKE_PATCH, new_callable=AsyncMock, return_value=result),
            patch(_SLEEP_PATCH, new_callable=AsyncMock),
        ):
            got = await invoke_with_cap_retry(gate, 'lbl', prompt='hi')
        assert got.api_error_status == 500
        assert gate._accounts[0].auth_failed is False

    async def test_no_api_error_status_is_noop(self):
        """api_error_status=None: the result is returned as-is, no auth handling."""
        gate = make_gate(['a'])
        result = make_result(success=True, cost_usd=0.5)  # api_error_status defaults to None
        with (
            patch(_INVOKE_PATCH, new_callable=AsyncMock, return_value=result),
            patch(_SLEEP_PATCH, new_callable=AsyncMock),
        ):
            got = await invoke_with_cap_retry(gate, 'lbl', prompt='hi')
        assert got.success is True
        assert gate._accounts[0].auth_failed is False


# ===================================================================
# TestInvokeFnParameter
# ===================================================================


@pytest.mark.asyncio
class TestInvokeFnParameter:
    """invoke_with_cap_retry accepts an invoke_fn callable for backend dispatch."""

    async def test_custom_invoke_fn_called(self):
        """When invoke_fn is passed, it is used instead of invoke_claude_agent."""
        gate = make_gate(['a'])
        custom = AsyncMock(return_value=make_result(
            success=True, cost_usd=0.1, duration_ms=6000, turns=2,
        ))
        await invoke_with_cap_retry(
            gate, 'lbl', invoke_fn=custom, backend='claude', prompt='hi',
        )
        custom.assert_awaited_once()

    async def test_invoke_fn_receives_kwargs(self):
        """Custom invoke_fn receives prompt + oauth_token + config_dir kwargs."""
        gate = make_gate(['a'])
        custom = AsyncMock(return_value=make_result(
            success=True, cost_usd=0.1, duration_ms=6000, turns=2,
        ))
        await invoke_with_cap_retry(
            gate, 'lbl', invoke_fn=custom, prompt='hello', model='sonnet',
        )
        kw = custom.call_args.kwargs
        assert kw['prompt'] == 'hello'
        assert kw['oauth_token'] == 'fake-token-a'
        assert kw['model'] == 'sonnet'

    async def test_backend_forwarded_to_detect_cap_hit(self):
        """backend parameter is forwarded to slot.detect_cap_hit."""
        gate = make_gate(['a'])
        detect_calls: list[str] = []
        orig_detect = gate.detect_cap_hit

        def spy(stderr, result_text, backend='claude', oauth_token=None):
            detect_calls.append(backend)
            return orig_detect(stderr, result_text, backend, oauth_token=oauth_token)

        gate.detect_cap_hit = spy  # type: ignore[assignment]

        result = make_result(
            success=True, cost_usd=0.1, duration_ms=6000, turns=2,
        )
        with patch(_INVOKE_PATCH, new_callable=AsyncMock, return_value=result):
            await invoke_with_cap_retry(
                gate, 'lbl', backend='codex', prompt='hi',
            )
        assert detect_calls == ['codex']

    async def test_default_invoke_fn_is_claude(self):
        """When invoke_fn is omitted, invoke_claude_agent is used."""
        gate = make_gate(['a'])
        result = make_result(
            success=True, cost_usd=0.1, duration_ms=6000, turns=2,
        )
        with patch(_INVOKE_PATCH, new_callable=AsyncMock, return_value=result) as mock_inv:
            await invoke_with_cap_retry(gate, 'lbl', prompt='hi')
        mock_inv.assert_awaited_once()


# ===================================================================
# TestVestigialParamsRemoved  (step-1 regression guard)
# ===================================================================


class TestVestigialParamsRemoved:
    """Regression guard: vestigial params and constants must not exist after task-1401/step-2.

    RED until step-2 removes max_cap_retries / cap_retry_deadline_secs from the
    invoke_with_cap_retry signature and deletes the corresponding module constants.
    """

    def test_signature_has_no_max_cap_retries(self):
        """invoke_with_cap_retry must NOT have a max_cap_retries parameter."""
        params = inspect.signature(invoke_with_cap_retry).parameters
        assert 'max_cap_retries' not in params, (
            'max_cap_retries is vestigial (task-1401): remove it from the signature'
        )

    def test_signature_has_no_cap_retry_deadline_secs(self):
        """invoke_with_cap_retry must NOT have a cap_retry_deadline_secs parameter."""
        params = inspect.signature(invoke_with_cap_retry).parameters
        assert 'cap_retry_deadline_secs' not in params, (
            'cap_retry_deadline_secs is vestigial (task-1401): remove it from the signature'
        )

    def test_default_max_cap_retries_not_importable(self):
        """_DEFAULT_MAX_CAP_RETRIES must NOT be importable from shared.cli_invoke."""
        with pytest.raises(ImportError):
            from shared.cli_invoke import _DEFAULT_MAX_CAP_RETRIES  # type: ignore[attr-defined]  # noqa: PLC0415, F401, I001

    def test_default_cap_retry_deadline_secs_not_importable(self):
        """_DEFAULT_CAP_RETRY_DEADLINE_SECS must NOT be importable from shared.cli_invoke."""
        with pytest.raises(ImportError):
            from shared.cli_invoke import _DEFAULT_CAP_RETRY_DEADLINE_SECS  # type: ignore[attr-defined]  # noqa: PLC0415, F401, I001
