"""Exhaustive tests for invoke_with_cap_retry — cap detection, failover, resume,
cooldown backoff, budget enforcement, cost-store integration, and edge cases.

Covers every branch in shared.cli_invoke.invoke_with_cap_retry (lines 136-274).
"""

from __future__ import annotations

import itertools
import logging
import os
from unittest.mock import AsyncMock, MagicMock, PropertyMock, call, patch

import pytest

from shared.cli_invoke import (
    _CAP_HIT_COOLDOWN_SECS,
    _DEFAULT_CAP_RETRY_DEADLINE_SECS,
    _DEFAULT_MAX_CAP_RETRIES,
    _MAX_CAP_COOLDOWN_SECS,
    CAP_HIT_RESUME_PROMPT,
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
    """Build a MagicMock UsageGate with sensible defaults."""
    gate = MagicMock()
    gate.account_count = overrides.pop('account_count', 1)
    gate.before_invoke = overrides.pop('before_invoke', AsyncMock(return_value='tok'))
    gate.detect_cap_hit = overrides.pop('detect_cap_hit', MagicMock(return_value=False))
    gate.active_account_name = overrides.pop('active_account_name', 'acct')
    gate.on_agent_complete = overrides.pop('on_agent_complete', MagicMock())
    gate.confirm_account_ok = overrides.pop('confirm_account_ok', MagicMock())
    for k, v in overrides.items():
        setattr(gate, k, v)
    return gate


# Shared patch targets
_INVOKE_PATCH = 'shared.cli_invoke.invoke_claude_agent'
_SLEEP_PATCH = 'shared.cli_invoke.asyncio.sleep'


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


# ===================================================================
# TestCapRetryCooldown
# ===================================================================


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
        assert isinstance(_DEFAULT_MAX_CAP_RETRIES, int)
        assert _DEFAULT_MAX_CAP_RETRIES == 20
        assert isinstance(_DEFAULT_CAP_RETRY_DEADLINE_SECS, float)
        assert _DEFAULT_CAP_RETRY_DEADLINE_SECS == 3600.0


# ===================================================================
# TestCapRetryMaxRetries
# ===================================================================


@pytest.mark.asyncio
class TestCapRetryMaxRetries:
    """max_cap_retries guard: raise AllAccountsCappedException after N cap hits."""

    async def test_raises_after_max_cap_retries(self):
        """With max_cap_retries=3, raises AllAccountsCappedException after exactly 3 cap hits."""
        gate = _mock_gate(
            account_count=1,
            before_invoke=AsyncMock(side_effect=['tok'] * 10),
            detect_cap_hit=MagicMock(return_value=True),
            active_account_name='acct',
        )
        result = make_result()
        with (
            patch(_INVOKE_PATCH, new_callable=AsyncMock, return_value=result) as mock_inv,
            patch(_SLEEP_PATCH, new_callable=AsyncMock),
            pytest.raises(AllAccountsCappedException) as exc_info,
        ):
            await invoke_with_cap_retry(
                gate, 'Task-3', max_cap_retries=3, prompt='hi',
            )
        assert exc_info.value.retries == 3
        assert 'Task-3' in str(exc_info.value)
        assert mock_inv.await_count == 3

    async def test_no_exception_when_cap_hits_under_limit(self):
        """2 cap hits with max_cap_retries=3 succeeds on the 3rd invocation."""
        gate = _mock_gate(
            account_count=1,
            before_invoke=AsyncMock(side_effect=['tok'] * 3),
            detect_cap_hit=MagicMock(side_effect=[True, True, False]),
            active_account_name='acct',
        )
        result = make_result()
        with (
            patch(_INVOKE_PATCH, new_callable=AsyncMock, return_value=result) as mock_inv,
            patch(_SLEEP_PATCH, new_callable=AsyncMock),
        ):
            got = await invoke_with_cap_retry(
                gate, 'lbl', max_cap_retries=3, prompt='hi',
            )
        assert got.success is True
        assert mock_inv.await_count == 3

    async def test_none_disables_max_cap_retries(self):
        """max_cap_retries=None disables the retry count guard; 5 cap hits then success."""
        gate = _mock_gate(
            account_count=1,
            before_invoke=AsyncMock(side_effect=['tok'] * 6),
            detect_cap_hit=MagicMock(side_effect=[True] * 5 + [False]),
            active_account_name='acct',
        )
        result = make_result()
        with (
            patch(_INVOKE_PATCH, new_callable=AsyncMock, return_value=result) as mock_inv,
            patch(_SLEEP_PATCH, new_callable=AsyncMock),
        ):
            got = await invoke_with_cap_retry(
                gate, 'lbl',
                max_cap_retries=None,
                cap_retry_deadline_secs=None,
                prompt='hi',
            )
        assert got.success is True
        assert mock_inv.await_count == 6


# ===================================================================
# TestCapRetryDeadline
# ===================================================================


@pytest.mark.asyncio
class TestCapRetryDeadline:
    """cap_retry_deadline_secs guard: raise AllAccountsCappedException when elapsed exceeds limit."""

    async def test_raises_when_deadline_exceeded(self):
        """When time.monotonic exceeds deadline, raises AllAccountsCappedException."""
        # monotonic returns 0.0 first (retry_start), then 4000.0 on the elapsed check
        # This simulates 4000 seconds elapsed, exceeding 3600s deadline
        gate = _mock_gate(
            account_count=1,
            before_invoke=AsyncMock(side_effect=['tok'] * 5),
            detect_cap_hit=MagicMock(return_value=True),
            active_account_name='acct',
        )
        result = make_result()
        # itertools.chain+repeat is resilient: first call → 0.0, all subsequent → 4000.0
        # so future extra monotonic() calls won't exhaust the iterator (unlike iter([...]))
        monotonic_values = itertools.chain([0.0], itertools.repeat(4000.0))

        with (
            patch(_INVOKE_PATCH, new_callable=AsyncMock, return_value=result),
            patch(_SLEEP_PATCH, new_callable=AsyncMock),
            patch('shared.cli_invoke.time.monotonic', side_effect=monotonic_values),
            pytest.raises(AllAccountsCappedException) as exc_info,
        ):
            await invoke_with_cap_retry(
                gate, 'deadline-task',
                cap_retry_deadline_secs=3600.0,
                max_cap_retries=None,
                prompt='hi',
            )
        exc = exc_info.value
        assert exc.elapsed_secs > 3600.0
        assert exc.label == 'deadline-task'
        assert exc.retries == 1

    async def test_deadline_fires_before_max_retries(self):
        """Deadline fires after 1 cap hit even when max_cap_retries=100 is far from exhausted.

        The deadline guard (cli_invoke.py:285) is checked independently from the
        max_cap_retries guard (cli_invoke.py:275), so whichever limit triggers first
        wins.  This test covers the interaction where deadline fires first.
        """
        gate = _mock_gate(
            account_count=1,
            before_invoke=AsyncMock(side_effect=['tok'] * 10),
            detect_cap_hit=MagicMock(return_value=True),
            active_account_name='acct',
        )
        result = make_result()
        # First call returns 0.0 (retry_start), all subsequent calls return 15.0
        # so elapsed == 15.0 > cap_retry_deadline_secs=10.0 after the very first hit
        monotonic_values = itertools.chain([0.0], itertools.repeat(15.0))

        with (
            patch(_INVOKE_PATCH, new_callable=AsyncMock, return_value=result),
            patch(_SLEEP_PATCH, new_callable=AsyncMock),
            patch('shared.cli_invoke.time.monotonic', side_effect=monotonic_values),
            pytest.raises(AllAccountsCappedException) as exc_info,
        ):
            await invoke_with_cap_retry(
                gate, 'deadline-first-task',
                cap_retry_deadline_secs=10.0,
                max_cap_retries=100,
                prompt='hi',
            )
        exc = exc_info.value
        # Deadline fires after 1 retry, not after 100
        assert exc.retries == 1, f'Expected 1 retry (deadline), got {exc.retries}'
        assert exc.elapsed_secs > 10.0, f'elapsed_secs should exceed deadline, got {exc.elapsed_secs}'
        assert exc.label == 'deadline-first-task'

    async def test_no_exception_when_within_deadline(self):
        """When elapsed time is well under deadline, no exception is raised."""
        gate = _mock_gate(
            account_count=1,
            before_invoke=AsyncMock(side_effect=['tok'] * 3),
            detect_cap_hit=MagicMock(side_effect=[True, True, False]),
            active_account_name='acct',
        )
        result = make_result()
        # All monotonic calls return 0.0, so elapsed is always 0.0 < 3600.0
        with (
            patch(_INVOKE_PATCH, new_callable=AsyncMock, return_value=result),
            patch(_SLEEP_PATCH, new_callable=AsyncMock),
            patch('shared.cli_invoke.time.monotonic', return_value=0.0),
        ):
            got = await invoke_with_cap_retry(
                gate, 'lbl',
                cap_retry_deadline_secs=3600.0,
                max_cap_retries=None,
                prompt='hi',
            )
        assert got.success is True


# ===================================================================
# TestCapRetryHeuristicGuard
# ===================================================================


@pytest.mark.asyncio
class TestCapRetryHeuristicGuard:
    """Heuristic cap-hit branch (zero-cost instant exit) also respects max_cap_retries."""

    def _make_heuristic_result(self) -> AgentResult:
        """Zero-cost instant exit result that triggers the heuristic branch."""
        return AgentResult(
            success=False,
            output='Usage limit reached',
            cost_usd=0.0,
            turns=1,
            duration_ms=100,
        )

    async def test_heuristic_raises_after_max_retries(self):
        """Heuristic cap hits count toward max_cap_retries and raise AllAccountsCappedException."""
        gate = _mock_gate(
            account_count=1,
            before_invoke=AsyncMock(side_effect=['tok'] * 5),
            detect_cap_hit=MagicMock(return_value=False),  # not caught by pattern
            active_account_name='acct',
        )
        gate._handle_cap_detected = MagicMock()
        heuristic_result = self._make_heuristic_result()
        with (
            patch(_INVOKE_PATCH, new_callable=AsyncMock, return_value=heuristic_result) as mock_inv,
            patch(_SLEEP_PATCH, new_callable=AsyncMock),
            pytest.raises(AllAccountsCappedException) as exc_info,
        ):
            await invoke_with_cap_retry(
                gate, 'heuristic-task', max_cap_retries=2, prompt='hi',
            )
        assert exc_info.value.retries == 2
        assert mock_inv.await_count == 2

    async def test_heuristic_succeeds_before_max_retries(self):
        """1 heuristic hit then success does not raise (under max_cap_retries=2)."""
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
                gate, 'lbl', max_cap_retries=2, prompt='hi',
            )
        assert got.success is True
        assert mock_inv.await_count == 2

    async def test_heuristic_deadline_exceeded(self):
        """Heuristic branch respects cap_retry_deadline_secs independently of max_cap_retries.

        The deadline guard in the heuristic branch (cli_invoke.py:347-356) is a
        separate code path from the pattern-match branch guard (line 285). This test
        covers that guard — previously had zero test coverage.
        """
        gate = _mock_gate(
            account_count=1,
            before_invoke=AsyncMock(side_effect=['tok'] * 10),
            detect_cap_hit=MagicMock(return_value=False),  # pattern branch skipped
            active_account_name='acct',
        )
        gate._handle_cap_detected = MagicMock()
        heuristic_result = self._make_heuristic_result()
        # First call → 0.0 (retry_start), subsequent calls → 4000.0
        # elapsed == 4000.0 > cap_retry_deadline_secs=3600.0 after first heuristic hit
        monotonic_values = itertools.chain([0.0], itertools.repeat(4000.0))

        with (
            patch(_INVOKE_PATCH, new_callable=AsyncMock, return_value=heuristic_result),
            patch(_SLEEP_PATCH, new_callable=AsyncMock),
            patch('shared.cli_invoke.time.monotonic', side_effect=monotonic_values),
            pytest.raises(AllAccountsCappedException) as exc_info,
        ):
            await invoke_with_cap_retry(
                gate, 'heuristic-deadline-task',
                max_cap_retries=None,
                cap_retry_deadline_secs=3600.0,
                prompt='hi',
            )
        exc = exc_info.value
        assert exc.retries == 1, f'Expected 1 retry (deadline), got {exc.retries}'
        assert exc.elapsed_secs > 3600.0, (
            f'elapsed_secs should exceed 3600.0 deadline, got {exc.elapsed_secs}'
        )
        assert exc.label == 'heuristic-deadline-task'


# ===================================================================
# TestCapRetryGuardLogging
# ===================================================================


@pytest.mark.asyncio
class TestCapRetryGuardLogging:
    """Verify logger.error is emitted with diagnostic info before raising."""

    async def test_error_logged_before_max_retries_raise(self, caplog):
        """logger.error includes label, retry count, and elapsed time on max_cap_retries hit."""
        gate = _mock_gate(
            account_count=2,
            before_invoke=AsyncMock(side_effect=['tok'] * 5),
            detect_cap_hit=MagicMock(return_value=True),
            active_account_name='acct',
        )
        result = make_result()
        with (
            patch(_INVOKE_PATCH, new_callable=AsyncMock, return_value=result),
            patch(_SLEEP_PATCH, new_callable=AsyncMock),
            caplog.at_level(logging.ERROR, logger='shared.cli_invoke'),
            pytest.raises(AllAccountsCappedException),
        ):
            await invoke_with_cap_retry(
                gate, 'my-label',
                max_cap_retries=2,
                cap_retry_deadline_secs=None,
                prompt='hi',
            )
        assert any(
            'my-label' in record.message and record.levelno == logging.ERROR
            for record in caplog.records
        ), f'Expected error log with label. Got: {[r.message for r in caplog.records]}'
        error_msgs = [r.message for r in caplog.records if r.levelno == logging.ERROR]
        assert len(error_msgs) >= 1
        assert any('2' in m for m in error_msgs), 'Error log should include retry count'

    async def test_error_logged_before_deadline_raise(self, caplog):
        """logger.error includes label and elapsed time on deadline hit."""
        gate = _mock_gate(
            account_count=1,
            before_invoke=AsyncMock(side_effect=['tok'] * 5),
            detect_cap_hit=MagicMock(return_value=True),
            active_account_name='acct',
        )
        result = make_result()
        # Resilient: first call → 0.0, all subsequent → 4000.0 (never exhausted)
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
                max_cap_retries=None,
                cap_retry_deadline_secs=3600.0,
                prompt='hi',
            )
        error_msgs = [r.message for r in caplog.records if r.levelno == logging.ERROR]
        assert len(error_msgs) >= 1
        assert any('deadline-label' in m for m in error_msgs), (
            f'Error log should include label. Got: {error_msgs}'
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
