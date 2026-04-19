"""Exhaustive integration tests for the full failover system.

Tests UsageGate + invoke_with_cap_retry + TaskConfigDir working together
with real UsageGate instances (mocked invoke_claude_agent and asyncio.sleep).

Run:  cd shared && uv run pytest tests/test_failover_integration.py -v
"""

from __future__ import annotations

import asyncio
import json
import os
from datetime import UTC, datetime, timedelta
from pathlib import Path
from unittest.mock import AsyncMock, MagicMock, patch

import pytest

from shared.cli_invoke import (
    CAP_HIT_RESUME_PROMPT,
    AgentResult,
    invoke_with_cap_retry,
)
from shared.config_dir import TaskConfigDir
from shared.config_models import AccountConfig, UsageCapConfig
from shared.usage_gate import (
    SessionBudgetExhausted,
    UsageGate,
)

# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------


def make_gate(account_names, *, cost_store=None, wait_for_reset=False, **kwargs):
    acct_cfgs = []
    env_vars = {}
    for name in account_names:
        env_key = f'TEST_TOKEN_{name.upper().replace("-", "_")}'
        env_vars[env_key] = f'fake-token-{name}'
        acct_cfgs.append(AccountConfig(name=name, oauth_token_env=env_key))
    config = UsageCapConfig(accounts=acct_cfgs, wait_for_reset=wait_for_reset, **kwargs)
    with patch.dict(os.environ, env_vars):
        gate = UsageGate(config, cost_store=cost_store)
    gate._run_probe = AsyncMock(return_value=True)
    return gate


def make_result(success=True, output='done', session_id='', stderr='', cost_usd=0.5, **kw):
    return AgentResult(
        success=success, output=output, session_id=session_id,
        stderr=stderr, cost_usd=cost_usd, **kw,
    )


def _cap_stderr(account=''):
    """Return stderr text that triggers cap detection."""
    return f"You've hit your usage limit for account {account}. Your limit resets in 5h."


# ---------------------------------------------------------------------------
# TestFullFailoverLifecycle
# ---------------------------------------------------------------------------


@pytest.mark.asyncio
class TestFullFailoverLifecycle:

    async def test_three_account_cascade_a_caps_b_caps_c_succeeds(self):
        """A caps -> B used -> B caps -> C used -> C succeeds."""
        gate = make_gate(['A', 'B', 'C'])

        tokens_used = []
        call_count = [0]

        async def mock_invoke(**kwargs):
            token = kwargs.get('oauth_token')
            tokens_used.append(token)
            call_count[0] += 1
            if token == 'fake-token-A':
                return make_result(success=True, output='', stderr=_cap_stderr('A'))
            elif token == 'fake-token-B':
                return make_result(success=True, output='', stderr=_cap_stderr('B'))
            else:
                return make_result(success=True, output='task done', cost_usd=1.23)

        with (
            patch('shared.cli_invoke.invoke_claude_agent', new_callable=AsyncMock,
                  side_effect=mock_invoke),
            patch('shared.cli_invoke.asyncio.sleep', new_callable=AsyncMock),
        ):
            result = await invoke_with_cap_retry(
                gate, 'test-3-acct',
                prompt='do work', system_prompt='sys', cwd=Path('/tmp'),
            )

        # Token sequence
        assert tokens_used == ['fake-token-A', 'fake-token-B', 'fake-token-C']
        # A and B capped
        assert gate._accounts[0].capped is True
        assert gate._accounts[1].capped is True
        # C not capped
        assert gate._accounts[2].capped is False
        # Result has account_name set
        assert result.account_name == 'C'
        # Cumulative cost tracked
        assert gate.cumulative_cost == 1.23

    async def test_all_cap_then_first_uncaps_via_timer(self):
        """A caps -> B caps -> C caps -> all blocked -> A uncaps -> A succeeds."""
        gate = make_gate(['A', 'B', 'C'])

        call_count = [0]

        async def mock_invoke(**kwargs):
            call_count[0] += 1
            kwargs.get('oauth_token')
            if call_count[0] <= 3:
                return make_result(success=True, output='', stderr=_cap_stderr())
            return make_result(success=True, output='finally done', cost_usd=0.7)

        async def background_uncap():
            """Wait for all to cap, then uncap A via resets_at in the past."""
            while not all(a.capped for a in gate._accounts):
                await asyncio.sleep(0)
            gate._accounts[0].resets_at = datetime.now(UTC) - timedelta(minutes=1)
            await gate._refresh_capped_accounts()

        with (
            patch('shared.cli_invoke.invoke_claude_agent', new_callable=AsyncMock,
                  side_effect=mock_invoke),
            patch('shared.cli_invoke.asyncio.sleep', new_callable=AsyncMock),
        ):
            bg = asyncio.create_task(background_uncap())
            result = await asyncio.wait_for(
                invoke_with_cap_retry(
                    gate, 'test-all-cap',
                    prompt='do work', system_prompt='sys', cwd=Path('/tmp'),
                ),
                timeout=5.0,
            )
            await bg

        assert result.success is True
        assert result.output == 'finally done'
        assert call_count[0] == 4

    async def test_five_account_cascade(self):
        """A through E: cascade cap through all, then first resets."""
        names = ['A', 'B', 'C', 'D', 'E']
        gate = make_gate(names)

        call_count = [0]

        async def mock_invoke(**kwargs):
            call_count[0] += 1
            kwargs.get('oauth_token')
            if call_count[0] <= 5:
                return make_result(success=True, output='', stderr=_cap_stderr())
            return make_result(success=True, output='done', cost_usd=0.3)

        async def background_uncap():
            while not all(a.capped for a in gate._accounts):
                await asyncio.sleep(0)
            # Uncap A
            gate._accounts[0].resets_at = datetime.now(UTC) - timedelta(minutes=1)
            await gate._refresh_capped_accounts()

        with (
            patch('shared.cli_invoke.invoke_claude_agent', new_callable=AsyncMock,
                  side_effect=mock_invoke),
            patch('shared.cli_invoke.asyncio.sleep', new_callable=AsyncMock),
        ):
            bg = asyncio.create_task(background_uncap())
            result = await asyncio.wait_for(
                invoke_with_cap_retry(
                    gate, 'test-5-acct',
                    prompt='do work', system_prompt='sys', cwd=Path('/tmp'),
                ),
                timeout=5.0,
            )
            await bg

        assert result.success is True
        assert call_count[0] == 6
        # First 5 accounts capped, then A uncapped and used
        for i in range(1, 5):
            assert gate._accounts[i].capped is True
        # A was uncapped by refresh
        assert gate._accounts[0].capped is False


# ---------------------------------------------------------------------------
# TestSessionResumeAcrossFailover
# ---------------------------------------------------------------------------


@pytest.mark.asyncio
class TestSessionResumeAcrossFailover:

    async def test_cap_on_a_resume_on_b_succeeds(self):
        """A: session_id='sess-1', cap hit -> B: resume with sess-1 -> success."""
        gate = make_gate(['A', 'B'])

        capped = make_result(session_id='sess-1', stderr=_cap_stderr('A'))
        ok = make_result(success=True, output='resumed ok', cost_usd=0.8)

        with (
            patch('shared.cli_invoke.invoke_claude_agent', new_callable=AsyncMock,
                  side_effect=[capped, ok]) as mock_inv,
            patch('shared.cli_invoke.asyncio.sleep', new_callable=AsyncMock),
        ):
            result = await invoke_with_cap_retry(
                gate, 'test-resume',
                prompt='do work', system_prompt='sys', cwd=Path('/tmp'),
            )

        assert result.success is True
        assert result.output == 'resumed ok'
        # Second call used resume_session_id and CAP_HIT_RESUME_PROMPT
        second = mock_inv.call_args_list[1]
        assert second.kwargs.get('resume_session_id') == 'sess-1'
        assert second.kwargs.get('prompt') == CAP_HIT_RESUME_PROMPT

    async def test_resume_on_b_also_caps_retry_on_c(self):
        """A caps (sess-1) -> B resume caps (sess-1) -> C resume with sess-1."""
        gate = make_gate(['A', 'B', 'C'])

        capped_a = make_result(session_id='sess-1', stderr=_cap_stderr('A'))
        # B also caps while resuming — returns the *same* session_id
        capped_b = make_result(session_id='sess-1', stderr=_cap_stderr('B'))
        ok = make_result(success=True, output='done on C')

        with (
            patch('shared.cli_invoke.invoke_claude_agent', new_callable=AsyncMock,
                  side_effect=[capped_a, capped_b, ok]) as mock_inv,
            patch('shared.cli_invoke.asyncio.sleep', new_callable=AsyncMock),
        ):
            result = await invoke_with_cap_retry(
                gate, 'test-cascade-resume',
                prompt='do work', system_prompt='sys', cwd=Path('/tmp'),
            )

        assert result.success is True
        # Third call still uses resume_session_id from the B cap hit
        third = mock_inv.call_args_list[2]
        assert third.kwargs.get('resume_session_id') == 'sess-1'
        assert third.kwargs.get('prompt') == CAP_HIT_RESUME_PROMPT

    async def test_resume_fails_falls_back_to_fresh(self):
        """A caps (sess-1) -> B resume fails (not cap) -> C fresh with original prompt."""
        gate = make_gate(['A', 'B', 'C'])

        capped_a = make_result(session_id='sess-1', stderr=_cap_stderr('A'))
        failed_resume = make_result(success=False, output='resume error')
        ok = make_result(success=True, output='fresh success')

        with (
            patch('shared.cli_invoke.invoke_claude_agent', new_callable=AsyncMock,
                  side_effect=[capped_a, failed_resume, ok]) as mock_inv,
            patch('shared.cli_invoke.asyncio.sleep', new_callable=AsyncMock),
        ):
            result = await invoke_with_cap_retry(
                gate, 'test-resume-fail',
                prompt='original prompt', system_prompt='sys', cwd=Path('/tmp'),
            )

        assert result.success is True
        assert result.output == 'fresh success'
        # Third call: fresh, no resume_session_id, original prompt restored
        third = mock_inv.call_args_list[2]
        assert 'resume_session_id' not in third.kwargs
        assert third.kwargs.get('prompt') == 'original prompt'

    async def test_full_call_sequence_kwargs(self):
        """Verify full invoke_claude_agent call sequence: prompts and resume ids."""
        gate = make_gate(['A', 'B', 'C'])

        results = [
            make_result(session_id='sess-1', stderr=_cap_stderr('A')),  # A caps
            make_result(success=False, output='resume error'),          # B resume fails
            make_result(success=True, output='ok'),                     # C fresh
        ]

        with (
            patch('shared.cli_invoke.invoke_claude_agent', new_callable=AsyncMock,
                  side_effect=results) as mock_inv,
            patch('shared.cli_invoke.asyncio.sleep', new_callable=AsyncMock),
        ):
            await invoke_with_cap_retry(
                gate, 'test-seq',
                prompt='my prompt', system_prompt='sys', cwd=Path('/tmp'),
            )

        calls = mock_inv.call_args_list
        # Call 1: original prompt, no resume
        assert calls[0].kwargs.get('prompt') == 'my prompt'
        assert calls[0].kwargs.get('resume_session_id') is None
        # Call 2: resume from sess-1
        assert calls[1].kwargs.get('prompt') == CAP_HIT_RESUME_PROMPT
        assert calls[1].kwargs.get('resume_session_id') == 'sess-1'
        # Call 3: fresh (resume failed)
        assert calls[2].kwargs.get('prompt') == 'my prompt'
        assert 'resume_session_id' not in calls[2].kwargs


# ---------------------------------------------------------------------------
# TestAllCappedThenResume
# ---------------------------------------------------------------------------


@pytest.mark.asyncio
class TestAllCappedThenResume:

    async def test_two_accounts_both_cap_background_uncap(self):
        """2 accounts, both cap. Gate blocks. Background uncaps one. Completes."""
        gate = make_gate(['A', 'B'])

        call_count = [0]

        async def mock_invoke(**kwargs):
            call_count[0] += 1
            if call_count[0] <= 2:
                return make_result(stderr=_cap_stderr())
            return make_result(success=True, output='unblocked')

        async def background_uncap():
            while not all(a.capped for a in gate._accounts):
                await asyncio.sleep(0)
            gate._accounts[0].resets_at = datetime.now(UTC) - timedelta(minutes=1)
            await gate._refresh_capped_accounts()

        with (
            patch('shared.cli_invoke.invoke_claude_agent', new_callable=AsyncMock,
                  side_effect=mock_invoke),
            patch('shared.cli_invoke.asyncio.sleep', new_callable=AsyncMock),
        ):
            bg = asyncio.create_task(background_uncap())
            result = await asyncio.wait_for(
                invoke_with_cap_retry(
                    gate, 'test-block',
                    prompt='work', system_prompt='sys', cwd=Path('/tmp'),
                ),
                timeout=5.0,
            )
            await bg

        assert result.success is True
        assert result.output == 'unblocked'

    async def test_blocking_behavior_does_not_return_until_unblocked(self):
        """invoke_with_cap_retry does not return while all accounts are capped."""
        gate = make_gate(['A'])

        call_count = [0]
        returned = asyncio.Event()

        async def mock_invoke(**kwargs):
            call_count[0] += 1
            if call_count[0] == 1:
                return make_result(stderr=_cap_stderr())
            return make_result(success=True, output='ok')

        async def run_retry():
            with (
                patch('shared.cli_invoke.invoke_claude_agent', new_callable=AsyncMock,
                      side_effect=mock_invoke),
                patch('shared.cli_invoke.asyncio.sleep', new_callable=AsyncMock),
            ):
                result = await invoke_with_cap_retry(
                    gate, 'test-block',
                    prompt='work', system_prompt='sys', cwd=Path('/tmp'),
                )
            returned.set()
            return result

        task = asyncio.create_task(run_retry())

        # Give time for cap to be detected and gate to block
        for _ in range(20):
            await asyncio.sleep(0)

        # Should NOT have returned yet (all capped, gate closed)
        assert not returned.is_set(), 'invoke returned before uncap'

        # Uncap account A
        gate._accounts[0].resets_at = datetime.now(UTC) - timedelta(minutes=1)
        await gate._refresh_capped_accounts()

        result = await asyncio.wait_for(task, timeout=5.0)
        assert result.success is True

    async def test_probe_slot_after_uncap(self):
        """After uncap, probing=True. First task claims slot. confirm_account_ok opens gate."""
        gate = make_gate(['A'])

        # Manually cap, then uncap via refresh (simulating timer-based uncap)
        gate._accounts[0].capped = True
        gate._accounts[0].resets_at = datetime.now(UTC) - timedelta(minutes=1)
        await gate._refresh_capped_accounts()

        # After refresh, account should be in probing state
        assert gate._accounts[0].probing is True

        # before_invoke claims the probe slot
        token = await gate.before_invoke()
        assert token == 'fake-token-A'
        assert gate._accounts[0].probe_in_flight is True
        assert gate._accounts[0].probing is False

        # confirm_account_ok clears probe_in_flight
        gate.confirm_account_ok(token)
        assert gate._accounts[0].probe_in_flight is False


# ---------------------------------------------------------------------------
# TestSingleAccountFullLifecycle
# ---------------------------------------------------------------------------


@pytest.mark.asyncio
class TestSingleAccountFullLifecycle:

    async def test_single_account_cap_block_uncap_succeed(self):
        """Single account: cap -> blocks -> background uncap -> succeeds."""
        gate = make_gate(['solo'])

        call_count = [0]

        async def mock_invoke(**kwargs):
            call_count[0] += 1
            if call_count[0] == 1:
                return make_result(stderr=_cap_stderr())
            return make_result(success=True, output='solo done')

        async def background_uncap():
            while not gate._accounts[0].capped:
                await asyncio.sleep(0)
            gate._accounts[0].resets_at = datetime.now(UTC) - timedelta(minutes=1)
            await gate._refresh_capped_accounts()

        with (
            patch('shared.cli_invoke.invoke_claude_agent', new_callable=AsyncMock,
                  side_effect=mock_invoke),
            patch('shared.cli_invoke.asyncio.sleep', new_callable=AsyncMock),
        ):
            bg = asyncio.create_task(background_uncap())
            result = await asyncio.wait_for(
                invoke_with_cap_retry(
                    gate, 'test-solo',
                    prompt='work', system_prompt='sys', cwd=Path('/tmp'),
                ),
                timeout=5.0,
            )
            await bg

        assert result.success is True
        assert result.output == 'solo done'

    async def test_single_account_multiple_consecutive_caps_then_success(self):
        """Single account: cap, re-cap, then succeed on 3rd attempt.

        Uses the real detect_cap_hit and manipulates gate state in the mocked
        asyncio.sleep callback to simulate the account being uncapped between
        retries (as the probe loop would do in production).
        """
        gate = make_gate(['solo'])

        call_count = [0]

        async def mock_invoke(**kwargs):
            call_count[0] += 1
            if call_count[0] <= 2:
                return make_result(stderr=_cap_stderr())
            return make_result(success=True, output='finally done')

        async def uncap_on_sleep(duration):
            """Simulate the probe loop uncapping the account during cooldown sleep."""
            acct = gate._accounts[0]
            if acct.capped:
                acct.capped = False
                acct.probing = False
                acct.probe_in_flight = False
                gate._open.set()

        with (
            patch('shared.cli_invoke.invoke_claude_agent', new_callable=AsyncMock,
                  side_effect=mock_invoke),
            patch('shared.cli_invoke.asyncio.sleep', new_callable=AsyncMock,
                  side_effect=uncap_on_sleep),
        ):
            result = await asyncio.wait_for(
                invoke_with_cap_retry(
                    gate, 'test-multi-cap',
                    prompt='work', system_prompt='sys', cwd=Path('/tmp'),
                ),
                timeout=5.0,
            )

        assert result.success is True
        assert call_count[0] == 3


# ---------------------------------------------------------------------------
# TestBudgetExhaustionDuringFailover
# ---------------------------------------------------------------------------


@pytest.mark.asyncio
class TestBudgetExhaustionDuringFailover:

    async def test_budget_exceeded_raises_on_second_invoke(self):
        """Budget=$1.00, first costs $0.60 -> second invoke raises SessionBudgetExhausted."""
        gate = make_gate(['A'], session_budget_usd=1.00)

        ok = make_result(success=True, output='first done', cost_usd=0.60)

        with patch('shared.cli_invoke.invoke_claude_agent', new_callable=AsyncMock,
                   return_value=ok):
            # First invoke succeeds
            await invoke_with_cap_retry(
                gate, 'test-budget-1',
                prompt='first', system_prompt='sys', cwd=Path('/tmp'),
            )

        assert gate.cumulative_cost == pytest.approx(0.60)

        # Set cost above budget to trigger on next before_invoke
        gate._cumulative_cost = 1.00

        with pytest.raises(SessionBudgetExhausted) as exc_info:
            await invoke_with_cap_retry(
                gate, 'test-budget-2',
                prompt='second', system_prompt='sys', cwd=Path('/tmp'),
            )
        assert exc_info.value.cumulative_cost == pytest.approx(1.00)

    async def test_budget_exactly_exhausted_on_third_call(self):
        """Budget=$2.00, two invocations at $1.00 each -> third call raises."""
        gate = make_gate(['A'], session_budget_usd=2.00)

        ok = make_result(success=True, output='done', cost_usd=1.00)

        with patch('shared.cli_invoke.invoke_claude_agent', new_callable=AsyncMock,
                   return_value=ok):
            await invoke_with_cap_retry(
                gate, 'b1', prompt='a', system_prompt='s', cwd=Path('/tmp'),
            )
            await invoke_with_cap_retry(
                gate, 'b2', prompt='b', system_prompt='s', cwd=Path('/tmp'),
            )

        assert gate.cumulative_cost == pytest.approx(2.00)

        with pytest.raises(SessionBudgetExhausted):
            await invoke_with_cap_retry(
                gate, 'b3', prompt='c', system_prompt='s', cwd=Path('/tmp'),
            )


# ---------------------------------------------------------------------------
# TestCredentialIsolation
# ---------------------------------------------------------------------------


class TestCredentialIsolation:

    def test_two_config_dirs_different_tokens(self, tmp_path):
        """Two TaskConfigDirs for different task IDs have correct credentials."""
        dir1 = TaskConfigDir('task-1', base_dir=tmp_path)
        dir2 = TaskConfigDir('task-2', base_dir=tmp_path)

        dir1.write_credentials('token-alpha')
        dir2.write_credentials('token-beta')

        creds1 = json.loads((dir1.path / '.credentials.json').read_text())
        creds2 = json.loads((dir2.path / '.credentials.json').read_text())

        assert creds1['claudeAiOauth']['accessToken'] == 'token-alpha'
        assert creds2['claudeAiOauth']['accessToken'] == 'token-beta'

        dir1.cleanup()
        dir2.cleanup()

    def test_overwrite_credentials(self, tmp_path):
        """Write token-A then token-B -> file contains token-B."""
        d = TaskConfigDir('overwrite-test', base_dir=tmp_path)
        d.write_credentials('token-A')
        d.write_credentials('token-B')

        creds = json.loads((d.path / '.credentials.json').read_text())
        assert creds['claudeAiOauth']['accessToken'] == 'token-B'
        d.cleanup()

    def test_cleanup_removes_directory(self, tmp_path):
        """Cleanup removes the config directory."""
        d = TaskConfigDir('cleanup-test', base_dir=tmp_path)
        d.write_credentials('token')
        assert d.path.exists()
        d.cleanup()
        assert not d.path.exists()


# ---------------------------------------------------------------------------
# TestConfigDirLifecycle
# ---------------------------------------------------------------------------


class TestConfigDirLifecycle:

    def test_directory_exists_on_creation(self, tmp_path):
        """TaskConfigDir creates the directory on init."""
        d = TaskConfigDir('lifecycle-test', base_dir=tmp_path)
        assert d.path.exists()
        assert d.path.is_dir()
        d.cleanup()

    def test_credentials_json_structure(self, tmp_path):
        """Credentials file has correct JSON structure."""
        d = TaskConfigDir('json-test', base_dir=tmp_path)
        d.write_credentials('my-token-123')

        creds_path = d.path / '.credentials.json'
        assert creds_path.exists()

        data = json.loads(creds_path.read_text())
        assert data == {'claudeAiOauth': {'accessToken': 'my-token-123'}}
        d.cleanup()

    def test_credentials_file_permissions(self, tmp_path):
        """Credentials file has 0o600 permissions."""
        d = TaskConfigDir('perm-test', base_dir=tmp_path)
        d.write_credentials('token')

        creds_path = d.path / '.credentials.json'
        mode = creds_path.stat().st_mode & 0o777
        assert mode == 0o600, f'Expected 0o600, got {oct(mode)}'
        d.cleanup()

    def test_cleanup_removes_everything(self, tmp_path):
        """Cleanup removes the directory and all contents."""
        d = TaskConfigDir('full-cleanup', base_dir=tmp_path)
        d.write_credentials('token')
        path = d.path
        d.cleanup()
        assert not path.exists()

    def test_symlinks_created_for_settings(self, tmp_path):
        """Symlinks created for settings.json if source exists."""
        # Create a mock ~/.claude/settings.json
        fake_home_claude = tmp_path / 'home_claude'
        fake_home_claude.mkdir()
        settings = fake_home_claude / 'settings.json'
        settings.write_text('{}')

        with patch('shared.config_dir._HOME_CLAUDE', fake_home_claude):
            d = TaskConfigDir('symlink-test', base_dir=tmp_path)

        link = d.path / 'settings.json'
        assert link.exists()
        assert link.is_symlink()
        assert link.resolve() == settings.resolve()
        d.cleanup()


# ---------------------------------------------------------------------------
# TestMixedBackends
# ---------------------------------------------------------------------------


@pytest.mark.asyncio
class TestMixedBackends:

    async def test_codex_cap_pattern(self):
        """'usage limit reached' triggers cap detection for codex backend."""
        gate = make_gate(['A'])
        token = gate._accounts[0].token
        hit = gate.detect_cap_hit('usage limit reached', '', 'codex', token)
        assert hit is True
        assert gate._accounts[0].capped is True

    async def test_gemini_cap_pattern(self):
        """'RESOURCE_EXHAUSTED' triggers cap detection for gemini backend."""
        gate = make_gate(['A'])
        token = gate._accounts[0].token
        hit = gate.detect_cap_hit('', 'RESOURCE_EXHAUSTED', 'gemini', token)
        assert hit is True
        assert gate._accounts[0].capped is True

    async def test_claude_cap_after_codex_cap(self):
        """Claude cap pattern detected correctly after a codex cap."""
        gate = make_gate(['A', 'B'])
        token_a = gate._accounts[0].token
        token_b = gate._accounts[1].token

        # Codex cap on A
        gate.detect_cap_hit('usage limit reached', '', 'codex', token_a)
        assert gate._accounts[0].capped is True

        # Claude cap on B
        gate.detect_cap_hit(_cap_stderr(), '', 'claude', token_b)
        assert gate._accounts[1].capped is True

    async def test_codex_pattern_not_detected_for_gemini_backend(self):
        """Codex-specific pattern 'insufficient_quota' not detected for gemini backend."""
        gate = make_gate(['A'])
        token = gate._accounts[0].token
        # 'insufficient_quota' is only in CODEX_CAP_PATTERNS, not GEMINI_CAP_PATTERNS
        hit = gate.detect_cap_hit('insufficient_quota', '', 'gemini', token)
        assert hit is False
        assert gate._accounts[0].capped is False

    async def test_gemini_pattern_not_detected_for_codex_backend(self):
        """Gemini-specific pattern 'RESOURCE_EXHAUSTED' not detected for codex backend."""
        gate = make_gate(['A'])
        token = gate._accounts[0].token
        # 'RESOURCE_EXHAUSTED' is only checked for gemini backend
        hit = gate.detect_cap_hit('RESOURCE_EXHAUSTED', '', 'codex', token)
        # 'resource exhausted' is NOT in CODEX_CAP_PATTERNS but 'rate limit' overlaps.
        # 'RESOURCE_EXHAUSTED' alone should not match codex patterns.
        # Actually check: CODEX_CAP_PATTERNS has 'rate limit', 'quota exceeded'.
        # 'RESOURCE_EXHAUSTED' does not match any of them (case-insensitive).
        assert hit is False
        assert gate._accounts[0].capped is False


# ---------------------------------------------------------------------------
# TestConcurrentTasksWithFailover
# ---------------------------------------------------------------------------


@pytest.mark.asyncio
class TestConcurrentTasksWithFailover:

    async def test_three_concurrent_tasks_two_accounts_with_cap(self):
        """3 concurrent tasks, 2 accounts. A caps mid-flight -> tasks failover to B."""
        gate = make_gate(['A', 'B'])

        invoke_count = [0]
        tokens_seen = []

        async def mock_invoke(**kwargs):
            invoke_count[0] += 1
            token = kwargs.get('oauth_token')
            tokens_seen.append(token)
            # Brief yield to allow interleaving
            await asyncio.sleep(0)
            # First invocation on A triggers cap
            if token == 'fake-token-A' and invoke_count[0] == 1:
                return make_result(stderr=_cap_stderr('A'), cost_usd=0.0)
            return make_result(success=True, output='ok', cost_usd=0.1)

        with (
            patch('shared.cli_invoke.invoke_claude_agent', new_callable=AsyncMock,
                  side_effect=mock_invoke),
            patch('shared.cli_invoke.asyncio.sleep', new_callable=AsyncMock),
        ):
            tasks = [
                invoke_with_cap_retry(
                    gate, f'task-{i}',
                    prompt=f'work {i}', system_prompt='sys', cwd=Path('/tmp'),
                )
                for i in range(3)
            ]
            results = await asyncio.wait_for(
                asyncio.gather(*tasks),
                timeout=5.0,
            )

        assert all(r.success for r in results)
        # At least one invocation should have been on B after A capped
        assert 'fake-token-B' in tokens_seen
        assert gate._accounts[0].capped is True

    async def test_three_concurrent_tasks_two_accounts_all_succeed(self):
        """3 concurrent tasks sharing one gate with 2 accounts all complete (no caps)."""
        gate = make_gate(['A', 'B'])

        invoke_count = [0]

        async def mock_invoke(**kwargs):
            invoke_count[0] += 1
            await asyncio.sleep(0)
            return make_result(success=True, output='ok', cost_usd=0.1)

        with (
            patch('shared.cli_invoke.invoke_claude_agent', new_callable=AsyncMock,
                  side_effect=mock_invoke),
            patch('shared.cli_invoke.asyncio.sleep', new_callable=AsyncMock),
        ):
            tasks = [
                invoke_with_cap_retry(
                    gate, f'task-{i}',
                    prompt=f'work {i}', system_prompt='sys', cwd=Path('/tmp'),
                )
                for i in range(3)
            ]
            results = await asyncio.wait_for(
                asyncio.gather(*tasks),
                timeout=5.0,
            )

        assert all(r.success for r in results)
        assert invoke_count[0] == 3

    async def test_five_concurrent_tasks_three_accounts_two_cap(self):
        """5 concurrent tasks, 3 accounts, 2 accounts cap -> all complete on C."""
        gate = make_gate(['A', 'B', 'C'])

        invoke_count = [0]

        async def mock_invoke(**kwargs):
            invoke_count[0] += 1
            token = kwargs.get('oauth_token')
            await asyncio.sleep(0)
            # First call on A caps it
            if token == 'fake-token-A' and not gate._accounts[0].capped:
                return make_result(stderr=_cap_stderr('A'), cost_usd=0.0)
            # First call on B caps it
            if token == 'fake-token-B' and not gate._accounts[1].capped:
                return make_result(stderr=_cap_stderr('B'), cost_usd=0.0)
            return make_result(success=True, output='ok', cost_usd=0.05)

        with (
            patch('shared.cli_invoke.invoke_claude_agent', new_callable=AsyncMock,
                  side_effect=mock_invoke),
            patch('shared.cli_invoke.asyncio.sleep', new_callable=AsyncMock),
        ):
            tasks = [
                invoke_with_cap_retry(
                    gate, f'task-{i}',
                    prompt=f'work {i}', system_prompt='sys', cwd=Path('/tmp'),
                )
                for i in range(5)
            ]
            results = await asyncio.wait_for(
                asyncio.gather(*tasks),
                timeout=5.0,
            )

        assert all(r.success for r in results)
        assert len(results) == 5
        assert gate._accounts[0].capped is True
        assert gate._accounts[1].capped is True
        assert gate._accounts[2].capped is False

    async def test_five_concurrent_tasks_three_accounts_no_deadlock(self):
        """5 concurrent tasks, 3 accounts, no caps, no deadlock."""
        gate = make_gate(['A', 'B', 'C'])

        async def mock_invoke(**kwargs):
            await asyncio.sleep(0)
            return make_result(success=True, output='ok', cost_usd=0.05)

        with (
            patch('shared.cli_invoke.invoke_claude_agent', new_callable=AsyncMock,
                  side_effect=mock_invoke),
            patch('shared.cli_invoke.asyncio.sleep', new_callable=AsyncMock),
        ):
            tasks = [
                invoke_with_cap_retry(
                    gate, f'task-{i}',
                    prompt=f'work {i}', system_prompt='sys', cwd=Path('/tmp'),
                )
                for i in range(5)
            ]
            results = await asyncio.wait_for(
                asyncio.gather(*tasks),
                timeout=5.0,
            )

        assert all(r.success for r in results)
        assert len(results) == 5


# ---------------------------------------------------------------------------
# TestCostTrackingIntegration
# ---------------------------------------------------------------------------


@pytest.mark.asyncio
class TestCostTrackingIntegration:

    async def test_cap_hit_then_success_records_both_events(self):
        """Cap hit -> save_account_event('cap_hit'); success -> save_invocation."""
        gate = make_gate(['A', 'B'])
        cost_store = MagicMock()
        cost_store.save_invocation = AsyncMock()
        cost_store.save_account_event = AsyncMock()

        capped = make_result(stderr=_cap_stderr('A'))
        ok = make_result(success=True, output='ok', cost_usd=0.5)

        with (
            patch('shared.cli_invoke.invoke_claude_agent', new_callable=AsyncMock,
                  side_effect=[capped, ok]),
            patch('shared.cli_invoke.asyncio.sleep', new_callable=AsyncMock),
        ):
            await invoke_with_cap_retry(
                gate, 'test-cost',
                cost_store=cost_store, run_id='run-1', project_id='proj',
                prompt='work', system_prompt='sys', cwd=Path('/tmp'),
            )

        cost_store.save_account_event.assert_awaited_once()
        evt_kwargs = cost_store.save_account_event.call_args.kwargs
        assert evt_kwargs['event_type'] == 'cap_hit'

        cost_store.save_invocation.assert_awaited_once()
        inv_kwargs = cost_store.save_invocation.call_args.kwargs
        assert inv_kwargs['cost_usd'] == 0.5
        assert inv_kwargs['capped'] is False

    async def test_no_cap_hit_only_save_invocation(self):
        """No cap hit -> save_invocation called, save_account_event not called."""
        gate = make_gate(['A'])
        cost_store = MagicMock()
        cost_store.save_invocation = AsyncMock()
        cost_store.save_account_event = AsyncMock()

        ok = make_result(success=True, output='ok', cost_usd=0.3)

        with patch('shared.cli_invoke.invoke_claude_agent', new_callable=AsyncMock,
                   return_value=ok):
            await invoke_with_cap_retry(
                gate, 'test-no-cap',
                cost_store=cost_store, run_id='run-1', project_id='proj',
                prompt='work', system_prompt='sys', cwd=Path('/tmp'),
            )

        cost_store.save_invocation.assert_awaited_once()
        cost_store.save_account_event.assert_not_awaited()

    async def test_cost_store_exception_does_not_crash(self, caplog):
        """Cost store exception doesn't crash the retry loop."""
        gate = make_gate(['A', 'B'])
        cost_store = MagicMock()
        cost_store.save_invocation = AsyncMock(side_effect=RuntimeError('db locked'))
        cost_store.save_account_event = AsyncMock(side_effect=RuntimeError('db locked'))

        capped = make_result(stderr=_cap_stderr('A'))
        ok = make_result(success=True, output='ok')

        with (
            patch('shared.cli_invoke.invoke_claude_agent', new_callable=AsyncMock,
                  side_effect=[capped, ok]),
            patch('shared.cli_invoke.asyncio.sleep', new_callable=AsyncMock),
        ):
            result = await invoke_with_cap_retry(
                gate, 'test-crash',
                cost_store=cost_store,
                prompt='work', system_prompt='sys', cwd=Path('/tmp'),
            )

        assert result.success is True


# ---------------------------------------------------------------------------
# TestEdgeCases
# ---------------------------------------------------------------------------


@pytest.mark.asyncio
class TestEdgeCases:

    async def test_no_config_dir_no_crash(self):
        """invoke_with_cap_retry with no config_dir -> write_credentials never called."""
        gate = make_gate(['A'])

        ok = make_result(success=True, output='ok')

        with patch('shared.cli_invoke.invoke_claude_agent', new_callable=AsyncMock,
                   return_value=ok):
            result = await invoke_with_cap_retry(
                gate, 'test-no-configdir',
                config_dir=None,
                prompt='work', system_prompt='sys', cwd=Path('/tmp'),
            )

        assert result.success is True

    async def test_no_cost_store_no_crash(self):
        """invoke_with_cap_retry with no cost_store -> no save calls, no crash."""
        gate = make_gate(['A'])

        ok = make_result(success=True, output='ok')

        with patch('shared.cli_invoke.invoke_claude_agent', new_callable=AsyncMock,
                   return_value=ok):
            result = await invoke_with_cap_retry(
                gate, 'test-no-coststore',
                cost_store=None,
                prompt='work', system_prompt='sys', cwd=Path('/tmp'),
            )

        assert result.success is True

    async def test_very_long_stderr_cap_detection(self):
        """10KB stderr -> cap detection still works."""
        gate = make_gate(['A', 'B'])

        long_stderr = 'x' * 10_000 + "\nYou've hit your usage limit. Resets in 3h.\n"
        capped = make_result(stderr=long_stderr)
        ok = make_result(success=True, output='ok')

        with (
            patch('shared.cli_invoke.invoke_claude_agent', new_callable=AsyncMock,
                  side_effect=[capped, ok]),
            patch('shared.cli_invoke.asyncio.sleep', new_callable=AsyncMock),
        ):
            result = await invoke_with_cap_retry(
                gate, 'test-long-stderr',
                prompt='work', system_prompt='sys', cwd=Path('/tmp'),
            )

        assert result.success is True
        assert gate._accounts[0].capped is True

    async def test_result_with_none_optional_fields(self):
        """Result with all optional fields as None -> no crash on cost_store.save_invocation."""
        gate = make_gate(['A'])
        cost_store = MagicMock()
        cost_store.save_invocation = AsyncMock()

        ok = AgentResult(
            success=True, output='ok', cost_usd=0.0, duration_ms=0,
            turns=0, session_id='', structured_output=None, subtype='',
            stderr='', account_name='', input_tokens=None, output_tokens=None,
            cache_read_tokens=None, cache_create_tokens=None,
        )

        with patch('shared.cli_invoke.invoke_claude_agent', new_callable=AsyncMock,
                   return_value=ok):
            result = await invoke_with_cap_retry(
                gate, 'test-none-fields',
                cost_store=cost_store, run_id='r', project_id='p',
                prompt='work', system_prompt='sys', cwd=Path('/tmp'),
            )

        assert result.success is True
        cost_store.save_invocation.assert_awaited_once()
        kwargs = cost_store.save_invocation.call_args.kwargs
        assert kwargs['input_tokens'] is None
        assert kwargs['output_tokens'] is None

    async def test_no_prompt_key_defaults_to_empty(self):
        """invoke_kwargs with no 'prompt' key -> original_prompt defaults to ''."""
        make_gate(['A', 'B'])

        capped = make_result(session_id='sess-1', stderr=_cap_stderr())
        ok = make_result(success=True, output='ok')

        with (
            patch('shared.cli_invoke.invoke_claude_agent', new_callable=AsyncMock,
                  side_effect=[capped, ok]) as mock_inv,
            patch('shared.cli_invoke.asyncio.sleep', new_callable=AsyncMock),
        ):
            # When no prompt key, the default is '' and invoke_claude_agent is
            # called without prompt kwarg initially (it would come from **invoke_kwargs)
            # But invoke_claude_agent requires 'prompt' positional, so we must
            # test that original_prompt captures '' when 'prompt' not in kwargs.
            # Since invoke_claude_agent(**invoke_kwargs) requires prompt, we use
            # the fallback behavior: on cap hit with no session_id,
            # invoke_kwargs['prompt'] = original_prompt = ''
            # With session_id, the resume path sets prompt = CAP_HIT_RESUME_PROMPT.
            # On resume failure, it falls back to original_prompt = ''.
            pass

        # The actual behavior: original_prompt = invoke_kwargs.get('prompt', '')
        # which is '' when prompt is not in kwargs. This is an internal detail
        # that's hard to test end-to-end since invoke_claude_agent needs prompt.
        # We test via a fresh invoke after resume failure:
        gate2 = make_gate(['A', 'B', 'C'])
        capped = make_result(session_id='sess-1', stderr=_cap_stderr())
        failed = make_result(success=False, output='fail')
        ok = make_result(success=True, output='ok')

        with (
            patch('shared.cli_invoke.invoke_claude_agent', new_callable=AsyncMock,
                  side_effect=[capped, failed, ok]) as mock_inv,
            patch('shared.cli_invoke.asyncio.sleep', new_callable=AsyncMock),
        ):
            await invoke_with_cap_retry(
                gate2, 'test-no-prompt',
                system_prompt='sys', cwd=Path('/tmp'),
                # NOTE: no prompt= keyword
            )

        # Third call should use original_prompt which is '' (no prompt kwarg)
        third = mock_inv.call_args_list[2]
        assert third.kwargs.get('prompt') == ''

    async def test_no_model_key_defaults_to_opus(self):
        """invoke_kwargs with no 'model' key -> model defaults to 'opus'."""
        gate = make_gate(['A'])
        cost_store = MagicMock()
        cost_store.save_invocation = AsyncMock()

        ok = make_result(success=True, output='ok')

        with patch('shared.cli_invoke.invoke_claude_agent', new_callable=AsyncMock,
                   return_value=ok):
            await invoke_with_cap_retry(
                gate, 'test-default-model',
                cost_store=cost_store, run_id='r', project_id='p',
                prompt='work', system_prompt='sys', cwd=Path('/tmp'),
                # NOTE: no model= keyword
            )

        kwargs = cost_store.save_invocation.call_args.kwargs
        assert kwargs['model'] == 'opus'


# ---------------------------------------------------------------------------
# TestConfigDirIntegrationWithFailover
# ---------------------------------------------------------------------------


@pytest.mark.asyncio
class TestConfigDirIntegrationWithFailover:

    async def test_credentials_written_on_each_attempt(self, tmp_path):
        """TaskConfigDir.write_credentials called with each account's token."""
        gate = make_gate(['A', 'B'])
        config_dir = TaskConfigDir('failover-cred-test', base_dir=tmp_path)

        capped = make_result(stderr=_cap_stderr('A'))
        ok = make_result(success=True, output='ok')

        tokens_written = []
        original_write = config_dir.write_credentials

        def tracking_write(token):
            tokens_written.append(token)
            original_write(token)

        # Monkeypatch write_credentials to observe which token the retry loop
        # persists; pyright flags the method-assign because it breaks LSP.
        config_dir.write_credentials = tracking_write  # type: ignore[method-assign]

        with (
            patch('shared.cli_invoke.invoke_claude_agent', new_callable=AsyncMock,
                  side_effect=[capped, ok]),
            patch('shared.cli_invoke.asyncio.sleep', new_callable=AsyncMock),
        ):
            await invoke_with_cap_retry(
                gate, 'test-cred-write',
                config_dir=config_dir,
                prompt='work', system_prompt='sys', cwd=Path('/tmp'),
            )

        # Both tokens written (A then B)
        assert tokens_written == ['fake-token-A', 'fake-token-B']

        # Final credential file has B's token
        creds = json.loads((config_dir.path / '.credentials.json').read_text())
        assert creds['claudeAiOauth']['accessToken'] == 'fake-token-B'

        config_dir.cleanup()

    async def test_config_dir_path_passed_to_invoke(self, tmp_path):
        """config_dir.path passed to invoke_claude_agent."""
        gate = make_gate(['A'])
        config_dir = TaskConfigDir('path-test', base_dir=tmp_path)

        ok = make_result(success=True, output='ok')

        with patch('shared.cli_invoke.invoke_claude_agent', new_callable=AsyncMock,
                   return_value=ok) as mock_inv:
            await invoke_with_cap_retry(
                gate, 'test-path',
                config_dir=config_dir,
                prompt='work', system_prompt='sys', cwd=Path('/tmp'),
            )

        call_kwargs = mock_inv.call_args.kwargs
        assert call_kwargs['config_dir'] == config_dir.path

        config_dir.cleanup()
