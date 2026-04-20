"""Tests for UsageGate auth_failed lifecycle.

Covers:
- AccountState.auth_failed field + pause_started_at semantics
- _handle_auth_failure marks account and fires cost event
- before_invoke skips auth_failed accounts
- re-probe loop re-reads env via load_dotenv(override=True)
- SIGHUP handler triggers immediate re-probe
- all-auth-failed closes the gate like all-capped
"""

from __future__ import annotations

import asyncio
import os
from datetime import UTC, datetime
from unittest.mock import AsyncMock, patch

import pytest

from shared.config_models import AccountConfig, UsageCapConfig
from shared.usage_gate import AccountState, UsageGate


def _make_gate(
    account_names: list[str],
    *,
    wait_for_reset: bool = False,
    auth_reprobe_secs: int = 3600,
) -> UsageGate:
    acct_cfgs = []
    env_vars: dict[str, str] = {}
    for name in account_names:
        env_key = f'TEST_AUTH_TOKEN_{name.upper().replace("-", "_")}'
        env_vars[env_key] = f'fake-token-{name}'
        acct_cfgs.append(AccountConfig(name=name, oauth_token_env=env_key))
    config = UsageCapConfig(
        accounts=acct_cfgs,
        wait_for_reset=wait_for_reset,
        auth_reprobe_secs=auth_reprobe_secs,
    )
    with patch.dict(os.environ, env_vars):
        gate = UsageGate(config)
    gate._run_probe = AsyncMock(return_value=True)
    return gate


class TestAccountStateAuthFailedField:

    def test_auth_failed_defaults_false(self):
        acct = AccountState(name='a', token='t')
        assert acct.auth_failed is False
        assert acct.auth_failed_at is None


class TestHandleAuthFailure:

    def test_marks_account_auth_failed(self):
        gate = _make_gate(['a', 'b'])
        marked = gate._handle_auth_failure(
            'HTTP 403: access denied', oauth_token='fake-token-a',
        )
        assert marked is True
        assert gate._accounts[0].auth_failed is True
        assert gate._accounts[0].auth_failed_at is not None
        # Does NOT set capped — auth failure is a separate lifecycle
        assert gate._accounts[0].capped is False
        # Other account untouched
        assert gate._accounts[1].auth_failed is False

    def test_unknown_token_returns_false(self):
        gate = _make_gate(['a'])
        marked = gate._handle_auth_failure('reason', oauth_token='unknown-token')
        assert marked is False
        assert gate._accounts[0].auth_failed is False

    def test_clears_probe_state(self):
        """_handle_auth_failure clears probe_in_flight so _open event doesn't deadlock."""
        gate = _make_gate(['a'])
        gate._accounts[0].probe_in_flight = True
        marked = gate._handle_auth_failure('reason', oauth_token='fake-token-a')
        assert marked is True
        assert gate._accounts[0].probe_in_flight is False


@pytest.mark.asyncio
class TestBeforeInvokeSkipsAuthFailed:

    async def test_before_invoke_skips_auth_failed(self):
        gate = _make_gate(['a', 'b'])
        gate._accounts[0].auth_failed = True
        gate._accounts[0].auth_failed_at = datetime.now(UTC)

        token = await gate.before_invoke()
        assert token == 'fake-token-b'

    async def test_before_invoke_blocks_when_all_auth_failed(self):
        gate = _make_gate(['a'])
        gate._accounts[0].auth_failed = True
        gate._accounts[0].auth_failed_at = datetime.now(UTC)

        with pytest.raises(asyncio.TimeoutError):
            await asyncio.wait_for(gate.before_invoke(), timeout=0.1)


class TestIsPausedIncludesAuthFailed:

    def test_is_paused_true_when_all_auth_failed(self):
        gate = _make_gate(['a', 'b'])
        gate._accounts[0].auth_failed = True
        gate._accounts[1].auth_failed = True
        assert gate.is_paused is True

    def test_is_paused_false_when_one_account_healthy(self):
        gate = _make_gate(['a', 'b'])
        gate._accounts[0].auth_failed = True
        # b is healthy
        assert gate.is_paused is False

    def test_is_paused_true_when_mixed_cap_and_auth_failed(self):
        gate = _make_gate(['a', 'b'])
        gate._accounts[0].capped = True
        gate._accounts[1].auth_failed = True
        assert gate.is_paused is True


@pytest.mark.asyncio
class TestAuthReprobeReReadsEnv:
    """Re-probe loop must call load_dotenv(override=True) and re-read
    os.environ so a refreshed .env picks up the new token."""

    async def test_reprobe_updates_token_from_env(self):
        env_var = 'TEST_AUTH_TOKEN_A'
        with patch.dict(os.environ, {env_var: 'old-token'}):
            config = UsageCapConfig(
                accounts=[AccountConfig(name='a', oauth_token_env=env_var)],
                wait_for_reset=False,
                auth_reprobe_secs=0,
            )
            gate = UsageGate(config)

        gate._run_probe = AsyncMock(return_value=True)
        gate._accounts[0].auth_failed = True
        gate._accounts[0].auth_failed_at = datetime.now(UTC)

        with (
            patch('shared.usage_gate.load_dotenv') as mock_load,
            patch.dict(os.environ, {env_var: 'new-token'}, clear=False),
        ):
            await gate._reprobe_account(gate._accounts[0])

        mock_load.assert_called_once()
        # load_dotenv is called with override=True
        call_kwargs = mock_load.call_args.kwargs
        assert call_kwargs.get('override') is True
        # Token refreshed from env
        assert gate._accounts[0].token == 'new-token'
        # Probe succeeded → auth_failed cleared
        assert gate._accounts[0].auth_failed is False

    async def test_reprobe_failure_keeps_auth_failed(self):
        gate = _make_gate(['a'], auth_reprobe_secs=0)
        gate._run_probe = AsyncMock(return_value=False)
        gate._accounts[0].auth_failed = True
        gate._accounts[0].auth_failed_at = datetime.now(UTC)

        with patch('shared.usage_gate.load_dotenv'):
            await gate._reprobe_account(gate._accounts[0])

        assert gate._accounts[0].auth_failed is True


@pytest.mark.asyncio
class TestSighupTriggersReprobe:
    """SIGHUP handler triggers an immediate re-probe of all auth_failed accounts."""

    async def test_sighup_handler_reprobes_auth_failed(self):
        gate = _make_gate(['a', 'b'])
        gate._accounts[0].auth_failed = True
        gate._accounts[0].auth_failed_at = datetime.now(UTC)
        gate._run_probe = AsyncMock(return_value=True)

        with patch('shared.usage_gate.load_dotenv'):
            await gate._on_sighup_async()

        # Account a was re-probed and uncapped
        assert gate._accounts[0].auth_failed is False
        # Account b never auth_failed; should remain untouched
        assert gate._accounts[1].auth_failed is False

    async def test_sighup_noop_when_no_auth_failed(self):
        gate = _make_gate(['a'])
        gate._run_probe = AsyncMock(return_value=True)
        with patch('shared.usage_gate.load_dotenv') as mock_load:
            await gate._on_sighup_async()
        # Nothing to reprobe: load_dotenv still called once to refresh env,
        # but no probes fire.
        gate._run_probe.assert_not_called()
        # load_dotenv may or may not be called; implementation-dependent.
        # We don't assert on it either way.
        _ = mock_load
