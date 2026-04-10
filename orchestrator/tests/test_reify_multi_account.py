"""Tests validating multi-account failover configuration for reify's orchestrator.

Tests are organized to mirror the implementation steps:
  - step-1/2:  Reify accounts YAML structure (UsageCapConfig parsing)
  - step-3/4:  Config loading via load_config() with accounts_file
  - step-5/6:  UsageGate initialization from 5-account config
  - step-7/8:  Cap hit on first account rotates to second
  - step-9/10: All accounts capped → blocks → one uncapped → resumes
"""

from __future__ import annotations

import asyncio
import os
from datetime import UTC, datetime, timedelta
from unittest.mock import AsyncMock, MagicMock, patch

import pytest
import yaml
from shared.config_models import AccountConfig, UsageCapConfig
from shared.usage_gate import AccountState, UsageGate

# ---------------------------------------------------------------------------
# Constants — expected reify automation account pool (F→E→C→B→D)
# ---------------------------------------------------------------------------

REIFY_ACCOUNT_DEFS = [
    {'name': 'max-f', 'oauth_token_env': 'CLAUDE_OAUTH_TOKEN_F'},
    {'name': 'max-e', 'oauth_token_env': 'CLAUDE_OAUTH_TOKEN_E'},
    {'name': 'max-c', 'oauth_token_env': 'CLAUDE_OAUTH_TOKEN_C'},
    {'name': 'max-b', 'oauth_token_env': 'CLAUDE_OAUTH_TOKEN_B'},
    {'name': 'max-d', 'oauth_token_env': 'CLAUDE_OAUTH_TOKEN_D'},
]

REIFY_ACCOUNT_NAMES = ['max-f', 'max-e', 'max-c', 'max-b', 'max-d']
REIFY_ENV_VARS = [
    'CLAUDE_OAUTH_TOKEN_F',
    'CLAUDE_OAUTH_TOKEN_E',
    'CLAUDE_OAUTH_TOKEN_C',
    'CLAUDE_OAUTH_TOKEN_B',
    'CLAUDE_OAUTH_TOKEN_D',
]


# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------

def _make_reify_gate(wait_for_reset: bool = False) -> UsageGate:
    """Create a UsageGate with reify's 5 automation accounts.

    Tokens are injected directly (no env var lookup) for test isolation.
    Mirrors the _make_gate() pattern in test_usage_gate.py.
    """
    acct_cfgs = [AccountConfig(**d) for d in REIFY_ACCOUNT_DEFS]
    config = UsageCapConfig(accounts=acct_cfgs, wait_for_reset=wait_for_reset)
    gate = UsageGate.__new__(UsageGate)
    gate._config = config
    gate._open = asyncio.Event()
    gate._open.set()
    gate._lock = asyncio.Lock()
    gate._cumulative_cost = 0.0
    gate._paused_reason = ''
    gate._pause_started_at = None
    gate._total_pause_secs = 0.0
    gate._cost_store = None
    gate._project_id = None
    gate._run_id = None
    gate._last_account_name = None
    gate._background_tasks = set()
    gate._probe_config_dir = MagicMock()
    gate._run_probe = AsyncMock(return_value=True)
    gate._accounts = [
        AccountState(name=d['name'], token=f'token-{d["name"]}')
        for d in REIFY_ACCOUNT_DEFS
    ]
    return gate


# ---------------------------------------------------------------------------
# step-1: Validate the YAML structure for a reify accounts file
# ---------------------------------------------------------------------------


class TestReifyAccountsYaml:
    """Validates that a reify-style accounts YAML parses into 5 AccountConfig entries."""

    def test_reify_accounts_yaml_has_five_automation_accounts(self, tmp_path):
        """A YAML fixture with reify's 5 accounts parses correctly via UsageCapConfig."""
        fixture = tmp_path / 'usage-accounts.yaml'
        fixture.write_text(yaml.dump({'accounts': REIFY_ACCOUNT_DEFS}))

        config = UsageCapConfig(accounts_file=str(fixture))

        assert len(config.accounts) == 5, (
            f"Expected 5 accounts, got {len(config.accounts)}"
        )

        names = [a.name for a in config.accounts]
        assert names == REIFY_ACCOUNT_NAMES, (
            f"Account names don't match expected order: {names}"
        )

        env_keys = [a.oauth_token_env for a in config.accounts]
        assert env_keys == REIFY_ENV_VARS, (
            f"Env var keys don't match: {env_keys}"
        )

    def test_account_names_follow_max_letter_convention(self, tmp_path):
        """All account names follow the 'max-<letter>' naming convention."""
        fixture = tmp_path / 'usage-accounts.yaml'
        fixture.write_text(yaml.dump({'accounts': REIFY_ACCOUNT_DEFS}))

        config = UsageCapConfig(accounts_file=str(fixture))

        for acct in config.accounts:
            assert acct.name.startswith('max-'), (
                f"Account name {acct.name!r} doesn't follow 'max-<letter>' convention"
            )
            assert len(acct.name) == len('max-f'), (
                f"Account name {acct.name!r} has wrong length (expected 'max-X' pattern)"
            )

    def test_env_var_names_follow_claude_oauth_token_convention(self, tmp_path):
        """All oauth_token_env fields follow CLAUDE_OAUTH_TOKEN_<LETTER> pattern."""
        fixture = tmp_path / 'usage-accounts.yaml'
        fixture.write_text(yaml.dump({'accounts': REIFY_ACCOUNT_DEFS}))

        config = UsageCapConfig(accounts_file=str(fixture))

        for acct in config.accounts:
            assert acct.oauth_token_env.startswith('CLAUDE_OAUTH_TOKEN_'), (
                f"{acct.oauth_token_env!r} doesn't start with 'CLAUDE_OAUTH_TOKEN_'"
            )

    def test_max_a_not_in_automation_pool(self, tmp_path):
        """max-a (interactive account) is NOT in the automation pool."""
        fixture = tmp_path / 'usage-accounts.yaml'
        fixture.write_text(yaml.dump({'accounts': REIFY_ACCOUNT_DEFS}))

        config = UsageCapConfig(accounts_file=str(fixture))

        names = [a.name for a in config.accounts]
        assert 'max-a' not in names, "max-a (interactive account) should not be in automation pool"
        assert 'CLAUDE_OAUTH_TOKEN_A' not in [a.oauth_token_env for a in config.accounts]


# ---------------------------------------------------------------------------
# step-3: Validate config loading with accounts_file via load_config()
# ---------------------------------------------------------------------------


class TestReifyConfigLoadsUsageCap:
    """Validates that a reify-style orchestrator YAML with usage_cap section loads correctly."""

    def test_reify_config_loads_usage_cap_with_accounts_file(self, tmp_path):
        """An orchestrator YAML with usage_cap.accounts_file loads 5 accounts."""
        from orchestrator.config import load_config

        # Create the accounts file
        accounts_file = tmp_path / 'usage-accounts.yaml'
        accounts_file.write_text(yaml.dump({'accounts': REIFY_ACCOUNT_DEFS}))

        # Create orchestrator config that references the accounts file
        config_data = {
            'project_root': str(tmp_path),
            'usage_cap': {
                'enabled': True,
                'pause_threshold': 0.96,
                'wait_for_reset': True,
                'probe_interval_secs': 300,
                'max_probe_interval_secs': 1800,
                'accounts_file': str(accounts_file),
            },
        }
        config_path = tmp_path / 'orchestrator.yaml'
        config_path.write_text(yaml.dump(config_data))

        config = load_config(config_path)

        assert config.usage_cap.enabled is True
        assert config.usage_cap.pause_threshold == pytest.approx(0.96)
        assert config.usage_cap.wait_for_reset is True
        assert config.usage_cap.probe_interval_secs == 300
        assert config.usage_cap.max_probe_interval_secs == 1800
        assert len(config.usage_cap.accounts) == 5, (
            f"Expected 5 accounts, got {len(config.usage_cap.accounts)}"
        )

    def test_usage_cap_accounts_have_correct_names_after_loading(self, tmp_path):
        """Accounts loaded via accounts_file have the correct names and env vars."""
        from orchestrator.config import load_config

        accounts_file = tmp_path / 'usage-accounts.yaml'
        accounts_file.write_text(yaml.dump({'accounts': REIFY_ACCOUNT_DEFS}))

        config_path = tmp_path / 'orchestrator.yaml'
        config_path.write_text(yaml.dump({
            'project_root': str(tmp_path),
            'usage_cap': {'accounts_file': str(accounts_file)},
        }))

        config = load_config(config_path)

        names = [a.name for a in config.usage_cap.accounts]
        assert names == REIFY_ACCOUNT_NAMES

        env_keys = [a.oauth_token_env for a in config.usage_cap.accounts]
        assert env_keys == REIFY_ENV_VARS


# ---------------------------------------------------------------------------
# step-5: Validate UsageGate initializes with 5 accounts when tokens are set
# ---------------------------------------------------------------------------


class TestGateFromReifyConfig:
    """Validates UsageGate initializes correctly from a 5-account config."""

    def test_gate_from_reify_config_initializes_five_accounts(self):
        """UsageGate with all 5 tokens set initializes 5 AccountState entries."""
        acct_cfgs = [AccountConfig(**d) for d in REIFY_ACCOUNT_DEFS]
        config = UsageCapConfig(accounts=acct_cfgs)

        env_vars = {d['oauth_token_env']: f'token-{d["name"]}' for d in REIFY_ACCOUNT_DEFS}

        with patch.dict(os.environ, env_vars):
            gate = UsageGate(config)

        assert len(gate._accounts) == 5, (
            f"Expected 5 AccountState entries, got {len(gate._accounts)}"
        )
        for state, defn in zip(gate._accounts, REIFY_ACCOUNT_DEFS, strict=False):
            assert state.name == defn['name']
            assert state.token is not None, (
                f"Account {state.name!r} has None token — env var not resolved"
            )
            assert state.token == f'token-{defn["name"]}'

    def test_gate_skips_account_when_token_missing(self):
        """UsageGate skips accounts whose env var is not set."""
        acct_cfgs = [AccountConfig(**d) for d in REIFY_ACCOUNT_DEFS]
        config = UsageCapConfig(accounts=acct_cfgs)

        # Set only 3 of the 5 tokens
        partial_env = {
            'CLAUDE_OAUTH_TOKEN_F': 'token-f',
            'CLAUDE_OAUTH_TOKEN_E': 'token-e',
            'CLAUDE_OAUTH_TOKEN_C': 'token-c',
        }

        with patch.dict(os.environ, partial_env, clear=True):
            gate = UsageGate(config)

        # Should have exactly 3 accounts (skipped B and D)
        assert len(gate._accounts) == 3
        names = [a.name for a in gate._accounts]
        assert names == ['max-f', 'max-e', 'max-c']

    def test_gate_account_names_match_reify_pool(self):
        """Gate._accounts names match reify's 5-account pool in correct order."""
        gate = _make_reify_gate()

        names = [a.name for a in gate._accounts]
        assert names == REIFY_ACCOUNT_NAMES


# ---------------------------------------------------------------------------
# step-7: Cap hit on first account rotates to second
# ---------------------------------------------------------------------------


class TestCapHitRotation:
    """Validates that capping one account causes before_invoke() to return the next."""

    @pytest.mark.asyncio
    async def test_cap_hit_on_first_account_rotates_to_second(self):
        """Capping max-f causes before_invoke() to return max-e's token."""
        gate = _make_reify_gate()

        # First call returns max-f's token
        token = await asyncio.wait_for(gate.before_invoke(), timeout=0.5)
        assert token == 'token-max-f', f"Expected 'token-max-f', got {token!r}"

        # Simulate cap hit on max-f
        gate._handle_cap_detected(
            reason='You hit your limit',
            resets_at=datetime.now(UTC) + timedelta(hours=1),
            oauth_token='token-max-f',
        )

        # Second call should return max-e's token
        token = await asyncio.wait_for(gate.before_invoke(), timeout=0.5)
        assert token == 'token-max-e', f"Expected 'token-max-e', got {token!r}"

    @pytest.mark.asyncio
    async def test_failover_chain_f_to_e_to_c(self):
        """Capping F then E causes before_invoke() to return C's token."""
        gate = _make_reify_gate()

        # Cap max-f
        gate._handle_cap_detected(
            reason='cap-f',
            resets_at=datetime.now(UTC) + timedelta(hours=1),
            oauth_token='token-max-f',
        )
        # Cap max-e
        gate._handle_cap_detected(
            reason='cap-e',
            resets_at=datetime.now(UTC) + timedelta(hours=1),
            oauth_token='token-max-e',
        )

        # Should return max-c's token
        token = await asyncio.wait_for(gate.before_invoke(), timeout=0.5)
        assert token == 'token-max-c', f"Expected 'token-max-c', got {token!r}"

    @pytest.mark.asyncio
    async def test_failover_chain_full_f_e_c_b_to_d(self):
        """Capping F, E, C, B causes before_invoke() to return D's token."""
        gate = _make_reify_gate()

        for acct_name in ('max-f', 'max-e', 'max-c', 'max-b'):
            gate._handle_cap_detected(
                reason=f'cap-{acct_name}',
                resets_at=datetime.now(UTC) + timedelta(hours=1),
                oauth_token=f'token-{acct_name}',
            )

        token = await asyncio.wait_for(gate.before_invoke(), timeout=0.5)
        assert token == 'token-max-d', f"Expected 'token-max-d', got {token!r}"


# ---------------------------------------------------------------------------
# step-9: All five accounts capped → gate blocks → one uncapped → resumes
# ---------------------------------------------------------------------------


class TestAllCappedBlockResume:
    """Validates full block-and-resume cycle with reify's 5-account set."""

    @pytest.mark.asyncio
    async def test_all_five_reify_accounts_capped_blocks_then_resumes(self):
        """Capping all 5 accounts blocks gate; uncapping one lets it resume."""
        gate = _make_reify_gate()

        # Cap all 5 accounts
        for defn in REIFY_ACCOUNT_DEFS:
            gate._handle_cap_detected(
                reason=f'cap-{defn["name"]}',
                resets_at=datetime.now(UTC) + timedelta(hours=1),
                oauth_token=f'token-{defn["name"]}',
            )

        # Gate should be paused (all capped)
        assert gate.is_paused, "Expected gate to be paused when all accounts are capped"
        assert not gate._open.is_set(), "Expected _open event to be cleared when all capped"

        # Uncap max-e and signal the gate to open
        gate._accounts[1].capped = False  # max-e is index 1
        gate._open.set()

        # Gate should unblock and return max-e's token
        token = await asyncio.wait_for(gate.before_invoke(), timeout=0.5)
        assert token == 'token-max-e', (
            f"Expected 'token-max-e' after uncapping, got {token!r}"
        )

    @pytest.mark.asyncio
    async def test_gate_blocks_while_all_capped(self):
        """before_invoke() blocks (asyncio.TimeoutError) when all 5 are capped."""
        gate = _make_reify_gate()

        # Cap all 5 — resets_at in future so auto-resume won't trigger
        for defn in REIFY_ACCOUNT_DEFS:
            gate._handle_cap_detected(
                reason=f'cap-{defn["name"]}',
                resets_at=datetime.now(UTC) + timedelta(hours=1),
                oauth_token=f'token-{defn["name"]}',
            )

        assert gate.is_paused

        with pytest.raises(asyncio.TimeoutError):
            await asyncio.wait_for(gate.before_invoke(), timeout=0.1)

    @pytest.mark.asyncio
    async def test_uncapping_first_account_resumes_from_beginning(self):
        """Uncapping max-f (first) gives max-f's token on resume."""
        gate = _make_reify_gate()

        # Cap all
        for defn in REIFY_ACCOUNT_DEFS:
            gate._handle_cap_detected(
                reason=f'cap-{defn["name"]}',
                resets_at=datetime.now(UTC) + timedelta(hours=1),
                oauth_token=f'token-{defn["name"]}',
            )

        # Uncap max-f (index 0)
        gate._accounts[0].capped = False
        gate._open.set()

        token = await asyncio.wait_for(gate.before_invoke(), timeout=0.5)
        assert token == 'token-max-f'

    def test_is_paused_reflects_all_capped_state(self):
        """gate.is_paused is True only when all 5 accounts are capped."""
        gate = _make_reify_gate()

        # Not paused initially
        assert not gate.is_paused

        # Cap accounts one by one — gate is paused only when ALL are capped
        for i, _defn in enumerate(REIFY_ACCOUNT_DEFS):
            gate._accounts[i].capped = True
            if i < len(REIFY_ACCOUNT_DEFS) - 1:
                assert not gate.is_paused, (
                    f"Gate should not be paused after capping only {i+1} of 5 accounts"
                )
            else:
                assert gate.is_paused, "Gate should be paused when all 5 accounts are capped"


# ---------------------------------------------------------------------------
# Unknown-token fallback: _handle_cap_detected with a token that matches no account
# ---------------------------------------------------------------------------


class TestCapDetectedUnknownToken:
    """Characterizes the best-guess fallback in _handle_cap_detected.

    When oauth_token does not match any configured account,
    _handle_cap_detected walks _accounts and picks the first non-capped
    account as the victim (usage_gate.py ~line 283).  If all accounts are
    already capped it logs a warning and returns without mutating state.
    """

    def test_unknown_token_falls_back_to_first_uncapped_account(self):
        """Unknown token causes _handle_cap_detected to cap the first uncapped account."""
        gate = _make_reify_gate()  # all 5 accounts uncapped

        gate._handle_cap_detected(
            reason='unknown-token-cap',
            resets_at=datetime.now(UTC) + timedelta(hours=1),
            oauth_token='not-a-real-token',
        )

        # max-f (index 0) is the first uncapped account — it becomes the victim
        assert gate._accounts[0].capped, "max-f should be capped (best-guess fallback victim)"
        # The remaining 4 accounts should be untouched
        for acct in gate._accounts[1:]:
            assert not acct.capped, (
                f"{acct.name} should still be uncapped after unknown-token fallback"
            )
        # Gate is not paused because 4 accounts remain available
        assert not gate.is_paused, (
            "Gate should not be paused: 4 of 5 accounts are still available"
        )

    def test_unknown_token_when_all_capped_is_noop(self):
        """Unknown token with all accounts capped is a no-op (logs warning, no state change)."""
        gate = _make_reify_gate()

        # Cap all 5 accounts using the same loop pattern as test_gate_blocks_while_all_capped
        future_reset = datetime.now(UTC) + timedelta(hours=1)
        for defn in REIFY_ACCOUNT_DEFS:
            gate._handle_cap_detected(
                reason=f'cap-{defn["name"]}',
                resets_at=future_reset,
                oauth_token=f'token-{defn["name"]}',
            )

        assert gate.is_paused, "Gate should be paused after capping all 5 accounts"

        # Snapshot the state of all 5 accounts before the unknown-token call
        snapshot = [
            (a.name, a.capped, a.resets_at)
            for a in gate._accounts
        ]

        # Call with an unknown token — should be a no-op since no uncapped account exists
        gate._handle_cap_detected(
            reason='stray-token-cap',
            resets_at=None,
            oauth_token='not-a-real-token',
        )

        # All 5 accounts must still be capped
        assert len(gate._accounts) == 5
        for acct in gate._accounts:
            assert acct.capped, (
                f"{acct.name} should still be capped after noop unknown-token call"
            )

        # No account state was mutated: compare against snapshot
        for i, (name, was_capped, was_resets_at) in enumerate(snapshot):
            acct = gate._accounts[i]
            assert acct.name == name
            assert acct.capped == was_capped, (
                f"{name}: capped changed unexpectedly"
            )
            assert acct.resets_at == was_resets_at, (
                f"{name}: resets_at changed unexpectedly"
            )

        # Gate remains paused
        assert gate.is_paused, "Gate must remain paused after noop unknown-token call"
