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
from pathlib import Path
from unittest.mock import patch

import pytest
import yaml
from shared.config_models import AccountConfig, UsageCapConfig
from shared.usage_gate import UsageGate

from tests.conftest import build_usage_gate

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
    Delegates to the shared build_usage_gate() helper in conftest.py.
    """
    acct_cfgs = [AccountConfig(**d) for d in REIFY_ACCOUNT_DEFS]
    tokens = [f'token-{d["name"]}' for d in REIFY_ACCOUNT_DEFS]
    return build_usage_gate(acct_cfgs, tokens, wait_for_reset=wait_for_reset)


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
        for state, defn in zip(gate._accounts, REIFY_ACCOUNT_DEFS, strict=True):
            assert state.name == defn['name']
            assert state.token is not None, (
                f"Account {state.name!r} has None token — env var not resolved"
            )
            assert state.token == f'token-{defn["name"]}'

    def test_gate_skips_account_when_token_missing(self, monkeypatch):
        """UsageGate skips accounts whose env var is not set."""
        acct_cfgs = [AccountConfig(**d) for d in REIFY_ACCOUNT_DEFS]
        config = UsageCapConfig(accounts=acct_cfgs)

        # Set only the 3 present tokens; explicitly delete the 2 missing ones so
        # ambient environment values (if any) don't accidentally satisfy the lookup.
        # monkeypatch restores the original env on test teardown automatically.
        monkeypatch.setenv('CLAUDE_OAUTH_TOKEN_F', 'token-f')
        monkeypatch.setenv('CLAUDE_OAUTH_TOKEN_E', 'token-e')
        monkeypatch.setenv('CLAUDE_OAUTH_TOKEN_C', 'token-c')
        monkeypatch.delenv('CLAUDE_OAUTH_TOKEN_B', raising=False)
        monkeypatch.delenv('CLAUDE_OAUTH_TOKEN_D', raising=False)

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

        # TEST SHORTCUT — bypasses the real production resume path.
        # In production, accounts are uncapped in one of two ways:
        #   (a) _refresh_capped_accounts() wakes periodically and checks whether
        #       resets_at has elapsed, then clears `capped` and calls _open.set(); or
        #   (b) _account_resume_probe_loop() fires a successful probe via _run_probe()
        #       and calls _open.set() itself after clearing `capped`.
        # Here we mutate `capped` and set the event directly so the test stays fast
        # and synchronous without needing to run background tasks or mock the clock.
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
    """Verifies that _handle_cap_detected is a no-op for explicit unknown tokens.

    When oauth_token does not match any configured account, _resolve_account logs a
    config-drift warning and returns None — no best-guess fallback applies.
    _handle_cap_detected then logs 'no matching account' and returns without mutating state.
    If all accounts are already capped the same no-op behaviour applies.
    """

    def test_unknown_token_does_not_cap_any_account(self):
        """Unknown token: no account is capped and gate is not paused."""
        gate = _make_reify_gate()  # all 5 accounts uncapped

        gate._handle_cap_detected(
            reason='unknown-token-cap',
            resets_at=datetime.now(UTC) + timedelta(hours=1),
            oauth_token='not-a-real-token',
        )

        # No account should be capped — unknown token is now a no-op
        for acct in gate._accounts:
            assert not acct.capped, (
                f"{acct.name} should remain uncapped after unknown-token call"
            )
        # Gate is not paused because all 5 accounts remain available
        assert not gate.is_paused, (
            "Gate should not be paused: all 5 accounts are still available"
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


# ---------------------------------------------------------------------------
# Production config pin: TestDarkFactoryProductionPool reads the REAL YAML
# ---------------------------------------------------------------------------


class TestDarkFactoryProductionPool:
    """Pins the real config/usage-accounts.yaml to the expected production account pool.

    Unlike the tests above (which use tmp_path fixtures), these tests load the
    actual shared config file from the repository root. They serve as a regression
    guard: if someone accidentally re-adds max-b or removes an active account
    during a merge conflict, the test fails immediately.

    Tests are skipped (not failed) when run outside a full repo checkout so that
    CI environments without the config file are not broken.
    """

    @staticmethod
    def _get_accounts_file():
        """Walk up from this file to the repo root and return the config file path.

        Returns None if the file is not found (e.g. in non-repo checkouts).
        """
        here = Path(__file__).resolve()
        for parent in list(here.parents):
            candidate = parent / 'config' / 'usage-accounts.yaml'
            if candidate.exists():
                return candidate
        return None

    def test_production_pool_accounts_in_expected_order(self):
        """Real config/usage-accounts.yaml has exactly [max-g, max-f, max-e, max-c, max-d] in order."""
        accounts_file = self._get_accounts_file()
        if accounts_file is None:
            pytest.skip("config/usage-accounts.yaml not found — not a full repo checkout")

        config = UsageCapConfig(accounts_file=str(accounts_file))
        names = [a.name for a in config.accounts]
        assert names == ['max-g', 'max-f', 'max-e', 'max-c', 'max-d'], (
            f"Production pool mismatch. Got: {names!r}. "
            "Expected max-b to be removed (permanently dead HTTP 403)."
        )

    def test_max_b_removed_from_production_pool(self):
        """max-b (permanently dead, HTTP 403 since 2026-04-20) must not appear in the pool."""
        accounts_file = self._get_accounts_file()
        if accounts_file is None:
            pytest.skip("config/usage-accounts.yaml not found — not a full repo checkout")

        config = UsageCapConfig(accounts_file=str(accounts_file))
        names = [a.name for a in config.accounts]
        env_vars = [a.oauth_token_env for a in config.accounts]
        assert 'max-b' not in names, (
            "max-b is permanently dead (HTTP 403) and must not be in the production pool"
        )
        assert 'CLAUDE_OAUTH_TOKEN_B' not in env_vars, (
            "CLAUDE_OAUTH_TOKEN_B must not appear in production pool (max-b is dead)"
        )

    def test_max_a_excluded_from_production_pool(self):
        """max-a (reserved for interactive/eval sessions) must not be in config/usage-accounts.yaml."""
        accounts_file = self._get_accounts_file()
        if accounts_file is None:
            pytest.skip("config/usage-accounts.yaml not found — not a full repo checkout")

        config = UsageCapConfig(accounts_file=str(accounts_file))
        names = [a.name for a in config.accounts]
        env_vars = [a.oauth_token_env for a in config.accounts]
        assert 'max-a' not in names, (
            "max-a (interactive account) must not be in the automation pool"
        )
        assert 'CLAUDE_OAUTH_TOKEN_A' not in env_vars, (
            "CLAUDE_OAUTH_TOKEN_A must not appear in production pool"
        )
