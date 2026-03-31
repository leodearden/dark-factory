"""Tests for UsageGate — race condition fixes and robustness improvements."""

from __future__ import annotations

import asyncio
import os
from datetime import UTC, datetime, timedelta
from unittest.mock import AsyncMock, MagicMock, patch

import pytest

from shared.config_models import AccountConfig, UsageCapConfig
from shared.usage_gate import UsageGate


# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------


def make_gate(
    account_names: list[str],
    *,
    cost_store=None,
    wait_for_reset: bool = False,
    session_budget_usd: float | None = None,
) -> UsageGate:
    """Create a UsageGate with the given account names and auto-generated env vars.

    Tokens are injected via patch.dict during construction so _init_accounts()
    finds them.  After construction the gate's _accounts list holds AccountState
    objects with token='fake-token-<name>'.
    """
    acct_cfgs = []
    env_vars: dict[str, str] = {}
    for name in account_names:
        env_key = f'TEST_TOKEN_{name.upper().replace("-", "_")}'
        env_vars[env_key] = f'fake-token-{name}'
        acct_cfgs.append(AccountConfig(name=name, oauth_token_env=env_key))

    config = UsageCapConfig(
        accounts=acct_cfgs,
        wait_for_reset=wait_for_reset,
        session_budget_usd=session_budget_usd,
    )

    with patch.dict(os.environ, env_vars):
        gate = UsageGate(config, cost_store=cost_store)

    return gate


def make_mock_cost_store() -> AsyncMock:
    """Return an AsyncMock that satisfies the CostStore interface."""
    store = AsyncMock()
    store.save_account_event = AsyncMock(return_value=None)
    return store
