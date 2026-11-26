"""The reconciliation subsystem's single usage gate, built from its config (task 11783)."""

import pytest
from shared.config_models import AccountConfig, UsageCapConfig
from shared.usage_gate import UsageGate

from fused_memory.config.schema import ReconciliationConfig
from fused_memory.reconciliation.harness import build_reconciliation_usage_gate


def test_no_gate_when_the_reconciliation_usage_cap_is_disabled():
    config = ReconciliationConfig(usage_cap=UsageCapConfig(enabled=False))

    assert build_reconciliation_usage_gate(config) is None


@pytest.mark.asyncio
async def test_an_enabled_usage_cap_builds_a_gate_over_its_accounts(monkeypatch):
    monkeypatch.setenv('TEST_RECON_TOKEN_A', 'fake-token-acct-a')
    config = ReconciliationConfig(
        usage_cap=UsageCapConfig(
            accounts=[AccountConfig(name='acct-a', oauth_token_env='TEST_RECON_TOKEN_A')],
            wait_for_reset=False,
        ),
    )

    gate = build_reconciliation_usage_gate(config)
    try:
        assert isinstance(gate, UsageGate)
        assert gate.account_count == 1
    finally:
        if gate is not None:
            await gate.shutdown()
