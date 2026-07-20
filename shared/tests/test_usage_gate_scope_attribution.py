"""Scope-aware cap attribution (PRD task β — contract C3, invariant S5).

Threads each invocation's cap *scope* (its model when scoped, else ``None``)
through ``InvokeSlot.report`` / ``InvokeSlot.detect_cap_hit`` into the gate's
cap handlers, so a scoped (fable) ``CapHit`` caps ONLY that account's
model-scope (writing ``AccountState.scope_caps[m]``) while leaving the
account-level phase machine untouched — the account keeps serving general
work. Attribution is by *invoked model*, never by cap-message text
(PRD decision 2). This is the "write half" of boundary B1; γ (task 2857)
consumes it for scope-aware selection.

Isolated file with local minimal helpers (mirroring test_cap_retry.py /
test_usage_gate_exhaustive.py) so it stays independent of the big suites.
"""

from __future__ import annotations

import json
import os
from datetime import UTC, datetime
from unittest.mock import AsyncMock, patch

from shared.config_models import AccountConfig, UsageCapConfig
from shared.usage_gate import AccountPhase, UsageGate

SCOPE = 'claude-fable-5'


# ---------------------------------------------------------------------------
# Local helpers (mirroring test_cap_retry.py / test_usage_gate_exhaustive.py)
# ---------------------------------------------------------------------------

def make_gate(account_names: list[str], *, cost_store=None, **cfg) -> UsageGate:
    """Create a real UsageGate with fake accounts, probe + reset-wait disabled."""
    acct_cfgs = []
    env_vars: dict[str, str] = {}
    for name in account_names:
        env_key = f'TEST_TOKEN_{name.upper().replace("-", "_")}'
        env_vars[env_key] = f'fake-token-{name}'
        acct_cfgs.append(AccountConfig(name=name, oauth_token_env=env_key))
    cfg.setdefault('wait_for_reset', False)
    config = UsageCapConfig(accounts=acct_cfgs, **cfg)
    with patch.dict(os.environ, env_vars):
        gate = UsageGate(config, cost_store=cost_store)
    # Mock _run_probe to prevent real subprocess spawning.
    gate._run_probe = AsyncMock(return_value=True)
    return gate


def make_mock_cost_store() -> AsyncMock:
    store = AsyncMock()
    store.save_account_event = AsyncMock(return_value=None)
    store.save_invocation = AsyncMock(return_value=None)
    return store


# ===========================================================================
# Step 1 — scoped attribution in the cap/near-cap handlers (invariant S5)
# ===========================================================================


class TestScopedHandlerAttribution:
    """A scoped ``scope=<model>`` CapHit/NearCap writes ``acct.scope_caps[m]``
    and does NOT touch the account-level phase machine or flags."""

    # --- scoped cap: state -------------------------------------------------

    def test_scoped_cap_writes_scope_cap_not_account_phase(self):
        gate = make_gate(['a'])
        acct = gate._accounts[0]
        target = datetime(2026, 5, 1, 18, 0, tzinfo=UTC)

        assert (
            gate._handle_cap_detected('reason', target, acct.token, scope=SCOPE)
            is True
        )

        sc = acct.scope_caps[SCOPE]
        assert sc.capped is True
        assert sc.resets_at == target
        assert sc.capped_at is not None
        # Account-level phase machine untouched (invariant S5).
        assert acct.phase == AccountPhase.AVAILABLE
        # No stray scope keys created for other models.
        assert set(acct.scope_caps) == {SCOPE}

    def test_scoped_cap_stores_unknown_resets_at_as_none(self):
        """resets_at reaches the handler already parsed (decision 2); an
        unknown (None) is stored verbatim — the None-backoff policy is γ's."""
        gate = make_gate(['a'])
        acct = gate._accounts[0]

        gate._handle_cap_detected('reason', None, acct.token, scope=SCOPE)

        sc = acct.scope_caps[SCOPE]
        assert sc.capped is True
        assert sc.resets_at is None
        assert acct.phase == AccountPhase.AVAILABLE

    # --- scoped cap: cost event -------------------------------------------

    def test_scoped_cap_fires_cap_hit_event_with_scope(self):
        gate = make_gate(['a'], cost_store=make_mock_cost_store())
        acct = gate._accounts[0]
        target = datetime(2026, 5, 1, 18, 0, tzinfo=UTC)

        with patch.object(gate, '_fire_cost_event') as mock_fire:
            gate._handle_cap_detected('reason', target, acct.token, scope=SCOPE)

        (name, event_type, details_json), _ = mock_fire.call_args
        assert name == acct.name
        assert event_type == 'cap_hit'  # event NAME unchanged (tactical Q3)
        details = json.loads(details_json)
        assert details['scope'] == SCOPE
        assert details['reason'] == 'reason'
        assert details['resets_at'] == target.isoformat()

    def test_scoped_cap_event_omits_resets_at_when_unknown(self):
        gate = make_gate(['a'], cost_store=make_mock_cost_store())
        acct = gate._accounts[0]

        with patch.object(gate, '_fire_cost_event') as mock_fire:
            gate._handle_cap_detected('reason', None, acct.token, scope=SCOPE)

        (_, event_type, details_json), _ = mock_fire.call_args
        assert event_type == 'cap_hit'
        details = json.loads(details_json)
        assert 'resets_at' not in details
        assert details == {'reason': 'reason', 'scope': SCOPE}

    # --- scoped near-cap: state -------------------------------------------

    def test_scoped_near_cap_writes_scope_cap_not_account_flag(self):
        gate = make_gate(['a'])
        acct = gate._accounts[0]

        assert (
            gate._handle_near_cap_warning('reason', acct.token, scope=SCOPE) is True
        )

        assert acct.scope_caps[SCOPE].near_cap is True
        # Account-level near_cap flag untouched (invariant S5).
        assert acct.near_cap is False
        assert acct.phase == AccountPhase.AVAILABLE

    # --- scoped near-cap: cost event --------------------------------------

    def test_scoped_near_cap_fires_near_cap_event_with_scope(self):
        gate = make_gate(['a'], cost_store=make_mock_cost_store())
        acct = gate._accounts[0]

        with patch.object(gate, '_fire_cost_event') as mock_fire:
            gate._handle_near_cap_warning('reason', acct.token, scope=SCOPE)

        (name, event_type, details_json), _ = mock_fire.call_args
        assert name == acct.name
        assert event_type == 'near_cap'  # event NAME unchanged (tactical Q3)
        details = json.loads(details_json)
        assert details['scope'] == SCOPE
        assert details['reason'] == 'reason'


class TestUnscopedHandlerByteEquivalence:
    """``scope=None`` (the general scope) keeps today's account-level paths
    byte-identical: account phase machine + flags + cost-event shape."""

    def test_unscoped_cap_transitions_account_no_scope_caps(self):
        gate = make_gate(['a'])
        acct = gate._accounts[0]
        target = datetime(2026, 5, 1, 18, 0, tzinfo=UTC)

        assert (
            gate._handle_cap_detected('reason', target, acct.token, scope=None)
            is True
        )

        assert acct.phase == AccountPhase.CAPPED
        assert acct.scope_caps == {}

    def test_unscoped_cap_event_has_no_scope_key(self):
        gate = make_gate(['a'], cost_store=make_mock_cost_store())
        acct = gate._accounts[0]
        target = datetime(2026, 5, 1, 18, 0, tzinfo=UTC)

        with patch.object(gate, '_fire_cost_event') as mock_fire:
            gate._handle_cap_detected('reason', target, acct.token, scope=None)

        (_, event_type, details_json), _ = mock_fire.call_args
        assert event_type == 'cap_hit'
        details = json.loads(details_json)
        assert details == {'reason': 'reason', 'resets_at': target.isoformat()}
        assert 'scope' not in details

    def test_unscoped_near_cap_sets_account_flag_no_scope_caps(self):
        gate = make_gate(['a'])
        acct = gate._accounts[0]

        assert gate._handle_near_cap_warning('reason', acct.token, scope=None) is True

        assert acct.near_cap is True
        assert acct.scope_caps == {}

    def test_unscoped_near_cap_event_has_no_scope_key(self):
        gate = make_gate(['a'], cost_store=make_mock_cost_store())
        acct = gate._accounts[0]

        with patch.object(gate, '_fire_cost_event') as mock_fire:
            gate._handle_near_cap_warning('reason', acct.token, scope=None)

        (_, event_type, details_json), _ = mock_fire.call_args
        assert event_type == 'near_cap'
        details = json.loads(details_json)
        assert details == {'reason': 'reason'}
        assert 'scope' not in details
