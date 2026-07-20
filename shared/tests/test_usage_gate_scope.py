"""Tests for the UsageGate model-scoped-cap scope substrate (PRD task α / contract C1).

Covers the three purely-additive pieces of the scope overlay:
  1. ``UsageCapConfig.scoped_cap_models`` — the config list naming which models
     get their own per-(account, model) cap scope (``[]`` = kill switch).
  2. ``ScopeCap`` dataclass + ``AccountState.scope_caps`` — the flag+timer
     overlay state (lazily populated by later tasks; empty by default).
  3. ``scope_for(model, config)`` — the module-level derivation mapping a model
     to its scope string (the model itself) or ``None`` (the general scope).

Byte-equivalence (PRD decision 6 / boundary B6): these all default to inert, so
every existing UsageGate suite stays green unmodified. This file touches no
existing test file. ``ScopeCap`` and ``scope_for`` are deliberately excluded
from ``usage_gate.__all__`` (mirroring the AccountLease precedent), so they are
imported here via explicit ``from shared.usage_gate import ...``.
"""

from __future__ import annotations

from datetime import UTC, datetime

import pydantic
import pytest

from shared.config_models import UsageCapConfig
from shared.usage_gate import AccountState, ScopeCap


class TestScopedCapModelsConfig:
    """The additive ``UsageCapConfig.scoped_cap_models`` list field."""

    def test_default_is_claude_fable_5(self):
        assert UsageCapConfig().scoped_cap_models == ['claude-fable-5']

    def test_instances_get_independent_lists(self):
        # Guards the default_factory choice against a shared mutable default:
        # mutating one instance's list must not leak into a fresh instance.
        a = UsageCapConfig()
        a.scoped_cap_models.append('x')
        assert 'x' not in UsageCapConfig().scoped_cap_models

    def test_kill_switch_empty_list(self):
        assert UsageCapConfig(scoped_cap_models=[]).scoped_cap_models == []

    def test_custom_list_round_trips(self):
        assert UsageCapConfig(scoped_cap_models=['opus']).scoped_cap_models == ['opus']

    def test_rejects_non_list(self):
        # An int is unambiguously not a list[str] — pydantic must reject it.
        with pytest.raises(pydantic.ValidationError):
            UsageCapConfig(scoped_cap_models=123)


class TestScopeCapState:
    """The ``ScopeCap`` overlay dataclass and ``AccountState.scope_caps`` field."""

    def test_scope_cap_defaults(self):
        sc = ScopeCap()
        assert sc.capped is False
        assert sc.resets_at is None
        assert sc.near_cap is False
        assert sc.capped_at is None

    def test_scope_cap_explicit_round_trips(self):
        dt = datetime(2026, 1, 1, tzinfo=UTC)
        sc = ScopeCap(capped=True, resets_at=dt, near_cap=True, capped_at=dt)
        assert sc.capped is True
        assert sc.resets_at == dt
        assert sc.near_cap is True
        assert sc.capped_at == dt

    def test_account_state_scope_caps_default_empty(self):
        assert AccountState(name='a', token=None).scope_caps == {}

    def test_account_state_scope_caps_independent(self):
        # Guards the default_factory=dict choice against a shared mutable default:
        # populating one account's overlay must not leak into another's.
        a = AccountState(name='a', token=None)
        b = AccountState(name='b', token=None)
        a.scope_caps['claude-fable-5'] = ScopeCap(capped=True)
        assert b.scope_caps == {}
