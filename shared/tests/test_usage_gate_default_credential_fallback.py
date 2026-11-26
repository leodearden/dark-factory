"""The ``~/.claude`` fallback is an explicit opt-in per gate (tasks 6042, 8161).

When no roster account resolves a token, ``UsageGate._init_accounts`` adopts
``~/.claude/.credentials.json`` as a single account named ``default`` only if
``UsageCapConfig.fallback_to_default_credential`` is True: that is the
operator's own interactive login, which fleet tooling must never ride
(ruling 2026-11-01). Left unset, the pool stays EMPTY.
"""

from __future__ import annotations

import json

import pytest

from shared import usage_gate as usage_gate_mod
from shared.config_models import AccountConfig, UsageCapConfig
from shared.usage_gate import UsageGate

_SENTINEL_TOKEN = 'sk-ant-oat01-SENTINEL-operator-login'


@pytest.fixture
def sentinel_operator_login(tmp_path, monkeypatch):
    """A valid-looking ~/.claude credential the gate must never adopt silently."""
    path = tmp_path / '.credentials.json'
    path.write_text(json.dumps({'claudeAiOauth': {'accessToken': _SENTINEL_TOKEN}}))
    monkeypatch.setattr(usage_gate_mod, 'CREDENTIALS_PATH', path)
    return path


@pytest.fixture
def unresolvable_roster(monkeypatch):
    names = ('max-x', 'max-y')
    for name in names:
        monkeypatch.delenv(f'TEST_UNSET_TOKEN_{name.upper().replace("-", "_")}', raising=False)
    return [
        AccountConfig(name=name, oauth_token_env=f'TEST_UNSET_TOKEN_{name.upper().replace("-", "_")}')
        for name in names
    ]


def test_a_gate_with_the_fallback_unset_resolves_no_account(
    sentinel_operator_login, unresolvable_roster,
):
    gate = UsageGate(UsageCapConfig(accounts=unresolvable_roster))

    assert gate.account_names == ()


def test_a_gate_with_the_fallback_disabled_resolves_no_account(
    sentinel_operator_login, unresolvable_roster,
):
    gate = UsageGate(UsageCapConfig(
        accounts=unresolvable_roster, fallback_to_default_credential=False,
    ))

    assert gate.account_count == 0


def test_an_explicit_opt_in_adopts_the_operator_login_as_one_default_account(
    sentinel_operator_login, unresolvable_roster,
):
    gate = UsageGate(UsageCapConfig(
        accounts=unresolvable_roster, fallback_to_default_credential=True,
    ))

    assert gate.account_count == 1
    assert gate.active_account_name == 'default'
