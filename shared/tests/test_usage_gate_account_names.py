"""``UsageGate.account_names``: the resolved roster, in failover order (task 6042).

Published so a caller can name the pool's accounts without reaching into the
gate's account states. It lists every account the gate RESOLVED, whatever its
phase: capping or rejecting an account does not remove it from the pool.
"""

from __future__ import annotations

import os
from datetime import UTC, datetime, timedelta
from unittest.mock import patch

from _usage_gate_test_helpers import make_gate

from shared.config_models import AccountConfig, UsageCapConfig
from shared.invocation_outcome import AuthFailed, CapHit
from shared.usage_gate import UsageGate


def test_lists_the_resolved_accounts_in_roster_order():
    gate = make_gate(['max-c', 'max-a', 'max-b'])

    assert gate.account_names == ('max-c', 'max-a', 'max-b')


def test_omits_an_account_whose_token_did_not_resolve():
    config = UsageCapConfig(
        accounts=[
            AccountConfig(name='resolved', oauth_token_env='TEST_TOKEN_RESOLVED'),
            AccountConfig(name='unresolved', oauth_token_env='TEST_TOKEN_UNRESOLVED'),
        ],
        fallback_to_default_credential=False,
    )
    with patch.dict(os.environ, {'TEST_TOKEN_RESOLVED': 'tok'}, clear=False):
        os.environ.pop('TEST_TOKEN_UNRESOLVED', None)
        gate = UsageGate(config)

    assert gate.account_names == ('resolved',)


async def test_keeps_capped_and_auth_failed_accounts():
    gate = make_gate(['a', 'b', 'c'])
    try:
        for outcome in (
            CapHit(resets_at=datetime.now(UTC) + timedelta(hours=5), reason='test cap'),
            AuthFailed(status=403),
        ):
            async with gate.invoke_slot() as slot:
                slot.report(outcome)
    finally:
        await gate.shutdown()

    assert gate.account_names == ('a', 'b', 'c')
