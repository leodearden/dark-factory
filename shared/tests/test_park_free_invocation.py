"""A park-free caller must be TOLD the pool is frozen, never parked (task 6042).

``UsageGate.before_invoke`` ends in an unbounded wait when no account is
admissible: ``_wait_for_any_account_to_reopen`` for a frozen fleet, the
per-scope waiter for an exhausted model scope. That is the right shape for
the orchestrator's long-lived workers, which can afford to wait out a reset,
and the wrong one for a nightly oneshot (the legibility trickle and census),
which must DEFER and exit rather than hang until the weekly reset (task 4736).

``cap_wait_sanity_secs`` cannot bound that wait (it is consulted only after a
RETURNED cap), and a caller-side ``asyncio.wait_for`` cannot tell a frozen
pool from a slow agent. So the bound is a ``park`` keyword on the gate itself,
surfaced through ``invoke_slot`` and ``invoke_with_cap_retry``'s
``park_on_frozen_pool``. ``park=True`` stays the default and stays
byte-identical for the fleet.

Every await is bounded by ``asyncio.wait_for(..., 2.0)``, so a regression that
parks FAILS this suite rather than hanging it.
"""

from __future__ import annotations

import asyncio
from datetime import UTC, datetime, timedelta
from pathlib import Path

import pytest
from _usage_gate_test_helpers import SCOPE, make_gate, set_scope_cap

from shared.cli_invoke import AgentResult, invoke_with_cap_retry
from shared.usage_gate import AccountPhase, PoolFrozen

_BOUND_SECS = 2.0


def _cap(acct) -> None:
    """Cap *acct* until well past any test's runtime, so no refresh reopens it."""
    acct.capped = True
    acct.resets_at = datetime.now(UTC) + timedelta(hours=5)


def _all_capped_gate(names=('a', 'b')):
    gate = make_gate(list(names))
    for acct in gate._accounts:
        _cap(acct)
    return gate


def _all_auth_failed_gate(names=('a', 'b')):
    gate = make_gate(list(names))
    for acct in gate._accounts:
        acct.auth_failed = True
    return gate


async def test_before_invoke_without_park_raises_when_every_account_is_capped():
    gate = _all_capped_gate()

    with pytest.raises(PoolFrozen):
        await asyncio.wait_for(gate.before_invoke(park=False), _BOUND_SECS)


async def test_before_invoke_without_park_raises_when_every_account_is_auth_failed():
    gate = _all_auth_failed_gate()

    with pytest.raises(PoolFrozen):
        await asyncio.wait_for(gate.before_invoke(park=False), _BOUND_SECS)


async def test_an_exhausted_scope_raises_rather_than_parking_on_the_scope_waiter():
    """The fleet is NOT frozen here — every account is generally AVAILABLE —
    so the park that must be refused is the per-scope waiter's, not ``_open``'s."""
    gate = make_gate(['a', 'b'])
    for acct in gate._accounts:
        set_scope_cap(acct, resets_at=datetime.now(UTC) + timedelta(hours=5))
    assert not gate.is_paused, 'fixture must leave the fleet open, or this is case (a)'

    with pytest.raises(PoolFrozen):
        await asyncio.wait_for(gate.before_invoke(scope=SCOPE, park=False), _BOUND_SECS)


async def test_the_default_still_parks_on_a_frozen_pool():
    """Byte-compat guard: omitting ``park`` keeps the fleet's patient wait."""
    gate = _all_capped_gate()

    with pytest.raises(TimeoutError):
        await asyncio.wait_for(gate.before_invoke(), 0.3)


async def test_invoke_slot_without_park_propagates_and_claims_no_account():
    gate = _all_capped_gate(('a', 'b', 'c'))
    gate._accounts[1].capped = False
    gate._accounts[1].auth_failed = True

    async def _enter_slot():
        async with gate.invoke_slot(park=False):
            pytest.fail('a frozen pool must not yield a slot')

    with pytest.raises(PoolFrozen):
        await asyncio.wait_for(_enter_slot(), _BOUND_SECS)

    assert [a.phase for a in gate._accounts] == [
        AccountPhase.CAPPED, AccountPhase.AUTH_FAILED, AccountPhase.CAPPED,
    ]


async def test_invoke_with_cap_retry_fails_over_both_auth_rejections_then_raises():
    """A 401 on one account and a 403 on the other: each fails over without
    counting a cap, and once both are AUTH_FAILED the park-free caller gets
    ``PoolFrozen`` rather than a park that would last until an operator acts."""
    gate = make_gate(['a', 'b'])
    status_by_token = {'fake-token-a': 401, 'fake-token-b': 403}
    calls = []

    async def _fake_invoke(**kwargs):
        calls.append(kwargs['oauth_token'])
        return AgentResult(
            success=False,
            output='Failed to authenticate.',
            api_error_status=status_by_token[kwargs['oauth_token']],
        )

    try:
        with pytest.raises(PoolFrozen):
            await asyncio.wait_for(
                invoke_with_cap_retry(
                    gate, 'lbl',
                    park_on_frozen_pool=False,
                    invoke_fn=_fake_invoke,
                    prompt='p', system_prompt='s', cwd=Path('/tmp'),
                ),
                _BOUND_SECS,
            )
    finally:
        await gate.shutdown()

    assert calls == ['fake-token-a', 'fake-token-b']
    assert gate.auth_failed_account_names == ('a', 'b')


async def test_pool_frozen_names_the_count_and_which_accounts_are_capped_or_auth_failed():
    gate = _all_capped_gate(('max-x', 'max-y', 'max-z'))
    gate._accounts[1].capped = False
    gate._accounts[1].auth_failed = True

    with pytest.raises(PoolFrozen) as excinfo:
        await asyncio.wait_for(gate.before_invoke(park=False), _BOUND_SECS)

    frozen = excinfo.value
    assert frozen.account_count == 3
    assert frozen.capped_account_names == ('max-x', 'max-z')
    assert frozen.auth_failed_account_names == ('max-y',)
    message = str(frozen)
    assert '3' in message
    for name in ('max-x', 'max-y', 'max-z'):
        assert name in message
    assert 'capped' in message and 'auth-failed' in message
