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
parks FAILS this suite rather than hanging it. Accounts are capped and
auth-failed through the gate's public slot API, never by writing its state.
"""

from __future__ import annotations

import asyncio
from datetime import UTC, datetime, timedelta
from pathlib import Path

import pytest
from _usage_gate_test_helpers import SCOPE, make_gate

from shared.cli_invoke import AgentResult, invoke_with_cap_retry
from shared.invocation_outcome import AuthFailed, CapHit, InvocationOutcome
from shared.usage_gate import PoolFrozen

_BOUND_SECS = 2.0


def _cap() -> CapHit:
    """A cap resetting well past any test's runtime, so no refresh reopens it."""
    return CapHit(resets_at=datetime.now(UTC) + timedelta(hours=5), reason='test cap')


_AUTH = AuthFailed(status=401)


async def _gate_reporting(names, *outcomes: InvocationOutcome, scope=None):
    """A gate over *names* with each outcome reported, in order, by the slot
    each next invocation in *scope* would get: the first outcome lands on the
    first admissible account, the next on the one after it, and so on."""
    gate = make_gate(list(names))
    for outcome in outcomes:
        async with gate.invoke_slot(scope=scope) as slot:
            slot.report(outcome)
    return gate


@pytest.fixture
async def gates():
    """Build gates through this, so each is shut down after the test (an
    auth failure starts a background re-probe)."""
    built = []

    async def _build(names, *outcomes, scope=None):
        gate = await _gate_reporting(names, *outcomes, scope=scope)
        built.append(gate)
        return gate

    yield _build
    for gate in built:
        await gate.shutdown()


async def test_before_invoke_without_park_raises_when_every_account_is_capped(gates):
    gate = await gates(('a', 'b'), _cap(), _cap())

    with pytest.raises(PoolFrozen):
        await asyncio.wait_for(gate.before_invoke(park=False), _BOUND_SECS)


async def test_before_invoke_without_park_raises_when_every_account_is_auth_failed(gates):
    gate = await gates(('a', 'b'), _AUTH, _AUTH)

    with pytest.raises(PoolFrozen):
        await asyncio.wait_for(gate.before_invoke(park=False), _BOUND_SECS)


async def test_an_exhausted_scope_raises_rather_than_parking_on_the_scope_waiter(gates):
    """The fleet is NOT frozen here — a scoped cap leaves every account
    generally AVAILABLE — so the park that must be refused is the per-scope
    waiter's, not ``_open``'s."""
    gate = await gates(('a', 'b'), _cap(), _cap(), scope=SCOPE)
    assert not gate.is_paused, 'fixture must leave the fleet open, or this is case (a)'

    with pytest.raises(PoolFrozen) as excinfo:
        await asyncio.wait_for(gate.before_invoke(scope=SCOPE, park=False), _BOUND_SECS)

    assert excinfo.value.scope == SCOPE


async def test_the_default_still_parks_on_a_frozen_pool(gates):
    """Byte-compat guard: omitting ``park`` keeps the fleet's patient wait."""
    gate = await gates(('a', 'b'), _cap(), _cap())

    with pytest.raises(TimeoutError):
        await asyncio.wait_for(gate.before_invoke(), 0.3)


async def test_invoke_slot_without_park_propagates_and_claims_no_account(gates):
    gate = await gates(('a', 'b', 'c'), _cap(), _AUTH, _cap())

    async def _enter_slot():
        async with gate.invoke_slot(park=False):
            pytest.fail('a frozen pool must not yield a slot')

    with pytest.raises(PoolFrozen) as refused:
        await asyncio.wait_for(_enter_slot(), _BOUND_SECS)
    with pytest.raises(PoolFrozen) as after:
        await asyncio.wait_for(gate.before_invoke(park=False), _BOUND_SECS)

    for frozen in (refused.value, after.value):
        assert frozen.capped_account_names == ('a', 'c')
        assert frozen.auth_failed_account_names == ('b',)


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


async def test_pool_frozen_names_the_count_and_which_accounts_are_capped_or_auth_failed(gates):
    gate = await gates(('max-x', 'max-y', 'max-z'), _cap(), _AUTH, _cap())

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
