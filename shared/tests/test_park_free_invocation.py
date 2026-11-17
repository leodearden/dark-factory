"""How long a caller parks on a pool with no admissible account (tasks 6042, 4002).

``UsageGate.before_invoke`` waits when no account is admissible:
``_wait_for_any_account_to_reopen`` for a frozen fleet, the per-scope waiter
for an exhausted model scope. Unbounded is the right shape for a long-lived
worker that can wait out a reset, and the wrong one for a nightly oneshot (the
legibility trickle and census), which must DEFER and exit rather than hang
until the weekly reset (task 4736).

A caller-side ``asyncio.wait_for`` cannot tell a frozen pool from a slow
agent, so the bound lives in the gate: ``max_wait_secs`` bounds the park and
``park=False`` is its zero allowance. ``invoke_with_cap_retry`` forwards the
time left on its ``cap_wait_sanity_secs`` deadline as that allowance, and
``park_on_frozen_pool=False`` as the zero. The default, ``park=True`` with no
``max_wait_secs``, stays byte-identical for direct gate callers.

Every await is bounded by ``asyncio.wait_for(..., 2.0)``, so a regression that
parks FAILS this suite rather than hanging it. Accounts are capped and
auth-failed through the gate's public slot API, never by writing its state.
"""

from __future__ import annotations

import asyncio
import logging
import time
from datetime import UTC, datetime, timedelta
from pathlib import Path

import pytest
from _usage_gate_test_helpers import SCOPE, make_gate

from shared.cli_invoke import AgentResult, AllAccountsCappedException, invoke_with_cap_retry
from shared.invocation_outcome import AuthFailed, CapHit, InvocationOutcome
from shared.usage_gate import PoolFrozen

_BOUND_SECS = 2.0


def _cap() -> CapHit:
    """A cap resetting well past any test's runtime, so no refresh reopens it."""
    return CapHit(resets_at=datetime.now(UTC) + timedelta(hours=5), reason='test cap')


_AUTH = AuthFailed(status=401)


async def _gate_reporting(names, *outcomes: InvocationOutcome, scope=None, **cfg):
    """A gate over *names* with each outcome reported, in order, by the slot
    each next invocation in *scope* would get: the first outcome lands on the
    first admissible account, the next on the one after it, and so on."""
    gate = make_gate(list(names), **cfg)
    for outcome in outcomes:
        async with gate.invoke_slot(scope=scope) as slot:
            slot.report(outcome)
    return gate


@pytest.fixture
async def gates():
    """Build gates through this, so each is shut down after the test (an
    auth failure starts a background re-probe)."""
    built = []

    async def _build(names, *outcomes, scope=None, **cfg):
        gate = await _gate_reporting(names, *outcomes, scope=scope, **cfg)
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


async def _raises_pool_frozen_after(awaitable) -> tuple[PoolFrozen, float]:
    """Await *awaitable* within the suite bound; return its PoolFrozen and how
    long it took to raise."""
    started = time.monotonic()
    with pytest.raises(PoolFrozen) as excinfo:
        await asyncio.wait_for(awaitable, _BOUND_SECS)
    return excinfo.value, time.monotonic() - started


async def test_a_bounded_park_on_a_frozen_fleet_raises_pool_frozen_once_spent(gates):
    gate = await gates(('a', 'b'), _cap(), _cap())

    frozen, elapsed = await _raises_pool_frozen_after(gate.before_invoke(max_wait_secs=0.2))

    assert elapsed >= 0.15, f'raised after {elapsed:.3f}s: it refused to park instead of parking'
    assert frozen.capped_account_names == ('a', 'b')


async def test_a_bounded_park_on_an_exhausted_scope_raises_with_the_scope(gates):
    gate = await gates(('a', 'b'), _cap(), _cap(), scope=SCOPE)
    assert not gate.is_paused, 'fixture must leave the fleet open, or this is the fleet park'

    frozen, elapsed = await _raises_pool_frozen_after(
        gate.before_invoke(scope=SCOPE, max_wait_secs=0.2),
    )

    assert elapsed >= 0.15, f'raised after {elapsed:.3f}s: it refused to park instead of parking'
    assert frozen.scope == SCOPE


async def test_a_zero_allowance_refuses_to_park(gates):
    gate = await gates(('a', 'b'), _cap(), _cap())

    await _raises_pool_frozen_after(gate.before_invoke(max_wait_secs=0.0))


async def test_invoke_slot_forwards_the_allowance(gates):
    gate = await gates(('a', 'b', 'c'), _cap(), _AUTH, _cap())

    async def _enter_slot():
        async with gate.invoke_slot(max_wait_secs=0.2):
            pytest.fail('a frozen pool must not yield a slot')

    refused, _ = await _raises_pool_frozen_after(_enter_slot())
    after, _ = await _raises_pool_frozen_after(gate.before_invoke(park=False))

    for frozen in (refused, after):
        assert frozen.capped_account_names == ('a', 'c')
        assert frozen.auth_failed_account_names == ('b',)


async def test_a_bounded_park_still_serves_an_account_that_reopens_in_time(gates):
    """The account reopens through the gate's real reset path: its resume
    probe (stubbed to succeed by ``make_gate``) fires at its reset time."""
    soon = CapHit(resets_at=datetime.now(UTC) + timedelta(seconds=0.5), reason='test cap')
    gate = await gates(('a', 'b'), _cap(), soon, wait_for_reset=True)
    assert gate.is_paused, 'fixture must start frozen, or nothing parks'
    allowance = 2.0

    started = time.monotonic()
    lease = await asyncio.wait_for(
        gate.before_invoke(max_wait_secs=allowance), _BOUND_SECS + allowance,
    )
    elapsed = time.monotonic() - started

    assert lease is not None and lease.name == 'b', lease
    assert elapsed < allowance, f'served after {elapsed:.3f}s, not within the {allowance}s bound'


class _RecordingInvoke:
    """An ``invoke_fn`` that records each dispatch and succeeds."""

    def __init__(self) -> None:
        self.calls: list[dict] = []

    async def __call__(self, **kwargs) -> AgentResult:
        self.calls.append(kwargs)
        return AgentResult(success=True, output='done')


def _invoke_parking(gate, invoke_fn, *, cap_wait_sanity_secs):
    return invoke_with_cap_retry(
        gate, 'lbl',
        cap_wait_sanity_secs=cap_wait_sanity_secs,
        invoke_fn=invoke_fn,
        prompt='p', system_prompt='s', cwd=Path('/tmp'),
    )


async def test_cap_wait_sanity_secs_bounds_the_park_of_a_pool_frozen_at_call_start(
    gates, caplog,
):
    gate = await gates(('a', 'b'), _cap(), _AUTH)
    invoke_fn = _RecordingInvoke()

    with (
        caplog.at_level(logging.ERROR, logger='shared.cli_invoke'),
        pytest.raises(AllAccountsCappedException) as excinfo,
    ):
        await asyncio.wait_for(
            _invoke_parking(gate, invoke_fn, cap_wait_sanity_secs=0.2), _BOUND_SECS,
        )

    assert excinfo.value.label == 'lbl'
    frozen = excinfo.value.__cause__
    assert isinstance(frozen, PoolFrozen), frozen
    assert frozen.capped_account_names == ('a',)
    assert frozen.auth_failed_account_names == ('b',)
    assert not invoke_fn.calls, 'a frozen pool must never be dispatched on'
    assert any(
        rec.levelno == logging.ERROR and rec.name == 'shared.cli_invoke'
        and 'lbl' in rec.getMessage()
        for rec in caplog.records
    ), [(rec.name, rec.getMessage()) for rec in caplog.records]


async def test_a_park_spent_on_an_all_auth_failed_pool_does_not_report_a_cap(gates):
    """Rejected credentials never reopen at a reset, so the exception a parking
    caller gets carries the gate's snapshot and does not call the freeze a cap."""
    gate = await gates(('a', 'b'), _AUTH, _AUTH)
    invoke_fn = _RecordingInvoke()

    with pytest.raises(AllAccountsCappedException) as excinfo:
        await asyncio.wait_for(
            _invoke_parking(gate, invoke_fn, cap_wait_sanity_secs=0.2), _BOUND_SECS,
        )

    frozen = excinfo.value.pool_frozen
    assert isinstance(frozen, PoolFrozen), frozen
    assert frozen is excinfo.value.__cause__, (frozen, excinfo.value.__cause__)
    assert frozen.capped_account_names == ()
    assert frozen.auth_failed_account_names == ('a', 'b')
    message = str(excinfo.value)
    assert 'all accounts capped' not in message, message
    assert 'auth-failed' in message and 'a, b' in message, message


async def test_a_cap_spent_in_the_retry_loop_carries_no_pool_snapshot():
    """Only a park the gate refused to end has a ``PoolFrozen`` to report."""
    exc = AllAccountsCappedException(retries=3, elapsed_secs=1.0, label='lbl')

    assert exc.pool_frozen is None
    assert 'all accounts capped' in str(exc)


async def test_without_a_sanity_bound_the_park_is_still_unbounded(gates):
    gate = await gates(('a', 'b'), _cap(), _cap())
    invoke_fn = _RecordingInvoke()

    with pytest.raises(TimeoutError):
        await asyncio.wait_for(
            _invoke_parking(gate, invoke_fn, cap_wait_sanity_secs=None), 0.3,
        )

    assert not invoke_fn.calls
