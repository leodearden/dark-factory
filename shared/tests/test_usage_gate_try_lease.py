"""``UsageGate.try_lease()`` — the synchronous, non-blocking selection knob
(task 5488).

WHY THIS EXISTS. ``before_invoke`` is ``async`` and, when nothing is
admissible, does not return: it awaits ``_open`` / a per-scope waiter and
BLOCKS, potentially until the weekly reset. The legibility trickle cannot
use that. Task 4736 rules that an all-capped night must DEFER at exit 0
within the nightly's own runtime, so ``asyncio.run(before_invoke())`` per
digest would hang the 03:00 unit for hours instead of deferring.

Every line of ``before_invoke``'s selection walk is already synchronous —
only the surrounding lock and the blocking retry are async. ``try_lease`` is
that walk, extracted, with the opposite admission policy: return ``None``
rather than wait. ONE selection implementation, two admission policies
(docs/code-quality.md heuristic 11) — which is the whole point, because the
alternative the ruling forbids is a second hand-rolled rotation in
``scripts/legibility/``.

The characterization pin at the bottom is what keeps the extraction honest:
``await before_invoke()`` and ``try_lease()`` must select the SAME account
on an identical gate, or the two policies have silently drifted apart.
"""

from __future__ import annotations

import pytest
from _usage_gate_test_helpers import make_gate

from shared.usage_gate import AccountLease


def test_try_lease_returns_the_first_admissible_account():
    """First-fit, exactly as before_invoke's own walk does it."""
    gate = make_gate(['acct-1', 'acct-2', 'acct-3'])

    lease = gate.try_lease()

    assert isinstance(lease, AccountLease)
    assert lease.name == 'acct-1'
    assert lease.token == 'fake-token-acct-1', (
        'the lease must carry the SAME account\'s token as its name — the '
        'in-lock capture AccountLease exists to guarantee'
    )


def test_try_lease_skips_a_capped_account():
    gate = make_gate(['acct-1', 'acct-2', 'acct-3'])
    gate._accounts[0].capped = True

    lease = gate.try_lease()

    assert lease is not None
    assert lease.name == 'acct-2'


def test_try_lease_skips_auth_failed_and_probe_in_flight_too():
    """The skip predicate is `capped or probe_in_flight or auth_failed` —
    all three, because all three mean "this account cannot serve a turn"."""
    gate = make_gate(['acct-1', 'acct-2', 'acct-3'])
    gate._accounts[0].auth_failed = True
    gate._accounts[1].probe_in_flight = True

    lease = gate.try_lease()

    assert lease is not None
    assert lease.name == 'acct-3'


@pytest.mark.timeout(15)
def test_try_lease_returns_none_immediately_when_everything_is_capped():
    """THE point of the extraction: no admissible account is a RETURN, not a
    wait.

    The short timeout is the assertion that matters. A regression that
    reintroduced ``before_invoke``'s awaiting arm here would not fail a value
    assertion — it would HANG, until the weekly reset in production and until
    the suite's own 300s default in CI. Failing loudly at 15s says the right
    thing instead.
    """
    gate = make_gate(['acct-1', 'acct-2'])
    gate._accounts[0].capped = True
    gate._accounts[1].auth_failed = True

    assert gate.try_lease() is None


@pytest.mark.timeout(15)
def test_try_lease_needs_no_running_event_loop():
    """Callable from plain synchronous code — which is the entire reason it
    exists, and is not implied by the tests above merely being ``def``
    rather than ``async def``: this one proves there is no event loop to
    borrow, so any hidden ``await``/``asyncio.get_running_loop()`` on the
    path fails here rather than in the 03:00 unit.

    The trickle is exactly this caller: one process, 33 sequential one-shot
    subprocess invocations, no event loop anywhere.
    """
    import asyncio

    with pytest.raises(RuntimeError):
        asyncio.get_running_loop()  # pin the premise: really no loop running

    gate = make_gate(['acct-1', 'acct-2'])
    lease = gate.try_lease()

    assert lease is not None and lease.name == 'acct-1'


def test_try_lease_claims_the_probe_slot_like_before_invoke_does():
    """A PROBING account is claimed as PROBE_IN_FLIGHT by the selector, so a
    second caller does not pile onto an account still being tested. Carried
    over from before_invoke's walk verbatim — not reimplemented."""
    gate = make_gate(['acct-1', 'acct-2'])
    gate._accounts[0].capped = True
    gate._accounts[1].probing = True

    lease = gate.try_lease()

    assert lease is not None and lease.name == 'acct-2'
    assert gate._accounts[1].probe_in_flight, (
        'selecting a PROBING account must claim its probe slot, exactly as '
        'before_invoke does — otherwise concurrent callers all pile onto an '
        'account whose headroom is still unproven'
    )
    # And the claim is what makes the account inadmissible to the NEXT call.
    assert gate.try_lease() is None


@pytest.mark.asyncio
async def test_before_invoke_still_selects_what_try_lease_selects():
    """Characterization pin on the extraction (not a new behaviour).

    ``before_invoke`` keeps its own responsibilities — the session-budget
    check, the empty-roster RuntimeError, the blocking retry loop — and
    delegates only the SELECTION. If the two ever name different accounts on
    identical gates, the single-implementation property this extraction
    exists to create has been lost.
    """
    for capped_indices in ([], [0], [0, 1]):
        gate_a = make_gate(['acct-1', 'acct-2', 'acct-3'])
        gate_b = make_gate(['acct-1', 'acct-2', 'acct-3'])
        for i in capped_indices:
            gate_a._accounts[i].capped = True
            gate_b._accounts[i].capped = True

        sync_lease = gate_a.try_lease()
        async_lease = await gate_b.before_invoke()

        assert sync_lease is not None and async_lease is not None
        assert sync_lease.name == async_lease.name, (
            f'try_lease and before_invoke disagreed with '
            f'{capped_indices} capped: {sync_lease.name} vs {async_lease.name}'
        )
        assert sync_lease.token == async_lease.token


# ---------------------------------------------------------------------------
# reverse= — the ordering preference, opt-in (task 5488 / step-7)
#
# The trickle's 33 haiku one-shots drain the roster from the END while the
# orchestrator takes first-available from the START, so the two workloads meet
# only when the pool is nearly exhausted — which is exactly when contention is
# unavoidable anyway. This is the "smallest such knob" the ruling authorizes:
# an ordering preference expressed IN the gate, rather than a parallel
# rotation bypassing it.
#
# The negative half is load-bearing: reverse defaults to False, so every
# existing caller — the orchestrator's whole fleet — keeps its b→h order.
# ---------------------------------------------------------------------------

_POOL = ['max-b', 'max-c', 'max-d', 'max-e', 'max-f', 'max-g', 'max-h']


def test_try_lease_reverse_drains_the_roster_from_the_end():
    gate = make_gate(_POOL)

    lease = gate.try_lease(reverse=True)

    assert lease is not None
    assert lease.name == 'max-h', (
        'the trickle drains h→b so it does not contend with the '
        "orchestrator's b→h first-available order"
    )


def test_try_lease_reverse_skips_capped_accounts_walking_backwards():
    gate = make_gate(_POOL)
    gate._accounts[-1].capped = True   # max-h
    gate._accounts[-2].capped = True   # max-g

    lease = gate.try_lease(reverse=True)

    assert lease is not None
    assert lease.name == 'max-f'


def test_try_lease_default_order_is_unchanged_by_the_new_knob():
    """reverse is OPT-IN. The default walk is still first-fit from the
    start — anything else would silently re-order the orchestrator's own
    account preference, which this task has no business touching."""
    gate = make_gate(_POOL)

    lease = gate.try_lease()

    assert lease is not None
    assert lease.name == 'max-b'


@pytest.mark.asyncio
async def test_before_invoke_never_reverses():
    """The complement, and the real non-regression assertion: before_invoke
    passes no `reverse`, so every existing async caller keeps b→h."""
    gate = make_gate(_POOL)

    lease = await gate.before_invoke()

    assert lease is not None and lease.name == 'max-b'
