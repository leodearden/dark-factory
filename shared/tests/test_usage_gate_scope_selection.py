"""Scope-aware selection, waiting, optimistic uncap + read APIs (PRD task γ).

Task 2857 (γ) consumes the scope substrate (task 2855 / α) and scope-aware
attribution (task 2856 / β) to add, all in ``shared/src/shared/usage_gate.py``:

* **selection** — ``before_invoke(scope=m)`` additionally skips accounts whose
  ``scope_caps[m]`` is capped with a future uncap-deadline (S2), while
  account-level CAPPED/AUTH_FAILED still dominates for every scope (S4) and
  ``scope=None`` stays byte-identical to today (S1);
* **optimistic uncap** — a selection-time sweep (``_refresh_scope_capped``)
  clears expired scope caps, using ``resets_at`` when known else
  ``capped_at + max_probe_interval_secs`` as the deadline (S6);
* **no-freeze waiting** — when every account is scope-capped for m but some are
  generally available, a per-scope ``asyncio.Event`` waiter sleeps toward the
  soonest scope reset WITHOUT ever clearing ``_open`` / setting a fleet pause
  (S3 — scope exhaustion never freezes the fleet);
* **read APIs** — ``scope_capacity_snapshot()`` (advisory per-model headroom,
  matching the invoke-time predicate, S8) and ``scope_status()`` (per-(account
  × scope) serialized state for a future digest/dashboard).

Isolated file with local minimal helpers (mirroring
``test_usage_gate_scope_attribution.py`` / ``test_cap_retry.py``) so it stays
independent of the big suites. Every existing scope=None suite stays green
unmodified (S1 / B6 byte-equivalence).
"""

from __future__ import annotations

import asyncio
import contextlib
import json
import os
from datetime import UTC, datetime, timedelta
from unittest.mock import AsyncMock, patch

import pytest

from shared.config_models import AccountConfig, UsageCapConfig
from shared.usage_gate import AccountPhase, ScopeCap, UsageGate

SCOPE = 'claude-fable-5'


# ---------------------------------------------------------------------------
# Local helpers (mirroring test_usage_gate_scope_attribution.py)
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


def set_scope_cap(
    acct,
    scope: str = SCOPE,
    *,
    capped: bool = True,
    resets_at: datetime | None = None,
    capped_at: datetime | None = None,
    near_cap: bool = False,
) -> ScopeCap:
    """Directly install a ScopeCap overlay on *acct* (bypassing the β handler)."""
    sc = ScopeCap(
        capped=capped, resets_at=resets_at, near_cap=near_cap, capped_at=capped_at,
    )
    acct.scope_caps[scope] = sc
    return sc


# ===========================================================================
# Step 1 — scope-uncap sweep + deadline mechanics (S6 core)
# ===========================================================================


class TestScopeUncapSweep:
    """``_refresh_scope_capped`` optimistically clears expired scope caps and
    ``_soonest_scope_reset`` reports the soonest uncap deadline. The deadline is
    ``resets_at`` when known, else ``capped_at + max_probe_interval_secs`` (S6)."""

    def test_past_resets_at_is_swept(self):
        gate = make_gate(['a'])
        acct = gate._accounts[0]
        now = datetime.now(UTC)
        set_scope_cap(acct, resets_at=now - timedelta(seconds=10), capped_at=now)

        assert gate._refresh_scope_capped(SCOPE) is True
        assert acct.scope_caps[SCOPE].capped is False

    def test_future_resets_at_stays_capped(self):
        gate = make_gate(['a'])
        acct = gate._accounts[0]
        now = datetime.now(UTC)
        set_scope_cap(acct, resets_at=now + timedelta(hours=1), capped_at=now)

        assert gate._refresh_scope_capped(SCOPE) is False
        assert acct.scope_caps[SCOPE].capped is True

    def test_unknown_resets_at_swept_after_max_probe_interval(self):
        """resets_at=None → deadline is capped_at + max_probe_interval_secs;
        a cap whose capped_at is older than that ceiling IS swept."""
        gate = make_gate(['a'])
        acct = gate._accounts[0]
        now = datetime.now(UTC)
        old = now - timedelta(seconds=gate._config.max_probe_interval_secs + 60)
        set_scope_cap(acct, resets_at=None, capped_at=old)

        assert gate._refresh_scope_capped(SCOPE) is True
        assert acct.scope_caps[SCOPE].capped is False

    def test_unknown_resets_at_fresh_cap_not_swept(self):
        """A just-set unknown-reset cap is within the deadline → NOT swept."""
        gate = make_gate(['a'])
        acct = gate._accounts[0]
        set_scope_cap(acct, resets_at=None, capped_at=datetime.now(UTC))

        assert gate._refresh_scope_capped(SCOPE) is False
        assert acct.scope_caps[SCOPE].capped is True

    def test_uncapped_scope_cap_ignored(self):
        """An already-uncapped scope cap is not touched and does not count."""
        gate = make_gate(['a'])
        acct = gate._accounts[0]
        now = datetime.now(UTC)
        set_scope_cap(acct, capped=False, resets_at=now - timedelta(hours=1), capped_at=now)

        assert gate._refresh_scope_capped(SCOPE) is False

    def test_refresh_scope_capped_sets_waiter_event(self):
        gate = make_gate(['a'])
        acct = gate._accounts[0]
        now = datetime.now(UTC)
        set_scope_cap(acct, resets_at=now - timedelta(seconds=10), capped_at=now)
        evt = gate._scope_waiter(SCOPE)
        evt.clear()

        assert gate._refresh_scope_capped(SCOPE) is True
        assert evt.is_set() is True

    def test_soonest_scope_reset_returns_min_deadline(self):
        gate = make_gate(['a', 'b'])
        a, b = gate._accounts
        now = datetime.now(UTC)
        soon = now + timedelta(minutes=10)
        later = now + timedelta(hours=2)
        set_scope_cap(a, resets_at=later, capped_at=now)
        set_scope_cap(b, resets_at=soon, capped_at=now)

        assert gate._soonest_scope_reset(SCOPE) == soon

    def test_soonest_scope_reset_none_when_none_capped(self):
        gate = make_gate(['a'])
        assert gate._soonest_scope_reset(SCOPE) is None

    def test_soonest_scope_reset_unknown_resets_uses_deadline(self):
        gate = make_gate(['a'])
        acct = gate._accounts[0]
        now = datetime.now(UTC)
        set_scope_cap(acct, resets_at=None, capped_at=now)
        expected = now + timedelta(seconds=gate._config.max_probe_interval_secs)

        assert gate._soonest_scope_reset(SCOPE) == expected


# ===========================================================================
# Step 3 — scope-aware selection (S1/S2/S4 — B1 general / B4 dominates / skip)
# ===========================================================================


class TestScopeAwareSelection:
    """``before_invoke(scope=m)`` skips accounts scope-capped for m (S2) while
    ``scope=None`` stays byte-identical (S1 / B1) and account-level CAPPED
    dominates every scope (S4 / B4). Headroom is always present, so no wait."""

    async def test_s1_b1_general_ignores_fable_scope_cap(self):
        # One account, fable-scope-capped (future deadline) but account AVAILABLE.
        gate = make_gate(['a'])
        acct = gate._accounts[0]
        now = datetime.now(UTC)
        set_scope_cap(acct, resets_at=now + timedelta(hours=1), capped_at=now)

        lease = await gate.before_invoke()  # scope=None
        assert lease is not None
        assert lease.name == 'a'  # a fable cap leaves general open (B1)

    async def test_s2_scoped_selection_skips_fable_capped(self):
        gate = make_gate(['a', 'b'])
        a, b = gate._accounts
        now = datetime.now(UTC)
        set_scope_cap(a, resets_at=now + timedelta(hours=1), capped_at=now)

        lease = await gate.before_invoke(scope=SCOPE)
        assert lease is not None
        assert lease.name == 'b'  # A skipped for the fable scope (S2)

    async def test_s4_b4_account_cap_dominates_every_scope(self):
        gate = make_gate(['a', 'b'])
        a, b = gate._accounts
        # Account-level CAPPED (general handler; future reset so no auto-uncap).
        gate._handle_cap_detected(
            'reason', datetime.now(UTC) + timedelta(hours=1), a.token, scope=None,
        )
        assert a.phase == AccountPhase.CAPPED

        scoped = await gate.before_invoke(scope=SCOPE)
        general = await gate.before_invoke()
        assert scoped.name == 'b'
        assert general.name == 'b'  # account cap dominates for every scope (S4/B4)

    async def test_invoke_slot_threads_scope_into_selection(self):
        gate = make_gate(['a', 'b'])
        a, b = gate._accounts
        now = datetime.now(UTC)
        set_scope_cap(a, resets_at=now + timedelta(hours=1), capped_at=now)

        async with gate.invoke_slot(scope=SCOPE) as slot:
            # scope threaded into selection → leases the fable-headroom account.
            assert slot.account_name == 'b'


# ===========================================================================
# Step 5 — scoped failover cost event carries scope (B2 event half); the
# general failover path stays byte-identical (S1)
# ===========================================================================


def _failover_calls(mock_fire):
    """The ('failover') calls recorded by a patched _fire_cost_event mock."""
    return [c for c in mock_fire.call_args_list if c.args[1] == 'failover']


class TestScopedFailoverEvent:
    """A scoped account switch fires a ``failover`` cost event whose JSON
    details include ``scope`` (B2), while the general (scope=None) failover
    path is byte-identical and the two trackers are independent (S1)."""

    async def test_scoped_failover_event_carries_scope(self):
        gate = make_gate(['a', 'b'], cost_store=make_mock_cost_store())
        a, b = gate._accounts
        now = datetime.now(UTC)

        with patch.object(gate, '_fire_cost_event') as mock_fire:
            # 1) First scoped selection lands on A (establishes per-scope last).
            first = await gate.before_invoke(scope=SCOPE)
            assert first.name == 'a'
            # 2) Scope-cap A → the next scoped selection fails over to B.
            set_scope_cap(a, resets_at=now + timedelta(hours=1), capped_at=now)
            second = await gate.before_invoke(scope=SCOPE)
            assert second.name == 'b'

        failovers = _failover_calls(mock_fire)
        assert len(failovers) == 1
        name, _event, details_json = failovers[0].args
        assert name == 'b'
        assert json.loads(details_json) == {'from': 'a', 'to': 'b', 'scope': SCOPE}

    async def test_general_failover_event_has_no_scope_key(self):
        gate = make_gate(['a', 'b'], cost_store=make_mock_cost_store())
        a, b = gate._accounts
        now = datetime.now(UTC)

        with patch.object(gate, '_fire_cost_event') as mock_fire:
            first = await gate.before_invoke()  # scope=None → A
            assert first.name == 'a'
            # Account-level cap A (general) → next general selection → B.
            gate._handle_cap_detected('r', now + timedelta(hours=1), a.token, scope=None)
            second = await gate.before_invoke()
            assert second.name == 'b'

        failovers = _failover_calls(mock_fire)
        assert len(failovers) == 1
        _name, _event, details_json = failovers[0].args
        details = json.loads(details_json)
        assert details == {'from': 'a', 'to': 'b'}
        assert 'scope' not in details

    async def test_scoped_selection_does_not_perturb_general_tracker(self):
        gate = make_gate(['a', 'b'], cost_store=make_mock_cost_store())
        a, b = gate._accounts
        now = datetime.now(UTC)

        # Establish general last-account = A via a general selection.
        g1 = await gate.before_invoke()
        assert g1.name == 'a'
        assert gate._last_account_name == 'a'

        # An interleaved SCOPED selection that lands on B must NOT move the
        # general tracker off 'a' (independent trackers, S1).
        set_scope_cap(a, resets_at=now + timedelta(hours=1), capped_at=now)
        s1 = await gate.before_invoke(scope=SCOPE)
        assert s1.name == 'b'
        assert gate._last_account_name == 'a'

        # A following general selection still lands on A with NO failover event.
        with patch.object(gate, '_fire_cost_event') as mock_fire:
            g2 = await gate.before_invoke()
            assert g2.name == 'a'
        assert _failover_calls(mock_fire) == []


# ===========================================================================
# Step 7 — S3 no-freeze wait (B3) + optimistic-uncap-returns-in-one-call (B5)
# ===========================================================================


class TestScopeWaitNoFreeze:
    """When every account is scope-capped for m but generally available, the
    scoped waiter parks WITHOUT freezing the fleet (S3/B3); a past-deadline
    scope cap is optimistically uncapped so the scoped call returns in one
    invocation (B5-return). All waits are bounded so RED surfaces as a
    failure/timeout, never an infinite hang."""

    async def test_s3_b3_scope_exhaustion_never_freezes_fleet(self):
        gate = make_gate(['a', 'b'])
        a, b = gate._accounts
        now = datetime.now(UTC)
        far = now + timedelta(hours=6)
        # Both fable-scope-capped (far-future deadline) but generally AVAILABLE.
        set_scope_cap(a, resets_at=far, capped_at=now)
        set_scope_cap(b, resets_at=far, capped_at=now)
        assert a.phase == AccountPhase.AVAILABLE
        assert b.phase == AccountPhase.AVAILABLE

        scoped_task = asyncio.create_task(gate.before_invoke(scope=SCOPE))
        try:
            # Let the scoped selection reach its (scope-local) wait.
            await asyncio.sleep(0.05)

            # Scope exhaustion must NOT freeze the fleet.
            assert gate._open.is_set() is True
            assert gate.paused_reason == ''
            # The scoped caller is parked (no fable headroom), not returned.
            assert scoped_task.done() is False

            # A concurrent GENERAL caller is served immediately (not delayed by
            # the scoped wait).
            general = await asyncio.wait_for(gate.before_invoke(), timeout=1.0)
            assert general is not None
            assert general.name in ('a', 'b')
        finally:
            scoped_task.cancel()
            with contextlib.suppress(asyncio.CancelledError):
                await scoped_task

    async def test_b5_optimistic_uncap_returns_in_one_call(self):
        gate = make_gate(['a', 'b'])
        a, b = gate._accounts
        now = datetime.now(UTC)
        # A's fable cap deadline is in the PAST → optimistically uncapped.
        set_scope_cap(a, resets_at=now - timedelta(seconds=5),
                      capped_at=now - timedelta(hours=1))
        # B's fable cap is still in the future.
        set_scope_cap(b, resets_at=now + timedelta(hours=6), capped_at=now)

        lease = await asyncio.wait_for(
            gate.before_invoke(scope=SCOPE), timeout=1.0
        )
        assert lease is not None
        assert lease.name == 'a'  # returned in one call (no wait, no timeout)


# ===========================================================================
# Step 9 — scope_capacity_snapshot() -> dict[str, bool] (advisory; S8)
# ===========================================================================


class TestScopeCapacitySnapshot:
    """``scope_capacity_snapshot()`` reports, per configured scoped model,
    whether ANY account has headroom for that scope — the same admission
    predicate ``before_invoke(scope=m)`` applies (S8). Pure read."""

    def test_one_key_per_configured_scoped_model(self):
        gate = make_gate(['a'])
        # default scoped_cap_models == ['claude-fable-5']
        assert set(gate.scope_capacity_snapshot()) == {SCOPE}

    def test_true_when_any_account_has_scope_headroom(self):
        gate = make_gate(['a', 'b'])
        a, b = gate._accounts
        now = datetime.now(UTC)
        set_scope_cap(a, resets_at=now + timedelta(hours=1), capped_at=now)
        # B has fable headroom → snapshot True.
        assert gate.scope_capacity_snapshot()[SCOPE] is True

    def test_false_when_all_lack_scope_headroom(self):
        gate = make_gate(['a', 'b'])
        a, b = gate._accounts
        now = datetime.now(UTC)
        # A account-level capped; B fable-scope-capped future → no fable headroom.
        gate._handle_cap_detected('r', now + timedelta(hours=1), a.token, scope=None)
        set_scope_cap(b, resets_at=now + timedelta(hours=1), capped_at=now)
        assert gate.scope_capacity_snapshot()[SCOPE] is False

    def test_past_deadline_scope_cap_counts_as_headroom(self):
        gate = make_gate(['a'])
        acct = gate._accounts[0]
        now = datetime.now(UTC)
        set_scope_cap(acct, resets_at=now - timedelta(seconds=5),
                      capped_at=now - timedelta(hours=1))
        # Snapshot agrees with the optimistic-uncap predicate (past deadline → headroom).
        assert gate.scope_capacity_snapshot()[SCOPE] is True

    def test_empty_scoped_cap_models_yields_empty_snapshot(self):
        gate = make_gate(['a'], scoped_cap_models=[])
        assert gate.scope_capacity_snapshot() == {}

    def test_only_account_fable_capped_future_yields_false(self):
        gate = make_gate(['a'])
        acct = gate._accounts[0]
        now = datetime.now(UTC)
        set_scope_cap(acct, resets_at=now + timedelta(hours=1), capped_at=now)
        # Generally available, but the sole account has no fable headroom.
        assert acct.phase == AccountPhase.AVAILABLE
        assert gate.scope_capacity_snapshot()[SCOPE] is False


# ===========================================================================
# Step 11 — scope_status() per-(account × scope) serialized read API
# ===========================================================================


class TestScopeStatus:
    """``scope_status()`` serializes per-(account × scope) cap state for a
    future digest/dashboard — ISO-string datetimes, None passed through, and
    only accounts with an observed scope cap included. Pure read."""

    def test_populated_scope_cap_serialized(self):
        gate = make_gate(['a'])
        acct = gate._accounts[0]
        resets = datetime(2026, 7, 20, 18, 0, tzinfo=UTC)
        capped_at = datetime(2026, 7, 20, 17, 0, tzinfo=UTC)
        set_scope_cap(acct, capped=True, resets_at=resets, near_cap=True,
                      capped_at=capped_at)

        assert gate.scope_status()['a'][SCOPE] == {
            'capped': True,
            'resets_at': resets.isoformat(),
            'near_cap': True,
            'capped_at': capped_at.isoformat(),
        }

    def test_none_datetimes_passed_through(self):
        gate = make_gate(['a'])
        acct = gate._accounts[0]
        set_scope_cap(acct, capped=True, resets_at=None, near_cap=False,
                      capped_at=None)

        assert gate.scope_status()['a'][SCOPE] == {
            'capped': True,
            'resets_at': None,
            'near_cap': False,
            'capped_at': None,
        }

    def test_accounts_with_empty_scope_caps_omitted(self):
        gate = make_gate(['a', 'b'])
        a, b = gate._accounts
        set_scope_cap(a)  # a observes a scope cap; b has none

        status = gate.scope_status()
        assert 'a' in status
        assert 'b' not in status

    def test_empty_gate_returns_empty(self):
        gate = make_gate(['a'])
        assert gate.scope_status() == {}

    def test_multiple_scopes_and_accounts_roundtrip(self):
        gate = make_gate(['a', 'b'])
        a, b = gate._accounts
        set_scope_cap(a, 'claude-fable-5', capped=True)
        set_scope_cap(b, 'other-model', capped=False, near_cap=True)

        status = gate.scope_status()
        assert set(status) == {'a', 'b'}
        assert set(status['a']) == {'claude-fable-5'}
        assert set(status['b']) == {'other-model'}
        assert status['a']['claude-fable-5']['capped'] is True
        assert status['b']['other-model']['near_cap'] is True
        assert status['b']['other-model']['capped'] is False
