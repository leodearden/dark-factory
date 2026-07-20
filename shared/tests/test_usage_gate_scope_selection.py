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
