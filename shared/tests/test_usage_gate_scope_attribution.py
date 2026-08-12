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

import asyncio
import json
import os
from datetime import UTC, datetime
from unittest.mock import AsyncMock, patch

import pytest

from shared.cli_invoke import (
    AgentResult,
    AllAccountsCappedException,
    invoke_with_cap_retry,
)
from shared.config_models import AccountConfig, UsageCapConfig
from shared.invocation_outcome import CapHit
from shared.usage_gate import AccountPhase, InvokeSlot, UsageGate

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


# ===========================================================================
# Step 3 — UsageGate.detect_cap_hit forwards `scope` to the handlers
# ===========================================================================

# Known-good cap / near-cap stderr strings (lifted from
# test_usage_gate_exhaustive.py): "You've hit your" + a confirm keyword
# ("usage limit"/"resets") → CapHit; "You're close to" + confirm → NearCap.
CAP_STDERR = "You've hit your usage limit resets in 3h"
NEAR_CAP_STDERR = "You're close to your usage limit. resets in 2h"


class TestDetectCapHitScope:
    """``UsageGate.detect_cap_hit(..., scope=<model>)`` routes the classified
    outcome into the scoped attribution branch of the handlers."""

    def test_detect_cap_hit_scoped_writes_scope_cap_not_phase(self):
        gate = make_gate(['a'])
        acct = gate._accounts[0]

        assert (
            gate.detect_cap_hit('', CAP_STDERR, oauth_token=acct.token, scope=SCOPE)
            is True
        )

        sc = acct.scope_caps[SCOPE]
        assert sc.capped is True
        # resets_at came from the generic classifier's parse of "resets in 3h".
        assert sc.resets_at is not None
        assert acct.phase == AccountPhase.AVAILABLE

    def test_detect_near_cap_scoped_writes_scope_cap_not_phase(self):
        gate = make_gate(['a'])
        acct = gate._accounts[0]

        assert (
            gate.detect_cap_hit(
                '', NEAR_CAP_STDERR, oauth_token=acct.token, scope=SCOPE
            )
            is True
        )

        assert acct.scope_caps[SCOPE].near_cap is True
        assert acct.near_cap is False
        assert acct.phase == AccountPhase.AVAILABLE

    def test_detect_cap_hit_unscoped_transitions_account(self):
        gate = make_gate(['a'])
        acct = gate._accounts[0]

        assert gate.detect_cap_hit('', CAP_STDERR, oauth_token=acct.token) is True

        assert acct.phase == AccountPhase.CAPPED
        assert acct.scope_caps == {}


# ===========================================================================
# Step 5 — InvokeSlot carries scope; report / detect_cap_hit ride slot.scope
# ===========================================================================


class TestInvokeSlotScope:
    """The slot carries the invocation scope; ``report`` / ``detect_cap_hit``
    forward ``slot.scope`` to the gate handlers."""

    def test_ctor_carries_scope(self):
        gate = make_gate(['a'])
        assert InvokeSlot(gate, None, scope=SCOPE).scope == SCOPE

    def test_ctor_scope_defaults_none(self):
        gate = make_gate(['a'])
        assert InvokeSlot(gate, None).scope is None

    async def test_invoke_slot_cm_exposes_scope_and_report_is_scoped(self):
        gate = make_gate(['a'])
        acct = gate._accounts[0]
        target = datetime(2026, 5, 1, 18, 0, tzinfo=UTC)

        async with gate.invoke_slot(scope=SCOPE) as slot:
            assert slot.scope == SCOPE
            slot.report(CapHit(resets_at=target, reason='x'))

        sc = acct.scope_caps[SCOPE]
        assert sc.capped is True
        assert sc.resets_at == target
        assert acct.phase == AccountPhase.AVAILABLE

    async def test_invoke_slot_cm_detect_cap_hit_is_scoped(self):
        gate = make_gate(['a'])
        acct = gate._accounts[0]

        async with gate.invoke_slot(scope=SCOPE) as slot:
            assert slot.detect_cap_hit(CAP_STDERR, '') is True

        assert acct.scope_caps[SCOPE].capped is True
        assert acct.phase == AccountPhase.AVAILABLE

    async def test_invoke_slot_cm_no_scope_report_caps_account(self):
        gate = make_gate(['a'])
        acct = gate._accounts[0]
        target = datetime(2026, 5, 1, 18, 0, tzinfo=UTC)

        async with gate.invoke_slot() as slot:
            assert slot.scope is None
            slot.report(CapHit(resets_at=target, reason='x'))

        assert acct.phase == AccountPhase.CAPPED
        assert acct.scope_caps == {}


# ===========================================================================
# Task 4096 — a scoped CapHit must still release the PROBE_IN_FLIGHT claim
# ===========================================================================


async def _probe_in_flight_slot(gate, *, scope):
    """Drive gate's sole account to PROBE_IN_FLIGHT through before_invoke().

    ``acct.probing = True`` uses the legacy boolean shim (no ``_transition``, so
    no resume-probe background task is started — this file's ``make_gate`` only
    AsyncMocks ``_run_probe``); ``before_invoke`` then takes the real
    ``if acct.probing:`` branch (usage_gate.py:867) and performs the genuine
    PROBING -> PROBE_IN_FLIGHT transition, so ``_open`` is correctly cleared and
    the lease generation matches (no spurious stale-lease warning).
    """
    acct = gate._accounts[0]
    acct.probing = True
    lease = await gate.before_invoke(scope=scope)
    assert acct.phase == AccountPhase.PROBE_IN_FLIGHT
    assert gate._open.is_set() is False  # single-account gate closed by the claim
    return acct, InvokeSlot(gate, lease, scope=scope)


class TestScopedCapHitReleasesProbeSlot:
    """A scoped ``CapHit`` reported while the account is PROBE_IN_FLIGHT must
    release the probe claim, exactly as the ``NearCap`` arm already does.

    The scoped ``_handle_cap_detected`` branch deliberately bypasses
    ``_transition`` (invariant S5), so — unlike the unscoped CapHit, which
    releases the claim as a side effect of the CAPPED transition — nothing else
    on this path can clear PROBE_IN_FLIGHT, and ``report()``'s
    ``finally: _settled = True`` suppresses ``invoke_slot()``'s ``__aexit__``
    safety net. This locks ``report()``'s own docstring invariant: the claim
    "is released on every path", and a scoped cap "leaves the account phase
    AVAILABLE".
    """

    async def test_scoped_cap_hit_releases_probe_and_leaves_account_selectable(self):
        gate = make_gate(['a'])
        target = datetime(2026, 5, 1, 18, 0, tzinfo=UTC)
        acct, slot = await _probe_in_flight_slot(gate, scope=SCOPE)

        slot.report(CapHit(resets_at=target, reason='x'))

        assert slot._settled is True
        # S5 attribution preserved — the fix must not weaken the scoped cap write.
        sc = acct.scope_caps[SCOPE]
        assert sc.capped is True
        assert sc.resets_at == target
        # The probe claim is released (report()'s own documented invariant).
        assert acct.phase == AccountPhase.AVAILABLE
        assert acct.probe_in_flight is False
        assert gate._open.is_set() is True  # DD-5: gate not deadlocked

        # Bounded: in the RED state before_invoke() blocks forever at
        # usage_gate.py:995-996, so a bare await would fail as a 60s suite
        # timeout instead of a sub-second, legible TimeoutError.
        lease = await asyncio.wait_for(gate.before_invoke(), timeout=1.0)
        assert lease is not None
        assert lease.name == 'a'

    async def test_unscoped_cap_hit_still_caps_account(self):
        gate = make_gate(['a'])
        target = datetime(2026, 5, 1, 18, 0, tzinfo=UTC)
        acct, slot = await _probe_in_flight_slot(gate, scope=None)

        slot.report(CapHit(resets_at=target, reason='x'))

        assert acct.phase == AccountPhase.CAPPED
        assert acct.resets_at == target
        assert acct.scope_caps == {}
        # The gate must STAY closed — the added release is a no-op here (the
        # idempotency already pinned by test_usage_gate_exhaustive.py's
        # test_noop_after_handle_cap_detected_already_cleared).
        assert gate._open.is_set() is False

    async def test_scoped_cap_hit_preserves_near_cap_annotation(self):
        gate = make_gate(['a'])
        acct, slot = await _probe_in_flight_slot(gate, scope=SCOPE)
        acct.near_cap = True

        slot.report(CapHit(resets_at=None, reason='x'))

        # release_probe_slot passes clear_near_cap=False (usage_gate.py:2127).
        assert acct.near_cap is True
        assert acct.phase == AccountPhase.AVAILABLE


# ===========================================================================
# Step 7 — end-to-end β: invoke_with_cap_retry derives + threads scope
# ===========================================================================


async def _fake_cap_invoke(**kwargs) -> AgentResult:
    """A heuristic-cap invoke_fn: zero-cost / instant / ≤1-turn / not-success.

    Accepts arbitrary kwargs (oauth_token / config_dir / backend / model /
    prompt) so it drops into invoke_with_cap_retry's dispatch. The empty
    output+stderr classify as a non-cap Failure, so the heuristic safety net
    fires ``slot.report(synthetic CapHit)`` — the deterministic β signal.
    """
    return AgentResult(success=False, output='', stderr='')


class TestCliThreadsScope:
    """``invoke_with_cap_retry`` derives the slot scope from the invoked model
    (claude-backend only) and threads it to the write-half handlers (boundary
    B1 write half; B8 non-claude bypass)."""

    async def test_claude_scoped_cap_lands_on_scope_not_account(self):
        gate = make_gate(['a'])
        acct = gate._accounts[0]

        with pytest.raises(AllAccountsCappedException):
            await invoke_with_cap_retry(
                gate,
                'lbl',
                invoke_fn=_fake_cap_invoke,
                backend='claude',
                max_cap_retries=1,
                model=SCOPE,
                prompt='hi',
            )

        # Write half of B1: scoped cap recorded, account phase machine untouched.
        assert acct.scope_caps[SCOPE].capped is True
        assert acct.phase == AccountPhase.AVAILABLE

    async def test_non_claude_backend_bypasses_scope(self):
        gate = make_gate(['a'])
        acct = gate._accounts[0]

        with pytest.raises(AllAccountsCappedException):
            await invoke_with_cap_retry(
                gate,
                'lbl',
                invoke_fn=_fake_cap_invoke,
                backend='codex',
                max_cap_retries=1,
                model=SCOPE,  # same model string, but codex → no scope derived
                prompt='hi',
            )

        # B8: a non-claude backend never derives scope, so attribution is
        # account-level (scope-blind), exactly as today.
        assert acct.scope_caps == {}
        assert acct.phase == AccountPhase.CAPPED
