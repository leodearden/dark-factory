"""Two-way boundary suite B1-B8 for the model-scoped usage-cap machinery
(PRD epsilon, plans/usage-gate-model-scoped-caps-prd.md).

Task epsilon is the integration gate: "B1-B8 green in CI" is the hard dep
that fable admission (task 2544) and the fable-architect-eval ratification
gate (tau3) consume as the safety evidence that scope-aware failover exists
before fable reaches production routing.

The alpha/beta/gamma/delta scope-cap machinery this suite exercises is
already merged to main (tasks 2855/2856/2857/2858). The existing per-task
unit suites (test_usage_gate_scope*.py) drive individual methods directly
(``before_invoke``/``_handle_cap_detected`` with a hand-installed
``set_scope_cap`` precondition). This file is different: every test here
drives the FULL loop through ``invoke_with_cap_retry``'s ``invoke_fn`` seam
with a scripted fake, so the scope is DERIVED FROM THE INVOKED MODEL
(``cli_invoke.py``'s ``scope = scope_for(model, gate._config)``) and a real
cap-hit stderr flows ``invoke_fn`` -> ``slot.detect_cap_hit`` ->
``gate.detect_cap_hit(scope=)`` -> ``_handle_cap_detected(scope=)`` ->
``scope_caps`` write -> retry -> ``before_invoke(scope=)`` failover.

This requires a REAL ``UsageGate`` (``make_gate``) with a default
``UsageCapConfig``, not a ``MagicMock`` gate (a mock has no ``_config`` and
would derive scope ``None`` unconditionally). Mirrors
``test_cap_retry.py::test_real_gate_failover``.

Boundaries covered here (B7 -- the resolver-composition boundary -- lives in
``orchestrator/tests/test_scope_capacity_gate_integration.py`` since it
imports both ``shared`` and ``orchestrator``):

* B1 -- a fable cap leaves general capacity open (scope=fable CapHit does
  not transition the account phase).
* B2 -- a scoped failover's cost event carries ``scope`` in its JSON details.
* B3 -- all-fable-capped is not a fleet freeze: a concurrent general dispatch
  completes immediately and a concurrent scope=fable caller merely parks.
* B4 -- a general (account-level) cap dominates for every scope.
* B5 -- timer uncap + reactive re-cap resolves in one bounded retry loop,
  with no tight spin (exactly one cooldown sleep).
* B6 -- ``scoped_cap_models=[]`` (the kill switch) is byte-equivalent to no
  scope machinery at all.
* B8 -- a non-claude backend bypasses scope derivation/state entirely, even
  when the model string is a configured scoped-cap model.

Isolated file (mirrors the ``test_usage_gate_scope_attribution.py`` /
``test_usage_gate_scope_selection.py`` convention) with local minimal
helpers so it stays independent of the big suites. Every cap-driving test
patches ``shared.cli_invoke.asyncio.sleep`` (cooldown neutralize); every
scope-wait/parking assertion is bounded (``asyncio.wait_for`` + cancel in a
``finally``) so a regression surfaces as a deterministic failure/timeout,
never an infinite hang in CI.

NO PRODUCTION CODE CHANGES: alpha/beta/gamma/delta are already merged, so
this is pure gate-test authoring -- every Bn test is expected GREEN on
authoring (it characterizes/locks landed behavior for the ratification
gate).
"""

from __future__ import annotations

import asyncio
import contextlib
import json
import os
from datetime import UTC, datetime, timedelta
from unittest.mock import AsyncMock, patch

from shared.cli_invoke import AgentResult, invoke_with_cap_retry
from shared.config_models import AccountConfig, UsageCapConfig
from shared.usage_gate import AccountPhase, ScopeCap, UsageGate

# NOTE: this scaffold (prerequisite P1) imports only what the harness below
# uses today; each Bn step below adds the specific additional imports its
# own test needs, keeping every commit ruff-clean rather than
# pre-importing the whole suite's eventual closure.

SCOPE = 'claude-fable-5'
_SLEEP_PATCH = 'shared.cli_invoke.asyncio.sleep'


# ---------------------------------------------------------------------------
# Local helpers (mirroring test_usage_gate_scope_selection.py:46-84 and
# test_cap_retry.py's make_gate/make_result/_SLEEP_PATCH)
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


class scripted_invoke:
    """Async fake ``invoke_fn`` yielding each of *results* in call order
    (the last result repeats if called more times than results supplied).

    Tolerates arbitrary ``**kwargs`` (``prompt``/``model``/``oauth_token``/
    ``config_dir``/``backend`` -- everything ``invoke_with_cap_retry``
    forwards to ``invoke_fn``) and records every call's kwargs onto
    ``self.calls`` for assertions (e.g. B8's backend-forwarding check).
    """

    def __init__(self, *results: AgentResult) -> None:
        self._results = list(results)
        self.calls: list[dict] = []

    async def __call__(self, **kwargs) -> AgentResult:
        self.calls.append(kwargs)
        idx = min(len(self.calls) - 1, len(self._results) - 1)
        return self._results[idx]


def cap_result(stderr: str, output: str = 'partial') -> AgentResult:
    """A capped AgentResult carrying a cap-hit stderr pattern.

    ``success=True`` mirrors ``test_cap_retry.py::test_real_gate_failover``'s
    ``capped`` fixture: cap detection happens via ``slot.detect_cap_hit``,
    which builds its OWN synthetic ``success=False`` result internally
    (``UsageGate.detect_cap_hit``), so the raw result's ``success`` value is
    irrelevant to cap classification here.
    """
    return AgentResult(success=True, output=output, stderr=stderr, cost_usd=0.5)


def ok_result(output: str = 'complete') -> AgentResult:
    """A successful AgentResult (mirrors make_result in test_cap_retry.py)."""
    return AgentResult(success=True, output=output, cost_usd=0.5)


# ===========================================================================
# B1 -- a fable cap leaves general capacity open
# ===========================================================================


class TestB1FableCapLeavesGeneralOpen:
    """B1: a scoped (fable) cap hit driven end-to-end through
    invoke_with_cap_retry fails over WITHOUT transitioning the capped
    account's phase (it stays serviceable for general work), and a
    following general invocation still selects and completes on it."""

    async def test_b1_fable_cap_leaves_general_open(self):
        gate = make_gate(['a', 'b'])
        a, b = gate._accounts

        with patch(_SLEEP_PATCH, new_callable=AsyncMock):
            got = await invoke_with_cap_retry(
                gate, 'lbl', model='claude-fable-5', backend='claude',
                invoke_fn=scripted_invoke(
                    cap_result("You've hit your usage limit. resets in 1h"),
                    ok_result(),
                ),
                prompt='hi',
            )

        assert got.account_name == 'b'  # failed over
        assert a.phase == AccountPhase.AVAILABLE  # NOT capped by the fable hit
        assert a.scope_caps[SCOPE].capped is True

        with patch(_SLEEP_PATCH, new_callable=AsyncMock):
            general = await invoke_with_cap_retry(
                gate, 'lbl', model='sonnet', backend='claude',
                invoke_fn=scripted_invoke(ok_result()),
                prompt='hi',
            )

        # The fable cap did not remove A's general capacity.
        assert general.account_name == 'a'
        assert general.output == 'complete'


# ===========================================================================
# B2 -- scoped failover cost event carries scope
# ===========================================================================


def _failover_calls(mock_fire):
    """The ('failover') calls recorded by a patched _fire_cost_event mock."""
    return [c for c in mock_fire.call_args_list if c.args[1] == 'failover']


class TestB2ScopedFailoverCarriesScope:
    """B2: a scoped account switch fires a ``failover`` cost event whose
    JSON details include ``scope``, driven end-to-end through the
    cap-retry loop (not by calling before_invoke directly)."""

    async def test_b2_scoped_failover_event_carries_scope(self):
        gate = make_gate(['a', 'b'], cost_store=make_mock_cost_store())

        with (
            patch.object(gate, '_fire_cost_event') as mock_fire,
            patch(_SLEEP_PATCH, new_callable=AsyncMock),
        ):
            got = await invoke_with_cap_retry(
                gate, 'lbl', model='claude-fable-5', backend='claude',
                invoke_fn=scripted_invoke(
                    cap_result("You've hit your usage limit. resets in 1h"),
                    ok_result(),
                ),
                prompt='hi',
            )

        assert got.account_name == 'b'

        failovers = _failover_calls(mock_fire)
        assert len(failovers) == 1
        name, _event, details_json = failovers[0].args
        assert name == 'b'
        assert json.loads(details_json) == {'from': 'a', 'to': 'b', 'scope': SCOPE}


# ===========================================================================
# B3 -- all-fable-capped is not a fleet freeze
# ===========================================================================


class TestB3AllFableCappedNotFleetFreeze:
    """B3: exhausting the fable scope on every account does not freeze the
    fleet -- a concurrent general dispatch still completes immediately
    (driven end-to-end through invoke_with_cap_retry), and a concurrent
    scope=fable caller merely parks, never blocking ``_open``. All waits
    are bounded so a regression surfaces as a timeout, never a hang."""

    async def test_b3_scope_exhaustion_never_freezes_fleet(self):
        gate = make_gate(['a', 'b'])
        a, b = gate._accounts
        now = datetime.now(UTC)
        far = now + timedelta(hours=6)
        # Both fable-scope-capped (far-future deadline) but generally AVAILABLE.
        set_scope_cap(a, resets_at=far, capped_at=now)
        set_scope_cap(b, resets_at=far, capped_at=now)
        assert a.phase == AccountPhase.AVAILABLE
        assert b.phase == AccountPhase.AVAILABLE

        # (1) A concurrent GENERAL dispatch completes immediately -- fable
        # exhaustion never blocks general work.
        general = await asyncio.wait_for(
            invoke_with_cap_retry(
                gate, 'lbl', model='sonnet', backend='claude',
                invoke_fn=scripted_invoke(ok_result()),
                prompt='hi',
            ),
            timeout=1.0,
        )
        assert general.account_name in ('a', 'b')
        assert general.output == 'complete'
        assert gate._open.is_set() is True
        assert gate.paused_reason == ''
        assert gate.is_paused is False

        # (2) A concurrent scope=fable caller parks -- it does NOT freeze
        # the fleet.
        scoped_task = asyncio.create_task(gate.before_invoke(scope=SCOPE))
        try:
            # Let the scoped selection reach its (scope-local) wait.
            await asyncio.sleep(0.05)
            assert scoped_task.done() is False
            assert gate._open.is_set() is True
            assert gate.paused_reason == ''
        finally:
            scoped_task.cancel()
            with contextlib.suppress(asyncio.CancelledError):
                await scoped_task


# ===========================================================================
# B4 -- a general (account-level) cap dominates every scope
# ===========================================================================


class TestB4GeneralCapDominatesEveryScope:
    """B4: a general (scope=None) cap hit driven end-to-end through
    invoke_with_cap_retry transitions the account to CAPPED via the
    account-level path, and that CAPPED phase dominates for EVERY scope
    (S4) -- both a fable-scope selection and a general selection skip it."""

    async def test_b4_general_cap_dominates_every_scope(self):
        gate = make_gate(['a', 'b'])
        a, b = gate._accounts

        with patch(_SLEEP_PATCH, new_callable=AsyncMock):
            got = await invoke_with_cap_retry(
                gate, 'lbl', model='sonnet', backend='claude',
                invoke_fn=scripted_invoke(
                    cap_result("You've used your usage limit. resets in 30m"),
                    ok_result(),
                ),
                prompt='hi',
            )

        assert got.account_name == 'b'
        # Account-level cap via the scope=None path (not a scope overlay).
        assert a.phase == AccountPhase.CAPPED

        # The account cap dominates for EVERY scope (S4) -- A is skipped for
        # both the fable scope and general selection.
        fable_lease = await gate.before_invoke(scope=SCOPE)
        assert fable_lease is not None
        assert fable_lease.name == 'b'
        general_lease = await gate.before_invoke()
        assert general_lease is not None
        assert general_lease.name == 'b'


# ===========================================================================
# B5 -- timer uncap + reactive re-cap, no tight loop
# ===========================================================================


class TestB5TimerUncapReactiveRecap:
    """B5: a fable scope cap whose uncap deadline has already passed is
    optimistically treated as open at selection time (S6), the very next
    invocation on that account re-caps it with a NEW future deadline
    (reactive re-cap), and the retry loop still resolves in exactly one
    bounded cooldown + failover -- never a tight spin."""

    async def test_b5_timer_uncap_then_reactive_recap_is_bounded(self):
        gate = make_gate(['a', 'b'])
        a, b = gate._accounts
        now = datetime.now(UTC)
        # A's fable-scope cap deadline has already passed -- the selection-time
        # predicate treats it as open (S6/S8). B has fable headroom (no scope
        # cap installed at all).
        set_scope_cap(
            a, resets_at=now - timedelta(seconds=5), capped_at=now - timedelta(hours=1),
        )

        with patch(_SLEEP_PATCH, new_callable=AsyncMock) as mock_sleep:
            got = await invoke_with_cap_retry(
                gate, 'lbl', model='claude-fable-5', backend='claude',
                invoke_fn=scripted_invoke(
                    cap_result("You've hit your usage limit. resets in 6h"),
                    ok_result(),
                ),
                prompt='hi',
            )

        assert got.account_name == 'b'
        sc = a.scope_caps[SCOPE]
        assert sc.capped is True
        # Reactive re-cap installed a NEW deadline, strictly in the future.
        assert sc.resets_at is not None
        assert sc.resets_at > datetime.now(UTC)
        # Exactly one cap cooldown -> retry: counters advanced, no tight spin.
        assert mock_sleep.await_count == 1


# ===========================================================================
# B6 -- byte-equivalence kill switch
# ===========================================================================


class TestB6KillSwitchByteEquivalence:
    """B6: scoped_cap_models=[] (the kill switch) is byte-equivalent to no
    scope machinery at all -- no scope state is ever allocated or read, and
    a fable-model invocation through the loop takes the exact same
    account-level path as any other model (decision 6)."""

    async def test_b6_no_scope_state_allocated(self):
        gate = make_gate(['a'], scoped_cap_models=[])

        assert gate.scope_capacity_snapshot() == {}
        assert gate.scope_status() == {}

        # With no scope caps possible, scope=fable selection behaves as
        # general.
        lease = await gate.before_invoke(scope=SCOPE)
        assert lease is not None
        assert lease.name == 'a'

    async def test_b6_fable_model_through_loop_is_byte_equivalent_to_general(self):
        gate = make_gate(['a', 'b'], scoped_cap_models=[])
        a, b = gate._accounts

        with patch(_SLEEP_PATCH, new_callable=AsyncMock):
            got = await invoke_with_cap_retry(
                gate, 'lbl', model='claude-fable-5', backend='claude',
                invoke_fn=scripted_invoke(
                    cap_result("You've hit your usage limit. resets in 1h"),
                    ok_result(),
                ),
                prompt='hi',
            )

        # scope_for('claude-fable-5', cfg-with-empty-scoped_cap_models) is
        # None -- the CapHit takes the account-level path.
        assert a.phase == AccountPhase.CAPPED
        assert a.scope_caps == {}  # no scope state written
        assert got.account_name == 'b'


# ===========================================================================
# B8 -- a non-claude backend bypasses scope derivation/state entirely
# ===========================================================================


class TestB8NonClaudeBackendBypass:
    """B8: a non-claude backend bypasses scope derivation/state entirely,
    even when the invoked model string is a configured scoped-cap model --
    the codex CapHit takes the account-level path and no scope_caps entry
    is ever written anywhere (decision 7). The T1 reconnect contract shapes
    (backend/prompt/oauth_token forwarded to invoke_fn) are preserved."""

    async def test_b8_codex_backend_bypasses_scope_derivation(self):
        gate = make_gate(['a', 'b'])
        a, b = gate._accounts
        invoke_fn = scripted_invoke(cap_result('usage limit reached'), ok_result())

        with patch(_SLEEP_PATCH, new_callable=AsyncMock):
            got = await invoke_with_cap_retry(
                gate, 'lbl', model='claude-fable-5', backend='codex',
                invoke_fn=invoke_fn,
                prompt='hi',
            )

        # No scope state written anywhere -- backend != 'claude' derives
        # scope None even though the model IS a scoped-cap model.
        assert a.scope_caps == {}
        assert b.scope_caps == {}
        assert gate.scope_status() == {}
        # The codex CapHit took the account-level path and failed over.
        assert a.phase == AccountPhase.CAPPED
        assert got.account_name == 'b'

        # T1 reconnect contract shapes preserved (mirrors TestInvokeFnParameter).
        first_call = invoke_fn.calls[0]
        assert first_call['backend'] == 'codex'
        assert first_call['prompt'] == 'hi'
        assert first_call['oauth_token'] == 'fake-token-a'
