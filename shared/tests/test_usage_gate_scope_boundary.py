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

import os
from datetime import datetime
from unittest.mock import AsyncMock, patch

from shared.cli_invoke import AgentResult
from shared.config_models import AccountConfig, UsageCapConfig
from shared.usage_gate import ScopeCap, UsageGate

# NOTE: this scaffold (prerequisite P1) imports only what the harness below
# uses today. Each Bn step below (B1-B6, B8) adds the specific additional
# imports (asyncio, contextlib, json, UTC, timedelta, invoke_with_cap_retry,
# AccountPhase, _SLEEP_PATCH) its own test needs, keeping every commit
# ruff-clean rather than pre-importing the whole suite's eventual closure.

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
