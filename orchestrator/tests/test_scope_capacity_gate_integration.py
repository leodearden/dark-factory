"""B7 -- resolver degrade wired from a REAL gate scope-capacity snapshot
(PRD epsilon, plans/usage-gate-model-scoped-caps-prd.md).

The sibling shared-side suite (shared/tests/test_usage_gate_scope_boundary.py)
drives boundaries B1-B6 and B8 end-to-end through invoke_with_cap_retry. B7
is the one boundary that crosses the shared/orchestrator seam: it binds a
REAL UsageGate's ``scope_capacity_snapshot()`` (task gamma, task 2857) to
``orchestrator.routing.resolve_route``'s scope-capacity fail-safe (task
delta, task 2858 -- see ``TestScopeCapacityFallback`` in
``orchestrator/tests/test_routing_resolver.py``).

delta's own unit tests already prove ``resolve_route`` degrades correctly on
a HAND-BUILT ``scope_capacity={'claude-fable-5': False}`` dict; the
non-redundant seam this file adds is a real gate PRODUCING that snapshot
from an actually-exhausted fable scope and feeding it to the resolver --
proving the two subsystems compose. The routing_decision-event surfacing
half is already delta-covered (``test_workflow_routing_decision.py``) and is
not duplicated here.

Isolated file: this module imports both ``shared`` and ``orchestrator``, so
it cannot live inside either package's existing scope-suite. Local minimal
harness only, mirroring ``orchestrator/tests/test_routing_resolver.py``'s
config-isolation fixture and ``_role_defaults``/``_inputs`` builders, plus
``shared/tests/test_usage_gate_scope_boundary.py``'s ``make_gate``/
``set_scope_cap`` (replicated locally -- ``orchestrator/tests`` cannot
import from ``shared/tests``).

NO PRODUCTION CODE CHANGES: gamma/delta are already merged to main -- this
is pure gate-test authoring, expected GREEN on authoring.
"""

from __future__ import annotations

import os
from datetime import UTC, datetime, timedelta
from unittest.mock import AsyncMock, patch

import pytest
from shared.config_models import AccountConfig, UsageCapConfig
from shared.usage_gate import ScopeCap, UsageGate

from orchestrator.config import ModelsConfig, OrchestratorConfig, RoutingConfig
from orchestrator.routing import (
    DEFAULT_ALLOWED_MODELS,
    PlanShape,
    RoleDefaults,
    RouteInputs,
    resolve_route,
)

SCOPE = 'claude-fable-5'


@pytest.fixture(autouse=True)
def _isolated_config_env(monkeypatch, tmp_path):
    """Every OrchestratorConfig() built in this module reads ONLY the
    package's shipped defaults.yaml -- no stray project config.yaml, no
    ambient ORCH_CONFIG_PATH. Mirrors test_routing_resolver.py's per-test
    convention.
    """
    monkeypatch.chdir(tmp_path)
    monkeypatch.setenv('ORCH_CONFIG_PATH', '')


# ---------------------------------------------------------------------------
# Local helpers (mirroring orchestrator/tests/test_routing_resolver.py:55-90
# and shared/tests/test_usage_gate_scope_boundary.py's make_gate/set_scope_cap)
# ---------------------------------------------------------------------------


def _role_defaults(
    *, model='opus', effort='high', budget_usd=10.0, max_turns=80,
) -> RoleDefaults:
    return RoleDefaults(model=model, effort=effort, budget_usd=budget_usd, max_turns=max_turns)


def _inputs(
    *,
    role_name: str = 'implementer',
    task_metadata: dict | None = None,
    plan_shape: PlanShape | None = None,
    routing_tier: int = 0,
    dispatch_count: int = 0,
    role_defaults: RoleDefaults | None = None,
    scope_capacity: dict[str, bool] | None = None,
) -> RouteInputs:
    """A RouteInputs with sensible defaults; role_defaults defaults to a
    'sonnet' sentinel distinguishable from any rule's opus/haiku set.model.

    ``scope_capacity`` defaults to None (the resolver's S7 fail-safe skips
    the capacity check entirely) -- B7 passes a real
    ``gate.scope_capacity_snapshot()`` explicitly.
    """
    return RouteInputs(
        role_name=role_name,
        task_id='1',
        task_metadata={} if task_metadata is None else task_metadata,
        plan_shape=plan_shape,
        routing_tier=routing_tier,
        dispatch_count=dispatch_count,
        role_defaults=role_defaults if role_defaults is not None else _role_defaults(model='sonnet'),
        scope_capacity=scope_capacity,
    )


def make_gate(account_names: list[str], *, cost_store=None, **cfg) -> UsageGate:
    """Create a real UsageGate with fake accounts, probe + reset-wait disabled.

    Mirrors ``shared/tests/test_usage_gate_scope_boundary.py``'s
    ``make_gate`` (P1) -- replicated locally since ``orchestrator/tests``
    cannot import from ``shared/tests``. This cross-package copy stays a
    separate replica even after the shared-side consolidation tracked in
    ticket tkt_0RRGS6H40KB28XBYSRX5HR4HNX (which only touches the
    shared/tests copies of make_gate/set_scope_cap).
    """
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


def set_scope_cap(
    acct,
    scope: str = SCOPE,
    *,
    capped: bool = True,
    resets_at: datetime | None = None,
    capped_at: datetime | None = None,
    near_cap: bool = False,
) -> ScopeCap:
    """Directly install a ScopeCap overlay on *acct* (bypassing the β handler).

    Mirrors ``shared/tests/test_usage_gate_scope_boundary.py``'s
    ``set_scope_cap``.
    """
    sc = ScopeCap(
        capped=capped, resets_at=resets_at, near_cap=near_cap, capped_at=capped_at,
    )
    acct.scope_caps[scope] = sc
    return sc


# ===========================================================================
# B7 -- resolver degrade wired from a REAL gate scope-capacity snapshot
# ===========================================================================


class TestB7ResolverDegradeFromRealGateSnapshot:
    """B7: a REAL ``UsageGate.scope_capacity_snapshot()`` -- not a hand-built
    dict, as delta's own unit tests use -- feeds ``resolve_route``'s
    scope-capacity fail-safe (S7): a fable-pinned metadata_override degrades
    to the next layer, dispatch still resolves a valid model, and the
    rejection is recorded namespaced by layer."""

    def test_b7_real_gate_snapshot_degrades_metadata_override(self):
        gate = make_gate(['a'])
        a = gate._accounts[0]
        now = datetime.now(UTC)
        # Exhaust A's (only account's) fable scope with a future deadline.
        set_scope_cap(a, resets_at=now + timedelta(hours=1), capped_at=now)

        snap = gate.scope_capacity_snapshot()
        assert snap == {SCOPE: False}  # the gate reports no fable headroom

        # routing.allowed_models must admit SCOPE for this test to reach the
        # scope-capacity check (S7) rather than being rejected earlier by
        # the unrelated allowlist check (invariant 2) -- 'claude-fable-5' is
        # deliberately NOT in DEFAULT_ALLOWED_MODELS yet (see CLAUDE.md
        # "Adaptive-routing substrate").
        cfg = OrchestratorConfig(
            models=ModelsConfig(implementer='opus'),
            routing=RoutingConfig(allowed_models=[*DEFAULT_ALLOWED_MODELS, SCOPE]),
        )
        inputs = _inputs(
            role_name='implementer',
            task_metadata={'model_overrides': {'implementer': SCOPE}},
            role_defaults=_role_defaults(model='sonnet'),
            scope_capacity=snap,
        )

        decision = resolve_route(inputs, cfg)

        assert decision.model != SCOPE  # degraded away from the exhausted model
        assert decision.model in DEFAULT_ALLOWED_MODELS  # a valid model resolved
        assert decision.source_layer == 'config'
        assert 'metadata_override:model-capacity-exhausted' in decision.rejected
