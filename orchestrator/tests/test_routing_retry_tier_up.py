"""Tests for the ``retry-tier-up`` routing rule (task μ,
plans/adaptive-model-routing-prd.md Phase 4).

RED (step-1) until the rule is appended to defaults.yaml's ``routing.rules``
(step-2). Exercises the rule end-to-end through the *shipped default*
``OrchestratorConfig`` (not a hand-built ``RoutingConfig``): on re-dispatch
after a failure the harness bumps ``metadata.routing.routing_tier`` (γ's
counter), and this rule is what turns that ``tier>=1`` into a
one-rung-stronger model via the resolver's *existing* closed vocabulary
(``min_routing_tier`` match + ``set.model=='+1'`` ladder-relative bump). No
``routing.py``/``config.py`` change is needed — the rule is fully expressible
in the resolver's current grammar (task μ plan, "Verified substrate").

Signal = PRD boundary test 5: dispatch #2's implementer routes one ladder
rung up (``rule_id=='retry-tier-up'``, ``source_layer=='policy_rule'``),
while a role already at the ladder top (architect) clamps there. The
tier-0 case asserts the rule is inert at the default tier (byte-equivalence,
invariant 3).

Fixtures are kept MODULE-LOCAL (not conftest.py) -- mirrors test_routing.py's
documented rationale (a conftest.py edit forces verify.py's has_conftest to
widen the merge-time scoped-test selection to the whole owning package).
"""

from __future__ import annotations

import pytest

from orchestrator.config import OrchestratorConfig
from orchestrator.routing import RoleDefaults, RouteInputs, resolve_route

RULE_ID = 'retry-tier-up'

# Deliberately nonsense -- config always wins for these known roles (each has
# a config field), so a sentinel role_defaults value makes a silent
# role_default-layer fallback (a bug where the config layer fails to apply)
# fail loudly rather than coincide with a real config value. Mirrors
# test_routing_byte_equivalence.py's _SENTINEL_ROLE_DEFAULTS.
_SENTINEL_ROLE_DEFAULTS = RoleDefaults(
    model='role-default-sentinel', effort='role-default-sentinel',
    budget_usd=-1.0, max_turns=-1,
)


@pytest.fixture(autouse=True)
def _isolated_config_env(monkeypatch, tmp_path):
    """Every OrchestratorConfig() built here reads ONLY the package's shipped
    defaults.yaml -- no stray project config.yaml, no ambient
    ORCH_CONFIG_PATH. Mirrors test_routing.py's per-test convention.
    """
    monkeypatch.chdir(tmp_path)
    monkeypatch.setenv('ORCH_CONFIG_PATH', '')


def _inputs(role_name: str, *, routing_tier: int) -> RouteInputs:
    """RouteInputs with NO plan_shape (so the large-plan rules are inert --
    they require a plan) and the given routing_tier, isolating the
    retry-tier-up trigger to the tier counter alone."""
    return RouteInputs(
        role_name=role_name,
        task_id='1',
        task_metadata={},
        plan_shape=None,
        routing_tier=routing_tier,
        dispatch_count=0,
        role_defaults=_SENTINEL_ROLE_DEFAULTS,
    )


def _find_rule(cfg: OrchestratorConfig, rule_id: str):
    return next((r for r in cfg.routing.rules if r.id == rule_id), None)


class TestRetryTierUpRuleShape:
    """The shipped defaults.yaml carries a well-formed retry-tier-up rule."""

    def test_rule_present_with_expected_shape(self):
        cfg = OrchestratorConfig()
        rule = _find_rule(cfg, RULE_ID)

        assert rule is not None, 'retry-tier-up rule missing from defaults.yaml routing.rules'
        assert rule.match.min_routing_tier == 1
        assert rule.match.role is not None
        assert {'implementer', 'debugger', 'architect'}.issubset(set(rule.match.role))
        assert rule.set.model == '+1'

    def test_rule_is_listed_last(self):
        """First-match-wins ordering: retry-tier-up is the LAST rule so
        large-plan retries match the large-plan rules first and keep their
        opus + max_turns=120 headroom (design decision)."""
        cfg = OrchestratorConfig()
        assert cfg.routing.rules[-1].id == RULE_ID


class TestRetryTierUpResolution:
    """resolve_route driven against the shipped default OrchestratorConfig."""

    def test_implementer_tier0_rule_inert(self):
        """(a) tier 0: ``min_routing_tier>=1`` is unmet -> the rule is inert;
        implementer resolves its config model (sonnet) and rule_id is NOT
        retry-tier-up -- byte-equivalence at the default tier (invariant 3)."""
        cfg = OrchestratorConfig()
        decision = resolve_route(_inputs('implementer', routing_tier=0), cfg)

        assert decision.model == 'sonnet'
        assert decision.source_layer == 'config'
        assert decision.rule_id != RULE_ID

    def test_implementer_tier1_bumps_one_rung(self):
        """(b) boundary test 5: at tier 1 the implementer routes one ladder
        rung up (sonnet -> opus) via the retry-tier-up policy rule."""
        cfg = OrchestratorConfig()
        decision = resolve_route(_inputs('implementer', routing_tier=1), cfg)

        assert decision.model == 'opus'
        assert decision.source_layer == 'policy_rule'
        assert decision.rule_id == RULE_ID

    def test_architect_tier1_clamps_at_ladder_top(self):
        """(c) architect's config model is already the ladder top (opus); a
        '+1' bump clamps there -- still opus, still attributed to the rule
        (the match fired and its model set applied without rejection)."""
        cfg = OrchestratorConfig()
        decision = resolve_route(_inputs('architect', routing_tier=1), cfg)

        assert decision.model == 'opus'
        assert decision.source_layer == 'policy_rule'
        assert decision.rule_id == RULE_ID
