"""Tests for the ``simple-saturated-full-path`` routing rule (task ν,
plans/adaptive-model-routing-prd.md Phase 4, boundary test 8).

RED (step-5) until the rule is added to defaults.yaml's ``routing.rules``
(step-6). Exercises the rule end-to-end through the *shipped default*
``OrchestratorConfig`` (not a hand-built ``RoutingConfig``): once a SIMPLE_TASK
dispatch exhausts its turn cap, ``workflow._stamp_simple_saturated`` stamps
``metadata.routing.simple_saturated=True`` (ν's flag), and THIS rule turns that
flag into a policy-rule attribution on the architect's routing decision — the
architect is always the first agent on the full path, so its
``routing_decision`` reliably NAMES the saturation (boundary test 8). No
``routing.py``/``config.py`` change is needed — the ``simple_saturated`` match
condition already exists (task ε substrate); only the defaults.yaml rule is new.

The rule sets ``model: opus`` (source_layer=policy_rule for unambiguous
attribution); since the architect's config model is already opus, this is
byte-equivalent — the actual re-route to the full path is done by the
SIMPLE_TASK gate (``_should_run_simple_task``), not by this rule's model set.

Fixtures are kept MODULE-LOCAL (not conftest.py) -- mirrors test_routing.py's
documented rationale (a conftest.py edit forces verify.py's has_conftest to
widen the merge-time scoped-test selection to the whole owning package), and
copies the structure of the sibling test_routing_retry_tier_up.py verbatim.
"""

from __future__ import annotations

import pytest

from orchestrator.config import OrchestratorConfig
from orchestrator.routing import RoleDefaults, RouteInputs, resolve_route

RULE_ID = 'simple-saturated-full-path'

# Deliberately nonsense -- config always wins for the architect role (it has a
# config field), so a sentinel role_defaults value makes a silent
# role_default-layer fallback (a bug where the config layer fails to apply)
# fail loudly rather than coincide with a real config value. Mirrors
# test_routing_retry_tier_up.py's _SENTINEL_ROLE_DEFAULTS.
_SENTINEL_ROLE_DEFAULTS = RoleDefaults(
    model='role-default-sentinel', effort='role-default-sentinel',
    budget_usd=-1.0, max_turns=-1,
)


@pytest.fixture(autouse=True)
def _isolated_config_env(monkeypatch, tmp_path):
    """Every OrchestratorConfig() built here reads ONLY the package's shipped
    defaults.yaml -- no stray project config.yaml, no ambient
    ORCH_CONFIG_PATH. Mirrors test_routing_retry_tier_up.py's convention.
    """
    monkeypatch.chdir(tmp_path)
    monkeypatch.setenv('ORCH_CONFIG_PATH', '')


def _inputs(role_name: str, *, simple_saturated: bool) -> RouteInputs:
    """RouteInputs whose ``metadata.routing.simple_saturated`` is set as given.

    ``routing_tier``/``dispatch_count`` are 0 and ``plan_shape`` is None so the
    retry-tier-up and large-plan rules are all inert -- isolating the
    simple-saturated trigger to the flag alone (mirrors
    ``_task_simple_saturated``'s read of ``metadata['routing']['simple_saturated']``).
    """
    return RouteInputs(
        role_name=role_name,
        task_id='1',
        task_metadata={'routing': {'simple_saturated': simple_saturated}},
        plan_shape=None,
        routing_tier=0,
        dispatch_count=0,
        role_defaults=_SENTINEL_ROLE_DEFAULTS,
    )


def _find_rule(cfg: OrchestratorConfig, rule_id: str):
    return next((r for r in cfg.routing.rules if r.id == rule_id), None)


class TestSimpleSaturatedRuleShape:
    """The shipped defaults.yaml carries a well-formed simple-saturated rule."""

    def test_rule_present_with_expected_shape(self):
        cfg = OrchestratorConfig()
        rule = _find_rule(cfg, RULE_ID)

        assert rule is not None, (
            'simple-saturated-full-path rule missing from defaults.yaml routing.rules'
        )
        assert rule.match.simple_saturated is True
        assert rule.match.role is not None
        assert 'architect' in rule.match.role
        assert rule.set.model == 'opus'

    def test_retry_tier_up_still_listed_last(self):
        """First-match-wins ordering: the new rule is inserted BEFORE
        retry-tier-up, so retry-tier-up remains the LAST rule
        (test_routing_retry_tier_up.py::test_rule_is_listed_last invariant)."""
        cfg = OrchestratorConfig()
        assert cfg.routing.rules[-1].id == 'retry-tier-up'


class TestSimpleSaturatedResolution:
    """resolve_route driven against the shipped default OrchestratorConfig."""

    def test_architect_saturated_names_the_rule(self):
        """(c) boundary test 8: a saturated task's architect routing decision
        is attributed to the simple-saturated-full-path policy rule (opus)."""
        cfg = OrchestratorConfig()
        decision = resolve_route(_inputs('architect', simple_saturated=True), cfg)

        assert decision.rule_id == RULE_ID
        assert decision.source_layer == 'policy_rule'
        assert decision.model == 'opus'

    def test_architect_not_saturated_rule_inert(self):
        """(d) at the non-saturated default the rule does NOT match, so the
        architect's decision is not attributed to it -- byte-equivalence at the
        default state (invariant 3)."""
        cfg = OrchestratorConfig()
        decision = resolve_route(_inputs('architect', simple_saturated=False), cfg)

        assert decision.rule_id != RULE_ID
