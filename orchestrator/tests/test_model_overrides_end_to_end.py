"""End-to-end seam + drift-guard regression test for metadata.model_overrides
(PRD adaptive-model-routing task ζ, plans/adaptive-model-routing-prd.md).

No orchestrator SOURCE change ships with task ζ: resolve_route's layer-1
metadata_override reader was delivered by dependency ε (task 2535) and the
typed ``TaskMetadata.model_overrides`` field by ζ's own shared/ steps (1-4).
This module is a regression guard locking that shared-field <-> resolver
contract in place -- green on arrival -- plus a drift guard tying the
hand-maintained ``shared.task_metadata.KNOWN_ROLE_NAMES`` mirror to the real
dispatch roles, so a role added to ``ROLES`` without updating the mirror is
caught here rather than silently leaving a role unpinnable.

Fixtures are kept MODULE-LOCAL (not conftest.py) -- mirrors
test_routing_resolver.py's documented rationale (a conftest.py edit forces
verify.py's has_conftest to widen the merge-time scoped-test selection to
the whole owning package).
"""

from __future__ import annotations

import pytest
from shared.task_metadata import KNOWN_ROLE_NAMES, parse_metadata

from orchestrator.agents.roles import ROLES
from orchestrator.config import ModelsConfig, OrchestratorConfig
from orchestrator.routing import RoleDefaults, RouteInputs, resolve_route


@pytest.fixture(autouse=True)
def _isolated_config_env(monkeypatch, tmp_path):
    """Every OrchestratorConfig() built in this module reads ONLY the
    package's shipped defaults.yaml -- no stray project config.yaml, no
    ambient ORCH_CONFIG_PATH. Mirrors test_routing_resolver.py's convention.
    """
    monkeypatch.chdir(tmp_path)
    monkeypatch.setenv('ORCH_CONFIG_PATH', '')


class TestModelOverridesTypedFieldRoundTripsIntoResolver:
    """The typed model_overrides field survives parse_metadata's
    write+enforce path unchanged, and resolve_route still honors it as the
    highest-precedence (metadata_override) layer -- exercising the reader
    added by dependency ε against the typed field added by ζ, with no
    resolver code change in between."""

    def test_typed_field_round_trip_wins_at_resolve_time(self):
        blob = {'model_overrides': {'implementer': 'haiku'}}
        model, warnings = parse_metadata(blob, direction='write', enforce=True)
        assert warnings == []
        task_metadata = model.model_dump()

        inputs = RouteInputs(
            role_name='implementer',
            task_id='1',
            task_metadata=task_metadata,
            plan_shape=None,
            routing_tier=0,
            dispatch_count=0,
            role_defaults=RoleDefaults(
                model='opus', effort='high', budget_usd=5.0, max_turns=50,
            ),
        )
        cfg = OrchestratorConfig(models=ModelsConfig(implementer='opus'))

        decision = resolve_route(inputs, cfg)

        # haiku is in the default allowlist (DEFAULT_ALLOWED_MODELS), so the
        # override is honored -- no fail-safe rejection.
        assert decision.model == 'haiku'
        assert decision.source_layer == 'metadata_override'
        assert decision.rejected == ()


class TestKnownRoleNamesDriftGuard:
    """``set(ROLES) <= KNOWN_ROLE_NAMES``: every dispatchable role is always
    pinnable.

    KNOWN_ROLE_NAMES lives in shared.task_metadata (fused-memory cannot
    import orchestrator, so the submit_task/update_task shape guard's
    "known role names" authority can't live here) and is therefore a
    hand-maintained mirror of ROLES. This test is the tripwire that catches
    the mirror silently diverging from the real dispatch roles -- a role
    added to ROLES without a matching KNOWN_ROLE_NAMES update fails here.
    """

    def test_every_dispatchable_role_is_a_known_role_name(self):
        assert set(ROLES) <= KNOWN_ROLE_NAMES

    def test_every_models_config_field_is_a_known_role_name(self):
        """KNOWN_ROLE_NAMES's docstring also claims it is a superset of
        ModelsConfig's collapsed keys (e.g. 'reviewer', 'triage',
        'module_tagger') -- accepted-but-inert at resolve time (they never
        match `inputs.role_name`), but still meant to shape-validate without
        an unknown_key rejection. This tripwires that second mirrored
        authority: a ModelsConfig field added without a matching
        KNOWN_ROLE_NAMES update would otherwise be unpinnable under its own
        config key with no test failure.
        """
        assert set(ModelsConfig.model_fields) <= KNOWN_ROLE_NAMES
