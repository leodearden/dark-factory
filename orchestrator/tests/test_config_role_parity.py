"""config_role is total over the orchestrator's registered role objects.

Those are ROLES, TRIAGE and the reviewers build_reviewer_role makes.

shared may not import orchestrator, so this invariant is checked here, against
the live role objects rather than a hand-kept list of names.
"""

from __future__ import annotations

import pytest
from shared.roles import DISPATCHABLE_ROLES, REVIEWER_VARIANTS, RoleName, config_role

from orchestrator.agents.roles import ROLES, build_reviewer_role
from orchestrator.agents.triage import TRIAGE


@pytest.mark.parametrize('dispatch_name', sorted(ROLES))
def test_every_registered_role_has_a_config_role(dispatch_name: str) -> None:
    role = config_role(dispatch_name)
    assert role.value == dispatch_name or (
        role is RoleName.REVIEWER and dispatch_name in REVIEWER_VARIANTS
    )


def test_registered_roles_are_within_the_role_wire_contract() -> None:
    assert set(ROLES) <= {r.value for r in DISPATCHABLE_ROLES} | REVIEWER_VARIANTS


def test_triage_has_its_own_config_role() -> None:
    assert config_role(TRIAGE.name) is RoleName.TRIAGE


def test_built_reviewer_roles_collapse_to_reviewer() -> None:
    reviewer = build_reviewer_role('security', 'any specialization')
    assert config_role(reviewer.name) is RoleName.REVIEWER
