"""The INV-12 blocks reach the roles that must read them.

PRD: ``plans/inv12-exceptions-owned-or-ratified-prd.md`` (D15) names the
implementer and the reviewer. Task 10464 adds the steward, whose merge-phase
changes land without review, so it reads the implementer's suppression rules
with its own closing sentence.

Each assertion is containment of a named constant or of structured form data in
a prompt. None of them pins the prompt's wording, so rewording the prose leaves
them green.
"""

from __future__ import annotations

import pytest
from shared.governed_exceptions import DECLARATION_FORMS, INLINE_MARKER_FORMS

from orchestrator.agents.detector_guidance import (
    IMPLEMENTER_SUPPRESSION_RULES,
    STEWARD_SUPPRESSION_RULES,
    SUPPRESSION_RULES,
    TYPE_ESCAPES,
)
from orchestrator.agents.roles import (
    IMPLEMENTER,
    REVIEWER_COMPREHENSIVE,
    REVIEWER_INV12_BLOCKERS,
    STEWARD,
)

_SUPPRESSION_READERS = pytest.mark.parametrize(
    'role', [IMPLEMENTER, STEWARD], ids=lambda role: role.name,
)


@_SUPPRESSION_READERS
@pytest.mark.parametrize('form', [*INLINE_MARKER_FORMS, *DECLARATION_FORMS])
def test_the_role_prompt_renders_every_accepted_form(role, form):
    assert form in role.system_prompt


def test_suppression_rules_are_nonempty():
    assert SUPPRESSION_RULES.strip(), (
        'SUPPRESSION_RULES is empty. The empty string is a substring of every '
        'string, so every containment check in this file would pass vacuously '
        'while no role reads any suppression rule at all.'
    )


@_SUPPRESSION_READERS
def test_both_roles_carry_the_shared_suppression_rules(role):
    assert SUPPRESSION_RULES in role.system_prompt


def test_each_role_carries_only_its_own_closing():
    assert IMPLEMENTER_SUPPRESSION_RULES in IMPLEMENTER.system_prompt
    assert IMPLEMENTER_SUPPRESSION_RULES not in STEWARD.system_prompt
    assert STEWARD_SUPPRESSION_RULES in STEWARD.system_prompt
    assert STEWARD_SUPPRESSION_RULES not in IMPLEMENTER.system_prompt
    assert IMPLEMENTER_SUPPRESSION_RULES != STEWARD_SUPPRESSION_RULES


def test_shared_rules_name_the_type_escapes_the_reviewer_blocks():
    assert TYPE_ESCAPES.strip()
    assert TYPE_ESCAPES in REVIEWER_INV12_BLOCKERS
    assert TYPE_ESCAPES in SUPPRESSION_RULES


def test_the_reviewer_blockers_sit_in_the_contract_a_pinned_artifact_cannot_replace():
    spec = REVIEWER_COMPREHENSIVE.prompt_spec
    assert spec is not None

    assert REVIEWER_INV12_BLOCKERS in spec.contract
    assert REVIEWER_INV12_BLOCKERS not in spec.baseline_heuristics
