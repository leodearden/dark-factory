"""The INV-12 blocks reach the roles D15 names.

PRD: ``plans/inv12-exceptions-owned-or-ratified-prd.md`` (D15). Each assertion
checks that a named constant reaches a prompt. None of them pins the prompt's
wording, so rewording the prose leaves them green.
"""

from __future__ import annotations

import pytest
from shared.governed_exceptions import DECLARATION_FORMS, INLINE_MARKER_FORMS

from orchestrator.agents.roles import (
    IMPLEMENTER,
    REVIEWER_COMPREHENSIVE,
    REVIEWER_INV12_BLOCKERS,
)


@pytest.mark.parametrize('form', [*INLINE_MARKER_FORMS, *DECLARATION_FORMS])
def test_the_implementer_prompt_renders_every_accepted_form(form):
    assert form in IMPLEMENTER.system_prompt


def test_the_reviewer_blockers_sit_in_the_contract_a_pinned_artifact_cannot_replace():
    spec = REVIEWER_COMPREHENSIVE.prompt_spec
    assert spec is not None

    assert REVIEWER_INV12_BLOCKERS in spec.contract
    assert REVIEWER_INV12_BLOCKERS not in spec.baseline_heuristics
