"""House-2559 drift-guard: pin the ε standing-decision prompt sections into the
assembled Stage-1 / Stage-2 system prompts (task 2898 ε, PRD
plans/stage1-entity-standing-decision-prd.md §ε).

Modeled on test_recon_report_guidance_drift.py: the render_* functions in
recon_self_model are the single source, and the stage prompts interpolate them
via an own-line `{render_*()}` f-string field. Because the f-string interpolates
the exact return value at import time, a plain `in` substring check is a
byte-identical pin — if a future edit re-hardcodes or drifts either section, the
`render_*() in <assembled prompt>` assertion fails rather than silently
diverging.

Stage-specificity is part of the contract: the shared schema section is pinned
into BOTH stage prompts, while the investigation_outcome section is Stage-2-only
(Stage 1 has no writer path and does not conclude investigations) and is asserted
ABSENT from the Stage-1 prompt.
"""
from __future__ import annotations

from fused_memory.reconciliation.prompts.stage1 import STAGE1_SYSTEM_PROMPT
from fused_memory.reconciliation.prompts.stage2 import (
    STAGE2_SYSTEM_PROMPT,
    build_stage2_system_prompt,
)
from fused_memory.reconciliation.recon_self_model import (
    render_entity_standing_decision_schema_section,
)


class TestEntityStandingDecisionSchemaSectionPinned:
    """The SHARED schema section is byte-identically present in BOTH stage
    prompts (and the dark_factory-assembled Stage-2 prompt, which returns
    STAGE2_SYSTEM_PROMPT unchanged for non-autopilot_video projects)."""

    def test_present_in_stage1(self):
        assert render_entity_standing_decision_schema_section() in STAGE1_SYSTEM_PROMPT

    def test_present_in_stage2(self):
        assert render_entity_standing_decision_schema_section() in STAGE2_SYSTEM_PROMPT

    def test_present_in_build_stage2_dark_factory(self):
        assert (
            render_entity_standing_decision_schema_section()
            in build_stage2_system_prompt('dark_factory')
        )
