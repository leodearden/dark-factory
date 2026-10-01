"""Pins on the Live-Workflow reading rules as DELIVERED to the recon stage LLMs.

The rules are rendered by the module that owns the section's vocabulary
(``reconciliation/live_workflow_section.py::render_live_workflow_authority_rules``),
so these tests pin tokens taken from that module's constants and the
identifiers an agent must act on, never the prose wording.
"""

from __future__ import annotations

from fused_memory.reconciliation.live_workflow_section import (
    LandedToken,
    render_live_workflow_authority_rules,
)
from fused_memory.reconciliation.prompts.stage1 import STAGE1_SYSTEM_PROMPT
from fused_memory.reconciliation.prompts.stage2 import STAGE2_SYSTEM_PROMPT
from fused_memory.reconciliation.prompts.stage3 import STAGE3_SYSTEM_PROMPT


def _paragraphs_containing(text: str, needle: str) -> list[str]:
    return [paragraph for paragraph in text.split('\n\n') if needle in paragraph]


class TestLandedRuleInStagePrompts:
    def test_rules_are_delivered_once_to_each_stage_that_receives_the_section(self) -> None:
        rules = render_live_workflow_authority_rules()

        assert STAGE1_SYSTEM_PROMPT.count(rules) == 1
        assert STAGE2_SYSTEM_PROMPT.count(rules) == 1
        assert rules not in STAGE3_SYSTEM_PROMPT, (
            'Stage 3 never receives the Live-Workflow Signals section, so it must not get rules over it'
        )

    def test_landed_true_forbids_resume_redispatch_and_reset_to_pending(self) -> None:
        paragraphs = _paragraphs_containing(render_live_workflow_authority_rules(), LandedToken.TRUE)

        assert paragraphs, f'no rule quotes the rendered {LandedToken.TRUE!s} token'
        assert any(
            all(word in paragraph for word in ('resume', 'redispatch', 'pending'))
            for paragraph in paragraphs
        ), 'the landed rule must forbid resuming, redispatching and resetting to pending'

    def test_done_provenance_is_the_landing_authority_and_a_refused_reopen_is_not_escalated(
        self,
    ) -> None:
        paragraphs = _paragraphs_containing(render_live_workflow_authority_rules(), 'done_provenance')

        assert any('escalat' in paragraph for paragraph in paragraphs), (
            'the done-task rule must name done_provenance and forbid escalating a refused reopen'
        )
