"""Pins on the Live-Workflow reading rules as DELIVERED to the recon stage LLMs.

The rules are rendered by the module that owns the section's vocabulary
(``reconciliation/live_workflow_section.py::render_live_workflow_authority_rules``),
so these tests pin tokens taken from that module's constants and the
identifiers an agent must act on, never the prose wording.
"""

from __future__ import annotations

import pytest

from fused_memory.reconciliation.live_workflow_section import (
    CLAIMANT_FIELD,
    PROJECT_LOCK_HELD,
    LandedToken,
    render_live_workflow_authority_rules,
)
from fused_memory.reconciliation.prompts.stage1 import STAGE1_SYSTEM_PROMPT
from fused_memory.reconciliation.prompts.stage2 import STAGE2_SYSTEM_PROMPT
from fused_memory.reconciliation.prompts.stage3 import STAGE3_SYSTEM_PROMPT
from fused_memory.services.live_workflow_detector import ClaimantLabel


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


class TestTieBreakerRule:
    @pytest.mark.parametrize('identifier', ['get_task', 'claimant_run_id', 'heartbeat_at'])
    def test_names_the_authoritative_check(self, identifier: str) -> None:
        assert identifier in render_live_workflow_authority_rules(), (
            'the tie-breaker must name the tool and both record fields that settle liveness'
        )

    @pytest.mark.parametrize('label', list(ClaimantLabel))
    def test_quotes_every_claimant_token_the_section_renders(self, label: ClaimantLabel) -> None:
        assert f'{CLAIMANT_FIELD}{label}' in render_live_workflow_authority_rules()

    def test_quotes_the_project_line_the_section_renders(self) -> None:
        assert PROJECT_LOCK_HELD in render_live_workflow_authority_rules()
