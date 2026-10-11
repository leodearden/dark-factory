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
    render_live_workflow_authority_rules,
)
from fused_memory.reconciliation.prompts.stage1 import STAGE1_SYSTEM_PROMPT
from fused_memory.reconciliation.prompts.stage2 import STAGE2_SYSTEM_PROMPT
from fused_memory.reconciliation.prompts.stage3 import STAGE3_SYSTEM_PROMPT
from fused_memory.services.live_workflow_detector import ClaimantLabel


class TestLandedRuleInStagePrompts:
    def test_rules_are_delivered_once_to_each_stage_that_receives_the_section(self) -> None:
        rules = render_live_workflow_authority_rules()

        assert STAGE1_SYSTEM_PROMPT.count(rules) == 1
        assert STAGE2_SYSTEM_PROMPT.count(rules) == 1
        assert rules not in STAGE3_SYSTEM_PROMPT, (
            'Stage 3 never receives the Live-Workflow Signals section, so it must not get rules over it'
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
