"""Prompt-guidance tests for the canonical gate-subject key (task 3588).

The recon stage prompts ARE the documentation for their only consumer —
the sleep-mode CLI agents that run each reconciliation stage have no other
source of guidance — so for a prompts module the rendered text IS the
runtime deliverable. This file follows the established
dedicated-prompt-guidance-module convention
(test_recon_amend_tool_advertisement.py, test_recon_gate_closure_guidance.py,
test_finding_provenance_prompt_guidance.py) and the assertion style of
test_recon_self_model.py::TestRenderSourceCompletionSection: assert the
load-bearing literals, not the prose around them.

WHY THIS EXISTS. The carrier→subject linkage key was LLM-invented and
inconsistent — reify carriers used `stranded_task_id`, dark-factory 3463
used `related_task_id` — so no deterministic consumer could join a gate to
its subject, and subject 5879 accumulated carriers 5902 → 5916 → 5929.
"""

from __future__ import annotations

import pytest

from fused_memory.reconciliation import recon_self_model as m


class TestSourceCompletionDeclaresGateSubject:
    """render_source_completion_section names the canonical subject key."""

    @pytest.mark.parametrize('can_file', [True, False])
    def test_names_the_canonical_metadata_key(self, can_file):
        text = m.render_source_completion_section(can_file_tasks=can_file)
        assert 'metadata.gate_subject' in text, (
            f'can_file_tasks={can_file} must name the canonical key'
        )

    @pytest.mark.parametrize('can_file', [True, False])
    def test_states_the_key_is_required_on_a_gate_filing(self, can_file):
        text = m.render_source_completion_section(can_file_tasks=can_file)
        # Scoped to the paragraph that introduces the key, so a stray 'MUST'
        # elsewhere in the section cannot satisfy this.
        para = next(
            (p for p in text.split('\n\n') if 'metadata.gate_subject' in p), ''
        )
        assert para, f'can_file_tasks={can_file} must introduce the key'
        assert 'MUST' in para, (
            f'can_file_tasks={can_file} must state the key is REQUIRED where '
            f'it introduces it; got {para!r}'
        )

    @pytest.mark.parametrize('can_file', [True, False])
    def test_names_the_boundary_rejection_consequence(self, can_file):
        text = m.render_source_completion_section(can_file_tasks=can_file)
        # A hard rejection invariant, not a lint warning — mirroring
        # render_execution_class_section's existing phrasing. Matched
        # case-insensitively: the section's voice uses emphasis caps
        # (MUST/REJECTED), which is style, not contract.
        lowered = text.lower()
        assert 'rejected' in lowered, (
            f'can_file_tasks={can_file} must name the rejection consequence'
        )
        assert 'non-terminal' in lowered, (
            f'can_file_tasks={can_file} must state the rejection applies only '
            f'while the first carrier is non-terminal'
        )

    def test_stage2_variant_names_update_task_as_the_amend_path(self):
        """Scoped to can_file_tasks=True deliberately.

        `update_task` is in DISALLOW_TASK_WRITES alongside `submit_task`, so
        Stage 1 holds NEITHER. render_source_completion_section's own
        docstring states the governing norm — "Never instruct Stage 1 to call
        a tool it does not hold (loud-over-silent)" — and
        test_operational_routing_boundary_matrix.py::
        test_recon_stage_prompts_carry_source_completion_directives enforces
        the `submit_task` half of it. Naming the amend tool in the Stage-1
        variant would violate both.
        """
        text = m.render_source_completion_section(can_file_tasks=True)
        assert 'update_task' in text, 'the filing stage must name the amend path'

    def test_stage1_variant_names_no_tool_it_does_not_hold(self):
        """The inverse pin, so a future edit cannot quietly reintroduce a
        task-write tool name into the relay-only stage."""
        text = m.render_source_completion_section(can_file_tasks=False)
        assert '`update_task`' not in text, f'got {text!r}'
        assert '`submit_task`' not in text, f'got {text!r}'

    def test_stage2_variant_names_the_read_side_aliases_as_legacy(self):
        """Stage 2 is the stage that actually files, so it is the one that
        needs to know the aliases exist AND that they are not for new work."""
        text = m.render_source_completion_section(can_file_tasks=True)
        assert 'stranded_task_id' in text
        assert 'related_task_id' in text
        assert 'read-side' in text, (
            'the aliases must be marked read-side only, so a NEW filing uses '
            'gate_subject'
        )

    @pytest.mark.parametrize('can_file', [True, False])
    def test_change_is_additive_preexisting_literals_survive(self, can_file):
        """The invariants test_recon_self_model.py already pins must survive —
        this section is extended, not rewritten."""
        text = m.render_source_completion_section(can_file_tasks=can_file)
        assert "metadata.operational_mode='gate'" in text
        assert "execution_class='operational'" in text
        assert '## Consolidation Gate' in text


# --------------------------------------------------------------------------- #
# Stage 2 `## Live-Workflow Authority` — amend, never cancel-and-remint
# --------------------------------------------------------------------------- #

_REGION_HEADER = '## Live-Workflow Authority'


def _live_workflow_region(prompt: str) -> str:
    """Return the `## Live-Workflow Authority` region only.

    Sliced between its header and the next `## ` header so every assertion
    below is scoped to the region — a literal appearing somewhere else in
    the (very long) Stage 2 prompt must not satisfy them.
    """
    _, _, after = prompt.partition(_REGION_HEADER)
    end = after.find('\n## ')
    return after if end == -1 else after[:end]


class TestStage2AmendDontRemintRule:
    """A liveness flicker on a gate's subject must never cancel the carrier.

    Cancel-and-remint is what orphaned esc-5881-1 / esc-5902-1 / esc-5916-1
    as permanently-pending L2 escalations, and what produced three carriers
    (5902 -> 5916 -> 5929) for the single subject 5879.
    """

    def test_region_header_appears_exactly_once(self):
        from fused_memory.reconciliation.prompts.stage2 import STAGE2_SYSTEM_PROMPT

        assert STAGE2_SYSTEM_PROMPT.count(_REGION_HEADER) == 1

    def test_region_forbids_cancelling_a_gate_carrier(self):
        from fused_memory.reconciliation.prompts.stage2 import STAGE2_SYSTEM_PROMPT

        region = _live_workflow_region(STAGE2_SYSTEM_PROMPT)
        assert 'gate_subject' in region, (
            'the region must identify the carrier by its canonical key'
        )
        assert 'cancel' in region.lower(), (
            'the region must state the cancel prohibition'
        )

    def test_region_directs_amend_in_place(self):
        from fused_memory.reconciliation.prompts.stage2 import STAGE2_SYSTEM_PROMPT

        region = _live_workflow_region(STAGE2_SYSTEM_PROMPT)
        assert 'update_task' in region, 'the region must name the amend tool'
        assert 'AMEND' in region, 'the region must direct amend-in-place'
        assert 'recurrence_count' in region, (
            'the region must name the counter to bump'
        )

    def test_region_warns_about_the_append_description_hazard(self):
        from fused_memory.reconciliation.prompts.stage2 import STAGE2_SYSTEM_PROMPT

        region = _live_workflow_region(STAGE2_SYSTEM_PROMPT)
        assert 'append=True' in region
        assert 'description' in region
        assert 'updated_task' in region, (
            'the region must tell the agent to verify the echoed updated_task'
        )

    def test_preexisting_rules_survive(self):
        from fused_memory.reconciliation.prompts.stage2 import STAGE2_SYSTEM_PROMPT

        region = _live_workflow_region(STAGE2_SYSTEM_PROMPT)
        # Rule 1 and the payload-section pointer must be untouched — this
        # change is additive.
        assert 'set_task_status' in region
        assert '### Live-Workflow Signals' in region

    def test_rule_reaches_the_real_consumer(self):
        from fused_memory.reconciliation.prompts.stage2 import (
            build_stage2_system_prompt,
        )

        # The non-autopilot passthrough is what the CLI stage runner
        # actually hands the agent.
        region = _live_workflow_region(build_stage2_system_prompt('dark_factory'))
        assert 'gate_subject' in region
        assert 'update_task' in region
