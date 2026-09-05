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

    @pytest.mark.parametrize('can_file', [True, False])
    def test_names_update_task_as_the_amend_path(self, can_file):
        text = m.render_source_completion_section(can_file_tasks=can_file)
        assert 'update_task' in text, (
            f'can_file_tasks={can_file} must name the amend path'
        )

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
