"""Composition-contract tests for the gate-owned finding-phrasing norm (task 4814).

The norm is one half of a two-halves fix: the Stage-1 prompt tells the model
how to phrase a gate-owned finding's ``suggested_action``, and
``normalize_gate_owned_suggested_actions`` deterministically writes the same
sentence when the model does not.  Both halves quote ONE constant,
``gate_owned_finding_phrasing.CANONICAL_HUMAN_GATE_ACTION`` (INV-5), so the
correction an operator reads can never be wording the stage was never given.

These tests pin the WIRING, not the prose — the same shape as
``test_recon_gate_closure_guidance.py``'s ``TestGateClosureArchiveGuidance``.
That module (and ``test_duplicate_finding_salvage_guidance.py``) each record a
prose substring pin that was tried and deliberately REMOVED because it passed
green on a reworded mis-instruction and failed red on a harmless reword.
Prose may be reworded freely here; the composition may not silently break.

Two composition facts are pinned:

1. The rendered norm reaches the assembled ``STAGE1_SYSTEM_PROMPT`` exactly
   once, carrying the canonical sentence with it — and reaches NO other stage
   prompt (plan design decision: Stage 1 only).
2. The prompt half and the code half of the ``GATE_RESOLUTION_FLAG_TYPE``
   carve-out agree.  They are pinned together, at the seam where a future edit
   could break only one of them.
"""

from __future__ import annotations

import pytest

from fused_memory.reconciliation.curator_gate_resolution_sweep import (
    GATE_RESOLUTION_FLAG_TYPE,
    build_gate_resolution_flag,
)
from fused_memory.reconciliation.gate_owned_finding_phrasing import (
    CANONICAL_HUMAN_GATE_ACTION,
    normalize_gate_owned_suggested_actions,
    render_gate_owned_action_norm,
    stamp_curator_gate_sweep_provenance,
)
from fused_memory.reconciliation.prompts.stage1 import STAGE1_SYSTEM_PROMPT
from fused_memory.reconciliation.prompts.stage2 import (
    STAGE2_SYSTEM_PROMPT,
    build_stage2_system_prompt,
)
from fused_memory.reconciliation.prompts.stage3 import STAGE3_SYSTEM_PROMPT

#: Named here so the scope assertions below read against a constant rather
#: than repeating the rationale in every message.
_STAGE1_ONLY_RATIONALE = (
    'plan design decision (task 4814): the norm is Stage 1 ONLY. Stage 2 is '
    'the stage that legitimately holds set_task_status/submit_task, and a '
    'section telling IT that gate-owned findings are awaiting human sign-off '
    'only sits one reword away from contradicting both '
    "build_gate_resolution_flag's instruction to transcribe a recorded ruling "
    "and render_source_completion_section's instruction to FILE the gate. If "
    'you are pasting it into another stage on purpose, revisit that decision '
    'first.'
)


class TestGateOwnedActionNormWiring:
    """The rendered norm reaches Stage 1's assembled prompt, and only Stage 1."""

    def test_present_in_stage1_exactly_once(self):
        """One copy in the assembled Stage-1 prompt — never a second pasted one.

        The exactly-once form is copied from
        ``test_recon_gate_closure_guidance.py``'s
        ``test_appears_exactly_once_per_prompt``: a second copy means the
        section was pasted separately instead of interpolated from its single
        renderer.
        """
        section = render_gate_owned_action_norm()

        assert section in STAGE1_SYSTEM_PROMPT, (
            'STAGE1_SYSTEM_PROMPT must carry render_gate_owned_action_norm() '
            'verbatim — without it the probabilistic half of task 4814 is not '
            'wired at all, and only the code gate is enforcing the rule'
        )
        assert STAGE1_SYSTEM_PROMPT.count(section) == 1, (
            'STAGE1_SYSTEM_PROMPT must carry the norm exactly once; a second '
            'copy means it was pasted rather than interpolated from its renderer'
        )

    def test_canonical_sentence_reaches_the_assembled_stage1_prompt(self):
        """Prompt and code gate quote the SAME sentence.

        This is the property that makes the deterministic normalizer's output
        consistent with what Stage 1 was told: an operator reading a corrected
        finding sees wording the stage actually received.
        """
        assert CANONICAL_HUMAN_GATE_ACTION in STAGE1_SYSTEM_PROMPT, (
            'CANONICAL_HUMAN_GATE_ACTION must survive prompt assembly — it is '
            'the one text both halves of task 4814 quote (INV-5), and if only '
            'the code half carries it, a corrected finding shows the operator '
            'wording the stage was never given'
        )

    def test_absent_from_stage2(self):
        """SCOPE PIN: the norm is deliberately not given to Stage 2."""
        assert render_gate_owned_action_norm() not in STAGE2_SYSTEM_PROMPT, (
            f'STAGE2_SYSTEM_PROMPT must NOT carry the norm — {_STAGE1_ONLY_RATIONALE}'
        )

    @pytest.mark.parametrize('project_id', ['dark_factory', 'autopilot_video'])
    def test_absent_from_both_build_stage2_branches(self, project_id: str):
        """Neither runtime Stage-2 branch may acquire the norm.

        ``autopilot_video`` injects an extra guardrail section at the
        ``## Available Tools`` sentinel; the scope pin must hold on that branch
        too, not just on the module-level constant.
        """
        built = build_stage2_system_prompt(project_id)

        assert render_gate_owned_action_norm() not in built, (
            f'build_stage2_system_prompt({project_id!r}) must NOT carry the '
            f'norm — {_STAGE1_ONLY_RATIONALE}'
        )

    def test_absent_from_stage3(self):
        """SCOPE PIN: the norm is deliberately not given to Stage 3."""
        assert render_gate_owned_action_norm() not in STAGE3_SYSTEM_PROMPT, (
            f'STAGE3_SYSTEM_PROMPT must NOT carry the norm — {_STAGE1_ONLY_RATIONALE}'
        )

    def test_new_section_reintroduces_no_banned_token(self):
        """ANTI-REGRESSION on the assembled Stage-1 prompt.

        Re-asserted here (not only on the section in isolation) because these
        are properties of the ASSEMBLED text, and this suite is what runs when
        the section's wiring changes.
        """
        for token, why in (
            ('escalate_blocker', 'test_stages.py bans the bare substring in Stage 1/3'),
            ('{{', 'test_stage1_consolidation_guidance.py bans surviving doubled braces'),
            ('}}', 'test_stage1_consolidation_guidance.py bans surviving doubled braces'),
            ('as_submit_task_kwargs',
             'test_recon_self_model.py bans this from the whole Stage-1 prompt'),
            ('write_entity_standing_decision',
             'test_recon_report_guidance_drift.py bans this Stage-1-denied tool'),
        ):
            assert token not in STAGE1_SYSTEM_PROMPT, (
                f'STAGE1_SYSTEM_PROMPT reintroduced {token!r}: {why}'
            )

    @pytest.mark.parametrize('key', ['source_finding_id', 'related_memory_ids'])
    def test_new_section_contributes_zero_provenance_key_occurrences(self, key: str):
        """The norm must not perturb the provenance derived-count invariant.

        ``test_finding_provenance_prompt_guidance.py`` pins
        ``prompt.count(key) == shared.count(key) + section.count(key)`` over
        the Stage-1 prompt.  Any mention of either key in a NEW section breaks
        that identity, so the section is asserted to contribute exactly zero
        — a stronger and more local statement than re-deriving the whole sum.
        """
        assert render_gate_owned_action_norm().count(key) == 0, (
            f'the gate-owned norm must not mention {key!r}: '
            'test_finding_provenance_prompt_guidance.py pins a DERIVED-COUNT '
            'invariant over the Stage-1 prompt that any new occurrence breaks'
        )


class TestGateOwnedNormDoesNotContradictTheCuratorGateFlag:
    """The prompt half and the code half of the carve-out agree.

    ``build_gate_resolution_flag`` deliberately tells Stage 2 to set a gate
    task's status — legitimately, because its evidence is a ruling a human
    curator ALREADY recorded in Mem0, so Stage 2 transcribes a decision rather
    than making one.  Without the carve-out, task 4814 would ship a rule that
    contradicts a live, deliberately-designed sibling flag on its very first
    cycle.

    The two halves are pinned TOGETHER here, at the seam, because a future
    edit could easily break only one of them: dropping the flag type from the
    prompt would leave the model believing the rule is unconditional, while
    dropping the code exemption would have the normalizer overwrite the
    sibling flag's instruction.

    The code half additionally requires sweep PROVENANCE, because
    ``GATE_RESOLUTION_FLAG_TYPE`` is a free-form value an LLM-authored finding
    may pick for itself; ``test_gate_owned_finding_phrasing.py`` pins that an
    unstamped finding carrying it IS corrected.
    """

    def test_prompt_half_names_the_carve_out_flag_type(self):
        """The rendered norm names the exempt flag type, by its imported constant."""
        assert GATE_RESOLUTION_FLAG_TYPE in render_gate_owned_action_norm(), (
            f'the norm must name the carve-out flag type ({GATE_RESOLUTION_FLAG_TYPE!r}) '
            'so the stage is told the rule is not unconditional'
        )

    def test_carve_out_survives_prompt_assembly(self):
        """The carve-out reaches the assembled prompt, not just the renderer."""
        assert GATE_RESOLUTION_FLAG_TYPE in STAGE1_SYSTEM_PROMPT, (
            'the carve-out flag type must survive Stage-1 prompt assembly'
        )

    def test_code_half_leaves_the_real_curator_gate_flag_untouched(self):
        """End-to-end against the REAL builder, not a hand-written fixture.

        Routed through the real ``stamp_curator_gate_sweep_provenance``, as
        Stage 1 does at the one site that appends the sweep's output: the
        exemption requires that provenance and not merely the free-form
        ``flag_type`` string an LLM-authored finding may also pick.
        """
        flag = stamp_curator_gate_sweep_provenance(
            [build_gate_resolution_flag('645', [{'id': 'mem-a'}])],
        )[0]
        original = dict(flag)

        result, count = normalize_gate_owned_suggested_actions([flag], ['645'])

        assert count == 0, (
            f'the {GATE_RESOLUTION_FLAG_TYPE} carve-out must not be normalized, '
            f'got count={count!r}'
        )
        assert result[0] == original, (
            'the real curator-gate-resolution flag must be returned unchanged — '
            'prepending "no stage may record a decision" to a flag whose whole '
            'purpose is to transcribe a recorded human ruling would contradict it'
        )
        assert CANONICAL_HUMAN_GATE_ACTION not in result[0]['suggested_action'], (
            'the canonical sentence must not leak into the carved-out flag'
        )
