"""Composition-contract tests for DUPLICATE_FINDING_SALVAGE_GUIDANCE (task 4184).

Every stage can be handed a `duplicate_finding` error: `add_finding` returns it
on a colliding `(task_id, flag_type)` signature, and `cite_task` returns it from
its two in-run cited-task folds (task-2425 project-scoped, task-2432
entity-scoped).  The fold path is the lossy one — the losing finding is purged
wholesale, so the agent's only way to preserve its work is to re-attach its
citations and any distinct context to the `existing_finding_id` the response
names.  That protocol was documented for Stage 1 only; Stages 2 and 3 hit the
same error and were never told.

These tests pin the WIRING — one shared constant, embedded in all three
assembled prompts and surviving both branches of ``build_stage2_system_prompt``
— rather than the prose, following ``test_recon_gate_closure_guidance.py``.
Prose may be reworded freely; the wiring may not silently break.

`.count(...) == 1` (not a bare `in`) is deliberate: it catches omission AND
double-interpolation in one assertion, and subsumes non-emptiness — a constant
accidentally emptied to `''` would make `in` vacuously true against every
prompt.

The single-source property is enforced structurally rather than by a
"text appears once in the tree" scan: because each stage prompt is asserted to
contain THIS constant, re-wording any one stage's copy in place would fail
these tests (INV-5 `no-lockstep-duplication`).
"""

from __future__ import annotations

import pytest

from fused_memory.reconciliation.prompts import DUPLICATE_FINDING_SALVAGE_GUIDANCE
from fused_memory.reconciliation.prompts.stage1 import STAGE1_SYSTEM_PROMPT
from fused_memory.reconciliation.prompts.stage2 import (
    STAGE2_SYSTEM_PROMPT,
    build_stage2_system_prompt,
)
from fused_memory.reconciliation.prompts.stage3 import STAGE3_SYSTEM_PROMPT


class TestDuplicateFindingSalvageGuidance:
    """All three stage prompts carry the salvage protocol exactly once."""

    def test_embedded_exactly_once_in_stage1_prompt(self):
        """Stage 1 — where the protocol originated — now rides the shared constant."""
        assert STAGE1_SYSTEM_PROMPT.count(DUPLICATE_FINDING_SALVAGE_GUIDANCE) == 1, (
            'STAGE1_SYSTEM_PROMPT must interpolate DUPLICATE_FINDING_SALVAGE_GUIDANCE '
            'exactly once — its former inline copy was replaced by the shared constant, '
            'so a second copy means the paragraph was pasted back in.'
        )

    def test_embedded_exactly_once_in_stage2_prompt(self):
        """Stage 2 files remediation tasks and cites them — it hits both folds."""
        assert STAGE2_SYSTEM_PROMPT.count(DUPLICATE_FINDING_SALVAGE_GUIDANCE) == 1, (
            'STAGE2_SYSTEM_PROMPT must interpolate DUPLICATE_FINDING_SALVAGE_GUIDANCE '
            'exactly once.'
        )

    def test_embedded_exactly_once_in_stage3_prompt(self):
        """Stage 3 cites tasks too, and the entity-scoped fold has no stage carve-out."""
        assert STAGE3_SYSTEM_PROMPT.count(DUPLICATE_FINDING_SALVAGE_GUIDANCE) == 1, (
            'STAGE3_SYSTEM_PROMPT must interpolate DUPLICATE_FINDING_SALVAGE_GUIDANCE '
            'exactly once.'
        )

    @pytest.mark.parametrize('project_id', ['dark_factory', 'autopilot_video'])
    def test_survives_build_stage2_system_prompt(self, project_id: str):
        """Both branches of the runtime builder keep the guidance.

        `autopilot_video` injects an extra guardrail section before
        `## Available Tools`; that injection must not displace this guidance.
        """
        built = build_stage2_system_prompt(project_id)

        assert built.count(DUPLICATE_FINDING_SALVAGE_GUIDANCE) == 1, (
            f'build_stage2_system_prompt({project_id!r}) must carry '
            'DUPLICATE_FINDING_SALVAGE_GUIDANCE exactly once.'
        )

    # No substring pin on the guidance prose lives here — an
    # `'existing_finding_id' in ...` / `'cite_task' in ...` assertion was tried
    # and removed, for the same defect-in-both-directions reason recorded at
    # test_recon_gate_closure_guidance.py:186-194. It is wrong whichever way it
    # fires: a faithful reword that names the same API surface differently
    # ("the id the response hands back", "the citation tool") goes red on a
    # correct edit, while a paragraph that drops both tokens in while garbling
    # the salvage protocol goes green — so it cannot detect the failure it
    # purports to guard. Pinning wording inside a module-level prompt string
    # literal is a documentation meta-test, not a behavioural contract.
    #
    # The four wiring assertions above already pin everything that can break
    # SILENTLY: the constant existing and being importable, being interpolated
    # into STAGE1/STAGE2/STAGE3_SYSTEM_PROMPT exactly once each, and surviving
    # both branches of `build_stage2_system_prompt`. The salvage protocol's
    # WORDING is maintained at its single source, prompts/__init__.py:244,
    # under review rather than by assertion.
