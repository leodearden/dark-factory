"""Prompt↔guard agreement for the canonical gate-subject key (task 3588).

NARROW SCOPE, deliberately. This module asserts ONE coupling: the recon
prompts tell the agent which metadata key to WRITE, and
``middleware/recurring_gate_guard.py`` READS that key at the ``submit_task``
boundary. If the two ever disagree the guard silently stops deduping, so the
agreement is load-bearing and is expressed here against the guard's own
imported symbols (``GATE_SUBJECT_KEY`` / ``GATE_SUBJECT_ALIASES``) rather
than against literals copied into the test.

DO NOT REINTRODUCE PROSE PINS. An earlier revision of this file locked
emphasis tokens and phrasings ('MUST', 'rejected', 'non-terminal', 'cancel',
'AMEND', 'read-side', 'recurrence_count', 'append=True', 'updated_task') —
cosmetic detail that breaks on a semantically neutral reword while catching
nothing that could regress the guard. Reviewer finding
(reviewer_comprehensive, prompt-wording-meta-test) removed them under this
repo's standing rule against documentation meta-tests; the prompt wording is
already correct and does not need a lock. The surviving assertions are either
symbol-coupled (above) or structural (the region-slicing sentinel).

COVERAGE LEDGER for the removed pins, so nobody re-adds one believing it was
the only guard:
- ``execution_class='operational'`` / ``metadata.operational_mode='gate'`` —
  still pinned in BOTH can_file_tasks modes by test_recon_self_model.py::
  TestRenderSourceCompletionSection::
  test_both_modes_state_operational_gate_filing_vocabulary, and on both
  ASSEMBLED prompts by test_operational_routing_boundary_matrix.py::
  test_recon_stage_prompts_carry_source_completion_directives.
- ``'`submit_task`' not in stage1_section`` — still pinned by that same
  boundary-matrix test.
- ``'`update_task`' not in stage1_section`` — GENUINELY LOST, accepted. The
  pin was backtick-only (an unbackticked `update_task` slipped straight
  through it) and it duplicated a norm already stated in
  render_source_completion_section's own docstring. Re-adding it in any
  tighter form is exactly what the review forbade; no follow-up is filed.
- Everything else ('## Consolidation Gate', set_task_status,
  '### Live-Workflow Signals', 'MUST', 'rejected', 'non-terminal', 'cancel',
  'AMEND', 'read-side', 'recurrence_count', 'append=True', 'updated_task') —
  prose this change never needed to own. Nothing to preserve.

WHY THE KEY EXISTS. The carrier→subject linkage key was LLM-invented and
inconsistent — reify carriers used ``stranded_task_id``, dark-factory 3463
used ``related_task_id`` — so no deterministic consumer could join a gate to
its subject, and subject 5879 accumulated carriers 5902 → 5916 → 5929.
"""

from __future__ import annotations

import pytest

from fused_memory.middleware.recurring_gate_guard import (
    GATE_SUBJECT_ALIASES,
    GATE_SUBJECT_KEY,
)
from fused_memory.reconciliation import recon_self_model as m


class TestSourceCompletionDeclaresGateSubject:
    """render_source_completion_section names the key the guard reads."""

    @pytest.mark.parametrize('can_file', [True, False])
    def test_names_the_canonical_metadata_key(self, can_file):
        """The prompt must name the exact key the guard resolves first.

        Asserted against the guard's own constant, so renaming
        GATE_SUBJECT_KEY without re-rendering the prompt fails here.
        """
        text = m.render_source_completion_section(can_file_tasks=can_file)
        assert GATE_SUBJECT_KEY in text, (
            f'can_file_tasks={can_file} must name the canonical key '
            f'{GATE_SUBJECT_KEY!r} that recurring_gate_guard reads'
        )

    def test_filing_stage_names_every_alias_the_guard_resolves(self):
        """Stage 2 is the stage that actually files, so it is the one that
        needs to know the full resolution order.

        Driven off GATE_SUBJECT_ALIASES: adding a new alias to the guard
        tuple fails this test until the prompt names it too.
        """
        text = m.render_source_completion_section(can_file_tasks=True)
        missing = [alias for alias in GATE_SUBJECT_ALIASES if alias not in text]
        assert not missing, (
            f'the filing stage must name every alias the guard resolves; '
            f'missing {missing!r}'
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
        """Structural, not prose: `_live_workflow_region` slices on this
        sentinel, so a duplicate header would silently make every other
        region-scoped assertion read the wrong region."""
        from fused_memory.reconciliation.prompts.stage2 import STAGE2_SYSTEM_PROMPT

        assert STAGE2_SYSTEM_PROMPT.count(_REGION_HEADER) == 1

    def test_region_identifies_the_carrier_by_the_key_the_guard_reads(self):
        from fused_memory.reconciliation.prompts.stage2 import STAGE2_SYSTEM_PROMPT

        region = _live_workflow_region(STAGE2_SYSTEM_PROMPT)
        assert GATE_SUBJECT_KEY in region, (
            f'the region must identify the carrier by {GATE_SUBJECT_KEY!r}, '
            f'the key recurring_gate_guard resolves'
        )

    def test_region_names_update_task_as_the_amend_path(self):
        from fused_memory.reconciliation.prompts.stage2 import STAGE2_SYSTEM_PROMPT

        region = _live_workflow_region(STAGE2_SYSTEM_PROMPT)
        assert '`update_task`' in region, 'the region must name the amend tool'

    def test_rule_reaches_the_real_consumer(self):
        from fused_memory.reconciliation.prompts.stage2 import (
            build_stage2_system_prompt,
        )

        # The non-autopilot passthrough is what the CLI stage runner
        # actually hands the agent.
        region = _live_workflow_region(build_stage2_system_prompt('dark_factory'))
        assert GATE_SUBJECT_KEY in region
        assert 'update_task' in region
