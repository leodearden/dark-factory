"""Stage 2 no longer authors a mem0 narrative reconstruction (task 3734).

That write landed in a store ``get_cycle_summary_presence`` never reads, so it
could never close the ledger-raised finding it was meant to repair.  These
tests pin the ABSENCE of the retired instruction through two tokens that cannot
occur incidentally; the prose of the replacement rule is deliberately not
pinned, per the house norm (``test_recon_gate_closure_guidance.py``,
``test_duplicate_finding_salvage_guidance.py``).
"""

from __future__ import annotations

import pytest

from fused_memory.reconciliation.policies.autopilot_video import AUTOPILOT_VIDEO_PROJECT_ID
from fused_memory.reconciliation.prompts.stage2 import build_stage2_system_prompt
from fused_memory.reconciliation.recon_pool_map import CYCLE_SUMMARY_RECORD_TYPE_NARRATIVE

_BOTH_BUILDER_BRANCHES = pytest.mark.parametrize(
    'project_id', ['dark_factory', AUTOPILOT_VIDEO_PROJECT_ID],
)


class TestStage2NeverAuthorsANarrativeReconstruction:
    @_BOTH_BUILDER_BRANCHES
    def test_prompt_never_names_the_narrative_record_type(self, project_id: str):
        """The quoted literal can only be the record_type value; the bare word
        'narrative' occurs incidentally elsewhere in the prompt."""
        forbidden = f"'{CYCLE_SUMMARY_RECORD_TYPE_NARRATIVE}'"

        assert forbidden not in build_stage2_system_prompt(project_id), (
            f'build_stage2_system_prompt({project_id!r}) names the retired '
            f'record_type {forbidden}; a prompt naming the value can instruct '
            'the LLM to write it.'
        )

    @_BOTH_BUILDER_BRANCHES
    def test_prompt_carries_no_dedup_defeating_retry_nonce(self, project_id: str):
        assert 'retry_nonce' not in build_stage2_system_prompt(project_id), (
            f'build_stage2_system_prompt({project_id!r}) still carries the '
            'retired dedup-defeating retry_nonce instruction (task 3734).'
        )
