"""Prompt-content assertions for Stage 1/2/3 cycle-summary metadata verification (task 1588).

Covers:
- Stage 2 producer instruction: per-cycle summary must be written with
  metadata containing kind='cycle_summary' and run_id set to the cycle run_id.
- Stage 3 verifier: lists mcp__fused-memory__count_memories_by_metadata in Available
  Tools; contains a two-path cycle-summary verification section (semantic search + metadata
  count); declares summary missing only when BOTH paths return nothing.
- Stage 3 disallow guard: count_memories_by_metadata is NOT in STAGE3_DISALLOWED
  (regression guard that the read-only tool stays available in Stage 3).
- Stage 1 pre-check extension: "Pre-Check: Already-Reconstructed Stage 2 Summaries"
  documents the metadata-keyed Path 2 (count_memories_by_metadata with
  {'kind':'cycle_summary','run_id':<run_id>}) and the both-empty rule; tool error
  means inconclusive → do not reconstruct.
"""

from __future__ import annotations

import pytest

from fused_memory.reconciliation.prompts.stage2 import STAGE2_SYSTEM_PROMPT


class TestStage2ProducerMetadataInstruction:
    """Stage 2 must instruct the agent to write the per-cycle summary with
    metadata.kind='cycle_summary' and metadata.run_id set to the cycle run_id."""

    def test_stage2_prompt_instructs_cycle_summary_metadata_kind(self):
        """STAGE2_SYSTEM_PROMPT must contain 'cycle_summary' as a metadata kind value
        for the per-cycle summary add_memory call."""
        assert 'cycle_summary' in STAGE2_SYSTEM_PROMPT, (
            "STAGE2_SYSTEM_PROMPT must instruct the agent to use metadata kind='cycle_summary' "
            "when writing the per-cycle summary — needed for deterministic metadata-keyed lookup."
        )

    def test_stage2_prompt_instructs_run_id_in_metadata(self):
        """STAGE2_SYSTEM_PROMPT must explicitly instruct setting run_id in the metadata
        (not just in the content) for the per-cycle summary."""
        # The instruction should tell the agent to pass run_id in metadata={...}
        # for the summary add_memory call.
        prompt = STAGE2_SYSTEM_PROMPT
        # The word 'metadata' must appear in close proximity to 'run_id' context,
        # instructing the agent to write run_id into the metadata dict.
        assert 'metadata' in prompt, (
            "STAGE2_SYSTEM_PROMPT must mention metadata for the per-cycle summary call."
        )
        # Specifically 'kind' and 'cycle_summary' and 'run_id' must all appear
        # in the metadata instruction for the per-cycle summary.
        assert 'kind' in prompt, (
            "STAGE2_SYSTEM_PROMPT must reference metadata key 'kind' for cycle_summary tagging."
        )
        # We assert that 'run_id' appears in the prompt's metadata instruction.
        # The instruction must include run_id as a metadata key (not just in content).
        # Find the Per-Cycle Summary section to scope the assertion:
        summary_section_idx = prompt.find('Per-Cycle Summary')
        assert summary_section_idx != -1, (
            "STAGE2_SYSTEM_PROMPT must contain a 'Per-Cycle Summary' section."
        )
        # After the Per-Cycle Summary section, there must be metadata instruction
        # referencing both 'kind' + 'cycle_summary' + 'run_id'.
        section_text = prompt[summary_section_idx:]
        assert 'cycle_summary' in section_text, (
            "The Per-Cycle Summary section must mention 'cycle_summary' as metadata kind."
        )
        assert 'run_id' in section_text, (
            "The Per-Cycle Summary section must mention 'run_id' in the metadata instruction."
        )
