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

from fused_memory.reconciliation.cli_stage_runner import STAGE3_DISALLOWED
from fused_memory.reconciliation.prompts.stage1 import STAGE1_SYSTEM_PROMPT
from fused_memory.reconciliation.prompts.stage2 import STAGE2_SYSTEM_PROMPT
from fused_memory.reconciliation.prompts.stage3 import STAGE3_SYSTEM_PROMPT


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


class TestStage3VerifierPrompt:
    """Stage 3 must list the new count tool and document two-path cycle-summary verification."""

    def test_stage3_prompt_lists_count_memories_by_metadata_tool(self):
        """STAGE3_SYSTEM_PROMPT must list mcp__fused-memory__count_memories_by_metadata
        under Available Tools so the agent knows it can use it."""
        assert 'mcp__fused-memory__count_memories_by_metadata' in STAGE3_SYSTEM_PROMPT, (
            "STAGE3_SYSTEM_PROMPT must include 'mcp__fused-memory__count_memories_by_metadata' "
            "in the Available Tools section so Stage 3 knows the tool is accessible."
        )

    def test_stage3_prompt_contains_cycle_summary_verification_section(self):
        """STAGE3_SYSTEM_PROMPT must contain a two-path cycle-summary verification section."""
        prompt = STAGE3_SYSTEM_PROMPT
        # Must mention semantic search as Path 1
        assert 'cycle' in prompt.lower() and 'summary' in prompt.lower(), (
            "STAGE3_SYSTEM_PROMPT must reference cycle summary verification."
        )
        # Must mention the metadata-keyed query with kind='cycle_summary' and run_id
        assert 'cycle_summary' in prompt, (
            "STAGE3_SYSTEM_PROMPT must reference {'kind':'cycle_summary'} in the "
            "metadata-keyed verification path."
        )
        assert 'count_memories_by_metadata' in prompt, (
            "STAGE3_SYSTEM_PROMPT must reference count_memories_by_metadata for Path 2."
        )

    def test_stage3_prompt_declares_both_paths_empty_rule(self):
        """STAGE3_SYSTEM_PROMPT must state the summary is declared missing only when
        BOTH paths return nothing (or equivalent wording)."""
        prompt = STAGE3_SYSTEM_PROMPT
        # The prompt must explicitly mention that BOTH paths must come up empty
        # before reporting the summary as missing.
        has_both_rule = (
            'both' in prompt.lower()
            and ('path' in prompt.lower() or 'search' in prompt.lower())
        )
        assert has_both_rule, (
            "STAGE3_SYSTEM_PROMPT must state that the summary is declared missing only "
            "when BOTH the semantic search path AND the metadata-count path return nothing."
        )

    def test_stage3_disallowed_does_not_contain_count_memories_by_metadata(self):
        """Regression guard: mcp__fused-memory__count_memories_by_metadata must NOT be
        in STAGE3_DISALLOWED so it stays auto-allowed in Stage 3's read-only mode."""
        assert 'mcp__fused-memory__count_memories_by_metadata' not in STAGE3_DISALLOWED, (
            "'mcp__fused-memory__count_memories_by_metadata' must NOT be in STAGE3_DISALLOWED — "
            "it is a read-only tool that Stage 3 must be able to use for metadata-keyed "
            "cycle-summary existence checks."
        )


class TestStage1PreCheckExtension:
    """Stage 1 'Pre-Check: Already-Reconstructed Stage 2 Summaries' block must be
    extended with the metadata-keyed Path 2 and the both-empty rule."""

    def test_stage1_precheck_references_count_memories_by_metadata(self):
        """STAGE1_SYSTEM_PROMPT must reference mcp__fused-memory__count_memories_by_metadata
        in the 'Pre-Check: Already-Reconstructed Stage 2 Summaries' block."""
        assert 'count_memories_by_metadata' in STAGE1_SYSTEM_PROMPT, (
            "STAGE1_SYSTEM_PROMPT must reference 'count_memories_by_metadata' in the "
            "Pre-Check: Already-Reconstructed Stage 2 Summaries section to document "
            "the metadata-keyed second verification path."
        )

    def test_stage1_precheck_references_cycle_summary_metadata_key(self):
        """STAGE1_SYSTEM_PROMPT must reference {'kind':'cycle_summary','run_id':<run_id>}
        as the filter for the metadata-keyed Path 2 in the pre-check."""
        assert 'cycle_summary' in STAGE1_SYSTEM_PROMPT, (
            "STAGE1_SYSTEM_PROMPT must reference 'cycle_summary' as the metadata kind "
            "filter for the metadata-keyed pre-check path."
        )

    def test_stage1_precheck_declares_both_paths_empty_rule(self):
        """STAGE1_SYSTEM_PROMPT pre-check must state that the missing-summary finding
        is emitted only when BOTH the semantic search AND the metadata count come up empty."""
        prompt = STAGE1_SYSTEM_PROMPT
        # Find the pre-check section
        precheck_idx = prompt.find('Pre-Check: Already-Reconstructed Stage 2 Summaries')
        assert precheck_idx != -1, (
            "STAGE1_SYSTEM_PROMPT must contain the "
            "'Pre-Check: Already-Reconstructed Stage 2 Summaries' block."
        )
        precheck_text = prompt[precheck_idx:]
        has_both_rule = (
            'both' in precheck_text.lower()
            and ('path' in precheck_text.lower() or 'empty' in precheck_text.lower())
        )
        assert has_both_rule, (
            "The Stage 1 pre-check block must state that a missing-summary finding is "
            "emitted only when BOTH the semantic search and the metadata count return nothing."
        )

    def test_stage1_precheck_instructs_tool_error_skip_reconstruction(self):
        """STAGE1_SYSTEM_PROMPT must instruct that a count_memories_by_metadata tool error
        means inconclusive — do not reconstruct (conservative/fail-safe semantics)."""
        prompt = STAGE1_SYSTEM_PROMPT
        precheck_idx = prompt.find('Pre-Check: Already-Reconstructed Stage 2 Summaries')
        assert precheck_idx != -1
        precheck_text = prompt[precheck_idx:]
        # The precheck must say something about tool error → skip reconstruction
        has_error_caveat = (
            'error' in precheck_text.lower()
            and ('inconclusive' in precheck_text.lower() or 'skip' in precheck_text.lower()
                 or 'not reconstruct' in precheck_text.lower()
                 or 'do not reconstruct' in precheck_text.lower())
        )
        assert has_error_caveat, (
            "Stage 1 pre-check must instruct that a count tool error means inconclusive "
            "and that reconstruction should be skipped (fail-safe: avoid false-positive "
            "reconstruction)."
        )
