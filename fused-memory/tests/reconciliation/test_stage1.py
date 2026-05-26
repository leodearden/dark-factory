"""Tests for Stage 1 Memory Consolidator payload behaviour and cross-stage prompt contracts.

Covers:
- project_root threading through assemble_payload / _format_assembled_payload
  (TestStage1PayloadThreadsProjectRootLegacy, TestStage1PayloadThreadsProjectRootAssembled)
- project_root omitted when empty (TestStage1PayloadOmitsProjectRootWhenUnset)
- STAGE2_SYSTEM_PROMPT mandates uniqueness_token in every per-cycle summary to defeat
  Mem0 cosine-similarity dedup of zero-flag/zero-task cycles (task 1473)
  (TestStage2PromptMandatesUniquenessToken)
- STAGE1_SYSTEM_PROMPT pre-checks for already-reconstructed Stage 2 summaries before
  emitting a missing-summary finding, preventing double-reconstruction in back-to-back
  remediation passes (task 1473)
  (TestStage1PromptPrechecksReconstructedSummaries)
"""

from __future__ import annotations

from unittest.mock import AsyncMock

import pytest

from fused_memory.config.schema import ReconciliationConfig
from fused_memory.models.reconciliation import (
    AssembledPayload,
    StageId,
    Watermark,
)
from fused_memory.reconciliation.stages.memory_consolidator import MemoryConsolidator


def _make_consolidator(project_root: str = '') -> MemoryConsolidator:
    """Build a MemoryConsolidator with mocked deps — mirrors test_stages.py ~L1418."""
    config = ReconciliationConfig()
    memory_mock = AsyncMock()
    memory_mock.get_episodes = AsyncMock(return_value=[])
    memory_mock.mem0 = AsyncMock()
    memory_mock.mem0.get_all = AsyncMock(return_value={'results': []})
    memory_mock.get_status = AsyncMock(return_value={})

    stage = MemoryConsolidator(
        StageId.memory_consolidator,
        memory_mock,
        AsyncMock(),  # taskmaster
        AsyncMock(),  # journal
        config,
    )
    stage.project_id = 'test_project'
    stage.project_root = project_root
    stage.episode_limit = 5
    stage.memory_limit = 10
    return stage


# ---------------------------------------------------------------------------
# project_root in legacy (time-windowed) assemble_payload
# ---------------------------------------------------------------------------


class TestStage1PayloadThreadsProjectRootLegacy:
    """assemble_payload includes project_root directive when project_root is set."""

    @pytest.mark.asyncio
    async def test_assemble_payload_includes_project_root(self):
        """Legacy assemble_payload emits 'Use project_root=...' when project_root is set."""
        stage = _make_consolidator(project_root='/home/leo/src/test_proj')
        watermark = Watermark(project_id='test_project')

        result = await stage.assemble_payload(
            events=[], watermark=watermark, prior_reports=[]
        )

        assert 'Use project_root="/home/leo/src/test_proj"' in result, (
            'assemble_payload should emit project_root directive when project_root is set'
        )
        assert result.rstrip().endswith(
            'Use project_root="/home/leo/src/test_proj" for tasks scoped to this project.'
        ), 'project_root directive should be the last line of the payload'
        assert result.count('Use project_root=') == 1, (
            'project_root directive should appear exactly once in the payload'
        )


# ---------------------------------------------------------------------------
# project_root in assembled-payload branch (_format_assembled_payload)
# ---------------------------------------------------------------------------


class TestStage1PayloadThreadsProjectRootAssembled:
    """_format_assembled_payload includes project_root directive when project_root is set."""

    @pytest.mark.asyncio
    async def test_format_assembled_payload_includes_project_root(self):
        """Assembled-payload branch emits 'Use project_root=...' when project_root is set."""
        stage = _make_consolidator(project_root='/home/leo/src/test_proj')

        # Set assembled_payload to trigger the ContextAssembler branch
        stage.assembled_payload = AssembledPayload(
            events=[],
            context_items={},
        )

        watermark = Watermark(project_id='test_project')
        result = await stage.assemble_payload(
            events=[], watermark=watermark, prior_reports=[]
        )

        assert 'Use project_root="/home/leo/src/test_proj"' in result, (
            '_format_assembled_payload should emit project_root directive when project_root is set'
        )
        assert result.rstrip().endswith(
            'Use project_root="/home/leo/src/test_proj" for tasks scoped to this project.'
        ), 'project_root directive should be the last line of the payload'
        assert result.count('Use project_root=') == 1, (
            'project_root directive should appear exactly once in the payload'
        )


# ---------------------------------------------------------------------------
# project_root omitted when empty (BaseStage default '')
# ---------------------------------------------------------------------------


class TestStage1PayloadOmitsProjectRootWhenUnset:
    """When project_root is '' (BaseStage default), no project_root line is emitted."""

    @pytest.mark.asyncio
    async def test_assemble_payload_omits_project_root_when_empty(self):
        """Legacy assemble_payload does NOT emit project_root line when project_root=''."""
        stage = _make_consolidator(project_root='')
        watermark = Watermark(project_id='test_project')

        result = await stage.assemble_payload(
            events=[], watermark=watermark, prior_reports=[]
        )

        assert 'Use project_root=' not in result, (
            'assemble_payload should omit project_root directive when project_root is empty'
        )
        assert 'Use project_root=""' not in result

    @pytest.mark.asyncio
    async def test_format_assembled_payload_omits_project_root_when_empty(self):
        """Assembled-payload branch does NOT emit project_root line when project_root=''."""
        stage = _make_consolidator(project_root='')
        stage.assembled_payload = AssembledPayload(
            events=[],
            context_items={},
        )

        watermark = Watermark(project_id='test_project')
        result = await stage.assemble_payload(
            events=[], watermark=watermark, prior_reports=[]
        )

        assert 'Use project_root=' not in result, (
            '_format_assembled_payload should omit project_root directive when project_root is empty'
        )
        assert 'Use project_root=""' not in result


# ---------------------------------------------------------------------------
# Step 1 (task-1473): STAGE2 prompt mandates uniqueness_token in cycle summaries
# ---------------------------------------------------------------------------


class TestStage2PromptMandatesUniquenessToken:
    """STAGE2_SYSTEM_PROMPT mandates a uniqueness_token in every per-cycle summary.

    Covers the fix for zero-flag/zero-task cycle summary deduplication (task 1473):
    a SHA-256-derived 8-hex-char token folds in the ISO timestamp so even empty
    cycles produce a distinct content string, defeating Mem0 cosine-similarity dedup.
    """

    def test_uniqueness_token_literal_present(self):
        """'uniqueness_token' must appear in STAGE2_SYSTEM_PROMPT."""
        from fused_memory.reconciliation.prompts.stage2 import STAGE2_SYSTEM_PROMPT

        assert 'uniqueness_token' in STAGE2_SYSTEM_PROMPT, (
            "STAGE2_SYSTEM_PROMPT must mandate a 'uniqueness_token' field in every "
            "per-cycle summary to defeat Mem0 cosine-similarity dedup of "
            "zero-flag/zero-task cycles."
        )

    def test_sha256_algorithm_named(self):
        """SHA-256 must be named as the derivation algorithm."""
        from fused_memory.reconciliation.prompts.stage2 import STAGE2_SYSTEM_PROMPT

        assert 'sha256' in STAGE2_SYSTEM_PROMPT.lower(), (
            "STAGE2_SYSTEM_PROMPT must specify SHA-256 as the algorithm for computing "
            "uniqueness_token."
        )

    def test_formula_inputs_named(self):
        """iso_timestamp and sorted flag_ids must appear as formula inputs."""
        from fused_memory.reconciliation.prompts.stage2 import STAGE2_SYSTEM_PROMPT

        assert 'iso_timestamp' in STAGE2_SYSTEM_PROMPT, (
            "STAGE2_SYSTEM_PROMPT must name 'iso_timestamp' as a formula input for "
            "uniqueness_token (folding in the timestamp ensures empty cycles differ)."
        )
        assert 'sorted' in STAGE2_SYSTEM_PROMPT, (
            "STAGE2_SYSTEM_PROMPT must reference sorted(flag_ids) as a formula input "
            "for deterministic ordering."
        )

    def test_eight_char_prefix_specified(self):
        """The [:8] first-8-hex-chars truncation must be specified."""
        from fused_memory.reconciliation.prompts.stage2 import STAGE2_SYSTEM_PROMPT

        assert '[:8]' in STAGE2_SYSTEM_PROMPT, (
            "STAGE2_SYSTEM_PROMPT must specify '[:8]' (first 8 hex chars) for "
            "uniqueness_token so the prompt is actionable as a hashlib one-liner."
        )

    def test_build_stage2_system_prompt_exposes_uniqueness_token(self):
        """build_stage2_system_prompt('dark_factory') must also expose uniqueness_token."""
        from fused_memory.reconciliation.prompts.stage2 import build_stage2_system_prompt

        result = build_stage2_system_prompt('dark_factory')
        assert 'uniqueness_token' in result, (
            "build_stage2_system_prompt('dark_factory') must expose uniqueness_token "
            "(it returns STAGE2_SYSTEM_PROMPT unmodified for non-autopilot projects)."
        )


# ---------------------------------------------------------------------------
# Step 3 (task-1473): STAGE1 prompt pre-checks for already-reconstructed summaries
# ---------------------------------------------------------------------------


class TestStage1PromptPrechecksReconstructedSummaries:
    """STAGE1_SYSTEM_PROMPT instructs a pre-check before flagging missing Stage 2 summaries.

    Covers the fix for double-reconstruction in back-to-back remediation passes (task 1473):
    before emitting a 'missing Stage 2 summary' finding for a run, Stage 1 must search
    Mem0 for run_id: <run_id> to confirm no prior remediation pass already wrote the
    summary.  If found, the finding must be suppressed.
    """

    def test_stage2_summary_concept_present(self):
        """The concept of existing Stage 2 summaries must be referenced in STAGE1_SYSTEM_PROMPT."""
        from fused_memory.reconciliation.prompts.stage1 import STAGE1_SYSTEM_PROMPT

        assert 'Stage 2 summary' in STAGE1_SYSTEM_PROMPT, (
            "STAGE1_SYSTEM_PROMPT must reference 'Stage 2 summary' in the context of "
            "the pre-check for already-written summaries from prior remediation passes."
        )

    def test_prior_remediation_concept_present(self):
        """STAGE1_SYSTEM_PROMPT must reference the 'prior remediation pass' concept."""
        from fused_memory.reconciliation.prompts.stage1 import STAGE1_SYSTEM_PROMPT

        assert 'prior remediation' in STAGE1_SYSTEM_PROMPT, (
            "STAGE1_SYSTEM_PROMPT must reference 'prior remediation' pass to explain "
            "why the pre-check is necessary (prevents double-reconstruction)."
        )

    def test_run_id_search_precheck_present(self):
        """STAGE1_SYSTEM_PROMPT must instruct a Mem0 search by run_id before flagging."""
        from fused_memory.reconciliation.prompts.stage1 import STAGE1_SYSTEM_PROMPT

        # The new pre-check section's search query must reference 'run_id:' (with colon),
        # matching the canonical search pattern used in the Flag Suppression Check section.
        assert 'run_id:' in STAGE1_SYSTEM_PROMPT, (
            "STAGE1_SYSTEM_PROMPT must instruct searching for 'run_id: <run_id>' to find "
            "existing Stage 2 summaries before emitting a missing-summary finding."
        )

    def test_do_not_emit_when_found(self):
        """STAGE1_SYSTEM_PROMPT must instruct NOT emitting the finding when a summary is found."""
        from fused_memory.reconciliation.prompts.stage1 import STAGE1_SYSTEM_PROMPT

        assert 'do NOT emit' in STAGE1_SYSTEM_PROMPT or 'double-reconstruction' in STAGE1_SYSTEM_PROMPT, (
            "STAGE1_SYSTEM_PROMPT must instruct NOT emitting the missing-summary finding "
            "when a prior Stage 2 summary for the same run_id is found, preventing "
            "double-reconstruction in back-to-back remediation passes."
        )

