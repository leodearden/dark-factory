"""Tests for Stage 1 Memory Consolidator payload behaviour and cross-stage prompt contracts.

Covers:
- project_root threading through assemble_payload / _format_assembled_payload
  (TestStage1PayloadThreadsProjectRootLegacy, TestStage1PayloadThreadsProjectRootAssembled)
- project_root omitted when empty (TestStage1PayloadOmitsProjectRootWhenUnset)
- STAGE2_SYSTEM_PROMPT uniqueness_token mechanism exists (task 1473): minimal existence
  check via build_stage2_system_prompt to guard against the section being dropped
  (TestStage2PromptMandatesUniquenessToken)
- A7b: harness._escalate fingerprint stamping and dedup routing
  (TestReconEscalationDedup)
"""

from __future__ import annotations

import json
from pathlib import Path
from unittest.mock import AsyncMock, MagicMock

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
    """Minimal existence check: build_stage2_system_prompt exposes the uniqueness_token mechanism."""

    def test_build_stage2_system_prompt_exposes_uniqueness_token(self):
        """build_stage2_system_prompt('dark_factory') must expose uniqueness_token."""
        from fused_memory.reconciliation.prompts.stage2 import build_stage2_system_prompt

        result = build_stage2_system_prompt('dark_factory')
        assert 'uniqueness_token' in result, (
            "build_stage2_system_prompt('dark_factory') must expose uniqueness_token "
            "(guards against the per-cycle summary uniqueness section being dropped)."
        )


# ---------------------------------------------------------------------------
# A7b: harness._escalate fingerprint stamping + dedup routing
# ---------------------------------------------------------------------------


def _make_dedup_harness(tmp_path: Path, queue_subdir: str = 'recon_esc'):
    """Build a minimal ReconciliationHarness wired to a real EscalationQueue.

    Uses mocked memory/journal/event_buffer so _escalate can be exercised
    without any I/O other than the queue filesystem writes under tmp_path.
    """
    from fused_memory.config.schema import FusedMemoryConfig, ReconciliationConfig
    from fused_memory.reconciliation.harness import ReconciliationHarness
    from escalation.queue import EscalationQueue

    config = FusedMemoryConfig(
        reconciliation=ReconciliationConfig(
            enabled=True,
            judge_enabled=False,  # no Judge; avoids needing real journal for Judge.init
            agent_llm_provider='anthropic',
            agent_llm_model='claude-sonnet-4-20250514',
        )
    )
    harness = ReconciliationHarness(
        memory_service=AsyncMock(),
        taskmaster=None,
        journal=MagicMock(),
        event_buffer=MagicMock(),
        config=config,
        known_projects={},
    )
    queue_dir = tmp_path / queue_subdir
    harness._escalation_queue = EscalationQueue(queue_dir)
    return harness, queue_dir


class TestReconEscalationDedup:
    """A7b: harness._escalate stamps dedupe_fingerprint and routes through submit_or_dedupe."""

    # ── Step 1: fingerprint stamping ────────────────────────────────────

    def test_escalate_stamps_finding_fingerprint(self, tmp_path):
        """_escalate with finding= kwarg stamps dedupe_fingerprint on the Escalation.

        RED before impl: _escalate has no finding param and never sets dedupe_fingerprint.
        """
        from escalation.dedupe import compute_content_fingerprint

        harness, queue_dir = _make_dedup_harness(tmp_path)

        finding = {
            'category': 'missing_knowledge',
            'affected_ids': ['452'],
            'description': 'Task 452 lacks completion summary',
            'actionable': False,
            'severity': 'minor',
        }

        harness._escalate(
            'recon_integrity_issue',
            run_id='abcd1234',
            summary='Non-actionable integrity finding: missing knowledge for task 452',
            detail='...',
            finding=finding,
        )

        files = list(queue_dir.glob('esc-*.json'))
        assert len(files) == 1, f'Expected 1 pending file, got {len(files)}'

        data = json.loads(files[0].read_text())
        expected_fp = compute_content_fingerprint(
            'recon_integrity_issue',
            'missing_knowledge',
            ['452'],
            'Task 452 lacks completion summary',
        )
        assert data['dedupe_fingerprint'] == expected_fp, (
            f"dedupe_fingerprint mismatch: got {data.get('dedupe_fingerprint')!r}, "
            f"expected {expected_fp!r}"
        )
