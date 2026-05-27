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

    # ── Step 3: dedup folding on recurrence ─────────────────────────────

    def test_recurring_finding_folds_into_one_parent(self, tmp_path):
        """Calling _escalate 3x with the same finding produces 1 file with dedupe_count==2.

        RED before impl: queue.submit() does not fold duplicates, so 3 files are created.
        """
        harness, queue_dir = _make_dedup_harness(tmp_path)

        finding = {
            'category': 'missing_knowledge',
            'affected_ids': ['452'],
            'description': 'Task 452 lacks completion summary',
        }

        for _ in range(3):
            harness._escalate(
                'recon_integrity_issue',
                run_id='aaaa0001',
                summary='Non-actionable integrity finding: missing knowledge for task 452',
                finding=finding,
            )

        files = list(queue_dir.glob('esc-*.json'))
        assert len(files) == 1, (
            f'Expected 1 pending file (deduped), got {len(files)}'
        )

        data = json.loads(files[0].read_text())
        assert data['dedupe_count'] == 2, (
            f"Expected dedupe_count==2 (2 children folded), got {data['dedupe_count']}"
        )

    def test_distinct_findings_stay_distinct(self, tmp_path):
        """Two calls with DIFFERENT affected_ids produce 2 separate files (not folded).

        Remains GREEN after impl: distinct fingerprints → no dedup parent → 2 submits.
        """
        harness, queue_dir = _make_dedup_harness(tmp_path)

        harness._escalate(
            'recon_integrity_issue',
            run_id='bbbb0001',
            summary='Non-actionable integrity finding: missing knowledge for task 452',
            finding={
                'category': 'missing_knowledge',
                'affected_ids': ['452'],
                'description': 'Task 452 lacks completion summary',
            },
        )
        harness._escalate(
            'recon_integrity_issue',
            run_id='bbbb0002',
            summary='Non-actionable integrity finding: missing knowledge for task 361',
            finding={
                'category': 'missing_knowledge',
                'affected_ids': ['361'],
                'description': 'Task 361 lacks completion summary',
            },
        )

        files = list(queue_dir.glob('esc-*.json'))
        assert len(files) == 2, (
            f'Expected 2 distinct pending files, got {len(files)}'
        )

    # ── Step 5: non-finding categories fold on summary ──────────────────

    @pytest.mark.parametrize('category', [
        # Three non-finding-only categories plus recon_integrity_issue (which has
        # finding-aware paths but also except-arm paths called WITHOUT a finding,
        # e.g. "Remediation orchestration failed" and "Remediation pass failed").
        # All four must fold on summary when finding=None.
        'recon_failure',
        'recon_stale_run',
        'recon_backlog_overflow',
        'recon_integrity_issue',
    ])
    def test_non_finding_categories_dedup_on_summary(self, tmp_path, category):
        """Non-finding recon categories fold identical summaries and keep distinct ones.

        Two calls with the same summary → 1 file with dedupe_count==1.
        One more call with a different summary → 2 files total.

        Verifies that _RECON_DEDUP_CONFIG covers all four recon categories and
        the summary-fallback branch in _escalate is wired correctly.
        """
        harness, queue_dir = _make_dedup_harness(tmp_path)

        repeated_summary = 'Stage memory_consolidator failed: timeout'

        # Two identical summary calls → should fold to 1 parent
        harness._escalate(category, run_id='aaaa1111', summary=repeated_summary, detail='')
        harness._escalate(category, run_id='aaaa1112', summary=repeated_summary, detail='')

        files = list(queue_dir.glob('esc-*.json'))
        assert len(files) == 1, (
            f'[{category}] Expected 1 pending file after 2 identical-summary calls, '
            f'got {len(files)}'
        )
        data = json.loads(files[0].read_text())
        assert data['dedupe_count'] == 1, (
            f'[{category}] Expected dedupe_count==1, got {data["dedupe_count"]}'
        )

        # A third call with a DIFFERENT summary → should stay distinct (2 files)
        harness._escalate(
            category, run_id='aaaa1113',
            summary='Stage task_knowledge_sync failed: timeout',
            detail='',
        )

        files = list(queue_dir.glob('esc-*.json'))
        assert len(files) == 2, (
            f'[{category}] Expected 2 pending files after distinct-summary call, '
            f'got {len(files)}'
        )

    # ── Step 7: call-site threading in _maybe_remediate / _run_remediation_pass ─

    @pytest.mark.asyncio
    async def test_maybe_remediate_passes_finding_to_escalate(self, tmp_path):
        """_maybe_remediate non-actionable loop uses finding-based fingerprint.

        RED before impl: _maybe_remediate calls _escalate without finding=, so the
        fingerprint is a summary hash instead of the per-target finding hash.
        """
        from datetime import UTC, datetime

        from escalation.dedupe import compute_content_fingerprint
        from fused_memory.models.reconciliation import (
            ReconciliationRun,
            RunStatus,
            RunType,
        )
        from fused_memory.reconciliation.harness import TierConfig

        harness, queue_dir = _make_dedup_harness(tmp_path)

        finding = {
            'category': 'memory_stale',
            'affected_ids': ['m1'],
            'description': 'd1',
            'actionable': False,
            'severity': 'minor',
        }
        parent_run = ReconciliationRun(
            id='parent-run-id',
            project_id='test_project',
            run_type=RunType.full,
            trigger_reason='buffer',
            started_at=datetime.now(UTC),
            events_processed=0,
            status=RunStatus.running,
            stage_reports={'integrity_check': {'items_flagged': [finding]}},
        )

        await harness._maybe_remediate(
            'test_project', 'parent-run-id', parent_run,
            TierConfig(), project_root='/tmp/x',
        )

        files = list(queue_dir.glob('esc-*.json'))
        assert len(files) == 1, f'Expected 1 pending file, got {len(files)}'

        data = json.loads(files[0].read_text())
        expected_fp = compute_content_fingerprint(
            'recon_integrity_issue', 'memory_stale', ['m1'], 'd1'
        )
        assert data['dedupe_fingerprint'] == expected_fp, (
            f'Expected finding-based fingerprint {expected_fp!r}, '
            f'got {data.get("dedupe_fingerprint")!r} (likely summary-based before impl)'
        )

    @pytest.mark.asyncio
    async def test_remediation_residue_passes_finding_to_escalate(self, tmp_path):
        """_run_remediation_pass residue loop uses finding-based fingerprint.

        RED before impl: _run_remediation_pass calls _escalate without finding=, so
        the fingerprint is summary-based rather than keyed on the finding identity.
        """
        from datetime import UTC, datetime
        from unittest.mock import AsyncMock as AM

        from escalation.dedupe import compute_content_fingerprint
        from fused_memory.models.reconciliation import StageId, StageReport
        from fused_memory.reconciliation.harness import TierConfig

        harness, queue_dir = _make_dedup_harness(tmp_path)
        # Override journal with AsyncMock so start_run/complete_run/etc. are awaitable
        harness.journal = AM()
        harness.journal.get_watermark = AM(return_value=None)
        harness.journal.write_journal = AM()
        # Make instance_id a plain string (ReconciliationRun expects str | None)
        harness.buffer.instance_id = 'test-instance'

        residue_finding = {
            'category': 'knowledge_stale',
            'affected_ids': ['t99'],
            'description': 'Task 99 stale after remediation',
        }
        now = datetime.now(UTC)

        # Use real stage instances (to pass isinstance checks in _run_remediation_pass)
        # with their run() methods patched to return quickly.
        stages = harness._make_stages()
        stages[0].run = AM(return_value=StageReport(
            stage=StageId.memory_consolidator, started_at=now, completed_at=now,
        ))
        stages[1].run = AM(return_value=StageReport(
            stage=StageId.task_knowledge_sync, started_at=now, completed_at=now,
        ))
        stages[2].run = AM(return_value=StageReport(
            stage=StageId.integrity_check, started_at=now, completed_at=now,
            items_flagged=[residue_finding],
        ))
        harness._make_stages = lambda: stages

        await harness._run_remediation_pass(
            'test_project', 'parent-run-id',
            findings=[{
                'category': 'missing_knowledge', 'affected_ids': ['t1'],
                'description': 'trigger finding', 'actionable': True,
            }],
            tier=TierConfig(),
            project_root='/tmp/x',
            filtered_task_tree=MagicMock(),  # pre-supply to skip _fetch_filtered_task_tree
        )

        files = list(queue_dir.glob('esc-*.json'))
        assert len(files) == 1, f'Expected 1 residue escalation, got {len(files)}'

        data = json.loads(files[0].read_text())
        expected_fp = compute_content_fingerprint(
            'recon_integrity_issue', 'knowledge_stale', ['t99'],
            'Task 99 stale after remediation',
        )
        assert data['dedupe_fingerprint'] == expected_fp, (
            f'Expected finding-based fingerprint {expected_fp!r}, '
            f'got {data.get("dedupe_fingerprint")!r} (likely summary-based before impl)'
        )
