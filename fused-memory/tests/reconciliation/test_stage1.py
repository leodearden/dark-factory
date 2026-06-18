"""Tests for Stage 1 Memory Consolidator payload behaviour and cross-stage prompt contracts.

Covers:
- project_root threading through assemble_payload / _format_assembled_payload
  (TestStage1PayloadThreadsProjectRootLegacy, TestStage1PayloadThreadsProjectRootAssembled)
- project_root omitted when empty (TestStage1PayloadOmitsProjectRootWhenUnset)
- STAGE2_SYSTEM_PROMPT uniqueness_token mechanism exists (task 1473): minimal existence
  check via build_stage2_system_prompt to guard against the section being dropped
  (TestStage2PromptMandatesUniquenessToken)
- CSPRNG summary_nonce injection in per-cycle payloads (task 1574):
  both legacy assemble_payload and assembled _format_assembled_payload paths must inject
  '### Per-Cycle Summary Nonce' with a fresh 8-hex nonce per call
  (TestStage1PayloadSummaryNonce)
- A7b: harness._escalate fingerprint stamping and dedup routing
  (TestReconEscalationDedup)
- Step-11: MemoryConsolidator.run() wiring — deletion guard (filter_false_absence_flags)
  and census inconsistency detection (detect_census_inconsistency)
  (TestMemoryConsolidatorRunWiring)
- Step-16: end-to-end fidelity — genuinely-absent task's flag survives run() when
  get_task RAISES the not-found TaskmasterError (real backend behavior)
  (TestMemoryConsolidatorRunWiring.test_genuine_absence_flag_survives_run)
"""

from __future__ import annotations

import json
import re
from datetime import UTC, datetime
from pathlib import Path
from unittest.mock import AsyncMock, MagicMock, patch

import pytest

from fused_memory.backends.task_backend_errors import TaskmasterError
from fused_memory.config.schema import ReconciliationConfig
from fused_memory.models.reconciliation import (
    AssembledPayload,
    EventSource,
    EventType,
    ReconciliationEvent,
    StageId,
    StageReport,
    Watermark,
)
from fused_memory.reconciliation.stages.base import BaseStage
from fused_memory.reconciliation.stages.memory_consolidator import MemoryConsolidator
from fused_memory.reconciliation.task_filter import FilteredTaskTree


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
# task-1574: CSPRNG summary_nonce injection in Stage 1 per-cycle payloads
# ---------------------------------------------------------------------------


class TestStage1PayloadSummaryNonce:
    """assemble_payload / _format_assembled_payload inject '### Per-Cycle Summary Nonce'
    with a fresh STAGE1-prefixed CSPRNG nonce per call (task 1590).

    Task 1590: nonce format updated from bare '<8-hex>' to 'STAGE1_<8-hex>' so
    Stage 1 and Stage 2 summaries always lead with structurally distinct tokens.
    """

    @pytest.mark.asyncio
    async def test_legacy_path_contains_nonce_section(self):
        """Legacy assemble_payload (time-windowed path) includes '### Per-Cycle Summary Nonce'
        and a summary_nonce: STAGE1_<8-hex> line.

        Task 1590: regex updated from bare '[0-9a-f]{8}' to 'STAGE1_[0-9a-f]{8}'.
        RED until step-6 impl changes _build_summary_nonce_section() to pass 'STAGE1'.
        """
        stage = _make_consolidator()
        watermark = Watermark(project_id='test_project')

        payload = await stage.assemble_payload(
            events=[], watermark=watermark, prior_reports=[]
        )

        assert '### Per-Cycle Summary Nonce' in payload, (
            "assemble_payload must inject '### Per-Cycle Summary Nonce' section "
            "(needed to defeat Mem0 cosine-similarity dedup; task 1574)"
        )
        m = re.search(r'summary_nonce: (STAGE1_[0-9a-f]{8})', payload)
        assert m is not None, (
            "assemble_payload must include 'summary_nonce: STAGE1_<8-hex>' in the payload "
            f"(regex r'summary_nonce: STAGE1_[0-9a-f]{{8}}' found no match); payload excerpt:\n"
            f"{payload[:500]}"
        )
        assert m.group(1).startswith('STAGE1_'), (
            f"summary_nonce must begin with 'STAGE1_'; got {m.group(1)!r}"
        )

    @pytest.mark.asyncio
    async def test_legacy_path_nonce_is_fresh_per_call(self):
        """Two consecutive assemble_payload calls produce distinct STAGE1-prefixed nonces.

        Proves each call generates a fresh CSPRNG nonce rather than caching one value.
        Task 1590: regex updated to match 'STAGE1_<8-hex>' form.
        RED until step-6 impl changes _build_summary_nonce_section() to pass 'STAGE1'.
        """
        stage = _make_consolidator()
        watermark = Watermark(project_id='test_project')

        payload1 = await stage.assemble_payload(
            events=[], watermark=watermark, prior_reports=[]
        )
        payload2 = await stage.assemble_payload(
            events=[], watermark=watermark, prior_reports=[]
        )

        m1 = re.search(r'summary_nonce: (STAGE1_[0-9a-f]{8})', payload1)
        m2 = re.search(r'summary_nonce: (STAGE1_[0-9a-f]{8})', payload2)
        assert m1 is not None and m2 is not None, (
            "Both assemble_payload calls must emit a 'summary_nonce: STAGE1_<8-hex>'; "
            "one or both were missing."
        )
        assert m1.group(1) != m2.group(1), (
            f"Consecutive assemble_payload calls must produce DISTINCT nonces "
            f"(got {m1.group(1)!r} both times — nonce is not fresh per call)"
        )

    @pytest.mark.asyncio
    async def test_assembled_path_contains_nonce_section(self):
        """_format_assembled_payload (ContextAssembler path) includes STAGE1-prefixed nonce.

        Task 1590: regex updated from bare '[0-9a-f]{8}' to 'STAGE1_[0-9a-f]{8}'.
        RED until step-6 impl changes _build_summary_nonce_section() to pass 'STAGE1'.
        """
        stage = _make_consolidator()
        stage.assembled_payload = AssembledPayload(events=[], context_items={})
        watermark = Watermark(project_id='test_project')

        payload = await stage.assemble_payload(
            events=[], watermark=watermark, prior_reports=[]
        )

        assert '### Per-Cycle Summary Nonce' in payload, (
            "_format_assembled_payload must inject '### Per-Cycle Summary Nonce' section "
            "(needed to defeat Mem0 cosine-similarity dedup; task 1574)"
        )
        m = re.search(r'summary_nonce: (STAGE1_[0-9a-f]{8})', payload)
        assert m is not None, (
            "_format_assembled_payload must include 'summary_nonce: STAGE1_<8-hex>' in the payload "
            f"(regex r'summary_nonce: STAGE1_[0-9a-f]{{8}}' found no match); payload excerpt:\n"
            f"{payload[:500]}"
        )
        assert m.group(1).startswith('STAGE1_'), (
            f"summary_nonce must begin with 'STAGE1_'; got {m.group(1)!r}"
        )


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
    from escalation.queue import EscalationQueue

    from fused_memory.config.schema import FusedMemoryConfig, ReconciliationConfig
    from fused_memory.reconciliation.harness import ReconciliationHarness

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
        """Non-actionable findings in _maybe_remediate are logged, NOT escalated.

        Per Task 1512 / plans/afk-A7-recon-closure.md: non-actionable findings
        are forward-fed into the next cycle and emitted as structured log records.
        They must NOT be placed in the escalation queue — escalating them is a
        category error since the only human action ('accept as known') is achieved
        by not filing.
        """
        from datetime import UTC, datetime

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
        assert len(files) == 0, (
            f'Non-actionable findings must NOT be escalated (Task 1512 design: '
            f'they are logged/forward-fed instead); got {len(files)} file(s)'
        )

    # ── Step 9: regression guard — harness never resolves its own escalations ──

    def test_harness_never_resolves_its_own_escalations(self, tmp_path):
        """Harness must never call EscalationQueue.resolve() — watcher is sole closer.

        Pins the watcher-is-sole-closer contract from plans/afk-A7-recon-closure.md.
        Already-passing behaviourally; exists as a regression guard so any future
        accidental resolve() call in _escalate is caught immediately.
        """
        from unittest.mock import patch

        from escalation.queue import EscalationQueue

        harness, queue_dir = _make_dedup_harness(tmp_path)

        sentinel_called = []

        def _never_resolve(*args, **kwargs):
            sentinel_called.append(('resolve', args, kwargs))
            raise AssertionError(
                'ReconciliationHarness must never call EscalationQueue.resolve() — '
                'the escalation-watcher session is the sole closer '
                '(plans/afk-A7-recon-closure.md)'
            )

        with patch.object(EscalationQueue, 'resolve', side_effect=_never_resolve):
            # Simulate all four _escalate categories used by the harness:
            # (a) recon_stale_run — _recover_stale_runs path
            harness._escalate('recon_stale_run', 'run1111', 'Run stale (>300s), recovered')
            # (b) recon_failure — run_full_cycle except arm
            harness._escalate('recon_failure', 'run2222', 'Stage s1 failed: timeout')
            # (c) recon_integrity_issue with finding — _maybe_remediate non-actionable loop
            harness._escalate(
                'recon_integrity_issue', 'run3333',
                'Non-actionable integrity finding: task 452 stale',
                finding={'category': 'knowledge_stale', 'affected_ids': ['452'], 'description': 'x'},
            )
            # (d) recon_integrity_issue without finding — _maybe_remediate except arm
            harness._escalate('recon_integrity_issue', 'run4444', 'Remediation orchestration failed: err')
            # (e) recon_backlog_overflow — BacklogIterator except arm
            harness._escalate('recon_backlog_overflow', 'run5555', 'Backlog chunk 1 failed: oom')

        # resolve must never have been invoked
        assert sentinel_called == [], (
            f'EscalationQueue.resolve() was unexpectedly called: {sentinel_called}'
        )

        # All submitted escalations must still be pending (not resolved/archived)
        files = list(queue_dir.glob('esc-*.json'))
        assert files, 'Expected at least one pending escalation after all _escalate calls'
        for f in files:
            data = json.loads(f.read_text())
            assert data['status'] == 'pending', (
                f'{f.name}: expected status=pending, got {data["status"]!r}'
            )

    # ── Amendment 3: severity promotion on fold ─────────────────────────

    def test_severity_promoted_to_max_on_fold(self, tmp_path):
        """Folding a 'blocking' child into an 'info' parent promotes parent severity.

        Documents the attach_dedupe_child / _max_severity contract so that any
        future refactor that breaks severity promotion is caught immediately.
        The watcher (port 8103) reads parent.severity for triage — a blocking
        recurrence must remain visible even if the parent was filed at 'info'.

        Note: _escalate maps recon_integrity_issue → 'info' and recon_failure →
        'blocking' consistently, so different severities for the same fingerprint
        cannot arise via _escalate alone (same category ⇒ same fingerprint ⇒
        same severity slot).  This test exercises the promotion path directly via
        submit_or_dedupe so the _max_severity contract is pinned regardless of
        how _escalate's severity mapping may evolve.
        """
        from escalation.dedupe import (
            DedupeConfig,
            compute_content_fingerprint,
            content_fingerprint_key,
            submit_or_dedupe,
        )
        from escalation.models import Escalation
        from escalation.queue import EscalationQueue

        queue_dir = tmp_path / 'sev_test'
        queue = EscalationQueue(queue_dir)
        cfg = DedupeConfig(
            infra_dedupe_enabled=True,
            infra_dedupe_window_secs=float('inf'),
            infra_dedupe_categories=('test_sev_category',),
            key_fn=content_fingerprint_key,
        )
        fp = compute_content_fingerprint('test_sev_category', 'stale_k', ['t1'], '')

        # First submission: 'info' severity → becomes the parent.
        esc1 = Escalation(
            id=queue.make_id('sev-parent'),
            task_id='sev-parent',
            agent_role='test',
            severity='info',
            category='test_sev_category',
            summary='stale_k for t1',
            dedupe_fingerprint=fp,
        )
        submit_or_dedupe(queue, esc1, cfg)

        # Second submission: 'blocking' severity → folds into parent, promoting severity.
        esc2 = Escalation(
            id=queue.make_id('sev-child'),
            task_id='sev-child',
            agent_role='test',
            severity='blocking',
            category='test_sev_category',
            summary='stale_k for t1',
            dedupe_fingerprint=fp,
        )
        submit_or_dedupe(queue, esc2, cfg)

        files = list(queue_dir.glob('esc-*.json'))
        assert len(files) == 1, f'Expected 1 file (folded), got {len(files)}'
        data = json.loads(files[0].read_text())
        assert data['severity'] == 'blocking', (
            f'Expected severity promoted to "blocking", got {data["severity"]!r}. '
            'attach_dedupe_child must call _max_severity() so the parent always '
            'reflects the highest-urgency signal across all folded occurrences.'
        )
        assert data['dedupe_count'] == 1, (
            f'Expected dedupe_count==1 (one child folded), got {data["dedupe_count"]}'
        )

    # ── Amendment 4: producer/consumer contract for fingerprint key ──────

    def test_content_fingerprint_key_matches_dedupe_fingerprint(self, tmp_path):
        """content_fingerprint_key(esc) == esc.dedupe_fingerprint — producer/consumer contract.

        Guards against future divergence where the watcher re-hashes independently
        and starts treating the same finding as distinct (causing dedupe_count to
        rise while separate files reopen for the same finding).

        The watcher uses content_fingerprint_key to find the dedup parent.
        The harness stamps esc.dedupe_fingerprint via compute_content_fingerprint.
        Both must agree on identity — this test pins that invariant end-to-end
        by reading a real escalation off disk and asserting the key function
        returns its stored fingerprint.
        """
        from escalation.dedupe import (
            compute_content_fingerprint,
            content_fingerprint_key,
        )
        from escalation.models import Escalation

        harness, queue_dir = _make_dedup_harness(tmp_path)

        harness._escalate(
            'recon_integrity_issue',
            run_id='contract-run',
            summary='Contract test finding',
            finding={
                'category': 'contract_cat',
                'affected_ids': ['id1'],
                'description': 'desc contract',
            },
        )

        files = list(queue_dir.glob('esc-*.json'))
        assert len(files) == 1
        esc = Escalation.from_json(files[0].read_text())

        # The key the watcher uses must equal the producer-stamped fingerprint.
        assert content_fingerprint_key(esc) == esc.dedupe_fingerprint, (
            'content_fingerprint_key(esc) diverges from esc.dedupe_fingerprint — '
            'producer (harness._escalate) and consumer (watcher find_dedupe_parent) '
            'use different identity paths, which would cause dedup to silently break.'
        )
        expected_fp = compute_content_fingerprint(
            'recon_integrity_issue', 'contract_cat', ['id1'], 'desc contract',
        )
        assert esc.dedupe_fingerprint == expected_fp, (
            f'Fingerprint mismatch: stored {esc.dedupe_fingerprint!r}, '
            f'expected {expected_fp!r}'
        )

    @pytest.mark.asyncio
    async def test_remediation_residue_passes_finding_to_escalate(self, tmp_path):
        """_run_remediation_pass residue loop uses finding-based fingerprint.

        When an actionable residue finding persists after remediation (persistence
        count >= threshold), _escalate is called with finding= so the fingerprint
        is keyed on finding identity rather than a summary hash.
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
            'actionable': True,  # must be actionable to reach the escalation path
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

        # Mock _finding_persistence_count to return the threshold value (4) so
        # the escalation gate fires without needing a real journal history.
        with patch.object(
            harness, '_finding_persistence_count', new=AM(return_value=4),
        ):
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


# ---------------------------------------------------------------------------
# Step-11: MemoryConsolidator.run() wiring — deletion guard + census check
# ---------------------------------------------------------------------------


class TestMemoryConsolidatorRunWiring:
    """MemoryConsolidator.run() must:

    (a) pipe items_flagged through filter_false_absence_flags after dedup_flags,
        dropping absence-type flags whose tasks are confirmed PRESENT (fail-closed
        guard against phantom-task false positives), and
    (b) detect census inconsistency from this cycle's events and set
        report.stats['task_tree_census_inconsistent'] when events reference task IDs
        that exceed the census max from filtered_task_tree.

    RED until step-12 wires filter_false_absence_flags and detect_census_inconsistency
    into MemoryConsolidator.run().
    """

    @pytest.mark.asyncio
    async def test_false_absence_flag_dropped_for_present_task(self):
        """Absence-type flag for a PRESENT task must be dropped from items_flagged.

        Simulates the main fix: a 'task_absent' flag emitted by Stage 1 for task 3438,
        but get_task returns a real record -> filter_false_absence_flags drops the flag.
        A non-absence flag for the same task must survive unchanged.

        RED before step-12: filter_false_absence_flags is not called, so the
        absence flag remains in items_flagged.
        """
        stage = _make_consolidator(project_root='/tmp/reify')
        stage.project_id = 'test_project'
        # get_task returns a present record — task is real, not absent
        assert stage.taskmaster is not None  # AsyncMock() from _make_consolidator
        stage.taskmaster.get_task.return_value = {'id': '3438', 'title': 'real task 3438'}  # type: ignore[union-attr]

        absence_flag = {
            'task_id': '3438',
            'flag_type': 'task_absent',
            'description': 'Task 3438 not found in task tree — may be phantom',
        }
        normal_flag = {
            'task_id': '100',
            'flag_type': 'missing_deliverable',
            'description': 'Task 100 has no deliverable',
        }
        all_flags = [absence_flag, normal_flag]

        base_report = StageReport(
            stage=StageId.memory_consolidator,
            started_at=datetime.now(UTC),
            completed_at=datetime.now(UTC),
            items_flagged=list(all_flags),
            stats={},
        )
        # dedup_flags passes all flags through unchanged
        dedup_mock = AsyncMock(return_value=list(all_flags))

        with (
            patch.object(BaseStage, 'run', new=AsyncMock(return_value=base_report)),
            patch(
                'fused_memory.reconciliation.stages.memory_consolidator.dedup_flags',
                new=dedup_mock,
            ),
        ):
            report = await stage.run(
                events=[],
                watermark=Watermark(project_id='test_project'),
                prior_reports=[],
                run_id='run-step11-a',
            )

        flag_types = [f.get('flag_type') for f in report.items_flagged]
        assert 'task_absent' not in flag_types, (
            "filter_false_absence_flags must drop 'task_absent' flags for tasks that "
            "are PRESENT (get_task returns a real record); "
            f"got items_flagged={report.items_flagged!r}. "
            "RED: filter_false_absence_flags is not yet wired into run()."
        )
        assert 'missing_deliverable' in flag_types, (
            "Non-absence flag 'missing_deliverable' must survive filter_false_absence_flags; "
            f"got items_flagged={report.items_flagged!r}"
        )

    @pytest.mark.asyncio
    async def test_census_inconsistency_detected_from_events(self):
        """Census inconsistency: events referencing task IDs > census max must set stats.

        Sets filtered_task_tree.max_task_id=1515 and passes an event with task_id=3438
        (> 1515). After run(), report.stats['task_tree_census_inconsistent'] must be
        set and contain 3438 (or its count).

        RED before step-12: detect_census_inconsistency is not called, so the stat
        is not set.
        """
        stage = _make_consolidator(project_root='/tmp/reify')
        stage.project_id = 'test_project'
        stage.filtered_task_tree = FilteredTaskTree(max_task_id=1515, total_count=1515)

        # Event with task_id exceeding the census max
        event_high_id = ReconciliationEvent(
            id='evt-3438',
            type=EventType.task_status_changed,
            source=EventSource.agent,
            project_id='test_project',
            timestamp=datetime.now(UTC),
            payload={'task_id': 3438},
        )
        # Event with task_id within census max — must NOT appear in stat
        event_low_id = ReconciliationEvent(
            id='evt-0012',
            type=EventType.task_created,
            source=EventSource.agent,
            project_id='test_project',
            timestamp=datetime.now(UTC),
            payload={'task_id': 12},
        )

        base_report = StageReport(
            stage=StageId.memory_consolidator,
            started_at=datetime.now(UTC),
            completed_at=datetime.now(UTC),
            items_flagged=[],
            stats={},
        )
        dedup_mock = AsyncMock(return_value=[])

        with (
            patch.object(BaseStage, 'run', new=AsyncMock(return_value=base_report)),
            patch(
                'fused_memory.reconciliation.stages.memory_consolidator.dedup_flags',
                new=dedup_mock,
            ),
        ):
            report = await stage.run(
                events=[event_high_id, event_low_id],
                watermark=Watermark(project_id='test_project'),
                prior_reports=[],
                run_id='run-step11-b',
            )

        assert 'task_tree_census_inconsistent' in report.stats, (
            "run() must set report.stats['task_tree_census_inconsistent'] when events "
            "reference task IDs exceeding the census max (1515 < 3438); "
            f"got stats={report.stats!r}. "
            "RED: detect_census_inconsistency is not yet wired into run()."
        )
        val = report.stats['task_tree_census_inconsistent']
        assert val, (
            f"task_tree_census_inconsistent must be truthy (non-empty); got {val!r}"
        )
        # The offending ID 3438 must be represented; the in-range ID 12 must not
        if isinstance(val, list):
            assert 3438 in val, f"Expected 3438 in census inconsistency list {val!r}"
            assert 12 not in val, f"In-range task 12 must not appear in {val!r}"

    @pytest.mark.asyncio
    async def test_remediation_run_skips_absence_filter(self):
        """Remediation runs (remediation_findings set) skip dedup AND the absence filter.

        The early-return for remediation_findings must fire before filter_false_absence_flags
        is applied, so absence flags are passed through unchanged.

        This test remains GREEN after step-12 as long as the early-return is preserved.
        """
        stage = _make_consolidator(project_root='/tmp/reify')
        stage.project_id = 'test_project'
        stage.remediation_findings = [{'description': 'fix this'}]  # remediation mode
        # get_task must NOT be called at all for remediation runs
        assert stage.taskmaster is not None  # AsyncMock() from _make_consolidator
        stage.taskmaster.get_task.side_effect = AssertionError(  # type: ignore[union-attr]
            'get_task must NOT be called during a remediation run'
        )

        absence_flag = {
            'task_id': '3438',
            'flag_type': 'task_absent',
            'description': 'phantom task',
        }
        base_report = StageReport(
            stage=StageId.memory_consolidator,
            started_at=datetime.now(UTC),
            completed_at=datetime.now(UTC),
            items_flagged=[absence_flag],
            stats={},
        )
        dedup_mock = AsyncMock(return_value=[absence_flag])

        with (
            patch.object(BaseStage, 'run', new=AsyncMock(return_value=base_report)),
            patch(
                'fused_memory.reconciliation.stages.memory_consolidator.dedup_flags',
                new=dedup_mock,
            ),
        ):
            report = await stage.run(
                events=[],
                watermark=Watermark(project_id='test_project'),
                prior_reports=[],
                run_id='run-step11-c',
            )

        # Remediation early-return fires before filter_false_absence_flags —
        # get_task must never be called, and the flag must survive unchanged.
        assert report is not None
        assert report.items_flagged == [absence_flag], (
            "Remediation run must return items_flagged unchanged (early return before "
            f"filter_false_absence_flags); got {report.items_flagged!r}"
        )

    @pytest.mark.asyncio
    async def test_genuine_absence_flag_survives_run(self):
        """Step-16 end-to-end: a genuinely-absent task's flag SURVIVES run().

        Mirrors real sqlite backend behavior: self.taskmaster.get_task RAISES
        TaskmasterError('TASKMASTER_TOOL_ERROR', 'No tasks found for ID(s): 3438')
        on absence (sqlite_task_backend.py:497-499).  The interceptor re-raises
        (middleware/task_interceptor.py:3361-3363).

        After step-17's fix, filter_false_absence_flags normalizes the raised
        exception to {error: str(exc), error_type: typename} and passes it to
        confirm_task_absent, which recognises the not-found phrase and returns
        True → the flag is KEPT (task positively absent).

        RED against current impl: the `except Exception` handler unconditionally
        drops every raise as inconclusive, so the flag is removed from items_flagged
        even though the task is genuinely absent.
        """
        stage = _make_consolidator(project_root='/tmp/reify')
        stage.project_id = 'test_project'
        # Simulate real sqlite backend: RAISES not-found on absence
        assert stage.taskmaster is not None  # AsyncMock() from _make_consolidator
        stage.taskmaster.get_task.side_effect = TaskmasterError(  # type: ignore[union-attr]
            'TASKMASTER_TOOL_ERROR', 'No tasks found for ID(s): 3438'
        )

        absence_flag = {
            'task_id': '3438',
            'flag_type': 'task_absent',
            'description': 'Task 3438 not found — genuinely absent',
        }
        all_flags = [absence_flag]

        base_report = StageReport(
            stage=StageId.memory_consolidator,
            started_at=datetime.now(UTC),
            completed_at=datetime.now(UTC),
            items_flagged=list(all_flags),
            stats={},
        )
        dedup_mock = AsyncMock(return_value=list(all_flags))

        with (
            patch.object(BaseStage, 'run', new=AsyncMock(return_value=base_report)),
            patch(
                'fused_memory.reconciliation.stages.memory_consolidator.dedup_flags',
                new=dedup_mock,
            ),
        ):
            report = await stage.run(
                events=[],
                watermark=Watermark(project_id='test_project'),
                prior_reports=[],
                run_id='run-step16',
            )

        flag_types = [f.get('flag_type') for f in report.items_flagged]
        assert 'task_absent' in flag_types, (
            "filter_false_absence_flags must KEEP 'task_absent' flags when get_task "
            "RAISES the not-found TaskmasterError (real sqlite backend behavior for "
            "genuinely absent tasks). "
            f"got items_flagged={report.items_flagged!r}. "
            "RED: current impl's except handler drops ALL raises as inconclusive."
        )


# ---------------------------------------------------------------------------
# Step-11 (RED) / step-12 (GREEN): task_count_verification stat wiring
# ---------------------------------------------------------------------------


class TestMemoryConsolidatorTaskCountVerificationWiring:
    """MemoryConsolidator.run() must surface task_count_verification in report.stats.

    step-11 (RED): task_count_verification attribute and stat-wiring don't exist yet.
    step-12 (GREEN): add class attribute + stat block in run().
    """

    @pytest.mark.asyncio
    async def test_task_count_verification_stat_set_when_inconsistent(self):
        """When task_count_verification has consistent=False, report.stats is set and WARNING logged."""
        stage = _make_consolidator(project_root='/tmp/reify')
        stage.project_id = 'test_project'

        verification_record = {
            'available': True,
            'consistent': False,
            'done_mismatch': True,
            'total_mismatch': False,
            'authoritative': {'done': 608, 'total': 635},
            'tree': {'done': 600, 'total': 635},
        }
        stage.task_count_verification = verification_record

        base_report = StageReport(
            stage=StageId.memory_consolidator,
            started_at=datetime.now(UTC),
            completed_at=datetime.now(UTC),
            items_flagged=[],
            stats={},
        )
        dedup_mock = AsyncMock(return_value=[])

        with (
            patch.object(BaseStage, 'run', new=AsyncMock(return_value=base_report)),
            patch(
                'fused_memory.reconciliation.stages.memory_consolidator.dedup_flags',
                new=dedup_mock,
            ),
        ):
            report = await stage.run(
                events=[],
                watermark=Watermark(project_id='test_project'),
                prior_reports=[],
                run_id='run-step11-tcv',
            )

        assert 'task_count_verification' in report.stats, (
            "run() must set report.stats['task_count_verification'] when "
            "stage.task_count_verification is set; "
            f"got stats={report.stats!r}"
        )
        assert report.stats['task_count_verification'] == verification_record

    @pytest.mark.asyncio
    async def test_task_count_verification_warning_when_inconsistent(self, caplog):
        """run() logs a WARNING when task_count_verification.consistent is False."""
        import logging

        stage = _make_consolidator(project_root='/tmp/reify')
        stage.project_id = 'test_project'
        stage.task_count_verification = {
            'available': True,
            'consistent': False,
            'done_mismatch': True,
            'total_mismatch': False,
            'authoritative': {'done': 608, 'total': 635},
            'tree': {'done': 600, 'total': 635},
        }

        base_report = StageReport(
            stage=StageId.memory_consolidator,
            started_at=datetime.now(UTC),
            completed_at=datetime.now(UTC),
            items_flagged=[],
            stats={},
        )
        dedup_mock = AsyncMock(return_value=[])

        with (
            patch.object(BaseStage, 'run', new=AsyncMock(return_value=base_report)),
            patch(
                'fused_memory.reconciliation.stages.memory_consolidator.dedup_flags',
                new=dedup_mock,
            ),
            caplog.at_level(logging.WARNING),
        ):
            await stage.run(
                events=[],
                watermark=Watermark(project_id='test_project'),
                prior_reports=[],
                run_id='run-step11-tcv-warn',
            )

        assert any(
            r.levelno >= logging.WARNING for r in caplog.records
        ), 'Expected at least one WARNING when task_count_verification.consistent=False'

    @pytest.mark.asyncio
    async def test_task_count_verification_absent_when_none(self):
        """When stage.task_count_verification is None, the key must be absent from report.stats."""
        stage = _make_consolidator(project_root='/tmp/reify')
        stage.project_id = 'test_project'
        # Explicitly leave task_count_verification at the default (None)

        base_report = StageReport(
            stage=StageId.memory_consolidator,
            started_at=datetime.now(UTC),
            completed_at=datetime.now(UTC),
            items_flagged=[],
            stats={},
        )
        dedup_mock = AsyncMock(return_value=[])

        with (
            patch.object(BaseStage, 'run', new=AsyncMock(return_value=base_report)),
            patch(
                'fused_memory.reconciliation.stages.memory_consolidator.dedup_flags',
                new=dedup_mock,
            ),
        ):
            report = await stage.run(
                events=[],
                watermark=Watermark(project_id='test_project'),
                prior_reports=[],
                run_id='run-step11-tcv-none',
            )

        assert 'task_count_verification' not in report.stats, (
            "When stage.task_count_verification is None, the key must be absent from "
            f"report.stats; got stats={report.stats!r}"
        )


# ---------------------------------------------------------------------------
# Step-13 (RED) / step-14 (GREEN): graphiti_queue_health stat wiring
# ---------------------------------------------------------------------------


class TestMemoryConsolidatorGraphitiQueueHealthWiring:
    """MemoryConsolidator.run() must surface graphiti_queue_health in report.stats.

    step-13 (RED): graphiti_queue_health attribute and stat-wiring don't exist yet.
    step-14 (GREEN): add class attribute + stat block in run().
    """

    @pytest.mark.asyncio
    async def test_graphiti_queue_health_stat_set_when_unhealthy(self):
        """When graphiti_queue_health.healthy=False, report.stats is set."""
        stage = _make_consolidator(project_root='/tmp/reify')
        stage.project_id = 'test_project'

        health_record = {
            'dead_count': 2,
            'pending_count': 0,
            'retry_count': 0,
            'oldest_pending_age_seconds': None,
            'healthy': False,
        }
        stage.graphiti_queue_health = health_record

        base_report = StageReport(
            stage=StageId.memory_consolidator,
            started_at=datetime.now(UTC),
            completed_at=datetime.now(UTC),
            items_flagged=[],
            stats={},
        )
        dedup_mock = AsyncMock(return_value=[])

        with (
            patch.object(BaseStage, 'run', new=AsyncMock(return_value=base_report)),
            patch(
                'fused_memory.reconciliation.stages.memory_consolidator.dedup_flags',
                new=dedup_mock,
            ),
        ):
            report = await stage.run(
                events=[],
                watermark=Watermark(project_id='test_project'),
                prior_reports=[],
                run_id='run-step13-gqh',
            )

        assert 'graphiti_queue_health' in report.stats, (
            "run() must set report.stats['graphiti_queue_health'] when "
            "stage.graphiti_queue_health is set; "
            f"got stats={report.stats!r}"
        )
        assert report.stats['graphiti_queue_health'] == health_record

    @pytest.mark.asyncio
    async def test_graphiti_queue_health_warning_when_unhealthy(self, caplog):
        """run() logs a WARNING when graphiti_queue_health.healthy=False."""
        import logging

        stage = _make_consolidator(project_root='/tmp/reify')
        stage.project_id = 'test_project'
        stage.graphiti_queue_health = {
            'dead_count': 2, 'pending_count': 0, 'retry_count': 0,
            'oldest_pending_age_seconds': None, 'healthy': False,
        }

        base_report = StageReport(
            stage=StageId.memory_consolidator,
            started_at=datetime.now(UTC),
            completed_at=datetime.now(UTC),
            items_flagged=[],
            stats={},
        )
        dedup_mock = AsyncMock(return_value=[])

        with (
            patch.object(BaseStage, 'run', new=AsyncMock(return_value=base_report)),
            patch(
                'fused_memory.reconciliation.stages.memory_consolidator.dedup_flags',
                new=dedup_mock,
            ),
            caplog.at_level(logging.WARNING),
        ):
            await stage.run(
                events=[],
                watermark=Watermark(project_id='test_project'),
                prior_reports=[],
                run_id='run-step13-gqh-warn',
            )

        assert any(
            r.levelno >= logging.WARNING for r in caplog.records
        ), 'Expected at least one WARNING when graphiti_queue_health.healthy=False'

    @pytest.mark.asyncio
    async def test_graphiti_queue_health_absent_when_none(self):
        """When stage.graphiti_queue_health is None, the key must be absent from report.stats."""
        stage = _make_consolidator(project_root='/tmp/reify')
        stage.project_id = 'test_project'
        # Explicitly leave graphiti_queue_health at the default (None)

        base_report = StageReport(
            stage=StageId.memory_consolidator,
            started_at=datetime.now(UTC),
            completed_at=datetime.now(UTC),
            items_flagged=[],
            stats={},
        )
        dedup_mock = AsyncMock(return_value=[])

        with (
            patch.object(BaseStage, 'run', new=AsyncMock(return_value=base_report)),
            patch(
                'fused_memory.reconciliation.stages.memory_consolidator.dedup_flags',
                new=dedup_mock,
            ),
        ):
            report = await stage.run(
                events=[],
                watermark=Watermark(project_id='test_project'),
                prior_reports=[],
                run_id='run-step13-gqh-none',
            )

        assert 'graphiti_queue_health' not in report.stats, (
            "When stage.graphiti_queue_health is None, the key must be absent from "
            f"report.stats; got stats={report.stats!r}"
        )


# ---------------------------------------------------------------------------
# Step-15 (RED) / step-16 (GREEN): Task Count Census payload section
# ---------------------------------------------------------------------------


class TestStage1PayloadTaskCountCensus:
    """Task Count Census section appears in both payload paths when verification is set.

    step-15 (RED): _build_task_count_census_section does not exist yet.
    step-16 (GREEN): add the helper + wire into both payload methods.
    """

    _CENSUS_HEADER = '### Task Count Census'

    def _make_stage_with_census(self) -> MemoryConsolidator:
        stage = _make_consolidator(project_root='/tmp/reify')
        stage.task_count_verification = {
            'available': True,
            'consistent': False,
            'done_mismatch': True,
            'total_mismatch': False,
            'authoritative': {'done': 608, 'total': 635},
            'tree': {'done': 600, 'total': 635},
        }
        return stage

    @pytest.mark.asyncio
    async def test_census_section_in_legacy_assemble_payload(self):
        """assemble_payload (legacy time-windowed path) includes Task Count Census section."""
        stage = self._make_stage_with_census()
        watermark = Watermark(project_id='test_project')

        result = await stage.assemble_payload(
            events=[], watermark=watermark, prior_reports=[]
        )

        assert self._CENSUS_HEADER in result, (
            f"assemble_payload must include '{self._CENSUS_HEADER}' when "
            f"task_count_verification is set; got:\n{result!r}"
        )
        # Authoritative counts must appear in the section
        assert '635' in result, 'Authoritative total=635 must appear in the census section'
        assert '608' in result, 'Authoritative done=608 must appear in the census section'
        # Source attribution
        assert 'get_statuses' in result, "'get_statuses' must be named as the source"
        # Divergence note: fixture has consistent=False (tree done=600 vs authoritative done=608)
        assert 'Divergence detected' in result, (
            "Divergence note must appear when consistent=False; got:\n{result!r}"
        )
        assert '600' in result, 'Tree done=600 must appear in the divergence note'

    @pytest.mark.asyncio
    async def test_census_section_in_assembled_payload_path(self):
        """_format_assembled_payload (ContextAssembler path) includes Task Count Census section."""
        stage = self._make_stage_with_census()
        stage.assembled_payload = AssembledPayload(events=[], context_items={})
        watermark = Watermark(project_id='test_project')

        result = await stage.assemble_payload(
            events=[], watermark=watermark, prior_reports=[]
        )

        assert self._CENSUS_HEADER in result, (
            f"_format_assembled_payload must include '{self._CENSUS_HEADER}' when "
            f"task_count_verification is set; got:\n{result!r}"
        )
        assert '635' in result, 'Authoritative total=635 must appear in census section'
        assert '608' in result, 'Authoritative done=608 must appear in census section'
        assert 'get_statuses' in result, "'get_statuses' must be named as the source"
        # Divergence note: fixture has consistent=False (tree done=600 vs authoritative done=608)
        assert 'Divergence detected' in result, (
            "Divergence note must appear when consistent=False; got:\n{result!r}"
        )
        assert '600' in result, 'Tree done=600 must appear in the divergence note'

    @pytest.mark.asyncio
    async def test_census_section_absent_when_verification_is_none(self):
        """Census section must be absent when task_count_verification is None."""
        stage = _make_consolidator(project_root='/tmp/reify')
        # task_count_verification defaults to None — don't set it
        watermark = Watermark(project_id='test_project')

        result = await stage.assemble_payload(
            events=[], watermark=watermark, prior_reports=[]
        )

        assert self._CENSUS_HEADER not in result, (
            f"Census section must be absent when task_count_verification is None; "
            f"got:\n{result!r}"
        )


# ---------------------------------------------------------------------------
# task-1786 step-3 (RED) / step-4 (GREEN): stale snapshot correction wiring
# ---------------------------------------------------------------------------


class TestStaleCountSnapshotCorrectionWiring:
    """MemoryConsolidator.run() must apply filter_stale_count_snapshot_corrections
    as the FIRST post-processor in items_flagged (before filter_terminal_metadata_flags
    and dedup_flags), and surface
    report.stats['stale_count_snapshot_corrections_dropped'].

    RED until step-4 wires filter_stale_count_snapshot_corrections into run()
    and sets the stat.
    """

    @pytest.mark.asyncio
    async def test_incident_finding_dropped_and_benign_survives(self):
        """The incident snapshot-correction finding is dropped; benign finding survives.

        Seeds base_report.items_flagged with:
        - incident: flag_type='count_snapshot_mismatch', description containing
          the 634/607 → 635/608 correction text (triggers the filter)
        - benign: flag_type='missing_deliverable' (must pass through unchanged)

        After stage.run():
        - incident finding must be absent from report.items_flagged
        - benign finding must be present
        - report.stats['stale_count_snapshot_corrections_dropped'] == 1

        RED: filter_stale_count_snapshot_corrections is not yet wired into
        run() and the stat is unset.
        """
        stage = _make_consolidator(project_root='/tmp/reify')
        stage.project_id = 'test_project'

        incident_flag = {
            'task_id': None,
            'flag_type': 'count_snapshot_mismatch',
            'description': (
                'Snapshot edge for autopilot_video reports 634/607 but is off by 1; '
                'should be 635/608 to match the Active Task Tree header.'
            ),
            'suggested_action': 'Correct the snapshot edge to 635/608.',
        }
        benign_flag = {
            'task_id': '100',
            'flag_type': 'missing_deliverable',
            'description': 'Task 100 has no deliverable',
        }
        all_flags = [incident_flag, benign_flag]

        base_report = StageReport(
            stage=StageId.memory_consolidator,
            started_at=datetime.now(UTC),
            completed_at=datetime.now(UTC),
            items_flagged=list(all_flags),
            stats={},
        )
        # dedup_flags passes all flags through unchanged so we can inspect
        # the stale-snapshot filter's effect directly
        dedup_mock = AsyncMock(side_effect=lambda **kw: kw['flags'])

        with (
            patch.object(BaseStage, 'run', new=AsyncMock(return_value=base_report)),
            patch(
                'fused_memory.reconciliation.stages.memory_consolidator.dedup_flags',
                new=dedup_mock,
            ),
        ):
            report = await stage.run(
                events=[],
                watermark=Watermark(project_id='test_project'),
                prior_reports=[],
                run_id='run-1786-step3',
            )

        flag_types = [f.get('flag_type') for f in report.items_flagged]

        assert 'count_snapshot_mismatch' not in flag_types, (
            "filter_stale_count_snapshot_corrections must drop the incident "
            "'count_snapshot_mismatch' finding (634/607→635/608, delta=1); "
            f"got items_flagged={report.items_flagged!r}. "
            "RED: filter_stale_count_snapshot_corrections is not yet wired into run()."
        )
        assert 'missing_deliverable' in flag_types, (
            "Benign 'missing_deliverable' flag must survive the stale-snapshot filter; "
            f"got items_flagged={report.items_flagged!r}"
        )
        assert report.stats.get('stale_count_snapshot_corrections_dropped') == 1, (
            "run() must set report.stats['stale_count_snapshot_corrections_dropped'] = 1 "
            "when one stale-snapshot-correction finding is dropped; "
            f"got stats={report.stats!r}. "
            "RED: stat not yet surfaced."
        )


# ---------------------------------------------------------------------------
# step-01 (RED) / step-02 (GREEN): assemble_payload fetch degraded tracking
# ---------------------------------------------------------------------------


class TestConsolidatorFetchDegradedSources:
    """assemble_payload must track fetch failures in _fetch_degraded_sources.

    Case A (episodes): when get_episodes raises, a WARNING must be emitted AND
    'episodes' must appear in stage._fetch_degraded_sources.

    Case B (mem0): when mem0.get_all raises, 'mem0' must appear in
    stage._fetch_degraded_sources.

    Case C (status): when get_status raises, a WARNING must be emitted AND
    'status' must appear in stage._fetch_degraded_sources.

    RED until step-02 wires the tracking and logging.
    """

    @pytest.mark.asyncio
    async def test_episode_fetch_failure_logs_warning_and_tracks_degraded_source(
        self, caplog
    ):
        """Case A: get_episodes raises → WARNING emitted AND 'episodes' in _fetch_degraded_sources."""
        import logging

        stage = _make_consolidator()
        stage.memory.get_episodes = AsyncMock(side_effect=RuntimeError('graphiti down'))
        watermark = Watermark(project_id='test_project')

        with caplog.at_level(logging.WARNING, logger='fused_memory'):
            await stage.assemble_payload(events=[], watermark=watermark, prior_reports=[])

        # Must emit a WARNING for the episodes fetch failure, with the specific event key
        warnings = [r for r in caplog.records if r.levelno == logging.WARNING]
        assert any(
            'reconciliation.stage1_episodes_fetch_failed' in r.getMessage()
            for r in warnings
        ), (
            "Expected a WARNING log containing 'reconciliation.stage1_episodes_fetch_failed' "
            'when get_episodes raises, got none. '
            'RED: the episodes except is a bare `except Exception: episodes=[]` with no log.'
        )

        # Must track the degraded source
        degraded = getattr(stage, '_fetch_degraded_sources', None)
        assert degraded is not None, (
            '_fetch_degraded_sources attribute does not exist on MemoryConsolidator. '
            'RED: attribute not yet declared.'
        )
        assert 'episodes' in degraded, (
            f"Expected 'episodes' in _fetch_degraded_sources, got: {degraded!r}. "
            "RED: episodes except does not append to _fetch_degraded_sources."
        )

    @pytest.mark.asyncio
    async def test_mem0_fetch_failure_tracks_degraded_source(self):
        """Case B: mem0.get_all raises → 'mem0' in _fetch_degraded_sources.

        NOTE: the mem0 except already emits a WARNING on main (added by a sibling task),
        so the RED driver is solely the missing _fetch_degraded_sources tracking
        (AttributeError or missing 'mem0' entry), not a missing warning.
        """
        stage = _make_consolidator()
        stage.memory.mem0.get_all = AsyncMock(side_effect=RuntimeError('mem0 down'))
        watermark = Watermark(project_id='test_project')

        await stage.assemble_payload(events=[], watermark=watermark, prior_reports=[])

        degraded = getattr(stage, '_fetch_degraded_sources', None)
        assert degraded is not None, (
            '_fetch_degraded_sources attribute does not exist on MemoryConsolidator. '
            'RED: attribute not yet declared.'
        )
        assert 'mem0' in degraded, (
            f"Expected 'mem0' in _fetch_degraded_sources, got: {degraded!r}. "
            "RED: mem0 except does not append to _fetch_degraded_sources."
        )

    @pytest.mark.asyncio
    async def test_status_fetch_failure_logs_warning_and_tracks_degraded_source(
        self, caplog
    ):
        """Case C: get_status raises → WARNING emitted AND 'status' in _fetch_degraded_sources."""
        import logging

        stage = _make_consolidator()
        stage.memory.get_status = AsyncMock(side_effect=RuntimeError('status backend down'))
        watermark = Watermark(project_id='test_project')

        with caplog.at_level(logging.WARNING, logger='fused_memory'):
            await stage.assemble_payload(events=[], watermark=watermark, prior_reports=[])

        # Must emit a WARNING for the status fetch failure, with the specific event key
        warnings = [r for r in caplog.records if r.levelno == logging.WARNING]
        assert any(
            'reconciliation.stage1_status_fetch_failed' in r.getMessage()
            for r in warnings
        ), (
            "Expected a WARNING log containing 'reconciliation.stage1_status_fetch_failed' "
            'when get_status raises, got none. '
            'RED: the store-stats except is a bare `except Exception: status={}` with no log.'
        )

        # Must track the degraded source
        degraded = getattr(stage, '_fetch_degraded_sources', None)
        assert degraded is not None, (
            '_fetch_degraded_sources attribute does not exist on MemoryConsolidator. '
            'RED: attribute not yet declared.'
        )
        assert 'status' in degraded, (
            f"Expected 'status' in _fetch_degraded_sources, got: {degraded!r}. "
            "RED: store-stats except does not append to _fetch_degraded_sources."
        )

    @pytest.mark.asyncio
    async def test_clean_fetch_leaves_degraded_sources_empty(self):
        """When no fetch fails, _fetch_degraded_sources is empty (genuine empty corpus)."""
        stage = _make_consolidator()
        watermark = Watermark(project_id='test_project')

        await stage.assemble_payload(events=[], watermark=watermark, prior_reports=[])

        degraded = getattr(stage, '_fetch_degraded_sources', None)
        assert degraded is not None, (
            '_fetch_degraded_sources attribute does not exist on MemoryConsolidator.'
        )
        assert degraded == [], (
            f"Expected empty _fetch_degraded_sources on clean fetch, got: {degraded!r}. "
            "Empty list = genuine empty corpus (distinguishable from fetch failure)."
        )

    @pytest.mark.asyncio
    async def test_assembled_path_status_fetch_failure_logs_warning_and_tracks_degraded_source(
        self, caplog
    ):
        """Case D: assembled-payload path, get_status raises → WARNING + 'status' in _fetch_degraded_sources.

        RED on base: _format_assembled_payload has a silent bare-except (no warning, no append).
        """
        import logging

        stage = _make_consolidator()
        stage.assembled_payload = AssembledPayload(events=[], context_items={})
        stage.memory.get_status = AsyncMock(side_effect=RuntimeError('status backend down'))
        watermark = Watermark(project_id='test_project')

        with caplog.at_level(logging.WARNING, logger='fused_memory'):
            await stage.assemble_payload(events=[], watermark=watermark, prior_reports=[])

        # Must emit a WARNING for the status fetch failure, with the specific event key
        warnings = [r for r in caplog.records if r.levelno == logging.WARNING]
        assert any(
            'reconciliation.stage1_status_fetch_failed' in r.getMessage()
            for r in warnings
        ), (
            "Expected a WARNING log containing 'reconciliation.stage1_status_fetch_failed' "
            'when get_status raises on assembled path, got none. '
            'RED: _format_assembled_payload has a silent bare-except with no log.'
        )

        # Must track the degraded source
        degraded = getattr(stage, '_fetch_degraded_sources', None)
        assert degraded is not None, (
            '_fetch_degraded_sources attribute does not exist on MemoryConsolidator.'
        )
        assert 'status' in degraded, (
            f"Expected 'status' in _fetch_degraded_sources for assembled path, got: {degraded!r}. "
            "RED: _format_assembled_payload bare-except does not append to _fetch_degraded_sources."
        )

    @pytest.mark.asyncio
    async def test_assembled_path_reset_prevents_degraded_leak_across_runs(self):
        """Reused-instance reset guard: call-1 (failure) tracks 'status'; call-2 (clean) resets to [].

        RED on base: call-1 has no append; also RED on naive-append-without-reset (call-2 leaks).
        This test pins the shared-class-mutable-default subtlety: _fetch_degraded_sources must be
        reset to a fresh instance [] at the top of _format_assembled_payload so reused instances
        never leak stale state from a prior run.
        """
        stage = _make_consolidator()
        stage.assembled_payload = AssembledPayload(events=[], context_items={})
        watermark = Watermark(project_id='test_project')

        # Call 1: get_status raises → 'status' should be tracked
        stage.memory.get_status = AsyncMock(side_effect=RuntimeError('status backend down'))
        await stage.assemble_payload(events=[], watermark=watermark, prior_reports=[])
        degraded_after_call1 = getattr(stage, '_fetch_degraded_sources', None)
        assert degraded_after_call1 is not None, (
            '_fetch_degraded_sources attribute does not exist on MemoryConsolidator.'
        )
        assert 'status' in degraded_after_call1, (
            f"Call 1 (failure): expected 'status' in _fetch_degraded_sources, got: {degraded_after_call1!r}. "
            'RED: _format_assembled_payload does not append to _fetch_degraded_sources on failure.'
        )

        # Call 2: get_status succeeds → _fetch_degraded_sources must be reset to []
        stage.memory.get_status = AsyncMock(return_value={})
        await stage.assemble_payload(events=[], watermark=watermark, prior_reports=[])
        degraded_after_call2 = getattr(stage, '_fetch_degraded_sources', None)
        assert degraded_after_call2 == [], (
            f"Call 2 (clean): expected _fetch_degraded_sources == [], got: {degraded_after_call2!r}. "
            'RED: _format_assembled_payload does not reset _fetch_degraded_sources before fetches, '
            'leaking stale state from a prior run (shared class-level mutable default).'
        )


# ---------------------------------------------------------------------------
# step-03 (RED) / step-04 (GREEN): run() surfacing stage1_fetch_degraded stat
# ---------------------------------------------------------------------------


class TestConsolidatorRunFetchDegradedStat:
    """MemoryConsolidator.run() must copy _fetch_degraded_sources to
    report.stats['stage1_fetch_degraded'] (empty list = clean, non-empty = failure).

    RED until step-04 adds the copy in run().
    """

    @pytest.mark.asyncio
    async def test_run_surfaces_degraded_episodes_in_stats(self):
        """When episodes fetch fails, report.stats['stage1_fetch_degraded'] == ['episodes']."""
        stage = _make_consolidator(project_root='/tmp/test_run_degraded')
        stage.project_id = 'test_project'
        stage.memory.get_episodes = AsyncMock(side_effect=RuntimeError('graphiti down'))

        # Use real assemble_payload (no BaseStage.run mock) — let MemoryConsolidator.run
        # call assemble_payload() internally, then check the stats it surfaces.
        # We need to mock BaseStage.run to capture how run() populates report.stats.
        # Strategy: let the real assemble_payload run (to populate _fetch_degraded_sources),
        # then replace BaseStage.run with a minimal mock that returns a fresh StageReport,
        # then check that MemoryConsolidator.run() copies _fetch_degraded_sources onto it.

        # Pre-populate _fetch_degraded_sources by calling assemble_payload directly first.
        # Then test the run() wiring via the BaseStage.run mock approach.
        watermark = Watermark(project_id='test_project')
        await stage.assemble_payload(events=[], watermark=watermark, prior_reports=[])

        # Now run() should copy _fetch_degraded_sources to report.stats
        base_report = StageReport(
            stage=StageId.memory_consolidator,
            started_at=datetime.now(UTC),
            completed_at=datetime.now(UTC),
            items_flagged=[],
            stats={},
        )

        with patch.object(BaseStage, 'run', new=AsyncMock(return_value=base_report)):
            report = await stage.run(
                events=[],
                watermark=watermark,
                prior_reports=[],
                run_id='run-step03-degraded',
            )

        assert 'stage1_fetch_degraded' in report.stats, (
            "run() must set report.stats['stage1_fetch_degraded']; key is missing. "
            "RED: run() does not yet copy _fetch_degraded_sources to stats."
        )
        assert report.stats['stage1_fetch_degraded'] == ['episodes'], (
            f"Expected report.stats['stage1_fetch_degraded'] == ['episodes'], "
            f"got: {report.stats.get('stage1_fetch_degraded')!r}. "
            "RED: run() does not yet copy _fetch_degraded_sources to stats."
        )

    @pytest.mark.asyncio
    async def test_run_surfaces_empty_degraded_on_clean_fetch(self):
        """When no fetch fails, report.stats['stage1_fetch_degraded'] == []."""
        stage = _make_consolidator(project_root='/tmp/test_run_clean')
        stage.project_id = 'test_project'

        watermark = Watermark(project_id='test_project')
        # Clean fetch — no exceptions
        await stage.assemble_payload(events=[], watermark=watermark, prior_reports=[])

        base_report = StageReport(
            stage=StageId.memory_consolidator,
            started_at=datetime.now(UTC),
            completed_at=datetime.now(UTC),
            items_flagged=[],
            stats={},
        )

        with patch.object(BaseStage, 'run', new=AsyncMock(return_value=base_report)):
            report = await stage.run(
                events=[],
                watermark=watermark,
                prior_reports=[],
                run_id='run-step03-clean',
            )

        assert report.stats.get('stage1_fetch_degraded') == [], (
            f"Expected report.stats['stage1_fetch_degraded'] == [] on clean fetch, "
            f"got: {report.stats.get('stage1_fetch_degraded')!r}. "
            "RED: stat not yet surfaced by run()."
        )

    @pytest.mark.asyncio
    async def test_run_surfaces_degraded_status_for_assembled_path(self):
        """Assembled-path, get_status raises → report.stats['stage1_fetch_degraded'] == ['status'].

        RED on base: _format_assembled_payload has no append, so _fetch_degraded_sources stays []
        and report.stats['stage1_fetch_degraded'] is [] rather than ['status'].
        Mirrors test_run_surfaces_degraded_episodes_in_stats but for the assembled-payload path.
        """
        stage = _make_consolidator(project_root='/tmp/test_run_degraded_assembled')
        stage.project_id = 'test_project'
        stage.assembled_payload = AssembledPayload(events=[], context_items={})
        stage.memory.get_status = AsyncMock(side_effect=RuntimeError('status backend down'))

        watermark = Watermark(project_id='test_project')
        # Pre-populate _fetch_degraded_sources by calling assemble_payload directly.
        # This exercises _format_assembled_payload (the assembled path).
        await stage.assemble_payload(events=[], watermark=watermark, prior_reports=[])

        # Now run() should copy _fetch_degraded_sources to report.stats
        base_report = StageReport(
            stage=StageId.memory_consolidator,
            started_at=datetime.now(UTC),
            completed_at=datetime.now(UTC),
            items_flagged=[],
            stats={},
        )

        with patch.object(BaseStage, 'run', new=AsyncMock(return_value=base_report)):
            report = await stage.run(
                events=[],
                watermark=watermark,
                prior_reports=[],
                run_id='run-assembled-degraded-status',
            )

        assert 'stage1_fetch_degraded' in report.stats, (
            "run() must set report.stats['stage1_fetch_degraded']; key is missing."
        )
        assert report.stats['stage1_fetch_degraded'] == ['status'], (
            f"Expected report.stats['stage1_fetch_degraded'] == ['status'] for assembled path, "
            f"got: {report.stats.get('stage1_fetch_degraded')!r}. "
            "RED: _format_assembled_payload does not append 'status' to _fetch_degraded_sources."
        )
