"""Tests for Stage 1 Memory Consolidator payload behaviour and cross-stage prompt contracts.

Covers:
- project_root threading through assemble_payload / _format_assembled_payload
  (TestStage1PayloadThreadsProjectRootLegacy, TestStage1PayloadThreadsProjectRootAssembled)
  (the former "project_root omitted when empty" cases were removed in task 2146:
  ProjectScope now rejects an empty project_root at construction)
- STAGE2_SYSTEM_PROMPT uniqueness_token mechanism exists (task 1473): minimal existence
  check via build_stage2_system_prompt to guard against the section being dropped
  (TestStage2PromptMandatesUniquenessToken)
- Task 2229 (W5-λ): deterministic Python cycle-summary write to the recon ledger
  (PRD plans/recon-reliability-prd.md §10, boundary test D1) — supersedes the
  former task-1574/1590 CSPRNG summary_nonce injection mechanism
  (TestMemoryConsolidatorDeterministicCycleSummaryWrite)
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
from datetime import UTC, datetime, timedelta
from pathlib import Path
from unittest.mock import AsyncMock, MagicMock, patch

import pytest
import pytest_asyncio

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
from fused_memory.models.scope import ProjectId, ProjectRoot, ProjectScope
from fused_memory.reconciliation.stages.base import BaseStage
from fused_memory.reconciliation.stages.memory_consolidator import MemoryConsolidator
from fused_memory.reconciliation.task_filter import FilteredTaskTree


def _scope(project_id: str, project_root: str) -> ProjectScope:
    """Build a ProjectScope from raw strings — DRYs the many test call sites."""
    return ProjectScope(ProjectId(project_id), ProjectRoot(project_root))


def _rescope(stages, scope: ProjectScope) -> list:
    """Re-scope each pinned stage instance in *stages* to *scope*, in place.

    Local mirror of test_harness.py's `_rescope` — lets this file's single
    `_make_stages` monkeypatch shim (below) honor whatever scope production
    code passes in.
    """
    for s in stages:
        s.scope = scope
    return stages


def _make_consolidator(project_root: str = '/tmp/test') -> MemoryConsolidator:
    """Build a MemoryConsolidator with mocked deps — mirrors test_stages.py ~L1418.

    NOTE: callers must pass a non-empty absolute ``project_root``. Passing
    ``project_root=''`` (the pre-task-2146 "unset root" sentinel) raises
    ``InputValidationError`` from ``ProjectScope.__post_init__`` at this call
    site — a required ``scope`` can no longer carry a falsy root. The former
    empty-root tests were removed accordingly (task 2146).
    """
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
        scope=_scope('test_project', project_root),
    )
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
# Task 2552: pin that _assemble_remediation_payload includes the project_root
# directive — it was the third Stage-1 payload builder, missed when task 2150
# hardened the other two. The legacy time-windowed assemble_payload branch and
# the assembled-path _format_assembled_payload branch already have dedicated
# coverage above (TestStage1PayloadThreadsProjectRootLegacy /
# ...Assembled) — no need to duplicate those here (reviewer finding, amendment
# pass round 1).
# ---------------------------------------------------------------------------


class TestStage1RemediationPayloadIncludesProjectRootDirective:
    """_assemble_remediation_payload includes the project_root directive
    emitted by ``_build_project_root_directive``.

    Covers only the remediation payload (``_assemble_remediation_payload``,
    reached via ``assemble_payload`` when ``remediation_findings`` is set) —
    the gap task 2150 missed. The other two Stage-1 payload builders (legacy
    ``assemble_payload`` and ``_format_assembled_payload``) are already
    covered by ``TestStage1PayloadThreadsProjectRootLegacy`` and
    ``TestStage1PayloadThreadsProjectRootAssembled`` above — task 2552.
    """

    _EXPECTED_ROOT = '/home/leo/src/test_proj'
    _DIRECTIVE = f'Use project_root="{_EXPECTED_ROOT}" for tasks scoped to this project.'

    @pytest.mark.asyncio
    async def test_remediation_payload_includes_directive(self):
        stage = _make_consolidator(project_root=self._EXPECTED_ROOT)
        stage.remediation_findings = []
        watermark = Watermark(project_id='test_project')

        result = await stage.assemble_payload(
            events=[], watermark=watermark, prior_reports=[]
        )

        assert self._DIRECTIVE in result, (
            '_assemble_remediation_payload must include the project_root directive '
            '(task 2552 regression: it was the third builder missed by task 2150)'
        )
        assert result.rstrip().endswith(self._DIRECTIVE), (
            'project_root directive should be the last line of the remediation payload'
        )
        assert result.count('Use project_root=') == 1, (
            'project_root directive should appear exactly once in the remediation payload'
        )


# ---------------------------------------------------------------------------
# project_root omitted when empty — REMOVED (task 2146 / recon-project-scope PRD).
#
# The former class TestStage1PayloadOmitsProjectRootWhenUnset pinned a scenario
# (a stage constructed with project_root='') that ProjectScope (task α, task
# 2144) now makes unconstructable: __post_init__ calls require_project_root(''),
# which raises InputValidationError. Task β deleted the BaseStage '' defaults, so
# no legitimate construction can yield a falsy self.project_root anywhere. The
# defensive `if not self.project_root:` guard in memory_consolidator's
# _build_project_root_directive (and the parallel guard in task_knowledge_sync's
# _render_live_workflow_section) is now dead code; task γ (task 2147) owns any
# cleanup of those branches. These tests exercised an impossible state and are
# deleted rather than migrated.
# ---------------------------------------------------------------------------


# ---------------------------------------------------------------------------
# Task 2229 (W5-λ): deterministic Python cycle-summary write to the recon
# ledger (PRD plans/recon-reliability-prd.md §10, boundary test D1).
#
# Supersedes the former task-1574/1590 CSPRNG summary_nonce injection tests
# (TestStage1PayloadSummaryNonce, deleted here): MemoryConsolidator.run() now
# calls write_cycle_summary in place of the old pretrim_summary_pool call
# plus the verify_cycle_summary_written / reconstruct_cycle_summary_stub
# self-heal chain, and assemble_payload() / _format_assembled_payload() no
# longer inject a nonce section for the LLM to prepend.
# ---------------------------------------------------------------------------


class TestMemoryConsolidatorDeterministicCycleSummaryWrite:
    """MemoryConsolidator.run() writes the authoritative per-cycle
    ``cycle_summary`` directly via the deterministic ``write_cycle_summary``
    helper (task 2229 W5-λ, PRD plans/recon-reliability-prd.md §10, boundary
    test D1) — no LLM turn, no nonce, no verify/reconstruct self-heal.

    RED until step-08 rewires ``run()`` to call ``write_cycle_summary`` in
    place of the ``pretrim_summary_pool`` call plus the
    ``verify_cycle_summary_written`` / ``reconstruct_cycle_summary_stub``
    self-heal chain, and strips the nonce injection from
    ``assemble_payload()`` / ``_format_assembled_payload()`` and the Stage 1
    system prompt.
    """

    @pytest_asyncio.fixture
    async def ledger_store(self, tmp_path):
        from fused_memory.reconciliation.recon_ledger import ReconLedgerStore

        s = ReconLedgerStore(tmp_path / 'reconciliation.db')
        await s.initialize()
        yield s
        await s.close()

    def _base_report(self) -> StageReport:
        return StageReport(
            stage=StageId.memory_consolidator,
            started_at=datetime(2026, 7, 10, 11, 0, 0, tzinfo=UTC),
            completed_at=datetime(2026, 7, 10, 11, 5, 0, tzinfo=UTC),
            items_flagged=[],
            stats={},
            llm_calls=2,
            tokens_used=500,
        )

    async def _run_stage(
        self, ledger_store, run_id: str, *, remediation: bool = False,
    ) -> StageReport:
        stage = _make_consolidator(project_root='/tmp/reify')
        stage.memory.recon_ledger = ledger_store
        if remediation:
            stage.remediation_findings = [{'description': 'fix this'}]

        watermark = Watermark(project_id='test_project')
        with patch.object(BaseStage, 'run', new=AsyncMock(return_value=self._base_report())):
            return await stage.run(
                events=[], watermark=watermark, prior_reports=[], run_id=run_id,
            )

    @pytest.mark.asyncio
    async def test_run_writes_one_authoritative_ledger_row(self, ledger_store):
        """Exactly one cycle_summary ledger row exists for (stage=
        'memory_consolidator', run_id) after run(), written from the report
        Python derives — not an LLM-authored Mem0 write."""
        run_id = 'run-d1-stage1'

        await self._run_stage(ledger_store, run_id)

        record = await ledger_store.get_by_identity(
            'test_project', 'cycle_summary', flag_type='memory_consolidator', run_id=run_id,
        )
        assert record is not None
        assert record.record_kind == 'cycle_summary'
        assert record.task_id == ''
        assert record.flag_type == 'memory_consolidator'
        assert record.run_id == run_id
        assert record.state == 'active'

    @pytest.mark.asyncio
    async def test_run_sets_cycle_summary_ledger_written_stat(self, ledger_store):
        report = await self._run_stage(ledger_store, 'run-d1-stat')

        # Renamed from 'stage1_cycle_summary_written' (reviewer finding
        # observability, task 2229 amendment pass round 2) — the key now
        # makes explicit that it tracks the authoritative ledger write only,
        # not the best-effort Mem0 mirror (see write_cycle_summary's
        # docstring "Returns" section).
        assert report.stats.get('stage1_cycle_summary_ledger_written') == 1

    @pytest.mark.asyncio
    async def test_retired_verify_reconstruct_stats_absent(self, ledger_store):
        report = await self._run_stage(ledger_store, 'run-d1-retired-stats')

        for retired_key in (
            'stage1_cycle_summary_verified_count',
            'stage1_cycle_summary_reconstructed',
            'stage1_cycle_summary_pool_trimmed',
        ):
            assert retired_key not in report.stats, (
                f'{retired_key!r} is retired LLM self-heal telemetry (task 2229 '
                'W5-λ) and must no longer be set by run().'
            )

    @pytest.mark.asyncio
    async def test_remediation_run_writes_no_cycle_summary_row(self, ledger_store):
        """Remediation passes never asked for a cycle_summary before this
        task, and must continue not writing one — self-healing there would
        fabricate a spurious cycle_summary every remediation pass (the
        remediation payload never asks the LLM for one either)."""
        run_id = 'run-d1-remediation'

        await self._run_stage(ledger_store, run_id, remediation=True)

        record = await ledger_store.get_by_identity(
            'test_project', 'cycle_summary', flag_type='memory_consolidator', run_id=run_id,
        )
        assert record is None, (
            'Remediation-pass run() must not write a cycle_summary ledger row.'
        )

    @pytest.mark.asyncio
    async def test_payload_has_no_nonce_text(self):
        """assemble_payload() no longer injects the '### Per-Cycle Summary
        Nonce' section or a 'summary_nonce'/'retry_nonce' line — Python owns
        the per-cycle summary write now, so there is nothing for the LLM to
        prepend a dedup-defeating nonce to."""
        stage = _make_consolidator(project_root='/tmp/reify')
        watermark = Watermark(project_id='test_project')

        payload = await stage.assemble_payload([], watermark, [])

        for forbidden in ('### Per-Cycle Summary Nonce', 'summary_nonce', 'retry_nonce'):
            assert forbidden not in payload, (
                f'assemble_payload() must not contain {forbidden!r} (task 2229 W5-λ).'
            )

    def test_system_prompt_uniqueness_directive_removed(self):
        """The Stage 1 system prompt no longer instructs the LLM to write a
        nonce'd per-cycle summary — the directive block is fully deleted, not
        merely made optional."""
        from fused_memory.reconciliation.prompts.stage1 import STAGE1_SYSTEM_PROMPT

        assert '**Per-Cycle Summary Uniqueness**' not in STAGE1_SYSTEM_PROMPT, (
            "The 'Per-Cycle Summary Uniqueness' directive must be deleted "
            '(task 2229 W5-λ) — Python writes the per-cycle summary '
            'deterministically now.'
        )
        assert '### Per-Cycle Summary Nonce' not in STAGE1_SYSTEM_PROMPT


# ---------------------------------------------------------------------------
# Task 2440: module-level `write_stage1_cycle_summary` helper — extracted
# from the inline write_cycle_summary(...) call in MemoryConsolidator.run()
# (task 2229 W5-λ) so the Stage-1 pool constants (recon_pool, cap,
# trim_source, stage) bind in exactly one place, shared by both the in-stage
# fast-path write and the harness-level raise-path backstop (task 2440).
#
# RED — the helper does not exist yet.
# ---------------------------------------------------------------------------


class TestWriteStage1CycleSummaryHelper:
    """`write_stage1_cycle_summary` writes the authoritative ledger row and
    binds the Stage-1 recon_pool/cap/trim_source/stage constants — parity
    with TestMemoryConsolidatorDeterministicCycleSummaryWrite's in-stage
    coverage above, but exercising the extracted helper directly."""

    @pytest_asyncio.fixture
    async def ledger_store(self, tmp_path):
        from fused_memory.reconciliation.recon_ledger import ReconLedgerStore

        s = ReconLedgerStore(tmp_path / 'reconciliation.db')
        await s.initialize()
        yield s
        await s.close()

    def _report(self) -> StageReport:
        return StageReport(
            stage=StageId.memory_consolidator,
            started_at=datetime(2026, 7, 11, 11, 0, 0, tzinfo=UTC),
            completed_at=datetime(2026, 7, 11, 11, 5, 0, tzinfo=UTC),
            items_flagged=[],
            stats={},
            llm_calls=2,
            tokens_used=500,
        )

    @pytest.mark.asyncio
    async def test_writes_one_ledger_row_and_binds_stage1_pool(self, ledger_store):
        from fused_memory.reconciliation.stages.memory_consolidator import (
            write_stage1_cycle_summary,
        )

        memory_service = AsyncMock()
        memory_service.recon_ledger = ledger_store
        memory_service.get_memories_by_metadata = AsyncMock(return_value=[])

        run_id = 'run-helper-stage1'
        report = self._report()

        written = await write_stage1_cycle_summary(
            memory_service, 'test_project', report, run_id,
        )

        assert written is True

        record = await ledger_store.get_by_identity(
            'test_project', 'cycle_summary', flag_type='memory_consolidator', run_id=run_id,
        )
        assert record is not None
        assert record.record_kind == 'cycle_summary'
        assert record.flag_type == 'memory_consolidator'
        assert record.task_id == ''
        assert record.run_id == run_id

        # Binds the Stage-1 pool constants: the best-effort Mem0 mirror write
        # tags metadata.stage with Stage 1's own stage identifier.
        memory_service.add_system_record.assert_awaited_once()
        assert memory_service.add_system_record.call_args.kwargs['metadata']['stage'] == (
            'memory_consolidator'
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
            TierConfig(), scope=_scope('test_project', '/tmp/x'),
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
        stages = harness._make_stages(_scope('test_project', '/tmp/x'))
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
        harness._make_stages = lambda scope: stages

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
                scope=_scope('test_project', '/tmp/x'),
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

    @pytest.mark.asyncio
    async def test_completion_markers_self_deleted_stat_present_when_zero(self):
        """report.stats['stage1_completion_markers_self_deleted'] is ALWAYS present,
        even when items_flagged contains no completion-marker-annotated flags (== 0).

        RED before step-6: run() does not set this stat at all (task-2312 step-5).
        """
        # project_id='test_project' is already set via the scope passed into
        # _make_consolidator (task 2146) — project_id is now a read-only
        # property, so the old post-construction assignment is deleted.
        stage = _make_consolidator(project_root='/tmp/reify')

        plain_flag = {
            'task_id': '100',
            'flag_type': 'missing_deliverable',
            'description': 'Task 100 has no deliverable',
        }
        base_report = StageReport(
            stage=StageId.memory_consolidator,
            started_at=datetime.now(UTC),
            completed_at=datetime.now(UTC),
            items_flagged=[plain_flag],
            stats={},
        )
        # dedup_flags passes the flag through unannotated (no completion marker).
        dedup_mock = AsyncMock(return_value=[plain_flag])

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
                run_id='run-step5-a',
            )

        assert 'stage1_completion_markers_self_deleted' in report.stats, (
            "run() must ALWAYS set report.stats['stage1_completion_markers_self_deleted']; "
            f"got stats={report.stats!r}. "
            "RED: the stat is not yet wired into run()."
        )
        assert report.stats['stage1_completion_markers_self_deleted'] == 0

    @pytest.mark.asyncio
    async def test_completion_markers_self_deleted_stat_counts_annotated_flags(self):
        """report.stats['stage1_completion_markers_self_deleted'] equals the count of
        flags dedup_flags annotated completion_marker_self_deleted=True.

        RED before step-6: run() does not set this stat at all (task-2312 step-5).
        """
        # project_id='test_project' is already set via the scope passed into
        # _make_consolidator (task 2146) — project_id is now a read-only
        # property, so the old post-construction assignment is deleted.
        stage = _make_consolidator(project_root='/tmp/reify')

        completion_flag = {
            'task_id': '77',
            'flag_type': 'duplicate_flag_marker_cleanup',
            'description': 'cleaned up an orphaned duplicate flag marker',
            'flag_for_stage2': False,
            'completion_marker_self_deleted': True,
            'last_seen_run_id': 'run-step5-b',
        }
        plain_flag = {
            'task_id': '100',
            'flag_type': 'missing_deliverable',
            'description': 'Task 100 has no deliverable',
        }
        base_report = StageReport(
            stage=StageId.memory_consolidator,
            started_at=datetime.now(UTC),
            completed_at=datetime.now(UTC),
            items_flagged=[completion_flag, plain_flag],
            stats={},
        )
        # dedup_flags returns exactly the annotated flags it would produce for this
        # input — the completion flag self-deleted (annotated), the plain flag untouched.
        dedup_mock = AsyncMock(return_value=[completion_flag, plain_flag])

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
                run_id='run-step5-b',
            )

        assert report.stats.get('stage1_completion_markers_self_deleted') == 1, (
            "Expected exactly 1 completion-marker self-delete counted; "
            f"got stats={report.stats!r}"
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
# Task 3709 (PRD δ): index_health stat wiring
# ---------------------------------------------------------------------------


class TestMemoryConsolidatorIndexHealthWiring:
    """MemoryConsolidator.run() must surface index_health in report.stats.

    The stage is a SURFACING POINT ONLY — it deliberately does not file the
    escalation (unlike the HOR/gate-backlog path). The startup sweep has no
    stage, so the filing lives in the harness detector both Q3 paths share; a
    stage-resident filer would fork δ into two divergent detectors.

    step-13 (RED): the index_health attribute and stat-wiring don't exist yet.
    step-14 (GREEN): add class attribute + stat block in run().
    """

    def _base_report(self) -> StageReport:
        return StageReport(
            stage=StageId.memory_consolidator,
            started_at=datetime.now(UTC),
            completed_at=datetime.now(UTC),
            items_flagged=[],
            stats={},
        )

    async def _run(self, stage, run_id: str) -> StageReport:
        dedup_mock = AsyncMock(return_value=[])
        with (
            patch.object(
                BaseStage, 'run', new=AsyncMock(return_value=self._base_report())
            ),
            patch(
                'fused_memory.reconciliation.stages.memory_consolidator.dedup_flags',
                new=dedup_mock,
            ),
        ):
            return await stage.run(
                events=[],
                watermark=Watermark(project_id='test_project'),
                prior_reports=[],
                run_id=run_id,
            )

    @pytest.mark.asyncio
    async def test_index_health_stat_set_when_unhealthy(self):
        """An unhealthy record is carried into report.stats verbatim."""
        stage = _make_consolidator(project_root='/tmp/reify')
        health_record = {
            'healthy': False,
            'missing': [['Entity', 'NODE', 'name', 'RANGE']],
            'unexpected': [],
            'expected_total': 38,
            'actual_total': 37,
        }
        stage.index_health = health_record

        report = await self._run(stage, 'run-3709-ih-unhealthy')

        assert 'index_health' in report.stats, (
            "run() must set report.stats['index_health'] when stage.index_health "
            f'is set; got stats={report.stats!r}'
        )
        assert report.stats['index_health'] == health_record

    @pytest.mark.asyncio
    async def test_index_health_warning_when_unhealthy(self, caplog):
        """run() logs a WARNING naming the project_id, run_id and missing count."""
        import logging

        stage = _make_consolidator(project_root='/tmp/reify')
        stage.index_health = {
            'healthy': False,
            'missing': [['Entity', 'NODE', 'name', 'RANGE']],
            'unexpected': [],
            'expected_total': 38,
            'actual_total': 37,
        }

        dedup_mock = AsyncMock(return_value=[])
        with (
            patch.object(
                BaseStage, 'run', new=AsyncMock(return_value=self._base_report())
            ),
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
                run_id='run-3709-ih-warn',
            )

        drift_warnings = [
            r
            for r in caplog.records
            if r.levelno >= logging.WARNING
            and 'index_drift_stage1' in r.getMessage()
        ]
        assert drift_warnings, (
            'Expected a reconciliation.index_drift_stage1 WARNING when '
            'index_health.healthy=False'
        )
        record = drift_warnings[0]
        assert getattr(record, 'missing_count', None) == 1
        assert getattr(record, 'run_id', None) == 'run-3709-ih-warn'
        assert getattr(record, 'project_id', None) is not None

    @pytest.mark.asyncio
    async def test_healthy_record_is_still_surfaced_without_warning(self, caplog):
        """A HEALTHY record must be observable too — ζ's activation check reads it."""
        import logging

        stage = _make_consolidator(project_root='/tmp/reify')
        health_record = {
            'healthy': True,
            'missing': [],
            'unexpected': [],
            'expected_total': 38,
            'actual_total': 38,
        }
        stage.index_health = health_record

        dedup_mock = AsyncMock(return_value=[])
        with (
            patch.object(
                BaseStage, 'run', new=AsyncMock(return_value=self._base_report())
            ),
            patch(
                'fused_memory.reconciliation.stages.memory_consolidator.dedup_flags',
                new=dedup_mock,
            ),
            caplog.at_level(logging.WARNING),
        ):
            report = await stage.run(
                events=[],
                watermark=Watermark(project_id='test_project'),
                prior_reports=[],
                run_id='run-3709-ih-healthy',
            )

        assert report.stats['index_health'] == health_record, (
            'ζ verifies activation by reading this record, so the healthy case '
            'must be observable, not just the drifted one'
        )
        assert not [
            r
            for r in caplog.records
            if r.levelno >= logging.WARNING
            and 'index_drift_stage1' in r.getMessage()
        ], 'A healthy graph must not warn'

    @pytest.mark.asyncio
    async def test_index_health_key_absent_when_none(self):
        """None must leave the key ABSENT — never a null a consumer reads as 'fine'."""
        stage = _make_consolidator(project_root='/tmp/reify')
        # Explicitly leave index_health at the default (None)

        report = await self._run(stage, 'run-3709-ih-none')

        assert 'index_health' not in report.stats, (
            'When stage.index_health is None the key must be ABSENT — a null '
            'would read as "checked and fine"; got '
            f'stats={report.stats!r}'
        )


# ---------------------------------------------------------------------------
# Tests for task 1938: MemoryConsolidator.run() status_correction_reconciliation
# surfacing
# ---------------------------------------------------------------------------


class TestMemoryConsolidatorStatusCorrectionReconciliationWiring:
    """MemoryConsolidator.run() must surface status_correction_reconciliation
    in report.stats.

    step-17 (RED): status_correction_reconciliation attribute and stat-wiring
        don't exist yet.
    step-18 (GREEN): add class attribute + stat block in run().
    """

    @pytest.mark.asyncio
    async def test_status_correction_reconciliation_stat_set_when_present(self):
        """When stage.status_correction_reconciliation is set, report.stats carries it."""
        stage = _make_consolidator(project_root='/tmp/reify')

        reconciliation_record = {
            'superseded': True, 'diverged': True, 'memory_id': 'x',
        }
        stage.status_correction_reconciliation = reconciliation_record

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
                run_id='run-step17-scr',
            )

        assert 'status_correction_reconciliation' in report.stats, (
            "run() must set report.stats['status_correction_reconciliation'] when "
            "stage.status_correction_reconciliation is set; "
            f"got stats={report.stats!r}"
        )
        assert report.stats['status_correction_reconciliation'] == reconciliation_record

    @pytest.mark.asyncio
    async def test_status_correction_reconciliation_absent_when_none(self):
        """When stage.status_correction_reconciliation is None, the key is absent."""
        stage = _make_consolidator(project_root='/tmp/reify')
        # Explicitly leave status_correction_reconciliation at the default (None)

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
                run_id='run-step17-scr-none',
            )

        assert 'status_correction_reconciliation' not in report.stats, (
            "When stage.status_correction_reconciliation is None, the key must be "
            f"absent from report.stats; got stats={report.stats!r}"
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


# ---------------------------------------------------------------------------
# Task 1977 step-1/step-3: Stage 1 payload renders '### Live-Workflow Signals'
# ---------------------------------------------------------------------------


class TestStage1PayloadLiveWorkflowSignalsSection:
    """assemble_payload() / _format_assembled_payload() render a
    '### Live-Workflow Signals' section when an active task has a live workflow
    (mirrors Stage 2's TestAssemblePayloadLiveWorkflowSignalsSection, task 1655).

    The renderer (_render_live_workflow_section) is imported into
    memory_consolidator.py from task_knowledge_sync.py, so tests monkeypatch
    the detector at its home namespace (tks_module.detect_live_workflow) —
    the same namespace the renderer's module-level call resolves against.

    RED until step-2 wires _build_live_workflow_section() into assemble_payload
    (legacy path); the assembled-path cases (test_assembled_path_*) stay RED
    until step-4 additionally wires it into _format_assembled_payload.
    """

    def _make_tree(self, tasks: list[dict]) -> FilteredTaskTree:
        """Build a FilteredTaskTree with the given tasks as active_tasks."""
        return FilteredTaskTree(
            active_tasks=tasks,
            done_tasks=[],
            cancelled_tasks=[],
            done_count=0,
            cancelled_count=0,
            other_count=0,
            total_count=len(tasks),
            max_task_id=max((t.get('id', 0) for t in tasks), default=0),
        )

    # ── legacy (time-windowed) assemble_payload path ────────────────────

    @pytest.mark.asyncio
    async def test_legacy_path_includes_section_for_live_task(self, monkeypatch):
        """Legacy assemble_payload lists the live task id under '### Live-Workflow Signals'."""
        import fused_memory.reconciliation.stages.task_knowledge_sync as tks_module
        from fused_memory.services.live_workflow_detector import WorkflowLiveness

        live_task_id = '4321'
        not_live_task_id = '100'
        live_task = {'id': int(live_task_id), 'title': 'Live task', 'status': 'in-progress'}
        other_task = {'id': int(not_live_task_id), 'title': 'Other task', 'status': 'blocked'}

        def _fake_detect(task_id, project_root, **kwargs):
            if str(task_id) == live_task_id:
                return WorkflowLiveness(
                    is_live=True,
                    worktree_registered=True,
                    recent_commit=False,
                    orchestrator_live=False,
                    branch=f'task/{live_task_id}',
                    last_commit_at=None,
                )
            return WorkflowLiveness(
                is_live=False,
                worktree_registered=False,
                recent_commit=False,
                orchestrator_live=False,
                branch=f'task/{task_id}',
                last_commit_at=None,
            )

        monkeypatch.setattr(tks_module, 'detect_live_workflow', _fake_detect)

        stage = _make_consolidator(project_root='/project')
        stage.filtered_task_tree = self._make_tree([live_task, other_task])
        watermark = Watermark(project_id='test_project')

        payload = await stage.assemble_payload(events=[], watermark=watermark, prior_reports=[])

        assert '### Live-Workflow Signals' in payload, (
            f"Expected '### Live-Workflow Signals' section in payload; "
            f"got snippet:\n{payload[-800:]!r}"
        )
        assert live_task_id in payload, (
            f"Expected live task id {live_task_id!r} listed under Live-Workflow Signals; "
            f"got snippet:\n{payload[-800:]!r}"
        )

    @pytest.mark.asyncio
    async def test_legacy_path_omits_section_when_no_task_live(self, monkeypatch):
        """Section absent when no active task is live (keeps the payload tight)."""
        import fused_memory.reconciliation.stages.task_knowledge_sync as tks_module
        from fused_memory.services.live_workflow_detector import WorkflowLiveness

        def _fake_detect(task_id, project_root, **kwargs):
            return WorkflowLiveness(
                is_live=False,
                worktree_registered=False,
                recent_commit=False,
                orchestrator_live=False,
                branch=f'task/{task_id}',
                last_commit_at=None,
            )

        monkeypatch.setattr(tks_module, 'detect_live_workflow', _fake_detect)

        stage = _make_consolidator(project_root='/project')
        stage.filtered_task_tree = self._make_tree(
            [{'id': 100, 'title': 'Other', 'status': 'blocked'}]
        )
        watermark = Watermark(project_id='test_project')

        payload = await stage.assemble_payload(events=[], watermark=watermark, prior_reports=[])

        assert '### Live-Workflow Signals' not in payload, (
            f"Expected '### Live-Workflow Signals' absent when no task is live; "
            f"got snippet:\n{payload[-800:]!r}"
        )

    # NOTE: the former test_legacy_path_omits_section_when_project_root_empty was
    # removed (task 2146 / recon-project-scope PRD). It constructed a stage with
    # project_root='' — a state ProjectScope (task α) now rejects at construction,
    # so the "omit section when project_root empty" branch is unreachable dead
    # code (task γ owns any cleanup of the guard in _render_live_workflow_section).

    @pytest.mark.asyncio
    async def test_legacy_path_omits_section_when_no_filtered_task_tree(self, monkeypatch):
        """Section absent when filtered_task_tree is None (the _make_consolidator default)."""
        import fused_memory.reconciliation.stages.task_knowledge_sync as tks_module
        from fused_memory.services.live_workflow_detector import WorkflowLiveness

        def _fake_detect(task_id, project_root, **kwargs):
            return WorkflowLiveness(
                is_live=True,
                worktree_registered=True,
                recent_commit=False,
                orchestrator_live=False,
                branch=f'task/{task_id}',
                last_commit_at=None,
            )

        monkeypatch.setattr(tks_module, 'detect_live_workflow', _fake_detect)

        stage = _make_consolidator(project_root='/project')
        assert stage.filtered_task_tree is None
        watermark = Watermark(project_id='test_project')

        payload = await stage.assemble_payload(events=[], watermark=watermark, prior_reports=[])

        assert '### Live-Workflow Signals' not in payload, (
            f"Expected '### Live-Workflow Signals' absent when filtered_task_tree is None; "
            f"got snippet:\n{payload[-800:]!r}"
        )

    # ── assembled (token-budget) _format_assembled_payload path ─────────

    @pytest.mark.asyncio
    async def test_assembled_path_includes_section_for_live_task(self, monkeypatch):
        """_format_assembled_payload lists the live task id under '### Live-Workflow Signals'.

        Mirrors test_format_assembled_payload_includes_tree_when_set (tests/test_stages.py)
        by calling _format_assembled_payload directly.

        RED after step-2 alone: only assemble_payload's legacy branch is wired; the
        assembled path is a separate method that must be wired independently (step-4).
        """
        import fused_memory.reconciliation.stages.task_knowledge_sync as tks_module
        from fused_memory.services.live_workflow_detector import WorkflowLiveness

        live_task_id = '4321'

        def _fake_detect(task_id, project_root, **kwargs):
            return WorkflowLiveness(
                is_live=True,
                worktree_registered=True,
                recent_commit=False,
                orchestrator_live=False,
                branch=f'task/{live_task_id}',
                last_commit_at=None,
            )

        monkeypatch.setattr(tks_module, 'detect_live_workflow', _fake_detect)

        stage = _make_consolidator(project_root='/project')
        stage.filtered_task_tree = self._make_tree(
            [{'id': int(live_task_id), 'title': 'Live task', 'status': 'in-progress'}]
        )
        stage.assembled_payload = AssembledPayload(events=[], context_items={})
        watermark = Watermark(project_id='test_project')

        payload = await stage._format_assembled_payload(watermark)

        assert '### Live-Workflow Signals' in payload, (
            f"Expected '### Live-Workflow Signals' section in assembled-path payload; "
            f"got snippet:\n{payload[-800:]!r}"
        )
        assert live_task_id in payload, (
            f"Expected live task id {live_task_id!r} listed under Live-Workflow Signals "
            f"in assembled-path payload; got snippet:\n{payload[-800:]!r}"
        )


# ---------------------------------------------------------------------------
# task 2107 step-7 (RED) / step-8 (GREEN): degenerate task-node sweep wiring
# ---------------------------------------------------------------------------


class TestDegenerateTaskNodeSweepWiring:
    """MemoryConsolidator.run() must invoke sweep_degenerate_task_nodes with the
    terminal (done + cancelled) task ids drawn from filtered_task_tree, and
    surface its stats as report.stats['degenerate_task_nodes_swept'] /
    report.stats['degenerate_task_nodes_scanned'].

    RED until step-8 wires extract_terminal_task_ids + sweep_degenerate_task_nodes
    into run().
    """

    @pytest.mark.asyncio
    async def test_run_sweeps_terminal_task_ids_and_surfaces_stats(self):
        """run() awaits the sweep with done+cancelled ids (incl. cancelled 142 & 144)
        and surfaces its scanned/deleted stats under the degenerate_task_nodes_* keys."""
        stage = _make_consolidator(project_root='/tmp/reify')
        stage.filtered_task_tree = FilteredTaskTree(
            done_tasks=[{'id': 148}],
            cancelled_tasks=[{'id': 142}, {'id': 144}],
        )

        base_report = StageReport(
            stage=StageId.memory_consolidator,
            started_at=datetime.now(UTC),
            completed_at=datetime.now(UTC),
            items_flagged=[],
            stats={},
        )
        # dedup_flags passes all flags through unchanged (mirrors the
        # TestStaleCountSnapshotCorrectionWiring pattern); items_flagged is
        # empty here so dedup_flags is not actually invoked, but patching it
        # keeps this test isolated from that unrelated post-processor.
        dedup_mock = AsyncMock(side_effect=lambda **kw: kw['flags'])
        sweep_mock = AsyncMock(
            return_value={'scanned': 3, 'degenerate': 2, 'deleted': 2, 'errors': 0}
        )

        with (
            patch.object(BaseStage, 'run', new=AsyncMock(return_value=base_report)),
            patch(
                'fused_memory.reconciliation.stages.memory_consolidator.dedup_flags',
                new=dedup_mock,
            ),
            patch(
                'fused_memory.reconciliation.stages.memory_consolidator.sweep_degenerate_task_nodes',
                new=sweep_mock,
            ),
        ):
            report = await stage.run(
                events=[],
                watermark=Watermark(project_id='test_project'),
                prior_reports=[],
                run_id='run-2107-step7',
            )

        sweep_mock.assert_awaited_once_with(stage.memory, 'test_project', ['148', '142', '144'])
        assert report.stats.get('degenerate_task_nodes_swept') == 2, (
            f"Expected report.stats['degenerate_task_nodes_swept'] == 2 (stats['deleted']); "
            f'got stats={report.stats!r}. '
            'RED: sweep_degenerate_task_nodes is not yet wired into run().'
        )
        assert report.stats.get('degenerate_task_nodes_scanned') == 3, (
            f"Expected report.stats['degenerate_task_nodes_scanned'] == 3 (stats['scanned']); "
            f'got stats={report.stats!r}'
        )


# ---------------------------------------------------------------------------
# task 2107 step-9 (RED) / step-10 (GREEN): degenerate task-node sweep guards
# ---------------------------------------------------------------------------


class TestDegenerateTaskNodeSweepGuards:
    """Guard/robustness behavior around the run() sweep wiring:

    (a) a remediation pass never sweeps (already true structurally — the
        sweep block sits after the remediation early-return).
    (b) a None filtered_task_tree never sweeps and sets no stats (already
        true — extract_terminal_task_ids(None) == [] short-circuits).
    (c) a sweep failure must be swallowed (best-effort) — run() still
        returns a StageReport and other post-processor stats remain intact.

    RED until step-10 wraps the sweep call in a best-effort try/except; (a)
    and (b) already pass as of step-8 (asserted here as regression locks),
    (c) is the case that actually fails before step-10.
    """

    @pytest.mark.asyncio
    async def test_remediation_pass_never_sweeps(self):
        """remediation_findings set -> run() returns before reaching the sweep block."""
        stage = _make_consolidator(project_root='/tmp/reify')
        stage.remediation_findings = [{'description': 'some finding'}]
        stage.filtered_task_tree = FilteredTaskTree(done_tasks=[{'id': 148}])

        base_report = StageReport(
            stage=StageId.memory_consolidator,
            started_at=datetime.now(UTC),
            completed_at=datetime.now(UTC),
            items_flagged=[],
            stats={},
        )
        sweep_mock = AsyncMock(
            return_value={'scanned': 1, 'degenerate': 1, 'deleted': 1, 'errors': 0}
        )

        with (
            patch.object(BaseStage, 'run', new=AsyncMock(return_value=base_report)),
            patch(
                'fused_memory.reconciliation.stages.memory_consolidator.sweep_degenerate_task_nodes',
                new=sweep_mock,
            ),
        ):
            report = await stage.run(
                events=[],
                watermark=Watermark(project_id='test_project'),
                prior_reports=[],
                run_id='run-2107-step9a',
            )

        sweep_mock.assert_not_awaited()
        assert 'degenerate_task_nodes_swept' not in report.stats

    @pytest.mark.asyncio
    async def test_none_filtered_task_tree_never_sweeps(self):
        """filtered_task_tree left at its None default -> sweep not awaited, no stats set."""
        stage = _make_consolidator(project_root='/tmp/reify')
        assert stage.filtered_task_tree is None

        base_report = StageReport(
            stage=StageId.memory_consolidator,
            started_at=datetime.now(UTC),
            completed_at=datetime.now(UTC),
            items_flagged=[],
            stats={},
        )
        sweep_mock = AsyncMock(
            return_value={'scanned': 1, 'degenerate': 1, 'deleted': 1, 'errors': 0}
        )

        with (
            patch.object(BaseStage, 'run', new=AsyncMock(return_value=base_report)),
            patch(
                'fused_memory.reconciliation.stages.memory_consolidator.sweep_degenerate_task_nodes',
                new=sweep_mock,
            ),
        ):
            report = await stage.run(
                events=[],
                watermark=Watermark(project_id='test_project'),
                prior_reports=[],
                run_id='run-2107-step9b',
            )

        sweep_mock.assert_not_awaited()
        assert 'degenerate_task_nodes_swept' not in report.stats
        assert 'degenerate_task_nodes_scanned' not in report.stats

    @pytest.mark.asyncio
    async def test_sweep_failure_is_swallowed_and_other_stats_remain_intact(self):
        """sweep_degenerate_task_nodes raising must not blow up run() or blank other stats.

        RED: the sweep call has no try/except yet, so the RuntimeError propagates
        out of stage.run() instead of being swallowed.
        """
        stage = _make_consolidator(project_root='/tmp/reify')
        stage.filtered_task_tree = FilteredTaskTree(done_tasks=[{'id': 148}])

        incident_flag = {
            'task_id': None,
            'flag_type': 'count_snapshot_mismatch',
            'description': (
                'Snapshot edge for autopilot_video reports 634/607 but is off by 1; '
                'should be 635/608 to match the Active Task Tree header.'
            ),
            'suggested_action': 'Correct the snapshot edge to 635/608.',
        }
        base_report = StageReport(
            stage=StageId.memory_consolidator,
            started_at=datetime.now(UTC),
            completed_at=datetime.now(UTC),
            items_flagged=[incident_flag],
            stats={},
        )
        dedup_mock = AsyncMock(side_effect=lambda **kw: kw['flags'])
        sweep_mock = AsyncMock(side_effect=RuntimeError('graphiti backend down'))

        with (
            patch.object(BaseStage, 'run', new=AsyncMock(return_value=base_report)),
            patch(
                'fused_memory.reconciliation.stages.memory_consolidator.dedup_flags',
                new=dedup_mock,
            ),
            patch(
                'fused_memory.reconciliation.stages.memory_consolidator.sweep_degenerate_task_nodes',
                new=sweep_mock,
            ),
        ):
            report = await stage.run(
                events=[],
                watermark=Watermark(project_id='test_project'),
                prior_reports=[],
                run_id='run-2107-step9c',
            )

        assert isinstance(report, StageReport), (
            'run() must still return a StageReport when the sweep raises. '
            'RED: the sweep call is not yet wrapped in a best-effort try/except.'
        )
        assert report.stats.get('stale_count_snapshot_corrections_dropped') == 1, (
            'The stale-snapshot-correction post-processor must still have run and set '
            f'its stat even though the sweep raised; got stats={report.stats!r}'
        )
        assert 'degenerate_task_nodes_swept' not in report.stats, (
            'A raised sweep must not leave a partial/incorrect degenerate_task_nodes_swept stat'
        )


# ---------------------------------------------------------------------------
# task 2613 step-11 (RED) / step-12 (GREEN): stale status-snapshot edge sweep wiring
# ---------------------------------------------------------------------------


class TestStaleStatusSnapshotEdgeSweepWiring:
    """MemoryConsolidator.run() must invoke sweep_stale_status_snapshot_edges
    against self.memory/self.taskmaster/self.project_id/self.project_root, and
    surface its stats as report.stats['stale_status_snapshot_edges_invalidated']
    / report.stats['stale_status_snapshot_edges_scanned'].

    RED until step-12 wires sweep_stale_status_snapshot_edges into run().
    """

    @pytest.mark.asyncio
    async def test_run_invalidates_stale_edge_and_surfaces_stats(self):
        """One stale edge (references done task 142) + one healthy edge
        (references still-pending task 999). run() must invalidate only the
        stale edge (via the real sweep_stale_status_snapshot_edges orchestration,
        exercised end-to-end through the wired backend calls) and surface both
        stats keys."""
        stage = _make_consolidator(project_root='/tmp/reify')

        stale_edge = {
            'uuid': 'edge-stale', 'fact': 'Task 142 is an active pending task', 'name': '',
        }
        healthy_edge = {
            'uuid': 'edge-healthy', 'fact': 'Task 999 is an active pending task', 'name': '',
        }
        stage.memory.graphiti.get_all_valid_edges = AsyncMock(
            return_value={'entity-a': [stale_edge, healthy_edge]},
        )
        assert stage.taskmaster is not None  # AsyncMock() from _make_consolidator
        stage.taskmaster.get_statuses = AsyncMock(
            return_value={'142': 'done', '999': 'pending'},
        )
        stage.memory.update_edge = AsyncMock()

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
                watermark=Watermark(project_id='test_project'),
                prior_reports=[],
                run_id='run-2613-step11a',
            )

        stage.memory.update_edge.assert_awaited_once()
        assert stage.memory.update_edge.await_args is not None
        update_call = stage.memory.update_edge.await_args
        assert update_call.args[0] == 'edge-stale', (
            f'Expected update_edge awaited for the stale edge uuid only, got {update_call!r}'
        )

        assert report.stats.get('stale_status_snapshot_edges_invalidated') == 1, (
            f"Expected report.stats['stale_status_snapshot_edges_invalidated'] == 1; "
            f'got stats={report.stats!r}. '
            'RED: sweep_stale_status_snapshot_edges is not yet wired into run().'
        )
        assert report.stats.get('stale_status_snapshot_edges_scanned') == 2, (
            f"Expected report.stats['stale_status_snapshot_edges_scanned'] == 2; "
            f'got stats={report.stats!r}'
        )

    @pytest.mark.asyncio
    async def test_sweep_failure_is_swallowed_and_other_stats_remain_intact(self):
        """sweep_stale_status_snapshot_edges raising must not blow up run() or
        blank other stats — mirrors the degenerate-sweep backstop.

        RED: the sweep call has no try/except yet, so the RuntimeError propagates
        out of stage.run() instead of being swallowed.
        """
        stage = _make_consolidator(project_root='/tmp/reify')

        incident_flag = {
            'task_id': None,
            'flag_type': 'count_snapshot_mismatch',
            'description': (
                'Snapshot edge for autopilot_video reports 634/607 but is off by 1; '
                'should be 635/608 to match the Active Task Tree header.'
            ),
            'suggested_action': 'Correct the snapshot edge to 635/608.',
        }
        base_report = StageReport(
            stage=StageId.memory_consolidator,
            started_at=datetime.now(UTC),
            completed_at=datetime.now(UTC),
            items_flagged=[incident_flag],
            stats={},
        )
        dedup_mock = AsyncMock(side_effect=lambda **kw: kw['flags'])
        sweep_mock = AsyncMock(side_effect=RuntimeError('graphiti backend down'))

        with (
            patch.object(BaseStage, 'run', new=AsyncMock(return_value=base_report)),
            patch(
                'fused_memory.reconciliation.stages.memory_consolidator.dedup_flags',
                new=dedup_mock,
            ),
            patch(
                'fused_memory.reconciliation.stages.memory_consolidator.'
                'sweep_stale_status_snapshot_edges',
                new=sweep_mock,
            ),
        ):
            report = await stage.run(
                events=[],
                watermark=Watermark(project_id='test_project'),
                prior_reports=[],
                run_id='run-2613-step11b',
            )

        assert isinstance(report, StageReport), (
            'run() must still return a StageReport when the sweep raises. '
            'RED: the sweep call is not yet wrapped in a best-effort try/except.'
        )
        assert report.stats.get('stale_count_snapshot_corrections_dropped') == 1, (
            'The stale-snapshot-correction post-processor must still have run and set '
            f'its stat even though the sweep raised; got stats={report.stats!r}'
        )
        assert 'stale_status_snapshot_edges_invalidated' not in report.stats, (
            'A raised sweep must not leave a partial/incorrect '
            'stale_status_snapshot_edges_invalidated stat'
        )
        assert 'stale_status_snapshot_edges_scanned' not in report.stats


class TestStalePriorityOverrideEdgeSweepWiring:
    """MemoryConsolidator.run() must invoke sweep_stale_priority_override_edges
    against self.memory/self.project_id/self.project_root, and surface its
    stats as report.stats['stale_priority_override_edges_invalidated'] /
    report.stats['stale_priority_override_edges_scanned'] (task 2781).

    RED until step-14 wires sweep_stale_priority_override_edges into run().
    """

    @pytest.mark.asyncio
    async def test_run_invalidates_stale_override_edge_and_surfaces_stats(self):
        """One stale priority-override edge (boost for task 5166, absent from
        the live override map) + one healthy edge (boost for task 999, present
        in the live map). run() must invalidate only the stale edge and
        surface both stats keys."""
        stage = _make_consolidator(project_root='/tmp/reify')

        stale_edge = {
            'uuid': 'edge-po-stale',
            'fact': "Set priority override for task 5166: {'boost_tier': 'high'}",
            'name': '',
        }
        healthy_edge = {
            'uuid': 'edge-po-healthy',
            'fact': "Set priority override for task 999: {'boost_tier': 'high'}",
            'name': '',
        }
        stage.memory.graphiti.get_all_valid_edges = AsyncMock(
            return_value={'entity-a': [stale_edge, healthy_edge]},
        )
        stage.memory.update_edge = AsyncMock()

        base_report = StageReport(
            stage=StageId.memory_consolidator,
            started_at=datetime.now(UTC),
            completed_at=datetime.now(UTC),
            items_flagged=[],
            stats={},
        )

        with (
            patch.object(BaseStage, 'run', new=AsyncMock(return_value=base_report)),
            patch(
                'fused_memory.reconciliation.stale_priority_override_edge_sweep.'
                'read_live_override_state',
                new=AsyncMock(return_value={'999': {'ttl_until': None}}),
            ),
        ):
            report = await stage.run(
                events=[],
                watermark=Watermark(project_id='test_project'),
                prior_reports=[],
                run_id='run-2781-step13a',
            )

        assert report.stats.get('stale_priority_override_edges_invalidated') == 1, (
            f"Expected report.stats['stale_priority_override_edges_invalidated'] == 1; "
            f'got stats={report.stats!r}. '
            'RED: sweep_stale_priority_override_edges is not yet wired into run().'
        )
        assert report.stats.get('stale_priority_override_edges_scanned') == 2, (
            f"Expected report.stats['stale_priority_override_edges_scanned'] == 2; "
            f'got stats={report.stats!r}'
        )

    @pytest.mark.asyncio
    async def test_sweep_failure_is_swallowed_and_other_stats_remain_intact(self):
        """sweep_stale_priority_override_edges raising must not blow up run()
        or blank other stats — mirrors the 2613 / degenerate-sweep backstop.

        RED: the sweep call has no try/except yet (nor is the sweep imported
        into memory_consolidator), so the patch target does not exist.
        """
        stage = _make_consolidator(project_root='/tmp/reify')

        # The task 2613 stale-status-snapshot sweep runs immediately before the
        # 2781 sweep and is NOT patched here, so it exercises the real
        # orchestration. Give it an empty valid-edge set so it succeeds (0
        # scanned) and sets its stat — that stat is the "other post-processing
        # was untouched" proof asserted below. Without this mock the 2613 sweep
        # itself fails on an unconfigured get_all_valid_edges and never sets it.
        stage.memory.graphiti.get_all_valid_edges = AsyncMock(return_value={})

        base_report = StageReport(
            stage=StageId.memory_consolidator,
            started_at=datetime.now(UTC),
            completed_at=datetime.now(UTC),
            items_flagged=[],
            stats={},
        )
        sweep_mock = AsyncMock(side_effect=RuntimeError('graphiti backend down'))

        with (
            patch.object(BaseStage, 'run', new=AsyncMock(return_value=base_report)),
            patch(
                'fused_memory.reconciliation.stages.memory_consolidator.'
                'sweep_stale_priority_override_edges',
                new=sweep_mock,
            ),
        ):
            report = await stage.run(
                events=[],
                watermark=Watermark(project_id='test_project'),
                prior_reports=[],
                run_id='run-2781-step13b',
            )

        assert isinstance(report, StageReport), (
            'run() must still return a StageReport when the sweep raises. '
            'RED: the sweep call is not yet wrapped in a best-effort try/except.'
        )
        assert 'stale_priority_override_edges_invalidated' not in report.stats, (
            'A raised sweep must not leave a partial/incorrect '
            'stale_priority_override_edges_invalidated stat'
        )
        assert 'stale_priority_override_edges_scanned' not in report.stats
        # Proof other post-processing was untouched: the task 2613 sweep (which
        # runs immediately before this one) still set its stat.
        assert 'stale_status_snapshot_edges_scanned' in report.stats, (
            'The task 2613 sweep must still have run and set its stat even '
            f'though the 2781 sweep raised; got stats={report.stats!r}'
        )


# ---------------------------------------------------------------------------
# The former class TestMemoryConsolidatorCycleSummaryFallback (task 2366)
# tested the LLM-era verify_cycle_summary_written / reconstruct_cycle_summary_stub
# self-heal chain. Task 2229 (W5-λ) retired that chain in favor of the
# deterministic write_cycle_summary helper — see
# TestMemoryConsolidatorDeterministicCycleSummaryWrite above, which covers the
# replacement behavior (including the remediation-skip case).
# ---------------------------------------------------------------------------


class TestAlreadyTrackedSystemicPatternWiring:
    """MemoryConsolidator.run() must apply filter_already_tracked_systemic_patterns
    to items_flagged (task 2416), dropping a systemic_pattern 'never tracked'
    finding when a done dark_factory task already covers the idea, and
    surfacing report.stats['systemic_pattern_already_tracked_dropped'].

    Hardens against the e61b38f9/1938 false-positive incident: a finding
    claimed the 'diff project_status_correction cache vs live get_statuses
    every cycle' idea was never tracked, despite dark_factory task 1938
    (done, merged 2026-07-01) already implementing it — spawning duplicate
    dark_factory task 2412.

    RED until step-10 wires filter_already_tracked_systemic_patterns into
    run() and sets the stat.
    """

    def _make_never_tracked_flag(self) -> dict:
        return {
            'task_id': None,
            'category': 'systemic_pattern',
            'flag_type': 'systemic_pattern',
            'description': (
                'This systemic pattern was never converted to a tracked task: diff '
                'the project_status_correction cache against live get_statuses every '
                'cycle to catch drift.'
            ),
            'suggested_action': (
                'File a task to diff the cache against live status each cycle.'
            ),
        }

    def _make_matching_done_task(self) -> dict:
        return {
            'id': '1938',
            'status': 'done',
            'title': (
                'Diff project_status_correction cache against live get_statuses '
                'every cycle'
            ),
            'description': (
                'Implemented a periodic diff of the cached project_status_correction '
                'value against a live get_statuses call each cycle to catch drift and '
                'correct stale cache entries before they propagate.'
            ),
        }

    @pytest.mark.asyncio
    async def test_already_tracked_finding_dropped_and_benign_survives(self):
        """The e61b38f9/1938 false finding is dropped; benign finding survives."""
        stage = _make_consolidator(project_root='/tmp/reify')
        stage.known_projects = {'dark_factory': '/df'}
        assert stage.taskmaster is not None  # AsyncMock() from _make_consolidator
        stage.taskmaster.get_tasks = AsyncMock(return_value={
            'tasks': [self._make_matching_done_task()],
        })

        never_tracked_flag = self._make_never_tracked_flag()
        benign_flag = {
            'task_id': '100',
            'flag_type': 'missing_deliverable',
            'description': 'Task 100 has no deliverable',
        }
        all_flags = [never_tracked_flag, benign_flag]

        base_report = StageReport(
            stage=StageId.memory_consolidator,
            started_at=datetime.now(UTC),
            completed_at=datetime.now(UTC),
            items_flagged=list(all_flags),
            stats={},
        )
        # dedup_flags passes all flags through unchanged so we can inspect
        # the already-tracked filter's effect directly.
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
                run_id='run-2416-step9',
            )

        assert never_tracked_flag not in report.items_flagged, (
            'systemic_pattern never-tracked finding must be DROPPED when done '
            "dark_factory task 1938 already covers the idea; got "
            f'items_flagged={report.items_flagged!r}. '
            'RED: filter_already_tracked_systemic_patterns is not yet wired into run().'
        )
        assert benign_flag in report.items_flagged, (
            f'Benign missing_deliverable flag must survive; got {report.items_flagged!r}'
        )
        assert report.stats.get('systemic_pattern_already_tracked_dropped') == 1, (
            "run() must set report.stats['systemic_pattern_already_tracked_dropped'] = 1 "
            f'when one already-tracked finding is dropped; got stats={report.stats!r}. '
            'RED: stat not yet surfaced.'
        )

    @pytest.mark.asyncio
    async def test_finding_kept_when_dark_factory_not_a_known_project(self):
        """No-op guard: finding survives when known_projects lacks 'dark_factory'.

        Exercises the fail-open path when the harness never registers
        dark_factory (e.g. a single-project test/deployment harness) — the
        resolved dark_factory_root is None, so the filter must degrade to a
        pass-through rather than erroring or dropping anything.
        """
        stage = _make_consolidator(project_root='/tmp/reify')
        stage.known_projects = {}  # dark_factory not registered
        assert stage.taskmaster is not None  # AsyncMock() from _make_consolidator
        stage.taskmaster.get_tasks = AsyncMock(return_value={
            'tasks': [self._make_matching_done_task()],
        })

        never_tracked_flag = self._make_never_tracked_flag()
        base_report = StageReport(
            stage=StageId.memory_consolidator,
            started_at=datetime.now(UTC),
            completed_at=datetime.now(UTC),
            items_flagged=[never_tracked_flag],
            stats={},
        )
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
                run_id='run-2416-step9-noop',
            )

        assert never_tracked_flag in report.items_flagged, (
            'Finding must be KEPT when dark_factory is not in known_projects '
            f'(no-op guard); got items_flagged={report.items_flagged!r}'
        )


class TestMemoryConsolidatorCitationVerificationWiring:
    """MemoryConsolidator.run() must re-verify each flagged finding's cited
    Mem0 memories (verify_cited_memories) AFTER super().run() and BEFORE the
    remediation early-return, so a phantom (non-resolving) mem0 citation is
    stripped + marked, and the stage1_* citation stats are always present on
    report.stats (task 2978).

    Driven through the remediation path: its early-return fires immediately
    after the verifier, isolating the verifier wiring from the full-path
    filter chain AND proving the check runs BEFORE that early-return (so both
    full and remediation passes are covered). RED until step-8 wires
    verify_cited_memories into run().
    """

    @pytest.mark.asyncio
    async def test_phantom_citation_stripped_and_marked_via_run(self):
        stage = _make_consolidator(project_root='/tmp/reify')
        # Remediation mode: run() early-returns right after the verifier, before
        # the full-path filter chain.
        stage.remediation_findings = [{'description': 'remediation'}]

        async def _get(project_id, memory_id):
            # 'good-id' resolves; 'phantom-id' is genuinely not found.
            if memory_id == 'good-id':
                return {'id': 'good-id', 'content': 'x', 'metadata': {}}
            return None

        assert stage.memory is not None  # AsyncMock() from _make_consolidator
        stage.memory.get_memory_by_id = AsyncMock(side_effect=_get)  # type: ignore[union-attr]

        finding = {
            'description': 'a stage-1 finding',
            'severity': 'medium',
            'cited_memories': [
                {'memory_id': 'good-id', 'store': 'mem0'},
                {'memory_id': 'phantom-id', 'store': 'mem0'},
            ],
        }
        base_report = StageReport(
            stage=StageId.memory_consolidator,
            started_at=datetime.now(UTC),
            completed_at=datetime.now(UTC),
            items_flagged=[finding],
            stats={},
        )

        with patch.object(BaseStage, 'run', new=AsyncMock(return_value=base_report)):
            report = await stage.run(
                events=[],
                watermark=Watermark(project_id='test_project'),
                prior_reports=[],
                run_id='run-citation-verify',
            )

        verified_finding = report.items_flagged[0]
        # The phantom is stripped; the resolving id remains.
        assert [c['memory_id'] for c in verified_finding['cited_memories']] == ['good-id'], (
            'run() must strip the non-resolving mem0 citation via verify_cited_memories; '
            f'got cited_memories={verified_finding.get("cited_memories")!r}. '
            'RED: verify_cited_memories is not yet wired into run().'
        )
        # The dropped phantom is named on the finding.
        assert verified_finding.get('citation_failures') == [
            {'memory_id': 'phantom-id', 'store': 'mem0', 'reason': 'memory_not_found'},
        ]
        # Citation stats are merged onto the returned report (present on this
        # remediation path — and, by construction, on the full path too).
        assert report.stats['stage1_phantom_citations_dropped'] == 1
        assert 'stage1_citations_verified' in report.stats
        assert 'stage1_citation_verification_errors' in report.stats


# ---------------------------------------------------------------------------
# Gate-backlog age-check wiring in MemoryConsolidator.run (task 3017)
# ---------------------------------------------------------------------------


_GATE_FIXED_NOW = datetime(2026, 7, 24, 12, 0, 0, tzinfo=UTC)


class _FrozenGateDatetime(datetime):
    """datetime subclass whose .now() returns a fixed instant (task 3017 wiring).

    Patched onto the memory_consolidator module so the gate-backlog block's
    ``datetime.now(UTC)`` is deterministic; ``gate_escalated_at`` stamps are
    built relative to ``_GATE_FIXED_NOW``.
    """

    @classmethod
    def now(cls, tz=None):
        return _GATE_FIXED_NOW


def _blocked_gate_task(tid, *, hours_ago: float) -> dict:
    """A blocked human-decision gate task aged ``hours_ago`` relative to fixed-now."""
    stamp = (_GATE_FIXED_NOW - timedelta(hours=hours_ago)).isoformat()
    return {
        'id': tid,
        'status': 'blocked',
        'title': f'Gate task {tid}',
        'metadata': {'operational_mode': 'gate', 'gate_escalated_at': stamp},
    }


class TestMemoryConsolidatorGateBacklogWiring:
    """MemoryConsolidator.run() files a level-1 reconciliation_stale_gate_backlog
    escalation for a blocked gate task whose gate_escalated_at has aged past 48h,
    and records stage1_gate_backlog_stalled / stage1_gate_backlog_escalated stats.

    Reuses the real-EscalationQueue-on-tmp_path + filtered_task_tree wiring harness
    and freezes datetime.now(UTC) in the memory_consolidator module so the age is
    deterministic.  RED until step-8 wires the gate-backlog block into run().
    """

    def _make_tree(self, tasks: list[dict]) -> FilteredTaskTree:
        return FilteredTaskTree(
            active_tasks=tasks,
            done_tasks=[],
            cancelled_tasks=[],
            done_count=0,
            cancelled_count=0,
            other_count=0,
            total_count=len(tasks),
            max_task_id=0,
        )

    def _base_report(self) -> StageReport:
        return StageReport(
            stage=StageId.memory_consolidator,
            started_at=datetime.now(UTC),
            completed_at=datetime.now(UTC),
            items_flagged=[],
            stats={},
        )

    async def _run(self, stage, monkeypatch, run_id: str) -> StageReport:
        monkeypatch.setattr(
            'fused_memory.reconciliation.stages.memory_consolidator.datetime',
            _FrozenGateDatetime,
        )
        with (
            patch.object(
                BaseStage, 'run', new=AsyncMock(return_value=self._base_report())
            ),
            patch(
                'fused_memory.reconciliation.stages.memory_consolidator.dedup_flags',
                new=AsyncMock(return_value=[]),
            ),
        ):
            return await stage.run(
                events=[],
                watermark=Watermark(project_id='test_project'),
                prior_reports=[],
                run_id=run_id,
            )

    @pytest.mark.asyncio
    async def test_stale_gate_backlog_files_l1_and_sets_stats(self, tmp_path, monkeypatch):
        """(a) blocked gate aged 49h → one pending L1 gate-backlog + stalled=1, escalated=1."""
        from escalation.queue import EscalationQueue

        stage = _make_consolidator(project_root='/tmp/reify')
        queue = EscalationQueue(tmp_path / 'gate_esc')
        stage._escalation_queue = queue
        stage.filtered_task_tree = self._make_tree([_blocked_gate_task('645', hours_ago=49)])

        report = await self._run(stage, monkeypatch, 'run-gate-a')

        pending = queue.get_by_task('645', status='pending', level=1)
        gate_backlog = [
            e for e in pending if e.category == 'reconciliation_stale_gate_backlog'
        ]
        assert len(gate_backlog) == 1, (
            'run() must file exactly one level-1 reconciliation_stale_gate_backlog '
            f'escalation for the >48h blocked gate task; got {pending!r}. '
            'RED: the gate-backlog block is not yet wired into run().'
        )
        assert gate_backlog[0].task_id == '645'
        assert report.stats['stage1_gate_backlog_stalled'] == 1
        assert report.stats['stage1_gate_backlog_escalated'] == 1

    @pytest.mark.asyncio
    async def test_fresh_gate_no_escalation_stats_zero(self, tmp_path, monkeypatch):
        """(b) control: gate aged 1h (< 48h) → no gate-backlog L1; both stats 0."""
        from escalation.queue import EscalationQueue

        stage = _make_consolidator(project_root='/tmp/reify')
        queue = EscalationQueue(tmp_path / 'gate_esc')
        stage._escalation_queue = queue
        stage.filtered_task_tree = self._make_tree([_blocked_gate_task('645', hours_ago=1)])

        report = await self._run(stage, monkeypatch, 'run-gate-b')

        assert not any(
            e.category == 'reconciliation_stale_gate_backlog' for e in queue.get_pending()
        )
        assert report.stats['stage1_gate_backlog_stalled'] == 0
        assert report.stats['stage1_gate_backlog_escalated'] == 0

    @pytest.mark.asyncio
    async def test_none_tree_noops(self, tmp_path, monkeypatch):
        """(c1) filtered_task_tree is None → block no-ops: no escalation, no raise, no stat."""
        from escalation.queue import EscalationQueue

        stage = _make_consolidator(project_root='/tmp/reify')
        queue = EscalationQueue(tmp_path / 'gate_esc')
        stage._escalation_queue = queue
        stage.filtered_task_tree = None

        report = await self._run(stage, monkeypatch, 'run-gate-c1')

        assert not any(
            e.category == 'reconciliation_stale_gate_backlog' for e in queue.get_pending()
        )
        assert 'stage1_gate_backlog_stalled' not in report.stats

    @pytest.mark.asyncio
    async def test_none_queue_noops(self, monkeypatch):
        """(c2) _escalation_queue is None → block no-ops: no raise, no stat."""
        stage = _make_consolidator(project_root='/tmp/reify')
        stage._escalation_queue = None
        stage.filtered_task_tree = self._make_tree([_blocked_gate_task('645', hours_ago=49)])

        report = await self._run(stage, monkeypatch, 'run-gate-c2')

        assert 'stage1_gate_backlog_stalled' not in report.stats
