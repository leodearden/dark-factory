"""Tests for the Stage 2 recon_report polling channel (task-1966 scope item 2).

Stage 2 additionally polls recon_report findings for the current run_id — an
independent channel that does NOT pass through filter_suppressed, so
systemic_pattern findings survive even when the Mem0 flagged-items channel is
suppressed by a task_id-scoped stage1_flag_suppression record.
"""

from __future__ import annotations

import logging
from datetime import UTC, datetime
from types import SimpleNamespace
from unittest.mock import AsyncMock

import pytest

# ---------------------------------------------------------------------------
# _query_recon_report_findings — RED until step-11 adds the helper
# ---------------------------------------------------------------------------


class _FakeReconReportState:
    """Minimal stand-in for ReconReportState exposing get_findings_for_run."""

    def __init__(self, findings=None, *, raises: Exception | None = None):
        self._findings = findings or []
        self._raises = raises
        self.calls: list[str] = []

    def get_findings_for_run(self, run_id: str):
        self.calls.append(run_id)
        if self._raises is not None:
            raise self._raises
        return self._findings


class TestQueryReconReportFindings:
    """Tests for _query_recon_report_findings(recon_report_state, run_id)."""

    def test_none_state_returns_empty(self):
        """(a) recon_report_state is None -> returns [] without raising.

        This only asserts the empty-return; there is no observable object to
        assert a "no call" contract against when state is None (None has no
        ``get_findings_for_run`` to spy on). The guard is the early
        ``if recon_report_state is None: return []`` check in the helper —
        see test_filters_to_systemic_pattern_category_only below for a case
        that positively confirms ``get_findings_for_run`` IS called (and
        exactly once) when a real state object is supplied."""
        from fused_memory.reconciliation.stages.task_knowledge_sync import (
            _query_recon_report_findings,
        )

        result = _query_recon_report_findings(None, 'r1')
        assert result == []

    def test_filters_to_systemic_pattern_category_only(self):
        """(b) Given findings mixing category=='systemic_pattern' and other
        categories, only the systemic_pattern findings are returned."""
        from fused_memory.reconciliation.stages.task_knowledge_sync import (
            _query_recon_report_findings,
        )

        systemic = {
            'finding_id': 'f1',
            'category': 'systemic_pattern',
            'task_id': '452',
            'flag_type': 'live_workflow_recurrence_counter_needed',
            'description': 'd1',
        }
        other = {
            'finding_id': 'f2',
            'category': 'task_memory_mismatch',
            'task_id': '99',
            'flag_type': 'memory_task_mismatch',
            'description': 'd2',
        }
        state = _FakeReconReportState(findings=[systemic, other])

        result = _query_recon_report_findings(state, 'r1')

        assert result == [systemic]
        assert state.calls == ['r1']

    def test_get_findings_for_run_exception_returns_empty_and_warns(self, caplog):
        """(c) get_findings_for_run raising -> [] + WARNING (best-effort,
        mirrors _query_stage2_flags)."""
        from fused_memory.reconciliation.stages.task_knowledge_sync import (
            _query_recon_report_findings,
        )

        state = _FakeReconReportState(raises=RuntimeError('recon_report down'))

        with caplog.at_level(
            logging.WARNING,
            logger='fused_memory.reconciliation.stages.task_knowledge_sync',
        ):
            result = _query_recon_report_findings(state, 'r1')

        assert result == []
        warning_messages = [r.message for r in caplog.records if r.levelno >= logging.WARNING]
        assert any('_query_recon_report_findings' in m for m in warning_messages), (
            f'Expected WARNING mentioning _query_recon_report_findings but got: {warning_messages}'
        )


# ---------------------------------------------------------------------------
# _flag_content_fingerprint — cross-channel content-fingerprint dedup key
# (task-2078). RED until task_knowledge_sync defines the helper.
# ---------------------------------------------------------------------------


class TestFlagContentFingerprint:
    """Tests for _flag_content_fingerprint(item: dict) -> str | None.

    Pure helper used to cross-channel-dedup a polled recon_report_systemic
    finding (keyed on ``description``) against combined_flags items already
    assembled from Stage 1 structured output or a surviving Mem0
    active-query flag (both keyed on ``content``) — finding_id is never
    available on the Mem0 side, so a normalized-content fingerprint is the
    only reliable join key across the two channels.
    """

    def test_content_key_returns_content_fingerprint(self):
        """(a) An item with a `content` key returns
        flag_dedup._content_fingerprint(content)."""
        from fused_memory.reconciliation.flag_dedup import _content_fingerprint
        from fused_memory.reconciliation.stages.task_knowledge_sync import (
            _flag_content_fingerprint,
        )

        text = 'live workflow recurrence counter needed for task 452'

        assert _flag_content_fingerprint({'content': text}) == _content_fingerprint(text)

    def test_description_only_key_returns_fingerprint_of_description(self):
        """(b) An item with only a `description` key (no `content`) returns
        the fingerprint of that description."""
        from fused_memory.reconciliation.flag_dedup import _content_fingerprint
        from fused_memory.reconciliation.stages.task_knowledge_sync import (
            _flag_content_fingerprint,
        )

        text = 'systemic pattern: repeated live workflow recurrence for task 452'

        assert _flag_content_fingerprint({'description': text}) == _content_fingerprint(text)

    @pytest.mark.parametrize(
        'item',
        [
            {},
            {'content': ''},
            {'content': '   '},
            {'description': ''},
            {'description': '   \n\t  '},
            {'content': None, 'description': None},
        ],
        ids=[
            'missing_both',
            'blank_content',
            'whitespace_content',
            'blank_description',
            'whitespace_description',
            'explicit_none_values',
        ],
    )
    def test_blank_or_missing_text_returns_none(self, item):
        """(c) blank / whitespace-only / missing text returns None (mirrors
        compute_content_fingerprint_signature's non-blank gate)."""
        from fused_memory.reconciliation.stages.task_knowledge_sync import (
            _flag_content_fingerprint,
        )

        assert _flag_content_fingerprint(item) is None

    def test_cross_channel_equality_content_vs_description(self):
        """(d) CROSS-CHANNEL EQUALITY — a mem0-style item {'content': X} and a
        recon-style item {'description': X} with identical text X produce the
        SAME fingerprint. This is the join property the cross-channel dedup
        in assemble_payload relies on."""
        from fused_memory.reconciliation.stages.task_knowledge_sync import (
            _flag_content_fingerprint,
        )

        text = 'live workflow recurrence counter needed for task 452'
        mem0_style = {'_source': 'mem0_active_query', 'content': text}
        recon_style = {'_source': 'recon_report_systemic', 'description': text}

        fp_mem0 = _flag_content_fingerprint(mem0_style)
        fp_recon = _flag_content_fingerprint(recon_style)

        assert fp_mem0 is not None
        assert fp_mem0 == fp_recon

    def test_different_text_produces_different_fingerprint(self):
        """(e) items with different text produce different fingerprints."""
        from fused_memory.reconciliation.stages.task_knowledge_sync import (
            _flag_content_fingerprint,
        )

        fp1 = _flag_content_fingerprint({'content': 'finding text A'})
        fp2 = _flag_content_fingerprint({'content': 'finding text B'})

        assert fp1 != fp2


# ---------------------------------------------------------------------------
# assemble_payload integration — recon_report channel reaches Stage 2 even
# when the Mem0 channel is suppressed (task-1966 step-12)
#
# RED until step-13 wires _query_recon_report_findings into assemble_payload.
# ---------------------------------------------------------------------------


def _make_configured_stage(
    deps: dict, *, project_id: str, project_root: str, run_id: str = 'test-run'
):
    """Mirror tests/test_stages.py's make_configured_task_knowledge_sync_stage."""
    from fused_memory.models.reconciliation import StageId
    from fused_memory.reconciliation.stages.task_knowledge_sync import TaskKnowledgeSync

    stage = TaskKnowledgeSync(StageId.task_knowledge_sync, **deps)
    stage.project_id = project_id
    stage.project_root = project_root
    stage._current_run_id = run_id
    return stage


class TestAssemblePayloadReconReportChannel:
    """Integration: assemble_payload merges systemic_pattern recon_report
    findings into the flagged section, independent of Mem0-channel suppression."""

    @pytest.fixture
    def mock_deps(self):
        from fused_memory.config.schema import ReconciliationConfig

        config = ReconciliationConfig(enabled=True, explore_codebase_root='/tmp/test')
        memory_service = AsyncMock()
        memory_service.search.return_value = []  # _query_stage2_flags yields nothing
        memory_service.count_memories_by_metadata.return_value = 0
        taskmaster = AsyncMock()
        taskmaster.get_tasks.return_value = {'tasks': []}
        journal = AsyncMock()
        journal.get_run.return_value = SimpleNamespace(started_at=datetime.now(UTC))
        return {
            'memory_service': memory_service,
            'taskmaster': taskmaster,
            'journal': journal,
            'config': config,
        }

    @pytest.fixture
    def watermark(self):
        from fused_memory.models.reconciliation import Watermark
        return Watermark(project_id='autopilot_video')

    def _systemic_finding(
        self, *, task_id='452', flag_type='live_workflow_recurrence_counter_needed'
    ):
        return {
            'finding_id': 'rr-finding-1',
            'severity': 'moderate',
            'category': 'systemic_pattern',
            'description': 'live workflow recurrence counter needed for task 452',
            'suggested_action': 'file a tracking task',
            'actionable': True,
            'task_id': task_id,
            'flag_type': flag_type,
            'cited_entities': [],
            'cited_edges': [],
            'cited_tasks': [],
            'cited_memories': [],
        }

    def _stage1_report(self, items_flagged):
        from fused_memory.models.reconciliation import StageId, StageReport

        now = datetime.now(UTC)
        return StageReport(
            stage=StageId.memory_consolidator,
            started_at=now,
            completed_at=now,
            items_flagged=items_flagged,
            stats={},
            llm_calls=1,
            tokens_used=100,
        )

    @pytest.mark.asyncio
    async def test_systemic_finding_reaches_stage2_despite_mem0_suppression(
        self, mock_deps, watermark
    ):
        """A systemic_pattern finding absent from stage1_report.items_flagged
        (simulating Mem0-channel suppression) still reaches the Stage 2 payload
        via the recon_report channel, tagged _source='recon_report_systemic'."""
        stage = _make_configured_stage(
            mock_deps, project_id='autopilot_video', project_root='/home/leo/src/autopilot-video'
        )
        finding = self._systemic_finding()
        stage._recon_report_state = _FakeReconReportState(findings=[finding])
        stage1_report = self._stage1_report(items_flagged=[])  # Mem0 channel: suppressed, nothing here

        payload = await stage.assemble_payload([], watermark, [stage1_report])

        assert '452' in payload
        assert 'live_workflow_recurrence_counter_needed' in payload
        assert 'recon_report_systemic' in payload
        assert stage._recon_report_systemic_polled == 1

    @pytest.mark.asyncio
    async def test_finding_already_in_stage1_report_not_double_rendered(
        self, mock_deps, watermark
    ):
        """When the same (task_id, flag_type) is already in stage1_report.items_flagged,
        the recon_report channel poll must not render it a second time."""
        stage = _make_configured_stage(
            mock_deps, project_id='autopilot_video', project_root='/home/leo/src/autopilot-video'
        )
        finding = self._systemic_finding()
        stage._recon_report_state = _FakeReconReportState(findings=[finding])
        stage1_report = self._stage1_report(items_flagged=[
            {
                'task_id': '452',
                'flag_type': 'live_workflow_recurrence_counter_needed',
                'description': 'already flagged by stage 1',
            },
        ])

        payload = await stage.assemble_payload([], watermark, [stage1_report])

        assert payload.count('live_workflow_recurrence_counter_needed') == 1
        assert stage._recon_report_systemic_polled == 0

    @pytest.mark.asyncio
    async def test_systemic_finding_deduped_against_surviving_mem0_flag(
        self, mock_deps, watermark
    ):
        """Regression (reviewer finding dedup_gap): a surviving Mem0
        active-query flag (mem0_active_query source, from a live
        flag_for_stage2 marker) carries flag_type sourced from the marker's
        own metadata, so compute_flag_signature can dedup the recon_report
        channel poll against it too — not only against the Stage 1
        structured-output channel. Without surfacing flag_type here, this
        finding would be rendered a second time via the recon_report poll."""
        mock_deps['memory_service'].search.return_value = [
            SimpleNamespace(
                id='mem0-marker-1',
                content='live workflow recurrence counter needed for task 452',
                metadata={
                    'flag_for_stage2': True,
                    'task_id': '452',
                    'flag_type': 'live_workflow_recurrence_counter_needed',
                    'run_id': 'test-run',
                },
                created_at=None,
            ),
        ]
        stage = _make_configured_stage(
            mock_deps, project_id='autopilot_video', project_root='/home/leo/src/autopilot-video'
        )
        finding = self._systemic_finding()
        stage._recon_report_state = _FakeReconReportState(findings=[finding])
        stage1_report = self._stage1_report(items_flagged=[])  # Stage 1 channel empty

        payload = await stage.assemble_payload([], watermark, [stage1_report])

        assert payload.count('live_workflow_recurrence_counter_needed') == 1
        assert stage._recon_report_systemic_polled == 0
