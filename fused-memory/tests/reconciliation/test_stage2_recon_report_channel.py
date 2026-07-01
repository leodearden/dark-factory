"""Tests for the Stage 2 recon_report polling channel (task-1966 scope item 2).

Stage 2 additionally polls recon_report findings for the current run_id — an
independent channel that does NOT pass through filter_suppressed, so
systemic_pattern findings survive even when the Mem0 flagged-items channel is
suppressed by a task_id-scoped stage1_flag_suppression record.
"""

from __future__ import annotations

import logging

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

    def test_none_state_returns_empty_no_call(self):
        """(a) recon_report_state is None -> returns [] and performs no call (no error)."""
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
