"""Tests for the recon_report MCP namespace scaffold (task α).

Covers:
- Config schema fields (TestConfigSchema)
- ReconReportState happy path (TestReconReportStateHappyPath)
- In-run dedup (TestReconReportInRunDedup)
- complete() idempotence (TestReconReportCompleteIdempotence)
- Unknown run_id errors (TestReconReportRunIdMismatch)
- State isolation between runs (TestReconReportStateIsolation)
- TTL reaper (TestReconReportReaper)
- FastMCP server factory (TestCreateReconReportServer)
- Main.py boot helper (TestReconReportBoot)
- Reaper not started by _build_recon_report_components (TestReconReportReaperWiredAtBoot)
"""

import pytest


# ---------------------------------------------------------------------------
# step-1: Config schema — RED until step-2 adds the fields
# ---------------------------------------------------------------------------


class TestConfigSchema:
    """Verify the two new config fields exist with correct defaults."""

    def test_server_config_recon_report_port_default(self):
        from fused_memory.config.schema import ServerConfig

        cfg = ServerConfig()
        assert cfg.recon_report_port == 8003

    def test_reconciliation_config_ttl_default(self):
        from fused_memory.config.schema import ReconciliationConfig

        cfg = ReconciliationConfig()
        assert cfg.recon_report_state_ttl_seconds == 300


# ---------------------------------------------------------------------------
# step-3: ReconReportState happy path — RED until step-4 creates the module
# ---------------------------------------------------------------------------


class TestReconReportStateHappyPath:
    """Drive ReconReportState through a full lifecycle without spinning up MCP."""

    def _make_state(self):
        from fused_memory.server.recon_report import ReconReportState

        t = [0.0]
        return ReconReportState(ttl_seconds=300, clock=lambda: t[0]), t

    def test_start_report_then_get_assembled(self):
        state, _ = self._make_state()
        state.start_report(run_id='r1', stage='memory_consolidator', project_id='dark_factory')
        report = state.get_assembled_report('r1', 'memory_consolidator')
        assert report is not None
        assert report['summary'] == ''
        assert report['stats'] == {}
        assert report['flagged_items'] == []

    def test_add_finding_returns_uuid(self):
        state, _ = self._make_state()
        state.start_report(run_id='r1', stage='memory_consolidator', project_id='dark_factory')
        result = state.add_finding(
            run_id='r1',
            severity='moderate',
            category='memory_stale',
            description='d',
            suggested_action='a',
            actionable=True,
            task_id='42',
            flag_type='orphaned_knowledge',
        )
        assert 'finding_id' in result
        assert len(result['finding_id']) == 36  # uuid4 canonical form

    def test_set_stat_and_inc_stat(self):
        state, _ = self._make_state()
        state.start_report(run_id='r1', stage='memory_consolidator', project_id='dark_factory')
        state.set_stat('r1', 'scanned', 100)
        result = state.inc_stat('r1', 'scanned', 5)
        assert result == {'value': 105}

    def test_complete_returns_flagged_count_and_stats(self):
        state, _ = self._make_state()
        state.start_report(run_id='r1', stage='memory_consolidator', project_id='dark_factory')
        state.add_finding(
            run_id='r1',
            severity='moderate',
            category='memory_stale',
            description='d',
            suggested_action='a',
            actionable=True,
            task_id='42',
            flag_type='orphaned_knowledge',
        )
        state.set_stat('r1', 'scanned', 100)
        state.inc_stat('r1', 'scanned', 5)
        result = state.complete('r1', 'summary text')
        assert result == {'flagged_count': 1, 'stats': {'scanned': 105}}

    def test_assembled_report_full_shape_after_complete(self):
        state, _ = self._make_state()
        state.start_report(run_id='r1', stage='memory_consolidator', project_id='dark_factory')
        fid = state.add_finding(
            run_id='r1',
            severity='moderate',
            category='memory_stale',
            description='d',
            suggested_action='a',
            actionable=True,
            task_id='42',
            flag_type='orphaned_knowledge',
        )['finding_id']
        state.set_stat('r1', 'scanned', 100)
        state.inc_stat('r1', 'scanned', 5)
        state.complete('r1', 'summary text')

        report = state.get_assembled_report('r1', 'memory_consolidator')
        assert report is not None
        assert report['summary'] == 'summary text'
        assert report['stats'] == {'scanned': 105}
        assert len(report['flagged_items']) == 1

        item = report['flagged_items'][0]
        assert item['finding_id'] == fid
        assert item['severity'] == 'moderate'
        assert item['category'] == 'memory_stale'
        assert item['description'] == 'd'
        assert item['suggested_action'] == 'a'
        assert item['actionable'] is True
        assert item['task_id'] == '42'
        assert item['flag_type'] == 'orphaned_knowledge'
        # cited_* must be present as empty lists in α (task β populates them)
        assert item['cited_entities'] == []
        assert item['cited_edges'] == []
        assert item['cited_tasks'] == []
        assert item['cited_memories'] == []


# ---------------------------------------------------------------------------
# step-5: In-run dedup — RED until step-6 wires the dedup logic
# ---------------------------------------------------------------------------


class TestReconReportInRunDedup:
    """Verify (task_id, flag_type) dedup inside a single (run_id, stage) pair."""

    def _make_state(self):
        from fused_memory.server.recon_report import ReconReportState

        t = [0.0]
        state = ReconReportState(ttl_seconds=300, clock=lambda: t[0])
        state.start_report(run_id='r1', stage='memory_consolidator', project_id='dark_factory')
        return state

    def _finding(self, state, task_id='42', flag_type='orphaned_knowledge', **kwargs):
        defaults = dict(
            run_id='r1',
            severity='moderate',
            category='memory_stale',
            description='d',
            suggested_action='a',
            actionable=True,
            task_id=task_id,
            flag_type=flag_type,
        )
        defaults.update(kwargs)
        return state.add_finding(**defaults)

    def test_second_same_sig_returns_duplicate_error(self):
        state = self._make_state()
        first = self._finding(state, task_id='42', flag_type='orphaned_knowledge')
        assert 'finding_id' in first
        id1 = first['finding_id']

        second = self._finding(state, task_id='42', flag_type='orphaned_knowledge',
                               description='different text still same sig')
        assert second['error'] == 'duplicate_finding'
        assert second['error_type'] == 'ReconReportDuplicateFinding'
        assert second['existing_finding_id'] == id1

    def test_different_sig_both_succeed(self):
        state = self._make_state()
        r1 = self._finding(state, task_id='42', flag_type='orphaned_knowledge')
        r2 = self._finding(state, task_id='99', flag_type='stale_edge')
        assert 'finding_id' in r1
        assert 'finding_id' in r2
        assert r1['finding_id'] != r2['finding_id']

    def test_both_none_no_dedup(self):
        """Two (None, None) findings are both allocated (informational)."""
        state = self._make_state()
        r1 = self._finding(state, task_id=None, flag_type=None)
        r2 = self._finding(state, task_id=None, flag_type=None)
        assert 'finding_id' in r1
        assert 'finding_id' in r2
        assert r1['finding_id'] != r2['finding_id']
