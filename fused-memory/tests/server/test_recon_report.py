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

    def test_server_config_port_collision_raises(self):
        """recon_report_port == port must be a config-time error (suggestion 3)."""
        import pytest
        from pydantic import ValidationError

        from fused_memory.config.schema import ServerConfig

        with pytest.raises(ValidationError, match='recon_report_port'):
            ServerConfig(port=8002, recon_report_port=8002)


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

    def _finding(self, state, task_id: str | None = '42', flag_type: str | None = 'orphaned_knowledge', **kwargs):
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
        """Two (None, None) findings with DISTINCT descriptions are both allocated."""
        state = self._make_state()
        r1 = self._finding(state, task_id=None, flag_type=None, description='first observation')
        r2 = self._finding(state, task_id=None, flag_type=None, description='second observation')
        assert 'finding_id' in r1
        assert 'finding_id' in r2
        assert r1['finding_id'] != r2['finding_id']


# ---------------------------------------------------------------------------
# step-7: complete() idempotence — RED until step-8 implements it
# ---------------------------------------------------------------------------


class TestReconReportCompleteIdempotence:
    """Verify PRD §9.2 idempotence rules for complete()."""

    def _make_state(self):
        from fused_memory.server.recon_report import ReconReportState

        t = [0.0]
        state = ReconReportState(ttl_seconds=300, clock=lambda: t[0])
        state.start_report(run_id='r1', stage='s1', project_id='dark_factory')
        state.add_finding(
            run_id='r1',
            severity='low',
            category='cat',
            description='d',
            suggested_action='a',
            task_id='1',
            flag_type='f',
        )
        state.set_stat('r1', 'k', 7)
        return state, t

    def test_first_complete_stamps_summary(self):
        state, _ = self._make_state()
        result = state.complete('r1', 'summary A')
        assert result == {'flagged_count': 1, 'stats': {'k': 7}}
        report = state.get_assembled_report('r1', 's1')
        assert report is not None
        assert report['summary'] == 'summary A'

    def test_second_same_summary_is_noop(self):
        state, _ = self._make_state()
        r1 = state.complete('r1', 'summary A')
        r2 = state.complete('r1', 'summary A')
        assert r2 == r1  # identical response
        report = state.get_assembled_report('r1', 's1')
        assert report is not None
        assert report['summary'] == 'summary A'
        assert report['summary_warnings'] == []

    def test_second_different_summary_warns_does_not_overwrite(self):
        state, _ = self._make_state()
        state.complete('r1', 'summary A')
        result = state.complete('r1', 'summary B')
        # Response is the cached one, not an error
        assert result == {'flagged_count': 1, 'stats': {'k': 7}}
        report = state.get_assembled_report('r1', 's1')
        assert report is not None
        # Original summary preserved
        assert report['summary'] == 'summary A'
        # Warning recorded
        assert len(report['summary_warnings']) == 1
        assert 'summary B' in report['summary_warnings'][0]


# ---------------------------------------------------------------------------
# amend: Post-completion mutation guard (suggestion 4)
# ---------------------------------------------------------------------------


class TestReconReportPostCompletionGuard:
    """add_finding / set_stat / inc_stat must be rejected after complete() stamps
    completed_at, preventing silent corruption of the assembled report.
    """

    def _make_completed_state(self):
        from fused_memory.server.recon_report import ReconReportState

        t = [0.0]
        state = ReconReportState(ttl_seconds=300, clock=lambda: t[0])
        state.start_report(run_id='r1', stage='s1', project_id='p')
        state.add_finding(
            run_id='r1',
            severity='low',
            category='c',
            description='d',
            suggested_action='a',
            task_id='1',
            flag_type='f',
        )
        state.complete('r1', 'summary')
        return state

    def test_add_finding_after_complete_is_rejected(self):
        state = self._make_completed_state()
        result = state.add_finding(
            run_id='r1',
            severity='high',
            category='c',
            description='late finding',
            suggested_action='a',
        )
        assert result['error'] == 'report_already_completed'
        assert result['error_type'] == 'ReconReportAlreadyCompleted'
        # The finding must NOT have been added
        report = state.get_assembled_report('r1', 's1')
        assert report is not None
        assert len(report['flagged_items']) == 1  # only the original one

    def test_set_stat_after_complete_is_rejected(self):
        state = self._make_completed_state()
        result = state.set_stat('r1', 'new_key', 42)
        assert result['error'] == 'report_already_completed'
        assert result['error_type'] == 'ReconReportAlreadyCompleted'
        # Stat must NOT have been written
        report = state.get_assembled_report('r1', 's1')
        assert report is not None
        assert 'new_key' not in report['stats']

    def test_inc_stat_after_complete_is_rejected(self):
        state = self._make_completed_state()
        result = state.inc_stat('r1', 'counter', 5)
        assert result['error'] == 'report_already_completed'
        assert result['error_type'] == 'ReconReportAlreadyCompleted'


# ---------------------------------------------------------------------------
# amend: inc_stat type mismatch (suggestion 5)
# ---------------------------------------------------------------------------


class TestReconReportIncStatTypeMismatch:
    """inc_stat on a string-valued stat key must return a structured error,
    not silently coerce the string to 0 and lose the original value.
    """

    def _make_state(self):
        from fused_memory.server.recon_report import ReconReportState

        t = [0.0]
        state = ReconReportState(ttl_seconds=300, clock=lambda: t[0])
        state.start_report(run_id='r1', stage='s1', project_id='p')
        return state

    def test_inc_stat_on_string_valued_key_returns_error(self):
        state = self._make_state()
        state.set_stat('r1', 'label', 'some-string')
        result = state.inc_stat('r1', 'label', 1)
        assert result['error'] == 'stat_type_mismatch'
        assert result['error_type'] == 'ReconReportStatTypeMismatch'
        assert result['key'] == 'label'
        assert result['current_type'] == 'str'

    def test_inc_stat_preserves_original_string(self):
        """The string value must not be overwritten by the failed inc."""
        state = self._make_state()
        state.set_stat('r1', 'label', 'keep-me')
        state.inc_stat('r1', 'label', 1)  # rejected
        report = state.get_assembled_report('r1', 's1')
        assert report is not None
        assert report['stats']['label'] == 'keep-me'

    def test_inc_stat_on_numeric_key_still_works(self):
        """Normal numeric increment path is unaffected."""
        state = self._make_state()
        state.set_stat('r1', 'count', 10)
        result = state.inc_stat('r1', 'count', 3)
        assert result == {'value': 13}

    def test_inc_stat_on_absent_key_initialises_to_zero(self):
        """Missing key is still treated as 0 (original behaviour)."""
        state = self._make_state()
        result = state.inc_stat('r1', 'fresh', 7)
        assert result == {'value': 7}


# ---------------------------------------------------------------------------
# step-9: Unknown run_id returns structured error — RED until step-10
# ---------------------------------------------------------------------------


class TestReconReportRunIdMismatch:
    """All mutation tools return run_id_unknown for an unregistered run_id."""

    _ERR = {'error': 'run_id_unknown', 'error_type': 'ReconReportRunUnknown'}

    def _make_state(self):
        from fused_memory.server.recon_report import ReconReportState

        t = [0.0]
        return ReconReportState(ttl_seconds=300, clock=lambda: t[0])

    def test_add_finding_unknown_run_id(self):
        state = self._make_state()
        result = state.add_finding(
            run_id='ghost',
            severity='low',
            category='cat',
            description='d',
            suggested_action='a',
        )
        assert result['error'] == 'run_id_unknown'
        assert result['error_type'] == 'ReconReportRunUnknown'

    def test_set_stat_unknown_run_id(self):
        state = self._make_state()
        result = state.set_stat('ghost', 'k', 1)
        assert result['error'] == 'run_id_unknown'

    def test_inc_stat_unknown_run_id(self):
        state = self._make_state()
        result = state.inc_stat('ghost', 'k', 1)
        assert result['error'] == 'run_id_unknown'

    def test_complete_unknown_run_id(self):
        state = self._make_state()
        result = state.complete('ghost', 'summary')
        assert result['error'] == 'run_id_unknown'

    def test_get_assembled_report_unknown_returns_none(self):
        state = self._make_state()
        assert state.get_assembled_report('ghost', 'some_stage') is None


# ---------------------------------------------------------------------------
# step-11: State isolation between two concurrent runs
# ---------------------------------------------------------------------------


class TestReconReportStateIsolation:
    """Two runs must never see each other's findings or stats."""

    def _make_state(self):
        from fused_memory.server.recon_report import ReconReportState

        t = [0.0]
        return ReconReportState(ttl_seconds=300, clock=lambda: t[0])

    def test_findings_are_isolated(self):
        state = self._make_state()
        state.start_report('r1', 'stage1', 'p')
        state.start_report('r2', 'stage2', 'p')

        state.add_finding(run_id='r1', severity='low', category='c',
                          description='r1-finding', suggested_action='a',
                          task_id='1', flag_type='f1')
        state.add_finding(run_id='r2', severity='high', category='c',
                          description='r2-finding', suggested_action='a',
                          task_id='2', flag_type='f2')

        r1 = state.get_assembled_report('r1', 'stage1')
        r2 = state.get_assembled_report('r2', 'stage2')
        assert r1 is not None
        assert r2 is not None
        assert len(r1['flagged_items']) == 1
        assert r1['flagged_items'][0]['description'] == 'r1-finding'
        assert len(r2['flagged_items']) == 1
        assert r2['flagged_items'][0]['description'] == 'r2-finding'

    def test_stats_are_isolated(self):
        state = self._make_state()
        state.start_report('r1', 's1', 'p')
        state.start_report('r2', 's2', 'p')
        state.set_stat('r1', 'scanned', 10)
        state.set_stat('r2', 'scanned', 99)
        r1 = state.get_assembled_report('r1', 's1')
        r2 = state.get_assembled_report('r2', 's2')
        assert r1 is not None
        assert r2 is not None
        assert r1['stats'] == {'scanned': 10}
        assert r2['stats'] == {'scanned': 99}

    def test_finding_ids_are_distinct(self):
        state = self._make_state()
        state.start_report('r1', 's1', 'p')
        state.start_report('r2', 's2', 'p')
        id1 = state.add_finding(run_id='r1', severity='low', category='c',
                                 description='d', suggested_action='a',
                                 task_id='1', flag_type='f')['finding_id']
        id2 = state.add_finding(run_id='r2', severity='low', category='c',
                                 description='d', suggested_action='a',
                                 task_id='1', flag_type='f')['finding_id']
        assert id1 != id2

    def test_completing_one_does_not_affect_other(self):
        state = self._make_state()
        state.start_report('r1', 's1', 'p')
        state.start_report('r2', 's2', 'p')
        state.complete('r1', 'done r1')
        r2 = state.get_assembled_report('r2', 's2')
        assert r2 is not None
        assert r2['summary'] == ''  # still in-progress


# ---------------------------------------------------------------------------
# step-13: TTL reaper via tick() — RED until step-14
# ---------------------------------------------------------------------------


class TestReconReportReaper:
    """tick() evicts completed entries past TTL; in-progress entries are immortal."""

    def _make_state(self, ttl=300):
        from fused_memory.server.recon_report import ReconReportState

        t = [0.0]
        state = ReconReportState(ttl_seconds=ttl, clock=lambda: t[0])
        return state, t

    def test_tick_before_ttl_keeps_entry(self):
        state, t = self._make_state(ttl=300)
        state.start_report('r1', 's1', 'p')
        state.complete('r1', 'done')
        t[0] = 100.0
        evicted = state.tick()
        assert evicted == 0
        assert state.get_assembled_report('r1', 's1') is not None

    def test_tick_after_ttl_evicts_entry(self):
        state, t = self._make_state(ttl=300)
        state.start_report('r1', 's1', 'p')
        state.complete('r1', 'done')
        t[0] = 301.0
        evicted = state.tick()
        assert evicted == 1
        assert state.get_assembled_report('r1', 's1') is None

    def test_inprogress_not_evicted_by_ttl(self):
        state, t = self._make_state(ttl=300)
        state.start_report('r1', 's1', 'p')
        # Do NOT call complete — entry remains in-progress
        t[0] = 1_000_000.0
        evicted = state.tick()
        assert evicted == 0
        assert state.get_assembled_report('r1', 's1') is not None

    def test_tick_returns_count(self):
        state, t = self._make_state(ttl=300)
        state.start_report('r1', 's1', 'p')
        state.start_report('r2', 's2', 'p')
        state.complete('r1', 'done')
        state.complete('r2', 'done')
        t[0] = 301.0
        evicted = state.tick()
        assert evicted == 2


# ---------------------------------------------------------------------------
# step-15: FastMCP server factory — RED until step-16
# ---------------------------------------------------------------------------


class TestCreateReconReportServer:
    """Verify create_recon_report_server wires all five tools to state."""

    def _make(self):
        from fused_memory.server.recon_report import ReconReportState, create_recon_report_server

        t = [0.0]
        state = ReconReportState(ttl_seconds=300, clock=lambda: t[0])
        mcp = create_recon_report_server(state)
        return state, mcp

    def test_returns_fastmcp_instance(self):
        from mcp.server.fastmcp import FastMCP

        _, mcp = self._make()
        assert isinstance(mcp, FastMCP)

    def test_mcp_name_is_recon_report(self):
        _, mcp = self._make()
        assert mcp.name == 'Recon Report'

    def test_all_five_tools_registered(self):
        _, mcp = self._make()
        tools = set(mcp._tool_manager._tools.keys())
        assert {'start_report', 'add_finding', 'set_stat', 'inc_stat', 'complete'} <= tools

    @pytest.mark.asyncio
    async def test_end_to_end_via_call_tool(self):
        """Drive the full lifecycle through mcp._tool_manager.call_tool."""
        state, mcp = self._make()
        tm = mcp._tool_manager

        # start_report
        result = await tm.call_tool('start_report', {
            'run_id': 'r1', 'stage': 'memory_consolidator', 'project_id': 'dark_factory',
        })
        assert isinstance(result, dict)
        assert result.get('run_id') == 'r1'

        # add_finding
        result = await tm.call_tool('add_finding', {
            'run_id': 'r1',
            'severity': 'low',
            'category': 'cat',
            'description': 'd',
            'suggested_action': 'a',
            'task_id': '42',
            'flag_type': 'f',
        })
        assert 'finding_id' in result

        # complete
        result = await tm.call_tool('complete', {'run_id': 'r1', 'summary': 'done'})
        assert 'flagged_count' in result

        # Verify state mutated
        report = state.get_assembled_report('r1', 'memory_consolidator')
        assert report is not None
        assert report['summary'] == 'done'
        assert len(report['flagged_items']) == 1


# ---------------------------------------------------------------------------
# step-17: _build_recon_report_components helper in main.py — RED until step-18
# ---------------------------------------------------------------------------


class TestReconReportBoot:
    """Verify _build_recon_report_components returns correct types and config."""

    def _make_config(self, recon_port=8003, ttl=300):
        """Build a minimal FusedMemoryConfig sufficient for the helper."""
        from fused_memory.config.schema import (
            FusedMemoryConfig,
            ReconciliationConfig,
            ServerConfig,
        )

        return FusedMemoryConfig(
            server=ServerConfig(recon_report_port=recon_port, host='127.0.0.1'),
            reconciliation=ReconciliationConfig(recon_report_state_ttl_seconds=ttl),
        )

    def test_returns_three_tuple(self):
        from fused_memory.server.main import _build_recon_report_components

        config = self._make_config()
        result = _build_recon_report_components(config)
        assert len(result) == 3

    def test_state_type_and_ttl(self):
        from fused_memory.server.main import _build_recon_report_components
        from fused_memory.server.recon_report import ReconReportState

        config = self._make_config(ttl=600)
        state, _, _ = _build_recon_report_components(config)
        assert isinstance(state, ReconReportState)
        assert state._ttl_seconds == 600

    def test_mcp_name_is_recon_report(self):
        from mcp.server.fastmcp import FastMCP

        from fused_memory.server.main import _build_recon_report_components

        config = self._make_config()
        _, mcp, _ = _build_recon_report_components(config)
        assert isinstance(mcp, FastMCP)
        assert mcp.name == 'Recon Report'

    def test_uvicorn_config_port(self):
        import uvicorn

        from fused_memory.server.main import _build_recon_report_components

        config = self._make_config(recon_port=8099)
        _, _, uv_cfg = _build_recon_report_components(config)
        assert isinstance(uv_cfg, uvicorn.Config)
        assert uv_cfg.port == 8099

    def test_uvicorn_config_host(self):
        from fused_memory.server.main import _build_recon_report_components

        config = self._make_config()
        _, _, uv_cfg = _build_recon_report_components(config)
        assert uv_cfg.host == '127.0.0.1'

    def test_asgi_shield_applied(self):
        from fused_memory.server.main import _ASGIExceptionShield, _build_recon_report_components

        config = self._make_config()
        _, _, uv_cfg = _build_recon_report_components(config)
        assert isinstance(uv_cfg.app, _ASGIExceptionShield)

    def test_safe_tool_wrapper_applied(self):
        from fused_memory.server.main import _build_recon_report_components

        config = self._make_config()
        _, mcp, _ = _build_recon_report_components(config)
        assert getattr(mcp._tool_manager, '_fused_memory_safe_wrapped', False)

    def test_json_404_handler_applied(self):
        """The recon_report Starlette app must have the JSON HTTPException handler
        registered (same guard as the primary server — suggestion 2).

        The handler lives on the inner starlette app (uv_cfg.app.app) because
        _ASGIExceptionShield wraps the app at the outer ASGI layer.
        """
        from starlette.exceptions import HTTPException

        from fused_memory.server.main import _build_recon_report_components

        config = self._make_config()
        _, _, uv_cfg = _build_recon_report_components(config)
        inner_app = uv_cfg.app.app  # _ASGIExceptionShield.app → the Starlette app
        assert HTTPException in inner_app.exception_handlers


# ---------------------------------------------------------------------------
# step-19: Reaper interface exposed but NOT started by _build_recon_report_components
# ---------------------------------------------------------------------------


class TestReconReportReaperWiredAtBoot:
    """Verify the reaper interface contract and that the factory stays test-friendly."""

    def _make_config(self):
        from fused_memory.config.schema import (
            FusedMemoryConfig,
            ReconciliationConfig,
            ServerConfig,
        )

        return FusedMemoryConfig(
            server=ServerConfig(recon_report_port=8003, host='127.0.0.1'),
            reconciliation=ReconciliationConfig(recon_report_state_ttl_seconds=300),
        )

    def test_state_exposes_start_reaper_callable(self):
        from fused_memory.server.main import _build_recon_report_components

        config = self._make_config()
        state, _, _ = _build_recon_report_components(config)
        assert callable(state.start_reaper)

    def test_state_exposes_stop_reaper_callable(self):
        from fused_memory.server.main import _build_recon_report_components

        config = self._make_config()
        state, _, _ = _build_recon_report_components(config)
        assert callable(state.stop_reaper)

    def test_state_exposes_tick_callable(self):
        from fused_memory.server.main import _build_recon_report_components

        config = self._make_config()
        state, _, _ = _build_recon_report_components(config)
        assert callable(state.tick)

    def test_reaper_not_started_by_factory(self):
        """_build_recon_report_components must NOT start the reaper task.

        Starting the reaper is run_server()'s responsibility; the factory
        must remain safe to call in unit tests (no asyncio.Task created).
        """
        from fused_memory.server.main import _build_recon_report_components

        config = self._make_config()
        state, _, _ = _build_recon_report_components(config)
        assert state._reaper_task is None


# ---------------------------------------------------------------------------
# step-21: Shutdown callback stops both servers — RED until step-22
# ---------------------------------------------------------------------------


class TestReconReportShutdownCallback:
    """Regression guard: SIGTERM/SIGINT must stop both primary AND recon_server.

    Prior to step-22, _install_operator_stop_handler received
    ``lambda: setattr(server, 'should_exit', True)``, which flipped only the
    primary server's should_exit. asyncio.gather(server.serve(),
    recon_server.serve()) therefore never resolved, the finally-block teardown
    was never reached, and run_server() hung until SIGKILL.
    """

    class _FakeServer:
        """Minimal fake uvicorn.Server exposing should_exit."""

        def __init__(self) -> None:
            self.should_exit = False

    def test_callback_stops_both_servers(self):
        from fused_memory.server.main import _make_operator_stop_callback

        primary = self._FakeServer()
        recon = self._FakeServer()

        cb = _make_operator_stop_callback(primary, recon)

        # Before invocation — both still running
        assert primary.should_exit is False
        assert recon.should_exit is False

        cb()

        # After invocation — BOTH must have been stopped
        assert primary.should_exit is True
        assert recon.should_exit is True

    def test_callback_with_single_server(self):
        """Verify the variadic signature works with just one server too."""
        from fused_memory.server.main import _make_operator_stop_callback

        srv = self._FakeServer()
        cb = _make_operator_stop_callback(srv)
        cb()
        assert srv.should_exit is True

    def test_callback_with_no_servers(self):
        """Verify graceful no-op when called with zero servers."""
        from fused_memory.server.main import _make_operator_stop_callback

        cb = _make_operator_stop_callback()
        cb()  # Must not raise


# ---------------------------------------------------------------------------
# task-1568: Cross-stage in-run dedup — RED until step-2 extends add_finding
# ---------------------------------------------------------------------------


class TestReconReportCrossStageDedup:
    """Extend the §9.2 in-run dedup so it spans stage boundaries within a run_id.

    Within one reconciliation cycle (shared run_id), Stage 2 must not be able
    to allocate a second finding row for a (task_id, flag_type) already filed
    by Stage 1. Cross-run isolation must be preserved: two runs with the same
    signature must still produce distinct finding_ids.
    """

    def _make_state(self):
        from fused_memory.server.recon_report import ReconReportState

        t = [0.0]
        return ReconReportState(ttl_seconds=300, clock=lambda: t[0]), t

    def _finding(
        self,
        state,
        run_id: str = 'r1',
        task_id: str | None = '3803',
        flag_type: str | None = 'task_stuck_pending_merge',
        **kwargs,
    ):
        defaults = dict(
            run_id=run_id,
            severity='high',
            category='stuck_task',
            description='d',
            suggested_action='a',
            actionable=True,
            task_id=task_id,
            flag_type=flag_type,
        )
        defaults.update(kwargs)
        return state.add_finding(**defaults)

    def test_cross_stage_same_sig_returns_duplicate_error(self):
        """Stage 2 filing the same (task_id, flag_type) as Stage 1 must return
        duplicate_finding with existing_finding_id pointing at Stage 1's finding.
        """
        state, _ = self._make_state()

        # Stage 1 — memory_consolidator
        state.start_report(run_id='r1', stage='memory_consolidator', project_id='dark_factory')
        result1 = self._finding(state, run_id='r1')
        assert 'finding_id' in result1, f'Stage 1 add_finding failed: {result1}'
        finding_id_a = result1['finding_id']

        # Stage 2 — task_knowledge_sync (same run_id, different stage)
        state.start_report(run_id='r1', stage='task_knowledge_sync', project_id='dark_factory')
        result2 = self._finding(state, run_id='r1')

        # Must be detected as a cross-stage duplicate
        assert result2.get('error') == 'duplicate_finding', (
            f'Expected duplicate_finding but got: {result2}'
        )
        assert result2['error_type'] == 'ReconReportDuplicateFinding'
        assert result2['existing_finding_id'] == finding_id_a

        # Stage 2's report must be empty (dup never entered it)
        s2_report = state.get_assembled_report('r1', 'task_knowledge_sync')
        assert s2_report is not None
        assert s2_report['flagged_items'] == []

        # Stage 1's report must still have exactly one finding
        s1_report = state.get_assembled_report('r1', 'memory_consolidator')
        assert s1_report is not None
        assert len(s1_report['flagged_items']) == 1
        assert s1_report['flagged_items'][0]['finding_id'] == finding_id_a

    def test_cross_stage_different_sig_both_allocate(self):
        """Different (task_id, flag_type) signatures across two stages of the same
        run must both succeed — cross-stage dedup must not over-suppress.
        """
        state, _ = self._make_state()

        # Stage 1
        state.start_report(run_id='r1', stage='memory_consolidator', project_id='dark_factory')
        r1 = self._finding(state, run_id='r1', task_id='42', flag_type='memory_task_mismatch')
        assert 'finding_id' in r1, f'Stage 1 add_finding failed: {r1}'

        # Stage 2 — different signature
        state.start_report(run_id='r1', stage='task_knowledge_sync', project_id='dark_factory')
        r2 = self._finding(state, run_id='r1', task_id='99', flag_type='missing_completion_memory')
        assert 'finding_id' in r2, f'Stage 2 add_finding failed: {r2}'

        assert r1['finding_id'] != r2['finding_id']

    def test_cross_run_same_sig_not_deduped(self):
        """Two different run_ids with the same (task_id, flag_type) must each
        allocate distinct finding_ids — cross-run isolation must be preserved.
        """
        state, _ = self._make_state()

        state.start_report(run_id='r1', stage='memory_consolidator', project_id='dark_factory')
        res1 = self._finding(state, run_id='r1')
        assert 'finding_id' in res1, f'Run r1 add_finding failed: {res1}'

        state.start_report(run_id='r2', stage='memory_consolidator', project_id='dark_factory')
        res2 = self._finding(state, run_id='r2')
        assert 'finding_id' in res2, f'Run r2 add_finding failed: {res2}'

        assert res1['finding_id'] != res2['finding_id']


# ---------------------------------------------------------------------------
# task-1568: Cross-stage finding citability — RED until step-4 widens
#            _resolve_finding to scan all (run_id, *) entries
# ---------------------------------------------------------------------------


class TestReconReportCrossStageCitability:
    """A finding_id returned via a cross-stage duplicate_finding must remain
    citable from the stage that received the duplicate_finding response.

    After Stage 2 gets duplicate_finding(existing_finding_id=A) where A was
    created in Stage 1, Stage 2 should be able to cite_entity(..., finding_id=A)
    to attach citations to Stage 1's finding.  Currently _resolve_finding only
    searches the *active* stage's entry, so cite_entity returns finding_unknown.
    """

    def _make_state_with_fake_services(self):
        """Return a ReconReportState wired with a fake memory service."""
        from fused_memory.server.recon_report import ReconReportState

        class _FakeMemSvc:
            async def get_entity(self, name: str, project_id: str) -> dict:
                return {'nodes': [{'uuid': 'aaaa-bbbb', 'name': name}]}

        t = [0.0]
        state = ReconReportState(
            ttl_seconds=300,
            clock=lambda: t[0],
            memory_service=_FakeMemSvc(),
        )
        return state, t

    @pytest.mark.asyncio
    async def test_cross_stage_duplicate_finding_is_citable(self):
        """cite_entity on a finding from a prior stage must succeed, not return
        finding_unknown, when the active stage has switched to a later stage.
        """
        state, _ = self._make_state_with_fake_services()

        # Stage 1: file finding A
        state.start_report(run_id='r1', stage='memory_consolidator', project_id='dark_factory')
        res1 = state.add_finding(
            run_id='r1',
            severity='high',
            category='stuck_task',
            description='task is stuck',
            suggested_action='investigate',
            task_id='3803',
            flag_type='task_stuck_pending_merge',
        )
        assert 'finding_id' in res1, f'Stage 1 add_finding failed: {res1}'
        finding_id_a = res1['finding_id']

        # Stage 2: receives duplicate_finding pointing at A
        state.start_report(run_id='r1', stage='task_knowledge_sync', project_id='dark_factory')
        dup = state.add_finding(
            run_id='r1',
            severity='high',
            category='stuck_task',
            description='same task is stuck (Stage 2 view)',
            suggested_action='investigate',
            task_id='3803',
            flag_type='task_stuck_pending_merge',
        )
        assert dup.get('error') == 'duplicate_finding'
        assert dup['existing_finding_id'] == finding_id_a

        # Stage 2 is now active; try to cite onto Stage 1's finding_id
        cite_result = await state.cite_entity(
            run_id='r1',
            finding_id=finding_id_a,
            name='SomeEntity',
        )

        # Must NOT return finding_unknown — the finding is in Stage 1's entry
        assert cite_result.get('error') != 'finding_unknown', (
            f'cite_entity returned finding_unknown instead of resolving cross-stage finding: {cite_result}'
        )
        assert 'entity_uuid' in cite_result, f'Expected entity_uuid in result, got: {cite_result}'

        # Citation must appear on Stage 1's finding in the assembled report
        s1_report = state.get_assembled_report('r1', 'memory_consolidator')
        assert s1_report is not None
        assert len(s1_report['flagged_items']) == 1
        item = s1_report['flagged_items'][0]
        assert item['finding_id'] == finding_id_a
        assert len(item['cited_entities']) == 1
        assert item['cited_entities'][0]['canonical_name'] == 'SomeEntity'

    @pytest.mark.asyncio
    async def test_cite_after_owning_stage_completed(self):
        """cite_entity on a cross-stage finding must succeed even if the finding's
        owning stage has already been completed — cite_* bypasses the completed()
        guard and now reads across stages via _resolve_finding.
        """
        state, _ = self._make_state_with_fake_services()

        # Stage 1: file finding A, then complete the stage
        state.start_report(run_id='r1', stage='memory_consolidator', project_id='dark_factory')
        res1 = state.add_finding(
            run_id='r1',
            severity='high',
            category='stuck_task',
            description='task is stuck',
            suggested_action='investigate',
            task_id='3803',
            flag_type='task_stuck_pending_merge',
        )
        assert 'finding_id' in res1, f'Stage 1 add_finding failed: {res1}'
        finding_id_a = res1['finding_id']
        state.complete(run_id='r1', summary='stage 1 done')

        # Stage 2: gets duplicate_finding pointing at A (Stage 1 already completed)
        state.start_report(run_id='r1', stage='task_knowledge_sync', project_id='dark_factory')
        dup = state.add_finding(
            run_id='r1',
            severity='high',
            category='stuck_task',
            description='same task (Stage 2 view)',
            suggested_action='investigate',
            task_id='3803',
            flag_type='task_stuck_pending_merge',
        )
        assert dup.get('error') == 'duplicate_finding'
        assert dup['existing_finding_id'] == finding_id_a

        # Stage 2 cites the cross-stage finding whose owning stage is completed
        cite_result = await state.cite_entity(
            run_id='r1',
            finding_id=finding_id_a,
            name='AnotherEntity',
        )

        # Must NOT return finding_unknown despite Stage 1 being completed
        assert cite_result.get('error') != 'finding_unknown', (
            f'cite_entity returned finding_unknown for a completed stage finding: {cite_result}'
        )
        assert 'entity_uuid' in cite_result, f'Expected entity_uuid, got: {cite_result}'

        # Citation must appear on Stage 1's finding in the assembled report
        s1_report = state.get_assembled_report('r1', 'memory_consolidator')
        assert s1_report is not None
        assert len(s1_report['flagged_items']) == 1
        item = s1_report['flagged_items'][0]
        assert item['finding_id'] == finding_id_a
        assert any(c['canonical_name'] == 'AnotherEntity' for c in item['cited_entities'])


# ---------------------------------------------------------------------------
# task-1652: Null-task_id/null-flag_type description-hash dedup
# ---------------------------------------------------------------------------


class TestReconReportNullDescDedup:
    """Verify description-content dedup for (None, None) findings.

    When both task_id and flag_type are None, add_finding must dedup by a
    normalized SHA-256 hash of the description, returning the existing
    _duplicate_finding_error shape.
    """

    def _make_state(self, ttl=300):
        from fused_memory.server.recon_report import ReconReportState

        t = [0.0]
        state = ReconReportState(ttl_seconds=ttl, clock=lambda: t[0])
        return state, t

    def _finding(
        self,
        state,
        run_id: str = 'r1',
        task_id: str | None = None,
        flag_type: str | None = None,
        description: str = 'informational observation',
        **kwargs,
    ):
        defaults = dict(
            run_id=run_id,
            severity='low',
            category='informational',
            description=description,
            suggested_action='none',
            actionable=False,
            task_id=task_id,
            flag_type=flag_type,
        )
        defaults.update(kwargs)
        return state.add_finding(**defaults)

    def test_same_description_returns_duplicate_error(self):
        """Second null-null finding with identical description → duplicate_finding."""
        state, _ = self._make_state()
        state.start_report(run_id='r1', stage='s1', project_id='dark_factory')

        first = self._finding(state, description='same text')
        assert 'finding_id' in first, f'First add_finding failed: {first}'
        first_id = first['finding_id']

        second = self._finding(state, description='same text')
        assert second.get('error') == 'duplicate_finding', (
            f'Expected duplicate_finding, got: {second}'
        )
        assert second['error_type'] == 'ReconReportDuplicateFinding'
        assert second['existing_finding_id'] == first_id

        # Stage report retains exactly ONE finding
        report = state.get_assembled_report('r1', 's1')
        assert report is not None
        assert len(report['flagged_items']) == 1
        assert report['flagged_items'][0]['finding_id'] == first_id

    def test_different_descriptions_both_allocate(self):
        """Two null-null findings with different descriptions both get distinct ids."""
        state, _ = self._make_state()
        state.start_report(run_id='r1', stage='s1', project_id='dark_factory')

        r1 = self._finding(state, description='alpha observation')
        r2 = self._finding(state, description='beta observation')
        assert 'finding_id' in r1, f'First failed: {r1}'
        assert 'finding_id' in r2, f'Second failed: {r2}'
        assert r1['finding_id'] != r2['finding_id']

    def test_normalized_whitespace_and_case_dedups(self):
        """Descriptions differing only by whitespace/case → second is a duplicate."""
        state, _ = self._make_state()
        state.start_report(run_id='r1', stage='s1', project_id='dark_factory')

        first = self._finding(state, description='Memory is stale')
        assert 'finding_id' in first, f'First failed: {first}'
        first_id = first['finding_id']

        # Differs only in case and extra internal whitespace
        second = self._finding(state, description='memory  is  STALE')
        assert second.get('error') == 'duplicate_finding', (
            f'Expected dedup on normalized description, got: {second}'
        )
        assert second['existing_finding_id'] == first_id

        # Differs only in leading/trailing whitespace
        third = self._finding(state, description='  Memory is stale  ')
        assert third.get('error') == 'duplicate_finding', (
            f'Expected dedup on whitespace-stripped description, got: {third}'
        )
        assert third['existing_finding_id'] == first_id

    def test_null_null_independent_from_signature_dedup(self):
        """A null-null finding and a signature finding with the same description
        must both allocate — separate dedup namespaces."""
        state, _ = self._make_state()
        state.start_report(run_id='r1', stage='s1', project_id='dark_factory')

        null_null = self._finding(state, task_id=None, flag_type=None, description='shared text')
        assert 'finding_id' in null_null, f'null-null failed: {null_null}'

        with_sig = state.add_finding(
            run_id='r1',
            severity='low',
            category='informational',
            description='shared text',
            suggested_action='none',
            actionable=False,
            task_id='42',
            flag_type='some_flag',
        )
        assert 'finding_id' in with_sig, f'sig finding failed: {with_sig}'
        assert with_sig['finding_id'] != null_null['finding_id']

    def test_cross_stage_same_description_dedups(self):
        """Null-null identical description filed in Stage 1 then Stage 2 of the
        SAME run_id → Stage 2 gets duplicate_finding."""
        state, _ = self._make_state()

        # Stage 1
        state.start_report(run_id='r1', stage='stage_one', project_id='dark_factory')
        first = self._finding(state, run_id='r1', description='cross-stage observation')
        assert 'finding_id' in first, f'Stage 1 failed: {first}'
        first_id = first['finding_id']

        # Stage 2 — same run_id, different stage
        state.start_report(run_id='r1', stage='stage_two', project_id='dark_factory')
        second = self._finding(state, run_id='r1', description='cross-stage observation')
        assert second.get('error') == 'duplicate_finding', (
            f'Expected cross-stage dedup, got: {second}'
        )
        assert second['existing_finding_id'] == first_id

        # Stage 2's report must be empty
        s2_report = state.get_assembled_report('r1', 'stage_two')
        assert s2_report is not None
        assert s2_report['flagged_items'] == []

    def test_cross_run_isolation_same_description(self):
        """Identical null-null description under different run_ids both allocate."""
        state, _ = self._make_state()

        state.start_report(run_id='r1', stage='s1', project_id='dark_factory')
        r1 = self._finding(state, run_id='r1', description='shared observation')
        assert 'finding_id' in r1, f'run r1 failed: {r1}'

        state.start_report(run_id='r2', stage='s1', project_id='dark_factory')
        r2 = self._finding(state, run_id='r2', description='shared observation')
        assert 'finding_id' in r2, f'run r2 failed: {r2}'

        assert r1['finding_id'] != r2['finding_id']

    def test_eviction_clears_desc_index(self):
        """After TTL eviction via tick(), _run_desc_index must not retain the
        evicted run's hashes (prevents unbounded growth across runs)."""
        state, t = self._make_state(ttl=300)
        state.start_report(run_id='r1', stage='s1', project_id='dark_factory')
        self._finding(state, run_id='r1', description='an informational note')
        state.complete(run_id='r1', summary='done')

        # Advance past TTL and evict
        t[0] = 301.0
        evicted = state.tick()
        assert evicted == 1

        # Entry is gone
        assert state.get_assembled_report('r1', 's1') is None

        # _run_desc_index must not retain r1's hashes
        assert 'r1' not in state._run_desc_index

    def test_empty_description_both_allocate(self):
        """Blank or whitespace-only descriptions normalize to '' — each allocates
        independently (no dedup key) rather than collapsing into one row.

        This pins the design choice: empty string is not a meaningful dedup key.
        Two observations with no description text are treated as distinct findings.
        """
        state, _ = self._make_state()
        state.start_report(run_id='r1', stage='s1', project_id='dark_factory')

        # Two findings with empty description must both allocate
        r1 = self._finding(state, description='')
        r2 = self._finding(state, description='')
        assert 'finding_id' in r1, f'First blank-desc failed: {r1}'
        assert 'finding_id' in r2, f'Second blank-desc failed: {r2}'
        assert r1['finding_id'] != r2['finding_id'], (
            'Blank-description findings must not dedup — empty string is not a dedup key'
        )

        # Whitespace-only descriptions also skip dedup
        r3 = self._finding(state, description='   ')
        r4 = self._finding(state, description='\t  \n')
        assert 'finding_id' in r3, f'Whitespace-only r3 failed: {r3}'
        assert 'finding_id' in r4, f'Whitespace-only r4 failed: {r4}'
        assert r3['finding_id'] != r4['finding_id'], (
            'Whitespace-only findings must not dedup'
        )

    def test_eviction_partial_run_canonical_stage_cleared(self):
        """Stage 1 holds the canonical finding; stage 2 dedups against it.
        When stage 1 is evicted via tick(), the desc index entry must be cleaned
        up correctly — the cleanup guard must not leave the hash dangling just
        because another stage of the same run referenced it via duplicate_finding.
        """
        state, t = self._make_state(ttl=300)

        # Stage 1: file the canonical finding
        state.start_report(run_id='r1', stage='stage_one', project_id='dark_factory')
        first = self._finding(state, run_id='r1', description='shared observation')
        assert 'finding_id' in first, f'Stage 1 add_finding failed: {first}'
        first_id = first['finding_id']

        # Complete stage 1 so it becomes eligible for TTL eviction
        state.complete(run_id='r1', summary='stage 1 done')

        # Stage 2: same description → duplicate_finding pointing at stage 1's finding
        state.start_report(run_id='r1', stage='stage_two', project_id='dark_factory')
        second = self._finding(state, run_id='r1', description='shared observation')
        assert second.get('error') == 'duplicate_finding', (
            f'Expected cross-stage dedup, got: {second}'
        )
        assert second['existing_finding_id'] == first_id

        # Advance past TTL — only stage_one has completed_at set, so only it evicts
        t[0] = 301.0
        evicted = state.tick()
        assert evicted == 1, f'Expected 1 eviction, got {evicted}'

        # Stage 1 entry must be gone
        assert state.get_assembled_report('r1', 'stage_one') is None

        # Stage 2 entry still exists (not completed → not eligible for eviction)
        assert state.get_assembled_report('r1', 'stage_two') is not None

        # _run_desc_index must NOT retain r1's hash after stage_one is evicted.
        # The cleanup guard (guard against removing a hash a later entry re-registered)
        # must not block the delete here, since stage_two never registered any hash.
        assert 'r1' not in state._run_desc_index
