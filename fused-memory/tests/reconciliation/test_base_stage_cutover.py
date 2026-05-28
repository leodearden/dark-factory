"""Tests for PRD γ cutover: BaseStage MCP config wiring and ReconReportState lifecycle."""

from __future__ import annotations

from unittest.mock import AsyncMock, MagicMock, patch

import pytest
import pytest_asyncio

from fused_memory.config.schema import ReconciliationConfig
from fused_memory.models.reconciliation import StageId, StageReport, Watermark
from fused_memory.reconciliation.cli_stage_runner import STAGE_REPORT_SCHEMA
from fused_memory.reconciliation.stages.base import BaseStage
from fused_memory.reconciliation.stages.memory_consolidator import MemoryConsolidator


class _StubStage(BaseStage):
    """Minimal BaseStage stub for testing BaseStage.run logic directly."""

    def get_disallowed_tools(self) -> list[str]:
        return []

    def get_system_prompt(self) -> str:
        return 'test prompt'

    def get_report_schema(self) -> dict:
        return STAGE_REPORT_SCHEMA

    async def assemble_payload(self, events, watermark, prior_reports) -> str:
        return 'test payload'


def _make_stage(
    recon_report_port: int = 8003,
    recon_report_state=None,
) -> _StubStage:
    """Build a _StubStage with minimal mock deps."""
    config = ReconciliationConfig()
    memory_mock = AsyncMock()

    stage = _StubStage(
        StageId.memory_consolidator,
        memory_mock,
        AsyncMock(),  # taskmaster
        AsyncMock(),  # journal
        config,
        recon_report_port=recon_report_port,
        recon_report_state=recon_report_state,
    )
    stage.project_id = 'test_project'
    stage.project_root = '/tmp/test'
    return stage


def _make_consolidator(recon_report_port: int = 8003) -> MemoryConsolidator:
    """Build a MemoryConsolidator with mocked deps for _build_mcp_config tests."""
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
        recon_report_port=recon_report_port,
    )
    stage.project_id = 'test_project'
    stage.project_root = '/tmp/test'
    stage.episode_limit = 5
    stage.memory_limit = 10
    return stage


class _FakeReconState:
    """Recording fake for ReconReportState — logs calls for assertion."""

    def __init__(self, assembled_report: dict | None = None):
        self.calls: list[tuple] = []
        self._assembled = assembled_report

    def start_report(self, run_id: str, stage: str, project_id: str) -> dict:
        self.calls.append(('start_report', run_id, stage, project_id))
        return {'run_id': run_id, 'stage': stage}

    def get_assembled_report(self, run_id: str, stage: str) -> dict | None:
        self.calls.append(('get_assembled_report', run_id, stage))
        return self._assembled


class TestReconStateLifecycle:
    """BaseStage.run must call start_report before the CLI and get_assembled_report after."""

    @pytest.mark.asyncio
    async def test_start_report_called_before_cli(self):
        """start_report is called once with (run_id, stage_id.value, project_id)."""
        call_log: list[str] = []

        assembled = {
            'summary': 'ok',
            'stats': {'items_reviewed': 5},
            'flagged_items': [],
            'summary_warnings': [],
        }

        class _RecordingState(_FakeReconState):
            def start_report(self, run_id, stage, project_id):
                call_log.append('start_report')
                return super().start_report(run_id, stage, project_id)

            def get_assembled_report(self, run_id, stage):
                call_log.append('get_assembled_report')
                return super().get_assembled_report(run_id, stage)

        state = _RecordingState(assembled_report=assembled)
        stage = _make_stage(recon_report_state=state)
        watermark = Watermark(project_id='test_project')

        from fused_memory.reconciliation.cli_stage_runner import StageResult

        fake_result = StageResult(report={}, success=True)

        with patch(
            'fused_memory.reconciliation.stages.base.run_stage_via_cli',
            new=AsyncMock(return_value=fake_result),
        ) as mock_cli:
            report = await stage.run([], watermark, [], run_id='run-001')

        # start_report must be called before the CLI (first in log)
        assert call_log[0] == 'start_report', 'start_report must be called before CLI'
        assert call_log[1] == 'get_assembled_report', 'get_assembled_report called after CLI'

    @pytest.mark.asyncio
    async def test_start_report_args(self):
        """start_report receives the correct run_id, stage_id.value, and project_id."""
        assembled = {'summary': '', 'stats': {}, 'flagged_items': [], 'summary_warnings': []}
        state = _FakeReconState(assembled_report=assembled)
        stage = _make_stage(recon_report_state=state)
        watermark = Watermark(project_id='test_project')

        from fused_memory.reconciliation.cli_stage_runner import StageResult

        with patch(
            'fused_memory.reconciliation.stages.base.run_stage_via_cli',
            new=AsyncMock(return_value=StageResult(report={}, success=True)),
        ):
            await stage.run([], watermark, [], run_id='run-abc')

        start_call = next(c for c in state.calls if c[0] == 'start_report')
        assert start_call[1] == 'run-abc'  # run_id
        assert start_call[2] == StageId.memory_consolidator.value  # stage
        assert start_call[3] == 'test_project'  # project_id

    @pytest.mark.asyncio
    async def test_assembled_report_overrides_stage_result(self):
        """StageReport.items_flagged and stats come from assembled report, not stage_result."""
        assembled = {
            'summary': 'test summary',
            'stats': {'items_reviewed': 42},
            'flagged_items': [{'description': 'test finding', 'severity': 'minor'}],
            'summary_warnings': [],
        }
        state = _FakeReconState(assembled_report=assembled)
        stage = _make_stage(recon_report_state=state)
        watermark = Watermark(project_id='test_project')

        from fused_memory.reconciliation.cli_stage_runner import StageResult

        # stage_result.report has different content — should be ignored
        dummy_result = StageResult(
            report={'flagged_items': [], 'stats': {}, 'summary': 'ignored'},
            success=True,
        )

        with patch(
            'fused_memory.reconciliation.stages.base.run_stage_via_cli',
            new=AsyncMock(return_value=dummy_result),
        ):
            report = await stage.run([], watermark, [], run_id='run-xyz')

        assert report.items_flagged == assembled['flagged_items']
        assert report.stats == assembled['stats']


class TestReconStateFallbackToEmpty:
    """When get_assembled_report returns None, BaseStage.run produces an empty StageReport."""

    @pytest.mark.asyncio
    async def test_none_assembled_produces_empty_stage_report(self):
        """get_assembled_report returning None → items_flagged=[], stats={}, no exception."""
        state = _FakeReconState(assembled_report=None)  # Simulates agent crash
        stage = _make_stage(recon_report_state=state)
        watermark = Watermark(project_id='test_project')

        from fused_memory.reconciliation.cli_stage_runner import StageResult

        with patch(
            'fused_memory.reconciliation.stages.base.run_stage_via_cli',
            new=AsyncMock(return_value=StageResult(report={}, success=True)),
        ):
            report = await stage.run([], watermark, [], run_id='run-none')

        assert report.items_flagged == [], 'items_flagged must be empty on None assembled'
        assert report.stats == {}, 'stats must be empty on None assembled'

    @pytest.mark.asyncio
    async def test_timestamps_still_set_on_none_assembled(self):
        """started_at and completed_at are populated even on None assembled."""
        state = _FakeReconState(assembled_report=None)
        stage = _make_stage(recon_report_state=state)
        watermark = Watermark(project_id='test_project')

        from fused_memory.reconciliation.cli_stage_runner import StageResult

        with patch(
            'fused_memory.reconciliation.stages.base.run_stage_via_cli',
            new=AsyncMock(return_value=StageResult(report={}, success=True)),
        ):
            report = await stage.run([], watermark, [], run_id='run-ts')

        assert report.started_at is not None
        assert report.completed_at is not None


class TestBuildMcpConfigReconReport:
    """_build_mcp_config must inject a recon-report HTTP entry from the configured port."""

    def test_recon_report_entry_default_port(self):
        """recon-report server entry present with default port 8003."""
        stage = _make_consolidator(recon_report_port=8003)
        mcp_config = stage._build_mcp_config()
        servers = mcp_config['mcpServers']
        assert 'recon-report' in servers, 'recon-report must be in mcpServers'
        entry = servers['recon-report']
        assert entry == {'type': 'http', 'url': 'http://127.0.0.1:8003/mcp/'}

    def test_recon_report_entry_custom_port(self):
        """recon-report entry uses the port passed at construction, not a hard-coded value."""
        stage = _make_consolidator(recon_report_port=9999)
        mcp_config = stage._build_mcp_config()
        entry = mcp_config['mcpServers']['recon-report']
        assert entry == {'type': 'http', 'url': 'http://127.0.0.1:9999/mcp/'}

    def test_existing_entries_preserved(self):
        """fused-memory and jcodemunch entries still present after recon-report injection."""
        stage = _make_consolidator(recon_report_port=8003)
        servers = stage._build_mcp_config()['mcpServers']
        assert 'fused-memory' in servers, 'fused-memory entry must remain'
        assert 'jcodemunch' in servers, 'jcodemunch entry must remain'
