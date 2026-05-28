"""Tests for PRD γ cutover: BaseStage MCP config wiring and ReconReportState lifecycle."""

from __future__ import annotations

from unittest.mock import AsyncMock

import pytest
import pytest_asyncio

from fused_memory.config.schema import ReconciliationConfig
from fused_memory.models.reconciliation import StageId
from fused_memory.reconciliation.stages.memory_consolidator import MemoryConsolidator


def _make_stage(recon_report_port: int = 8003) -> MemoryConsolidator:
    """Build a MemoryConsolidator stub with minimal mock deps and a given recon_report_port."""
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


class TestBuildMcpConfigReconReport:
    """_build_mcp_config must inject a recon-report HTTP entry from the configured port."""

    def test_recon_report_entry_default_port(self):
        """recon-report server entry present with default port 8003."""
        stage = _make_stage(recon_report_port=8003)
        mcp_config = stage._build_mcp_config()
        servers = mcp_config['mcpServers']
        assert 'recon-report' in servers, 'recon-report must be in mcpServers'
        entry = servers['recon-report']
        assert entry == {'type': 'http', 'url': 'http://127.0.0.1:8003/mcp/'}

    def test_recon_report_entry_custom_port(self):
        """recon-report entry uses the port passed at construction, not a hard-coded value."""
        stage = _make_stage(recon_report_port=9999)
        mcp_config = stage._build_mcp_config()
        entry = mcp_config['mcpServers']['recon-report']
        assert entry == {'type': 'http', 'url': 'http://127.0.0.1:9999/mcp/'}

    def test_existing_entries_preserved(self):
        """fused-memory and jcodemunch entries still present after recon-report injection."""
        stage = _make_stage(recon_report_port=8003)
        servers = stage._build_mcp_config()['mcpServers']
        assert 'fused-memory' in servers, 'fused-memory entry must remain'
        assert 'jcodemunch' in servers, 'jcodemunch entry must remain'
