"""Tests for get_scheduler_state and get_scheduler_events MCP tools.

All tools read from orchestrator-owned files under
``<project_root>/data/orchestrator/``.  Tests use a ``tmp_path``-rooted
project_root with the ``passthrough_main_checkout`` autouse fixture so the
path isn't rejected by the git-worktree validator.
"""

from __future__ import annotations

import json
from datetime import UTC, datetime
from pathlib import Path
from unittest.mock import AsyncMock

import pytest

from fused_memory.server.tools import create_mcp_server


# ---------------------------------------------------------------------------
# Fixtures (mirror test_scheduler_overrides_tools.py pattern)
# ---------------------------------------------------------------------------


@pytest.fixture(autouse=True)
def passthrough_main_checkout(monkeypatch):
    """Stub resolve_main_checkout to pass its argument through unchanged.

    These tests use synthetic project_root values rooted in tmp_path that
    aren't real git working trees; the real resolver would reject them.
    """
    monkeypatch.setattr(
        'fused_memory.server.tools.resolve_main_checkout', lambda p: str(p),
    )


@pytest.fixture
def memory_service():
    """Mocked MemoryService with all methods as AsyncMocks."""
    svc = AsyncMock()
    svc.add_memory = AsyncMock(return_value=None)
    return svc


@pytest.fixture
def task_interceptor():
    """Mocked task interceptor."""
    ti = AsyncMock()
    ti.set_task_status = AsyncMock(return_value={'success': True})
    return ti


@pytest.fixture
def mcp_server(memory_service, task_interceptor):
    """MCP server with mocked MemoryService and task interceptor."""
    return create_mcp_server(memory_service, task_interceptor=task_interceptor)


# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------

_EMPTY_SKELETON = {
    'skip_counts': {},
    'parks': {},
    'effective_priorities': {},
    'pin_queue': [],
    'overrides': {},
    'current_holders': {},
    'snapshot_at': None,
}


def _write_snapshot(project_root: Path, data: dict) -> Path:
    """Write a synthetic scheduler_state.json under project_root."""
    path = project_root / 'data' / 'orchestrator' / 'scheduler_state.json'
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps(data), encoding='utf-8')
    return path


def _runs_db_path(project_root: Path) -> Path:
    return project_root / 'data' / 'orchestrator' / 'runs.db'


# ===========================================================================
# Step-19: get_scheduler_state tool
# ===========================================================================


class TestGetSchedulerStateTool:
    """get_scheduler_state reads the on-disk JSON snapshot."""

    @pytest.mark.asyncio
    async def test_returns_snapshot_from_disk(self, tmp_path, mcp_server):
        """Tool round-trips a synthetic snapshot dict written to disk."""
        synthetic = {
            'skip_counts': {'T1': 3},
            'parks': {'T2': {'modules': ['m/src'], 'installed_at': '2026-01-01T00:00:00+00:00'}},
            'effective_priorities': {'T1': 'high'},
            'pin_queue': [{'task_id': 'T3', 'order': 1}],
            'overrides': {'T1': {'boost_tier': 'high', 'pinned': False,
                                 'reserve_now': False, 'ttl_until': None}},
            'current_holders': {'m/src': 'T4'},
            'snapshot_at': '2026-05-14T12:00:00+00:00',
        }
        _write_snapshot(tmp_path, synthetic)

        result = await mcp_server._tool_manager.call_tool(
            'get_scheduler_state',
            {'project_root': str(tmp_path)},
        )
        assert result == synthetic, (
            f'Expected round-tripped snapshot, got {result!r}'
        )

    @pytest.mark.asyncio
    async def test_returns_empty_skeleton_when_file_missing(
        self, tmp_path, mcp_server
    ):
        """Tool returns the empty skeleton when scheduler_state.json is missing."""
        result = await mcp_server._tool_manager.call_tool(
            'get_scheduler_state',
            {'project_root': str(tmp_path)},
        )
        assert result == _EMPTY_SKELETON, (
            f'Expected empty skeleton, got {result!r}'
        )

    @pytest.mark.asyncio
    async def test_invalid_project_root_returns_validation_error(
        self, tmp_path, mcp_server
    ):
        """Empty string project_root triggers the ValidationError envelope."""
        result = await mcp_server._tool_manager.call_tool(
            'get_scheduler_state',
            {'project_root': ''},
        )
        assert isinstance(result, dict), f'Expected dict, got {type(result)}'
        assert result.get('error_type') == 'ValidationError', (
            f'Expected ValidationError, got {result!r}'
        )
        assert 'error' in result
