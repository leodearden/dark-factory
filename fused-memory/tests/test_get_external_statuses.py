"""Tests for get_external_statuses MCP tool (cross-project status read)."""

from __future__ import annotations

from unittest.mock import AsyncMock

import pytest

from fused_memory.server.tools import create_mcp_server


# Foreign DB data: {task_id: status}
_FOREIGN_DB = {'13': 'done', '14': 'pending', '20': 'cancelled'}


def _make_get_statuses_side_effect(db: dict):
    """Return a get_statuses side_effect that filters by ids (like the real interceptor).

    When ids is None, returns all entries.  When ids is provided, returns only
    matching entries (unknown ids silently omitted), exactly mirroring
    task_interceptor.get_statuses behaviour.
    """
    async def _side_effect(project_root, ids=None, tag=None):
        if ids is None:
            return dict(db)
        return {k: v for k, v in db.items() if k in ids}
    return _side_effect


@pytest.fixture(autouse=True)
def passthrough_main_checkout(monkeypatch):
    """Stub resolve_main_checkout to pass its argument through unchanged.

    These tests use synthetic project_root values like ``/df`` that are not
    real git working trees; the real resolver would reject them.  End-to-end
    resolver behaviour is exercised in test_main_checkout_resolver.py.
    """
    monkeypatch.setattr(
        'fused_memory.server.tools.resolve_main_checkout', lambda p: str(p),
    )


@pytest.fixture
def ext_task_interceptor():
    """Task interceptor mock with a foreign-DB get_statuses side_effect."""
    ti = AsyncMock()
    ti.get_statuses = AsyncMock(side_effect=_make_get_statuses_side_effect(_FOREIGN_DB))
    return ti


@pytest.fixture
def mcp_server(ext_task_interceptor):
    """MCP server wired with known_projects={'dark_factory': '/df'}."""
    mock_service = AsyncMock()
    return create_mcp_server(
        mock_service,
        task_interceptor=ext_task_interceptor,
        known_projects={'dark_factory': '/df'},
    )


# ------------------------------------------------------------------
# step-01: happy path — real status returned keyed by verbatim dep
# ------------------------------------------------------------------


@pytest.mark.asyncio
async def test_known_project_known_task_returns_status(mcp_server):
    """A known project + known task id returns the real task status, key is verbatim dep."""
    result = await mcp_server._tool_manager.call_tool(
        'get_external_statuses',
        {'deps': ['dark_factory:13']},
    )
    assert result == {'dark_factory:13': 'done'}
