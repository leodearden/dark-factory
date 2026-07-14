"""Contract/characterization test pinning the deliberate get_statuses vs
get_external_statuses envelope shape asymmetry (task 2562 item 3).

``get_statuses`` returns a WRAPPED ``{'statuses': {id: status}}`` envelope
while ``get_external_statuses`` intentionally returns a FLAT
``{dep: status}`` dict — its documented scheduler contract, guard-defused
by tasks 1799/1807 (see CLAUDE.md's "Cross-project task dependencies"
section). This asymmetry is deliberate, not a bug: unifying the shapes
would break the scheduler and the documented contract (see the docstrings
on both tools). This test pins both runtime shapes as a regression guard
so any future "unify the envelope" change must consciously update it.

Runtime-shape assertions only — no docstring-wording assertions.
"""

from __future__ import annotations

from unittest.mock import AsyncMock

import pytest

from fused_memory.server.tools import create_mcp_server

_STATUSES = {'1': 'pending', '2': 'done'}


@pytest.fixture(autouse=True)
def passthrough_main_checkout(monkeypatch):
    """Stub resolve_main_checkout to pass its argument through unchanged.

    Mirrors test_get_external_statuses.py's fixture of the same purpose —
    without this the real resolver rejects synthetic roots like '/project'
    because they are not real git working trees.
    """
    monkeypatch.setattr(
        'fused_memory.server.tools.resolve_main_checkout',
        lambda p: str(p),
    )


@pytest.fixture
def contract_task_interceptor():
    """AsyncMock task_interceptor whose get_statuses filters by ids like the
    real interceptor (mirrors test_get_external_statuses.py's fixture)."""
    ti = AsyncMock()

    async def _get_statuses(project_root, ids=None, tag=None):
        if ids is None:
            return dict(_STATUSES)
        return {k: v for k, v in _STATUSES.items() if k in ids}

    ti.get_statuses = AsyncMock(side_effect=_get_statuses)
    return ti


@pytest.fixture
def contract_server(contract_task_interceptor):
    """MCP server wired with known_projects for the get_external_statuses case."""
    mock_service = AsyncMock()
    return create_mcp_server(
        mock_service,
        task_interceptor=contract_task_interceptor,
        known_projects={'dark_factory': '/project'},
    )


@pytest.mark.asyncio
async def test_get_statuses_returns_wrapped_envelope(contract_server):
    """get_statuses wraps the {id: status} map under a top-level 'statuses' key."""
    result = await contract_server._tool_manager.call_tool(
        'get_statuses',
        {'project_root': '/project'},
    )
    assert isinstance(result, dict)
    assert 'statuses' in result, f"Expected wrapped 'statuses' key, got: {result!r}"
    assert result['statuses'] == _STATUSES
    # The wrapped envelope's outer dict is NOT itself the {id: status} map.
    assert result != _STATUSES


@pytest.mark.asyncio
async def test_get_external_statuses_returns_flat_dict(contract_server):
    """get_external_statuses returns a FLAT {dep: status} dict — no 'statuses' key."""
    result = await contract_server._tool_manager.call_tool(
        'get_external_statuses',
        {'deps': ['dark_factory:1', 'dark_factory:2']},
    )
    assert isinstance(result, dict)
    assert 'statuses' not in result, f"Flat shape must not have a 'statuses' key: {result!r}"
    assert result == {'dark_factory:1': 'pending', 'dark_factory:2': 'done'}


@pytest.mark.asyncio
async def test_envelope_shapes_are_asymmetric_by_design(contract_server):
    """Direct side-by-side: the two tools' response shapes are NOT
    interchangeable for the same underlying status data — pins the
    asymmetry as a single characterization assertion so a future caller
    (or refactor) cannot accidentally assume a shared shape.
    """
    wrapped = await contract_server._tool_manager.call_tool(
        'get_statuses',
        {'project_root': '/project'},
    )
    flat = await contract_server._tool_manager.call_tool(
        'get_external_statuses',
        {'deps': ['dark_factory:1']},
    )
    assert 'statuses' in wrapped
    assert 'statuses' not in flat
    assert set(wrapped.keys()) == {'statuses'}
    assert set(flat.keys()) == {'dark_factory:1'}
