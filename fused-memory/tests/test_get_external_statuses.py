"""Tests for get_external_statuses MCP tool (cross-project status read)."""

from __future__ import annotations

from unittest.mock import AsyncMock

import pytest
from mcp.server.fastmcp.exceptions import ToolError

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

    This fixture is load-bearing: ``get_external_statuses`` now pipes each
    registered project_root through ``_normalize_project_root`` (which calls
    ``resolve_main_checkout``) to redirect any worktree path to the main
    checkout, mirroring all other read tools.  Without this stub the real
    resolver would reject synthetic roots like ``/df`` and ``/proj_a`` because
    they are not real git working trees.  End-to-end resolver behaviour is
    exercised in test_main_checkout_resolver.py.
    """
    monkeypatch.setattr(
        'fused_memory.server.tools.resolve_main_checkout',
        lambda p: str(p),
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


# ------------------------------------------------------------------
# step-03: unknown_project sentinel
# ------------------------------------------------------------------


@pytest.mark.asyncio
async def test_unknown_project_returns_sentinel(mcp_server):
    """A project_id absent from known_projects returns 'unknown_project' (not a backend call)."""
    result = await mcp_server._tool_manager.call_tool(
        'get_external_statuses',
        {'deps': ['nope:1']},
    )
    assert result == {'nope:1': 'unknown_project'}


# ------------------------------------------------------------------
# step-05: unknown_task sentinel
# ------------------------------------------------------------------


@pytest.mark.asyncio
async def test_known_project_unknown_task_returns_sentinel(mcp_server):
    """A known project but a task_id absent from the foreign DB returns 'unknown_task'."""
    result = await mcp_server._tool_manager.call_tool(
        'get_external_statuses',
        {'deps': ['dark_factory:999999']},
    )
    assert result == {'dark_factory:999999': 'unknown_task'}


# ------------------------------------------------------------------
# step-07: malformed sentinel
# ------------------------------------------------------------------


@pytest.mark.asyncio
@pytest.mark.parametrize(
    'dep',
    [
        'garbage',  # no colon at all
        'dark_factory:',  # empty task_id
        ':13',  # empty project_id
        'dark_factory:abc',  # non-numeric task_id
        'dark_factory:15.2',  # dotted subtask form — out of scope
    ],
)
async def test_malformed_dep_returns_sentinel(mcp_server, ext_task_interceptor, dep):
    """Deps that cannot be parsed as <project_id>:<int> return 'malformed', no backend call."""
    result = await mcp_server._tool_manager.call_tool(
        'get_external_statuses',
        {'deps': [dep]},
    )
    assert result == {dep: 'malformed'}
    # No backend read should be made for malformed deps
    ext_task_interceptor.get_statuses.assert_not_called()


# ------------------------------------------------------------------
# step-09: project_id normalisation (hyphen → underscore)
# ------------------------------------------------------------------


@pytest.mark.asyncio
async def test_hyphen_project_id_normalised_and_key_verbatim(mcp_server):
    """'dark-factory:13' (hyphen form) resolves identically to 'dark_factory:13'.

    The result key must be the original hyphen string verbatim.
    """
    result = await mcp_server._tool_manager.call_tool(
        'get_external_statuses',
        {'deps': ['dark-factory:13']},
    )
    assert result == {'dark-factory:13': 'done'}


# ------------------------------------------------------------------
# step-11: acceptance test + batching + read-only
# ------------------------------------------------------------------


@pytest.mark.asyncio
async def test_acceptance_mixed_deps(mcp_server):
    """User-observable signal: mixed deps return the correct sentinel/status for each.

    Reproduces the task's acceptance criteria exactly.
    """
    result = await mcp_server._tool_manager.call_tool(
        'get_external_statuses',
        {
            'deps': [
                'dark_factory:13',  # real status
                'nope:1',  # unknown project
                'dark_factory:999999',  # unknown task
                'garbage',  # malformed (no colon)
                'dark-factory:13',  # hyphen form → same real status
            ]
        },
    )
    assert result == {
        'dark_factory:13': 'done',
        'nope:1': 'unknown_project',
        'dark_factory:999999': 'unknown_task',
        'garbage': 'malformed',
        'dark-factory:13': 'done',
    }


@pytest.mark.asyncio
async def test_batching_same_project_one_backend_call(mcp_server, ext_task_interceptor):
    """Two deps targeting the same project issue exactly ONE get_statuses backend call."""
    await mcp_server._tool_manager.call_tool(
        'get_external_statuses',
        {'deps': ['dark_factory:13', 'dark_factory:14']},
    )
    assert ext_task_interceptor.get_statuses.call_count == 1


@pytest.mark.asyncio
async def test_read_only_no_mutations(mcp_server, ext_task_interceptor):
    """The tool must not call any task-mutation or reconciliation methods."""
    await mcp_server._tool_manager.call_tool(
        'get_external_statuses',
        {'deps': ['dark_factory:13']},
    )
    ext_task_interceptor.set_task_status.assert_not_called()
    ext_task_interceptor.update_task.assert_not_called()
    ext_task_interceptor.add_dependency.assert_not_called()
    ext_task_interceptor.trigger_reconciliation.assert_not_called()


# ------------------------------------------------------------------
# step-13: transient error propagates — NOT mapped to a sentinel
# ------------------------------------------------------------------


@pytest.mark.asyncio
async def test_transient_backend_error_raises_not_sentinel(mcp_server, ext_task_interceptor):
    """A transient backend error (RuntimeError) must propagate as ToolError.

    It must NOT be caught and returned as 'unknown_project'/'unknown_task'/'malformed'.
    This guards the load-bearing distinction between transient failure (raise →
    scheduler fail-safe wait) and semantic unresolvability (sentinel → grace-then-escalate).
    """
    ext_task_interceptor.get_statuses = AsyncMock(side_effect=RuntimeError('db down'))

    with pytest.raises(ToolError):
        await mcp_server._tool_manager.call_tool(
            'get_external_statuses',
            {'deps': ['dark_factory:13']},
        )


# ------------------------------------------------------------------
# Amendment: additional edge-case coverage (suggestion 4)
# ------------------------------------------------------------------


@pytest.mark.asyncio
async def test_empty_deps_returns_empty_dict(mcp_server, ext_task_interceptor):
    """Empty deps list returns {} with zero backend calls."""
    result = await mcp_server._tool_manager.call_tool(
        'get_external_statuses',
        {'deps': []},
    )
    assert result == {}
    ext_task_interceptor.get_statuses.assert_not_called()


@pytest.fixture
def two_project_interceptor():
    """Task interceptor with a two-project foreign DB (proj_a and proj_b)."""
    ti = AsyncMock()
    db_a = {'1': 'done'}
    db_b = {'2': 'pending'}

    async def side_effect(project_root, ids=None, tag=None):
        db = db_a if project_root == '/proj_a' else db_b
        if ids is None:
            return dict(db)
        return {k: v for k, v in db.items() if k in ids}

    ti.get_statuses = AsyncMock(side_effect=side_effect)
    return ti


@pytest.fixture
def mcp_server_two_projects(two_project_interceptor):
    """MCP server with two known projects for cross-project batching tests."""
    return create_mcp_server(
        AsyncMock(),
        task_interceptor=two_project_interceptor,
        known_projects={'proj_a': '/proj_a', 'proj_b': '/proj_b'},
    )


@pytest.mark.asyncio
async def test_two_distinct_projects_two_backend_calls(
    mcp_server_two_projects, two_project_interceptor
):
    """Deps from two distinct known projects issue exactly one get_statuses call each.

    This exercises the cross-project branch of the per-project batching loop
    and verifies the call_count==2 invariant (one call per distinct project).
    """
    result = await mcp_server_two_projects._tool_manager.call_tool(
        'get_external_statuses',
        {'deps': ['proj_a:1', 'proj_b:2']},
    )
    assert result == {'proj_a:1': 'done', 'proj_b:2': 'pending'}
    assert two_project_interceptor.get_statuses.call_count == 2
