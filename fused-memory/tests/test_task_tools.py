"""Tests for MCP task-tool behavior (update_task, set_task_status, etc.)."""

from unittest.mock import AsyncMock

import pytest

from fused_memory.server.tools import create_mcp_server


@pytest.fixture(autouse=True)
def passthrough_main_checkout(monkeypatch):
    """Stub resolve_main_checkout to pass its argument through unchanged.

    These tests use synthetic project_root values like ``/project`` that
    aren't real git working trees; the real resolver would reject them.
    End-to-end resolver behavior is exercised in
    test_main_checkout_resolver.py and test_canonical_tasks_json.py.
    """
    monkeypatch.setattr(
        'fused_memory.server.tools.resolve_main_checkout', lambda p: str(p),
    )


@pytest.fixture
def task_interceptor():
    ti = AsyncMock()
    ti.update_task = AsyncMock(return_value={'success': True})
    return ti


@pytest.fixture
def mcp_server_with_tasks(task_interceptor):
    """MCP server with a mocked task interceptor."""
    mock_service = AsyncMock()
    return create_mcp_server(mock_service, task_interceptor=task_interceptor)


# ------------------------------------------------------------------
# update_task metadata coercion
# ------------------------------------------------------------------


@pytest.mark.asyncio
async def test_update_task_metadata_dict_coerced_to_json_string(
    mcp_server_with_tasks, task_interceptor,
):
    """When metadata is passed as a dict (as MCP callers naturally do),
    the tool should JSON-serialize it before forwarding to the interceptor."""
    await mcp_server_with_tasks._tool_manager.call_tool(
        'update_task',
        {'id': '1', 'project_root': '/project', 'metadata': {'key': 'value'}},
    )
    task_interceptor.update_task.assert_called_once()
    _, kwargs = task_interceptor.update_task.call_args
    assert kwargs['metadata'] == '{"key": "value"}'


@pytest.mark.asyncio
async def test_update_task_metadata_string_passed_through(
    mcp_server_with_tasks, task_interceptor,
):
    """When metadata is already a JSON string, it should pass through unchanged."""
    await mcp_server_with_tasks._tool_manager.call_tool(
        'update_task',
        {'id': '1', 'project_root': '/project', 'metadata': '{"key": "value"}'},
    )
    task_interceptor.update_task.assert_called_once()
    _, kwargs = task_interceptor.update_task.call_args
    assert kwargs['metadata'] == '{"key": "value"}'


@pytest.mark.asyncio
async def test_update_task_metadata_none_passed_through(
    mcp_server_with_tasks, task_interceptor,
):
    """When metadata is None/omitted, it should pass through as None."""
    await mcp_server_with_tasks._tool_manager.call_tool(
        'update_task',
        {'id': '1', 'project_root': '/project'},
    )
    task_interceptor.update_task.assert_called_once()
    _, kwargs = task_interceptor.update_task.call_args
    assert kwargs['metadata'] is None


# ------------------------------------------------------------------
# update_task parameter forwarding (prompt, append, tag)
# ------------------------------------------------------------------


@pytest.mark.asyncio
@pytest.mark.parametrize(
    'tool_args, expected_overrides',
    [
        pytest.param(
            {'id': '1', 'project_root': '/project', 'prompt': 'Update the description'},
            {'prompt': 'Update the description', 'append': False, 'tag': None},
            id='prompt-forwarded',
        ),
        pytest.param(
            {'id': '1', 'project_root': '/project', 'prompt': 'Extra info', 'append': True},
            {'prompt': 'Extra info', 'append': True, 'tag': None},
            id='append-true',
        ),
        pytest.param(
            {'id': '1', 'project_root': '/project', 'tag': 'v2'},
            {'prompt': None, 'append': False, 'tag': 'v2'},
            id='tag-forwarded',
        ),
        pytest.param(
            {'id': '1', 'project_root': '/project'},
            {'prompt': None, 'append': False, 'tag': None},
            id='tag-none',
        ),
    ],
)
async def test_update_task_param_forwarding(
    tool_args, expected_overrides, mcp_server_with_tasks, task_interceptor,
):
    """update_task forwards all parameters to the interceptor with exact kwargs."""
    result = await mcp_server_with_tasks._tool_manager.call_tool('update_task', tool_args)
    assert result == {'success': True}
    base_kwargs = {
        'task_id': '1',
        'project_root': '/project',
        'metadata': None,
    }
    expected_kwargs = {**base_kwargs, **expected_overrides}
    task_interceptor.update_task.assert_called_once_with(**expected_kwargs)


# ------------------------------------------------------------------
# update_task error handling
# ------------------------------------------------------------------


@pytest.mark.asyncio
async def test_update_task_interceptor_error_returns_error_dict(
    mcp_server_with_tasks, task_interceptor,
):
    """When the interceptor raises an Exception, the tool returns {'error': str(e)}."""
    task_interceptor.update_task.side_effect = RuntimeError('backend unavailable')
    result = await mcp_server_with_tasks._tool_manager.call_tool(
        'update_task',
        {'id': '1', 'project_root': '/project'},
    )
    assert isinstance(result, dict)
    assert 'error' in result
    assert 'backend unavailable' in result['error']
    assert result['error_type'] == RuntimeError.__name__


@pytest.mark.asyncio
async def test_update_task_relative_path_returns_validation_error(
    mcp_server_with_tasks,
):
    """When project_root is a relative path, update_task returns a ValidationError dict."""
    result = await mcp_server_with_tasks._tool_manager.call_tool(
        'update_task',
        {'id': '1', 'project_root': 'relative/path'},
    )
    assert isinstance(result, dict)
    assert 'error' in result
    assert result['error_type'] == 'ValidationError'


# ------------------------------------------------------------------
# update_task rejects metadata.done_provenance (2026-04-27 hardening)
# ------------------------------------------------------------------


@pytest.mark.asyncio
async def test_update_task_rejects_metadata_done_provenance_dict(
    mcp_server_with_tasks, task_interceptor,
):
    """A dict-shaped metadata carrying done_provenance is rejected with a pointer to set_task_status."""
    result = await mcp_server_with_tasks._tool_manager.call_tool(
        'update_task',
        {
            'id': '1', 'project_root': '/project',
            'metadata': {'done_provenance': {'kind': 'merged', 'commit': 'abc'}},
        },
    )
    assert isinstance(result, dict)
    assert result.get('error_type') == 'ValidationError'
    assert 'set_task_status' in result['error']
    task_interceptor.update_task.assert_not_called()


@pytest.mark.asyncio
async def test_update_task_rejects_metadata_done_provenance_json_string(
    mcp_server_with_tasks, task_interceptor,
):
    """A JSON-string metadata carrying done_provenance is also rejected."""
    result = await mcp_server_with_tasks._tool_manager.call_tool(
        'update_task',
        {
            'id': '1', 'project_root': '/project',
            'metadata': '{"done_provenance": {"kind": "merged", "commit": "abc"}}',
        },
    )
    assert isinstance(result, dict)
    assert result.get('error_type') == 'ValidationError'
    assert 'set_task_status' in result['error']
    task_interceptor.update_task.assert_not_called()


@pytest.mark.asyncio
async def test_update_task_allows_unrelated_metadata(
    mcp_server_with_tasks, task_interceptor,
):
    """Other metadata keys still pass through; the gate only blocks done_provenance."""
    result = await mcp_server_with_tasks._tool_manager.call_tool(
        'update_task',
        {
            'id': '1', 'project_root': '/project',
            'metadata': {'modules': ['orchestrator/'], 'priority': 'high'},
        },
    )
    assert result == {'success': True}
    task_interceptor.update_task.assert_called_once()


# ------------------------------------------------------------------
# Defensive tool registration (always registered, even without Taskmaster)
# ------------------------------------------------------------------


def test_task_tools_registered_without_interceptor():
    """Task tools are registered even when no task_interceptor is provided."""
    mock_service = AsyncMock()
    server = create_mcp_server(mock_service)  # No task_interceptor
    tool_names = [t.name for t in server._tool_manager.list_tools()]
    for name in ['get_tasks', 'get_task', 'set_task_status',
                 'update_task', 'add_subtask', 'remove_task', 'add_dependency',
                 'remove_dependency']:
        assert name in tool_names, f'{name} should be registered'
    for name in ['expand_task', 'parse_prd']:
        assert name not in tool_names, (
            f'{name} was retired with the Taskmaster cutover'
        )


def test_add_task_mcp_tool_not_registered():
    """The deprecated add_task MCP tool binding must not exist after facade removal."""
    mock_service = AsyncMock()
    server = create_mcp_server(mock_service)
    tool_names = [t.name for t in server._tool_manager.list_tools()]
    assert 'add_task' not in tool_names, 'add_task MCP tool must be removed'


@pytest.mark.asyncio
async def test_task_tool_error_without_taskmaster():
    """Calling a task tool with no-taskmaster interceptor returns structured error."""
    mock_service = AsyncMock()
    server = create_mcp_server(mock_service)  # No task_interceptor → fallback
    result = await server._tool_manager.call_tool(
        'get_tasks', {'project_root': '/project'},
    )
    assert 'error' in result
    assert 'not configured' in result['error'].lower()


# ------------------------------------------------------------------
# set_task_status input validation
# ------------------------------------------------------------------


@pytest.mark.asyncio
async def test_set_task_status_rejects_invalid_status(mcp_server_with_tasks):
    """set_task_status with an invalid status returns an error dict."""
    result = await mcp_server_with_tasks._tool_manager.call_tool(
        'set_task_status',
        {'id': '1', 'project_root': '/project', 'status': 'bogus'},
    )
    assert isinstance(result, dict)
    assert 'error' in result
    assert 'bogus' in result['error'] or 'invalid' in result['error'].lower()
    # Should mention valid statuses
    assert 'done' in result['error'] or 'pending' in result['error']


@pytest.mark.asyncio
async def test_set_task_status_valid_status_passes_through(
    mcp_server_with_tasks, task_interceptor,
):
    """set_task_status with a valid status passes through to the interceptor."""
    task_interceptor.set_task_status = AsyncMock(return_value={'success': True})
    result = await mcp_server_with_tasks._tool_manager.call_tool(
        'set_task_status',
        {'id': '1', 'project_root': '/project', 'status': 'done'},
    )
    task_interceptor.set_task_status.assert_called_once()
    assert 'error' not in result


# ------------------------------------------------------------------
# trigger_reconciliation without taskmaster
# ------------------------------------------------------------------


@pytest.mark.asyncio
async def test_trigger_reconciliation_without_taskmaster_returns_not_configured():
    """trigger_reconciliation without a task_interceptor returns a clear 'not configured' error."""
    mock_service = AsyncMock()
    server = create_mcp_server(mock_service)  # No task_interceptor
    result = await server._tool_manager.call_tool(
        'trigger_reconciliation',
        {'project_id': 'proj'},
    )
    assert isinstance(result, dict)
    assert 'error' in result
    assert 'not configured' in result['error'].lower()


# ------------------------------------------------------------------
# error_type in exception handler responses
# ------------------------------------------------------------------


@pytest.mark.asyncio
async def test_set_task_status_exception_includes_error_type(
    mcp_server_with_tasks, task_interceptor,
):
    """When set_task_status interceptor raises RuntimeError, result includes error_type='RuntimeError'."""
    task_interceptor.set_task_status = AsyncMock(
        side_effect=RuntimeError('backend unavailable')
    )
    result = await mcp_server_with_tasks._tool_manager.call_tool(
        'set_task_status',
        {'id': '1', 'project_root': '/project', 'status': 'done'},
    )
    assert isinstance(result, dict)
    assert 'error' in result
    assert 'backend unavailable' in result['error']
    assert result.get('error_type') == RuntimeError.__name__


@pytest.mark.asyncio
async def test_update_task_exception_includes_error_type(
    mcp_server_with_tasks, task_interceptor,
):
    """When update_task interceptor raises ValueError, result includes error_type='ValueError'."""
    task_interceptor.update_task = AsyncMock(
        side_effect=ValueError('invalid field')
    )
    result = await mcp_server_with_tasks._tool_manager.call_tool(
        'update_task',
        {'id': '1', 'project_root': '/project'},
    )
    assert isinstance(result, dict)
    assert 'error' in result
    assert 'invalid field' in result['error']
    assert result.get('error_type') == ValueError.__name__


# ------------------------------------------------------------------
# [REVIEW FIX] Regression: 'blocked' must be a valid task status
# ------------------------------------------------------------------


@pytest.mark.asyncio
async def test_set_task_status_blocked_passes_through(
    mcp_server_with_tasks, task_interceptor,
):
    """[Regression] set_task_status with status='blocked' must pass through — not rejected.

    'blocked' is a TaskInterceptor.STATUS_TRIGGERS value and is documented in the
    set_task_status docstring. Rejecting it would be a functional regression.
    """
    task_interceptor.set_task_status = AsyncMock(return_value={'success': True})
    result = await mcp_server_with_tasks._tool_manager.call_tool(
        'set_task_status',
        {'id': '1', 'project_root': '/project', 'status': 'blocked'},
    )
    # Must NOT be a validation error
    assert isinstance(result, dict)
    assert 'error' not in result, (
        f"'blocked' should be accepted as a valid status, got error: {result.get('error')}"
    )
    task_interceptor.set_task_status.assert_called_once()


@pytest.mark.asyncio
@pytest.mark.parametrize('status', ['done', 'blocked', 'cancelled', 'deferred'])
async def test_set_task_status_all_trigger_statuses_pass_through(
    status, mcp_server_with_tasks, task_interceptor,
):
    """All TaskInterceptor.STATUS_TRIGGERS values must be accepted by validation."""
    task_interceptor.set_task_status = AsyncMock(return_value={'success': True})
    result = await mcp_server_with_tasks._tool_manager.call_tool(
        'set_task_status',
        {'id': '1', 'project_root': '/project', 'status': status},
    )
    assert isinstance(result, dict)
    assert 'error' not in result, (
        f"STATUS_TRIGGERS value {status!r} should be accepted, got: {result}"
    )


# ---------------------------------------------------------------------------
# step-17: ticket-shaped id rejection for all id-accepting tools
# ---------------------------------------------------------------------------

@pytest.mark.asyncio
@pytest.mark.parametrize('tool_name,extra_kwargs', [
    ('set_task_status', {'status': 'done'}),
    ('update_task', {'title': 'new title'}),
    ('add_subtask', {'title': 'sub'}),
    ('remove_task', {}),
    ('add_dependency', {'depends_on': '5'}),
    ('remove_dependency', {'depends_on': '5'}),
])
async def test_id_accepting_tools_reject_ticket_shaped_ids(
    tool_name, extra_kwargs, mcp_server_with_tasks, task_interceptor,
):
    """Tools that accept an ``id`` arg must reject tkt_-prefixed ids with a ValidationError."""
    task_interceptor.set_task_status = AsyncMock(return_value={'success': True})
    task_interceptor.update_task = AsyncMock(return_value={'success': True})
    task_interceptor.add_subtask = AsyncMock(return_value={'success': True})
    task_interceptor.remove_task = AsyncMock(return_value={'success': True})
    task_interceptor.add_dependency = AsyncMock(return_value={'success': True})
    task_interceptor.remove_dependency = AsyncMock(return_value={'success': True})

    args = {'id': 'tkt_abc', 'project_root': '/project', **extra_kwargs}
    result = await mcp_server_with_tasks._tool_manager.call_tool(tool_name, args)

    assert isinstance(result, dict), f'Expected dict, got {type(result)}: {result!r}'
    assert result.get('error_type') == 'ValidationError', (
        f'Expected ValidationError, got: {result}'
    )
    assert 'tkt_' in result.get('error', '') or 'ticket' in result.get('error', '').lower(), (
        f'Error message should mention ticket resolution: {result.get("error")!r}'
    )
    # Must NOT have called the backend
    getattr(task_interceptor, tool_name.replace('remove_', 'remove_')).assert_not_called()


@pytest.mark.asyncio
@pytest.mark.parametrize('tool_name', ['add_dependency', 'remove_dependency'])
async def test_dependency_tools_reject_ticket_shaped_depends_on(
    tool_name, mcp_server_with_tasks, task_interceptor,
):
    """Dependency tools must also reject tkt_-prefixed depends_on values."""
    task_interceptor.add_dependency = AsyncMock(return_value={'success': True})
    task_interceptor.remove_dependency = AsyncMock(return_value={'success': True})

    args = {'id': '1', 'project_root': '/project', 'depends_on': 'tkt_abc'}
    result = await mcp_server_with_tasks._tool_manager.call_tool(tool_name, args)

    assert isinstance(result, dict)
    assert result.get('error_type') == 'ValidationError', (
        f'Expected ValidationError for ticket depends_on, got: {result}'
    )
    getattr(task_interceptor, tool_name).assert_not_called()


# ---------------------------------------------------------------------------
# get_statuses MCP tool tests (step-7)
# ---------------------------------------------------------------------------


def test_get_statuses_registered(mcp_server_with_tasks):
    """get_statuses is registered as a tool in the MCP server."""
    tool_names = [t.name for t in mcp_server_with_tasks._tool_manager.list_tools()]
    assert 'get_statuses' in tool_names


@pytest.mark.asyncio
async def test_get_statuses_forwards_to_interceptor(mcp_server_with_tasks, task_interceptor):
    """get_statuses wraps the interceptor result in {'statuses': ...}."""
    from unittest.mock import AsyncMock
    task_interceptor.get_statuses = AsyncMock(return_value={'1': 'done'})
    result = await mcp_server_with_tasks._tool_manager.call_tool(
        'get_statuses', {'project_root': '/project'},
    )
    assert result == {'statuses': {'1': 'done'}}


@pytest.mark.asyncio
async def test_get_statuses_relative_path_returns_validation_error(mcp_server_with_tasks):
    """Relative project_root returns a ValidationError dict."""
    result = await mcp_server_with_tasks._tool_manager.call_tool(
        'get_statuses', {'project_root': 'relative/path'},
    )
    assert isinstance(result, dict)
    assert 'error' in result
    assert result['error_type'] == 'ValidationError'


@pytest.mark.asyncio
async def test_get_statuses_interceptor_exception_returns_error_type(
    mcp_server_with_tasks, task_interceptor,
):
    """RuntimeError from the interceptor surfaces as {'error': ..., 'error_type': 'RuntimeError'}."""
    from unittest.mock import AsyncMock
    task_interceptor.get_statuses = AsyncMock(side_effect=RuntimeError('backend failure'))
    result = await mcp_server_with_tasks._tool_manager.call_tool(
        'get_statuses', {'project_root': '/project'},
    )
    assert isinstance(result, dict)
    assert 'error' in result
    assert 'backend failure' in result['error']
    assert result['error_type'] == 'RuntimeError'


# ------------------------------------------------------------------
# planning_mode + resolve_ticket idempotency + commit_planning
# ------------------------------------------------------------------


@pytest.mark.asyncio
async def test_submit_task_forwards_planning_mode_flag(
    mcp_server_with_tasks, task_interceptor,
):
    """submit_task MCP tool forwards planning_mode=True to the interceptor."""
    from unittest.mock import AsyncMock
    task_interceptor.submit_task = AsyncMock(
        return_value={'task_id': '7', 'status': 'deferred', 'planning_mode': True},
    )
    result = await mcp_server_with_tasks._tool_manager.call_tool(
        'submit_task',
        {'project_root': '/project', 'title': 'X', 'planning_mode': True},
    )
    assert result == {'task_id': '7', 'status': 'deferred', 'planning_mode': True}
    kwargs = task_interceptor.submit_task.call_args.kwargs
    assert kwargs.get('planning_mode') is True


@pytest.mark.asyncio
async def test_submit_task_planning_mode_default_false(
    mcp_server_with_tasks, task_interceptor,
):
    """submit_task defaults planning_mode=False when omitted."""
    from unittest.mock import AsyncMock
    task_interceptor.submit_task = AsyncMock(return_value={'ticket': 'tkt_x'})
    await mcp_server_with_tasks._tool_manager.call_tool(
        'submit_task', {'project_root': '/project', 'title': 'X'},
    )
    kwargs = task_interceptor.submit_task.call_args.kwargs
    assert kwargs.get('planning_mode') is False


@pytest.mark.asyncio
async def test_resolve_ticket_idempotent_passthrough_for_numeric_id(
    mcp_server_with_tasks, task_interceptor,
):
    """A numeric task id passed to resolve_ticket short-circuits to created/idempotent."""
    from unittest.mock import AsyncMock
    task_interceptor.resolve_ticket = AsyncMock()  # Should not be called.
    result = await mcp_server_with_tasks._tool_manager.call_tool(
        'resolve_ticket', {'ticket': '42', 'project_root': '/project'},
    )
    assert result == {
        'status': 'created',
        'task_id': '42',
        'reason': 'idempotent_passthrough',
    }
    task_interceptor.resolve_ticket.assert_not_called()


@pytest.mark.asyncio
async def test_resolve_ticket_idempotent_passthrough_strips_whitespace(
    mcp_server_with_tasks, task_interceptor,
):
    """Numeric ids with surrounding whitespace are accepted and stripped.

    The MCP wire schema enforces ``ticket: str``, so int passthrough is
    only meaningful at the interceptor layer; it's covered by the
    ``_looks_like_task_id`` unit tests in test_task_interceptor.py.
    """
    from unittest.mock import AsyncMock
    task_interceptor.resolve_ticket = AsyncMock()
    result = await mcp_server_with_tasks._tool_manager.call_tool(
        'resolve_ticket', {'ticket': '  42  ', 'project_root': '/project'},
    )
    assert result == {
        'status': 'created',
        'task_id': '42',
        'reason': 'idempotent_passthrough',
    }
    task_interceptor.resolve_ticket.assert_not_called()


@pytest.mark.asyncio
async def test_resolve_ticket_rejects_non_ticket_non_numeric(
    mcp_server_with_tasks,
):
    """resolve_ticket still rejects strings that are neither tickets nor numeric ids."""
    result = await mcp_server_with_tasks._tool_manager.call_tool(
        'resolve_ticket', {'ticket': 'not-a-ticket', 'project_root': '/project'},
    )
    assert result['error_type'] == 'ValidationError'
    assert 'tkt_' in result['error']


@pytest.mark.asyncio
async def test_resolve_ticket_real_ticket_still_resolves(
    mcp_server_with_tasks, task_interceptor,
):
    """tkt_-prefixed tickets still flow to the interceptor's resolve_ticket."""
    from unittest.mock import AsyncMock
    task_interceptor.resolve_ticket = AsyncMock(
        return_value={'status': 'created', 'task_id': '99'},
    )
    result = await mcp_server_with_tasks._tool_manager.call_tool(
        'resolve_ticket',
        {'ticket': 'tkt_abc', 'project_root': '/project'},
    )
    assert result == {'status': 'created', 'task_id': '99'}
    task_interceptor.resolve_ticket.assert_called_once()


@pytest.mark.asyncio
async def test_commit_planning_forwards_to_set_task_status(
    mcp_server_with_tasks, task_interceptor,
):
    """commit_planning bulk-flips ids via set_task_status with the target_status."""
    from unittest.mock import AsyncMock
    task_interceptor.set_task_status = AsyncMock(
        return_value={'success': True, 'results': []},
    )
    result = await mcp_server_with_tasks._tool_manager.call_tool(
        'commit_planning',
        {'project_root': '/project', 'task_ids': '42,43,44'},
    )
    assert result == {'success': True, 'results': []}
    kwargs = task_interceptor.set_task_status.call_args.kwargs
    assert kwargs['task_id'] == '42,43,44'
    assert kwargs['status'] == 'pending'
    assert kwargs['project_root'] == '/project'


@pytest.mark.asyncio
async def test_commit_planning_target_status_defaults_to_pending(
    mcp_server_with_tasks, task_interceptor,
):
    from unittest.mock import AsyncMock
    task_interceptor.set_task_status = AsyncMock(return_value={'success': True})
    await mcp_server_with_tasks._tool_manager.call_tool(
        'commit_planning',
        {'project_root': '/project', 'task_ids': '7'},
    )
    assert task_interceptor.set_task_status.call_args.kwargs['status'] == 'pending'


@pytest.mark.asyncio
async def test_commit_planning_accepts_alternate_targets(
    mcp_server_with_tasks, task_interceptor,
):
    """deferred and cancelled are valid commit targets (commit / abort / discard)."""
    from unittest.mock import AsyncMock
    task_interceptor.set_task_status = AsyncMock(return_value={'success': True})
    for target in ('deferred', 'cancelled'):
        await mcp_server_with_tasks._tool_manager.call_tool(
            'commit_planning',
            {'project_root': '/project', 'task_ids': '7', 'target_status': target},
        )
    statuses = [c.kwargs['status'] for c in task_interceptor.set_task_status.call_args_list]
    assert statuses == ['deferred', 'cancelled']


@pytest.mark.asyncio
async def test_commit_planning_rejects_invalid_target_status(
    mcp_server_with_tasks, task_interceptor,
):
    """Statuses other than pending/deferred/cancelled are rejected at the MCP layer."""
    from unittest.mock import AsyncMock
    task_interceptor.set_task_status = AsyncMock()
    result = await mcp_server_with_tasks._tool_manager.call_tool(
        'commit_planning',
        {'project_root': '/project', 'task_ids': '7', 'target_status': 'in-progress'},
    )
    assert result['error_type'] == 'ValidationError'
    assert 'in-progress' in result['error']
    task_interceptor.set_task_status.assert_not_called()


@pytest.mark.asyncio
async def test_commit_planning_rejects_empty_task_ids(
    mcp_server_with_tasks, task_interceptor,
):
    """Empty / whitespace task_ids string is rejected."""
    from unittest.mock import AsyncMock
    task_interceptor.set_task_status = AsyncMock()
    for ids in ('', '   ', ',,,'):
        result = await mcp_server_with_tasks._tool_manager.call_tool(
            'commit_planning',
            {'project_root': '/project', 'task_ids': ids},
        )
        assert result['error_type'] == 'ValidationError'
    task_interceptor.set_task_status.assert_not_called()


@pytest.mark.asyncio
async def test_commit_planning_rejects_ticket_id_in_batch(
    mcp_server_with_tasks, task_interceptor,
):
    """commit_planning rejects ticket UUIDs — only resolved task ids are valid."""
    from unittest.mock import AsyncMock
    task_interceptor.set_task_status = AsyncMock()
    result = await mcp_server_with_tasks._tool_manager.call_tool(
        'commit_planning',
        {'project_root': '/project', 'task_ids': '42,tkt_abc,44'},
    )
    assert result['error_type'] == 'ValidationError'
    assert 'tkt_abc' in result['error']
    task_interceptor.set_task_status.assert_not_called()


@pytest.mark.asyncio
async def test_commit_planning_relative_path_returns_validation_error(
    mcp_server_with_tasks,
):
    """Relative project_root rejected with the standard ValidationError shape."""
    result = await mcp_server_with_tasks._tool_manager.call_tool(
        'commit_planning',
        {'project_root': 'relative/path', 'task_ids': '7'},
    )
    assert result['error_type'] == 'ValidationError'


@pytest.mark.asyncio
async def test_commit_planning_interceptor_exception_returns_error_type(
    mcp_server_with_tasks, task_interceptor,
):
    """Exceptions from set_task_status surface as {'error', 'error_type'}."""
    from unittest.mock import AsyncMock
    task_interceptor.set_task_status = AsyncMock(
        side_effect=RuntimeError('backend down'),
    )
    result = await mcp_server_with_tasks._tool_manager.call_tool(
        'commit_planning',
        {'project_root': '/project', 'task_ids': '7'},
    )
    assert result['error_type'] == 'RuntimeError'
    assert 'backend down' in result['error']


def test_commit_planning_registered(mcp_server_with_tasks):
    """commit_planning shows up in the MCP server's tool list."""
    tool_names = [t.name for t in mcp_server_with_tasks._tool_manager.list_tools()]
    assert 'commit_planning' in tool_names
