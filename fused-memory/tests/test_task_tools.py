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
# Defensive tool registration (always registered, even without Taskmaster)
# ------------------------------------------------------------------


def test_task_tools_registered_without_interceptor():
    """Task tools are registered even when no task_interceptor is provided."""
    mock_service = AsyncMock()
    server = create_mcp_server(mock_service)  # No task_interceptor
    tool_names = [t.name for t in server._tool_manager.list_tools()]
    for name in ['get_tasks', 'get_task', 'set_task_status', 'add_task',
                 'update_task', 'add_subtask', 'remove_task', 'add_dependency',
                 'remove_dependency', 'expand_task', 'parse_prd']:
        assert name in tool_names, f'{name} should be registered'


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
