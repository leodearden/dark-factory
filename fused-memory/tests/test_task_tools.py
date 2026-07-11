"""Tests for MCP task-tool behavior (update_task, set_task_status, etc.)."""

import json
import logging
from unittest.mock import AsyncMock

import pytest
import pytest_asyncio

from fused_memory.server.tools import create_mcp_server

_TOOLS_LOGGER = 'fused_memory.server.tools'


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


@pytest_asyncio.fixture
async def real_task_stack(tmp_path):
    """Live SqliteTaskBackend + TaskInterceptor + MCP server stack.

    The status/done_provenance write-authority floor lives in
    ``SqliteTaskBackend.update_task`` (task C1); the mocked
    ``task_interceptor`` fixture above can't exercise it, so BT-C1
    rejection tests need a real backend + interceptor + create_mcp_server
    stack instead. No git repo or created task is needed — the floor
    raises before the row SELECT.
    """
    from fused_memory.backends.sqlite_task_backend import SqliteTaskBackend
    from fused_memory.config.schema import TaskmasterConfig
    from fused_memory.middleware.task_interceptor import TaskInterceptor
    from fused_memory.reconciliation.event_buffer import EventBuffer

    backend = SqliteTaskBackend(TaskmasterConfig(project_root=str(tmp_path)))
    await backend.start()
    event_buffer = EventBuffer(db_path=tmp_path / 'real_stack_eb.db', buffer_size_threshold=100)
    await event_buffer.initialize()
    interceptor = TaskInterceptor(backend, None, event_buffer)
    server = create_mcp_server(AsyncMock(), task_interceptor=interceptor)
    try:
        yield server, interceptor
    finally:
        await event_buffer.close()
        await backend.close()


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
            {'prompt': 'Update the description', 'append': None, 'metadata_mode': None, 'tag': None},
            id='prompt-forwarded',
        ),
        pytest.param(
            {'id': '1', 'project_root': '/project', 'prompt': 'Extra info', 'append': True},
            {'prompt': 'Extra info', 'append': True, 'metadata_mode': None, 'tag': None},
            id='append-true',
        ),
        pytest.param(
            {'id': '1', 'project_root': '/project', 'tag': 'v2'},
            {'prompt': None, 'append': None, 'metadata_mode': None, 'tag': 'v2'},
            id='tag-forwarded',
        ),
        pytest.param(
            {'id': '1', 'project_root': '/project'},
            {'prompt': None, 'append': None, 'metadata_mode': None, 'tag': None},
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
        'title': None,
        'description': None,
        'details': None,
        'priority': None,
        'status': None,
        'dependencies': None,
        'agent_id': None,
    }
    expected_kwargs = {**base_kwargs, **expected_overrides}
    task_interceptor.update_task.assert_called_once_with(**expected_kwargs)


# ------------------------------------------------------------------
# update_task metadata_mode wire-forwarding (step-7 RED / step-8 GREEN)
# ------------------------------------------------------------------


@pytest.mark.asyncio
async def test_update_task_metadata_mode_replace_forwarded(
    mcp_server_with_tasks, task_interceptor,
):
    """metadata_mode='replace' is forwarded raw to the interceptor."""
    await mcp_server_with_tasks._tool_manager.call_tool(
        'update_task',
        {'id': '1', 'project_root': '/project', 'metadata_mode': 'replace'},
    )
    _, kwargs = task_interceptor.update_task.call_args
    assert kwargs['metadata_mode'] == 'replace', (
        f"Expected metadata_mode='replace'; got {kwargs.get('metadata_mode')!r}"
    )
    assert kwargs['append'] is None, (
        f"Expected append=None (unresolved); got {kwargs.get('append')!r}"
    )


@pytest.mark.asyncio
async def test_update_task_append_true_forwarded_raw(
    mcp_server_with_tasks, task_interceptor,
):
    """append=True is forwarded raw (unresolved) alongside metadata_mode=None."""
    await mcp_server_with_tasks._tool_manager.call_tool(
        'update_task',
        {'id': '1', 'project_root': '/project', 'append': True},
    )
    _, kwargs = task_interceptor.update_task.call_args
    assert kwargs['append'] is True, (
        f"Expected append=True; got {kwargs.get('append')!r}"
    )
    assert kwargs['metadata_mode'] is None, (
        f"Expected metadata_mode=None; got {kwargs.get('metadata_mode')!r}"
    )


@pytest.mark.asyncio
async def test_update_task_no_arg_default_forwards_none_none(
    mcp_server_with_tasks, task_interceptor,
):
    """No-arg default: both append=None and metadata_mode=None forwarded (backend resolves)."""
    await mcp_server_with_tasks._tool_manager.call_tool(
        'update_task',
        {'id': '1', 'project_root': '/project'},
    )
    _, kwargs = task_interceptor.update_task.call_args
    assert kwargs['append'] is None, (
        f"Expected append=None; got {kwargs.get('append')!r}"
    )
    assert kwargs['metadata_mode'] is None, (
        f"Expected metadata_mode=None; got {kwargs.get('metadata_mode')!r}"
    )


# ------------------------------------------------------------------
# update_task dotted-ID forwarding (post-SQLite-cutover regression lock)
# ------------------------------------------------------------------


@pytest.mark.asyncio
async def test_update_task_forwards_dotted_subtask_id_to_interceptor(
    mcp_server_with_tasks, task_interceptor,
):
    """Dotted subtask IDs pass through the MCP boundary to the interceptor unchanged.

    Locks the post-SQLite-cutover contract: the stale Node Taskmaster MCP
    wrapper used to reject dotted IDs with "taskId must be a positive integer".
    That wrapper was removed in the SQLite cutover. This test ensures that a
    future defence-in-depth integer-validation guard (analogous to
    _reject_if_ticket_id) cannot silently re-introduce the same rejection.
    """
    result = await mcp_server_with_tasks._tool_manager.call_tool(
        'update_task',
        {
            'id': '2696.1',
            'project_root': '/project',
            'metadata': {'memory_hints': {'entities': ['X']}},
        },
    )
    # (a) No ValidationError or numeric-id rejection.
    assert isinstance(result, dict)
    assert result.get('error_type') != 'ValidationError'
    assert result == {'success': True}

    # (b) The interceptor was called once with the dotted id unchanged.
    task_interceptor.update_task.assert_called_once()
    _, kwargs = task_interceptor.update_task.call_args
    assert kwargs['task_id'] == '2696.1'

    # (c) Dict metadata is JSON-serialized before forwarding (standard coercion).
    # Use json.loads for the comparison so key-order / spacing differences don't
    # cause spurious failures (exact serialization form is covered by the dedicated
    # test_update_task_metadata_dict_coerced_to_json_string test).
    assert json.loads(kwargs['metadata']) == {'memory_hints': {'entities': ['X']}}


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
async def test_update_task_done_provenance_rejection_byte_identical_across_surfaces(
    real_task_stack, tmp_path,
):
    """BT-C1: the tools.py surface and the interceptor surface return the
    byte-identical canonical done_provenance rejection dict.

    Today they diverge: tools.py's ``_reject_done_provenance_in_metadata``
    returns ``{'error_type': 'ValidationError', ...}`` while the interceptor
    returns the canonical ``{'success': False, 'error':
    'done_provenance_via_update_task', ...}`` shape. Drives both surfaces
    against a live SqliteTaskBackend + TaskInterceptor stack so the C1
    backend floor actually fires. No git repo / created task needed — the
    floor raises before the row SELECT.
    """
    from fused_memory.backends.task_backend_errors import done_provenance_via_update_task_error

    server, interceptor = real_task_stack
    payload = {'done_provenance': {'kind': 'merged', 'commit': 'abc'}}

    dict_tools = await server._tool_manager.call_tool(
        'update_task',
        {'id': '1', 'project_root': str(tmp_path), 'metadata': payload},
    )
    dict_interceptor = await interceptor.update_task(
        '1', str(tmp_path), metadata=json.dumps(payload),
    )

    expected = done_provenance_via_update_task_error('1')
    assert dict_tools == expected, f'tools.py surface diverged from canonical shape: {dict_tools}'
    assert dict_interceptor == expected, (
        f'interceptor surface diverged from canonical shape: {dict_interceptor}'
    )
    assert dict_tools == dict_interceptor


@pytest.mark.asyncio
async def test_update_task_rejects_metadata_done_provenance_dict(
    real_task_stack, tmp_path,
):
    """A dict-shaped metadata carrying done_provenance is rejected with the
    canonical done_provenance_via_update_task shape (BT-C1)."""
    from fused_memory.backends.task_backend_errors import done_provenance_via_update_task_error

    server, _interceptor = real_task_stack
    result = await server._tool_manager.call_tool(
        'update_task',
        {
            'id': '1', 'project_root': str(tmp_path),
            'metadata': {'done_provenance': {'kind': 'merged', 'commit': 'abc'}},
        },
    )
    assert result == done_provenance_via_update_task_error('1')


@pytest.mark.asyncio
async def test_update_task_rejects_metadata_done_provenance_json_string(
    real_task_stack, tmp_path,
):
    """A JSON-string metadata carrying done_provenance is also rejected (BT-C1)."""
    from fused_memory.backends.task_backend_errors import done_provenance_via_update_task_error

    server, _interceptor = real_task_stack
    result = await server._tool_manager.call_tool(
        'update_task',
        {
            'id': '1', 'project_root': str(tmp_path),
            'metadata': '{"done_provenance": {"kind": "merged", "commit": "abc"}}',
        },
    )
    assert result == done_provenance_via_update_task_error('1')


@pytest.mark.asyncio
async def test_update_task_allows_unrelated_metadata(
    mcp_server_with_tasks, task_interceptor,
):
    """Other metadata keys still pass through; the gate only blocks done_provenance
    and directory-charter violations."""
    result = await mcp_server_with_tasks._tool_manager.call_tool(
        'update_task',
        {
            'id': '1', 'project_root': '/project',
            'metadata': {'files': ['orchestrator/src/foo.py'], 'priority': 'high'},
        },
    )
    assert result == {'success': True}
    task_interceptor.update_task.assert_called_once()


# ------------------------------------------------------------------
# update_task metadata discard WARN (task 2166 step-11)
#
# _reject_done_provenance_in_metadata's inline str->dict json.loads branch
# must delegate its malformed-input policy to
# shared.task_metadata.parse_metadata(direction='read') and emit a
# task_metadata.schema_warning WARNING on a genuine whole-metadata discard
# (unparseable JSON / non-object) instead of silently returning None — I4.
# The done_provenance rejection/pass-through behavior itself is unchanged.
# ------------------------------------------------------------------


@pytest.mark.asyncio
async def test_update_task_json_string_with_done_provenance_still_rejected(
    real_task_stack, tmp_path, caplog,
):
    """A parseable JSON-string metadata carrying done_provenance is still
    rejected with the canonical shape, with no spurious discard warning."""
    from fused_memory.backends.task_backend_errors import done_provenance_via_update_task_error

    server, _interceptor = real_task_stack
    with caplog.at_level(logging.WARNING, logger=_TOOLS_LOGGER):
        result = await server._tool_manager.call_tool(
            'update_task',
            {
                'id': '1', 'project_root': str(tmp_path),
                'metadata': '{"done_provenance": {"kind": "merged", "commit": "abc"}}',
            },
        )
    assert result == done_provenance_via_update_task_error('1')
    warns = [
        r for r in caplog.records
        if r.name == _TOOLS_LOGGER and r.levelno >= logging.WARNING
    ]
    assert not warns, (
        f'a parseable metadata string must not emit a discard WARNING; got '
        f'{[r.message for r in warns]!r}'
    )


@pytest.mark.asyncio
async def test_update_task_json_string_without_done_provenance_not_rejected(
    mcp_server_with_tasks, task_interceptor, caplog,
):
    """A parseable JSON-string metadata without done_provenance is forwarded,
    with no spurious discard warning."""
    with caplog.at_level(logging.WARNING, logger=_TOOLS_LOGGER):
        result = await mcp_server_with_tasks._tool_manager.call_tool(
            'update_task',
            {
                'id': '1', 'project_root': '/project',
                'metadata': '{"priority": "high"}',
            },
        )
    assert result == {'success': True}
    task_interceptor.update_task.assert_called_once()
    warns = [
        r for r in caplog.records
        if r.name == _TOOLS_LOGGER and r.levelno >= logging.WARNING
    ]
    assert not warns, (
        f'a parseable metadata string must not emit a discard WARNING; got '
        f'{[r.message for r in warns]!r}'
    )


# ------------------------------------------------------------------
# update_task lock-charter guard (Change #2)
# ------------------------------------------------------------------


@pytest.mark.asyncio
async def test_update_task_rejects_directory_in_files_dict_metadata(
    mcp_server_with_tasks, task_interceptor,
):
    """update_task rejects metadata.files containing a directory entry (dict form).

    Mirrors the submit_task guard: a directory path must not be written to
    metadata.files via update_task either.
    """
    result = await mcp_server_with_tasks._tool_manager.call_tool(
        'update_task',
        {
            'id': '1', 'project_root': '/project',
            'metadata': {'files': ['orchestrator/']},
        },
    )
    assert result.get('error_type') == 'LockCharterViolation', (
        f'Expected LockCharterViolation, got: {result}'
    )
    assert 'orchestrator/' in result.get('directory_paths', []), (
        f'Expected orchestrator/ in directory_paths; got {result}'
    )
    task_interceptor.update_task.assert_not_called()


@pytest.mark.asyncio
async def test_update_task_rejects_directory_in_files_merge_mode_files_only(
    mcp_server_with_tasks, task_interceptor,
):
    """update_task (metadata_mode='merge', files-only payload) rejects directory entries.

    Covers the REQUIRED merge-mode case where a caller supplies only
    {'files': [...]} — this is the scheduler's writeback shape.
    """
    result = await mcp_server_with_tasks._tool_manager.call_tool(
        'update_task',
        {
            'id': '1', 'project_root': '/project',
            'metadata': {'files': ['src']},
            'metadata_mode': 'merge',
        },
    )
    assert result.get('error_type') == 'LockCharterViolation', (
        f'Expected LockCharterViolation, got: {result}'
    )
    task_interceptor.update_task.assert_not_called()


@pytest.mark.asyncio
async def test_update_task_accepts_file_level_files(
    mcp_server_with_tasks, task_interceptor,
):
    """update_task forwards when metadata.files contains only file-level paths."""
    result = await mcp_server_with_tasks._tool_manager.call_tool(
        'update_task',
        {
            'id': '1', 'project_root': '/project',
            'metadata': {'files': ['pkg/mod/foo.py']},
        },
    )
    assert result == {'success': True}, f'Expected forwarded success; got {result}'
    task_interceptor.update_task.assert_called_once()


@pytest.mark.asyncio
async def test_update_task_accepts_empty_files_list(
    mcp_server_with_tasks, task_interceptor,
):
    """update_task forwards when metadata.files is [] (defer-to-architect value)."""
    result = await mcp_server_with_tasks._tool_manager.call_tool(
        'update_task',
        {
            'id': '1', 'project_root': '/project',
            'metadata': {'files': []},
        },
    )
    assert result == {'success': True}, f'Expected forwarded success; got {result}'
    task_interceptor.update_task.assert_called_once()


# ------------------------------------------------------------------
# update_task rejects status= kwarg (2026-05-08 hardening)
# ------------------------------------------------------------------


@pytest.mark.asyncio
@pytest.mark.parametrize('bad_status', ['done', 'pending', 'cancelled', 'in-progress', 'blocked'])
async def test_update_task_rejects_status_kwarg(
    real_task_stack, tmp_path, bad_status,
):
    """update_task with status= is rejected — agents must use set_task_status.

    Closes the bypass route used to mark reify tasks done without going through
    the terminal-exit, phantom-done, and done-provenance gates. The
    enforcement point is the SqliteTaskBackend write-authority floor (task
    C1), surfaced through the interceptor as the canonical
    status_via_update_task rejection dict (task C2) — drive through the real
    stack rather than asserting on a mocked interceptor.
    """
    from fused_memory.backends.task_backend_errors import status_via_update_task_error

    server, _interceptor = real_task_stack
    result = await server._tool_manager.call_tool(
        'update_task',
        {'id': '1', 'project_root': str(tmp_path), 'status': bad_status},
    )
    assert result == status_via_update_task_error('1', bad_status)


@pytest.mark.asyncio
async def test_update_task_status_none_still_allowed(
    mcp_server_with_tasks, task_interceptor,
):
    """status=None (the default) is the metadata-only path and must still work."""
    await mcp_server_with_tasks._tool_manager.call_tool(
        'update_task',
        {'id': '1', 'project_root': '/project', 'status': None},
    )
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
                 'update_task', 'remove_task', 'add_dependency',
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


def test_add_subtask_mcp_tool_not_registered():
    """add_subtask MCP tool must not be registered after DF-D (task 1543).

    RED assertion: fails while add_subtask is still in server/tools.py,
    passes once step-4 removes it.
    """
    from fused_memory.backends.sqlite_task_backend import SqliteTaskBackend
    from fused_memory.middleware.task_interceptor import TaskInterceptor

    mock_service = AsyncMock()
    server = create_mcp_server(mock_service)
    tool_names = [t.name for t in server._tool_manager.list_tools()]
    assert 'add_subtask' not in tool_names, (
        "'add_subtask' is still registered as an MCP tool; "
        'DF-D (task 1543) step-4 must delete it from server/tools.py.'
    )
    assert not hasattr(SqliteTaskBackend, 'add_subtask'), (
        'SqliteTaskBackend.add_subtask still exists; step-4 must delete it.'
    )
    assert not hasattr(TaskInterceptor, 'add_subtask'), (
        'TaskInterceptor.add_subtask still exists; step-4 must delete it.'
    )


def test_add_subtask_result_not_importable():
    """AddSubtaskResult must not be importable after DF-D (task 1543).

    RED assertion: fails while AddSubtaskResult is still defined in
    task_backend_types.py, passes once step-4 removes it.
    """
    import importlib

    # If already in sys.modules (cached), reload to re-check the symbol.
    module_name = 'fused_memory.backends.task_backend_types'
    mod = importlib.import_module(module_name)
    assert not hasattr(mod, 'AddSubtaskResult'), (
        'AddSubtaskResult is still exported from fused_memory.backends.task_backend_types; '
        'DF-D step-4 must delete it.'
    )


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


@pytest.mark.asyncio
async def test_set_task_status_accepts_merge_deferred(
    mcp_server_with_tasks, task_interceptor,
):
    """set_task_status with 'merge-deferred' passes validation and forwards to the interceptor.

    merge-deferred is the non-terminal holding state for atomic-train members
    (PRD orchestrator-atomic-train-merge §9.2, task 1519).
    """
    task_interceptor.set_task_status = AsyncMock(return_value={'success': True})
    result = await mcp_server_with_tasks._tool_manager.call_tool(
        'set_task_status',
        {'id': '1', 'project_root': '/project', 'status': 'merge-deferred'},
    )
    assert 'error' not in result, (
        f"Expected no error for 'merge-deferred', got: {result.get('error')}"
    )
    task_interceptor.set_task_status.assert_called_once_with(
        task_id='1',
        status='merge-deferred',
        project_root='/project',
        tag=None,
        done_provenance=None,
        reopen_reason=None,
        agent_id=None,
    )


@pytest.mark.asyncio
async def test_set_task_status_forwards_agent_id_to_interceptor(
    mcp_server_with_tasks, task_interceptor,
):
    """set_task_status forwards caller identity (agent_id) to the interceptor
    so the transition-legality gate (task 2175/rho1b) can classify an actor.
    """
    task_interceptor.set_task_status = AsyncMock(return_value={'success': True})
    await mcp_server_with_tasks._tool_manager.call_tool(
        'set_task_status',
        {
            'id': '1',
            'project_root': '/project',
            'status': 'blocked',
            'agent_id': 'recon-stage-7',
        },
    )
    task_interceptor.set_task_status.assert_awaited_once()
    assert task_interceptor.set_task_status.call_args.kwargs.get('agent_id') == 'recon-stage-7', (
        f'Expected agent_id to be forwarded to the interceptor; '
        f'call_args={task_interceptor.set_task_status.call_args!r}'
    )


# ------------------------------------------------------------------
# submit_task / update_task forward ctx-resolved agent_id to the interceptor
# (task 2175 step-11/step-12 added handler acceptance; task 2223/W5-ε completes
# the plumbing by forwarding agent_id into interceptor.submit_task /
# interceptor.update_task — the substrate W5-ζ ReconWritePolicy consults inside
# interceptor.update_task, gated on a recon-stage agent_id. The interceptor
# captures agent_id as an explicit keyword-only param so it never leaks into the
# ticket blob / tm.update_task; see test_task_write_agent_id.py.)
# ------------------------------------------------------------------


@pytest.mark.asyncio
async def test_submit_task_accepts_agent_id_and_forwards_existing_kwargs_unchanged(
    mcp_server_with_tasks, task_interceptor,
):
    """submit_task forwards agent_id (caller identity) to the interceptor without
    disturbing its existing forwarding of prompt/priority/etc."""
    task_interceptor.submit_task = AsyncMock(return_value={'ticket': 'tkt_x'})
    result = await mcp_server_with_tasks._tool_manager.call_tool(
        'submit_task',
        {
            'project_root': '/project',
            'prompt': 'Do the thing',
            'priority': 'high',
            'agent_id': 'recon-stage-7',
            # A recon-stage caller must now declare metadata.execution_class
            # (η guard, task 2225) or the submit is rejected before the
            # interceptor is reached — this test only exercises agent_id
            # forwarding, so declare a valid class to stay past the guard.
            'metadata': {'execution_class': 'code_tdd'},
        },
    )
    assert 'error' not in result
    task_interceptor.submit_task.assert_awaited_once()
    kwargs = task_interceptor.submit_task.call_args.kwargs
    assert kwargs.get('prompt') == 'Do the thing', f'Expected prompt forwarded; got {kwargs!r}'
    assert kwargs.get('priority') == 'high', f'Expected priority forwarded; got {kwargs!r}'
    assert kwargs.get('agent_id') == 'recon-stage-7', (
        f'submit_task must forward agent_id to the interceptor; got {kwargs!r}'
    )


@pytest.mark.asyncio
async def test_update_task_accepts_agent_id_and_forwards_existing_kwargs_unchanged(
    mcp_server_with_tasks, task_interceptor,
):
    """update_task forwards agent_id (caller identity) to the interceptor without
    disturbing its existing exact-kwarg forwarding."""
    result = await mcp_server_with_tasks._tool_manager.call_tool(
        'update_task',
        {
            'id': '1',
            'project_root': '/project',
            'title': 'New title',
            'priority': 'high',
            'agent_id': 'recon-stage-7',
        },
    )
    assert 'error' not in result
    task_interceptor.update_task.assert_called_once_with(
        task_id='1',
        project_root='/project',
        prompt=None,
        metadata=None,
        append=None,
        metadata_mode=None,
        tag=None,
        title='New title',
        description=None,
        details=None,
        priority='high',
        status=None,
        dependencies=None,
        agent_id='recon-stage-7',
    )


# ------------------------------------------------------------------
# submit_task execution-class guard η (task 2225/W5-η) — boundary test E1
#
# The unit-level invariant matrix for execution_class_error /
# inject_execution_class lives in test_execution_class_guard.py; these
# tests assert the guard is WIRED into the submit_task tool boundary so the
# E1 user-observable signal is actually delivered: a recon-stage caller that
# omits metadata.execution_class is rejected BEFORE the interceptor is
# reached, a valid class is accepted and persists into the forwarded
# metadata, and non-recon callers are entirely unaffected.
# ------------------------------------------------------------------


@pytest.mark.asyncio
async def test_submit_task_recon_stage_without_execution_class_rejected(
    mcp_server_with_tasks, task_interceptor,
):
    """E1: a recon-stage submit_task with no metadata.execution_class is
    rejected with a ValidationError and never reaches the interceptor."""
    task_interceptor.submit_task = AsyncMock(return_value={'ticket': 'tkt_x'})
    result = await mcp_server_with_tasks._tool_manager.call_tool(
        'submit_task',
        {
            'project_root': '/project',
            'prompt': 'Reconcile task 7',
            'agent_id': 'recon-stage-task_knowledge_sync',
        },
    )
    assert result.get('error_type') == 'ValidationError', f'got {result!r}'
    assert 'execution_class' in result.get('error', '')
    task_interceptor.submit_task.assert_not_awaited()


@pytest.mark.asyncio
async def test_submit_task_recon_stage_with_valid_execution_class_accepted_and_persisted(
    mcp_server_with_tasks, task_interceptor,
):
    """E1: a recon-stage submit_task with a valid execution_class is accepted
    and the declared class persists into the metadata forwarded to the
    interceptor (for 2085's routing + ξ's premise-lint to consume)."""
    task_interceptor.submit_task = AsyncMock(return_value={'ticket': 'tkt_x'})
    result = await mcp_server_with_tasks._tool_manager.call_tool(
        'submit_task',
        {
            'project_root': '/project',
            'prompt': 'Reconcile task 7',
            'agent_id': 'recon-stage-task_knowledge_sync',
            'metadata': {'execution_class': 'operational'},
        },
    )
    assert 'error' not in result
    task_interceptor.submit_task.assert_awaited_once()
    forwarded = task_interceptor.submit_task.call_args.kwargs.get('metadata')
    assert isinstance(forwarded, dict), f'metadata not normalised to dict; got {forwarded!r}'
    assert forwarded.get('execution_class') == 'operational', (
        f'declared execution_class must persist into forwarded metadata; got {forwarded!r}'
    )


@pytest.mark.asyncio
async def test_submit_task_non_recon_without_execution_class_accepted(
    mcp_server_with_tasks, task_interceptor,
):
    """A non-recon caller (e.g. claude-interactive) is never enforced: a
    submit_task with no execution_class is accepted and reaches the
    interceptor unchanged."""
    task_interceptor.submit_task = AsyncMock(return_value={'ticket': 'tkt_x'})
    result = await mcp_server_with_tasks._tool_manager.call_tool(
        'submit_task',
        {
            'project_root': '/project',
            'prompt': 'Do the thing',
            'agent_id': 'claude-interactive',
        },
    )
    assert 'error' not in result
    task_interceptor.submit_task.assert_awaited_once()


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
# rebuild_candidate_key_index (task 2402)
# ------------------------------------------------------------------


@pytest.mark.asyncio
async def test_rebuild_candidate_key_index_forwards_project_root_and_returns_result(
    mcp_server_with_tasks, task_interceptor,
):
    """rebuild_candidate_key_index forwards project_root to
    task_interceptor.taskmaster.reaudit_candidate_key_index (the live,
    on-demand re-run of the self-heal migration) and returns its result
    dict unchanged."""
    canned_result = {
        'index_built': True, 'healed': [], 'flagged': [], 'user_version': 4,
    }
    task_interceptor.taskmaster.reaudit_candidate_key_index.return_value = canned_result
    result = await mcp_server_with_tasks._tool_manager.call_tool(
        'rebuild_candidate_key_index',
        {'project_root': '/project'},
    )
    task_interceptor.taskmaster.reaudit_candidate_key_index.assert_called_once_with(
        '/project',
    )
    assert result == canned_result


@pytest.mark.asyncio
async def test_rebuild_candidate_key_index_without_taskmaster_returns_not_configured():
    """rebuild_candidate_key_index without a task_interceptor returns a clear
    'not configured' error, matching trigger_reconciliation's guard, instead
    of raising (e.g. AttributeError on ``task_interceptor.taskmaster``)."""
    mock_service = AsyncMock()
    server = create_mcp_server(mock_service)  # No task_interceptor
    result = await server._tool_manager.call_tool(
        'rebuild_candidate_key_index',
        {'project_root': '/project'},
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
# pre-done hook error passthrough at the MCP surface
# ---------------------------------------------------------------------------


@pytest.mark.asyncio
async def test_set_task_status_mcp_tool_surfaces_pre_done_hook_error(
    mcp_server_with_tasks, task_interceptor,
):
    """The set_task_status MCP tool passes pre_done_hook_rejected errors through unchanged.

    Confirms the existing dict-passthrough at tools.py propagates the new
    error code correctly — no exception raised, no field stripping.
    """
    task_interceptor.set_task_status = AsyncMock(
        return_value={
            'success': False,
            'error': 'pre_done_hook_rejected',
            'task_id': '1',
            'exit_code': 1,
            'stderr': 'hook validation failed',
            'command': '/bin/false',
            'hint': 'Fix the underlying issue and retry.',
        }
    )
    result = await mcp_server_with_tasks._tool_manager.call_tool(
        'set_task_status',
        {'id': '1', 'project_root': '/project', 'status': 'done'},
    )
    assert isinstance(result, dict)
    assert result['error'] == 'pre_done_hook_rejected'
    assert result['task_id'] == '1'
    assert result['exit_code'] == 1
    # error_type is only added by the exception handler — must not appear here
    assert 'error_type' not in result


# ---------------------------------------------------------------------------
# step-17: ticket-shaped id rejection for all id-accepting tools
# ---------------------------------------------------------------------------

@pytest.mark.asyncio
@pytest.mark.parametrize('tool_name,extra_kwargs', [
    ('set_task_status', {'status': 'done'}),
    ('update_task', {'title': 'new title'}),
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
    task_interceptor.remove_tasks = AsyncMock(return_value={'success': True})
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
    # Must NOT have called the backend. The wire tool ``remove_task``
    # dispatches to the interceptor's ``remove_tasks`` method (renamed at
    # the protocol layer); map the wire name to the interceptor attr.
    interceptor_attr = 'remove_tasks' if tool_name == 'remove_task' else tool_name
    getattr(task_interceptor, interceptor_attr).assert_not_called()


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
# remove_task wire-shape normalisation
# ---------------------------------------------------------------------------


@pytest.mark.asyncio
async def test_remove_task_single_id_normalises_to_singleton_list(
    mcp_server_with_tasks, task_interceptor,
):
    """A single id on the wire reaches the interceptor as ['<id>']."""
    task_interceptor.remove_tasks = AsyncMock(
        return_value={'successful': 1, 'failed': 0, 'removed_ids': ['5']},
    )
    await mcp_server_with_tasks._tool_manager.call_tool(
        'remove_task', {'id': '5', 'project_root': '/project'},
    )
    task_interceptor.remove_tasks.assert_called_once()
    _, kwargs = task_interceptor.remove_tasks.call_args
    assert kwargs['ids'] == ['5']


@pytest.mark.asyncio
async def test_remove_task_csv_normalises_to_list(
    mcp_server_with_tasks, task_interceptor,
):
    """The MCP tool accepts a comma-separated id string and forwards a
    structured ``list[str]`` to the interceptor — the wire-shape boundary."""
    task_interceptor.remove_tasks = AsyncMock(
        return_value={'successful': 3, 'failed': 0,
                      'removed_ids': ['966.1', '966.2', '1680.1']},
    )
    await mcp_server_with_tasks._tool_manager.call_tool(
        'remove_task',
        {'id': '966.1, 966.2 ,1680.1', 'project_root': '/project'},
    )
    task_interceptor.remove_tasks.assert_called_once()
    _, kwargs = task_interceptor.remove_tasks.call_args
    # CSV split + per-element strip
    assert kwargs['ids'] == ['966.1', '966.2', '1680.1']


@pytest.mark.asyncio
async def test_remove_task_empty_after_strip_is_noop(
    mcp_server_with_tasks, task_interceptor,
):
    """All-whitespace / empty-CSV input returns the empty-noop DTO without
    touching the interceptor."""
    task_interceptor.remove_tasks = AsyncMock()
    result = await mcp_server_with_tasks._tool_manager.call_tool(
        'remove_task', {'id': ' , ,', 'project_root': '/project'},
    )
    assert result == {
        'successful': 0,
        'failed': 0,
        'removed_ids': [],
        'message': 'no ids supplied',
    }
    task_interceptor.remove_tasks.assert_not_called()


@pytest.mark.asyncio
async def test_remove_task_taskmaster_error_returns_structured_dict(
    mcp_server_with_tasks, task_interceptor,
):
    """TaskmasterError from the backend is converted to a structured error dict.

    Wire-contract regression: when the interceptor raises
    ``TaskmasterError('INVALID_TASK_ID', ...)`` the MCP exception handler must
    return ``{'error': '...INVALID_TASK_ID...', 'error_type': 'TaskmasterError'}``
    — the shape that programmatic callers parse.  A nested-subtask-id value is
    used as a realistic driver but the test's actual coverage is the generic
    exception-to-dict adapter in the MCP handler.  Also asserts that the CSV
    split forwarded the malformed id as ``ids=['1.2.3']`` (i.e. the MCP
    boundary doesn't pre-validate grammar; it delegates to the backend).
    """
    from fused_memory.backends.task_backend_errors import TaskmasterError

    task_interceptor.remove_tasks = AsyncMock(
        side_effect=TaskmasterError(
            'INVALID_TASK_ID', "nested subtask ids not supported: '1.2.3'"
        )
    )

    result = await mcp_server_with_tasks._tool_manager.call_tool(
        'remove_task', {'id': '1.2.3', 'project_root': '/project'},
    )

    assert result['error_type'] == 'TaskmasterError'
    assert 'INVALID_TASK_ID' in result['error']
    assert 'nested subtask ids not supported' in result['error']

    # Confirm the CSV split forwarded the id without pre-validation.
    task_interceptor.remove_tasks.assert_called_once()
    _, kwargs = task_interceptor.remove_tasks.call_args
    assert kwargs['ids'] == ['1.2.3']


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


# ---------------------------------------------------------------------------
# add_dependency / remove_dependency qualified depends_on wire tests (step-11)
# ---------------------------------------------------------------------------


@pytest.mark.asyncio
async def test_add_dependency_qualified_not_ticket_rejected_and_forwarded(
    mcp_server_with_tasks, task_interceptor,
):
    """A qualified depends_on ("project_id:task_id") is not tkt_-rejected and
    is forwarded verbatim to task_interceptor.add_dependency.

    Guards that _reject_if_ticket_id only catches the 'tkt_' prefix — a
    colon-separated qualified id is NOT tkt_-shaped and must pass through
    unchanged.
    """
    task_interceptor.add_dependency = AsyncMock(
        return_value={'id': '1', 'dependency_id': 'dark_factory:13', 'message': 'ok'},
    )
    result = await mcp_server_with_tasks._tool_manager.call_tool(
        'add_dependency',
        {'id': '1', 'depends_on': 'dark_factory:13', 'project_root': '/project'},
    )
    # Must NOT be a ValidationError ticket rejection.
    assert result.get('error_type') != 'ValidationError', (
        f'Qualified depends_on was incorrectly ticket-rejected: {result}'
    )
    task_interceptor.add_dependency.assert_awaited_once()
    _, kwargs = task_interceptor.add_dependency.call_args
    assert kwargs['depends_on'] == 'dark_factory:13'


@pytest.mark.asyncio
async def test_add_dependency_self_loop_taskmaster_error_wire_shape(
    mcp_server_with_tasks, task_interceptor,
):
    """When the backend raises TaskmasterError for self-loop, the tool returns
    {'error': 'TASKMASTER_TOOL_ERROR: add_dependency: task cannot depend on itself',
     'error_type': 'TaskmasterError'}.

    Confirms the specific error wire shape from the observable signal (plan
    §OBSERVABLE SIGNAL).
    """
    from fused_memory.backends.task_backend_errors import TaskmasterError

    task_interceptor.add_dependency = AsyncMock(
        side_effect=TaskmasterError(
            'TASKMASTER_TOOL_ERROR',
            'add_dependency: task cannot depend on itself',
        )
    )
    result = await mcp_server_with_tasks._tool_manager.call_tool(
        'add_dependency',
        {'id': '1', 'depends_on': 'my_project:1', 'project_root': '/project'},
    )
    assert result['error_type'] == 'TaskmasterError'
    assert 'TASKMASTER_TOOL_ERROR' in result['error']
    assert 'task cannot depend on itself' in result['error']


# ---------------------------------------------------------------------------
# get_tasks / get_task project provenance stamp (task 1661)
# ---------------------------------------------------------------------------


@pytest.mark.asyncio
async def test_get_tasks_stamps_project_provenance(mcp_server_with_tasks, task_interceptor):
    """get_tasks result envelope carries project_id and project_root provenance stamps.

    The stamped keys allow the caller (including Stage-3 LLM agents) to verify
    which project a bulk dump came from.  Existing 'tasks' list must be untouched.
    """
    task_interceptor.get_tasks = AsyncMock(
        return_value={'tasks': [{'id': '1', 'title': 'real df task', 'status': 'pending'}]}
    )
    result = await mcp_server_with_tasks._tool_manager.call_tool(
        'get_tasks',
        {'project_root': '/home/leo/src/dark-factory'},
    )
    # Provenance stamps must be present
    assert result.get('project_id') == 'dark_factory', (
        f"Expected project_id='dark_factory', got: {result.get('project_id')!r}"
    )
    assert result.get('project_root') == '/home/leo/src/dark-factory', (
        f"Expected project_root='/home/leo/src/dark-factory', got: {result.get('project_root')!r}"
    )
    # The original tasks list must be preserved untouched
    assert result.get('tasks') == [{'id': '1', 'title': 'real df task', 'status': 'pending'}]


@pytest.mark.asyncio
async def test_get_task_stamps_project_id(mcp_server_with_tasks, task_interceptor):
    """get_task result dict carries project_id and project_root provenance stamps.

    Full symmetry with get_tasks lets callers cross-check single-task and bulk reads.
    The existing id/title/status fields must be preserved intact.
    """
    task_interceptor.get_task = AsyncMock(
        return_value={'id': 1654, 'title': 'real df task', 'status': 'done'}
    )
    result = await mcp_server_with_tasks._tool_manager.call_tool(
        'get_task',
        {'id': '1654', 'project_root': '/home/leo/src/dark-factory'},
    )
    # Provenance stamps must be present (symmetric with get_tasks)
    assert result.get('project_id') == 'dark_factory', (
        f"Expected project_id='dark_factory', got: {result.get('project_id')!r}"
    )
    assert result.get('project_root') == '/home/leo/src/dark-factory', (
        f"Expected project_root='/home/leo/src/dark-factory', got: {result.get('project_root')!r}"
    )
    # Existing fields must be preserved
    assert result.get('id') == 1654
    assert result.get('title') == 'real df task'
    assert result.get('status') == 'done'


@pytest.mark.asyncio
async def test_remove_dependency_qualified_not_ticket_rejected_and_forwarded(
    mcp_server_with_tasks, task_interceptor,
):
    """A qualified depends_on is not tkt_-rejected and is forwarded verbatim to
    task_interceptor.remove_dependency.
    """
    task_interceptor.remove_dependency = AsyncMock(
        return_value={'id': '1', 'dependency_id': 'dark_factory:13', 'message': 'ok'},
    )
    result = await mcp_server_with_tasks._tool_manager.call_tool(
        'remove_dependency',
        {'id': '1', 'depends_on': 'dark_factory:13', 'project_root': '/project'},
    )
    # Must NOT be a ValidationError ticket rejection.
    assert result.get('error_type') != 'ValidationError', (
        f'Qualified depends_on was incorrectly ticket-rejected: {result}'
    )
    task_interceptor.remove_dependency.assert_awaited_once()
    _, kwargs = task_interceptor.remove_dependency.call_args
    assert kwargs['depends_on'] == 'dark_factory:13'


# ---------------------------------------------------------------------------
# get_tasks pagination (task 1727)
# ---------------------------------------------------------------------------

_FIVE_TASKS = [
    {'id': str(i), 'title': f'task {i}', 'status': 'pending'} for i in range(1, 6)
]


@pytest.mark.asyncio
async def test_get_tasks_pagination_slices_and_reports_metadata(
    mcp_server_with_tasks, task_interceptor
):
    """get_tasks with page_size slices the task list and attaches a pagination envelope.

    Provenance stamps (project_id, project_root) must survive pagination.
    Three sub-scenarios in one fixture setup:
      (a) first page  → tasks 1-2, has_more=True
      (b) last page   → task 5 only, has_more=False
      (c) beyond end  → empty list, returned=0, has_more=False
    """
    task_interceptor.get_tasks = AsyncMock(return_value={'tasks': list(_FIVE_TASKS)})

    # (a) First page: offset=0, page_size=2
    result = await mcp_server_with_tasks._tool_manager.call_tool(
        'get_tasks',
        {'project_root': '/home/leo/src/dark-factory', 'page_size': 2, 'offset': 0},
    )
    assert result.get('tasks') == _FIVE_TASKS[:2], f'Expected first 2 tasks, got: {result.get("tasks")}'
    assert result.get('pagination') == {
        'total': 5,
        'offset': 0,
        'page_size': 2,
        'returned': 2,
        'has_more': True,
    }, f'Unexpected pagination dict: {result.get("pagination")}'
    # Provenance stamps must survive pagination
    assert result.get('project_id') == 'dark_factory'
    assert result.get('project_root') == '/home/leo/src/dark-factory'

    # (b) Last item: offset=4, page_size=2 → only task 5
    task_interceptor.get_tasks = AsyncMock(return_value={'tasks': list(_FIVE_TASKS)})
    result2 = await mcp_server_with_tasks._tool_manager.call_tool(
        'get_tasks',
        {'project_root': '/home/leo/src/dark-factory', 'page_size': 2, 'offset': 4},
    )
    assert result2.get('tasks') == _FIVE_TASKS[4:], f'Expected last task, got: {result2.get("tasks")}'
    assert result2['pagination']['returned'] == 1
    assert result2['pagination']['has_more'] is False
    assert result2['pagination']['total'] == 5

    # (c) Past end: offset=10
    task_interceptor.get_tasks = AsyncMock(return_value={'tasks': list(_FIVE_TASKS)})
    result3 = await mcp_server_with_tasks._tool_manager.call_tool(
        'get_tasks',
        {'project_root': '/home/leo/src/dark-factory', 'page_size': 2, 'offset': 10},
    )
    assert result3.get('tasks') == [], f'Expected empty list, got: {result3.get("tasks")}'
    assert result3['pagination']['returned'] == 0
    assert result3['pagination']['has_more'] is False


@pytest.mark.asyncio
async def test_get_tasks_pagination_validation_and_backward_compat(
    mcp_server_with_tasks, task_interceptor
):
    """get_tasks pagination: backward-compat + input validation.

    (a) Backward-compat: no page_size → full list returned, no 'pagination' key.
    (b) page_size=0  → ValidationError, interceptor NOT called.
    (c) page_size=-1 → ValidationError, interceptor NOT called.
    (d) offset=-1 with page_size=2 → ValidationError, interceptor NOT called.
    """
    # (a) Backward-compat: default call (no page_size)
    task_interceptor.get_tasks = AsyncMock(return_value={'tasks': list(_FIVE_TASKS)})
    result = await mcp_server_with_tasks._tool_manager.call_tool(
        'get_tasks',
        {'project_root': '/home/leo/src/dark-factory'},
    )
    assert result.get('tasks') == _FIVE_TASKS, (
        f'Backward-compat: full list expected, got: {result.get("tasks")}'
    )
    assert 'pagination' not in result, (
        f'Backward-compat: pagination key must be absent, got: {result}'
    )

    # (b) page_size=0 → ValidationError
    task_interceptor.get_tasks = AsyncMock(return_value={'tasks': list(_FIVE_TASKS)})
    bad0 = await mcp_server_with_tasks._tool_manager.call_tool(
        'get_tasks',
        {'project_root': '/home/leo/src/dark-factory', 'page_size': 0},
    )
    assert bad0.get('error_type') == 'ValidationError', f'Expected ValidationError for page_size=0, got: {bad0}'
    assert 'page_size' in bad0.get('error', '').lower(), f'Error message should mention page_size: {bad0}'
    task_interceptor.get_tasks.assert_not_awaited()

    # (c) page_size=-1 → ValidationError
    task_interceptor.get_tasks = AsyncMock(return_value={'tasks': list(_FIVE_TASKS)})
    bad_neg = await mcp_server_with_tasks._tool_manager.call_tool(
        'get_tasks',
        {'project_root': '/home/leo/src/dark-factory', 'page_size': -1},
    )
    assert bad_neg.get('error_type') == 'ValidationError', f'Expected ValidationError for page_size=-1, got: {bad_neg}'
    task_interceptor.get_tasks.assert_not_awaited()

    # (d) offset=-1 with page_size=2 → ValidationError
    task_interceptor.get_tasks = AsyncMock(return_value={'tasks': list(_FIVE_TASKS)})
    bad_off = await mcp_server_with_tasks._tool_manager.call_tool(
        'get_tasks',
        {'project_root': '/home/leo/src/dark-factory', 'page_size': 2, 'offset': -1},
    )
    assert bad_off.get('error_type') == 'ValidationError', f'Expected ValidationError for offset=-1, got: {bad_off}'
    assert 'offset' in bad_off.get('error', '').lower(), f'Error message should mention offset: {bad_off}'
    task_interceptor.get_tasks.assert_not_awaited()


# ---------------------------------------------------------------------------
# get_tasks statuses filter (task 1758)
# ---------------------------------------------------------------------------

_PENDING_TASKS = [
    {'id': '1', 'title': 'task 1', 'status': 'pending'},
    {'id': '3', 'title': 'task 3', 'status': 'pending'},
]


@pytest.mark.asyncio
async def test_get_tasks_status_filter_forwarded_and_composes(
    mcp_server_with_tasks, task_interceptor
):
    """get_tasks tool: statuses forwarded to interceptor; composes with pagination; validation.

    (a) statuses=['pending'] forwarded as kwarg; result tasks + provenance stamps pass through.
    (b) statuses=['pending'] + page_size=1 → pagination slices the already-filtered list.
    (c) no statuses (default) → interceptor called with statuses=None, full result returned.
    (d) statuses=[] → forwarded as-is (NOT a ValidationError).
    (e) statuses='done' (non-list) → ValidationError, interceptor NOT awaited.
    """
    # (a) Forwarding: statuses=['pending'] kwarg forwarded, result + provenance pass through
    task_interceptor.get_tasks = AsyncMock(return_value={'tasks': list(_PENDING_TASKS)})
    result_a = await mcp_server_with_tasks._tool_manager.call_tool(
        'get_tasks',
        {'project_root': '/home/leo/src/dark-factory', 'statuses': ['pending']},
    )
    task_interceptor.get_tasks.assert_awaited_once()
    _, kwargs_a = task_interceptor.get_tasks.call_args
    assert kwargs_a.get('statuses') == ['pending'], (
        f'Expected statuses kwarg forwarded, got call_args: {task_interceptor.get_tasks.call_args}'
    )
    assert result_a.get('tasks') == _PENDING_TASKS, (
        f'Expected pending tasks passed through, got: {result_a.get("tasks")}'
    )
    assert result_a.get('project_id') == 'dark_factory', f'Missing project_id: {result_a}'
    assert result_a.get('project_root') == '/home/leo/src/dark-factory', f'Missing project_root: {result_a}'

    # (b) Compose: statuses + page_size → pagination slices the already-filtered list
    task_interceptor.get_tasks = AsyncMock(return_value={'tasks': list(_PENDING_TASKS)})
    result_b = await mcp_server_with_tasks._tool_manager.call_tool(
        'get_tasks',
        {'project_root': '/home/leo/src/dark-factory', 'statuses': ['pending'], 'page_size': 1},
    )
    assert result_b.get('tasks') == _PENDING_TASKS[:1], (
        f'Expected first 1 task from filtered list, got: {result_b.get("tasks")}'
    )
    assert result_b.get('pagination') == {
        'total': 2,
        'offset': 0,
        'page_size': 1,
        'returned': 1,
        'has_more': True,
    }, f'Unexpected pagination: {result_b.get("pagination")}'

    # (c) Default (no statuses) → interceptor called with statuses=None, full result returned
    task_interceptor.get_tasks = AsyncMock(return_value={'tasks': list(_PENDING_TASKS)})
    result_c = await mcp_server_with_tasks._tool_manager.call_tool(
        'get_tasks',
        {'project_root': '/home/leo/src/dark-factory'},
    )
    task_interceptor.get_tasks.assert_awaited_once()
    _, kwargs_c = task_interceptor.get_tasks.call_args
    assert kwargs_c.get('statuses') is None, (
        f'Expected statuses=None when omitted, got: {kwargs_c.get("statuses")}'
    )
    assert result_c.get('tasks') == _PENDING_TASKS, f'Expected full result, got: {result_c}'

    # (d) statuses=[] is valid ("match nothing") — forwarded as-is, not a ValidationError
    task_interceptor.get_tasks = AsyncMock(return_value={'tasks': []})
    result_d = await mcp_server_with_tasks._tool_manager.call_tool(
        'get_tasks',
        {'project_root': '/home/leo/src/dark-factory', 'statuses': []},
    )
    assert result_d.get('error_type') != 'ValidationError', (
        f'statuses=[] should not be a ValidationError, got: {result_d}'
    )
    task_interceptor.get_tasks.assert_awaited_once()
    _, kwargs_d = task_interceptor.get_tasks.call_args
    assert kwargs_d.get('statuses') == [], (
        f'Expected statuses=[] forwarded, got: {kwargs_d.get("statuses")}'
    )

    # (e) statuses='done' (bare string, non-list) → ValidationError, interceptor NOT awaited
    task_interceptor.get_tasks = AsyncMock(return_value={'tasks': []})
    result_e = await mcp_server_with_tasks._tool_manager.call_tool(
        'get_tasks',
        {'project_root': '/home/leo/src/dark-factory', 'statuses': 'done'},
    )
    assert result_e.get('error_type') == 'ValidationError', (
        f'Expected ValidationError for non-list statuses, got: {result_e}'
    )
    assert 'statuses' in result_e.get('error', '').lower(), (
        f'Error message should mention statuses: {result_e}'
    )
    task_interceptor.get_tasks.assert_not_awaited()

    # (f) statuses=[None] (list with non-string elements) → ValidationError, interceptor NOT awaited
    task_interceptor.get_tasks = AsyncMock(return_value={'tasks': []})
    result_f = await mcp_server_with_tasks._tool_manager.call_tool(
        'get_tasks',
        {'project_root': '/home/leo/src/dark-factory', 'statuses': [None]},
    )
    assert result_f.get('error_type') == 'ValidationError', (
        f'Expected ValidationError for statuses=[None], got: {result_f}'
    )
    assert 'statuses' in result_f.get('error', '').lower(), (
        f'Error message should mention statuses: {result_f}'
    )
    task_interceptor.get_tasks.assert_not_awaited()


# ---------------------------------------------------------------------------
# Lock-charter guard γ — submit_task wiring tests (step-5 RED / step-6 GREEN)
# ---------------------------------------------------------------------------


@pytest.mark.asyncio
async def test_submit_task_rejects_directory_in_files_dict_metadata(
    mcp_server_with_tasks, task_interceptor,
):
    """submit_task rejects metadata.files containing a directory (dict form)."""
    task_interceptor.submit_task = AsyncMock(return_value={'task_id': '1'})
    result = await mcp_server_with_tasks._tool_manager.call_tool(
        'submit_task',
        {
            'project_root': '/project',
            'title': 'X',
            'metadata': {'files': ['orchestrator/']},
        },
    )
    assert result.get('error_type') == 'LockCharterViolation'
    assert 'orchestrator/' in result.get('directory_paths', [])
    task_interceptor.submit_task.assert_not_called()


@pytest.mark.asyncio
async def test_submit_task_rejects_directory_in_files_json_string_metadata(
    mcp_server_with_tasks, task_interceptor,
):
    """submit_task rejects metadata.files containing a directory (JSON-string form)."""
    task_interceptor.submit_task = AsyncMock(return_value={'task_id': '2'})
    result = await mcp_server_with_tasks._tool_manager.call_tool(
        'submit_task',
        {
            'project_root': '/project',
            'title': 'Y',
            'metadata': '{"files": ["src"]}',
        },
    )
    assert result.get('error_type') == 'LockCharterViolation'
    assert 'src' in result.get('directory_paths', [])
    task_interceptor.submit_task.assert_not_called()


@pytest.mark.asyncio
async def test_submit_task_planning_mode_rejects_directory_in_files(
    mcp_server_with_tasks, task_interceptor,
):
    """planning_mode=True submit_task is also guarded (catches #4552 human-decompose class)."""
    task_interceptor.submit_task = AsyncMock(
        return_value={'task_id': '3', 'status': 'deferred', 'planning_mode': True},
    )
    result = await mcp_server_with_tasks._tool_manager.call_tool(
        'submit_task',
        {
            'project_root': '/project',
            'title': 'Z',
            'planning_mode': True,
            'metadata': {'files': ['crates/reify-eval/src']},
        },
    )
    assert result.get('error_type') == 'LockCharterViolation'
    assert 'crates/reify-eval/src' in result.get('directory_paths', [])
    task_interceptor.submit_task.assert_not_called()


@pytest.mark.asyncio
async def test_submit_task_accepts_file_level_files(
    mcp_server_with_tasks, task_interceptor,
):
    """submit_task forwards when metadata.files contains only file-level paths."""
    task_interceptor.submit_task = AsyncMock(return_value={'task_id': '10'})
    result = await mcp_server_with_tasks._tool_manager.call_tool(
        'submit_task',
        {
            'project_root': '/project',
            'title': 'A',
            'metadata': {'files': ['pkg/mod/foo.py']},
        },
    )
    assert result == {'task_id': '10'}
    task_interceptor.submit_task.assert_called_once()


@pytest.mark.asyncio
async def test_submit_task_accepts_empty_files_list(
    mcp_server_with_tasks, task_interceptor,
):
    """submit_task forwards when metadata.files is [] (defer-to-architect value)."""
    task_interceptor.submit_task = AsyncMock(return_value={'task_id': '11'})
    result = await mcp_server_with_tasks._tool_manager.call_tool(
        'submit_task',
        {
            'project_root': '/project',
            'title': 'B',
            'metadata': {'files': []},
        },
    )
    assert result == {'task_id': '11'}
    task_interceptor.submit_task.assert_called_once()


@pytest.mark.asyncio
async def test_submit_task_accepts_none_metadata(
    mcp_server_with_tasks, task_interceptor,
):
    """submit_task forwards when metadata is None (no files declared)."""
    task_interceptor.submit_task = AsyncMock(return_value={'task_id': '12'})
    result = await mcp_server_with_tasks._tool_manager.call_tool(
        'submit_task',
        {
            'project_root': '/project',
            'title': 'C',
        },
    )
    assert result == {'task_id': '12'}
    task_interceptor.submit_task.assert_called_once()


# ---------------------------------------------------------------------------
# Lock-charter guard γ — commit_planning wiring tests (step-7 RED / step-8 GREEN)
# ---------------------------------------------------------------------------


@pytest.mark.asyncio
async def test_commit_planning_rejects_directory_in_task_files(
    mcp_server_with_tasks, task_interceptor,
):
    """commit_planning rejects atomically when any task has directory files."""
    # Task 42 has file-level files; task 43 has a directory → whole batch rejected.
    async def _get_task(tid, *args, **kwargs):
        if tid == '43':
            return {'id': '43', 'metadata': {'files': ['orchestrator/']}}
        return {'id': tid, 'metadata': {'files': ['a/b.rs']}}

    task_interceptor.get_task = AsyncMock(side_effect=_get_task)
    task_interceptor.set_task_status = AsyncMock()

    result = await mcp_server_with_tasks._tool_manager.call_tool(
        'commit_planning',
        {'project_root': '/project', 'task_ids': '42,43'},
    )

    assert result.get('error_type') == 'LockCharterViolation'
    # The offending task id must appear in the error message.
    assert '43' in result.get('error', '')
    # The offending directory path must appear in directory_paths.
    assert 'orchestrator/' in result.get('directory_paths', [])
    # set_task_status must NOT have been called (whole-batch atomic reject).
    task_interceptor.set_task_status.assert_not_called()


@pytest.mark.asyncio
async def test_commit_planning_accepts_all_file_level_tasks(
    mcp_server_with_tasks, task_interceptor,
):
    """commit_planning forwards when all tasks have file-level metadata.files."""
    task_interceptor.get_task = AsyncMock(
        return_value={'id': '7', 'metadata': {'files': ['src/main.py']}},
    )
    task_interceptor.set_task_status = AsyncMock(return_value={'success': True})

    result = await mcp_server_with_tasks._tool_manager.call_tool(
        'commit_planning',
        {'project_root': '/project', 'task_ids': '7'},
    )

    assert result == {'success': True}
    task_interceptor.set_task_status.assert_called_once()


@pytest.mark.asyncio
async def test_commit_planning_accepts_empty_files_list(
    mcp_server_with_tasks, task_interceptor,
):
    """commit_planning accepts tasks with files=[] (defer-to-architect value)."""
    task_interceptor.get_task = AsyncMock(
        return_value={'id': '8', 'metadata': {'files': []}},
    )
    task_interceptor.set_task_status = AsyncMock(return_value={'success': True})

    result = await mcp_server_with_tasks._tool_manager.call_tool(
        'commit_planning',
        {'project_root': '/project', 'task_ids': '8'},
    )

    assert result == {'success': True}
    task_interceptor.set_task_status.assert_called_once()


@pytest.mark.asyncio
async def test_commit_planning_early_rejection_paths_do_not_call_get_task(
    mcp_server_with_tasks, task_interceptor,
):
    """Early-rejection paths (invalid target/empty ids/ticket) skip get_task and set_task_status."""
    task_interceptor.get_task = AsyncMock()
    task_interceptor.set_task_status = AsyncMock()

    # Invalid target_status
    r1 = await mcp_server_with_tasks._tool_manager.call_tool(
        'commit_planning',
        {'project_root': '/project', 'task_ids': '7', 'target_status': 'in-progress'},
    )
    assert r1['error_type'] == 'ValidationError'

    # Empty ids
    r2 = await mcp_server_with_tasks._tool_manager.call_tool(
        'commit_planning',
        {'project_root': '/project', 'task_ids': ''},
    )
    assert r2['error_type'] == 'ValidationError'

    # Ticket id in batch
    r3 = await mcp_server_with_tasks._tool_manager.call_tool(
        'commit_planning',
        {'project_root': '/project', 'task_ids': '42,tkt_abc'},
    )
    assert r3['error_type'] == 'ValidationError'

    task_interceptor.get_task.assert_not_called()
    task_interceptor.set_task_status.assert_not_called()


@pytest.mark.asyncio
async def test_commit_planning_bad_task_id_returns_structured_error(
    mcp_server_with_tasks, task_interceptor,
):
    """commit_planning returns structured error dict when get_task raises (e.g. unknown id).

    Previously the guard loop ran OUTSIDE the try/except, so a TaskmasterError
    from get_task would escape and change the response shape.  Now the whole
    guard + set_task_status block is inside a single try/except, guaranteeing
    a consistent {'error', 'error_type'} response even for missing/invalid ids.
    """
    from unittest.mock import AsyncMock

    from fused_memory.backends.task_backend_errors import TaskmasterError

    task_interceptor.get_task = AsyncMock(
        side_effect=TaskmasterError('INVALID_TASK_ID', 'task not found: 99999'),
    )
    task_interceptor.set_task_status = AsyncMock()

    result = await mcp_server_with_tasks._tool_manager.call_tool(
        'commit_planning',
        {'project_root': '/project', 'task_ids': '99999'},
    )

    assert 'error' in result
    assert result.get('error_type') == 'TaskmasterError'
    task_interceptor.set_task_status.assert_not_called()
