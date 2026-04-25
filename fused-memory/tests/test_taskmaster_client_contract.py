"""Contract tests for TaskmasterBackend wrapper-method DTOs.

Each test feeds the mocked MCP session a real Taskmaster wire-envelope
payload (``{data: {...}, version, tag}`` on success;
``{text: 'Error: ...'}`` for tool-level failure via
``createErrorResponse``) and asserts that the wrapper returns the flat
DTO declared in :mod:`fused_memory.backends.taskmaster_types`.

These tests are the primary tripwire for Taskmaster wire-shape drift.
Integration-level coverage against a real Taskmaster MCP subprocess
lives in ``tests/integration/test_taskmaster_mcp_contract.py`` — that
suite is skipped automatically until ``taskmaster-ai/dist/mcp-server.js``
is built; see its module docstring for the bootstrap command.
"""

from __future__ import annotations

import json
from unittest.mock import AsyncMock, MagicMock

import pytest
from mcp.types import TextContent

from fused_memory.backends.taskmaster_client import TaskmasterBackend
from fused_memory.backends.taskmaster_types import TaskmasterError
from fused_memory.config.schema import TaskmasterConfig

# ── fixtures ────────────────────────────────────────────────────────


@pytest.fixture
def config():
    return TaskmasterConfig(
        transport='stdio',
        command='node',
        args=['server.js'],
        project_root='/project',
    )


@pytest.fixture
def client(config):
    c = TaskmasterBackend(config)
    mock_session = AsyncMock()
    c._session = mock_session
    return c, mock_session


def _success_result(data: dict | list) -> MagicMock:
    """Real wire-envelope success payload shape that Taskmaster emits."""
    envelope = {
        'data': data,
        'version': {'version': '0.27.0', 'name': 'task-master-ai'},
        'tag': {'currentTag': 'master', 'availableTags': ['master']},
    }
    result = MagicMock()
    result.content = [TextContent(type='text', text=json.dumps(envelope))]
    result.isError = False
    return result


def _error_result(message: str = 'boom') -> MagicMock:
    """Real wire error payload shape that createErrorResponse emits."""
    text = f'Error: {message}\nVersion: 0.27.0\nName: task-master-ai'
    result = MagicMock()
    result.content = [TextContent(type='text', text=text)]
    result.isError = True
    return result


# ── add_task ────────────────────────────────────────────────────────


@pytest.mark.asyncio
async def test_add_task_returns_flat_dto(client):
    c, session = client
    session.call_tool = AsyncMock(return_value=_success_result({
        'taskId': 42,
        'message': 'Successfully added new task #42',
        'telemetryData': {'tokens': 123},
        'tagInfo': {'currentTag': 'master'},
    }))

    dto = await c.add_task(project_root='/project', prompt='do thing')

    assert dto['id'] == '42'
    assert dto['message'] == 'Successfully added new task #42'


@pytest.mark.asyncio
async def test_add_task_raises_taskmaster_error_on_tool_failure(client):
    c, session = client
    session.call_tool = AsyncMock(
        return_value=_error_result('Either the prompt parameter or both title and description are required'),
    )

    with pytest.raises(TaskmasterError) as exc_info:
        await c.add_task(project_root='/project')

    assert 'prompt parameter' in exc_info.value.message
    assert exc_info.value.code == 'TASKMASTER_TOOL_ERROR'


# ── update_task ─────────────────────────────────────────────────────


@pytest.mark.asyncio
async def test_update_task_returns_flat_dto(client):
    c, session = client
    session.call_tool = AsyncMock(return_value=_success_result({
        'message': 'Task 7 updated',
        'taskId': 7,
        'tasksPath': '/project/.taskmaster/tasks/tasks.json',
        'useResearch': False,
        'updated': True,
        'updatedTask': {'id': '7', 'title': 'updated title'},
        'telemetryData': {},
    }))

    dto = await c.update_task('7', project_root='/project', prompt='refine')

    assert dto['id'] == '7'
    assert dto['updated'] is True
    assert dto['updated_task'] == {'id': '7', 'title': 'updated title'}
    assert 'updated' in dto['message']


@pytest.mark.asyncio
async def test_update_task_raises_on_error(client):
    c, session = client
    session.call_tool = AsyncMock(return_value=_error_result('No prompt provided'))
    with pytest.raises(TaskmasterError):
        await c.update_task('7', project_root='/project')


# ── set_task_status ─────────────────────────────────────────────────


@pytest.mark.asyncio
async def test_set_task_status_returns_flat_dto(client):
    c, session = client
    session.call_tool = AsyncMock(return_value=_success_result({
        'message': 'Successfully updated 1 task(s) to "done"',
        'tasks': [
            {'taskId': '7', 'oldStatus': 'pending', 'newStatus': 'done'},
        ],
    }))

    dto = await c.set_task_status('7', 'done', project_root='/project')

    assert 'done' in dto['message']
    assert dto['tasks'][0]['newStatus'] == 'done'


@pytest.mark.asyncio
async def test_set_task_status_raises_on_error(client):
    c, session = client
    session.call_tool = AsyncMock(return_value=_error_result('invalid task id'))
    with pytest.raises(TaskmasterError):
        await c.set_task_status('nope', 'done', project_root='/project')


# ── add_subtask ─────────────────────────────────────────────────────


@pytest.mark.asyncio
async def test_add_subtask_returns_flat_dto(client):
    c, session = client
    session.call_tool = AsyncMock(return_value=_success_result({
        'message': 'New subtask 3.2 successfully created',
        'subtask': {
            'id': '3.2',
            'title': 'Sub',
            'status': 'pending',
        },
    }))

    dto = await c.add_subtask('3', project_root='/project', title='Sub')

    assert dto['id'] == '3.2'
    assert dto['parent_id'] == '3'
    assert 'created' in dto['message']
    assert dto['subtask']['id'] == '3.2'


@pytest.mark.asyncio
async def test_add_subtask_raises_on_error(client):
    c, session = client
    session.call_tool = AsyncMock(return_value=_error_result('Parent task not found'))
    with pytest.raises(TaskmasterError):
        await c.add_subtask('999', project_root='/project', title='x')


# ── remove_task ─────────────────────────────────────────────────────


@pytest.mark.asyncio
async def test_remove_task_returns_flat_dto(client):
    c, session = client
    session.call_tool = AsyncMock(return_value=_success_result({
        'totalTasks': 1,
        'successful': 1,
        'failed': 0,
        'removedTasks': [{'id': '5', 'title': 'gone'}],
        'message': 'Removed 1 task',
        'tasksPath': '/p/.taskmaster/tasks/tasks.json',
        'tag': 'master',
    }))

    dto = await c.remove_task('5', project_root='/project')

    assert dto['successful'] == 1
    assert dto['failed'] == 0
    assert dto['removed_ids'] == ['5']
    assert dto['message'] == 'Removed 1 task'


@pytest.mark.asyncio
async def test_remove_task_raises_on_error(client):
    c, session = client
    session.call_tool = AsyncMock(return_value=_error_result('Task ID is required'))
    with pytest.raises(TaskmasterError):
        await c.remove_task('', project_root='/project')


# ── add_dependency / remove_dependency ──────────────────────────────


@pytest.mark.asyncio
async def test_add_dependency_returns_flat_dto(client):
    c, session = client
    session.call_tool = AsyncMock(return_value=_success_result({
        'message': 'Added',
        'taskId': 7,
        'dependencyId': 3,
    }))
    dto = await c.add_dependency('7', '3', project_root='/project')
    assert dto['id'] == '7'
    assert dto['dependency_id'] == '3'


@pytest.mark.asyncio
async def test_remove_dependency_returns_flat_dto(client):
    c, session = client
    session.call_tool = AsyncMock(return_value=_success_result({
        'message': 'Removed',
        'taskId': 7,
        'dependencyId': 3,
    }))
    dto = await c.remove_dependency('7', '3', project_root='/project')
    assert dto['id'] == '7'
    assert dto['dependency_id'] == '3'


@pytest.mark.asyncio
async def test_add_dependency_raises_on_error(client):
    c, session = client
    session.call_tool = AsyncMock(return_value=_error_result('cycle would be introduced'))
    with pytest.raises(TaskmasterError):
        await c.add_dependency('7', '3', project_root='/project')


# ── validate_dependencies ───────────────────────────────────────────


@pytest.mark.asyncio
async def test_validate_dependencies_returns_flat_dto(client):
    c, session = client
    session.call_tool = AsyncMock(return_value=_success_result({
        'message': 'Dependencies validated successfully',
        'tasksPath': '/p/.taskmaster/tasks/tasks.json',
    }))

    dto = await c.validate_dependencies(project_root='/project')

    assert dto['message'] == 'Dependencies validated successfully'


# ── expand_task ─────────────────────────────────────────────────────


@pytest.mark.asyncio
async def test_expand_task_returns_flat_dto(client):
    c, session = client
    session.call_tool = AsyncMock(return_value=_success_result({
        'task': {'id': '7', 'title': 'parent', 'subtasks': [{'id': '7.1'}, {'id': '7.2'}]},
        'subtasksAdded': 2,
        'hasExistingSubtasks': False,
        'telemetryData': {},
    }))

    dto = await c.expand_task('7', project_root='/project')

    assert dto['subtasks_added'] == 2
    assert dto['has_existing_subtasks'] is False
    assert dto['task']['id'] == '7'


@pytest.mark.asyncio
async def test_expand_task_skipped_variant(client):
    """The JS direct-function emits a skipped variant when subtasks exist
    and ``force`` isn't passed. DTO should still parse."""
    c, session = client
    session.call_tool = AsyncMock(return_value=_success_result({
        'message': 'Task 7 already has subtasks. Expansion skipped.',
        'task': {'id': '7', 'subtasks': []},
        'subtasksAdded': 0,
        'hasExistingSubtasks': True,
    }))

    dto = await c.expand_task('7', project_root='/project')

    assert dto['subtasks_added'] == 0
    assert dto['has_existing_subtasks'] is True


# ── parse_prd ───────────────────────────────────────────────────────


@pytest.mark.asyncio
async def test_parse_prd_returns_flat_dto(client):
    c, session = client
    session.call_tool = AsyncMock(return_value=_success_result({
        'message': 'Generated 7 tasks',
        'outputPath': '/p/.taskmaster/tasks/tasks.json',
        'telemetryData': {},
    }))

    dto = await c.parse_prd(input_path='/p/prd.md', project_root='/project')

    assert dto['output_path'] == '/p/.taskmaster/tasks/tasks.json'
    assert '7 tasks' in dto['message']


# ── get_tasks ───────────────────────────────────────────────────────


@pytest.mark.asyncio
async def test_get_tasks_returns_flat_dto(client):
    c, session = client
    session.call_tool = AsyncMock(return_value=_success_result({
        'tasks': [
            {'id': '1', 'title': 'A', 'status': 'pending'},
            {'id': '2', 'title': 'B', 'status': 'done'},
        ],
        'filter': 'all',
        'stats': {'total': 2},
    }))

    dto = await c.get_tasks(project_root='/project')

    assert isinstance(dto['tasks'], list)
    assert len(dto['tasks']) == 2
    assert dto['tasks'][0]['id'] == '1'


# ── get_task ────────────────────────────────────────────────────────


@pytest.mark.asyncio
async def test_get_task_returns_task_dict_directly(client):
    """For a single id, Taskmaster puts the task dict at ``data`` (no wrapper)."""
    c, session = client
    session.call_tool = AsyncMock(return_value=_success_result({
        'id': 7,
        'title': 'Real task',
        'status': 'in-progress',
        'description': 'desc',
        'priority': 'high',
        'subtasks': [],
        'dependencies': [],
    }))

    dto = await c.get_task('7', project_root='/project')

    assert dto['id'] == 7
    assert dto['title'] == 'Real task'
    assert dto['status'] == 'in-progress'


@pytest.mark.asyncio
async def test_get_task_raises_on_not_found(client):
    c, session = client
    session.call_tool = AsyncMock(return_value=_error_result('No tasks found for ID(s): 99'))
    with pytest.raises(TaskmasterError):
        await c.get_task('99', project_root='/project')


# ── malformed envelope guard ────────────────────────────────────────


@pytest.mark.asyncio
async def test_unexpected_shape_raises(client):
    """If neither `data` nor an `Error:` text is present, adapter must raise."""
    c, session = client
    result = MagicMock()
    result.content = [TextContent(type='text', text=json.dumps({'unexpected': True}))]
    session.call_tool = AsyncMock(return_value=result)

    with pytest.raises(TaskmasterError) as exc_info:
        await c.add_task(project_root='/project', prompt='x')

    assert exc_info.value.code == 'UNEXPECTED_RESPONSE_SHAPE'
