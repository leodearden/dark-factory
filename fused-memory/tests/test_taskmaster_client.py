"""Tests for Taskmaster MCP client backend."""

import json
from unittest.mock import AsyncMock, MagicMock

import pytest
from mcp.types import TextContent

from fused_memory.backends.taskmaster_client import TaskmasterBackend
from fused_memory.config.schema import TaskmasterConfig


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
    # Mock the session
    mock_session = AsyncMock()
    c._session = mock_session
    return c, mock_session


def _mock_tool_result(data: dict):
    """Create a mock MCP tool result."""
    result = MagicMock()
    text_block = TextContent(type='text', text=json.dumps(data))
    result.content = [text_block]
    return result


@pytest.mark.asyncio
async def test_call_tool(client):
    c, session = client
    session.call_tool = AsyncMock(return_value=_mock_tool_result({'tasks': []}))

    result = await c.call_tool('get_tasks', {'projectRoot': '/project'})
    assert result == {'tasks': []}
    session.call_tool.assert_called_once_with('get_tasks', {'projectRoot': '/project'})


@pytest.mark.asyncio
async def test_get_tasks(client):
    c, session = client
    session.call_tool = AsyncMock(return_value=_mock_tool_result({
        'tasks': [{'id': '1', 'title': 'Task 1'}]
    }))

    result = await c.get_tasks('/project')
    assert 'tasks' in result
    session.call_tool.assert_called_once_with('get_tasks', {'projectRoot': '/project'})


@pytest.mark.asyncio
async def test_get_task(client):
    c, session = client
    session.call_tool = AsyncMock(return_value=_mock_tool_result({
        'id': '1', 'title': 'Task 1', 'status': 'pending'
    }))

    result = await c.get_task('1', '/project')
    assert result['id'] == '1'


@pytest.mark.asyncio
async def test_set_task_status(client):
    c, session = client
    session.call_tool = AsyncMock(return_value=_mock_tool_result({'success': True}))

    result = await c.set_task_status('1', 'done', '/project')
    assert result['success'] is True
    call_args = session.call_tool.call_args[0]
    assert call_args[0] == 'set_task_status'
    assert call_args[1]['id'] == '1'
    assert call_args[1]['status'] == 'done'


@pytest.mark.asyncio
async def test_add_task(client):
    c, session = client
    session.call_tool = AsyncMock(return_value=_mock_tool_result({'id': '2'}))

    result = await c.add_task(prompt='New task', priority='high', project_root='/project')
    assert result['id'] == '2'
    call_args = session.call_tool.call_args[0]
    assert call_args[1]['prompt'] == 'New task'
    assert call_args[1]['priority'] == 'high'


@pytest.mark.asyncio
async def test_update_task_with_metadata(client):
    c, session = client
    session.call_tool = AsyncMock(return_value=_mock_tool_result({'success': True}))

    await c.update_task('1', metadata='{"key": "value"}', project_root='/project')
    call_args = session.call_tool.call_args[0]
    assert call_args[1]['metadata'] == '{"key": "value"}'


@pytest.mark.asyncio
async def test_expand_task(client):
    c, session = client
    session.call_tool = AsyncMock(return_value=_mock_tool_result({
        'subtasks': [{'id': '1.1'}, {'id': '1.2'}]
    }))

    result = await c.expand_task('1', '/project', num='3', force=True)
    assert len(result['subtasks']) == 2
    call_args = session.call_tool.call_args[0]
    assert call_args[1]['force'] is True
    assert call_args[1]['num'] == '3'


@pytest.mark.asyncio
async def test_base_args_with_tag(client):
    c, session = client
    args = c._base_args('/project', 'my-tag')
    assert args['projectRoot'] == '/project'
    assert args['tag'] == 'my-tag'


@pytest.mark.asyncio
async def test_base_args_default_project_root(client):
    c, session = client
    args = c._base_args()
    assert args['projectRoot'] == '/project'  # From config


@pytest.mark.asyncio
async def test_require_session_raises_without_init(config):
    c = TaskmasterBackend(config)
    with pytest.raises(RuntimeError, match='not connected'):
        c._require_session()


@pytest.mark.asyncio
async def test_call_tool_non_json_response(client):
    """Non-JSON response should be wrapped."""
    c, session = client
    result = MagicMock()
    text_block = TextContent(type='text', text='Not JSON')
    result.content = [text_block]
    session.call_tool = AsyncMock(return_value=result)

    data = await c.call_tool('some_tool', {})
    assert data == {'text': 'Not JSON'}
