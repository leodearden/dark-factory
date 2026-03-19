"""Tests for BaseStage tool closures using correct identifiers."""

from unittest.mock import AsyncMock

import pytest

from fused_memory.config.schema import ReconciliationConfig
from fused_memory.models.reconciliation import StageId
from fused_memory.reconciliation.stages.base import BaseStage


@pytest.fixture
def mock_memory_service():
    svc = AsyncMock()
    svc.search = AsyncMock(return_value=[])
    svc.get_entity = AsyncMock(return_value={'nodes': [], 'edges': []})
    svc.get_episodes = AsyncMock(return_value=[])
    svc.get_status = AsyncMock(return_value={'ok': True})
    svc.add_memory = AsyncMock(return_value=AsyncMock(model_dump=lambda: {}))
    svc.delete_memory = AsyncMock(return_value={'deleted': True})
    return svc


@pytest.fixture
def mock_taskmaster():
    tm = AsyncMock()
    tm.get_tasks = AsyncMock(return_value={'tasks': []})
    tm.get_task = AsyncMock(return_value={'id': '1', 'title': 'Test'})
    tm.set_task_status = AsyncMock(return_value={'success': True})
    tm.update_task = AsyncMock(return_value={'success': True})
    tm.add_task = AsyncMock(return_value={'id': '2'})
    tm.add_subtask = AsyncMock(return_value={'id': '1.1'})
    tm.remove_task = AsyncMock(return_value={'success': True})
    tm.add_dependency = AsyncMock(return_value={'success': True})
    tm.remove_dependency = AsyncMock(return_value={'success': True})
    return tm


@pytest.fixture
def config():
    return ReconciliationConfig(
        enabled=True,
        explore_codebase_root='/tmp/test',
        agent_llm_provider='anthropic',
        agent_llm_model='claude-sonnet-4-20250514',
    )


@pytest.fixture
def stage(mock_memory_service, mock_taskmaster, config):
    journal = AsyncMock()
    s = BaseStage(StageId.memory_consolidator, mock_memory_service, mock_taskmaster, journal, config)
    s.project_id = 'dark_factory'
    s.project_root = '/home/leo/src/dark-factory'
    return s


class TestMemoryReadToolsUseProjectId:
    """Memory read tools should use self.project_id (logical)."""

    @pytest.mark.asyncio
    async def test_search_uses_project_id(self, stage, mock_memory_service):
        tools = stage._memory_read_tools()
        await tools['search'].function(query='test query')
        mock_memory_service.search.assert_called_once()
        assert mock_memory_service.search.call_args.kwargs['project_id'] == 'dark_factory'

    @pytest.mark.asyncio
    async def test_get_entity_uses_project_id(self, stage, mock_memory_service):
        tools = stage._memory_read_tools()
        await tools['get_entity'].function(name='TestEntity')
        mock_memory_service.get_entity.assert_called_once()
        assert mock_memory_service.get_entity.call_args.kwargs['project_id'] == 'dark_factory'

    @pytest.mark.asyncio
    async def test_get_episodes_uses_project_id(self, stage, mock_memory_service):
        tools = stage._memory_read_tools()
        await tools['get_episodes'].function()
        mock_memory_service.get_episodes.assert_called_once()
        assert mock_memory_service.get_episodes.call_args.kwargs['project_id'] == 'dark_factory'


class TestMemoryWriteToolsUseProjectId:
    """Memory write tools should use self.project_id (logical)."""

    @pytest.mark.asyncio
    async def test_add_memory_uses_project_id(self, stage, mock_memory_service):
        tools = stage._memory_write_tools()
        await tools['add_memory'].function(content='test content')
        mock_memory_service.add_memory.assert_called_once()
        assert mock_memory_service.add_memory.call_args.kwargs['project_id'] == 'dark_factory'

    @pytest.mark.asyncio
    async def test_delete_memory_uses_project_id(self, stage, mock_memory_service):
        tools = stage._memory_write_tools()
        await tools['delete_memory'].function(memory_id='m1', store='mem0')
        mock_memory_service.delete_memory.assert_called_once()
        assert mock_memory_service.delete_memory.call_args.kwargs['project_id'] == 'dark_factory'


class TestTaskReadToolsUseProjectRoot:
    """Task read tools should use self.project_root (filesystem path)."""

    @pytest.mark.asyncio
    async def test_get_tasks_uses_project_root(self, stage, mock_taskmaster):
        tools = stage._task_read_tools()
        await tools['get_tasks'].function()
        mock_taskmaster.get_tasks.assert_called_once_with(
            project_root='/home/leo/src/dark-factory'
        )

    @pytest.mark.asyncio
    async def test_get_task_uses_project_root(self, stage, mock_taskmaster):
        tools = stage._task_read_tools()
        await tools['get_task'].function(id='1')
        mock_taskmaster.get_task.assert_called_once_with(
            task_id='1', project_root='/home/leo/src/dark-factory'
        )


class TestTaskWriteToolsUseProjectRoot:
    """Task write tools should use self.project_root (filesystem path)."""

    @pytest.mark.asyncio
    async def test_set_task_status_uses_project_root(self, stage, mock_taskmaster):
        tools = stage._task_write_tools()
        await tools['set_task_status'].function(id='1', status='done')
        mock_taskmaster.set_task_status.assert_called_once_with(
            task_id='1', status='done', project_root='/home/leo/src/dark-factory'
        )

    @pytest.mark.asyncio
    async def test_update_task_uses_project_root(self, stage, mock_taskmaster):
        tools = stage._task_write_tools()
        await tools['update_task'].function(id='1', prompt='new prompt')
        mock_taskmaster.update_task.assert_called_once_with(
            task_id='1', prompt='new prompt', metadata=None,
            project_root='/home/leo/src/dark-factory'
        )

    @pytest.mark.asyncio
    async def test_add_task_uses_project_root(self, stage, mock_taskmaster):
        tools = stage._task_write_tools()
        await tools['add_task'].function(title='New Task')
        mock_taskmaster.add_task.assert_called_once_with(
            prompt=None, title='New Task', project_root='/home/leo/src/dark-factory'
        )

    @pytest.mark.asyncio
    async def test_remove_task_uses_project_root(self, stage, mock_taskmaster):
        tools = stage._task_write_tools()
        await tools['remove_task'].function(id='1')
        mock_taskmaster.remove_task.assert_called_once_with(
            task_id='1', project_root='/home/leo/src/dark-factory'
        )
