"""Tests for the context builder — unit tests with mocked backends."""

from unittest.mock import AsyncMock, MagicMock

import pytest

from fused_memory.context.builder import ContextBuilder
from fused_memory.models.enums import MemoryCategory, SourceStore
from fused_memory.models.memory import MemoryResult


def _make_result(content: str, store: SourceStore = SourceStore.graphiti, category: MemoryCategory | None = None) -> MemoryResult:
    return MemoryResult(
        id='test-id',
        content=content,
        category=category,
        source_store=store,
        relevance_score=0.9,
    )


@pytest.fixture
def mock_memory():
    svc = MagicMock()
    svc.search = AsyncMock(return_value=[])
    svc.get_entity = AsyncMock(return_value={'nodes': [], 'edges': []})
    return svc


@pytest.fixture
def mock_taskmaster():
    tm = MagicMock()
    tm.get_tasks = AsyncMock(return_value={'tasks': []})
    tm.get_task = AsyncMock(return_value={'task': {'id': '7', 'title': 'Test task', 'status': 'in-progress'}})
    return tm


@pytest.fixture
def builder(mock_memory, mock_taskmaster):
    return ContextBuilder(memory_service=mock_memory, taskmaster=mock_taskmaster)


@pytest.fixture
def builder_no_tasks(mock_memory):
    return ContextBuilder(memory_service=mock_memory)


class TestBriefingStructure:
    @pytest.mark.asyncio
    async def test_empty_briefing(self, builder_no_tasks, mock_memory):
        """With no results, returns a fallback message."""
        briefing = await builder_no_tasks.build_briefing(project_id='test')
        assert 'No context available' in briefing

    @pytest.mark.asyncio
    async def test_briefing_has_header(self, builder, mock_memory):
        """Briefing always starts with the header."""
        mock_memory.search = AsyncMock(return_value=[
            _make_result('Project uses microservices architecture'),
        ])
        briefing = await builder.build_briefing(
            project_id='test', include_task_tree=False
        )
        assert briefing.startswith('# Context Briefing')

    @pytest.mark.asyncio
    async def test_overview_section(self, builder, mock_memory):
        """Overview search results appear under Project Overview."""
        mock_memory.search = AsyncMock(return_value=[
            _make_result('Dark Factory is a software factory'),
        ])
        briefing = await builder.build_briefing(
            project_id='test', include_task_tree=False
        )
        assert '## Project Overview' in briefing
        assert 'Dark Factory is a software factory' in briefing

    @pytest.mark.asyncio
    async def test_multiple_sections(self, builder, mock_memory):
        """Multiple search types produce multiple sections."""
        call_count = 0

        async def varying_search(**kwargs):
            nonlocal call_count
            call_count += 1
            if call_count == 1:
                return [_make_result('Architecture overview')]
            elif call_count == 2:
                return [_make_result('Chose PostgreSQL', category=MemoryCategory.decisions_and_rationale)]
            elif call_count == 3:
                return [_make_result('Use ruff for formatting', category=MemoryCategory.preferences_and_norms)]
            return []

        mock_memory.search = AsyncMock(side_effect=varying_search)
        briefing = await builder.build_briefing(
            project_id='test', include_task_tree=False
        )
        assert '## Project Overview' in briefing
        assert '## Recent Decisions' in briefing
        assert '## Active Conventions' in briefing


class TestTaskSection:
    @pytest.mark.asyncio
    async def test_task_details_included(self, builder, mock_memory, mock_taskmaster):
        """Task section shows title, description, status."""
        mock_taskmaster.get_task = AsyncMock(return_value={
            'task': {
                'id': '7',
                'title': 'Implement context builder',
                'description': 'Build the briefing module',
                'status': 'in-progress',
            }
        })
        briefing = await builder.build_briefing(
            project_id='test',
            task_id='7',
            project_root='/test/root',
            include_task_tree=False,
        )
        assert '## Current Task: 7' in briefing
        assert 'Implement context builder' in briefing
        assert 'in-progress' in briefing


class TestMemoryHints:
    @pytest.mark.asyncio
    async def test_hint_queries_executed(self, builder, mock_memory):
        """Memory hint queries are executed via search."""
        mock_memory.search = AsyncMock(return_value=[
            _make_result('Reconciliation uses three stages'),
        ])
        hints = {
            'queries': ['reconciliation pipeline'],
            'entities': [],
        }
        briefing = await builder.build_briefing(
            project_id='test',
            memory_hints=hints,
            include_task_tree=False,
        )
        # Check that search was called with the hint query
        search_calls = mock_memory.search.call_args_list
        hint_call = [c for c in search_calls if c.kwargs.get('query') == 'reconciliation pipeline']
        assert len(hint_call) > 0

    @pytest.mark.asyncio
    async def test_hint_entities_executed(self, builder, mock_memory):
        """Memory hint entities are looked up via get_entity."""
        mock_memory.get_entity = AsyncMock(return_value={
            'nodes': [{'name': 'TaskInterceptor', 'summary': 'Middleware for task events'}],
            'edges': [{'fact': 'TaskInterceptor wraps TaskmasterBackend'}],
        })
        hints = {
            'queries': [],
            'entities': ['TaskInterceptor'],
        }
        briefing = await builder.build_briefing(
            project_id='test',
            memory_hints=hints,
            include_task_tree=False,
        )
        mock_memory.get_entity.assert_called_once_with(
            name='TaskInterceptor', project_id='test'
        )
        assert 'TaskInterceptor' in briefing

    @pytest.mark.asyncio
    async def test_task_with_embedded_hints(self, builder, mock_memory, mock_taskmaster):
        """Hints in task metadata are extracted and executed."""
        mock_taskmaster.get_task = AsyncMock(return_value={
            'task': {
                'id': '5',
                'title': 'Fix reconciliation',
                'status': 'pending',
                'metadata': {
                    'memory_hints': {
                        'queries': ['reconciliation architecture'],
                        'entities': ['EventBuffer'],
                    }
                },
            }
        })
        mock_memory.search = AsyncMock(return_value=[
            _make_result('Three-stage pipeline'),
        ])
        mock_memory.get_entity = AsyncMock(return_value={
            'nodes': [{'name': 'EventBuffer', 'summary': 'Buffers reconciliation events'}],
            'edges': [],
        })
        briefing = await builder.build_briefing(
            project_id='test',
            task_id='5',
            project_root='/test/root',
            include_task_tree=False,
        )
        assert 'Fix reconciliation' in briefing
        assert 'Memory Hints' in briefing


class TestTaskTree:
    @pytest.mark.asyncio
    async def test_task_tree_rendered(self, builder, mock_memory, mock_taskmaster):
        """Task tree section renders tasks and subtasks."""
        mock_taskmaster.get_tasks = AsyncMock(return_value={
            'tasks': [
                {
                    'id': '1',
                    'title': 'Setup infrastructure',
                    'status': 'done',
                    'subtasks': [
                        {'id': '1.1', 'title': 'Docker config', 'status': 'done'},
                        {'id': '1.2', 'title': 'CI pipeline', 'status': 'pending'},
                    ],
                },
                {
                    'id': '2',
                    'title': 'Implement features',
                    'status': 'in-progress',
                    'subtasks': [],
                },
            ]
        })
        briefing = await builder.build_briefing(
            project_id='test',
            project_root='/test/root',
        )
        assert '## Task Tree' in briefing
        assert '[done] 1: Setup infrastructure' in briefing
        assert '[done] 1.1: Docker config' in briefing
        assert '[pending] 1.2: CI pipeline' in briefing
        assert '[in-progress] 2: Implement features' in briefing

    @pytest.mark.asyncio
    async def test_task_tree_excluded(self, builder, mock_memory, mock_taskmaster):
        """Task tree can be excluded."""
        briefing = await builder.build_briefing(
            project_id='test',
            project_root='/test/root',
            include_task_tree=False,
        )
        assert '## Task Tree' not in briefing


class TestErrorHandling:
    @pytest.mark.asyncio
    async def test_search_failure_skips_section(self, builder, mock_memory):
        """Search errors are caught and the section is skipped."""
        mock_memory.search = AsyncMock(side_effect=Exception('Connection refused'))
        briefing = await builder.build_briefing(
            project_id='test', include_task_tree=False
        )
        assert 'No context available' in briefing

    @pytest.mark.asyncio
    async def test_task_fetch_failure_skips_section(self, builder, mock_memory, mock_taskmaster):
        """Task fetch errors are caught and the section is skipped."""
        mock_taskmaster.get_task = AsyncMock(side_effect=Exception('Not found'))
        briefing = await builder.build_briefing(
            project_id='test',
            task_id='999',
            project_root='/test/root',
            include_task_tree=False,
        )
        assert '## Current Task' not in briefing
