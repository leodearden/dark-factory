"""Tests for the search_tasks MCP tool, TaskInterceptor.search_tasks, and
TaskCurator.search_corpus."""

from __future__ import annotations

from unittest.mock import AsyncMock, MagicMock, patch

import pytest

from fused_memory.config.schema import CuratorConfig, FusedMemoryConfig
from fused_memory.middleware.task_curator import TaskCurator
from fused_memory.server.tools import create_mcp_server


def _make_config() -> FusedMemoryConfig:
    cfg = FusedMemoryConfig()
    cfg.curator = CuratorConfig()
    return cfg


def _mock_point(payload: dict, score: float) -> MagicMock:
    point = MagicMock()
    point.payload = payload
    point.score = score
    return point


# ──────────────────────────────────────────────────────────────────────────────
# Curator layer: TaskCurator.search_corpus
# ──────────────────────────────────────────────────────────────────────────────


class TestSearchCorpus:
    @pytest.mark.asyncio
    async def test_search_corpus_returns_ranked_hits(self):
        """search_corpus embeds the query, calls query_points with limit +
        score_threshold, and maps payload + score into hit dicts."""
        curator = TaskCurator(config=_make_config(), taskmaster=None)
        project_id = 'myproject'

        results = MagicMock()
        results.points = [
            _mock_point(
                {
                    'task_id': '42',
                    'title': 'Fix the parser',
                    'description': 'Parser chokes on unicode',
                    'files_to_modify': ['src/parse.py'],
                    'priority': 'high',
                    'updated_at': '2026-05-26T00:00:00+00:00',
                },
                score=0.82,
            ),
            _mock_point({'task_id': '7', 'title': 'Other task'}, score=0.41),
        ]

        mock_client = AsyncMock()
        mock_client.collection_exists = AsyncMock(return_value=True)
        mock_client.query_points = AsyncMock(return_value=results)

        mock_embedder = AsyncMock()
        mock_embedder.create = AsyncMock(return_value=[0.1] * 1536)

        with patch.object(curator, '_get_qdrant', return_value=mock_client), \
             patch.object(curator, '_get_embedder', return_value=mock_embedder):
            hits = await curator.search_corpus(
                'parser bug', project_id, limit=5, score_threshold=0.3,
            )

        # Embedded the raw query string (not composed via _embedding_text).
        mock_embedder.create.assert_awaited_once_with('parser bug')

        # query_points forwarded limit + score_threshold.
        mock_client.query_points.assert_awaited_once()
        kwargs = mock_client.query_points.call_args.kwargs
        assert kwargs['collection_name'] == f'task_curator_{project_id}'
        assert kwargs['limit'] == 5
        assert kwargs['score_threshold'] == 0.3
        assert kwargs['with_payload'] is True

        # Hits carry task_id + score, in returned order.
        assert [h['task_id'] for h in hits] == ['42', '7']
        assert hits[0]['score'] == 0.82
        assert hits[0]['title'] == 'Fix the parser'
        assert hits[0]['files_to_modify'] == ['src/parse.py']

    @pytest.mark.asyncio
    async def test_search_corpus_missing_collection_returns_empty(self):
        """No collection (no tasks ever filed) → [], and query_points is never
        called (reads must not create the collection)."""
        curator = TaskCurator(config=_make_config(), taskmaster=None)

        mock_client = AsyncMock()
        mock_client.collection_exists = AsyncMock(return_value=False)
        mock_client.query_points = AsyncMock()

        mock_embedder = AsyncMock()
        mock_embedder.create = AsyncMock(return_value=[0.1] * 1536)

        with patch.object(curator, '_get_qdrant', return_value=mock_client), \
             patch.object(curator, '_get_embedder', return_value=mock_embedder):
            hits = await curator.search_corpus('anything', 'emptyproject')

        assert hits == []
        mock_client.query_points.assert_not_called()

    @pytest.mark.asyncio
    async def test_search_corpus_swallows_errors(self):
        """A raising query_points degrades to [] rather than propagating."""
        curator = TaskCurator(config=_make_config(), taskmaster=None)

        mock_client = AsyncMock()
        mock_client.collection_exists = AsyncMock(return_value=True)
        mock_client.query_points = AsyncMock(side_effect=RuntimeError('qdrant down'))

        mock_embedder = AsyncMock()
        mock_embedder.create = AsyncMock(return_value=[0.1] * 1536)

        with patch.object(curator, '_get_qdrant', return_value=mock_client), \
             patch.object(curator, '_get_embedder', return_value=mock_embedder):
            hits = await curator.search_corpus('boom', 'someproject')

        assert hits == []


# ──────────────────────────────────────────────────────────────────────────────
# MCP tool layer: search_tasks
# ──────────────────────────────────────────────────────────────────────────────


PROJECT_ROOT = '/home/leo/src/dark-factory'


def _make_mock_service() -> AsyncMock:
    mock_service = AsyncMock()
    mock_service.add_memory.return_value = MagicMock()
    return mock_service


@pytest.mark.asyncio
async def test_search_tasks_tool_returns_enriched_results():
    """The tool passes through the interceptor's status-enriched results."""
    mock_interceptor = AsyncMock()
    mock_interceptor.search_tasks = AsyncMock(
        return_value={'results': [{'task_id': '42', 'status': 'done', 'score': 0.9}]},
    )
    server = create_mcp_server(_make_mock_service(), mock_interceptor)

    result = await server._tool_manager.call_tool(
        'search_tasks',
        {'query': 'parser bug', 'project_root': PROJECT_ROOT},
    )

    assert result['results'][0]['status'] == 'done'
    assert result['results'][0]['score'] == 0.9
    mock_interceptor.search_tasks.assert_awaited_once()


@pytest.mark.asyncio
async def test_search_tasks_rejects_empty_query():
    """Blank query → ValidationError, interceptor not called."""
    mock_interceptor = AsyncMock()
    mock_interceptor.search_tasks = AsyncMock()
    server = create_mcp_server(_make_mock_service(), mock_interceptor)

    result = await server._tool_manager.call_tool(
        'search_tasks',
        {'query': '   ', 'project_root': PROJECT_ROOT},
    )

    assert result['error_type'] == 'ValidationError'
    mock_interceptor.search_tasks.assert_not_called()


@pytest.mark.asyncio
async def test_search_tasks_rejects_nonpositive_limit():
    """limit <= 0 → ValidationError, interceptor not called."""
    mock_interceptor = AsyncMock()
    mock_interceptor.search_tasks = AsyncMock()
    server = create_mcp_server(_make_mock_service(), mock_interceptor)

    result = await server._tool_manager.call_tool(
        'search_tasks',
        {'query': 'parser bug', 'project_root': PROJECT_ROOT, 'limit': 0},
    )

    assert result['error_type'] == 'ValidationError'
    mock_interceptor.search_tasks.assert_not_called()


@pytest.mark.asyncio
async def test_search_tasks_passes_through_curator_disabled():
    """ConfigurationError shape from the interceptor is passed through."""
    mock_interceptor = AsyncMock()
    mock_interceptor.search_tasks = AsyncMock(
        return_value={
            'error': 'Task curator is not enabled',
            'error_type': 'ConfigurationError',
            'results': [],
        },
    )
    server = create_mcp_server(_make_mock_service(), mock_interceptor)

    result = await server._tool_manager.call_tool(
        'search_tasks',
        {'query': 'parser bug', 'project_root': PROJECT_ROOT},
    )

    assert result['error_type'] == 'ConfigurationError'
    assert result['results'] == []
