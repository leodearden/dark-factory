"""Tests for the search_tasks MCP tool, TaskInterceptor.search_tasks, and
TaskCurator.search_corpus."""

from __future__ import annotations

import subprocess
from collections.abc import AsyncIterator
from contextlib import asynccontextmanager
from dataclasses import dataclass
from pathlib import Path
from typing import Any
from unittest.mock import AsyncMock, MagicMock, patch

import pytest
from _curator_helpers import (
    ExhaustibleEmbedder,
    exhaustible_embedder,
    in_memory_corpus_curator,
    make_config,
)
from mcp.server.fastmcp import FastMCP
from shared.whole_tree_guard_family import WHOLE_TREE_SCAN_TEST_TIMEOUT

from fused_memory.config.schema import CuratorConfig, FusedMemoryConfig
from fused_memory.middleware.curator_types import CandidateTask
from fused_memory.middleware.task_curator import CorpusSearchError, TaskCurator
from fused_memory.middleware.task_interceptor import TaskInterceptor
from fused_memory.models.scope import resolve_project_id
from fused_memory.reconciliation.event_buffer import EventBuffer
from fused_memory.server.tools import create_mcp_server

pytestmark = pytest.mark.timeout(WHOLE_TREE_SCAN_TEST_TIMEOUT)


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
    async def test_an_exhausted_embedder_raises_corpus_search_error(self):
        """A dead embedder makes the corpus unsearchable, which is not "no similar task"."""
        embedder = exhaustible_embedder()
        curator = in_memory_corpus_curator(embedder)
        try:
            await curator.record_task('42', CandidateTask(title='Fix the parser'), 'p')
            hits = await curator.search_corpus('Fix the parser', 'p')
            assert [h['task_id'] for h in hits] == ['42']

            embedder.exhaust()
            with pytest.raises(CorpusSearchError) as raised:
                await curator.search_corpus('Fix the parser', 'p')
        finally:
            await curator.close()

        err = raised.value
        assert err.error_type == 'RateLimitError'
        assert err.billing_exhausted == 'api_key_credits'
        assert 'api_key_credits' in str(err)

    @pytest.mark.asyncio
    async def test_an_unreachable_qdrant_raises_corpus_search_error(self):
        config = _make_config()
        config.mem0.qdrant_url = 'http://127.0.0.1:9'
        curator = TaskCurator(
            config=config, taskmaster=None,
            embedder=exhaustible_embedder(), qdrant_client=None,
        )
        try:
            with pytest.raises(CorpusSearchError) as raised:
                await curator.search_corpus('anything', 'p')
        finally:
            await curator.close()

        err = raised.value
        assert err.billing_exhausted is None
        assert isinstance(err.error_type, str)
        assert err.error_type not in ('', 'CorpusSearchError')

    @pytest.mark.asyncio
    async def test_nothing_filed_is_a_genuine_empty_result_even_with_a_dead_embedder(self):
        embedder = exhaustible_embedder()
        embedder.exhaust()
        curator = in_memory_corpus_curator(embedder)
        try:
            assert await curator.search_corpus('anything', 'p') == []
        finally:
            await curator.close()


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


# ──────────────────────────────────────────────────────────────────────────────
# MCP tool over a real interceptor and curator: a failed lookup is not "nothing filed"
# ──────────────────────────────────────────────────────────────────────────────


_FILED_TITLE = 'link-heal sweep caps its adjudications per run'


@dataclass
class SearchStack:
    server: FastMCP
    curator: TaskCurator
    root: str

    async def search(self, query: str, **arguments: Any) -> dict[str, Any]:
        return await self.server._tool_manager.call_tool(
            'search_tasks', {'project_root': self.root, 'query': query, **arguments},
        )

    async def file(self, task_id: str, title: str) -> None:
        await self.curator.record_task(
            task_id, CandidateTask(title=title), resolve_project_id(self.root),
        )


@asynccontextmanager
async def _search_stack(
    tmp_path: Path, embedder: ExhaustibleEmbedder,
) -> AsyncIterator[SearchStack]:
    """The search_tasks tool over a real TaskInterceptor whose curator has an in-memory corpus."""
    project_dir = tmp_path / 'project'
    project_dir.mkdir()
    subprocess.run(['git', 'init', '-q', str(project_dir)], check=True, capture_output=True)
    buffer = EventBuffer(db_path=tmp_path / 'eb.db', buffer_size_threshold=100)
    await buffer.initialize()
    taskmaster = AsyncMock()
    taskmaster.get_statuses_raw = AsyncMock(return_value={'1': 'pending'})
    curator = in_memory_corpus_curator(embedder)
    interceptor = TaskInterceptor(
        taskmaster, AsyncMock(), buffer, config=make_config(), curator=curator,
    )
    try:
        yield SearchStack(
            create_mcp_server(_make_mock_service(), interceptor),
            curator,
            str(project_dir.resolve()),
        )
    finally:
        await interceptor.close()
        await curator.close()
        await buffer.close()


@pytest.mark.asyncio
async def test_an_exhausted_embedder_answers_degraded_not_empty(tmp_path):
    embedder = exhaustible_embedder()
    async with _search_stack(tmp_path, embedder) as stack:
        await stack.file('1', _FILED_TITLE)
        healthy = await stack.search(_FILED_TITLE, score_threshold=0.1)
        embedder.exhaust()
        degraded = await stack.search(_FILED_TITLE, score_threshold=0.1)

    assert [hit['task_id'] for hit in healthy['results']] == ['1']
    assert healthy['results'][0]['status'] == 'pending'
    assert 'degraded' not in healthy
    assert 'error' not in healthy

    assert degraded['results'] == []
    assert degraded['degraded'] is True
    assert degraded['error_type'] == 'RateLimitError'
    assert degraded['billing_exhausted'] == 'api_key_credits'
    assert isinstance(degraded['error'], str)
    assert degraded['error']


@pytest.mark.asyncio
async def test_nothing_filed_stays_a_genuine_empty_answer(tmp_path):
    embedder = exhaustible_embedder()
    embedder.exhaust()
    async with _search_stack(tmp_path, embedder) as stack:
        reply = await stack.search(_FILED_TITLE)

    assert reply == {'results': []}
