"""Integration tests for the write-time near-duplicate guard in add_memory (task 2467).

Mirrors test_add_memory_snapshot_gate.py's harness: an AsyncMock memory
service wired through create_mcp_server, invoked via
server._tool_manager.call_tool('add_memory', {...}).
"""

from __future__ import annotations

from unittest.mock import AsyncMock, MagicMock

import pytest

from fused_memory.models.enums import MemoryCategory, SourceStore
from fused_memory.models.memory import MemoryResult
from fused_memory.server.tools import create_mcp_server

_PROJECT_ID = 'dark_factory'
_CONTENT = 'canonical .task gitignore gotcha'


def _near_duplicate_result(
    id_: str = 'm1',
    score: float = 0.97,
    content: str = 'canonical .task gitignore gotcha (existing entry)',
    category: MemoryCategory = MemoryCategory.procedural_knowledge,
) -> MemoryResult:
    return MemoryResult(
        id=id_,
        content=content,
        category=category,
        source_store=SourceStore.mem0,
        relevance_score=score,
    )


def _configure_pass_through_add_memory(mock_service: AsyncMock) -> None:
    """Configure mock_service.add_memory to return a dict-dumpable result.

    Mirrors test_add_memory_snapshot_gate.py: an unspecced AsyncMock's default
    return value chains AsyncMock all the way down, so ``result.model_dump()``
    would itself be an unawaited coroutine unless the return value is an
    explicit MagicMock with model_dump configured.
    """
    mem_result = MagicMock()
    mem_result.model_dump.return_value = {'id': 'ok'}
    mock_service.add_memory.return_value = mem_result


class TestAddMemoryNearDuplicateGate:
    """Write-gate: high-similarity procedural_knowledge writes are soft-blocked."""

    @pytest.mark.asyncio
    async def test_rejects_near_duplicate_procedural_knowledge_write(self):
        """A high-similarity search match blocks the write with a structured dict.

        The gate must return the standard soft-block error dict and must NOT
        call memory_service.add_memory.
        """
        mock_service = AsyncMock()
        mock_service.search.return_value = [_near_duplicate_result(id_='m1', score=0.97)]
        server = create_mcp_server(mock_service)

        result = await server._tool_manager.call_tool(
            'add_memory',
            {
                'content': _CONTENT,
                'category': 'procedural_knowledge',
                'agent_id': 'claude-interactive',
                'project_id': _PROJECT_ID,
            },
        )

        assert isinstance(result, dict), f'Expected dict, got {type(result)}: {result!r}'
        assert result.get('error') == 'procedural_knowledge_near_duplicate_write_blocked', (
            f'Expected near-duplicate block, got: {result!r}'
        )
        assert result.get('error_type') == 'ProceduralKnowledgeNearDuplicateWriteRejected', (
            f'Expected error_type=ProceduralKnowledgeNearDuplicateWriteRejected, got: {result!r}'
        )
        assert result.get('agent_id') == 'claude-interactive', (
            f'Expected agent_id echoed, got: {result!r}'
        )
        assert result.get('content_excerpt') == _CONTENT[:200], (
            f'Expected content_excerpt=content[:200], got: {result!r}'
        )
        assert result.get('matched_memory_id') == 'm1', (
            f'Expected matched_memory_id=m1, got: {result!r}'
        )
        assert isinstance(result.get('similarity'), int | float), (
            f'Expected a numeric similarity, got: {result!r}'
        )
        assert result.get('hint'), f'Expected a non-empty hint, got: {result!r}'
        mock_service.add_memory.assert_not_called()

    @pytest.mark.asyncio
    async def test_allows_write_when_top_match_below_threshold(self):
        """A search match scoring well below the threshold does not block the write."""
        mock_service = AsyncMock()
        mock_service.search.return_value = [_near_duplicate_result(id_='m1', score=0.50)]
        _configure_pass_through_add_memory(mock_service)
        server = create_mcp_server(mock_service)

        result = await server._tool_manager.call_tool(
            'add_memory',
            {
                'content': _CONTENT,
                'category': 'procedural_knowledge',
                'agent_id': 'claude-interactive',
                'project_id': _PROJECT_ID,
            },
        )

        assert result.get('error') != 'procedural_knowledge_near_duplicate_write_blocked', (
            f'Gate must not fire below threshold; got: {result!r}'
        )
        mock_service.add_memory.assert_called_once()

    @pytest.mark.asyncio
    async def test_guard_does_not_run_for_non_procedural_category(self):
        """The guard is scoped to category='procedural_knowledge' only.

        Even a high-similarity match must not trigger a search (or a block)
        when the write's own category is something else.
        """
        mock_service = AsyncMock()
        mock_service.search.return_value = [_near_duplicate_result(id_='m1', score=0.97)]
        _configure_pass_through_add_memory(mock_service)
        server = create_mcp_server(mock_service)

        result = await server._tool_manager.call_tool(
            'add_memory',
            {
                'content': _CONTENT,
                'category': 'observations_and_summaries',
                'agent_id': 'claude-interactive',
                'project_id': _PROJECT_ID,
            },
        )

        assert result.get('error') != 'procedural_knowledge_near_duplicate_write_blocked', (
            f'Gate must not fire for a non-procedural_knowledge category; got: {result!r}'
        )
        mock_service.search.assert_not_called()
        mock_service.add_memory.assert_called_once()

    @pytest.mark.asyncio
    async def test_allow_near_duplicate_override_bypasses_guard(self):
        """metadata={'allow_near_duplicate': True} bypasses the guard entirely.

        Even a high-similarity match must not block the write when the caller
        has explicitly opted in to writing a near-duplicate.
        """
        mock_service = AsyncMock()
        mock_service.search.return_value = [_near_duplicate_result(id_='m1', score=0.97)]
        _configure_pass_through_add_memory(mock_service)
        server = create_mcp_server(mock_service)

        result = await server._tool_manager.call_tool(
            'add_memory',
            {
                'content': _CONTENT,
                'category': 'procedural_knowledge',
                'agent_id': 'claude-interactive',
                'project_id': _PROJECT_ID,
                'metadata': {'allow_near_duplicate': True},
            },
        )

        assert result.get('error') != 'procedural_knowledge_near_duplicate_write_blocked', (
            f'Override must bypass the guard; got: {result!r}'
        )
        mock_service.add_memory.assert_called_once()

    @pytest.mark.asyncio
    async def test_fails_open_when_search_raises(self):
        """A search error must never block the write — fail-open, not fail-closed."""
        mock_service = AsyncMock()
        mock_service.search.side_effect = TimeoutError('search backend unavailable')
        _configure_pass_through_add_memory(mock_service)
        server = create_mcp_server(mock_service)

        result = await server._tool_manager.call_tool(
            'add_memory',
            {
                'content': _CONTENT,
                'category': 'procedural_knowledge',
                'agent_id': 'claude-interactive',
                'project_id': _PROJECT_ID,
            },
        )

        assert result.get('error') != 'procedural_knowledge_near_duplicate_write_blocked', (
            f'A search failure must not block the write; got: {result!r}'
        )
        mock_service.add_memory.assert_called_once()

    @pytest.mark.asyncio
    async def test_allows_write_when_search_returns_empty(self):
        """No candidate results at all → no match → write proceeds."""
        mock_service = AsyncMock()
        mock_service.search.return_value = []
        _configure_pass_through_add_memory(mock_service)
        server = create_mcp_server(mock_service)

        result = await server._tool_manager.call_tool(
            'add_memory',
            {
                'content': _CONTENT,
                'category': 'procedural_knowledge',
                'agent_id': 'claude-interactive',
                'project_id': _PROJECT_ID,
            },
        )

        assert result.get('error') != 'procedural_knowledge_near_duplicate_write_blocked', (
            f'Empty search results must not block the write; got: {result!r}'
        )
        mock_service.add_memory.assert_called_once()
