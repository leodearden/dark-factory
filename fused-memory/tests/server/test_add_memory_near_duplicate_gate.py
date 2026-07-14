"""Integration tests for the write-time near-duplicate guard in add_memory (task 2467).

Mirrors test_add_memory_snapshot_gate.py's harness: an AsyncMock memory
service wired through create_mcp_server, invoked via
server._tool_manager.call_tool('add_memory', {...}).
"""

from __future__ import annotations

from unittest.mock import AsyncMock

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
) -> MemoryResult:
    return MemoryResult(
        id=id_,
        content=content,
        category=MemoryCategory.procedural_knowledge,
        source_store=SourceStore.mem0,
        relevance_score=score,
    )


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
