"""Behaviour tests for the search MCP tool degraded-channel surfacing (task 1812).

step-15 (RED) / step-16 (GREEN):
  When memory_service.search returns a SearchResults with degraded=True, the MCP
  handler must surface degraded/failed_stores on the response.  Happy-path response
  shape must be unchanged (no degraded keys when degraded is False).
"""

from __future__ import annotations

from unittest.mock import AsyncMock

import pytest

from fused_memory.models.enums import SourceStore
from fused_memory.models.memory import MemoryResult
from fused_memory.server.tools import create_mcp_server
from fused_memory.services.memory_service import SearchResults

_PROJECT_ID = 'dark_factory'


class TestSearchToolDegradedSurfacing:
    """search MCP tool must surface degraded/failed_stores only on fault path."""

    @pytest.mark.asyncio
    async def test_degraded_search_includes_degraded_keys(self):
        """When service.search returns degraded=True, response has degraded=True and failed_stores."""
        mock_service = AsyncMock()
        mock_service.search = AsyncMock(
            return_value=SearchResults([], degraded=True, failed_stores=['mem0'])
        )
        server = create_mcp_server(mock_service)

        result = await server._tool_manager.call_tool(
            'search',
            {'query': 'test query', 'project_id': _PROJECT_ID},
        )

        assert isinstance(result, dict), f'Expected dict, got {type(result)}: {result!r}'
        assert 'error' not in result, f'Unexpected error in result: {result!r}'

        assert result.get('degraded') is True, (
            f"Expected result['degraded'] is True when search is degraded, "
            f"got result={result!r}. "
            "RED: handler does not yet surface degraded key."
        )
        assert result.get('failed_stores') == ['mem0'], (
            f"Expected result['failed_stores'] == ['mem0'], got {result.get('failed_stores')!r}. "
            "RED: handler does not yet surface failed_stores key."
        )

    @pytest.mark.asyncio
    async def test_clean_search_omits_degraded_keys(self):
        """When service.search returns degraded=False, response has NO degraded/failed_stores keys.

        Fault-only loudness: happy-path MCP response shape unchanged.
        """
        mock_service = AsyncMock()
        mock_service.search = AsyncMock(
            return_value=SearchResults([], degraded=False, failed_stores=[])
        )
        server = create_mcp_server(mock_service)

        result = await server._tool_manager.call_tool(
            'search',
            {'query': 'test query', 'project_id': _PROJECT_ID},
        )

        assert isinstance(result, dict), f'Expected dict, got {type(result)}: {result!r}'
        assert 'error' not in result, f'Unexpected error in result: {result!r}'
        assert 'degraded' not in result, (
            f"'degraded' must NOT appear in happy-path response, got result={result!r}. "
            "Fault-only loudness: clean-path shape must be byte-for-byte unchanged."
        )
        assert 'failed_stores' not in result, (
            f"'failed_stores' must NOT appear in happy-path response, got result={result!r}."
        )

    @pytest.mark.asyncio
    async def test_degraded_search_includes_failed_store_diagnostics(self):
        """When service.search returns failure_diagnostics (task 2653), the tool
        must surface them as result['failed_store_diagnostics'] alongside the
        existing degraded/failed_stores keys — closing the attribution gap where
        a caller sees degraded+failed_stores with no root-cause info.
        """
        diagnostics = [{
            'store': 'graphiti',
            'reason': 'exception',
            'error_type': 'RuntimeError',
            'rate_limit_or_quota': False,
            'query_len': 5,
            'project_id': 'dark_factory',
            'error': 'boom',
        }]
        mock_service = AsyncMock()
        mock_service.search = AsyncMock(
            return_value=SearchResults(
                [],
                degraded=True,
                failed_stores=['graphiti'],
                failure_diagnostics=diagnostics,
            )
        )
        server = create_mcp_server(mock_service)

        result = await server._tool_manager.call_tool(
            'search',
            {'query': 'q', 'project_id': _PROJECT_ID},
        )

        assert result.get('degraded') is True
        assert result.get('failed_stores') == ['graphiti']
        assert result.get('failed_store_diagnostics') == diagnostics, (
            f"Expected result['failed_store_diagnostics'] == {diagnostics!r}, "
            f'got {result.get("failed_store_diagnostics")!r}. '
            'RED: the tool does not surface failed_store_diagnostics yet.'
        )

    @pytest.mark.asyncio
    async def test_clean_search_omits_failed_store_diagnostics_key(self):
        """Fault-only loudness: a clean (non-degraded) response must NOT contain
        the 'failed_store_diagnostics' key."""
        mock_service = AsyncMock()
        mock_service.search = AsyncMock(
            return_value=SearchResults([], degraded=False, failed_stores=[])
        )
        server = create_mcp_server(mock_service)

        result = await server._tool_manager.call_tool(
            'search',
            {'query': 'test query', 'project_id': _PROJECT_ID},
        )

        assert 'failed_store_diagnostics' not in result, (
            f"'failed_store_diagnostics' must NOT appear in happy-path response, "
            f'got result={result!r}.'
        )

    @pytest.mark.asyncio
    async def test_plain_list_back_compat_does_not_add_degraded_keys(self):
        """When service.search returns a plain list (back-compat), response has no degraded keys."""
        mock_service = AsyncMock()
        mock_service.search = AsyncMock(return_value=[])
        server = create_mcp_server(mock_service)

        result = await server._tool_manager.call_tool(
            'search',
            {'query': 'test query', 'project_id': _PROJECT_ID},
        )

        assert 'degraded' not in result, (
            f"'degraded' must NOT appear when search returns a plain list, got result={result!r}."
        )


# ---------------------------------------------------------------------------
# Task 3129 (leaf δ) — grouped reads at the MCP `search` boundary
#
# step-13 (RED) / step-14 (GREEN): child records fold into their parent's
# grouped document, and the fault-only degrade channel keeps working.
# ---------------------------------------------------------------------------

_CANONICAL_ID = '11111111-1111-4111-8111-111111111111'
_AMEND_IDS = (
    '22222222-2222-4222-8222-222222222221',
    '22222222-2222-4222-8222-222222222222',
)
_SIGHT_IDS = (
    '33333333-3333-4333-8333-333333333331',
    '33333333-3333-4333-8333-333333333332',
    '33333333-3333-4333-8333-333333333333',
)


def _hit(memory_id: str, content: str, score: float, metadata: dict) -> MemoryResult:
    return MemoryResult(
        id=memory_id,
        content=content,
        source_store=SourceStore.mem0,
        relevance_score=score,
        metadata=metadata,
    )


def _service_with_a_parented_canonical(**search_kwargs) -> AsyncMock:
    """A service whose search returns a canonical PLUS its 2+3 child records."""
    hits = [_hit(_CANONICAL_ID, 'the canonical claim', 0.9, {'kind': 'canonical'})]
    hits += [
        _hit(amend_id, 'a correction', 0.8, {'parent_id': _CANONICAL_ID, 'kind': 'amendment'})
        for amend_id in _AMEND_IDS
    ]
    hits += [
        _hit(sight_id, 'seen again', 0.7, {'parent_id': _CANONICAL_ID, 'kind': 'sighting'})
        for sight_id in _SIGHT_IDS
    ]
    rows = [
        {
            'id': amend_id,
            'created_at': f'2026-08-0{index + 1}T00:00:00+00:00',
            'metadata': {'data': 'a correction', 'parent_id': _CANONICAL_ID, 'kind': 'amendment'},
        }
        for index, amend_id in enumerate(_AMEND_IDS)
    ]

    def _count(*, project_id: str, filters: dict):
        if filters.get('parent_id') != _CANONICAL_ID:
            return 0
        return {'sighting': 3, 'amendment': 2}.get(filters.get('kind', ''), 5)

    def _scroll(*, project_id: str, filters: dict, limit: int = 1000):
        return rows[:limit]

    service = AsyncMock()
    service.search = AsyncMock(return_value=SearchResults(hits, **search_kwargs))
    service.count_memories_by_metadata = AsyncMock(side_effect=_count)
    service.get_memories_by_metadata = AsyncMock(side_effect=_scroll)
    return service


class TestSearchToolGroupedReads:
    """The leaf's SIGNAL at the MCP boundary: one grouped hit, no loose children."""

    @pytest.mark.asyncio
    async def test_children_fold_into_one_grouped_canonical_hit(self):
        """6 raw hits (1 canonical + 2 amendments + 3 sightings) → 1 grouped result."""
        mock_service = _service_with_a_parented_canonical()
        server = create_mcp_server(mock_service)

        result = await server._tool_manager.call_tool(
            'search',
            {'query': 'test query', 'project_id': _PROJECT_ID},
        )

        assert 'error' not in result, f'Unexpected error: {result!r}'
        results = result['results']
        assert [r['id'] for r in results] == [_CANONICAL_ID], (
            'Child records must not appear as separate top-level hits — the '
            f'esc-5541 shape. Got {[r["id"] for r in results]!r}. '
            'RED: the search tool does not group yet.'
        )
        grouped = results[0]['grouped']
        assert len(grouped['amendments']) == 2, f'Expected 2 digests, got {grouped!r}'
        assert grouped['sighting_count'] == 3, f'Expected sighting_count 3, got {grouped!r}'

    @pytest.mark.asyncio
    async def test_grouping_does_not_clobber_the_degraded_channel(self):
        """Fault-only degrade keys must survive the grouping transform (task 1812)."""
        mock_service = _service_with_a_parented_canonical(
            degraded=True,
            failed_stores=['graphiti'],
            failure_diagnostics=[{'store': 'graphiti', 'error_type': 'TimeoutError'}],
        )
        server = create_mcp_server(mock_service)

        result = await server._tool_manager.call_tool(
            'search',
            {'query': 'test query', 'project_id': _PROJECT_ID},
        )

        assert result.get('degraded') is True, (
            'SearchResults.degraded does NOT survive a list transform — the tool '
            f'must read it BEFORE grouping. Got {result!r}'
        )
        assert result.get('failed_stores') == ['graphiti']
        assert result.get('failed_store_diagnostics') == [
            {'store': 'graphiti', 'error_type': 'TimeoutError'}
        ]
        assert [r['id'] for r in result['results']] == [_CANONICAL_ID]

    @pytest.mark.asyncio
    async def test_childless_results_are_byte_identical_to_today(self):
        """Zero-child corpus (today's fleet): no grouped key anywhere."""
        mock_service = AsyncMock()
        plain = _hit('44444444-4444-4444-8444-444444444444', 'a plain memory', 0.5, {})
        mock_service.search = AsyncMock(return_value=SearchResults([plain]))
        mock_service.count_memories_by_metadata = AsyncMock(return_value=0)
        server = create_mcp_server(mock_service)

        result = await server._tool_manager.call_tool(
            'search',
            {'query': 'test query', 'project_id': _PROJECT_ID},
        )

        assert result['results'] == [plain.model_dump()], (
            f'A childless result must dump exactly as today, got {result["results"]!r}'
        )
        assert mock_service.get_memories_by_metadata.await_count == 0, (
            'The zero-child common path must issue no scroll'
        )

    @pytest.mark.asyncio
    async def test_grouping_failure_degrades_to_the_ungrouped_list(self):
        """A grouping fault must never turn a working search into an error dict."""
        mock_service = AsyncMock()
        plain = _hit('44444444-4444-4444-8444-444444444444', 'a plain memory', 0.5, {})
        mock_service.search = AsyncMock(return_value=SearchResults([plain]))
        mock_service.count_memories_by_metadata = AsyncMock(
            side_effect=TimeoutError('qdrant down')
        )
        server = create_mcp_server(mock_service)

        result = await server._tool_manager.call_tool(
            'search',
            {'query': 'test query', 'project_id': _PROJECT_ID},
        )

        assert 'error' not in result, (
            f'A grouping fault must not convert a working search into an error, got {result!r}'
        )
        assert [r['id'] for r in result['results']] == [plain.id]
