"""End-to-end MCP tests for the topic-anchored canonical pin (task 3111).

Both consumer seams, exercised through ``create_mcp_server`` against a REAL
``MemoryService`` (only its Mem0 backend mocked), because the properties under
test are exactly the ones a service-level test cannot see:

1. SEARCH SEAM / COMPOSITION WITH TASK 3129. ``group_search_results`` runs at
   the MCP boundary, AFTER ``MemoryService.search``. This is the composition
   proof that could not be written before 3129 landed: grouping is append-only
   (no sort, no truncation), so a record pinned at index 0 by the service must
   still be at index 0 when the agent reads it — with the ONE exception that
   ``_suppress_child`` folds a child-shaped hit into its parent, which is why
   the pin refuses to select a child-shaped payload in the first place.

2. GUARD SEAM. The near-duplicate write guard's pre-check calls
   ``MemoryService.search`` DIRECTLY (``server/tools.py``), bypassing grouping,
   so every ``procedural_knowledge`` write now runs the anchoring block. A
   dissimilar write must still go through: the canonical joining the candidate
   set must not become a spurious block.
"""

from __future__ import annotations

import types
from unittest.mock import AsyncMock, MagicMock

import pytest
from _fm_helpers import install_identity_mocks

from fused_memory.server.tools import create_mcp_server
from fused_memory.services.memory_service import MemoryService

_PROJECT_ID = 'dark_factory'
_TOPIC = 'harness-layout-gate-decision-rule'
_CANONICAL_ID = 'canonical-0001'
_CANONICAL_BODY = (
    'The consolidated canonical for the harness layout gate decision rule, '
    'carrying every claim the narrow siblings each carry only one of.'
)


@pytest.fixture
def service(mock_config):
    """A REAL MemoryService with only its backends mocked.

    Real on purpose: the point is to run the production search tail and the
    production MCP boundary, not a re-implementation of either.
    """
    svc = MemoryService(mock_config)
    svc.graphiti = MagicMock()
    svc.graphiti.search = AsyncMock(return_value=[])
    svc.graphiti.search_nodes = AsyncMock(return_value=[])
    install_identity_mocks(svc.graphiti)

    svc.mem0 = MagicMock()
    svc.mem0.search = AsyncMock(return_value={'results': []})
    svc.mem0.scroll_by_metadata = AsyncMock(return_value=[])
    svc.mem0.add = AsyncMock(return_value={'results': [{'id': 'mem0-new'}]})
    svc.mem0.count_by_metadata = AsyncMock(return_value=0)
    return svc


def _sibling(n: int, *, topic: str = _TOPIC) -> dict:
    """A narrow sibling as Mem0Backend.search returns it; cosine below 0.92."""
    return {
        'id': f'sibling-{n}',
        'memory': f'narrow sibling {n} on the harness layout gate',
        'score': round(0.85 - 0.01 * n, 4),
        'metadata': {'category': 'procedural_knowledge', 'topic': topic},
    }


def _canonical_payload(**extra: object) -> dict:
    """The raw scroll payload; body under the raw 'data' key, no store_score."""
    metadata: dict = {
        'canonical': True,
        'topic': _TOPIC,
        'category': 'procedural_knowledge',
        'data': _CANONICAL_BODY,
    }
    metadata.update(extra)
    return {
        'id': _CANONICAL_ID,
        'created_at': '2026-08-09T00:00:00+00:00',
        'metadata': metadata,
    }


def _seed(service, *, scroll: list[dict] | None = None, siblings: int = 9) -> None:
    service.mem0.search = AsyncMock(
        return_value={'results': [_sibling(n) for n in range(1, siblings + 1)]}
    )
    service.mem0.scroll_by_metadata = AsyncMock(
        return_value=[_canonical_payload()] if scroll is None else scroll
    )


class TestSearchSeamComposesWithGroupedRead:
    """The pin survives the MCP boundary's grouped read at index 0."""

    @pytest.mark.asyncio
    async def test_pinned_canonical_is_first_in_the_mcp_response(self, service):
        _seed(service)
        server = create_mcp_server(service)

        response = await server._tool_manager.call_tool('search', {
            'query': 'which harness layout does the gate decision rule accept',
            'project_id': _PROJECT_ID,
            'categories': ['procedural_knowledge'],
            'stores': ['mem0'],
            'limit': 5,
        })

        first = response['results'][0]
        assert first['id'] == _CANONICAL_ID
        assert first['topic_anchored'] is True
        assert first['content'] == _CANONICAL_BODY

    @pytest.mark.asyncio
    async def test_topic_anchored_is_surfaced_on_every_result(self, service):
        """The key is unconditional, so an agent can read it without a guard."""
        _seed(service)
        server = create_mcp_server(service)

        response = await server._tool_manager.call_tool('search', {
            'query': 'harness layout gate',
            'project_id': _PROJECT_ID,
            'categories': ['procedural_knowledge'],
            'stores': ['mem0'],
            'limit': 5,
        })

        assert all('topic_anchored' in r for r in response['results'])
        assert [r['topic_anchored'] for r in response['results']] == [
            True, False, False, False, False,
        ]

    @pytest.mark.asyncio
    async def test_child_shaped_canonical_is_never_pinned(self, service):
        """THE ADVERSARIAL CASE that makes the composition proof meaningful.

        A malformed record carrying canonical=True AND child-shaped metadata
        (kind='amendment' + parent_id) must be rejected by
        select_canonical_payload, because grouped_read._suppress_child would
        otherwise fold it into its parent and silently seat a DIFFERENT record
        at slot 0 than the pin selected.
        """
        _seed(service, scroll=[
            _canonical_payload(kind='amendment', parent_id='parent-uuid-0001')
        ])
        server = create_mcp_server(service)

        response = await server._tool_manager.call_tool('search', {
            'query': 'harness layout gate',
            'project_id': _PROJECT_ID,
            'categories': ['procedural_knowledge'],
            'stores': ['mem0'],
            'limit': 5,
        })

        ids = [r['id'] for r in response['results']]
        assert _CANONICAL_ID not in ids
        assert 'parent-uuid-0001' not in ids
        assert not any(r['topic_anchored'] for r in response['results'])


class TestGuardSeamStillAdmitsDissimilarWrites:
    """A canonical joining the candidate set must not become a spurious block."""

    @pytest.mark.asyncio
    async def test_dissimilar_procedural_knowledge_write_is_not_blocked(self, service):
        """The pin reaches the guard's limit=5 pre-check and the write goes through.

        The pre-check calls MemoryService.search DIRECTLY, so grouping is
        bypassed and the anchoring block runs on every procedural_knowledge
        write. Every sibling here is below the 0.92 threshold and the injected
        anchor carries no store_score at all, so nothing qualifies.
        """
        _seed(service)
        service.config.reconciliation = types.SimpleNamespace(
            procedural_knowledge_near_dup_guard_enabled=True,
            procedural_knowledge_near_dup_threshold=0.92,
            procedural_knowledge_topic_guard_clusters=[],
            topic_anchored_recall_enabled=True,
        )
        added = MagicMock()
        added.model_dump.return_value = {'memory_ids': ['mem0-new']}
        service.add_memory = AsyncMock(return_value=added)
        server = create_mcp_server(service)

        result = await server._tool_manager.call_tool('add_memory', {
            'content': 'An entirely unrelated note about Qdrant collection prefixes.',
            'category': 'procedural_knowledge',
            'agent_id': 'claude-interactive',
            'project_id': _PROJECT_ID,
        })

        assert 'error' not in result
        assert result == {'memory_ids': ['mem0-new']}
        service.add_memory.assert_awaited_once()

    @pytest.mark.asyncio
    async def test_pin_reaches_the_guards_candidate_set(self, service):
        """Corroborates the mechanism: the pre-check search really is anchored."""
        _seed(service)
        service.config.reconciliation = types.SimpleNamespace(
            procedural_knowledge_near_dup_guard_enabled=True,
            procedural_knowledge_near_dup_threshold=0.92,
            procedural_knowledge_topic_guard_clusters=[],
            topic_anchored_recall_enabled=True,
        )
        added = MagicMock()
        added.model_dump.return_value = {'memory_ids': ['mem0-new']}
        service.add_memory = AsyncMock(return_value=added)
        server = create_mcp_server(service)

        await server._tool_manager.call_tool('add_memory', {
            'content': 'An entirely unrelated note about Qdrant collection prefixes.',
            'category': 'procedural_knowledge',
            'agent_id': 'claude-interactive',
            'project_id': _PROJECT_ID,
        })

        # The pre-check's own search performed the anchor lookup.
        assert service.mem0.scroll_by_metadata.await_count >= 1
