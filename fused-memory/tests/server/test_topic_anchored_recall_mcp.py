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
    """The write guard's candidate set is NOT anchored, and still admits writes.

    The pre-check calls MemoryService.search DIRECTLY, so it would otherwise
    run the anchoring block on every procedural_knowledge write. It opts out
    (``anchor_topics=False``) because it reads its 5 results as a candidate
    set rather than a presentation — see
    ``test_pin_never_reaches_the_guards_candidate_set`` for the full rationale.
    """

    @pytest.mark.asyncio
    async def test_dissimilar_procedural_knowledge_write_is_not_blocked(self, service):
        """An un-anchored pre-check still admits a genuinely dissimilar write.

        Every sibling here is below the 0.92 threshold, so nothing qualifies
        and the write goes through. This is the end-to-end proof that opting
        out of anchoring did not disturb the guard's admit path.
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
    async def test_pin_never_reaches_the_guards_candidate_set(self, service):
        """REGRESSION GUARD: the write guard's pre-check must NOT be anchored.

        This assertion is deliberately INVERTED from its original form, which
        asserted the pin *did* reach this candidate set and so encoded the
        defect rather than the contract. The guard reads its 5 results as a
        CANDIDATE SET and qualifies each on ``metadata['store_score'] >=
        threshold``; a pinned canonical carries no ``store_score`` at all, so
        every pin would spend one of those 5 slots on a record the guard can
        never qualify — while evicting a genuine cosine hit, because the pin
        promotes rather than adds. That shrinks an effective 5-candidate set
        toward 2 on exactly the consolidated topics this guard protects, and
        lets a true near-duplicate at rank 5 land as a duplicate.

        The guard therefore passes ``anchor_topics=False`` (server/tools.py).
        Asserting ZERO scroll round-trips also pins the cheap-gate ordering:
        the opt-out is checked before the harvest, so an opted-out caller pays
        no lookup at all.
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

        await server._tool_manager.call_tool('add_memory', {
            'content': 'An entirely unrelated note about Qdrant collection prefixes.',
            'category': 'procedural_knowledge',
            'agent_id': 'claude-interactive',
            'project_id': _PROJECT_ID,
        })

        # The pre-check's own search did NOT perform an anchor lookup.
        assert service.mem0.scroll_by_metadata.await_count == 0


_CHILD_ID = 'amendment-0001'
_CHILD_BODY = 'The amendment that narrows the gate decision rule to layout v3.'


class TestPinnedCanonicalWithChildrenInTheWindow:
    """THE INTERESTING HALF of the composition proof with task 3129.

    The class above covers only the NEGATIVE grouping interaction (a
    child-SHAPED canonical is refused). This covers the positive one: a
    canonical reached BOTH by the pin (index 0, relevance_score 0.0) AND by
    upward resolution from a lower-ranked child hit. That path lands in
    ``group_search_results``' ``existing is not None`` branch, which keeps the
    FIRST entry's position and takes ``max(...)`` of the two scores — so the
    pin's slot 0 must survive the dedup, the pinned entry must gain its
    ``grouped`` block, and the canonical must be emitted exactly ONCE.

    Nothing else asserts this: it is precisely where a pinned record could be
    silently duplicated, demoted to the child's rank, or emitted without the
    grouping the MCP boundary exists to add.
    """

    @staticmethod
    def _amendment_child_hit() -> dict:
        """A child of the canonical, as Mem0Backend.search returns it."""
        return {
            'id': _CHILD_ID,
            'memory': _CHILD_BODY,
            'score': 0.81,
            'metadata': {
                'category': 'procedural_knowledge',
                'topic': _TOPIC,
                'kind': 'amendment',
                'parent_id': _CANONICAL_ID,
            },
        }

    @classmethod
    def _wire_children(cls, service) -> None:
        """Dispatch the two backend reads by FILTER, not by call order.

        The anchor lookup and the amendment-digest scroll both land on
        ``scroll_by_metadata``, and grouping issues its counts concurrently, so
        a positional ``side_effect`` list would bind the wrong answer to the
        wrong question the moment either call order changed.
        """
        service.mem0.search = AsyncMock(return_value={'results': [
            _sibling(1),
            cls._amendment_child_hit(),
        ]})

        async def _scroll(_scope, filters, _limit):
            if filters.get('canonical') is True:
                return [_canonical_payload()]
            if filters.get('parent_id') == _CANONICAL_ID:
                return [{
                    'id': _CHILD_ID,
                    'created_at': '2026-08-11T00:00:00+00:00',
                    'metadata': {'kind': 'amendment', 'parent_id': _CANONICAL_ID,
                                 'data': _CHILD_BODY},
                }]
            return []

        async def _count(_scope, filters):
            if filters.get('parent_id') != _CANONICAL_ID:
                return 0
            return 0 if filters.get('kind') == 'sighting' else 1

        service.mem0.scroll_by_metadata = AsyncMock(side_effect=_scroll)
        service.mem0.count_by_metadata = AsyncMock(side_effect=_count)

    @pytest.mark.asyncio
    async def test_pin_keeps_slot_zero_and_absorbs_its_matched_child(self, service):
        self._wire_children(service)
        server = create_mcp_server(service)

        response = await server._tool_manager.call_tool('search', {
            'query': 'which harness layout does the gate decision rule accept',
            'project_id': _PROJECT_ID,
            'categories': ['procedural_knowledge'],
            'stores': ['mem0'],
            'limit': 5,
        })

        results = response['results']
        assert [r['id'] for r in results].count(_CANONICAL_ID) == 1, (
            'reached both by the pin and by upward resolution — still ONE entry'
        )
        first = results[0]
        assert first['id'] == _CANONICAL_ID
        assert first['topic_anchored'] is True, (
            "the dedup keeps the FIRST entry, so it must be the PIN's dump — "
            'an upward-resolved parent would carry topic_anchored False'
        )
        # The child folded into the pin rather than surviving as its own hit.
        assert _CHILD_ID not in [r['id'] for r in results]
        grouped = first['grouped']
        assert grouped['amendment_count'] == 1
        assert [a['id'] for a in grouped['amendments']] == [_CHILD_ID]
        # Pinned in full by _pin_matched_child: a swallowed child's text must
        # never be less reachable than it was before grouping ran.
        assert grouped['amendments'][0]['content'] == _CHILD_BODY

    @pytest.mark.asyncio
    async def test_dedup_keeps_the_higher_score_without_moving_the_pin(self, service):
        """``max(...)`` of the two scores, at the PIN's position.

        The pin is scored 0.0 by contract (order-only, never a synthetic
        score), so the surviving entry takes the child's real RRF score — while
        staying exactly where the pin put it.
        """
        self._wire_children(service)
        server = create_mcp_server(service)

        response = await server._tool_manager.call_tool('search', {
            'query': 'harness layout gate',
            'project_id': _PROJECT_ID,
            'categories': ['procedural_knowledge'],
            'stores': ['mem0'],
            'limit': 5,
        })

        first = response['results'][0]
        assert first['id'] == _CANONICAL_ID
        assert first['relevance_score'] > 0.0
        assert response['results'][1]['id'] == 'sibling-1'
