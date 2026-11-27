"""MemoryService.search hit attribution: every hit names its writer, task and stored scope.

Driven through the public ``search`` seam; each store's double returns records shaped as the store really returns them.
Stored scopes are written through Scope's own forward mappings, so a hit's project_id is checked against the
scope the record was written under rather than against a literal that would survive a change to those mappings.
"""

from __future__ import annotations

from unittest.mock import AsyncMock

import pytest
from _fm_helpers import MockEdge, mock_search_backends

from fused_memory.models.enums import SourceStore
from fused_memory.models.scope import Scope
from fused_memory.services.memory_service import MemoryService

_WRITTEN_UNDER = Scope(project_id='dark_factory')
_ATTRIBUTION_KEYS = ('agent_id', 'task_id', 'project_id')


@pytest.fixture
def service(mock_config):
    """A real MemoryService over mocked Graphiti and Mem0 backends."""
    return mock_search_backends(MemoryService(mock_config))


def _mem0_item(**fields: object) -> dict:
    """One record as Mem0Backend.search returns it: managed keys lifted to the top level."""
    item: dict = {
        'id': 'm1',
        'memory': 'x',
        'score': 0.9,
        'metadata': {'category': 'observations_and_summaries'},
    }
    item.update(fields)
    return item


async def _search_mem0_hit(service, item: dict):
    service.mem0.search = AsyncMock(return_value={'results': [item]})
    results = await service.search(
        query='who wrote this', project_id=_WRITTEN_UNDER.project_id, stores=['mem0'],
    )
    assert len(results) == 1
    return results[0]


class TestMem0HitAttribution:
    @pytest.mark.asyncio
    async def test_writer_task_and_scope_reach_the_hit(self, service):
        hit = await _search_mem0_hit(service, _mem0_item(
            created_at='2026-11-01T00:00:00+00:00',
            agent_id='claude-task-7',
            user_id=_WRITTEN_UNDER.mem0_user_id,
            metadata={'category': 'observations_and_summaries', 'task_id': '3137'},
        ))

        expected = {
            'agent_id': 'claude-task-7',
            'task_id': '3137',
            'project_id': _WRITTEN_UNDER.project_id,
            'created_at': '2026-11-01T00:00:00+00:00',
        }
        assert {key: getattr(hit, key) for key in expected} == expected
        dumped = hit.model_dump()
        assert {key: dumped[key] for key in expected} == expected
        assert hit.metadata['task_id'] == '3137'
        assert 'agent_id' not in hit.metadata
        assert 'user_id' not in hit.metadata

    @pytest.mark.asyncio
    async def test_unrecorded_attribution_is_none_but_present(self, service):
        hit = await _search_mem0_hit(service, _mem0_item())

        assert (hit.agent_id, hit.task_id, hit.project_id) == (None, None, None)
        dumped = hit.model_dump()
        assert {key: dumped[key] for key in _ATTRIBUTION_KEYS} == dict.fromkeys(_ATTRIBUTION_KEYS)

    @pytest.mark.asyncio
    @pytest.mark.parametrize(
        ('stored_agent_id', 'stored_user_id'),
        [
            pytest.param('', '', id='empty-str'),
            pytest.param(7, 123, id='int'),
            pytest.param(['claude-task-7'], ['dark_factory'], id='list'),
        ],
    )
    async def test_writer_and_scope_that_are_not_non_empty_strings_read_as_none(
        self, service, stored_agent_id, stored_user_id,
    ):
        hit = await _search_mem0_hit(service, _mem0_item(
            agent_id=stored_agent_id, user_id=stored_user_id,
        ))

        assert (hit.agent_id, hit.project_id) == (None, None)

    @pytest.mark.asyncio
    @pytest.mark.parametrize(
        ('stored_task_id', 'expected'),
        [
            pytest.param(3137, '3137', id='legacy-int'),
            pytest.param(True, None, id='bool'),
            pytest.param([5040, 5149], None, id='list'),
            pytest.param('', None, id='empty-str'),
        ],
    )
    async def test_task_id_projection(self, service, stored_task_id, expected):
        hit = await _search_mem0_hit(service, _mem0_item(
            metadata={'category': 'observations_and_summaries', 'task_id': stored_task_id},
        ))

        assert hit.task_id == expected


async def _search_graph_hit(service, edge: MockEdge):
    service.graphiti.search = AsyncMock(return_value=[edge])
    results = await service.search(
        query='Redis dependencies', project_id=_WRITTEN_UNDER.project_id, stores=['graphiti'],
    )
    assert len(results) == 1
    return results[0]


class TestGraphitiHitAttribution:
    @pytest.mark.asyncio
    async def test_graph_hit_reports_its_partition_and_no_writer(self, service):
        hit = await _search_graph_hit(service, MockEdge(
            fact='Auth service depends on Redis', uuid='edge-1',
            group_id=_WRITTEN_UNDER.graphiti_group_id,
        ))

        assert hit.source_store is SourceStore.graphiti
        assert hit.project_id == _WRITTEN_UNDER.project_id
        assert hit.agent_id is None
        assert hit.task_id is None

    @pytest.mark.asyncio
    async def test_unpartitioned_edge_does_not_borrow_the_queried_project(self, service):
        hit = await _search_graph_hit(service, MockEdge(
            fact='Auth service depends on Redis', uuid='edge-1',
        ))

        assert hit.project_id is None


_TOPIC = 'harness-layout-gate-decision-rule'
_CANONICAL_ID = 'canonical-0001'


def _topic_sibling(n: int) -> dict:
    """A narrow sibling in the canonical's topic, carrying no canonical flag."""
    return _mem0_item(
        id=f'sibling-{n}',
        memory=f'narrow sibling {n} on the harness layout gate',
        score=round(0.89 - 0.01 * n, 4),
        metadata={'category': 'procedural_knowledge', 'topic': _TOPIC},
    )


def _canonical_scroll_row() -> dict:
    """The raw scroll dict the pin reads: managed and custom keys mixed in one payload."""
    return {
        'id': _CANONICAL_ID,
        'created_at': '2026-08-09T00:00:00+00:00',
        'metadata': {
            'canonical': True,
            'topic': _TOPIC,
            'category': 'procedural_knowledge',
            'data': 'The consolidated canonical for the harness layout gate decision rule.',
            'agent_id': 'claude-task-4004',
            'user_id': _WRITTEN_UNDER.mem0_user_id,
            'task_id': '4004',
        },
    }


class TestTopicPinnedCanonicalAttribution:
    @pytest.mark.asyncio
    async def test_pinned_canonical_carries_writer_task_and_scope(self, service):
        service.mem0.search = AsyncMock(
            return_value={'results': [_topic_sibling(n) for n in range(1, 10)]}
        )
        service.mem0.scroll_by_metadata = AsyncMock(return_value=[_canonical_scroll_row()])

        results = await service.search(
            query='which harness layout does the gate decision rule accept',
            project_id=_WRITTEN_UNDER.project_id,
            categories=['procedural_knowledge'],
            stores=['mem0'],
            limit=5,
        )

        pinned = results[0]
        assert (pinned.id, pinned.topic_anchored) == (_CANONICAL_ID, True)
        assert (pinned.agent_id, pinned.task_id, pinned.project_id) == (
            'claude-task-4004', '4004', _WRITTEN_UNDER.project_id,
        )
        assert 'agent_id' not in pinned.metadata
        assert 'user_id' not in pinned.metadata
