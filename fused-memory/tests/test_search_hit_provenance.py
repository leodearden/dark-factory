"""MemoryService.search hit attribution: every hit names its writer, task and stored scope.

Driven through the public ``search`` seam; each store's double returns records shaped as the store really returns them.
"""

from __future__ import annotations

from unittest.mock import AsyncMock, MagicMock

import pytest
from _fm_helpers import install_identity_mocks

from fused_memory.services.memory_service import MemoryService

_PROJECT_ID = 'dark_factory'
_ATTRIBUTION_KEYS = ('agent_id', 'task_id', 'project_id')


@pytest.fixture
def service(mock_config):
    """A real MemoryService over mocked Graphiti and Mem0 backends."""
    svc = MemoryService(mock_config)

    svc.graphiti = MagicMock()
    svc.graphiti.search = AsyncMock(return_value=[])
    svc.graphiti.search_nodes = AsyncMock(return_value=[])
    svc.graphiti.get_node_names_by_uuids = AsyncMock(return_value={})
    install_identity_mocks(svc.graphiti)

    svc.mem0 = MagicMock()
    svc.mem0.search = AsyncMock(return_value={'results': []})
    svc.mem0.scroll_by_metadata = AsyncMock(return_value=[])
    return svc


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
        query='who wrote this', project_id=_PROJECT_ID, stores=['mem0'],
    )
    assert len(results) == 1
    return results[0]


class TestMem0HitAttribution:
    @pytest.mark.asyncio
    async def test_writer_task_and_scope_reach_the_hit(self, service):
        hit = await _search_mem0_hit(service, _mem0_item(
            created_at='2026-11-01T00:00:00+00:00',
            agent_id='claude-task-7',
            user_id='dark_factory',
            metadata={'category': 'observations_and_summaries', 'task_id': '3137'},
        ))

        expected = {
            'agent_id': 'claude-task-7',
            'task_id': '3137',
            'project_id': 'dark_factory',
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
