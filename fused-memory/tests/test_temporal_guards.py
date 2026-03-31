"""Integration tests for temporal guards — planned episode filtering pipeline."""

from __future__ import annotations

from unittest.mock import AsyncMock, MagicMock

import pytest
import pytest_asyncio

from fused_memory.models.scope import Scope
from fused_memory.services.memory_service import MemoryService
from fused_memory.services.planned_episode_registry import PlannedEpisodeRegistry


@pytest_asyncio.fixture
async def registry(tmp_path):
    """PlannedEpisodeRegistry backed by a real SQLite DB."""
    reg = PlannedEpisodeRegistry(data_dir=tmp_path / 'registry')
    await reg.initialize()
    yield reg
    await reg.close()


@pytest_asyncio.fixture
async def service_with_real_registry(mock_config, tmp_path):
    """MemoryService with real PlannedEpisodeRegistry, mocked graphiti/mem0 backends.

    The durable queue is also mocked so we can call internal methods directly
    without spinning up queue workers.
    """
    svc = MemoryService(mock_config)

    # Wire real registry (bypasses initialize() to avoid durable queue creation)
    reg = PlannedEpisodeRegistry(data_dir=tmp_path / 'reg')
    await reg.initialize()
    svc.planned_episode_registry = reg

    # Mock Graphiti backend — add_episode returns None (no edges to dedup)
    svc.graphiti = MagicMock()
    svc.graphiti.add_episode = AsyncMock(return_value=None)
    svc.graphiti.search = AsyncMock(return_value=[])
    svc.graphiti.bulk_remove_edges = AsyncMock(return_value=0)

    # Mock Mem0 backend
    svc.mem0 = MagicMock()
    svc.mem0.search = AsyncMock(return_value={'results': []})
    svc.mem0.add = AsyncMock(return_value={'results': [{'id': 'mem0-1'}]})

    # Mock durable queue (not invoked in these direct-method tests)
    svc.durable_queue = MagicMock()
    svc.durable_queue.enqueue = AsyncMock(return_value=1)
    svc.durable_queue.enqueue_batch = AsyncMock(return_value=[1])

    yield svc, reg

    await reg.close()


class TestTemporalGuardRoundTrip:
    """Integration: planning episode registry + search filter + promotion end-to-end."""

    @pytest.mark.asyncio
    async def test_execute_graphiti_write_registers_planning_episode(
        self, service_with_real_registry
    ):
        """_execute_graphiti_write with temporal_context='planning' registers the episode UUID."""
        svc, reg = service_with_real_registry
        ep_uuid = 'integration-ep-planning-001'
        project_id = 'integ-project'

        payload = {
            'uuid': ep_uuid,
            'name': 'test-episode',
            'content': 'CostStore extends AgentResult for cost tracking',
            'source': 'text',
            'group_id': project_id,
            'source_description': '[temporal:planning] PRD content',
            'temporal_context': 'planning',
        }
        await svc._execute_graphiti_write('add_episode', payload)

        assert await reg.is_planned(ep_uuid), (
            f'Episode {ep_uuid!r} should be in planned registry after planning write'
        )

    @pytest.mark.asyncio
    async def test_execute_graphiti_write_does_not_register_current_episode(
        self, service_with_real_registry
    ):
        """_execute_graphiti_write without temporal_context does NOT register the episode."""
        svc, reg = service_with_real_registry
        ep_uuid = 'integration-ep-current-001'
        project_id = 'integ-project'

        payload = {
            'uuid': ep_uuid,
            'name': 'test-episode',
            'content': 'CostStore was implemented in cost_store.py',
            'source': 'text',
            'group_id': project_id,
            'source_description': 'observed fact',
            # No temporal_context key
        }
        await svc._execute_graphiti_write('add_episode', payload)

        assert not await reg.is_planned(ep_uuid), (
            f'Episode {ep_uuid!r} should NOT be in planned registry for non-planning write'
        )

    @pytest.mark.asyncio
    async def test_search_excludes_planning_episode_edges(
        self, service_with_real_registry
    ):
        """After registering a planning episode, _search_graphiti excludes its edges."""
        from tests.conftest import MockEdge
        svc, reg = service_with_real_registry
        ep_uuid = 'integration-ep-plan-002'
        project_id = 'integ-project'

        # Register episode as planned
        await reg.register(ep_uuid, project_id)

        # Configure graphiti.search to return an edge with this episode in provenance
        planned_edge = MockEdge(
            fact='CostStore extends AgentResult',
            uuid='edge-uuid-plan-1',
            episodes=[ep_uuid],
        )
        svc.graphiti.search = AsyncMock(return_value=[planned_edge])

        scope = Scope(project_id=project_id)
        results = await svc._search_graphiti(
            'CostStore', scope, limit=10, include_planned=False
        )

        assert len(results) == 0, (
            'Edge whose entire provenance is from a planned episode should be excluded'
        )

    @pytest.mark.asyncio
    async def test_search_includes_planning_edges_when_flag_set(
        self, service_with_real_registry
    ):
        """With include_planned=True, planning edges are included and marked."""
        from tests.conftest import MockEdge
        svc, reg = service_with_real_registry
        ep_uuid = 'integration-ep-plan-003'
        project_id = 'integ-project'

        await reg.register(ep_uuid, project_id)

        planned_edge = MockEdge(
            fact='CostStore extends AgentResult',
            uuid='edge-uuid-plan-2',
            episodes=[ep_uuid],
        )
        svc.graphiti.search = AsyncMock(return_value=[planned_edge])

        scope = Scope(project_id=project_id)
        results = await svc._search_graphiti(
            'CostStore', scope, limit=10, include_planned=True
        )

        assert len(results) == 1, 'Planned edge should appear when include_planned=True'
        assert results[0].metadata.get('planned') is True, (
            'Planned edge should have metadata["planned"] = True'
        )

    @pytest.mark.asyncio
    async def test_promotion_makes_edge_visible_in_normal_search(
        self, service_with_real_registry
    ):
        """After promoting a planning episode, its edges appear in normal search."""
        from tests.conftest import MockEdge
        svc, reg = service_with_real_registry
        ep_uuid = 'integration-ep-plan-004'
        project_id = 'integ-project'

        # Register as planned
        await reg.register(ep_uuid, project_id)

        planned_edge = MockEdge(
            fact='CostStore extends AgentResult',
            uuid='edge-uuid-plan-3',
            episodes=[ep_uuid],
        )
        svc.graphiti.search = AsyncMock(return_value=[planned_edge])

        scope = Scope(project_id=project_id)

        # Before promotion: excluded
        results = await svc._search_graphiti(
            'CostStore', scope, limit=10, include_planned=False
        )
        assert len(results) == 0, 'Edge should be excluded before promotion'

        # Promote the episode
        await reg.promote(ep_uuid)
        assert not await reg.is_planned(ep_uuid), 'Episode should not be planned after promotion'

        # After promotion: visible in normal search
        results = await svc._search_graphiti(
            'CostStore', scope, limit=10, include_planned=False
        )
        assert len(results) == 1, 'Edge should be visible after promotion'
        assert not results[0].metadata.get('planned'), (
            'Promoted edge should not have planned metadata'
        )

    @pytest.mark.asyncio
    async def test_full_round_trip_write_to_visibility(
        self, service_with_real_registry
    ):
        """Complete pipeline: planning write → registered → excluded → promoted → visible."""
        from tests.conftest import MockEdge
        svc, reg = service_with_real_registry
        ep_uuid = 'integration-ep-full-001'
        project_id = 'integ-project'

        # Step 1: planning write via _execute_graphiti_write
        payload = {
            'uuid': ep_uuid,
            'name': 'prd-episode',
            'content': 'PRD: TaskStore manages task lifecycle',
            'source': 'text',
            'group_id': project_id,
            'source_description': '[temporal:planning] PRD',
            'temporal_context': 'planning',
        }
        await svc._execute_graphiti_write('add_episode', payload)

        # Verify registered
        assert await reg.is_planned(ep_uuid)

        # Step 2: configure search to return an edge from this episode
        edge = MockEdge(
            fact='TaskStore manages task lifecycle',
            uuid='edge-full-1',
            episodes=[ep_uuid],
        )
        svc.graphiti.search = AsyncMock(return_value=[edge])
        scope = Scope(project_id=project_id)

        # Step 3: normal search excludes it
        results = await svc._search_graphiti('TaskStore', scope, limit=10, include_planned=False)
        assert len(results) == 0, 'Planning edge must be excluded before task done'

        # Step 4: include_planned search finds it
        results = await svc._search_graphiti('TaskStore', scope, limit=10, include_planned=True)
        assert len(results) == 1
        assert results[0].metadata.get('planned') is True

        # Step 5: promote (task marked done)
        await reg.promote(ep_uuid)

        # Step 6: normal search now includes it
        results = await svc._search_graphiti('TaskStore', scope, limit=10, include_planned=False)
        assert len(results) == 1, 'Edge must be visible in normal search after promotion'
        assert not results[0].metadata.get('planned')
