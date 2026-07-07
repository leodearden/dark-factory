"""Integration tests for temporal guards — planned episode filtering pipeline."""

from __future__ import annotations

from unittest.mock import AsyncMock, MagicMock

import pytest
import pytest_asyncio

from fused_memory.models.scope import Scope
from fused_memory.server.tools import create_mcp_server
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
    from _fm_helpers import install_identity_mocks

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
    install_identity_mocks(svc.graphiti)

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
        from _fm_helpers import MockEdge
        svc, reg = service_with_real_registry
        ep_uuid = 'integration-ep-plan-002'
        project_id = 'integ-project'
        scope = Scope(project_id=project_id)

        # Register under the same canonical group_id _search_graphiti will look
        # up (mirrors memory_service.add_episode(), which derives group_id from
        # this same Scope for both the registry write and the search read).
        await reg.register(ep_uuid, scope.graphiti_group_id)

        # Configure graphiti.search to return an edge with this episode in provenance
        planned_edge = MockEdge(
            fact='CostStore extends AgentResult',
            uuid='edge-uuid-plan-1',
            episodes=[ep_uuid],
        )
        svc.graphiti.search = AsyncMock(return_value=[planned_edge])

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
        from _fm_helpers import MockEdge
        svc, reg = service_with_real_registry
        ep_uuid = 'integration-ep-plan-003'
        project_id = 'integ-project'
        scope = Scope(project_id=project_id)

        await reg.register(ep_uuid, scope.graphiti_group_id)

        planned_edge = MockEdge(
            fact='CostStore extends AgentResult',
            uuid='edge-uuid-plan-2',
            episodes=[ep_uuid],
        )
        svc.graphiti.search = AsyncMock(return_value=[planned_edge])

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
        from _fm_helpers import MockEdge
        svc, reg = service_with_real_registry
        ep_uuid = 'integration-ep-plan-004'
        project_id = 'integ-project'
        scope = Scope(project_id=project_id)

        # Register as planned
        await reg.register(ep_uuid, scope.graphiti_group_id)

        planned_edge = MockEdge(
            fact='CostStore extends AgentResult',
            uuid='edge-uuid-plan-3',
            episodes=[ep_uuid],
        )
        svc.graphiti.search = AsyncMock(return_value=[planned_edge])

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
        from _fm_helpers import MockEdge
        svc, reg = service_with_real_registry
        ep_uuid = 'integration-ep-full-001'
        project_id = 'integ-project'
        scope = Scope(project_id=project_id)

        # Step 1: planning write via _execute_graphiti_write. group_id is set
        # from scope.graphiti_group_id (not the raw project_id), mirroring
        # memory_service.add_episode()'s real payload construction so the
        # write and the later search below key off the same canonical value.
        payload = {
            'uuid': ep_uuid,
            'name': 'prd-episode',
            'content': 'PRD: TaskStore manages task lifecycle',
            'source': 'text',
            'group_id': scope.graphiti_group_id,
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


class TestBatchPlanAutoTagRoundTrip:
    """End-to-end regression (task 2022): the MCP add_episode tool auto-tags a
    batch-plan-shaped episode as planning, so its Graphiti-extracted completion
    edges are excluded from default search until a related task is promoted.

    Incident shape: episode 128442e1 ('Merge-queue modularization and
    invariant-enforcement batch were queued together as df 1985-2002') was
    ingested without temporal_context='planning', so its ~20 extracted
    completion edges polluted default factual search even though only 1 of
    tasks 1985-2002 was done.
    """

    @pytest.mark.asyncio
    async def test_batch_plan_episode_registered_and_excluded_until_promoted(
        self, service_with_real_registry
    ):
        """add_episode tool call with batch-plan content and no temporal_context:
        registers the episode as planned, excludes its edge from default search,
        and surfaces it with include_planned=True.
        """
        from _fm_helpers import MockEdge

        svc, reg = service_with_real_registry
        project_id = 'integ-batch-plan-001'

        # Inline the durable queue: capture the enqueued payload and run it
        # through _execute_graphiti_write synchronously so registration runs
        # within this test (mirrors the real dual_write_episode callback path).
        async def _inline_enqueue(**kwargs):
            if kwargs.get('operation') == 'add_episode':
                await svc._execute_graphiti_write('add_episode', kwargs['payload'])
            return 1

        svc.durable_queue.enqueue = AsyncMock(side_effect=_inline_enqueue)

        mcp_server = create_mcp_server(svc)

        result = await mcp_server._tool_manager.call_tool(
            'add_episode',
            {
                'content': (
                    'Merge-queue modularization and invariant-enforcement batch '
                    'were queued together as df 1985-2002'
                ),
                'project_id': project_id,
            },
        )
        episode_id = result['episode_id']

        # (a) auto-tag → registered as planned
        assert await reg.is_planned(episode_id) is True, (
            f'Batch-plan episode {episode_id!r} should be auto-tagged planning '
            f'and registered in the planned-episode registry'
        )

        # Simulate a premature completion edge extracted from this episode.
        svc.graphiti.search = AsyncMock(return_value=[
            MockEdge(
                fact='Merge-queue modularization was extracted',
                uuid='edge-batch-1',
                episodes=[episode_id],
            )
        ])
        scope = Scope(project_id=project_id)

        # Excluded from default search (this is the user-observable signal
        # this task fixes: planned-but-undone batch deliverables stay out of
        # factual search).
        default_results = await svc._search_graphiti(
            'Merge-queue modularization', scope, limit=10, include_planned=False
        )
        assert len(default_results) == 0, (
            'Batch-plan edge should be excluded from default search before '
            'the related task is done'
        )

        # Still recoverable via include_planned=True.
        planned_results = await svc._search_graphiti(
            'Merge-queue modularization', scope, limit=10, include_planned=True
        )
        assert len(planned_results) == 1, (
            'Batch-plan edge should be visible with include_planned=True'
        )
        assert planned_results[0].metadata.get('planned') is True, (
            'Batch-plan edge surfaced via include_planned=True should carry '
            "metadata['planned'] = True"
        )
