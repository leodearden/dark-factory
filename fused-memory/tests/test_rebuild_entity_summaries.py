"""Tests for the SummaryRebuilder maintenance utility."""
from __future__ import annotations

from unittest.mock import AsyncMock

import pytest


class TestSummaryRebuilderInspect:
    """SummaryRebuilder.inspect_summaries() queries entity summaries by UUID."""

    @pytest.fixture
    def mock_backend(self):
        backend = AsyncMock()
        backend.get_node_text = AsyncMock(return_value=('SomeEntity', 'summary text here'))
        return backend

    @pytest.mark.asyncio
    async def test_inspect_returns_uuid_name_summary_tuples(self, mock_backend):
        """inspect_summaries returns (uuid, name, summary) tuples for given UUIDs."""
        from fused_memory.maintenance.rebuild_entity_summaries import SummaryRebuilder
        rebuilder = SummaryRebuilder(backend=mock_backend)
        uuids = ['uuid-1', 'uuid-2']
        mock_backend.get_node_text.side_effect = [
            ('Entity A', 'Summary of A'),
            ('Entity B', 'Summary of B'),
        ]
        results = await rebuilder.inspect_summaries(uuids)
        assert len(results) == 2
        assert results[0] == ('uuid-1', 'Entity A', 'Summary of A')
        assert results[1] == ('uuid-2', 'Entity B', 'Summary of B')

    @pytest.mark.asyncio
    async def test_inspect_handles_node_not_found_gracefully(self, mock_backend):
        """NodeNotFoundError for a UUID produces a (uuid, None, None) entry rather than crashing."""
        from fused_memory.backends.graphiti_client import NodeNotFoundError
        from fused_memory.maintenance.rebuild_entity_summaries import SummaryRebuilder
        rebuilder = SummaryRebuilder(backend=mock_backend)
        mock_backend.get_node_text.side_effect = [
            ('Entity A', 'Summary A'),
            NodeNotFoundError('uuid-missing not found'),
        ]
        results = await rebuilder.inspect_summaries(['uuid-found', 'uuid-missing'])
        assert len(results) == 2
        assert results[0] == ('uuid-found', 'Entity A', 'Summary A')
        assert results[1] == ('uuid-missing', None, None)

    @pytest.mark.asyncio
    async def test_inspect_empty_uuids_returns_empty_list(self, mock_backend):
        """Empty UUID list returns empty result list."""
        from fused_memory.maintenance.rebuild_entity_summaries import SummaryRebuilder
        rebuilder = SummaryRebuilder(backend=mock_backend)
        results = await rebuilder.inspect_summaries([])
        assert results == []
        mock_backend.get_node_text.assert_not_called()


class TestSummaryRebuilderRebuild:
    """SummaryRebuilder.rebuild() calls build_communities on the GraphitiBackend."""

    @pytest.fixture
    def mock_backend(self):
        return AsyncMock()

    @pytest.mark.asyncio
    async def test_rebuild_calls_build_communities_with_project_id(self, mock_backend):
        """rebuild(project_id) calls backend.build_communities(group_ids=[project_id])."""
        from fused_memory.maintenance.rebuild_entity_summaries import SummaryRebuilder
        rebuilder = SummaryRebuilder(backend=mock_backend)
        await rebuilder.rebuild(project_id='my_project')
        mock_backend.build_communities.assert_called_once_with(group_ids=['my_project'])


class TestRunRebuildEntrypoint:
    """run_rebuild() initializes service, runs rebuild + inspect, returns results."""

    @pytest.mark.asyncio
    async def test_run_rebuild_returns_inspect_and_rebuild_results(self):
        """run_rebuild() returns a dict with 'inspected' and 'rebuilt' keys."""
        from unittest.mock import AsyncMock, MagicMock, patch

        mock_service = AsyncMock()
        mock_service.graphiti = AsyncMock()
        mock_service.graphiti.build_communities = AsyncMock()
        mock_service.graphiti.get_node_text = AsyncMock(
            return_value=('Test Entity', 'Some summary')
        )

        with patch(
            'fused_memory.maintenance.rebuild_entity_summaries.MemoryService',
            return_value=mock_service,
        ), patch(
            'fused_memory.maintenance.rebuild_entity_summaries.FusedMemoryConfig',
            return_value=MagicMock(),
        ):
            from fused_memory.maintenance.rebuild_entity_summaries import run_rebuild
            result = await run_rebuild(
                project_id='test_project',
                uuids=['uuid-1'],
                inspect_only=False,
            )

        assert 'inspected' in result
        assert 'rebuilt' in result

    @pytest.mark.asyncio
    async def test_run_rebuild_inspect_only_skips_build_communities(self):
        """When inspect_only=True, build_communities is NOT called."""
        from unittest.mock import AsyncMock, MagicMock, patch

        mock_service = AsyncMock()
        mock_service.graphiti = AsyncMock()
        mock_service.graphiti.build_communities = AsyncMock()
        mock_service.graphiti.get_node_text = AsyncMock(
            return_value=('Test Entity', 'Some summary')
        )

        with patch(
            'fused_memory.maintenance.rebuild_entity_summaries.MemoryService',
            return_value=mock_service,
        ), patch(
            'fused_memory.maintenance.rebuild_entity_summaries.FusedMemoryConfig',
            return_value=MagicMock(),
        ):
            from fused_memory.maintenance.rebuild_entity_summaries import run_rebuild
            result = await run_rebuild(
                project_id='test_project',
                uuids=['uuid-1'],
                inspect_only=True,
            )

        mock_service.graphiti.build_communities.assert_not_called()
        assert result['rebuilt'] is False
