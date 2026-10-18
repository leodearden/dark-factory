"""Tests for the get_entity_by_uuid diagnostic — service method + MCP tool.

Task 2086: read-only topological UUID→node lookup, exposing the existing
GraphitiBackend.get_node_text via a new service method and MCP tool. Unlike
get_entity (name-based, fuzzy, semantic edge-gather), this is a direct
UUID readback: {uuid, name, summary} with no edge gather.
"""

from __future__ import annotations

import json
from unittest.mock import AsyncMock, MagicMock

import pytest
from _id_resolver_fakes import FakeGraphiti, FakeMem0, make_resolver

from fused_memory.backends.graphiti_client import NodeNotFoundError
from fused_memory.services.memory_service import MemoryService


def _parse_tool_result(result):
    """Extract the dict from a FastMCP TextContent result or pass-through dict."""
    if isinstance(result, list):
        content = result[0].text if hasattr(result[0], 'text') else str(result[0])
        return json.loads(content)
    return result


@pytest.fixture
def service(mock_config):
    """MemoryService with a mocked graphiti backend (no real DB needed)."""
    svc = MemoryService(mock_config)
    svc.graphiti = MagicMock()
    svc.graphiti.get_node_text = AsyncMock(return_value=('NodeName', 'a summary'))
    return svc


class TestGetEntityByUuidService:
    """MemoryService.get_entity_by_uuid — thin wrapper over graphiti.get_node_text."""

    @pytest.mark.asyncio
    async def test_hit_returns_uuid_name_summary(self, service):
        result = await service.get_entity_by_uuid('node-uuid-1', project_id='test')
        assert result == {'uuid': 'node-uuid-1', 'name': 'NodeName', 'summary': 'a summary'}
        service.graphiti.get_node_text.assert_awaited_once_with('node-uuid-1', group_id='test')

    @pytest.mark.asyncio
    async def test_empty_summary_is_passed_through(self, service):
        service.graphiti.get_node_text = AsyncMock(return_value=('N', ''))
        result = await service.get_entity_by_uuid('node-uuid-2', project_id='test')
        assert result['summary'] == ''

    @pytest.mark.asyncio
    async def test_miss_raises_node_not_found_error(self, service):
        service.graphiti.get_node_text = AsyncMock(side_effect=NodeNotFoundError('missing'))
        with pytest.raises(NodeNotFoundError):
            await service.get_entity_by_uuid('node-uuid-missing', project_id='test')

    @pytest.mark.asyncio
    async def test_group_id_matches_project_id_for_non_default_project(self, service):
        """group_id passed to the backend must equal the caller's project_id.

        Documents the scoping guarantee: a lookup scoped to one project_id
        must query that project's group_id, not a default/fallback one —
        otherwise this diagnostic could silently leak nodes across projects.
        """
        service.graphiti.get_node_text = AsyncMock(return_value=('Other', 'other summary'))
        result = await service.get_entity_by_uuid('node-uuid-3', project_id='other_project')
        service.graphiti.get_node_text.assert_awaited_once_with(
            'node-uuid-3', group_id='other_project'
        )
        assert result == {'uuid': 'node-uuid-3', 'name': 'Other', 'summary': 'other summary'}


class TestGetEntityByUuidTool:
    """MCP tool get_entity_by_uuid — registration, delegation, and error handling."""

    @pytest.fixture
    def mock_service(self):
        svc = MagicMock()
        svc.get_entity_by_uuid = AsyncMock(
            return_value={'uuid': 'node-1', 'name': 'Alice', 'summary': 's'}
        )
        return svc

    @pytest.fixture
    def mcp_server(self, mock_service):
        from fused_memory.server.tools import create_mcp_server
        return create_mcp_server(mock_service)

    @pytest.mark.asyncio
    async def test_tool_is_registered(self, mcp_server):
        """get_entity_by_uuid is registered as an MCP tool."""
        tool_names = [t.name for t in await mcp_server.list_tools()]
        assert 'get_entity_by_uuid' in tool_names

    @pytest.mark.asyncio
    async def test_delegates_to_memory_service(self, mcp_server, mock_service):
        """Tool calls memory_service.get_entity_by_uuid with entity_uuid and project_id."""
        await mcp_server._tool_manager.call_tool(
            'get_entity_by_uuid',
            {'entity_uuid': 'node-1', 'project_id': 'dark_factory'},
        )
        mock_service.get_entity_by_uuid.assert_awaited_once()
        call_kwargs = mock_service.get_entity_by_uuid.call_args[1]
        assert call_kwargs.get('entity_uuid') == 'node-1'
        assert call_kwargs.get('project_id') == 'dark_factory'

    @pytest.mark.asyncio
    async def test_success_passthrough(self, mcp_server, mock_service):
        """Successful service result is returned to the caller unchanged."""
        result = await mcp_server._tool_manager.call_tool(
            'get_entity_by_uuid',
            {'entity_uuid': 'node-1', 'project_id': 'dark_factory'},
        )
        parsed = _parse_tool_result(result)
        assert parsed == {'uuid': 'node-1', 'name': 'Alice', 'summary': 's'}

    @pytest.mark.asyncio
    async def test_empty_project_id_returns_error(self, mcp_server, mock_service):
        """Empty project_id returns a validation error dict without calling the service."""
        result = await mcp_server._tool_manager.call_tool(
            'get_entity_by_uuid',
            {'entity_uuid': 'node-1', 'project_id': ''},
        )
        parsed = _parse_tool_result(result)
        assert 'error' in parsed
        assert parsed.get('error_type') == 'ValidationError'
        mock_service.get_entity_by_uuid.assert_not_awaited()

    @pytest.mark.asyncio
    async def test_miss_returns_error_dict_not_raised(self, mcp_server, mock_service):
        """NodeNotFoundError from the service surfaces as a structured error dict."""
        mock_service.get_entity_by_uuid = AsyncMock(
            side_effect=NodeNotFoundError('Entity node not found: node-x')
        )
        result = await mcp_server._tool_manager.call_tool(
            'get_entity_by_uuid',
            {'entity_uuid': 'node-x', 'project_id': 'dark_factory'},
        )
        parsed = _parse_tool_result(result)
        assert 'error' in parsed
        assert parsed.get('error_type') == 'NodeNotFoundError'


_NODE = 'bff81530-70e5-4ab6-939e-507299d1bcd2'
_SIBLING = 'bff81530-0000-4000-8000-000000000002'


class TestGetEntityByUuidPrefixArm:
    """A uuid PREFIX given as entity_uuid (PRD C4, task 5323).

    A prefix naming exactly one Entity node is read through to it; any other
    prefix gets the same miss report / refusal vocabulary get_memory_by_id
    uses. Full uuids and non-prefix ids keep the existing NodeNotFoundError
    shape — that asymmetry is deliberate and pinned here.
    """

    @staticmethod
    def _service(*, miss: bool = False) -> MagicMock:
        svc = MagicMock()
        if miss:
            svc.get_entity_by_uuid = AsyncMock(side_effect=NodeNotFoundError('Entity node not found'))
        else:
            svc.get_entity_by_uuid = AsyncMock(
                side_effect=lambda entity_uuid, project_id: {
                    'uuid': entity_uuid, 'name': 'Resolver', 'summary': 's',
                },
            )
        return svc

    @staticmethod
    async def _call(svc, resolver, entity_uuid: str) -> dict:
        from fused_memory.server.tools import create_mcp_server

        server = create_mcp_server(svc, id_resolver=resolver)
        return _parse_tool_result(await server._tool_manager.call_tool(
            'get_entity_by_uuid', {'entity_uuid': entity_uuid, 'project_id': 'reify'},
        ))

    @pytest.mark.asyncio
    async def test_a_prefix_naming_one_node_reads_it_and_says_it_was_resolved(self):
        svc = self._service()
        resolver = make_resolver(graphiti=FakeGraphiti(nodes={_NODE: 'Resolver'}))

        result = await self._call(svc, resolver, 'bff81530')

        assert result == {'uuid': _NODE, 'name': 'Resolver', 'summary': 's', 'resolved_from_prefix': 'bff81530'}
        svc.get_entity_by_uuid.assert_awaited_once_with(entity_uuid=_NODE, project_id='reify')

    @pytest.mark.asyncio
    async def test_a_node_deleted_between_resolution_and_read_still_says_it_was_resolved(self):
        svc = self._service(miss=True)
        resolver = make_resolver(graphiti=FakeGraphiti(nodes={_NODE: 'Resolver'}))

        result = await self._call(svc, resolver, 'bff81530')

        assert result == {
            'error': 'Entity node not found',
            'error_type': 'NodeNotFoundError',
            'entity_uuid': _NODE,
            'resolved_from_prefix': 'bff81530',
        }

    @pytest.mark.asyncio
    async def test_a_prefix_with_no_live_match_is_a_miss_report(self):
        svc = self._service()

        result = await self._call(svc, make_resolver(), 'bff81530')

        assert result['hint']
        assert {k: v for k, v in result.items() if k != 'hint'} == {
            'found': False,
            'prefix_checked': True,
            'prefix': 'bff81530',
            'project_id': 'reify',
            'namespaces': {'mem0': [], 'mem0_tombstone': [], 'graphiti_node': [], 'graphiti_edge': []},
        }
        svc.get_entity_by_uuid.assert_not_awaited()

    @pytest.mark.asyncio
    async def test_a_prefix_naming_a_live_memory_reports_it_rather_than_claiming_no_id(self):
        svc = self._service()
        resolver = make_resolver(mem0=FakeMem0({_NODE: {'data': 'a memory'}}))

        result = await self._call(svc, resolver, 'bff81530')

        assert result['found'] is False
        assert result['prefix_checked'] is True
        assert result['namespaces']['mem0'] == [{'id': _NODE, 'preview': 'a memory'}]
        svc.get_entity_by_uuid.assert_not_awaited()

    @pytest.mark.asyncio
    async def test_an_ambiguous_prefix_is_refused_with_its_candidates(self):
        svc = self._service()
        resolver = make_resolver(graphiti=FakeGraphiti(nodes={_NODE: 'a'}, edges={_SIBLING: 'b'}))

        result = await self._call(svc, resolver, 'bff81530')

        assert result['error_type'] == 'ambiguous_uuid_prefix'
        assert result['candidates'] == [
            {'namespace': 'graphiti_node', 'id': _NODE, 'preview': 'a'},
            {'namespace': 'graphiti_edge', 'id': _SIBLING, 'preview': 'b'},
        ]
        assert 'found' not in result
        svc.get_entity_by_uuid.assert_not_awaited()

    @pytest.mark.asyncio
    async def test_an_unreadable_store_is_unavailable(self):
        svc = self._service()
        resolver = make_resolver(mem0=FakeMem0(scroll_error=ConnectionError('qdrant down')))

        result = await self._call(svc, resolver, 'bff81530')

        assert result['error_type'] == 'resolver_unavailable'
        assert result['store'] == 'mem0'
        assert result['prefix_checked'] is False
        assert 'found' not in result

    @pytest.mark.asyncio
    async def test_a_full_uuid_miss_keeps_the_node_not_found_shape_without_resolving(self):
        svc = self._service(miss=True)
        mem0 = FakeMem0({_NODE: {'data': 'a memory'}})

        result = await self._call(svc, make_resolver(mem0=mem0), _NODE)

        assert result == {'error': 'Entity node not found', 'error_type': 'NodeNotFoundError'}
        assert mem0.scroll_calls == []

    @pytest.mark.asyncio
    async def test_a_non_prefix_id_goes_straight_to_the_service(self):
        svc = self._service()
        mem0 = FakeMem0()

        await self._call(svc, make_resolver(mem0=mem0), 'node-x')

        svc.get_entity_by_uuid.assert_awaited_once_with(entity_uuid='node-x', project_id='reify')
        assert mem0.scroll_calls == []
