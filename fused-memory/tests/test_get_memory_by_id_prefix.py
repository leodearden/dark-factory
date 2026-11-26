"""get_memory_by_id given a uuid PREFIX (PRD C4, B10/B11; task 5323).

A prefix that names exactly one live Mem0 memory is read through to that
memory and echoes ``resolved_from_prefix``. Any other prefix answers with a
miss report (``prefix_checked: True``) instead of leaking a Qdrant 400, an
ambiguous one is refused with its candidates, and an unchecked one says so
(``prefix_checked: False``) rather than claiming a miss (INV-11).
"""

from __future__ import annotations

import json
from unittest.mock import AsyncMock

import pytest
from _id_resolver_fakes import FakeGraphiti, FakeMem0, RecordingTombstones, make_resolver

from fused_memory.server.tools import create_mcp_server

_PROJECT = 'reify'
_FULL = 'bff81530-70e5-4ab6-939e-507299d1bcd2'
_SIBLING = 'bff81530-0000-4000-8000-000000000002'
_PAYLOAD = {'data': 'the raw memory content', 'category': 'observations_and_summaries'}


def _service(record: dict | None) -> AsyncMock:
    service = AsyncMock()
    service.get_memory_by_id = AsyncMock(return_value=record)
    service.get_mem0_deletion_tombstone = AsyncMock(return_value=None)
    service.count_memories_by_metadata = AsyncMock(return_value=0)
    return service


def _record() -> dict:
    return {'id': _FULL, 'content': 'the raw memory content', 'metadata': _PAYLOAD}


async def _call(service: AsyncMock, resolver, memory_id: str) -> dict:
    server = create_mcp_server(service, id_resolver=resolver)
    return await server._tool_manager.call_tool(
        'get_memory_by_id', {'project_id': _PROJECT, 'memory_id': memory_id},
    )


class TestUniquePrefix:
    @pytest.mark.asyncio
    async def test_reads_the_one_memory_it_names_and_says_it_was_resolved(self):
        service = _service(_record())
        resolver = make_resolver(mem0=FakeMem0({_FULL: _PAYLOAD}))

        result = await _call(service, resolver, 'bff81530')

        assert result['found'] is True
        assert result['memory_id'] == _FULL
        assert result['resolved_from_prefix'] == 'bff81530'
        assert result['content'] == 'the raw memory content'
        service.get_memory_by_id.assert_awaited_once_with(project_id=_PROJECT, memory_id=_FULL)

    @pytest.mark.asyncio
    async def test_a_memory_deleted_between_resolution_and_read_still_says_it_was_resolved(self):
        service = _service(None)
        resolver = make_resolver(mem0=FakeMem0({_FULL: _PAYLOAD}))

        result = await _call(service, resolver, 'bff81530')

        assert result == {
            'found': False,
            'memory_id': _FULL,
            'project_id': _PROJECT,
            'resolved_from_prefix': 'bff81530',
        }


class TestMissReport:
    @pytest.mark.asyncio
    async def test_no_live_match_reports_every_namespace_including_tombstones(self):
        tombstone = {'memory_id': '3f2a9c1e-0000-4000-8000-000000000001', 'deleter': 'stage1_cycle_summary_trim'}
        service = _service(_record())
        resolver = make_resolver(
            mem0=FakeMem0({_FULL: _PAYLOAD}), tombstones=RecordingTombstones([tombstone]),
        )

        result = await _call(service, resolver, '3f2a9c1e')

        hint = result.pop('hint')
        assert isinstance(hint, str) and hint
        assert result == {
            'found': False,
            'prefix_checked': True,
            'prefix': '3f2a9c1e',
            'project_id': _PROJECT,
            'namespaces': {
                'mem0': [],
                'mem0_tombstone': [tombstone],
                'graphiti_node': [],
                'graphiti_edge': [],
            },
        }
        rendered = json.dumps(result)
        assert 'UnexpectedResponse' not in rendered
        assert 'not a valid point ID' not in rendered
        service.get_memory_by_id.assert_not_awaited()

    @pytest.mark.asyncio
    async def test_a_prefix_naming_a_graph_node_reports_it_rather_than_reading_mem0(self):
        service = _service(_record())
        resolver = make_resolver(graphiti=FakeGraphiti(nodes={_FULL: 'resolver node'}))

        result = await _call(service, resolver, 'bff81530')

        assert result['found'] is False
        assert result['prefix_checked'] is True
        assert result['namespaces']['graphiti_node'] == [{'id': _FULL, 'preview': 'resolver node'}]
        service.get_memory_by_id.assert_not_awaited()


class TestRefusals:
    @pytest.mark.asyncio
    async def test_an_ambiguous_prefix_is_refused_with_its_candidates(self):
        service = _service(_record())
        resolver = make_resolver(mem0=FakeMem0({_FULL: {'data': 'a'}, _SIBLING: {'data': 'b'}}))

        result = await _call(service, resolver, 'bff81530')

        assert result['error_type'] == 'ambiguous_uuid_prefix'
        assert result['prefix'] == 'bff81530'
        assert result['prefix_checked'] is True
        assert result['candidates'] == [
            {'namespace': 'mem0', 'id': _SIBLING, 'preview': 'b'},
            {'namespace': 'mem0', 'id': _FULL, 'preview': 'a'},
        ]
        assert 'found' not in result
        service.get_memory_by_id.assert_not_awaited()

    @pytest.mark.asyncio
    async def test_an_unreadable_store_is_unavailable_never_a_miss(self):
        service = _service(_record())
        resolver = make_resolver(graphiti=FakeGraphiti(error=ConnectionError('falkordb down')))

        result = await _call(service, resolver, 'bff81530')

        assert result['error_type'] == 'resolver_unavailable'
        assert result['store'] == 'graphiti'
        assert result['prefix_checked'] is False
        assert 'found' not in result
        service.get_memory_by_id.assert_not_awaited()

    @pytest.mark.parametrize('malformed', ['xyz', 'bff8153', 'bff8153070e54ab6939e507299d1bcd2'])
    @pytest.mark.asyncio
    async def test_a_malformed_id_is_a_validation_error_before_any_store_read(self, malformed):
        service = _service(_record())
        mem0 = FakeMem0({_FULL: _PAYLOAD})

        result = await _call(service, make_resolver(mem0=mem0), malformed)

        assert result['error_type'] == 'ValidationError'
        assert result['hint']
        service.get_memory_by_id.assert_not_awaited()
        assert mem0.scroll_calls == []


class TestFullUuidPath:
    @pytest.mark.asyncio
    async def test_a_full_uuid_never_consults_the_resolver(self):
        service = _service(_record())
        mem0 = FakeMem0({_FULL: _PAYLOAD})

        result = await _call(service, make_resolver(mem0=mem0), _FULL)

        assert result == {
            'found': True,
            'memory_id': _FULL,
            'project_id': _PROJECT,
            'content': 'the raw memory content',
            'metadata': _PAYLOAD,
        }
        assert mem0.scroll_calls == []
