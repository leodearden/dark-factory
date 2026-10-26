"""The uuid-prefix resolver contract (PRD C2, task 5323).

Drives the REAL IdResolver over the in-memory stores in _id_resolver_fakes.
The rule every failure case below pins is INV-11: a prefix that could not be
checked is never ``none`` — the resolver raises ResolverUnavailable(store).
"""

from __future__ import annotations

import httpx
import pytest
from _id_resolver_fakes import FakeGraphiti, FakeMem0, RecordingTombstones, make_resolver
from qdrant_client.http.exceptions import UnexpectedResponse
from shared.uuid_prefix_guard import Candidate, Resolution, ResolverUnavailable

from fused_memory.backends.graphiti_client import IncompleteEnumerationError
from fused_memory.backends.mem0_client import ScrollPageBudgetExhausted
from fused_memory.services.id_resolver import IdResolver
from fused_memory.services.memory_service import MemoryService

_PROJECT = 'reify'
_FULL = 'bff81530-70e5-4ab6-939e-507299d1bcd2'
_SIBLING = 'bff81530-0000-4000-8000-000000000002'
_OTHER = 'cafe0000-0000-4000-8000-000000000001'
_LONG_TEXT = 'resolver preview ' * 10


def _memory(text: str) -> dict:
    return {'data': text, 'user_id': _PROJECT}


class TestBoundary:
    @pytest.mark.parametrize('bad', [
        '',
        'bff8153',
        'xyz12345',
        'BFF81530',
        'bff8153070e54ab6939e507299d1bcd2',
        '-bff81530',
    ])
    @pytest.mark.asyncio
    async def test_a_non_prefix_is_refused_before_any_store_is_touched(self, bad):
        mem0, graphiti, tombstones = FakeMem0({_FULL: _memory('x')}), FakeGraphiti(), RecordingTombstones()
        resolver = make_resolver(mem0=mem0, graphiti=graphiti, tombstones=tombstones)

        with pytest.raises(ValueError):
            await resolver.resolve_prefix(_PROJECT, bad)

        assert (mem0.scroll_calls, mem0.point_reads, graphiti.calls, tombstones.calls) == ([], [], [], [])

    @pytest.mark.parametrize('full', [_FULL, _FULL.upper()])
    @pytest.mark.asyncio
    async def test_a_full_uuid_resolves_as_an_exact_match(self, full):
        resolver = make_resolver(mem0=FakeMem0({_FULL: _memory('exact'), _SIBLING: _memory('s')}))

        resolution = await resolver.resolve_prefix(_PROJECT, full)

        assert resolution == Resolution('unique', (Candidate('mem0', _FULL, 'exact'),))


class TestMem0Walk:
    @pytest.mark.asyncio
    async def test_one_ids_only_walk_of_the_projects_collection(self):
        mem0 = FakeMem0({_FULL: _memory('x')})
        resolver = make_resolver(mem0=mem0, collection_prefix='fused')

        await resolver.resolve_prefix(_PROJECT, 'bff81530')

        assert len(mem0.scroll_calls) == 1
        collection, kwargs = mem0.scroll_calls[0]
        assert collection == 'fused_reify', 'the COLLECTION prefix, never the uuid prefix'
        assert kwargs['with_payload'] is False
        assert kwargs['page_size'] == 10_000

    @pytest.mark.asyncio
    async def test_unique_mem0_id_carries_an_80_char_preview_read_for_the_match_only(self):
        mem0 = FakeMem0({_OTHER: _memory('other'), _FULL: _memory(_LONG_TEXT)})
        resolver = make_resolver(mem0=mem0)

        resolution = await resolver.resolve_prefix(_PROJECT, 'bff81530')

        assert resolution == Resolution('unique', (Candidate('mem0', _FULL, _LONG_TEXT[:80]),))
        assert mem0.point_reads == [(_FULL,)]

    @pytest.mark.asyncio
    async def test_a_match_deleted_before_its_preview_read_keeps_its_candidate(self):
        mem0 = FakeMem0({_FULL: None})
        resolver = make_resolver(mem0=mem0)

        resolution = await resolver.resolve_prefix(_PROJECT, 'bff81530')

        assert resolution == Resolution('unique', (Candidate('mem0', _FULL, ''),))

    @pytest.mark.asyncio
    async def test_no_match_anywhere_is_none(self):
        resolver = make_resolver(mem0=FakeMem0({_OTHER: _memory('x')}))

        assert await resolver.resolve_prefix(_PROJECT, 'bff81530') == Resolution('none', ())

    @pytest.mark.asyncio
    async def test_two_mem0_ids_sharing_the_prefix_are_ambiguous(self):
        resolver = make_resolver(mem0=FakeMem0({_FULL: _memory('a'), _SIBLING: _memory('b')}))

        resolution = await resolver.resolve_prefix(_PROJECT, 'bff81530')

        assert resolution.outcome == 'ambiguous'
        assert {c.id for c in resolution.candidates} == {_FULL, _SIBLING}

    @pytest.mark.asyncio
    async def test_every_match_is_previewed_by_one_batched_read(self):
        mem0 = FakeMem0({_FULL: _memory('a'), _OTHER: _memory('o'), _SIBLING: _memory('b')})
        resolver = make_resolver(mem0=mem0)

        resolution = await resolver.resolve_prefix(_PROJECT, 'bff81530')

        assert [(c.id, c.preview) for c in resolution.candidates] == [(_SIBLING, 'b'), (_FULL, 'a')]
        assert mem0.point_reads == [(_FULL, _SIBLING)]

    @pytest.mark.asyncio
    async def test_no_match_reads_no_point(self):
        mem0 = FakeMem0({_OTHER: _memory('o')})

        await make_resolver(mem0=mem0).resolve_prefix(_PROJECT, 'bff81530')

        assert mem0.point_reads == []


class TestGraphitiNamespaces:
    @pytest.mark.asyncio
    async def test_a_mem0_id_and_a_graph_node_sharing_the_prefix_are_ambiguous(self):
        resolver = make_resolver(
            mem0=FakeMem0({_FULL: _memory('a')}),
            graphiti=FakeGraphiti(nodes={_SIBLING: 'node'}),
        )

        resolution = await resolver.resolve_prefix(_PROJECT, 'bff81530')

        assert resolution.outcome == 'ambiguous'
        assert {c.namespace for c in resolution.candidates} == {'mem0', 'graphiti_node'}

    @pytest.mark.asyncio
    async def test_unique_node_previews_its_name(self):
        resolver = make_resolver(graphiti=FakeGraphiti(nodes={_FULL: _LONG_TEXT}))

        resolution = await resolver.resolve_prefix(_PROJECT, 'bff81530')

        assert resolution == Resolution('unique', (Candidate('graphiti_node', _FULL, _LONG_TEXT[:80]),))

    @pytest.mark.asyncio
    async def test_unique_edge_previews_its_fact(self):
        resolver = make_resolver(graphiti=FakeGraphiti(edges={_FULL: _LONG_TEXT}))

        resolution = await resolver.resolve_prefix(_PROJECT, 'bff81530')

        assert resolution == Resolution('unique', (Candidate('graphiti_edge', _FULL, _LONG_TEXT[:80]),))

    @pytest.mark.asyncio
    async def test_candidates_are_ordered_by_namespace_then_id(self):
        resolver = make_resolver(
            mem0=FakeMem0({'bff81530-0000-4000-8000-000000000009': _memory('m')}),
            graphiti=FakeGraphiti(
                nodes={
                    'bff81530-0000-4000-8000-000000000003': 'n3',
                    'bff81530-0000-4000-8000-000000000002': 'n2',
                },
                edges={'bff81530-0000-4000-8000-000000000001': 'e1'},
            ),
        )

        resolution = await resolver.resolve_prefix(_PROJECT, 'bff81530')

        assert [(c.namespace, c.id[-1]) for c in resolution.candidates] == [
            ('mem0', '9'), ('graphiti_node', '2'), ('graphiti_node', '3'), ('graphiti_edge', '1'),
        ]

    @pytest.mark.asyncio
    async def test_both_graph_reads_run_on_the_projects_graph_every_time(self):
        graphiti = FakeGraphiti()
        resolver = make_resolver(mem0=FakeMem0({_FULL: _memory('a'), _SIBLING: _memory('b')}), graphiti=graphiti)

        await resolver.resolve_prefix(_PROJECT, 'bff81530')

        assert sorted(graphiti.calls) == [
            ('edge_uuids_with_prefix', 'bff81530', _PROJECT),
            ('entity_uuids_with_prefix', 'bff81530', _PROJECT),
        ]


def _missing_collection() -> UnexpectedResponse:
    return UnexpectedResponse(404, 'Not Found', b"Collection `fused_reify` doesn't exist!", httpx.Headers())


class TestUnavailableIsNeverAResolution:
    @pytest.mark.parametrize('error', [
        TimeoutError('page read timed out'),
        ConnectionError('qdrant down'),
        ScrollPageBudgetExhausted('budget'),
        UnexpectedResponse(500, 'Internal Server Error', b'boom', httpx.Headers()),
    ])
    @pytest.mark.asyncio
    async def test_a_failing_walk_is_mem0_unavailable(self, error):
        resolver = make_resolver(mem0=FakeMem0({_FULL: _memory('x')}, scroll_error=error))

        with pytest.raises(ResolverUnavailable) as excinfo:
            await resolver.resolve_prefix(_PROJECT, 'bff81530')

        assert excinfo.value.store == 'mem0'
        assert excinfo.value.__cause__ is error

    @pytest.mark.asyncio
    async def test_a_failing_preview_read_is_mem0_unavailable(self):
        error = ConnectionError('qdrant down mid-resolution')
        resolver = make_resolver(mem0=FakeMem0({_FULL: _memory('x')}, point_error=error))

        with pytest.raises(ResolverUnavailable) as excinfo:
            await resolver.resolve_prefix(_PROJECT, 'bff81530')

        assert excinfo.value.store == 'mem0'
        assert excinfo.value.__cause__ is error

    @pytest.mark.parametrize('error', [
        ConnectionError('falkordb down'),
        IncompleteEnumerationError('structurally incomplete'),
    ])
    @pytest.mark.asyncio
    async def test_a_failing_graph_read_is_graphiti_unavailable(self, error):
        resolver = make_resolver(graphiti=FakeGraphiti(error=error))

        with pytest.raises(ResolverUnavailable) as excinfo:
            await resolver.resolve_prefix(_PROJECT, 'bff81530')

        assert excinfo.value.store == 'graphiti'
        assert excinfo.value.__cause__ is error

    @pytest.mark.asyncio
    async def test_a_hanging_graph_read_is_graphiti_unavailable_after_the_read_timeout(self):
        resolver = make_resolver(graphiti=FakeGraphiti(hang=True), read_timeout_seconds=0.05)

        with pytest.raises(ResolverUnavailable) as excinfo:
            await resolver.resolve_prefix(_PROJECT, 'bff81530')

        assert excinfo.value.store == 'graphiti'

    @pytest.mark.asyncio
    async def test_a_hanging_walk_is_mem0_unavailable_after_the_read_timeout(self):
        graphiti = FakeGraphiti()
        resolver = make_resolver(mem0=FakeMem0(hang=True), graphiti=graphiti, read_timeout_seconds=0.05)

        with pytest.raises(ResolverUnavailable) as excinfo:
            await resolver.resolve_prefix(_PROJECT, 'bff81530')

        assert excinfo.value.store == 'mem0'
        assert graphiti.calls == []

    @pytest.mark.asyncio
    async def test_a_missing_collection_holds_zero_ids_rather_than_being_an_outage(self):
        resolver = make_resolver(
            mem0=FakeMem0(scroll_error=_missing_collection()),
            graphiti=FakeGraphiti(nodes={_FULL: 'node'}),
        )

        resolution = await resolver.resolve_prefix(_PROJECT, 'bff81530')

        assert resolution == Resolution('unique', (Candidate('graphiti_node', _FULL, 'node'),))


class TestMissReport:
    @pytest.mark.asyncio
    async def test_live_lists_come_from_the_resolution_and_tombstones_from_their_source(self):
        tombstone = {'memory_id': _SIBLING, 'deleter': 'stage1_cycle_summary_trim'}
        mem0, tombstones = FakeMem0(), RecordingTombstones([tombstone])
        resolver = make_resolver(
            mem0=mem0,
            graphiti=FakeGraphiti(nodes={_FULL: 'node'}),
            tombstones=tombstones,
        )
        resolution = await resolver.resolve_prefix(_PROJECT, 'bff81530')

        report = await resolver.miss_report(_PROJECT, 'bff81530', resolution)

        assert report == {
            'mem0': [],
            'mem0_tombstone': [tombstone],
            'graphiti_node': [{'id': _FULL, 'preview': 'node'}],
            'graphiti_edge': [],
        }
        assert tombstones.calls == [(_PROJECT, 'bff81530')]
        assert len(mem0.scroll_calls) == 1, 'the report must not walk the collection again'

    @pytest.mark.asyncio
    async def test_unreadable_tombstones_are_reported_unchecked_never_empty(self):
        resolver = make_resolver(tombstones=RecordingTombstones(unreadable=True))
        resolution = await resolver.resolve_prefix(_PROJECT, 'bff81530')

        report = await resolver.miss_report(_PROJECT, 'bff81530', resolution)

        assert report['mem0_tombstone'] is None
        assert report['mem0'] == report['graphiti_node'] == report['graphiti_edge'] == []


class TestForMemoryService:
    @pytest.mark.asyncio
    async def test_wires_the_services_stores_collection_prefix_and_tombstones(
        self, mock_config, monkeypatch,
    ):
        mem0, graphiti, tombstones = FakeMem0({_FULL: _memory('x')}), FakeGraphiti(), RecordingTombstones()
        service = MemoryService(mock_config)
        monkeypatch.setattr(service, 'mem0', mem0)
        monkeypatch.setattr(service, 'graphiti', graphiti)
        monkeypatch.setattr(service, 'find_mem0_deletion_tombstones', tombstones)
        resolver = IdResolver.for_memory_service(service)

        resolution = await resolver.resolve_prefix(_PROJECT, 'bff81530')
        await resolver.miss_report(_PROJECT, 'bff81530', resolution)

        assert mem0.scroll_calls[0][0] == f'{mock_config.mem0.collection_prefix}_reify'
        assert [group_id for _, _, group_id in graphiti.calls] == [_PROJECT, _PROJECT]
        assert tombstones.calls == [(_PROJECT, 'bff81530')]
