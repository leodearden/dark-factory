"""POST /resolve-prefix — the resolver's HTTP surface (PRD C5, B13; task 5323).

The route is mounted on the same Starlette app as the MCP endpoint, so it is
driven exactly like /health: TestClient over create_mcp_server(...)'s
streamable_http_app(), with the REAL IdResolver over in-memory stores.
"""

from __future__ import annotations

from unittest.mock import AsyncMock

import pytest
from _id_resolver_fakes import FakeGraphiti, FakeMem0, RecordingTombstones, make_resolver
from shared.uuid_prefix_guard import Candidate, Resolution
from starlette.testclient import TestClient

from fused_memory.server.tools import create_mcp_server

_KNOWN_PROJECTS = {'reify': '/abs/reify', 'dark_factory': '/abs/df'}
_FULL = 'bff81530-70e5-4ab6-939e-507299d1bcd2'
_SIBLING = 'bff81530-0000-4000-8000-000000000002'
_LONG_TEXT = 'resolver preview ' * 10


class _Rig:
    """One server over fake stores, plus the stores so a test can read their call logs."""

    def __init__(self, *, mem0: FakeMem0 | None = None, graphiti: FakeGraphiti | None = None) -> None:
        self.mem0 = mem0 or FakeMem0()
        self.graphiti = graphiti or FakeGraphiti()
        self.tombstones = RecordingTombstones()
        self.resolver = make_resolver(mem0=self.mem0, graphiti=self.graphiti, tombstones=self.tombstones)
        server = create_mcp_server(AsyncMock(), id_resolver=self.resolver, known_projects=_KNOWN_PROJECTS)
        self.client = TestClient(server.streamable_http_app())

    def post(self, body: object):
        return self.client.post('/resolve-prefix', json=body)

    def store_calls(self) -> int:
        return (
            len(self.mem0.scroll_calls) + len(self.mem0.point_reads)
            + len(self.graphiti.calls) + len(self.tombstones.calls)
        )


class TestResolutions:
    def test_unique_is_200_with_exactly_the_resolution_fields(self):
        rig = _Rig(mem0=FakeMem0({_FULL: {'data': _LONG_TEXT}}))

        response = rig.post({'project_id': 'reify', 'prefix': 'bff81530'})

        assert response.status_code == 200
        body = response.json()
        assert body == {
            'outcome': 'unique',
            'candidates': [{'namespace': 'mem0', 'id': _FULL, 'preview': _LONG_TEXT[:80]}],
        }
        decoded = Resolution(body['outcome'], tuple(Candidate(**c) for c in body['candidates']))
        assert decoded == Resolution('unique', (Candidate('mem0', _FULL, _LONG_TEXT[:80]),))

    def test_none_is_200(self):
        response = _Rig().post({'project_id': 'reify', 'prefix': 'bff81530'})

        assert response.status_code == 200
        assert response.json() == {'outcome': 'none', 'candidates': []}

    def test_ambiguous_is_200_naming_every_candidate(self):
        rig = _Rig(mem0=FakeMem0({_FULL: {'data': 'a'}}), graphiti=FakeGraphiti(nodes={_SIBLING: 'node'}))

        response = rig.post({'project_id': 'reify', 'prefix': 'bff81530'})

        assert response.status_code == 200
        assert response.json() == {
            'outcome': 'ambiguous',
            'candidates': [
                {'namespace': 'mem0', 'id': _FULL, 'preview': 'a'},
                {'namespace': 'graphiti_node', 'id': _SIBLING, 'preview': 'node'},
            ],
        }

    def test_a_hyphenated_project_id_is_canonicalized(self):
        rig = _Rig()

        response = rig.post({'project_id': 'dark-factory', 'prefix': 'bff81530'})

        assert response.status_code == 200
        assert {group_id for _, _, group_id in rig.graphiti.calls} == {'dark_factory'}


class TestValidation:
    @pytest.mark.parametrize(('body', 'field'), [
        ({'project_id': 'reify', 'prefix': 'bff8153'}, 'prefix'),
        ({'project_id': 'reify', 'prefix': 'xyz12345'}, 'prefix'),
        ({'project_id': 'reify', 'prefix': 'BFF81530'}, 'prefix'),
        ({'project_id': 'reify'}, 'prefix'),
        ({'project_id': 'reify', 'prefix': 12345678}, 'prefix'),
        ({'project_id': 'unknown_project', 'prefix': 'bff81530'}, 'project_id'),
        ({'project_id': '-home-leo-src-reify', 'prefix': 'bff81530'}, 'project_id'),
        ({'project_id': 'reify;drop', 'prefix': 'bff81530'}, 'project_id'),
        ({'prefix': 'bff81530'}, 'project_id'),
        ({'project_id': 7, 'prefix': 'bff81530'}, 'project_id'),
        (['reify', 'bff81530'], 'body'),
    ])
    def test_a_bad_request_is_400_naming_its_field_and_touches_no_store(self, body, field):
        rig = _Rig(mem0=FakeMem0({_FULL: {'data': 'x'}}))

        response = rig.post(body)

        assert response.status_code == 400
        payload = response.json()
        assert payload['error_type'] == 'ValidationError'
        assert payload['field'] == field
        assert payload['error']
        assert rig.store_calls() == 0

    @pytest.mark.parametrize('prefix', [
        'bff81530',
        'bff81530' + 'a' * 23,
        _FULL,
        _FULL.upper(),
        'bff8153',
        'BFF81530',
        'xyz12345',
        'bff8153070e54ab6939e507299d1bcd2',
        '-bff81530',
    ])
    @pytest.mark.asyncio
    async def test_the_route_accepts_exactly_what_the_resolver_accepts(self, prefix):
        rig = _Rig()
        try:
            await rig.resolver.resolve_prefix('reify', prefix)
            resolver_accepts = True
        except ValueError:
            resolver_accepts = False

        route_accepts = rig.post({'project_id': 'reify', 'prefix': prefix}).status_code != 400

        assert route_accepts == resolver_accepts

    def test_a_non_json_body_is_400_body(self):
        rig = _Rig()

        response = rig.client.post(
            '/resolve-prefix', content=b'not json', headers={'content-type': 'application/json'},
        )

        assert response.status_code == 400
        assert response.json()['field'] == 'body'
        assert rig.store_calls() == 0


class TestUnavailable:
    def test_a_graph_outage_is_503_naming_graphiti_never_a_none(self):
        rig = _Rig(graphiti=FakeGraphiti(error=ConnectionError('falkordb down')))

        response = rig.post({'project_id': 'reify', 'prefix': 'bff81530'})

        assert response.status_code == 503
        body = response.json()
        assert body['error_type'] == 'resolver_unavailable'
        assert body['store'] == 'graphiti'
        assert body['error']
        assert 'outcome' not in body

    def test_a_mem0_outage_is_503_naming_mem0(self):
        rig = _Rig(mem0=FakeMem0({_FULL: {'data': 'x'}}, scroll_error=ConnectionError('qdrant down')))

        response = rig.post({'project_id': 'reify', 'prefix': 'bff81530'})

        assert response.status_code == 503
        assert response.json()['store'] == 'mem0'


def test_get_is_405():
    assert _Rig().client.get('/resolve-prefix').status_code == 405
