"""Tests for fused_memory/backends/falkor_read_only.py, the read-only FalkorDB seam."""
from __future__ import annotations

import dataclasses
from collections.abc import Mapping, MutableMapping

import pytest

from fused_memory.backends.falkor_read_only import (
    COMMAND_BY_PRIMITIVE,
    RO_COMMAND,
    NonReadOnlyCommandError,
    PagedQuery,
    ReadOnlyGraphProxy,
    assert_read_only_command,
)


class _RecordingGraph:
    """A graph handle that records reads and refuses writes."""

    def __init__(self, *, resultset_cap: int = 10) -> None:
        self.resultset_cap = resultset_cap
        self.queries: list[tuple[str, dict | None]] = []
        self.write_calls: list[str] = []

    async def ro_query(self, cypher: str, params: dict | None = None) -> None:
        self.queries.append((cypher, params))

    async def query(self, cypher: str, params: dict | None = None) -> None:
        self.write_calls.append(cypher)
        raise AssertionError('the read-only seam may never issue query()')


class TestTheCommandGuard:
    def test_the_read_only_command_passes(self) -> None:
        assert RO_COMMAND == 'GRAPH.RO_QUERY'
        assert assert_read_only_command(RO_COMMAND) is None

    def test_a_writable_command_is_refused_with_a_typed_error(self) -> None:
        with pytest.raises(NonReadOnlyCommandError) as caught:
            assert_read_only_command('GRAPH.QUERY')
        error = caught.value
        assert isinstance(error, RuntimeError)
        assert error.command == 'GRAPH.QUERY'
        message = str(error)
        assert 'GRAPH.QUERY' in message
        assert 'GRAPH.RO_QUERY' in message
        assert 'read-only' in message

    def test_every_falkordb_primitive_the_proxy_knows_is_mapped(self) -> None:
        """A primitive absent from the map is forwarded UNGUARDED, so the map is what can go quietly wrong."""
        assert COMMAND_BY_PRIMITIVE['ro_query'] == RO_COMMAND
        assert COMMAND_BY_PRIMITIVE['query'] == 'GRAPH.QUERY'
        assert isinstance(COMMAND_BY_PRIMITIVE, Mapping)
        assert not isinstance(COMMAND_BY_PRIMITIVE, MutableMapping)
        assert not hasattr(COMMAND_BY_PRIMITIVE, '__setitem__')


class TestReadOnlyGraphProxy:
    def test_the_read_primitive_resolves_to_the_wrapped_graph(self) -> None:
        graph = _RecordingGraph()
        assert ReadOnlyGraphProxy(graph).ro_query.__self__ is graph

    def test_the_command_guard_is_reachable_on_a_real_read(self) -> None:
        """The write primitive is refused before any byte reaches the store, naming the command rather than the method."""
        graph = _RecordingGraph()
        proxy = ReadOnlyGraphProxy(graph)
        with pytest.raises(NonReadOnlyCommandError, match='GRAPH.QUERY'):
            _ = proxy.query
        assert graph.write_calls == []

    def test_unmapped_attributes_are_forwarded_untouched(self) -> None:
        """A handle legitimately carries bookkeeping; guarding it would make the proxy brittle without making it safer."""
        graph = _RecordingGraph(resultset_cap=7)
        proxy = ReadOnlyGraphProxy(graph)
        assert proxy.queries is graph.queries
        assert proxy.resultset_cap == graph.resultset_cap


_EDGE_MATCH = 'MATCH (a)-[r:RELATES_TO]->(b) WHERE r.invalid_at IS NULL'
_EDGE_QUERY = PagedQuery(match=_EDGE_MATCH, row='r', returns=('r.uuid', 'r.fact'))
_NODE_QUERY = PagedQuery(match='MATCH (n:Entity)', row='n', returns=('n.uuid', 'n.name'))


class TestPagedQuery:
    def test_an_edge_query_builds_its_page_and_census_from_one_match(self) -> None:
        assert _EDGE_QUERY.page_template == (
            'MATCH (a)-[r:RELATES_TO]->(b) WHERE r.invalid_at IS NULL '
            'RETURN r.uuid, r.fact ORDER BY r.uuid SKIP {skip} LIMIT {limit}'
        )
        assert _EDGE_QUERY.census_cypher == (
            'MATCH (a)-[r:RELATES_TO]->(b) WHERE r.invalid_at IS NULL RETURN count(r)'
        )

    def test_the_census_counts_the_row_variable_never_star(self) -> None:
        """count(*) over an edge pattern counts (source, target) pairs, so a multi-edge reads short."""
        assert _NODE_QUERY.census_cypher == 'MATCH (n:Entity) RETURN count(n)'
        for query in (_EDGE_QUERY, _NODE_QUERY):
            assert 'count(*)' not in query.census_cypher

    @pytest.mark.parametrize('query', [_EDGE_QUERY, _NODE_QUERY], ids=['edges', 'nodes'])
    def test_census_and_page_describe_one_population(self, query: PagedQuery) -> None:
        match = query.match.strip()
        assert query.page_template.startswith(match + ' RETURN ')
        assert query.census_cypher.startswith(match + ' RETURN ')

    def test_returns_that_omit_the_sort_key_are_refused(self) -> None:
        with pytest.raises(ValueError, match=r'r\.uuid') as caught:
            PagedQuery(match=_EDGE_MATCH, row='r', returns=('r.fact',))
        assert "('r.fact',)" in str(caught.value)

    def test_empty_returns_are_refused(self) -> None:
        with pytest.raises(ValueError, match='returns'):
            PagedQuery(match=_EDGE_MATCH, row='r', returns=())

    def test_a_row_that_is_not_an_identifier_is_refused(self) -> None:
        with pytest.raises(ValueError, match="'r.x'"):
            PagedQuery(match=_EDGE_MATCH, row='r.x', returns=('r.x.uuid',))

    @pytest.mark.parametrize('field', ['match', 'row', 'returns'])
    def test_the_query_is_frozen(self, field: str) -> None:
        with pytest.raises(dataclasses.FrozenInstanceError):
            setattr(_EDGE_QUERY, field, getattr(_NODE_QUERY, field))
