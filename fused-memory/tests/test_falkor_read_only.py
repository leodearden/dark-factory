"""Tests for fused_memory/backends/falkor_read_only.py, the read-only FalkorDB seam."""
from __future__ import annotations

import ast
import dataclasses
import logging
import re
from pathlib import Path
from typing import Any

import pytest
import redis.exceptions
from _fm_helpers import FALKOR_HOST, FALKOR_PORT, falkor_skipif, unique_graph_name
from falkordb.asyncio import FalkorDB
from falkordb.asyncio.graph import AsyncGraph

from fused_memory.backends import falkor_read_only
from fused_memory.backends.falkor_read_only import (
    FORWARDED_ATTRIBUTES,
    RO_COMMAND,
    NonReadOnlyCommandError,
    PagedQuery,
    ReadOnlyGraphProxy,
    ReadOnlyGraphReader,
    open_graph,
)
from fused_memory.backends.graphiti_client import (
    INCOMPLETE_STRUCTURAL_REFUSAL,
    PagedRead,
)


class _RecordingGraph:
    """A graph handle that records reads and refuses writes."""

    def __init__(self, *, resultset_cap: int = 10) -> None:
        self.name = 'g'
        self.resultset_cap = resultset_cap
        self.queries: list[tuple[str, dict | None]] = []
        self.write_calls: list[str] = []

    async def ro_query(self, cypher: str, params: dict | None = None) -> None:
        self.queries.append((cypher, params))

    async def query(self, cypher: str, params: dict | None = None) -> None:
        self.write_calls.append(cypher)
        raise AssertionError('the read-only seam may never issue query()')


class TestTheRefusal:
    def test_a_refusal_is_a_typed_error_naming_what_was_refused_and_what_is_allowed(self) -> None:
        error = NonReadOnlyCommandError('query')
        assert isinstance(error, RuntimeError)
        assert error.primitive == 'query'
        message = str(error)
        assert "'query'" in message
        assert RO_COMMAND in message
        assert 'read-only' in message

    def test_the_forwarded_set_is_the_read_primitive_and_the_graph_name_only(self) -> None:
        assert RO_COMMAND == 'GRAPH.RO_QUERY'
        assert sorted(FORWARDED_ATTRIBUTES) == ['name', 'ro_query']
        assert isinstance(FORWARDED_ATTRIBUTES, frozenset)


class TestReadOnlyGraphProxy:
    def test_the_read_primitive_resolves_to_the_wrapped_graph(self) -> None:
        graph = _RecordingGraph()
        assert ReadOnlyGraphProxy(graph).ro_query.__self__ is graph

    def test_the_graph_name_is_forwarded(self) -> None:
        assert ReadOnlyGraphProxy(_RecordingGraph()).name == 'g'

    def test_the_write_primitive_is_refused_before_any_byte_reaches_the_store(self) -> None:
        graph = _RecordingGraph()
        proxy = ReadOnlyGraphProxy(graph)
        with pytest.raises(NonReadOnlyCommandError, match="'query'"):
            _ = proxy.query
        assert graph.write_calls == []

    def test_an_attribute_the_handle_carries_is_refused_unless_forwarded(self) -> None:
        """Deny by default: carrying an attribute is no licence to reach it through the seam."""
        proxy = ReadOnlyGraphProxy(_RecordingGraph())
        for name in ('queries', 'resultset_cap', 'write_calls'):
            with pytest.raises(NonReadOnlyCommandError) as caught:
                getattr(proxy, name)
            assert caught.value.primitive == name

    def test_every_asyncgraph_primitive_but_the_read_is_refused(self) -> None:
        """Enumerated from the installed falkordb, so a primitive it adds later is refused, not forwarded."""
        primitives = {name for name in dir(AsyncGraph) if not name.startswith('__')}
        refused = primitives - FORWARDED_ATTRIBUTES
        assert {
            'query', '_query', 'delete', 'copy', 'call_procedure', 'profile',
            'create_node_range_index', 'create_edge_unique_constraint',
            'drop_node_vector_index', 'drop_edge_mandatory_constraint',
        } <= refused
        proxy = ReadOnlyGraphProxy(_RecordingGraph())
        for name in sorted(refused):
            with pytest.raises(NonReadOnlyCommandError) as caught:
                getattr(proxy, name)
            assert caught.value.primitive == name


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


_SKIP_LIMIT_RE = re.compile(r'SKIP\s+(\d+)\s+LIMIT\s+(\d+)', re.IGNORECASE)
_CENSUS_RE = re.compile(r'RETURN\s+count\(([*\w]+)\)\s*$', re.IGNORECASE)


class _Result:
    def __init__(self, result_set: list[list] | None) -> None:
        self.result_set = result_set


class _CappedGraph:
    """A store that cuts every answer to a silent row cap and answers a census as FalkorDB does.

    Rows are ``(source, target, uuid, ...)``. Over a ``RELATES_TO`` pattern,
    ``count(*)`` counts distinct (source, target) pairs, so a multi-edge
    collapses; any other count is the row count. ``truncate_at`` serves only
    that many rows, as a short read would. ``query`` raises.
    """

    def __init__(
        self,
        corpus: list[list],
        *,
        cap: int = 10,
        census_override: int | None = None,
        truncate_at: int | None = None,
        answers_none: bool = False,
    ) -> None:
        self.corpus = corpus
        self.cap = cap
        self.census_override = census_override
        self.truncate_at = truncate_at
        self.answers_none = answers_none
        self.queries: list[tuple[str, dict | None]] = []
        self.write_calls: list[str] = []

    @property
    def page_queries(self) -> list[str]:
        return [cypher for cypher, _ in self.queries if _SKIP_LIMIT_RE.search(cypher)]

    def _census(self, cypher: str, counted: str) -> int:
        if self.census_override is not None:
            return self.census_override
        if counted == '*' and 'RELATES_TO' in cypher:
            return len({(row[0], row[1]) for row in self.corpus})
        return len(self.corpus)

    async def ro_query(self, cypher: str, params: dict | None = None) -> _Result:
        self.queries.append((cypher, params))
        if self.answers_none:
            return _Result(None)
        if census := _CENSUS_RE.search(cypher.strip()):
            return _Result([[self._census(cypher, census.group(1))]])
        served = self.corpus[: self.truncate_at]
        if bounds := _SKIP_LIMIT_RE.search(cypher):
            skip, limit = int(bounds.group(1)), int(bounds.group(2))
            return _Result(served[skip: skip + limit][: self.cap])
        return _Result(served[: self.cap])

    async def query(self, cypher: str, params: dict | None = None) -> None:
        self.write_calls.append(cypher)
        raise AssertionError('the read-only seam may never issue query()')


def _edge_rows(n: int) -> list[list]:
    return [[f'src-{i}', f'dst-{i}', f'edge-{i:04d}'] for i in range(n)]


def _multi_edge_rows() -> list[list]:
    """Seven edges over six (source, target) pairs: edges 0 and 1 are a multi-edge."""
    rows = _edge_rows(7)
    rows[1][0], rows[1][1] = rows[0][0], rows[0][1]
    return rows


_EDGE_ROWS_QUERY = PagedQuery(
    match='MATCH (a)-[r:RELATES_TO]->(b)', row='r', returns=('a.name', 'b.name', 'r.uuid')
)


def _reader(
    graph: _CappedGraph, *, page_size: int = 5, resultset_size: int = 10
) -> ReadOnlyGraphReader:
    return ReadOnlyGraphReader(
        graph=graph, graph_name='g', page_size=page_size, resultset_size=resultset_size
    )


class TestReadPaged:
    @pytest.mark.asyncio
    async def test_a_corpus_larger_than_the_cap_is_read_in_full(self) -> None:
        graph = _CappedGraph(_edge_rows(25), cap=10)
        read = await _reader(graph).read_paged(_EDGE_ROWS_QUERY)
        assert isinstance(read, PagedRead)
        assert read.complete is True
        assert read.rows_seen == read.expected_rows == 25
        assert len(graph.page_queries) >= 5

    @pytest.mark.asyncio
    async def test_a_census_disagreement_surfaces_rather_than_raises(self) -> None:
        graph = _CappedGraph(_edge_rows(6), census_override=99)
        read = await _reader(graph).read_paged(_EDGE_ROWS_QUERY)
        assert read.complete is False
        assert read.reason is not None

    @pytest.mark.asyncio
    async def test_a_read_short_by_the_multi_edge_excess_is_incomplete(self) -> None:
        """A count(*) census would equal the pair count and pass this short read as complete."""
        corpus = _multi_edge_rows()
        pairs = len({(row[0], row[1]) for row in corpus})
        assert pairs < len(corpus)
        graph = _CappedGraph(corpus, truncate_at=pairs)
        read = await _reader(graph).read_paged(_EDGE_ROWS_QUERY)
        assert read.complete is False
        assert read.rows_seen == pairs
        assert read.expected_rows == len(corpus)

    @pytest.mark.asyncio
    async def test_params_reach_the_census_and_every_page(self) -> None:
        graph = _CappedGraph(_edge_rows(12))
        await _reader(graph).read_paged(_EDGE_ROWS_QUERY, params={'x': 1})
        assert len(graph.queries) >= 3
        assert all(params == {'x': 1} for _, params in graph.queries)

    @pytest.mark.asyncio
    async def test_the_paging_knobs_reach_the_primitive(self) -> None:
        graph = _CappedGraph(_edge_rows(3))
        read = await _reader(graph, page_size=10, resultset_size=10).read_paged(_EDGE_ROWS_QUERY)
        assert read.incomplete_kind == INCOMPLETE_STRUCTURAL_REFUSAL


class TestReadUnpaged:
    @pytest.mark.asyncio
    async def test_one_query_is_issued_with_its_params(self) -> None:
        graph = _CappedGraph(_edge_rows(3))
        rows = await _reader(graph).read_unpaged('MATCH (e:Episodic) RETURN e.uuid', {'p': 1})
        assert graph.queries == [('MATCH (e:Episodic) RETURN e.uuid', {'p': 1})]
        assert rows == _edge_rows(3)

    @pytest.mark.asyncio
    async def test_a_null_result_set_reads_as_no_rows(self) -> None:
        graph = _CappedGraph(_edge_rows(3), answers_none=True)
        assert await _reader(graph).read_unpaged('MATCH (e:Episodic) RETURN e.uuid') == []

    @pytest.mark.asyncio
    async def test_a_result_at_the_cap_warns_that_it_may_be_truncated(
        self, caplog: pytest.LogCaptureFixture
    ) -> None:
        graph = _CappedGraph(_edge_rows(25), cap=10)
        with caplog.at_level(logging.WARNING, logger='fused_memory.backends.falkor_read_only'):
            rows = await _reader(graph).read_unpaged('MATCH (e:Episodic) RETURN e.uuid')
        assert len(rows) == 10
        warnings = [
            r for r in caplog.records
            if r.name == 'fused_memory.backends.falkor_read_only' and r.levelno == logging.WARNING
        ]
        assert any('resultset_size=10' in r.getMessage() for r in warnings)

    @pytest.mark.asyncio
    async def test_a_result_below_the_cap_does_not_warn(
        self, caplog: pytest.LogCaptureFixture
    ) -> None:
        graph = _CappedGraph(_edge_rows(9), cap=10)
        with caplog.at_level(logging.WARNING, logger='fused_memory.backends.falkor_read_only'):
            await _reader(graph).read_unpaged('MATCH (e:Episodic) RETURN e.uuid')
        assert [r for r in caplog.records if r.name == 'fused_memory.backends.falkor_read_only'] == []


class TestTheGuardedHandle:
    @pytest.mark.asyncio
    async def test_the_writable_query_is_never_issued(self) -> None:
        graph = _CappedGraph(_edge_rows(12))
        reader = _reader(graph)
        await reader.read_paged(_EDGE_ROWS_QUERY)
        await reader.read_unpaged('MATCH (e:Episodic) RETURN e.uuid')
        assert graph.queries
        assert graph.write_calls == []

    def test_reads_go_through_the_command_guard(self) -> None:
        graph = _CappedGraph(_edge_rows(1))
        guarded = _reader(graph).guarded_graph()
        assert isinstance(guarded, ReadOnlyGraphProxy)
        assert guarded.ro_query.__self__ is graph
        for primitive in ('query', 'delete', 'create_node_range_index'):
            with pytest.raises(NonReadOnlyCommandError):
                getattr(guarded, primitive)


class TestOpening:
    def test_a_reader_opens_nothing_until_its_first_read(self) -> None:
        """FalkorDB's constructor probes the server synchronously, so a refused port surfaces at first use."""
        reader = ReadOnlyGraphReader(graph_name='g', host='127.0.0.1', port=1)
        with pytest.raises(redis.exceptions.ConnectionError):
            reader.guarded_graph()

    @pytest.mark.parametrize('address', [{'host': 'h'}, {'port': 1}], ids=['host', 'port'])
    def test_a_uri_with_a_host_or_port_is_refused_before_any_client_is_built(
        self, monkeypatch: pytest.MonkeyPatch, address: dict
    ) -> None:
        def _no_client(*args: object, **kwargs: object) -> None:
            raise AssertionError('open_graph built a client for an ambiguous address')

        monkeypatch.setattr(FalkorDB, '__init__', _no_client)
        with pytest.raises(ValueError, match='uri'):
            open_graph('g', uri='redis://x', **address)

    def test_the_module_never_constructs_or_directly_imports_the_graphiti_driver(self) -> None:
        """graphiti's FalkorDriver.__init__ fire-and-forgets build_indices_and_constraints(), a write.

        Direct imports only: graphiti_client, which this module imports, loads
        the driver module transitively.
        """
        tree = ast.parse(Path(falkor_read_only.__file__).read_text(encoding='utf-8'))
        calls = [
            node.lineno for node in ast.walk(tree)
            if isinstance(node, ast.Call)
            and (getattr(node.func, 'id', None) or getattr(node.func, 'attr', None)) == 'FalkorDriver'
        ]
        imported = [
            alias.name for node in ast.walk(tree) if isinstance(node, ast.Import)
            for alias in node.names
        ] + [
            node.module or '' for node in ast.walk(tree) if isinstance(node, ast.ImportFrom)
        ]
        assert calls == []
        assert [name for name in imported if name.split('.')[0] == 'graphiti_core'] == []


class _OpenedGraph(_CappedGraph):
    def __init__(self, client: _FakeClient, name: str) -> None:
        super().__init__(_edge_rows(1))
        self.client = client
        self.name = name


class _FakeClient:
    def __init__(self, **address: object) -> None:
        self.address = address
        self.closes = 0

    def select_graph(self, graph_name: str) -> _OpenedGraph:
        return _OpenedGraph(self, graph_name)

    async def aclose(self) -> None:
        self.closes += 1


@pytest.fixture
def opened_clients(monkeypatch: pytest.MonkeyPatch) -> list[_FakeClient]:
    """Every client open_graph builds, in order; none of them touches a network."""
    clients: list[_FakeClient] = []

    def _build(**address: object) -> _FakeClient:
        client = _FakeClient(**address)
        clients.append(client)
        return client

    monkeypatch.setattr(falkor_read_only, 'FalkorDB', _build)
    return clients


class TestClosing:
    @pytest.mark.asyncio
    async def test_aclose_closes_the_client_the_reader_opened(
        self, opened_clients: list[_FakeClient]
    ) -> None:
        reader = ReadOnlyGraphReader(graph_name='g', host='h', port=1)
        await reader.read_unpaged('MATCH (e:Episodic) RETURN e.uuid')
        await reader.aclose()
        assert [(client.address, client.closes) for client in opened_clients] == [
            ({'host': 'h', 'port': 1}, 1)
        ]

    @pytest.mark.asyncio
    async def test_a_reader_that_never_read_opens_and_closes_nothing(
        self, opened_clients: list[_FakeClient]
    ) -> None:
        await ReadOnlyGraphReader(graph_name='g', host='h', port=1).aclose()
        assert opened_clients == []

    @pytest.mark.asyncio
    async def test_an_injected_graph_is_left_to_its_owner(
        self, opened_clients: list[_FakeClient]
    ) -> None:
        owner = _FakeClient()
        reader = ReadOnlyGraphReader(graph_name='g', graph=_OpenedGraph(owner, 'g'))
        await reader.read_unpaged('MATCH (e:Episodic) RETURN e.uuid')
        await reader.aclose()
        assert owner.closes == 0
        assert opened_clients == []

    @pytest.mark.asyncio
    async def test_a_read_after_aclose_opens_a_fresh_client(
        self, opened_clients: list[_FakeClient]
    ) -> None:
        reader = ReadOnlyGraphReader(graph_name='g', host='h', port=1)
        await reader.read_unpaged('MATCH (e:Episodic) RETURN e.uuid')
        await reader.aclose()
        await reader.read_unpaged('MATCH (e:Episodic) RETURN e.uuid')
        await reader.aclose()
        assert [client.closes for client in opened_clients] == [1, 1]


@pytest.mark.integration
@falkor_skipif()
@pytest.mark.timeout(30)
@pytest.mark.asyncio
async def test_a_live_open_yields_a_falkordb_client_and_no_graphiti_driver(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    from graphiti_core.driver import falkordb_driver

    def _boom(*args: object, **kwargs: object) -> None:
        raise AssertionError('the read-only seam constructed a graphiti FalkorDriver')

    monkeypatch.setattr(falkordb_driver.FalkorDriver, '__init__', _boom)
    opened: list[FalkorDB] = []

    class _RecordingFalkorDB(FalkorDB):
        def __init__(self, **address: Any) -> None:
            super().__init__(**address)
            opened.append(self)

    monkeypatch.setattr(falkor_read_only, 'FalkorDB', _RecordingFalkorDB)
    reader = ReadOnlyGraphReader(
        graph_name=unique_graph_name('6136_ro_seam'), host=FALKOR_HOST, port=FALKOR_PORT
    )
    try:
        assert isinstance(reader.guarded_graph(), ReadOnlyGraphProxy)
    finally:
        await reader.aclose()
    assert len(opened) == 1
