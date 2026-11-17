"""The scratch FalkorDB's seed: a graph copied from the source server by DUMP and RESTORE.

The offline cases are refused before any connection, with both servers aimed at
refused ports. The live cases run on the scratch server alone: a seeded
``evalmem_test_`` graph is read, deleted and restored under its own name, which is
the cross-server copy seen from one server. The graph is deleted in ``finally``;
run them with ``uv run pytest -m integration tests/test_seed_scratch_reference_graph.py``.
"""

from __future__ import annotations

import asyncio
import contextlib
import itertools
import time
import uuid
from collections.abc import Iterator
from pathlib import Path
from types import SimpleNamespace

import pytest
from _falkor_index_doubles import LIVE_HEADER, rows_for
from _fm_falkor_live import FALKOR_HOST, FALKOR_PORT, await_index_operational, falkor_skipif
from _fm_helpers import load_script_module
from _fm_sockets import refused_port
from falkordb.asyncio import FalkorDB
from falkordb.asyncio.graph import AsyncGraph
from redis.exceptions import ConnectionError as RedisConnectionError
from redis.exceptions import ResponseError
from shared.whole_tree_guard_family import WHOLE_TREE_SCAN_TEST_TIMEOUT

from fused_memory.arm_harness.topology import read_topology, topology_hash
from fused_memory.backends.falkor_indices import (
    INDEX_CATALOG_QUERY,
    INDEX_DEFINITION_QUERY,
    IndexCatalogUnsettledError,
)
from fused_memory.config.falkordb_endpoints import PRODUCTION_FALKORDB

pytestmark = pytest.mark.timeout(WHOLE_TREE_SCAN_TEST_TIMEOUT)

SEED = load_script_module(
    Path(__file__).resolve().parents[1] / 'scripts' / 'seed_scratch_reference_graph.py'
)

SEED_CYPHER = (
    'CREATE (a:Entity {uuid: "n-merge", name: "merge worker", group_id: $g, '
    'name_embedding: vecf32($v)}), '
    '(b:Entity {uuid: "n-lock", name: "index lock", group_id: $g, '
    'name_embedding: vecf32($w)}), '
    '(a)-[:RELATES_TO {uuid: "r-1", name: "WAITS_ON", group_id: $g, '
    'fact: "the merge worker waits on the index lock"}]->(b)'
)
RANGE_INDEX_CYPHER = 'CREATE INDEX FOR (n:Entity) ON (n.uuid)'
FULLTEXT_INDEX_CYPHER = "CALL db.idx.fulltext.createNodeIndex('Entity', 'name')"
FULLTEXT_QUERY = (
    "CALL db.idx.fulltext.queryNodes('Entity', 'merge') YIELD node RETURN node.uuid"
)
VECTORS_QUERY = 'MATCH (n:Entity) RETURN n.uuid, n.name_embedding ORDER BY n.uuid'


def _scratch_name() -> str:
    return f'evalmem_test_{uuid.uuid4().hex[:8]}'


# --- refused before any connection ----------------------------------------------------


@pytest.fixture
def unreachable_servers(monkeypatch):
    """The source (FALKORDB_URI) and the scratch target (FALKOR_*) both on refused ports."""
    with refused_port() as source_port, refused_port() as scratch_port:
        monkeypatch.setenv('FALKORDB_URI', f'redis://127.0.0.1:{source_port}')
        monkeypatch.setenv('FALKOR_HOST', '127.0.0.1')
        monkeypatch.setenv('FALKOR_PORT', str(scratch_port))
        yield


def test_a_graph_outside_the_scratch_namespace_is_refused(unreachable_servers, capsys):
    code = SEED.main(['--graph', 'dark_factory'])

    err = capsys.readouterr().err
    assert code != 0
    assert "scratch guard refused 'dark_factory'" in err
    assert 'Traceback' not in err


def test_a_scratch_target_that_is_production_is_refused(unreachable_servers, monkeypatch, capsys):
    monkeypatch.delenv('FALKOR_HOST')
    monkeypatch.setenv('FALKOR_PORT', str(PRODUCTION_FALKORDB.port))

    code = SEED.main(['--graph', _scratch_name()])

    err = capsys.readouterr().err
    assert code != 0
    assert 'production' in err
    assert PRODUCTION_FALKORDB.uri in err
    assert 'Traceback' not in err


def test_a_source_that_is_the_scratch_server_is_refused(monkeypatch, capsys):
    with refused_port() as port:
        monkeypatch.setenv('FALKORDB_URI', f'redis://127.0.0.1:{port}')
        monkeypatch.setenv('FALKOR_HOST', '127.0.0.1')
        monkeypatch.setenv('FALKOR_PORT', str(port))

        code = SEED.main([])

    err = capsys.readouterr().err
    assert code != 0
    assert 'source and target are the same server' in err
    assert 'Traceback' not in err


# --- what a restore leaves behind when it cannot finish -------------------------------


class _EmptyGraph:
    """A restored graph with no nodes, edges or indexes, whose delete can be made to fail."""

    def __init__(self, delete_error: Exception | None = None) -> None:
        self.delete_error = delete_error
        self.deleted = False

    async def ro_query(self, query: str, *args: object, **kwargs: object) -> SimpleNamespace:
        return SimpleNamespace(header=LIVE_HEADER, result_set=[])

    async def delete(self) -> None:
        if self.delete_error is not None:
            raise self.delete_error
        self.deleted = True


_UNDER_CONSTRUCTION = '[Indexing] 1/2: UNDER CONSTRUCTION'
_STATUS_POSITION = [name for _, name in LIVE_HEADER].index('status')


class _IndexBuildingGraph(_EmptyGraph):
    """An empty restored graph whose catalog reports one index, with each read taking the next of *statuses*."""

    def __init__(self, statuses: Iterator[str]) -> None:
        super().__init__()
        self.statuses = statuses
        self.catalog_reads = 0

    async def ro_query(self, query: str, *args: object, **kwargs: object) -> SimpleNamespace:
        if query != INDEX_CATALOG_QUERY:
            return await super().ro_query(query, *args, **kwargs)
        self.catalog_reads += 1
        [row] = rows_for({('Entity', 'NODE', 'uuid', 'RANGE')})
        row[_STATUS_POSITION] = next(self.statuses)
        return SimpleNamespace(header=LIVE_HEADER, result_set=[row])


class _EmptyScratchServer:
    def __init__(self, graph: _EmptyGraph, save_error: Exception | None = None) -> None:
        self.graph = graph
        self.save_error = save_error
        self.restored: list[str] = []
        self.saves = 0
        self.connection = self

    async def list_graphs(self) -> list[str]:
        return []

    async def exists(self, name: str) -> int:
        return 0

    async def restore(self, name: str, ttl: int, payload: bytes) -> None:
        self.restored.append(name)

    async def save(self) -> None:
        self.saves += 1
        if self.save_error is not None:
            raise self.save_error

    def select_graph(self, name: str) -> _EmptyGraph:
        return self.graph


EMPTY_GRAPH_IDENTITY = SEED.GraphIdentity(
    node_count=0, edge_count=0, topology_hash=topology_hash((), ()), index_definitions=()
)
OTHER_GRAPH_IDENTITY = SEED.GraphIdentity(
    node_count=1, edge_count=0, topology_hash='0' * 64, index_definitions=()
)


def _restore(server: _EmptyScratchServer, name: str, identity) -> object:
    return asyncio.run(SEED.restore_graph(server, SEED.SourceGraph(name, b'dump', identity)))


def test_a_copy_that_differs_from_its_source_is_deleted_and_never_saved():
    name = _scratch_name()
    server = _EmptyScratchServer(_EmptyGraph())

    with pytest.raises(SEED.SeedMismatch, match='differs from its source'):
        _restore(server, name, OTHER_GRAPH_IDENTITY)

    assert server.restored == [name]
    assert server.graph.deleted
    assert server.saves == 0


def test_an_unverified_copy_that_cannot_be_deleted_is_named_with_both_errors():
    name = _scratch_name()
    server = _EmptyScratchServer(_EmptyGraph(delete_error=RedisConnectionError('connection lost')))

    with pytest.raises(SEED.CopyLeftBehind) as raised:
        _restore(server, name, OTHER_GRAPH_IDENTITY)

    message = str(raised.value)
    assert 'differs from its source' in message
    assert 'connection lost' in message
    assert f'GRAPH.DELETE {name}' in message
    assert isinstance(raised.value.__cause__, SEED.SeedMismatch)
    assert server.saves == 0


def test_a_failed_save_leaves_the_verified_copy_and_says_it_is_unsaved():
    name = _scratch_name()
    server = _EmptyScratchServer(
        _EmptyGraph(), save_error=ResponseError('Background save already in progress')
    )

    with pytest.raises(SEED.CopyNotSaved) as raised:
        _restore(server, name, EMPTY_GRAPH_IDENTITY)

    message = str(raised.value)
    assert name in message
    assert 'Background save already in progress' in message
    assert 'SAVE' in message
    assert not server.graph.deleted


def test_a_verified_copy_is_saved():
    name = _scratch_name()
    server = _EmptyScratchServer(_EmptyGraph())

    assert _restore(server, name, EMPTY_GRAPH_IDENTITY) == EMPTY_GRAPH_IDENTITY
    assert server.saves == 1
    assert not server.graph.deleted


@pytest.fixture
def fast_settle_poll(monkeypatch):
    monkeypatch.setattr(SEED, 'SETTLE_POLL_INTERVAL_S', 0.001)


def test_a_copy_whose_indexes_never_settle_is_deleted_and_never_saved(
    fast_settle_poll, monkeypatch
):
    monkeypatch.setattr(SEED, 'SETTLE_TIMEOUT_S', 0.02)
    name = _scratch_name()
    graph = _IndexBuildingGraph(itertools.repeat(_UNDER_CONSTRUCTION))
    server = _EmptyScratchServer(graph)

    with pytest.raises(IndexCatalogUnsettledError) as raised:
        _restore(server, name, EMPTY_GRAPH_IDENTITY)

    message = str(raised.value)
    assert name in message
    assert '0.02s' in message
    assert _UNDER_CONSTRUCTION in message
    assert graph.catalog_reads >= 2
    assert graph.deleted
    assert server.saves == 0


def test_a_copy_is_verified_and_saved_once_its_indexes_settle(fast_settle_poll):
    name = _scratch_name()
    graph = _IndexBuildingGraph(iter([_UNDER_CONSTRUCTION, 'OPERATIONAL']))
    server = _EmptyScratchServer(graph)

    assert _restore(server, name, EMPTY_GRAPH_IDENTITY) == EMPTY_GRAPH_IDENTITY
    assert graph.catalog_reads == 2
    assert server.saves == 1
    assert not graph.deleted


# --- live round trips on the scratch server -------------------------------------------


async def _seed_source(graph: AsyncGraph, name: str) -> None:
    await graph.query(SEED_CYPHER, {'g': name, 'v': [0.25, 0.5, 0.75], 'w': [0.5, 0.5, 0.0]})
    await graph.query(RANGE_INDEX_CYPHER)
    await graph.query(FULLTEXT_INDEX_CYPHER)
    await await_index_operational(graph)


async def _topology_hash(graph: AsyncGraph, name: str) -> str:
    return topology_hash(*await read_topology(graph, name))


@contextlib.asynccontextmanager
async def _seeded_scratch_graph():
    name = _scratch_name()
    client = FalkorDB(host=FALKOR_HOST, port=FALKOR_PORT)
    graph = client.select_graph(name)
    try:
        await _seed_source(graph, name)
        yield client, graph, name
    finally:
        with contextlib.suppress(Exception):
            await graph.delete()
        await client.aclose()


@falkor_skipif()
@pytest.mark.integration
@pytest.mark.timeout(300)
class TestLiveSeed:
    @pytest.mark.asyncio
    async def test_a_restored_graph_keeps_its_topology_vectors_and_indexes_and_is_saved(self):
        async with _seeded_scratch_graph() as (client, graph, name):
            topology_before = await _topology_hash(graph, name)
            definitions_before = (await graph.ro_query(INDEX_DEFINITION_QUERY)).result_set
            vectors_before = (await graph.ro_query(VECTORS_QUERY)).result_set
            source = await SEED.read_source(client, name)
            await graph.delete()
            restored_from = int(time.time())

            identity = await SEED.restore_graph(client, source)

            assert identity == source.identity
            assert identity.topology_hash == topology_before
            assert identity.index_definitions == tuple(map(tuple, definitions_before))
            assert name in await client.list_graphs()
            assert await _topology_hash(graph, name) == topology_before
            assert (await graph.ro_query(INDEX_DEFINITION_QUERY)).result_set == definitions_before
            assert (await graph.ro_query(VECTORS_QUERY)).result_set == vectors_before
            assert (await graph.ro_query(FULLTEXT_QUERY)).result_set == [['n-merge']]
            last_save = await client.connection.lastsave()
            assert last_save.timestamp() >= restored_from

    @pytest.mark.asyncio
    async def test_a_graph_the_target_already_holds_is_refused_and_left_untouched(self):
        async with _seeded_scratch_graph() as (client, graph, name):
            topology_before = await _topology_hash(graph, name)
            vectors_before = (await graph.ro_query(VECTORS_QUERY)).result_set

            with pytest.raises(SEED.TargetGraphExists, match=name):
                await SEED.seed_graph(client, client, name)

            assert await _topology_hash(graph, name) == topology_before
            assert (await graph.ro_query(VECTORS_QUERY)).result_set == vectors_before
