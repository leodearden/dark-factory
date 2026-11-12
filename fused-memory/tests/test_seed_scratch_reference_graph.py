"""The scratch FalkorDB's seed: a graph copied from the source server by DUMP and RESTORE.

The offline cases are refused before any connection, with both servers aimed at
refused ports. The live cases copy one ``evalmem_test_`` graph onto another on the
scratch server and delete both in ``finally``; run them with
``uv run pytest -m integration tests/test_seed_scratch_reference_graph.py``.
"""

from __future__ import annotations

import contextlib
import time
import uuid
from pathlib import Path

import pytest
from _fm_falkor_live import FALKOR_HOST, FALKOR_PORT, await_index_operational, falkor_skipif
from _fm_helpers import load_script_module
from arm_harness.test_live_replay_integration import INDEX_DEFINITION_QUERY
from falkordb.asyncio import FalkorDB
from falkordb.asyncio.graph import AsyncGraph
from test_lane_dependency_probe import refused_port

from fused_memory.arm_harness.topology import read_topology, topology_hash
from fused_memory.config.falkordb_endpoints import PRODUCTION_FALKORDB

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


# --- live copies on the scratch server ------------------------------------------------


async def _seed_source(graph: AsyncGraph, name: str) -> None:
    await graph.query(SEED_CYPHER, {'g': name, 'v': [0.25, 0.5, 0.75], 'w': [0.5, 0.5, 0.0]})
    await graph.query(RANGE_INDEX_CYPHER)
    await graph.query(FULLTEXT_INDEX_CYPHER)
    await await_index_operational(graph)


async def _topology_hash(graph: AsyncGraph, name: str) -> str:
    return topology_hash(*await read_topology(graph, name))


@contextlib.asynccontextmanager
async def _two_clients():
    source_client = FalkorDB(host=FALKOR_HOST, port=FALKOR_PORT)
    target_client = FalkorDB(host=FALKOR_HOST, port=FALKOR_PORT)
    try:
        yield source_client, target_client
    finally:
        await source_client.aclose()
        await target_client.aclose()


async def _delete_quietly(*graphs: AsyncGraph) -> None:
    for graph in graphs:
        with contextlib.suppress(Exception):
            await graph.delete()


@falkor_skipif()
@pytest.mark.integration
@pytest.mark.timeout(300)
class TestLiveSeed:
    @pytest.mark.asyncio
    async def test_a_copy_keeps_the_topology_vectors_and_indexes_and_is_saved(self):
        source_name, target_name = _scratch_name(), _scratch_name()
        async with _two_clients() as (source_client, target_client):
            source = source_client.select_graph(source_name)
            target = target_client.select_graph(target_name)
            try:
                await _seed_source(source, source_name)
                seeded_at = int(time.time())

                verdict = await SEED.seed_graph(
                    source_client, target_client, source_name, target_name
                )

                source_hash = await _topology_hash(source, source_name)
                source_definitions = (await source.ro_query(INDEX_DEFINITION_QUERY)).result_set
                assert verdict.topology_hash == source_hash
                assert await _topology_hash(target, target_name) == source_hash
                assert verdict.index_definitions == tuple(map(tuple, source_definitions))
                assert (
                    await target.ro_query(INDEX_DEFINITION_QUERY)
                ).result_set == source_definitions
                assert (await target.ro_query(VECTORS_QUERY)).result_set == (
                    await source.ro_query(VECTORS_QUERY)
                ).result_set
                assert (await target.ro_query(FULLTEXT_QUERY)).result_set == [['n-merge']]
                last_save = await target_client.connection.lastsave()
                assert last_save.timestamp() >= seeded_at
            finally:
                await _delete_quietly(source, target)

    @pytest.mark.asyncio
    async def test_an_existing_target_is_refused_and_left_untouched(self):
        source_name, target_name = _scratch_name(), _scratch_name()
        async with _two_clients() as (source_client, target_client):
            source = source_client.select_graph(source_name)
            target = target_client.select_graph(target_name)
            try:
                await _seed_source(source, source_name)
                await target.query('CREATE (:Entity {uuid: "n-existing", name: "already here"})')
                existing_hash = await _topology_hash(target, target_name)

                with pytest.raises(SEED.TargetGraphExists, match=target_name):
                    await SEED.seed_graph(source_client, target_client, source_name, target_name)

                assert await _topology_hash(target, target_name) == existing_hash
            finally:
                await _delete_quietly(source, target)
