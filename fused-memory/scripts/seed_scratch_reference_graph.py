#!/usr/bin/env python3
"""Seed one graph onto the scratch FalkorDB from the FalkorDB the config names.

The source is only read: its identity over ``GRAPH.RO_QUERY`` and its value by
``DUMP``. The copy is a ``RESTORE`` under the graph's own name. DUMP/RESTORE is
the primitive because GRAPH.COPY works within one server only, and a graph's name
travels inside its DUMP payload, so a restore under any other name is not a
faithful copy. The copy is refused if the scratch server already holds the name.
Once its indexes are serving it must match the source's topology hash and index
definitions, or it is deleted again. A verified copy is saved, so the RDB-only
scratch server keeps it across restarts.

It does not call the store-mutation preflight. That preflight guards the
shared mem0/Qdrant store, and this script touches neither. It writes FalkorDB
only on the scratch server (``scratch_falkordb_from_env`` refuses production)
and only under an ``evalmem_`` name (``require_scratch_name``). It never
replaces a graph, and its one delete removes the copy it has just restored.
That reasoning dies if the script gains a MemoryService, a Qdrant client, or a
target that can be production.

Usage, from ``fused-memory/``: ``python scripts/seed_scratch_reference_graph.py
[--graph NAME]`` (OPERATIONS.md "The scratch FalkorDB").
"""

from __future__ import annotations

import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[1] / 'src'))

import asyncio
import time
from dataclasses import dataclass
from enum import IntEnum
from typing import Any

from falkordb.asyncio import FalkorDB
from falkordb.asyncio.graph import AsyncGraph
from redis.exceptions import RedisError
from shared.cli_boundary import LoudArgumentParser, run_cli

from fused_memory.arm_harness.scratch_guard import (
    GuardCheckpoint,
    ScratchGuardError,
    require_scratch_name,
)
from fused_memory.arm_harness.topology import read_topology, topology_hash
from fused_memory.backends.falkor_indices import (
    IndexCatalogUnsettledError,
    resolve_header_positions,
    unsettled_index_statuses,
)
from fused_memory.config.falkordb_endpoints import FalkorEndpoint, scratch_falkordb_from_env
from fused_memory.config.schema import FusedMemoryConfig

REFERENCE_GRAPH = 'evalmem_lme_ref_incumbent_a'
INDEX_DEFINITION_QUERY = (
    'CALL db.indexes() YIELD label, properties, types, options, language, stopwords, entitytype '
    'RETURN label, properties, types, options, language, stopwords, entitytype '
    'ORDER BY entitytype, label'
)
INDEX_CATALOG_QUERY = 'CALL db.indexes()'
INDEX_STATUS_COLUMNS = {'label': 'label', 'status': 'status'}
SETTLE_TIMEOUT_S = 120.0
SETTLE_POLL_INTERVAL_S = 0.2


class Exit(IntEnum):
    SEEDED = 0
    FAILED = 1
    REFUSED = 3
    MISMATCH = 4


class SameServerError(RuntimeError):
    """The scratch target is the server the graph would be copied from."""


class SourceGraphMissing(Exception):
    """The source server holds no graph by that name."""


class TargetGraphExists(Exception):
    """The scratch server already holds the name; a stale seed is never replaced."""


class SeedMismatch(Exception):
    """The restored copy differs from its source; it has been deleted again."""


@dataclass(frozen=True)
class GraphIdentity:
    node_count: int
    edge_count: int
    topology_hash: str
    index_definitions: tuple[tuple[Any, ...], ...]


@dataclass(frozen=True)
class SourceGraph:
    name: str
    payload: bytes
    identity: GraphIdentity


@dataclass(frozen=True)
class Servers:
    source: FalkorEndpoint
    source_password: str | None
    target: FalkorEndpoint


async def read_source(client: FalkorDB, name: str) -> SourceGraph:
    """The graph's DUMP value and its identity; issues reads only."""
    require_scratch_name(name, checkpoint=GuardCheckpoint.GRAPH_COPY)
    payload = await client.connection.dump(name)
    if payload is None:
        raise SourceGraphMissing(f'the source server holds no graph {name!r}')
    return SourceGraph(name, payload, await _identity(client.select_graph(name), name))


async def restore_graph(client: FalkorDB, source: SourceGraph) -> GraphIdentity:
    """Restore *source* under its own name, verify it, and save; leaves a verified copy or none."""
    name = require_scratch_name(source.name, checkpoint=GuardCheckpoint.GRAPH_COPY)
    if name in await client.list_graphs() or await client.connection.exists(name):
        raise TargetGraphExists(
            f'the scratch server already holds {name!r}, and a stale seed is never replaced; '
            f'delete it there first (GRAPH.DELETE {name})'
        )
    await client.connection.restore(name, 0, source.payload)
    graph = client.select_graph(name)
    try:
        await _await_indexes_serving(graph, name)
        identity = await _identity(graph, name)
        if identity != source.identity:
            raise SeedMismatch(
                f'restored {name!r} differs from its source: {_summary(identity)} against '
                f'{_summary(source.identity)}; the copy was deleted'
            )
    except Exception:
        await graph.delete()
        raise
    await client.connection.save()
    return identity


async def seed_graph(source_client: FalkorDB, target_client: FalkorDB, name: str) -> GraphIdentity:
    return await restore_graph(target_client, await read_source(source_client, name))


async def _identity(graph: AsyncGraph, name: str) -> GraphIdentity:
    topology = await read_topology(graph, name)
    definitions = (await graph.ro_query(INDEX_DEFINITION_QUERY)).result_set
    return GraphIdentity(
        node_count=len(topology.nodes),
        edge_count=len(topology.edges),
        topology_hash=topology_hash(*topology),
        index_definitions=tuple(tuple(row) for row in definitions),
    )


async def _await_indexes_serving(graph: AsyncGraph, name: str) -> None:
    deadline = time.monotonic() + SETTLE_TIMEOUT_S
    while True:
        response = await graph.ro_query(INDEX_CATALOG_QUERY)
        positions = resolve_header_positions(response.header, INDEX_STATUS_COLUMNS)
        unsettled = unsettled_index_statuses(
            {key: row[position] for key, position in positions.items()}
            for row in response.result_set
        )
        if not unsettled:
            return
        if time.monotonic() >= deadline:
            raise IndexCatalogUnsettledError(
                f'restored {name!r} still has indexes that are not OPERATIONAL after '
                f'{SETTLE_TIMEOUT_S:.0f}s: {unsettled!r}; the copy was deleted'
            )
        await asyncio.sleep(SETTLE_POLL_INTERVAL_S)


def _summary(identity: GraphIdentity) -> str:
    return (
        f'{identity.node_count} nodes, {identity.edge_count} edges, '
        f'{len(identity.index_definitions)} index definitions, '
        f'topology {identity.topology_hash[:12]}'
    )


def resolve_servers() -> Servers:
    """The server the config names as the source, and the scratch target, which is neither it nor production."""
    configured = FusedMemoryConfig().graphiti.falkordb
    source = FalkorEndpoint.from_uri(configured.uri)
    target = scratch_falkordb_from_env()
    if target == source:
        raise SameServerError(
            f'source and target are the same server ({source.uri}); point FALKORDB_URI at the '
            f'server that holds the graph and FALKOR_HOST/FALKOR_PORT at the scratch server'
        )
    return Servers(source, configured.password or None, target)


async def _seed(servers: Servers, name: str) -> GraphIdentity:
    source_client = FalkorDB(
        host=servers.source.host, port=servers.source.port, password=servers.source_password
    )
    target_client = FalkorDB(host=servers.target.host, port=servers.target.port)
    try:
        return await seed_graph(source_client, target_client, name)
    finally:
        await source_client.aclose()
        await target_client.aclose()


def build_parser() -> LoudArgumentParser:
    parser = LoudArgumentParser(description=__doc__.splitlines()[0] if __doc__ else None)
    parser.add_argument(
        '--graph', default=REFERENCE_GRAPH, help='the graph to copy (default: %(default)s)'
    )
    return parser


def _report(status: Exit, error: object) -> int:
    print(f'{status.name.lower()}: {error}', file=sys.stderr)
    return status


def main(argv: list[str] | None = None) -> int:
    name = build_parser().parse_args(argv).graph
    try:
        require_scratch_name(name, checkpoint=GuardCheckpoint.GRAPH_COPY)
        servers = resolve_servers()
    except (ScratchGuardError, RuntimeError) as error:
        return _report(Exit.REFUSED, error)
    except ValueError as error:
        return _report(Exit.FAILED, error)
    try:
        identity = asyncio.run(_seed(servers, name))
    except (SourceGraphMissing, TargetGraphExists) as error:
        return _report(Exit.REFUSED, error)
    except (SeedMismatch, IndexCatalogUnsettledError) as error:
        return _report(Exit.MISMATCH, error)
    except (RedisError, OSError) as error:
        return _report(Exit.FAILED, f'{type(error).__name__}: {error}')
    print(f'seeded {name} on {servers.target.uri} from {servers.source.uri}: {_summary(identity)}')
    return Exit.SEEDED


if __name__ == '__main__':
    sys.exit(run_cli(main))
