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
import contextlib
from dataclasses import dataclass
from enum import IntEnum
from functools import partial
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
    INDEX_DEFINITION_QUERY,
    IndexCatalogUnsettledError,
    await_index_catalog_settled,
    read_index_records,
)
from fused_memory.config.falkordb_endpoints import FalkorEndpoint, scratch_falkordb_from_env
from fused_memory.config.schema import FusedMemoryConfig

REFERENCE_GRAPH = 'evalmem_lme_ref_incumbent_a'
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
    """The restored copy differs from its source."""


class CopyLeftBehind(Exception):
    """A copy that failed verification could not be deleted; the scratch server still holds it."""


class CopyNotSaved(Exception):
    """A verified copy is live on the scratch server, but SAVE failed, so a restart would lose it."""


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
    """Restore *source* under its own name, verify it, and save.

    Leaves a verified copy or none, unless an exception says otherwise:
    ``CopyLeftBehind`` for an unverified copy whose delete failed,
    ``CopyNotSaved`` for a verified copy SAVE did not persist.
    """
    name = require_scratch_name(source.name, checkpoint=GuardCheckpoint.GRAPH_COPY)
    if name in await client.list_graphs() or await client.connection.exists(name):
        raise TargetGraphExists(
            f'the scratch server already holds {name!r}, and a stale seed is never replaced; '
            f'delete it there first (GRAPH.DELETE {name})'
        )
    await client.connection.restore(name, 0, source.payload)
    graph = client.select_graph(name)
    try:
        identity = await _verified_identity(graph, source)
    except Exception as error:
        await _delete_unverified_copy(graph, name, error)
        raise
    try:
        await client.connection.save()
    except RedisError as error:
        raise CopyNotSaved(
            f'restored {name!r} matches its source, but SAVE failed ({type(error).__name__}: '
            f'{error}); the copy is live but a restart would lose it, so run SAVE on the '
            f'scratch server'
        ) from error
    return identity


async def seed_graph(source_client: FalkorDB, target_client: FalkorDB, name: str) -> GraphIdentity:
    return await restore_graph(target_client, await read_source(source_client, name))


async def _verified_identity(graph: AsyncGraph, source: SourceGraph) -> GraphIdentity:
    await await_index_catalog_settled(
        partial(read_index_records, graph),
        graph_name=source.name,
        timeout_s=SETTLE_TIMEOUT_S,
        interval=SETTLE_POLL_INTERVAL_S,
    )
    identity = await _identity(graph, source.name)
    if identity != source.identity:
        raise SeedMismatch(
            f'restored {source.name!r} differs from its source: {_summary(identity)} against '
            f'{_summary(source.identity)}'
        )
    return identity


async def _delete_unverified_copy(graph: AsyncGraph, name: str, cause: Exception) -> None:
    try:
        await graph.delete()
    except Exception as error:
        raise CopyLeftBehind(
            f'{cause}; deleting the unverified copy then failed ({type(error).__name__}: '
            f'{error}), so the scratch server still holds {name!r}: delete it there '
            f'(GRAPH.DELETE {name})'
        ) from cause


async def _identity(graph: AsyncGraph, name: str) -> GraphIdentity:
    topology = await read_topology(graph, name)
    definitions = (await graph.ro_query(INDEX_DEFINITION_QUERY)).result_set
    return GraphIdentity(
        node_count=len(topology.nodes),
        edge_count=len(topology.edges),
        topology_hash=topology_hash(*topology),
        index_definitions=tuple(tuple(row) for row in definitions),
    )


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
    """Each client is closed even when the other's constructor, which dials, raises."""
    async with contextlib.AsyncExitStack() as clients:
        source_client = FalkorDB(
            host=servers.source.host, port=servers.source.port, password=servers.source_password
        )
        clients.push_async_callback(source_client.aclose)
        target_client = FalkorDB(host=servers.target.host, port=servers.target.port)
        clients.push_async_callback(target_client.aclose)
        return await seed_graph(source_client, target_client, name)


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
        return _report(Exit.MISMATCH, f'{error}; the copy was deleted')
    except (CopyLeftBehind, CopyNotSaved) as error:
        return _report(Exit.FAILED, error)
    except (RedisError, OSError) as error:
        return _report(Exit.FAILED, f'{type(error).__name__}: {error}')
    print(f'seeded {name} on {servers.target.uri} from {servers.source.uri}: {_summary(identity)}')
    return Exit.SEEDED


if __name__ == '__main__':
    sys.exit(run_cli(main))
