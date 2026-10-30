"""Live-FalkorDB fixtures shared by the index-provisioning integration suites.

The live counterpart of ``_falkor_index_doubles``.  Every suite that drives
``GraphitiBackend``'s index provisioning against a real FalkorDB needs the same
three things: throwaway graphs, a backend over a real driver, and a normalized
reading of the indices a graph lists, with the production expected-set check
built on it.  The first two carry the HAZARD rules below, so they live here
once; a hazard fix made to one private copy would never reach the others.

HAZARD rules, enforced here so no caller can forget one:

* Every graph is a uuid-suffixed scratch graph (``unique_graph_name``), deleted
  on exit.  ``select_graph`` alone does not create the KEY, so a graph a test
  never writes to stays absent.
* ``GraphitiBackend.initialize()`` is never called: its W6-ε identity scan
  REPAIR-writes every graph on the server.
* Every backend gets an EXPLICIT ``registered_graph_ids``.  The derived registry
  names real project graphs in an operator shell carrying
  DASHBOARD_KNOWN_PROJECT_ROOTS.
* The driver is a fully constructed ``_MultiTenantFalkorDriver``, never a bare
  ``FalkorDriver``, whose ``__init__`` fire-and-forgets an index build under a
  running loop.  It is handed to ``GraphitiBackend.with_driver``, and
  ``LiveBackends.driver_of`` returns the driver this module holds.
"""

from __future__ import annotations

import contextlib
from collections.abc import AsyncIterator, Callable, Iterable

from _fm_helpers import FALKOR_HOST, FALKOR_PORT, unique_graph_name
from falkordb.asyncio import FalkorDB
from falkordb.asyncio.graph import AsyncGraph

from fused_memory.backends.falkor_indices import (
    IndexSpec,
    expected_index_set,
    normalize_index_records,
)
from fused_memory.backends.graphiti_client import GraphitiBackend, _MultiTenantFalkorDriver
from fused_memory.config.schema import FusedMemoryConfig
from fused_memory.reconciliation.index_health import summarize_index_health

ScratchGraphFactory = Callable[[str], tuple[str, AsyncGraph]]


@contextlib.asynccontextmanager
async def scratch_graphs(task_slug: str) -> AsyncIterator[ScratchGraphFactory]:
    """Yield ``make(slug) -> (name, graph)``; every graph made is deleted on exit.

    *task_slug* embeds the owning task id, so a graph orphaned by a killed run
    traces back to the suite that made it.
    """
    clients: list[FalkorDB] = []
    graphs: list[AsyncGraph] = []

    def make(slug: str) -> tuple[str, AsyncGraph]:
        name = unique_graph_name(f'{task_slug}_{slug}')
        client = FalkorDB(host=FALKOR_HOST, port=FALKOR_PORT)
        clients.append(client)
        graph = client.select_graph(name)
        graphs.append(graph)
        return name, graph

    try:
        yield make
    finally:
        for graph in graphs:
            with contextlib.suppress(Exception):
                await graph.delete()
        for client in clients:
            with contextlib.suppress(Exception):
                await client.aclose()


class LiveBackends:
    """``live_backends``' factory: each call builds a backend over its own live driver; ``driver_of`` returns that driver."""

    def __init__(self, config: FusedMemoryConfig) -> None:
        self._config = config
        self._drivers: dict[GraphitiBackend, _MultiTenantFalkorDriver] = {}

    def __call__(self, registered: Iterable[str]) -> GraphitiBackend:
        driver = _MultiTenantFalkorDriver(host=FALKOR_HOST, port=FALKOR_PORT)
        backend = GraphitiBackend.with_driver(
            self._config, driver, registered_graph_ids=registered,
        )
        self._drivers[backend] = driver
        return backend

    def driver_of(self, backend: GraphitiBackend) -> _MultiTenantFalkorDriver:
        return self._drivers[backend]

    async def close_all(self) -> None:
        for backend in self._drivers:
            await backend.close()


@contextlib.asynccontextmanager
async def live_backends(config: FusedMemoryConfig) -> AsyncIterator[LiveBackends]:
    """Yield a ``LiveBackends`` factory, ``make(registered) -> GraphitiBackend``; all closed on exit."""
    backends = LiveBackends(config)
    try:
        yield backends
    finally:
        await backends.close_all()


async def listed_index_specs(backend: GraphitiBackend, group_id: str) -> set[IndexSpec]:
    """The specs ``CALL db.indexes()`` LISTS, normalized: listed, not necessarily OPERATIONAL or serving."""
    return normalize_index_records(await backend.list_indices(group_id=group_id))


async def missing_production_indices(backend: GraphitiBackend, group_id: str) -> list[IndexSpec]:
    """Expected-but-absent specs, judged by the reader and summarizer δ's detector uses."""
    listed = await listed_index_specs(backend, group_id)
    return summarize_index_health(listed, expected_index_set())['missing']
