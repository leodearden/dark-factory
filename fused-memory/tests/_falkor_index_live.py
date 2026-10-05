"""Live-FalkorDB fixtures shared by the index-provisioning integration suites.

The live counterpart of ``_falkor_index_doubles``.  Every suite that drives
``GraphitiBackend``'s index provisioning against a real FalkorDB needs the same
three things: throwaway graphs, a backend over a real driver, and the production
expected-set check.  The first two carry the HAZARD rules below, so they live
here once; a hazard fix made to one private copy would never reach the others.

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
  running loop.  ``GraphitiBackend`` accepts no driver publicly, so it is
  written into ``backend._driver``; ``injected_driver`` is the matching read.
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
LiveBackendFactory = Callable[[Iterable[str]], GraphitiBackend]


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


@contextlib.asynccontextmanager
async def live_backends(config: FusedMemoryConfig) -> AsyncIterator[LiveBackendFactory]:
    """Yield ``make(registered) -> GraphitiBackend`` over a real driver; all closed on exit."""
    backends: list[GraphitiBackend] = []

    def make(registered: Iterable[str]) -> GraphitiBackend:
        backend = GraphitiBackend(config, registered_graph_ids=registered)
        backend._driver = _MultiTenantFalkorDriver(host=FALKOR_HOST, port=FALKOR_PORT)
        backends.append(backend)
        return backend

    try:
        yield make
    finally:
        for backend in backends:
            await backend.close()


def injected_driver(backend: GraphitiBackend) -> _MultiTenantFalkorDriver:
    """The driver ``live_backends`` gave *backend*; any other backend raises."""
    driver = backend._driver
    if not isinstance(driver, _MultiTenantFalkorDriver):
        raise TypeError(f'backend carries no live_backends driver: {driver!r}')
    return driver


async def missing_production_indices(backend: GraphitiBackend, group_id: str) -> list[IndexSpec]:
    """Expected-but-absent specs, judged by the reader and summarizer δ's detector uses."""
    actual = normalize_index_records(await backend.list_indices(group_id=group_id))
    return summarize_index_health(actual, expected_index_set())['missing']
