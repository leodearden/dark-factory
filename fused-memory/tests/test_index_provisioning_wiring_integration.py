"""Live FalkorDB boundary tests for γ's index-provisioning wiring (task 3708).

Requires a running FalkorDB; skipped automatically when one is not reachable.

The mock suite (``tests/test_index_provisioning_wiring.py``) pins WHEN γ calls
β and what it does with the outcome.  Only a live graph shows that the wiring
really leaves a registered graph carrying the full expected index set, judged by
the same summarizer δ's drift detector uses, and that an unregistered graph's
index state really is left byte-for-byte alone (PRD boundary test 8).

HAZARD compliance
-----------------
* Every graph here is a uuid-suffixed scratch graph (``unique_graph_name``),
  deleted in ``finally``.
* ``GraphitiBackend.initialize()`` is NEVER called: its W6-ε identity scan
  REPAIR-writes every graph on the server.
* Every backend is built with an EXPLICIT ``registered_graph_ids`` naming only
  scratch graphs.  Never rely on the derived registry: in an operator shell
  carrying DASHBOARD_KNOWN_PROJECT_ROOTS it names real project graphs.
* ``_MultiTenantFalkorDriver`` is injected into ``backend._driver`` directly; a
  bare ``FalkorDriver`` is never constructed, because its ``__init__``
  fire-and-forgets an index build under a running loop.
"""

from __future__ import annotations

import contextlib
from unittest.mock import MagicMock

import pytest
import pytest_asyncio
from _fm_helpers import (
    FALKOR_HOST,
    FALKOR_PORT,
    await_index_operational,
    falkor_skipif,
    unique_graph_name,
)
from _graphiti_fake import FakeGraphitiClient
from falkordb.asyncio import FalkorDB

from fused_memory.backends.falkor_indices import expected_index_set, normalize_index_records
from fused_memory.backends.graphiti_client import GraphitiBackend, _MultiTenantFalkorDriver
from fused_memory.reconciliation.index_health import summarize_index_health

pytestmark = [
    falkor_skipif(),
    # Provisioning writes ~30 statements and each test waits for them to reach
    # OPERATIONAL; `timeout_method = "thread"` os._exit(1)s the xdist worker, so
    # an under-budget timeout would read as an infrastructure crash.
    pytest.mark.timeout(120),
    pytest.mark.integration,
]


@pytest_asyncio.fixture
async def scratch():
    """Factory for uuid-suffixed throwaway graphs, each torn down in ``finally``.

    ``select_graph`` alone does not create the KEY, so a graph a test never
    writes to stays absent — which is what the first-write tests rely on.
    """
    clients: list[FalkorDB] = []
    graphs: list = []

    def _make(slug: str):
        name = unique_graph_name(f'3708_{slug}')
        client = FalkorDB(host=FALKOR_HOST, port=FALKOR_PORT)
        clients.append(client)
        graph = client.select_graph(name)
        graphs.append(graph)
        return name, graph

    try:
        yield _make
    finally:
        for graph in graphs:
            with contextlib.suppress(Exception):
                await graph.delete()
        for client in clients:
            with contextlib.suppress(Exception):
                await client.aclose()


@pytest_asyncio.fixture
async def live_backend_factory(mock_config):
    """Build backends wired to a REAL driver, with an explicit scratch-only registry."""
    backends: list[GraphitiBackend] = []

    def _make(registered: set[str]) -> GraphitiBackend:
        backend = GraphitiBackend(mock_config, registered_graph_ids=registered)
        backend._driver = _MultiTenantFalkorDriver(host=FALKOR_HOST, port=FALKOR_PORT)
        backends.append(backend)
        return backend

    try:
        yield _make
    finally:
        for backend in backends:
            await backend.close()


async def _seed_trap_state(graph) -> None:
    """An existing graph carrying only the trap indices esc-3375-1 recorded."""
    await graph.query('CREATE (:Probe {seed: 1})')
    await graph.query('CREATE INDEX FOR (n:Entity) ON (n.uuid)')
    await graph.query('CREATE INDEX FOR ()-[e:RELATES_TO]-() ON (e.uuid)')
    await await_index_operational(graph)


async def _normalized_indices(backend: GraphitiBackend, name: str) -> set:
    return normalize_index_records(await backend.list_indices(group_id=name))


async def _missing_indices(backend: GraphitiBackend, name: str) -> list:
    actual = await _normalized_indices(backend, name)
    return summarize_index_health(actual, expected_index_set())['missing']


class TestStartupSweepLive:
    """PRD D6's startup half, against real graphs."""

    @pytest.mark.asyncio
    async def test_startup_sweep_provisions_a_registered_trap_state_graph(
        self, scratch, live_backend_factory,
    ):
        name, graph = scratch('sweep_registered')
        await _seed_trap_state(graph)
        backend = live_backend_factory({name})

        await backend.provision_registered_graphs()
        await await_index_operational(graph)

        assert await _missing_indices(backend, name) == []

    @pytest.mark.asyncio
    async def test_startup_sweep_leaves_an_unregistered_probe_graph_untouched(
        self, scratch, live_backend_factory,
    ):
        """PRD boundary test 8, startup half: the skip is a FILTER decision."""
        probe_name, probe_graph = scratch('probe_e1_gw')
        registered_name, registered_graph = scratch('sweep_other')
        await _seed_trap_state(probe_graph)
        await _seed_trap_state(registered_graph)
        backend = live_backend_factory({registered_name})
        before = await _normalized_indices(backend, probe_name)

        await backend.provision_registered_graphs()
        await await_index_operational(registered_graph)

        assert probe_name in await backend._require_falkor_client().list_graphs(), (
            'the probe graph must exist, or "untouched" would be vacuous'
        )
        assert await _normalized_indices(backend, probe_name) == before
        assert await _missing_indices(backend, registered_name) == [], (
            'the sweep must have run and provisioned the graph it WAS given'
        )


class TestFirstWriteLive:
    """PRD D6's first-write half, against real graphs.

    Only the graphiti_core client is replaced (``FakeGraphitiClient``), so no LLM
    is involved: provisioning precedes the upstream call, and the real driver
    does every index read and write.
    """

    @staticmethod
    def _with_fake_upstream(backend: GraphitiBackend) -> FakeGraphitiClient:
        fake = FakeGraphitiClient()
        backend._client_for = MagicMock(return_value=fake)
        return fake

    @pytest.mark.asyncio
    async def test_first_write_provisions_a_registered_graph_with_no_restart(
        self, scratch, live_backend_factory,
    ):
        """PRD boundary test 7: a newly registered project needs no service restart."""
        name, graph = scratch('first_write')
        backend = live_backend_factory({name})
        assert name not in await backend._require_falkor_client().list_graphs(), (
            'the graph must not exist before its first write'
        )
        fake = self._with_fake_upstream(backend)

        await backend.add_episode(name='n', content='c', group_id=name)
        await await_index_operational(graph)

        assert len(fake.calls) == 1
        assert await _missing_indices(backend, name) == []

    @pytest.mark.asyncio
    async def test_first_write_leaves_an_unregistered_graph_untouched(
        self, scratch, live_backend_factory,
    ):
        """PRD boundary test 8, write half."""
        probe_name, probe_graph = scratch('probe_e1_gw')
        await _seed_trap_state(probe_graph)
        other_name, _ = scratch('first_write_other')
        backend = live_backend_factory({other_name})
        before = await _normalized_indices(backend, probe_name)
        fake = self._with_fake_upstream(backend)

        await backend.add_episode(name='n', content='c', group_id=probe_name)

        assert len(fake.calls) == 1, 'the write itself must still have happened'
        assert await _normalized_indices(backend, probe_name) == before
