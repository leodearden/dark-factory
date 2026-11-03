"""Live FalkorDB boundary tests for γ's index-provisioning wiring (task 3708).

Requires a running FalkorDB; skipped automatically when one is not reachable.

The mock suite (``tests/test_index_provisioning_wiring.py``) pins WHEN γ calls
β and what it does with the outcome.  Only a live graph shows that the wiring
really leaves a registered graph carrying the full expected index set, judged by
the same summarizer δ's drift detector uses, and that an unregistered graph's
index state really is left byte-for-byte alone (PRD boundary test 8).

Graphs and backends come only from ``_falkor_index_live``, which enforces the
live-FalkorDB HAZARD rules.
"""

from __future__ import annotations

from unittest.mock import MagicMock

import pytest
import pytest_asyncio
from _falkor_index_live import (
    listed_index_specs,
    live_backends,
    missing_production_indices,
    scratch_graphs,
)
from _fm_helpers import await_index_operational, falkor_skipif
from _graphiti_fake import FakeGraphitiClient

from fused_memory.backends.graphiti_client import GraphitiBackend

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
    """A graph a test never writes to stays absent, which the first-write tests rely on."""
    async with scratch_graphs('3708') as make:
        yield make


@pytest_asyncio.fixture
async def live_backend_factory(mock_config):
    async with live_backends(mock_config) as make:
        yield make


async def _seed_trap_state(graph) -> None:
    """An existing graph carrying only the trap indices esc-3375-1 recorded."""
    await graph.query('CREATE (:Probe {seed: 1})')
    await graph.query('CREATE INDEX FOR (n:Entity) ON (n.uuid)')
    await graph.query('CREATE INDEX FOR ()-[e:RELATES_TO]-() ON (e.uuid)')
    await await_index_operational(graph)


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

        assert await missing_production_indices(backend, name) == []

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
        before = await listed_index_specs(backend, probe_name)

        await backend.provision_registered_graphs()
        await await_index_operational(registered_graph)

        assert probe_name in await backend._require_falkor_client().list_graphs(), (
            'the probe graph must exist, or "untouched" would be vacuous'
        )
        assert await listed_index_specs(backend, probe_name) == before
        assert await missing_production_indices(backend, registered_name) == [], (
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
        assert await missing_production_indices(backend, name) == []

    @pytest.mark.asyncio
    async def test_first_write_leaves_an_unregistered_graph_untouched(
        self, scratch, live_backend_factory,
    ):
        """PRD boundary test 8, write half."""
        probe_name, probe_graph = scratch('probe_e1_gw')
        await _seed_trap_state(probe_graph)
        other_name, _ = scratch('first_write_other')
        backend = live_backend_factory({other_name})
        before = await listed_index_specs(backend, probe_name)
        fake = self._with_fake_upstream(backend)

        await backend.add_episode(name='n', content='c', group_id=probe_name)

        assert len(fake.calls) == 1, 'the write itself must still have happened'
        assert await listed_index_specs(backend, probe_name) == before
