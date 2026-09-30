"""Unit tests for γ's FalkorDB index-provisioning wiring (task 3708).

What this module pins
---------------------
The WIRING of β's ``GraphitiBackend.ensure_indices`` into the two places PRD D6
(docs/prds/falkordb-index-provisioning.md) names — the startup sweep and the
first write to a graph — and the D5 registry that scopes both: which graphs are
provisioned, when, under which lock, and what a provisioning failure does (and
does not) do to the write that triggered it.

HAZARD compliance: every test here is mock-driven.  No live FalkorDB, and no
``FalkorDriver`` / ``_MultiTenantFalkorDriver`` construction or
``GraphitiBackend.initialize()`` call anywhere — ``FalkorDriver.__init__``
fire-and-forgets index creation when an event loop is running, and
``initialize()``'s identity scan REPAIR-writes every graph on the server.  The
live lane lives in ``tests/test_index_provisioning_wiring_integration.py``.
"""

from __future__ import annotations

import asyncio
import logging
from unittest.mock import AsyncMock, MagicMock

import pytest
import redis.exceptions
from test_ensure_indices import _EMPTY_KEY_ERROR, _issued, _ro_issued, _rows_for, _StatefulGraph
from test_falkor_indices import _TRAP_PRESENT, LIVE_HEADER

from fused_memory.backends.falkor_indices import expected_index_set, plan_index_statements
from fused_memory.backends.graphiti_client import GraphitiBackend
from fused_memory.config.schema import TaskmasterConfig
from fused_memory.models.scope import KNOWN_PROJECT_ROOTS_ENV

_LOGGER = 'fused_memory.backends.graphiti_client'


def _plan_for(present: set) -> list[str]:
    """The statements β issues against a graph that already carries *present*."""
    return [statement for statement, _specs in plan_index_statements(expected_index_set() - present)]


def _route(backend, graphs: dict, listing: list[str]) -> None:
    """Resolve each graph key to its own mock, and point the RAW listing at *listing*."""
    backend._driver._get_graph = MagicMock(side_effect=graphs.__getitem__)
    backend._driver.client.list_graphs = AsyncMock(return_value=listing)


def _requested_graphs(backend) -> set[str]:
    return {c.args[0] for c in backend._driver._get_graph.call_args_list}


class TestRegisteredGraphIds:
    """PRD D5: the provisioning scope is the project registry, not a graph name."""

    @pytest.fixture
    def roots(self, tmp_path):
        """Three project roots with no manifest, so their ids derive from basenames."""
        made = {name: tmp_path / name for name in ('Primary-Root', 'Alpha-Proj', 'beta')}
        for path in made.values():
            path.mkdir()
        return made

    def test_default_registry_is_taskmaster_root_plus_known_project_roots_env(
        self, mock_config, roots, monkeypatch,
    ):
        monkeypatch.setenv(
            KNOWN_PROJECT_ROOTS_ENV, f"{roots['Alpha-Proj']},{roots['beta']}",
        )
        config = mock_config.model_copy(deep=True)
        config.taskmaster = TaskmasterConfig(project_root=str(roots['Primary-Root']))

        registered = GraphitiBackend(config).registered_graph_ids

        assert registered == frozenset({'primary_root', 'alpha_proj', 'beta'})
        assert isinstance(registered, frozenset)

    def test_without_a_taskmaster_section_the_registry_is_the_env_roots_only(
        self, mock_config, roots, monkeypatch,
    ):
        monkeypatch.setenv(KNOWN_PROJECT_ROOTS_ENV, str(roots['Alpha-Proj']))
        config = mock_config.model_copy(deep=True)
        config.taskmaster = None

        assert GraphitiBackend(config).registered_graph_ids == frozenset({'alpha_proj'})

    def test_an_injected_registry_wins_over_the_environment_and_is_canonicalized(
        self, mock_config, roots, monkeypatch,
    ):
        monkeypatch.setenv(KNOWN_PROJECT_ROOTS_ENV, str(roots['Alpha-Proj']))

        backend = GraphitiBackend(mock_config, registered_graph_ids=['My-Project'])

        assert backend.registered_graph_ids == frozenset({'my_project'})


class TestStartupSweep:
    """PRD D6, startup half: provision every registered graph that already exists."""

    @pytest.mark.asyncio
    async def test_only_registered_existing_graphs_are_provisioned(
        self, mock_config, make_backend, make_graph_mock,
    ):
        backend = make_backend(mock_config, registered_graph_ids={'reg_a', 'reg_absent'})
        graphs = {'reg_a': make_graph_mock(_rows_for(_TRAP_PRESENT), header=LIVE_HEADER)}
        _route(backend, graphs, ['unreg_probe', 'reg_a', 'default_db'])

        await backend.provision_registered_graphs()

        assert _issued(graphs['reg_a']) == _plan_for(_TRAP_PRESENT)
        # A registered but ABSENT graph is left to its first write (D6); an
        # unregistered one is never touched, whatever its name looks like (D5).
        assert _requested_graphs(backend) == {'reg_a'}

    @pytest.mark.asyncio
    async def test_one_graph_raising_does_not_stop_the_sweep(
        self, mock_config, make_backend, make_graph_mock, caplog,
    ):
        backend = make_backend(mock_config, registered_graph_ids={'reg_a', 'reg_b'})
        broken = make_graph_mock([], header=LIVE_HEADER)
        broken.ro_query = AsyncMock(side_effect=redis.exceptions.ConnectionError('down'))
        graphs = {
            'reg_a': broken,
            'reg_b': make_graph_mock(_rows_for(_TRAP_PRESENT), header=LIVE_HEADER),
        }
        _route(backend, graphs, ['reg_a', 'reg_b'])

        with caplog.at_level(logging.WARNING, logger=_LOGGER):
            await backend.provision_registered_graphs()

        assert _issued(graphs['reg_b']) == _plan_for(_TRAP_PRESENT)
        assert any(
            r.name == _LOGGER and r.levelno == logging.WARNING and 'reg_a' in r.getMessage()
            for r in caplog.records
        ), 'a graph whose provisioning raised must be named in a WARNING'

    @pytest.mark.asyncio
    async def test_a_graph_whose_provisioning_raised_is_retried_on_the_next_sweep(
        self, mock_config, make_backend, make_graph_mock,
    ):
        backend = make_backend(mock_config, registered_graph_ids={'reg_a'})
        graph = make_graph_mock(_rows_for(_TRAP_PRESENT), header=LIVE_HEADER)
        healthy_read = graph.ro_query.side_effect('CALL db.indexes()')
        graph.ro_query.side_effect = [redis.exceptions.ConnectionError('blip'), healthy_read]
        _route(backend, {'reg_a': graph}, ['reg_a'])

        await backend.provision_registered_graphs()
        assert _issued(graph) == []

        await backend.provision_registered_graphs()
        assert _issued(graph) == _plan_for(_TRAP_PRESENT)

    @pytest.mark.asyncio
    async def test_a_returned_run_is_cached_even_with_per_statement_failures(
        self, mock_config, make_backend, make_graph_mock,
    ):
        """β already WARNs per rejected statement; δ's detector owns the gap (INV-4)."""
        backend = make_backend(mock_config, registered_graph_ids={'reg_a'})
        graph = make_graph_mock(_rows_for(_TRAP_PRESENT), header=LIVE_HEADER)
        doomed = _plan_for(_TRAP_PRESENT)[0]

        async def _query(statement, *args, **kwargs):
            if statement == doomed:
                raise RuntimeError('mock-rejection')
            return MagicMock(result_set=[], header=LIVE_HEADER)

        graph.query = AsyncMock(side_effect=_query)
        _route(backend, {'reg_a': graph}, ['reg_a'])

        await backend.provision_registered_graphs()
        reads_after_first = len(_ro_issued(graph))
        await backend.provision_registered_graphs()

        assert reads_after_first == 1
        assert len(_ro_issued(graph)) == reads_after_first

    @pytest.mark.asyncio
    async def test_a_second_sweep_over_provisioned_graphs_reads_nothing(
        self, mock_config, make_backend, make_graph_mock,
    ):
        """INV-3: the in-process cache skips only redundant WORK."""
        backend = make_backend(mock_config, registered_graph_ids={'reg_a'})
        graph = make_graph_mock(_rows_for(_TRAP_PRESENT), header=LIVE_HEADER)
        _route(backend, {'reg_a': graph}, ['reg_a'])

        await backend.provision_registered_graphs()
        reads, writes = list(_ro_issued(graph)), list(_issued(graph))
        await backend.provision_registered_graphs()

        assert writes == _plan_for(_TRAP_PRESENT)
        assert _ro_issued(graph) == reads
        assert _issued(graph) == writes

    @pytest.mark.asyncio
    async def test_a_sweep_over_fully_provisioned_graphs_claims_nothing(
        self, mock_config, make_backend, make_graph_mock, caplog,
    ):
        """INV-2: a sweep that changed nothing must not emit a line read as "provisioned"."""
        backend = make_backend(mock_config, registered_graph_ids={'reg_a'})
        graph = make_graph_mock(_rows_for(expected_index_set()), header=LIVE_HEADER)
        _route(backend, {'reg_a': graph}, ['reg_a'])

        with caplog.at_level(logging.DEBUG, logger=_LOGGER):
            await backend.provision_registered_graphs()

        assert _ro_issued(graph) == ['CALL db.indexes()'], 'the graph must have been examined'
        assert [
            r for r in caplog.records if r.name == _LOGGER and r.levelno >= logging.INFO
        ] == []


class TestFirstWriteProvisioning:
    """PRD D6, first-write half: a registered graph is provisioned before its first write.

    Only the graphiti_core client is stubbed (the ``tests/_graphiti_fake.py``
    idiom), so the real ``add_episode`` / ``ensure_entity_node`` bodies run.
    """

    UPSTREAM_RESULT = object()

    def _wire(self, backend, graph, *, listing: list[str]) -> list[str]:
        """Route every graph key to *graph*, stub upstream, and return the shared event log."""
        events: list[str] = []

        async def _upstream(**kwargs):
            events.append('upstream')
            return self.UPSTREAM_RESULT

        client = MagicMock()
        client.add_episode = AsyncMock(side_effect=_upstream)
        backend._client_for = MagicMock(return_value=client)
        backend._driver._get_graph = MagicMock(return_value=graph)
        backend._driver.client.list_graphs = AsyncMock(return_value=listing)

        if isinstance(graph.query, AsyncMock):
            respond = graph.query.side_effect

            def _query(statement, *args, **kwargs):
                events.append(statement)
                return respond(statement, *args, **kwargs)

            graph.query = AsyncMock(side_effect=_query)
        return events

    @staticmethod
    async def _write(backend, group_id='reg'):
        return await backend.add_episode(name='n', content='c', group_id=group_id)

    @pytest.mark.asyncio
    async def test_first_add_episode_on_a_registered_absent_graph_provisions_before_the_write(
        self, mock_config, make_backend, make_graph_mock,
    ):
        backend = make_backend(mock_config, registered_graph_ids={'reg'})
        graph = make_graph_mock([], header=LIVE_HEADER)
        graph.ro_query = AsyncMock(side_effect=_EMPTY_KEY_ERROR)
        events = self._wire(backend, graph, listing=[])

        result = await self._write(backend)

        assert events == _plan_for(set()) + ['upstream']
        assert _ro_issued(graph) == ['CALL db.indexes()'], 'no readiness poll (INV-7)'
        assert result is self.UPSTREAM_RESULT

    @pytest.mark.asyncio
    async def test_add_episode_on_an_unregistered_graph_touches_no_index_state(
        self, mock_config, make_backend, make_graph_mock,
    ):
        backend = make_backend(mock_config, registered_graph_ids={'reg'})
        graph = make_graph_mock(_rows_for(_TRAP_PRESENT), header=LIVE_HEADER)
        events = self._wire(backend, graph, listing=['unreg'])

        await self._write(backend, group_id='unreg')

        graph.ro_query.assert_not_awaited()
        graph.query.assert_not_awaited()
        assert events == ['upstream']

    @pytest.mark.asyncio
    async def test_two_concurrent_first_writes_issue_the_plan_once(
        self, mock_config, make_backend,
    ):
        backend = make_backend(mock_config, registered_graph_ids={'reg'})
        graph = _StatefulGraph(_TRAP_PRESENT)
        self._wire(backend, graph, listing=['reg'])

        await asyncio.gather(self._write(backend), self._write(backend))

        assert graph.issued == _plan_for(_TRAP_PRESENT)

    @pytest.mark.asyncio
    async def test_first_write_under_the_callers_identity_lock_does_not_deadlock(
        self, mock_config, make_backend, make_graph_mock,
    ):
        """The ``MemoryService._execute_graphiti_write`` shape: the caller already holds it."""
        backend = make_backend(mock_config, registered_graph_ids={'reg'})
        graph = make_graph_mock(_rows_for(_TRAP_PRESENT), header=LIVE_HEADER)
        events = self._wire(backend, graph, listing=['reg'])

        async with backend._identity_lock_for('reg'):
            await asyncio.wait_for(self._write(backend), 5)

        assert events == _plan_for(_TRAP_PRESENT) + ['upstream']

    @pytest.mark.asyncio
    async def test_a_provisioning_failure_never_fails_the_write_and_is_retried(
        self, mock_config, make_backend, make_graph_mock, caplog,
    ):
        backend = make_backend(mock_config, registered_graph_ids={'reg'})
        graph = make_graph_mock([], header=LIVE_HEADER)
        graph.ro_query = AsyncMock(side_effect=redis.exceptions.ConnectionError('down'))
        self._wire(backend, graph, listing=['reg'])

        with caplog.at_level(logging.WARNING, logger=_LOGGER):
            result = await self._write(backend)

        assert result is self.UPSTREAM_RESULT
        assert any(
            r.name == _LOGGER and r.levelno == logging.WARNING and "'reg'" in r.getMessage()
            for r in caplog.records
        ), 'the failed provisioning must be named in a WARNING'
        assert graph.ro_query.await_count == 1

        await self._write(backend)

        assert graph.ro_query.await_count == 2, 'an uncached graph must be retried'

    @pytest.mark.asyncio
    async def test_a_hung_provisioning_read_is_bounded_and_releases_the_lock(
        self, mock_config, make_backend, make_graph_mock,
    ):
        config = mock_config.model_copy(deep=True)
        config.queue.backend_write_timeout_seconds = 0.05
        backend = make_backend(config, registered_graph_ids={'reg'})
        graph = make_graph_mock(_rows_for(_TRAP_PRESENT), header=LIVE_HEADER)
        healthy_read = graph.ro_query.side_effect
        never = asyncio.Event()
        reads = 0

        async def _ro_query(statement, *args, **kwargs):
            nonlocal reads
            reads += 1
            if reads == 1:
                await never.wait()
            return healthy_read(statement, *args, **kwargs)

        graph.ro_query = AsyncMock(side_effect=_ro_query)
        events = self._wire(backend, graph, listing=['reg'])

        await asyncio.wait_for(self._write(backend), 2)
        assert events == ['upstream'], 'the write must proceed past a hung provisioning read'

        await asyncio.wait_for(self._write(backend), 2)
        assert events == ['upstream', *_plan_for(_TRAP_PRESENT), 'upstream']

    @pytest.mark.asyncio
    async def test_later_writes_skip_provisioning_once_cached(
        self, mock_config, make_backend, make_graph_mock,
    ):
        backend = make_backend(mock_config, registered_graph_ids={'reg'})
        graph = make_graph_mock(_rows_for(_TRAP_PRESENT), header=LIVE_HEADER)
        self._wire(backend, graph, listing=['reg'])

        await self._write(backend)
        await self._write(backend)
        await self._write(backend)

        assert _ro_issued(graph) == ['CALL db.indexes()']
        assert _issued(graph) == _plan_for(_TRAP_PRESENT)

    @pytest.mark.asyncio
    async def test_ensure_entity_node_on_a_registered_absent_graph_provisions_before_minting(
        self, mock_config, make_backend, make_graph_mock,
    ):
        backend = make_backend(mock_config, registered_graph_ids={'reg'})
        graph = make_graph_mock([], header=LIVE_HEADER)

        async def _ro_query(statement, *args, **kwargs):
            if 'db.indexes' in statement:
                raise _EMPTY_KEY_ERROR
            return MagicMock(result_set=[], header=[])  # the resolve read: no such node

        graph.ro_query = AsyncMock(side_effect=_ro_query)
        self._wire(backend, graph, listing=[])
        backend.client.embedder.create = AsyncMock(return_value=[0.1])
        backend.update_node_embedding = AsyncMock()

        await backend.ensure_entity_node('Some Entity', group_id='reg')

        writes = _issued(graph)
        mint_at = next(i for i, w in enumerate(writes) if w.startswith('CREATE (n:Entity'))
        assert writes[:mint_at] == _plan_for(set())
