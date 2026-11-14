"""The bounded Cypher egress: every raw graph call runs under a named budget.

Every timing assertion runs under an outer ``asyncio.wait_for`` guard, so a
missing budget fails fast with an unnamed TimeoutError instead of freezing
the run.
"""

from __future__ import annotations

import asyncio
from collections.abc import Awaitable
from pathlib import Path
from typing import TypeVar
from unittest.mock import AsyncMock, MagicMock

import pytest
from _fm_falkor_live import FALKOR_HOST, FALKOR_PORT, falkor_skipif
from _graph_access_fake import HangingRawGraph, wait_forever, with_backend_budgets
from _import_closure import SRC, modules_loaded_by_importing
from pydantic import ValidationError

from fused_memory.backends import graphiti_client, paged_read
from fused_memory.backends.graph_access import (
    BudgetedGraph,
    FalkorGraphAccess,
    OpBoundReader,
    paged_ro_query,
)
from fused_memory.backends.graphiti_client import GraphitiBackend
from fused_memory.config.schema import (
    FalkorDBProviderConfig,
    FusedMemoryConfig,
    GraphitiBackendConfig,
    QueueConfig,
)

_T = TypeVar('_T')

_OUTER_GUARD_S = 5.0
_READ_S = 0.05
_WRITE_S = 0.08

_PAGE_TEMPLATE = 'MATCH (n:Entity) RETURN n.uuid ORDER BY n.uuid SKIP {skip} LIMIT {limit}'
_CENSUS = 'MATCH (n:Entity) RETURN count(*)'


async def _guarded(awaitable: Awaitable[_T]) -> _T:
    return await asyncio.wait_for(awaitable, timeout=_OUTER_GUARD_S)


def _budgeted(raw) -> BudgetedGraph:
    return BudgetedGraph(raw, name='g', read_budget_s=_READ_S, write_budget_s=_WRITE_S)


class TestBudgetedGraphBudgets:
    @pytest.mark.asyncio
    @pytest.mark.parametrize(
        ('call', 'budget'),
        [
            (lambda graph: graph.ro_query('MATCH (n) RETURN n', op='probe'), _READ_S),
            (lambda graph: graph.query('CREATE (n)', op='probe'), _WRITE_S),
            (lambda graph: graph.ro_query_compact('MATCH (n) RETURN n', op='probe'), _READ_S),
        ],
        ids=['ro_query-reads', 'query-writes', 'ro_query_compact-reads'],
    )
    async def test_an_overrun_names_the_op_and_the_budget_applied(self, call, budget):
        graph = _budgeted(HangingRawGraph())

        with pytest.raises(TimeoutError) as raised:
            await _guarded(call(graph))

        assert str(raised.value) == f'Graphiti probe timed out after {budget}s'


class TestBudgetedGraphForwardsTheRawCallShape:
    @pytest.mark.asyncio
    @pytest.mark.parametrize('method', ['ro_query', 'query'])
    async def test_no_params_reaches_the_raw_graph_as_one_positional_argument(self, method):
        raw = MagicMock()
        setattr(raw, method, AsyncMock())

        result = await getattr(_budgeted(raw), method)('C', op='x')

        getattr(raw, method).assert_awaited_once_with('C')
        assert result is getattr(raw, method).return_value

    @pytest.mark.asyncio
    @pytest.mark.parametrize('method', ['ro_query', 'query'])
    async def test_params_reach_the_raw_graph_as_a_second_positional_argument(self, method):
        raw = MagicMock()
        setattr(raw, method, AsyncMock())

        result = await getattr(_budgeted(raw), method)('C', {'k': 1}, op='x')

        getattr(raw, method).assert_awaited_once_with('C', {'k': 1})
        assert result is getattr(raw, method).return_value

    @pytest.mark.asyncio
    @pytest.mark.parametrize('method', ['ro_query', 'query'])
    async def test_a_raw_failure_propagates_untouched(self, method):
        raw = MagicMock()
        setattr(raw, method, AsyncMock(side_effect=RuntimeError('boom')))

        with pytest.raises(RuntimeError, match='^boom$'):
            await getattr(_budgeted(raw), method)('C', op='x')

    @pytest.mark.asyncio
    async def test_ro_query_compact_issues_the_compact_read_command_verbatim(self):
        raw = MagicMock()
        reply = [['header'], [[1, 'row']], ['stats']]
        raw.execute_command = AsyncMock(return_value=reply)
        cypher = "MATCH (n:Entity {uuid: 'u'}) RETURN n.name_embedding"

        result = await _budgeted(raw).ro_query_compact(cypher, op='x')

        raw.execute_command.assert_awaited_once_with('GRAPH.RO_QUERY', 'g', cypher, '--compact')
        assert result is reply


class TestFalkorGraphAccess:
    def _access(self, driver) -> FalkorGraphAccess:
        return FalkorGraphAccess(driver, read_budget_s=_READ_S, write_budget_s=_WRITE_S)

    @pytest.mark.asyncio
    async def test_a_graph_is_resolved_by_its_group_id_and_answers_from_it(
        self, make_graph_driver, make_graph_mock,
    ):
        access = self._access(make_graph_driver({'g': make_graph_mock([[7]])}))

        graph = access.graph('g')
        result = await graph.ro_query('MATCH (n) RETURN count(n)', op='x')

        assert graph.name == 'g'
        assert result.result_set == [[7]]

    def test_an_unwired_graph_raises_the_drivers_key_error(self, make_graph_driver):
        access = self._access(make_graph_driver({'g': MagicMock()}))

        with pytest.raises(KeyError, match="no graph wired for 'missing'"):
            access.graph('missing')

    @pytest.mark.asyncio
    async def test_a_resolved_graph_carries_the_budgets(self, make_graph_driver):
        access = self._access(make_graph_driver({'g': HangingRawGraph()}))

        with pytest.raises(TimeoutError) as raised:
            await _guarded(access.graph('g').query('CREATE (n)', op='probe'))

        assert str(raised.value) == f'Graphiti probe timed out after {_WRITE_S}s'

    @pytest.mark.asyncio
    async def test_list_graphs_answers_from_the_connection(self, make_graph_driver):
        driver = make_graph_driver({})
        driver.client.list_graphs = AsyncMock(return_value=['a', 'b'])

        assert await self._access(driver).list_graphs() == ['a', 'b']

    @pytest.mark.asyncio
    async def test_a_hanging_list_graphs_overruns_the_read_budget(self, make_graph_driver):
        driver = make_graph_driver({})
        driver.client.list_graphs = AsyncMock(side_effect=wait_forever)

        with pytest.raises(TimeoutError) as raised:
            await _guarded(self._access(driver).list_graphs())

        assert str(raised.value) == f'Graphiti list_graphs timed out after {_READ_S}s'


class TestTheEgressHoldsOnlyItsDeclaredState:
    @pytest.mark.parametrize('build', [
        pytest.param(lambda: _budgeted(MagicMock()), id='BudgetedGraph'),
        pytest.param(
            lambda: FalkorGraphAccess(MagicMock(), read_budget_s=_READ_S, write_budget_s=_WRITE_S),
            id='FalkorGraphAccess',
        ),
    ])
    def test_an_undeclared_attribute_cannot_be_set(self, build):
        attribute = 'undeclared'

        with pytest.raises(AttributeError):
            setattr(build(), attribute, 1)


class TestPagedRoQuery:
    @pytest.mark.asyncio
    async def test_it_reads_what_the_primitive_reads_over_the_raw_graph(self, make_graph_mock):
        raw = make_graph_mock([[f'u{i}'] for i in range(7)])

        through_handle = await paged_ro_query(
            _budgeted(raw), _PAGE_TEMPLATE, _CENSUS, op='enum', page_size=3,
        )
        through_raw = await paged_read.paged_ro_query(raw, _PAGE_TEMPLATE, _CENSUS, page_size=3)

        assert through_handle == through_raw
        assert through_handle.complete

    @pytest.mark.asyncio
    async def test_a_hanging_page_overruns_the_read_budget_under_the_callers_op(self):
        with pytest.raises(TimeoutError) as raised:
            await _guarded(
                paged_ro_query(_budgeted(HangingRawGraph()), _PAGE_TEMPLATE, _CENSUS, op='enum')
            )

        assert str(raised.value) == f'Graphiti enum timed out after {_READ_S}s'

    @pytest.mark.asyncio
    @pytest.mark.parametrize(
        ('bounds', 'expected_kind'),
        [
            ({'page_size': 3, 'resultset_size': 3}, paged_read.INCOMPLETE_STRUCTURAL_REFUSAL),
            ({'page_size': 2, 'max_pages': 1}, paged_read.INCOMPLETE_PAGE_CAP),
        ],
        ids=['page_size-and-resultset_size', 'max_pages'],
    )
    async def test_the_paging_bounds_reach_the_primitive(
        self, make_graph_mock, bounds, expected_kind,
    ):
        raw = make_graph_mock([[f'u{i}'] for i in range(7)])

        read = await paged_ro_query(_budgeted(raw), _PAGE_TEMPLATE, _CENSUS, op='enum', **bounds)

        assert read.incomplete_kind == expected_kind


class TestOpBoundReader:
    @pytest.mark.asyncio
    async def test_it_reads_through_the_handle_under_its_op(self):
        handle = MagicMock()
        handle.ro_query = AsyncMock()

        result = await OpBoundReader(handle, op='x').ro_query('C', {'k': 1})

        handle.ro_query.assert_awaited_once_with('C', {'k': 1}, op='x')
        assert result is handle.ro_query.return_value

    @pytest.mark.asyncio
    async def test_a_hanging_read_names_the_bound_op(self):
        reader = OpBoundReader(_budgeted(HangingRawGraph()), op='x')

        with pytest.raises(TimeoutError) as raised:
            await _guarded(reader.ro_query('C'))

        assert str(raised.value) == f'Graphiti x timed out after {_READ_S}s'

    def test_it_names_the_graph_it_reads(self):
        assert OpBoundReader(_budgeted(MagicMock()), op='x').name == 'g'


_UNPAGED_CYPHER_METHODS = [
    pytest.param(lambda b: b.retrieve_episodes(['g']), 'retrieve_episodes', _READ_S,
                 id='retrieve_episodes'),
    pytest.param(
        lambda b: b.redact_episode_content('ep', group_id='g', new_content='redacted'),
        'redact_episode_content.read', _READ_S, id='redact_episode_content',
    ),
    pytest.param(
        lambda b: b.reassign_edge('e', 'n2', which_end='source', group_id='g'),
        'reassign_edge', _READ_S, id='reassign_edge',
    ),
    pytest.param(lambda b: b.get_valid_edges_for_node('n', group_id='g'),
                 'get_valid_edges_for_node', _READ_S, id='get_valid_edges_for_node'),
    pytest.param(lambda b: b.count_foreign_relationships('n', group_id='g'),
                 'count_foreign_relationships', _READ_S, id='count_foreign_relationships'),
    pytest.param(lambda b: b.get_connected_entity_uuids('n', group_id='g'),
                 'get_connected_entity_uuids', _READ_S, id='get_connected_entity_uuids'),
    pytest.param(lambda b: b.bulk_remove_edges(['e'], group_id='g'),
                 'bulk_remove_edges', _WRITE_S, id='bulk_remove_edges'),
    pytest.param(lambda b: b.dedup_valid_edges_for_node('n', group_id='g'),
                 'dedup_valid_edges_for_node', _READ_S, id='dedup_valid_edges_for_node'),
    pytest.param(lambda b: b.redirect_node_edges('old', 'new', group_id='g'),
                 'redirect_node_edges', _WRITE_S, id='redirect_node_edges'),
    pytest.param(lambda b: b.redirect_node_mentions('old', 'new', group_id='g'),
                 'redirect_node_mentions', _READ_S, id='redirect_node_mentions'),
    pytest.param(lambda b: b.delete_entity_node('n', group_id='g'),
                 'delete_entity_node', _READ_S, id='delete_entity_node'),
    pytest.param(lambda b: b.get_node_text('n', group_id='g'),
                 'get_node_text', _READ_S, id='get_node_text'),
    pytest.param(lambda b: b.resolve_entity_by_name('Alice', group_id='g'),
                 'resolve_entity_by_name', _READ_S, id='resolve_entity_by_name'),
    pytest.param(lambda b: b.get_nodes_by_exact_name('Alice', group_id='g'),
                 'get_nodes_by_exact_name', _READ_S, id='get_nodes_by_exact_name'),
    pytest.param(lambda b: b.find_duplicate_entity_nodes('Alice', group_id='g'),
                 'find_duplicate_entity_nodes', _READ_S, id='find_duplicate_entity_nodes'),
    pytest.param(
        lambda b: b.find_entity_nodes_by_name_substring('Ali', group_id='g'),
        'find_entity_nodes_by_name_substring', _READ_S,
        id='find_entity_nodes_by_name_substring',
    ),
    pytest.param(lambda b: b.update_node_summary('n', 'summary', group_id='g'),
                 'update_node_summary', _WRITE_S, id='update_node_summary'),
    pytest.param(lambda b: b.update_node_name('n', 'Alice', group_id='g'),
                 'update_node_name', _WRITE_S, id='update_node_name'),
    pytest.param(lambda b: b.get_edge_text('e', group_id='g'),
                 'get_edge_text', _READ_S, id='get_edge_text'),
    pytest.param(lambda b: b.get_edges_by_uuids(['e'], group_id='g'),
                 'get_edges_by_uuids', _READ_S, id='get_edges_by_uuids'),
    pytest.param(lambda b: b.get_edge_invalid_at('e', group_id='g'),
                 'get_edge_invalid_at', _READ_S, id='get_edge_invalid_at'),
    pytest.param(lambda b: b.update_node_embedding('n', [0.5], group_id='g'),
                 'update_node_embedding', _WRITE_S, id='update_node_embedding'),
    pytest.param(lambda b: b.update_edge_embedding('e', [0.5], group_id='g'),
                 'update_edge_embedding', _WRITE_S, id='update_edge_embedding'),
    pytest.param(lambda b: b.list_indices(group_id='g'), 'list_indices', _READ_S,
                 id='list_indices'),
    pytest.param(lambda b: b.ensure_indices(group_id='g'), 'list_indices', _READ_S,
                 id='ensure_indices'),
    pytest.param(lambda b: b.drop_index('Entity', 'name', group_id='g'),
                 'drop_index', _WRITE_S, id='drop_index'),
    pytest.param(
        lambda b: b.drop_vector_index('Entity', 'name_embedding', entity_type='NODE', group_id='g'),
        'drop_vector_index', _WRITE_S, id='drop_vector_index',
    ),
    pytest.param(lambda b: b.node_count('g'), 'node_count', _READ_S, id='node_count'),
]


_PAGED_CYPHER_METHODS = [
    pytest.param(lambda b: b.enumerate_entity_nodes(group_id='g'),
                 'enumerate_entity_nodes', _READ_S, id='enumerate_entity_nodes'),
    pytest.param(lambda b: b.enumerate_ranked_entity_nodes(group_id='g'),
                 'enumerate_ranked_entity_nodes', _READ_S, id='enumerate_ranked_entity_nodes'),
    pytest.param(lambda b: b.enumerate_all_valid_edges(group_id='g'),
                 'enumerate_all_valid_edges', _READ_S, id='enumerate_all_valid_edges'),
    pytest.param(
        lambda b: b.enumerate_valid_edges_mentioning('Alice', group_id='g'),
        'enumerate_valid_edges_mentioning', _READ_S, id='enumerate_valid_edges_mentioning',
    ),
    pytest.param(
        lambda b: b.query_edges_by_time_range(
            '2026-01-01T00:00:00+00:00', '2026-02-01T00:00:00+00:00', group_id='g',
        ),
        'query_edges_by_time_range', _READ_S, id='query_edges_by_time_range',
    ),
    pytest.param(lambda b: b.query_stale_node_embeddings(1536, group_id='g'),
                 'query_stale_node_embeddings', _READ_S, id='query_stale_node_embeddings'),
    pytest.param(lambda b: b.query_stale_edge_embeddings(1536, group_id='g'),
                 'query_stale_edge_embeddings', _READ_S, id='query_stale_edge_embeddings'),
    pytest.param(lambda b: b.entity_uuids_with_prefix('ab', group_id='g'),
                 'entity_uuids_with_prefix', _READ_S, id='entity_uuids_with_prefix'),
    pytest.param(lambda b: b.edge_uuids_with_prefix('ab', group_id='g'),
                 'edge_uuids_with_prefix', _READ_S, id='edge_uuids_with_prefix'),
]


class ReadsNothingWritesHang:
    """A raw graph whose reads find nothing and whose writes wait forever."""

    async def ro_query(self, *_args):
        return MagicMock(result_set=[], header=[])

    def query(self, *args):
        return wait_forever(*args)

    def execute_command(self, *args):
        return wait_forever(*args)


class TestEveryBackendCypherMethodIsBudgeted:
    """Through the production provider, every raw Cypher call overruns a named, configured budget."""

    @pytest.mark.asyncio
    @pytest.mark.parametrize(
        ('call', 'op', 'budget'), _UNPAGED_CYPHER_METHODS + _PAGED_CYPHER_METHODS,
    )
    async def test_a_hanging_graph_fails_on_the_methods_budget(
        self, mock_config, make_backend, make_graph_driver, call, op, budget,
    ):
        driver = make_graph_driver({'g': HangingRawGraph()})
        driver.client.list_graphs = AsyncMock(return_value=['g'])
        backend = make_backend(
            with_backend_budgets(mock_config, read_s=_READ_S, write_s=_WRITE_S), driver=driver,
        )

        with pytest.raises(TimeoutError) as raised:
            await _guarded(call(backend))

        assert str(raised.value) == f'Graphiti {op} timed out after {budget}s'

    @pytest.mark.asyncio
    async def test_minting_an_entity_fails_on_the_write_budget(
        self, mock_config, make_backend, make_graph_driver,
    ):
        driver = make_graph_driver({'g': ReadsNothingWritesHang()})
        backend = make_backend(
            with_backend_budgets(mock_config, read_s=_READ_S, write_s=_WRITE_S), driver=driver,
        )

        with pytest.raises(TimeoutError) as raised:
            await _guarded(backend.ensure_entity_node('Alice', group_id='g'))

        assert str(raised.value) == f'Graphiti _mint_entity_node timed out after {_WRITE_S}s'


class TestTheConnectionHasASocketTimeout:
    """The FalkorDB connection itself is bounded, beyond every named budget."""

    def test_the_code_default_socket_timeout_outlasts_both_code_default_budgets(
        self, code_default_config,
    ):
        config = FusedMemoryConfig()

        assert config.graphiti.falkordb.socket_timeout_seconds > max(
            config.queue.backend_read_timeout_seconds,
            config.queue.backend_write_timeout_seconds,
        )

    @pytest.mark.parametrize('budget_field', [
        'backend_read_timeout_seconds', 'backend_write_timeout_seconds',
    ])
    def test_a_budget_reaching_the_socket_timeout_is_rejected(self, budget_field):
        with pytest.raises(ValidationError) as raised:
            FusedMemoryConfig(
                graphiti=GraphitiBackendConfig(
                    falkordb=FalkorDBProviderConfig(socket_timeout_seconds=130.0),
                ),
                queue=QueueConfig.model_validate({budget_field: 300.0}),
            )

        message = str(raised.value)
        assert 'graphiti.falkordb.socket_timeout_seconds (130.0) must exceed' in message
        assert f'queue.{budget_field} (300.0)' in message

    def test_a_socket_timeout_raised_with_the_budget_is_accepted(self):
        config = FusedMemoryConfig(
            graphiti=GraphitiBackendConfig(
                falkordb=FalkorDBProviderConfig(socket_timeout_seconds=310.0),
            ),
            queue=QueueConfig(backend_write_timeout_seconds=300.0),
        )

        assert config.graphiti.falkordb.socket_timeout_seconds == 310.0

    def test_the_driver_connects_with_the_socket_timeout_and_its_clones_share_it(
        self, monkeypatch,
    ):
        falkor_db = MagicMock()
        monkeypatch.setattr(graphiti_client, 'FalkorDB', falkor_db)

        driver = graphiti_client._MultiTenantFalkorDriver(
            host='h', port=1, password='p', socket_timeout=7.5,
        )

        falkor_db.assert_called_once_with(host='h', port=1, password='p', socket_timeout=7.5)
        assert driver.client is falkor_db.return_value
        assert driver.clone('g').client is driver.client

    @pytest.mark.asyncio
    async def test_initialize_connects_with_the_configured_socket_timeout(
        self, mock_config, monkeypatch,
    ):
        falkor_db = MagicMock()
        monkeypatch.setattr(graphiti_client, 'FalkorDB', falkor_db)
        monkeypatch.setattr(graphiti_client, 'Graphiti', MagicMock(return_value=AsyncMock()))
        mock_config.llm.providers.openai.api_key = ''
        mock_config.graphiti.falkordb.uri = 'redis://falkor-host:6390'
        mock_config.graphiti.falkordb.socket_timeout_seconds = 12.5

        await GraphitiBackend(mock_config, registered_graph_ids=()).initialize(
            skip_maintenance=True,
        )

        falkor_db.assert_called_once_with(
            host='falkor-host',
            port=6390,
            password=mock_config.graphiti.falkordb.password,
            socket_timeout=12.5,
        )

    @falkor_skipif()
    @pytest.mark.asyncio
    async def test_a_live_connection_carries_the_socket_timeout(self):
        driver = graphiti_client._MultiTenantFalkorDriver(
            host=FALKOR_HOST, port=FALKOR_PORT, socket_timeout=12.5,
        )
        try:
            assert driver.client.connection.get_connection_kwargs()['socket_timeout'] == 12.5
        finally:
            await driver.client.aclose()


def test_the_egress_module_loads_no_graphiti_and_no_falkordb():
    probe = modules_loaded_by_importing(
        'fused_memory.backends.graph_access',
        (
            'graphiti_core',
            'graphiti_core.driver.falkordb_driver',
            'fused_memory.backends.graphiti_client',
            'falkordb',
        ),
    )

    assert Path(probe.file).resolve() == SRC / 'fused_memory' / 'backends' / 'graph_access.py'
    assert probe.loaded == ()
