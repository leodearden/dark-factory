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
from _graph_access_fake import HangingRawGraph, wait_forever
from _import_closure import SRC, modules_loaded_by_importing

from fused_memory.backends import paged_read
from fused_memory.backends.graph_access import (
    BudgetedGraph,
    FalkorGraphAccess,
    OpBoundReader,
    paged_ro_query,
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
