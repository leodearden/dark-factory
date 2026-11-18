"""GraphitiBackend.get_edges_by_uuids: the post-write survivor read (task 6137).

The overreach shadow judges an episode's minted edges only after the identity
lock is released, by which time the reconcile sub-passes may have deleted a
duplicate edge or renamed an endpoint, so it reads the survivors back. Its O4
handoff reads them again to drop flags whose edge is gone or retired.
"""
from __future__ import annotations

from types import SimpleNamespace
from typing import Any
from unittest.mock import MagicMock

import pytest


class _ReadOnlyGraph:
    """Records every ro_query; any write path fails the test."""

    def __init__(self, rows: list[list[Any]]) -> None:
        self.rows = rows
        self.ro_calls: list[tuple[str, dict[str, Any]]] = []

    async def ro_query(self, cypher: str, params: dict[str, Any] | None = None) -> Any:
        self.ro_calls.append((cypher, params or {}))
        return SimpleNamespace(result_set=self.rows)

    async def query(self, *args: Any, **kwargs: Any) -> Any:
        raise AssertionError('get_edges_by_uuids is read-only: graph.query must not be called')


def _backend(make_backend, mock_config, rows: list[list[Any]]):
    backend = make_backend(mock_config)
    graph = _ReadOnlyGraph(rows)
    backend._driver._get_graph = MagicMock(return_value=graph)
    return backend, graph


def _row(uuid: str, *, fact: str = '', source: str = 'S', target: str = 'T',
         episodes: tuple[str, ...] | list[str] = (), active: bool = True) -> list[Any]:
    return [uuid, fact or f'fact of {uuid}', source, target, list(episodes), active]


class TestGetEdgesByUuids:
    @pytest.mark.asyncio
    async def test_rows_come_back_as_dicts_in_input_order(self, make_backend, mock_config) -> None:
        backend, _ = _backend(make_backend, mock_config, [
            _row('e2', source='B', target='C', episodes=['ep2']),
            _row('e1', source='A', target='B', episodes=['ep1', 'ep9']),
        ])
        edges = await backend.get_edges_by_uuids(['e1', 'e2'], group_id='test')
        assert edges == [
            {'uuid': 'e1', 'fact': 'fact of e1', 'source_name': 'A', 'target_name': 'B',
             'episodes': ['ep1', 'ep9'], 'active': True},
            {'uuid': 'e2', 'fact': 'fact of e2', 'source_name': 'B', 'target_name': 'C',
             'episodes': ['ep2'], 'active': True},
        ]

    @pytest.mark.asyncio
    async def test_the_uuids_are_a_parameter_never_interpolated(self, make_backend, mock_config) -> None:
        backend, graph = _backend(make_backend, mock_config, [])
        await backend.get_edges_by_uuids(['uuid-alpha', 'uuid-beta'], group_id='test')
        ((cypher, params),) = graph.ro_calls
        assert params == {'uuids': ['uuid-alpha', 'uuid-beta']}
        assert '$uuids' in cypher
        assert 'uuid-alpha' not in cypher and 'uuid-beta' not in cypher

    @pytest.mark.asyncio
    async def test_no_uuids_means_no_query(self, make_backend, mock_config) -> None:
        backend, graph = _backend(make_backend, mock_config, [_row('e1')])
        assert await backend.get_edges_by_uuids([], group_id='test') == []
        assert graph.ro_calls == []

    @pytest.mark.asyncio
    async def test_a_vanished_edge_is_simply_absent(self, make_backend, mock_config) -> None:
        backend, _ = _backend(make_backend, mock_config, [_row('e1')])
        edges = await backend.get_edges_by_uuids(['e1', 'deleted-by-dedup'], group_id='test')
        assert [edge['uuid'] for edge in edges] == ['e1']

    @pytest.mark.asyncio
    async def test_a_retired_edge_is_returned_marked_inactive(self, make_backend, mock_config) -> None:
        backend, _ = _backend(make_backend, mock_config, [_row('live'), _row('retired', active=False)])
        edges = await backend.get_edges_by_uuids(['live', 'retired'], group_id='test')
        assert [(edge['uuid'], edge['active']) for edge in edges] == [('live', True), ('retired', False)]

    @pytest.mark.asyncio
    async def test_duplicate_uuid_rows_collapse_to_the_first(self, make_backend, mock_config) -> None:
        backend, _ = _backend(make_backend, mock_config, [
            _row('e1', fact='first'), _row('e1', fact='second'),
        ])
        edges = await backend.get_edges_by_uuids(['e1'], group_id='test')
        assert [edge['fact'] for edge in edges] == ['first']

    @pytest.mark.asyncio
    async def test_nulls_become_empty(self, make_backend, mock_config) -> None:
        backend, _ = _backend(make_backend, mock_config, [
            ['e1', None, None, None, None, True],
        ])
        (edge,) = await backend.get_edges_by_uuids(['e1'], group_id='test')
        assert edge == {
            'uuid': 'e1', 'fact': '', 'source_name': '', 'target_name': '', 'episodes': [],
            'active': True,
        }
