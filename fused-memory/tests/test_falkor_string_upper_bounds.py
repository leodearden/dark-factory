"""Every string upper bound fused-memory sends to FalkorDB must be evaluated exactly.

When a RANGE index serves ``<`` or ``<=`` on a string property, FalkorDB
returns a SUPERSET of the matching rows, while ``>`` and ``>=`` stay exact
(task 7543). So each upper bound is built by
``falkor_string_bounds.py::exact_string_upper_bound`` and this module pins,
against a throwaway graph carrying production's RANGE index set, that the
reads using it return exactly the rows a client-side comparison selects. The
measurements and the per-site audit are in plans/falkordb-resultset-cap-audit.md.
"""

from __future__ import annotations

import contextlib
import uuid
from datetime import UTC, datetime
from typing import Any, NamedTuple
from unittest.mock import AsyncMock, MagicMock

import pytest
import pytest_asyncio
from _fm_helpers import FALKOR_HOST, FALKOR_PORT, falkor_skipif, unique_graph_name
from falkordb.asyncio import FalkorDB
from graphiti_core.cross_encoder.client import CrossEncoderClient
from graphiti_core.embedder import EmbedderClient
from graphiti_core.llm_client import LLMClient
from graphiti_core.nodes import EpisodeType

from fused_memory.backends.falkor_indices import expected_index_set, range_create_statement
from fused_memory.backends.falkor_string_bounds import exact_string_upper_bound
from fused_memory.backends.graphiti_client import GraphitiBackend, _MultiTenantFalkorDriver

_WINDOW_START = '2026-10-01T00:00:00Z'
_WINDOW_END = '2026-10-10T00:00:00Z'


class TestExactStringUpperBound:
    def test_inclusive_bound_is_a_concatenated_less_or_equal(self):
        assert exact_string_upper_bound('e.valid_at', 'end', inclusive=True) == (
            "(e.valid_at + '') <= $end"
        )

    def test_exclusive_bound_is_a_concatenated_less_than(self):
        assert exact_string_upper_bound('n.uuid', 'upper', inclusive=False) == (
            "(n.uuid + '') < $upper"
        )

    @pytest.mark.parametrize('param', ['$end', 'end date', ''])
    def test_a_param_that_is_not_a_bare_name_is_refused(self, param):
        with pytest.raises(ValueError) as excinfo:
            exact_string_upper_bound('e.valid_at', param, inclusive=True)

        assert repr(param) in str(excinfo.value)


# ---------------------------------------------------------------------------
# Live FalkorDB: a throwaway graph with production's RANGE index set
# ---------------------------------------------------------------------------


class LiveGraph(NamedTuple):
    client: FalkorDB
    graph: Any
    name: str


@pytest_asyncio.fixture
async def string_bound_live_graph():
    """Provision a throwaway graph carrying every production RANGE index, yield it, clean up."""
    name = unique_graph_name('7543_string_bounds')
    client = FalkorDB(host=FALKOR_HOST, port=FALKOR_PORT)
    with contextlib.suppress(Exception):
        stale = client.select_graph(name)
        await stale.delete()
    graph = client.select_graph(name)
    for spec in sorted(expected_index_set()):
        if spec[3] == 'RANGE':
            await graph.query(range_create_statement(spec))
    try:
        yield LiveGraph(client=client, graph=graph, name=name)
    finally:
        with contextlib.suppress(Exception):
            await graph.delete()
        with contextlib.suppress(Exception):
            await client.aclose()


async def _seed_window_edges(graph) -> dict[str, str]:
    """Create one RELATES_TO edge per day from 2026-09-30 to 2026-11-01; return {uuid: valid_at}."""
    valid_ats = [
        '2026-09-30T12:00:00+00:00',
        *(f'2026-10-{day:02d}T12:00:00+00:00' for day in range(1, 32)),
        '2026-11-01T12:00:00+00:00',
    ]
    seeded: dict[str, str] = {}
    for valid_at in valid_ats:
        edge_uuid = str(uuid.uuid4())
        await graph.query(
            'CREATE (:Entity {uuid: $a})'
            "-[:RELATES_TO {uuid: $u, valid_at: $v, fact: 'f', name: 'n'}]->"
            '(:Entity {uuid: $b})',
            {'a': str(uuid.uuid4()), 'b': str(uuid.uuid4()), 'u': edge_uuid, 'v': valid_at},
        )
        seeded[edge_uuid] = valid_at
    return seeded


def _in_window(seeded: dict[str, str], start: str, end: str) -> set[str]:
    return {edge_uuid for edge_uuid, valid_at in seeded.items() if start <= valid_at <= end}


_WINDOW_QUERY = (
    'MATCH ()-[e:RELATES_TO]->() WHERE e.valid_at >= $start AND '
    + exact_string_upper_bound('e.valid_at', 'end', inclusive=True)
    + ' RETURN e.uuid'
)
_WINDOW_PARAMS = {'start': _WINDOW_START, 'end': _WINDOW_END}


@falkor_skipif()
@pytest.mark.timeout(60)
@pytest.mark.integration
class TestExactStringUpperBoundLiveFalkorDB:
    @pytest.mark.asyncio
    async def test_window_with_exact_upper_bound_matches_ground_truth(
        self, string_bound_live_graph,
    ):
        live = string_bound_live_graph
        seeded = await _seed_window_edges(live.graph)

        result = await live.graph.ro_query(_WINDOW_QUERY, _WINDOW_PARAMS)

        assert {row[0] for row in result.result_set} == _in_window(
            seeded, _WINDOW_START, _WINDOW_END,
        )

    @pytest.mark.asyncio
    async def test_lower_bound_keeps_the_range_index(self, string_bound_live_graph):
        live = string_bound_live_graph
        await _seed_window_edges(live.graph)

        plan = str(await live.graph.explain(_WINDOW_QUERY, _WINDOW_PARAMS))

        assert 'Index Scan' in plan, plan


@falkor_skipif()
@pytest.mark.timeout(60)
@pytest.mark.integration
class TestQueryEdgesByTimeRangeLiveFalkorDB:
    @pytest.mark.asyncio
    async def test_returns_exactly_the_edges_in_the_window(
        self, mock_config, make_backend, string_bound_live_graph,
    ):
        live = string_bound_live_graph
        seeded = await _seed_window_edges(live.graph)
        backend = make_backend(mock_config)
        backend._driver._get_graph = MagicMock(return_value=live.graph)

        edges = await backend.query_edges_by_time_range(
            _WINDOW_START, _WINDOW_END, group_id='test',
        )

        assert {e['uuid'] for e in edges} == _in_window(seeded, _WINDOW_START, _WINDOW_END)
        assert all(_WINDOW_START <= e['valid_at'] <= _WINDOW_END for e in edges)

    @pytest.mark.asyncio
    async def test_both_bounds_are_inclusive(
        self, mock_config, make_backend, string_bound_live_graph,
    ):
        live = string_bound_live_graph
        seeded = await _seed_window_edges(live.graph)
        backend = make_backend(mock_config)
        backend._driver._get_graph = MagicMock(return_value=live.graph)
        start = '2026-10-01T12:00:00+00:00'
        end = '2026-10-05T12:00:00+00:00'

        edges = await backend.query_edges_by_time_range(start, end, group_id='test')

        assert sorted(e['valid_at'] for e in edges) == [
            f'2026-10-{day:02d}T12:00:00+00:00' for day in range(1, 6)
        ]
        assert {e['uuid'] for e in edges} == _in_window(seeded, start, end)


# ---------------------------------------------------------------------------
# graphiti-core's previous-episode read, on add_episode's per-group client
# ---------------------------------------------------------------------------


def _backend_over(mock_config, driver: _MultiTenantFalkorDriver) -> GraphitiBackend:
    backend = GraphitiBackend(mock_config)
    backend._driver = driver
    backend._llm_client = MagicMock(spec=LLMClient)
    backend._embedder = MagicMock(spec=EmbedderClient)
    backend._cross_encoder = MagicMock(spec=CrossEncoderClient)
    return backend


def _per_group_client(mock_config, records: list[dict]):
    """add_episode's client for group 'g', and the query double answering it with *records*."""
    driver = _MultiTenantFalkorDriver(falkor_db=MagicMock(), database='__base__')
    client = _backend_over(mock_config, driver)._client_for('g')
    execute_query = AsyncMock(return_value=(records, [], None))
    client.driver.execute_query = execute_query
    return client, execute_query


def _episode_record(name: str, valid_at: str) -> dict:
    return {
        'uuid': str(uuid.uuid4()),
        'name': name,
        'group_id': 'g',
        'created_at': valid_at,
        'source': 'text',
        'source_description': '',
        'content': name,
        'valid_at': valid_at,
        'entity_edges': [],
    }


class TestPerGroupClientPreviousEpisodeRead:
    @pytest.mark.asyncio
    async def test_valid_at_upper_bound_cannot_be_index_served(self, mock_config):
        client, execute_query = _per_group_client(mock_config, [])
        reference_time = datetime(2026, 9, 12, tzinfo=UTC)

        await client.retrieve_episodes(
            reference_time, last_n=3, group_ids=['g'], source=EpisodeType.text,
        )

        await_args = execute_query.await_args
        assert await_args is not None
        cypher = await_args.args[0]
        assert exact_string_upper_bound('e.valid_at', 'reference_time', inclusive=True) in cypher
        assert 'e.valid_at <= $reference_time' not in cypher
        assert await_args.kwargs['reference_time'] is reference_time
        assert await_args.kwargs['group_ids'] == ['g']
        assert await_args.kwargs['source'] == 'text'
        assert await_args.kwargs['num_episodes'] == 3

    @pytest.mark.asyncio
    async def test_returns_episodes_oldest_first(self, mock_config):
        client, _ = _per_group_client(mock_config, [
            _episode_record('sep-10', '2026-09-10T12:00:00+00:00'),
            _episode_record('sep-05', '2026-09-05T12:00:00+00:00'),
            _episode_record('sep-01', '2026-09-01T12:00:00+00:00'),
        ])

        episodes = await client.retrieve_episodes(
            datetime(2026, 9, 12, tzinfo=UTC), last_n=3, group_ids=['g'], source=EpisodeType.text,
        )

        assert [e.name for e in episodes] == ['sep-01', 'sep-05', 'sep-10']

    @pytest.mark.asyncio
    async def test_saga_is_refused_without_querying(self, mock_config):
        client, execute_query = _per_group_client(mock_config, [])

        with pytest.raises(NotImplementedError, match='saga'):
            await client.retrieve_episodes(
                datetime(2026, 9, 12, tzinfo=UTC), last_n=3, group_ids=['g'], saga='s',
            )

        execute_query.assert_not_awaited()


async def _seed_episodes(live: LiveGraph) -> None:
    episodes = [
        (f'ep-{month:02d}-{day:02d}', f'2026-{month:02d}-{day:02d}T12:00:00+00:00', 'text')
        for month in (8, 9, 10)
        for day in (1, 5, 10, 15, 20, 25)
    ]
    episodes.append(('msg-09-11', '2026-09-11T12:00:00+00:00', 'message'))
    for name, valid_at, source in episodes:
        await live.graph.query(
            'CREATE (:Episodic {uuid: $uuid, name: $name, group_id: $group_id, '
            'created_at: $valid_at, valid_at: $valid_at, source: $source, '
            "source_description: '', content: $name, entity_edges: []})",
            {
                'uuid': str(uuid.uuid4()), 'name': name, 'group_id': live.name,
                'valid_at': valid_at, 'source': source,
            },
        )


async def _previous_episode_names(mock_config, live: LiveGraph, reference_time: datetime) -> list[str]:
    driver = _MultiTenantFalkorDriver(falkor_db=live.client, database=live.name)
    client = _backend_over(mock_config, driver)._client_for(live.name)
    episodes = await client.retrieve_episodes(
        reference_time, last_n=3, group_ids=[live.name], source=EpisodeType.text,
    )
    return [e.name for e in episodes]


@falkor_skipif()
@pytest.mark.timeout(60)
@pytest.mark.integration
class TestPreviousEpisodeReadLiveFalkorDB:
    @pytest.mark.asyncio
    async def test_previous_episodes_precede_the_reference_time(
        self, mock_config, string_bound_live_graph,
    ):
        live = string_bound_live_graph
        await _seed_episodes(live)

        names = await _previous_episode_names(mock_config, live, datetime(2026, 9, 12, tzinfo=UTC))

        assert names == ['ep-09-01', 'ep-09-05', 'ep-09-10']

    @pytest.mark.asyncio
    async def test_reference_time_is_inclusive(self, mock_config, string_bound_live_graph):
        live = string_bound_live_graph
        await _seed_episodes(live)

        names = await _previous_episode_names(
            mock_config, live, datetime(2026, 9, 10, 12, tzinfo=UTC),
        )

        assert names[-1] == 'ep-09-10'

    @pytest.mark.asyncio
    async def test_reference_after_every_episode_returns_the_newest(
        self, mock_config, string_bound_live_graph,
    ):
        live = string_bound_live_graph
        await _seed_episodes(live)

        names = await _previous_episode_names(mock_config, live, datetime(2026, 11, 1, tzinfo=UTC))

        assert names == ['ep-10-15', 'ep-10-20', 'ep-10-25']
