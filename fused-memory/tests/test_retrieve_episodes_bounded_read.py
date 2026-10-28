"""GraphitiBackend.retrieve_episodes reads a group's newest episodes in one bounded FalkorDB query.

The unit tests pin the contract at the graph seam with a double that returns up
to ``limit`` rows in the order the test supplies them: the server's selection
is the server's job, and the live test pins it against a throwaway graph that
carries production's RANGE index set, where a string ``<`` over-returns (see
plans/falkordb-resultset-cap-audit.md).
"""

from __future__ import annotations

import contextlib
import re
from datetime import UTC, datetime, timedelta
from typing import Any, NamedTuple
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
from falkordb.asyncio import FalkorDB
from graphiti_core.nodes import EpisodeType, EpisodicNode

from fused_memory.backends.falkor_indices import expected_index_set, range_create_statement
from fused_memory.backends.graphiti_client import GraphitiBackend, _MultiTenantFalkorDriver


class _Result(NamedTuple):
    header: list[list[Any]]
    result_set: list[list[Any]]


class EpisodeGraph:
    """A FalkorDB graph double answering ro_query with up to ``limit`` of its episodes, in list order."""

    def __init__(self, episodes: list[dict]) -> None:
        self.episodes = episodes
        self.calls: list[tuple[str, dict]] = []

    async def ro_query(self, cypher: str, params: dict) -> _Result:
        self.calls.append((cypher, params))
        aliases = re.findall(r'\bAS\s+(\w+)', cypher)
        return _Result(
            header=[[1, alias] for alias in aliases],
            result_set=[
                [episode[alias] for alias in aliases]
                for episode in self.episodes[: params['limit']]
            ],
        )

    async def query(self, cypher: str, params: dict | None = None) -> _Result:
        raise AssertionError('read paths must use ro_query, never query')


def _row(uuid: str, created_at_iso: str, *, group_id: str = 'dark_factory') -> dict:
    return {
        'uuid': uuid,
        'name': f'episode-{uuid}',
        'group_id': group_id,
        'created_at': created_at_iso,
        'source': 'text',
        'source_description': '',
        'content': f'content-{uuid}',
        'valid_at': created_at_iso,
        'entity_edges': [],
    }


def _backend_over(mock_config, make_backend, graph: EpisodeGraph) -> GraphitiBackend:
    backend = make_backend(mock_config)
    backend._driver._get_graph = MagicMock(return_value=graph)
    return backend


async def _uuids(backend: GraphitiBackend, last_n: int) -> list[str]:
    episodes = await backend.retrieve_episodes(group_ids=['dark_factory'], last_n=last_n)
    return [episode.uuid for episode in episodes]


class TestRetrieveEpisodesQuery:
    @pytest.mark.asyncio
    async def test_issues_one_bounded_read_only_query_without_a_range_predicate(
        self, mock_config, make_backend,
    ):
        graph = EpisodeGraph([])
        get_graph = MagicMock(return_value=graph)
        backend = make_backend(mock_config)
        backend._driver._get_graph = get_graph

        await backend.retrieve_episodes(group_ids=['dark_factory'], last_n=5)

        assert len(graph.calls) == 1
        cypher, params = graph.calls[0]
        assert params == {'group_ids': ['dark_factory'], 'limit': 5}
        assert 'ORDER BY created_at DESC, uuid DESC' in cypher
        assert '<' not in cypher
        get_graph.assert_called_once_with('dark_factory')

    @pytest.mark.asyncio
    async def test_rows_decode_into_episodic_nodes(self, mock_config, make_backend):
        graph = EpisodeGraph([_row('uuid-1', '2026-06-22T08:30:00.123456+00:00')])
        backend = _backend_over(mock_config, make_backend, graph)

        [episode] = await backend.retrieve_episodes(group_ids=['dark_factory'], last_n=1)

        assert isinstance(episode, EpisodicNode)
        assert episode.uuid == 'uuid-1'
        assert episode.name == 'episode-uuid-1'
        assert episode.content == 'content-uuid-1'
        assert episode.group_id == 'dark_factory'
        assert episode.source == EpisodeType.text
        assert episode.created_at == datetime(2026, 6, 22, 8, 30, 0, 123456, tzinfo=UTC)
        assert episode.created_at.tzinfo is not None

    @pytest.mark.asyncio
    async def test_an_empty_group_returns_no_episodes_after_one_query(
        self, mock_config, make_backend,
    ):
        graph = EpisodeGraph([])
        backend = _backend_over(mock_config, make_backend, graph)

        assert await backend.retrieve_episodes(group_ids=['dark_factory'], last_n=5) == []
        assert len(graph.calls) == 1


class TestRetrieveEpisodesOrdering:
    @pytest.mark.asyncio
    async def test_returned_window_is_newest_first(self, mock_config, make_backend):
        graph = EpisodeGraph([
            _row('uuid-apr', '2026-04-07T00:00:00+00:00'),
            _row('uuid-jun', '2026-06-22T00:00:00+00:00'),
            _row('uuid-may', '2026-05-17T00:00:00+00:00'),
            _row('uuid-feb', '2026-02-01T00:00:00+00:00'),
            _row('uuid-mar', '2026-03-10T00:00:00+00:00'),
        ])
        backend = _backend_over(mock_config, make_backend, graph)

        assert await _uuids(backend, 5) == [
            'uuid-jun', 'uuid-may', 'uuid-apr', 'uuid-mar', 'uuid-feb',
        ]

    @pytest.mark.asyncio
    async def test_fewer_episodes_than_last_n_returns_them_all_newest_first(
        self, mock_config, make_backend,
    ):
        graph = EpisodeGraph([
            _row('uuid-mar', '2026-03-10T00:00:00+00:00'),
            _row('uuid-jun', '2026-06-22T00:00:00+00:00'),
            _row('uuid-feb', '2026-02-01T00:00:00+00:00'),
        ])
        backend = _backend_over(mock_config, make_backend, graph)

        assert await _uuids(backend, 10) == ['uuid-jun', 'uuid-mar', 'uuid-feb']

    @pytest.mark.asyncio
    @pytest.mark.parametrize('upstream', [
        ['uuid-tie-b', 'uuid-tie-a', 'uuid-recent'],
        ['uuid-tie-a', 'uuid-tie-b', 'uuid-recent'],
    ])
    async def test_a_created_at_tie_breaks_by_uuid_descending_whatever_the_server_order(
        self, mock_config, make_backend, upstream,
    ):
        created_at = {
            'uuid-recent': '2026-06-01T00:00:00+00:00',
            'uuid-tie-a': '2026-05-01T00:00:00+00:00',
            'uuid-tie-b': '2026-05-01T00:00:00+00:00',
        }
        graph = EpisodeGraph([_row(uuid, created_at[uuid]) for uuid in upstream])
        backend = _backend_over(mock_config, make_backend, graph)

        assert await _uuids(backend, 3) == ['uuid-recent', 'uuid-tie-b', 'uuid-tie-a']

    @pytest.mark.asyncio
    async def test_a_naive_stored_created_at_sorts_as_utc(self, mock_config, make_backend):
        graph = EpisodeGraph([
            _row('uuid-aware-early', '2026-01-01T00:00:00+00:00'),
            _row('uuid-naive-mid', '2026-03-01T00:00:00'),
            _row('uuid-aware-late', '2026-06-01T00:00:00+00:00'),
        ])
        backend = _backend_over(mock_config, make_backend, graph)

        assert await _uuids(backend, 3) == [
            'uuid-aware-late', 'uuid-naive-mid', 'uuid-aware-early',
        ]


class TestRetrieveEpisodesLastNBound:
    @pytest.mark.asyncio
    @pytest.mark.parametrize('last_n', [0, -1, 10000, 10001])
    async def test_last_n_outside_the_bound_is_refused_before_any_query(
        self, mock_config, make_backend, last_n,
    ):
        graph = EpisodeGraph([])
        backend = _backend_over(mock_config, make_backend, graph)

        with pytest.raises(ValueError) as excinfo:
            await backend.retrieve_episodes(group_ids=['dark_factory'], last_n=last_n)

        assert f'last_n={last_n}' in str(excinfo.value)
        assert '10000' in str(excinfo.value)
        assert graph.calls == []

    @pytest.mark.asyncio
    async def test_last_n_just_below_the_bound_is_read(self, mock_config, make_backend):
        graph = EpisodeGraph([])
        backend = _backend_over(mock_config, make_backend, graph)

        await backend.retrieve_episodes(group_ids=['dark_factory'], last_n=9999)

        assert [params['limit'] for _, params in graph.calls] == [9999]


# ---------------------------------------------------------------------------
# Live FalkorDB: a throwaway graph with production's RANGE index set
# ---------------------------------------------------------------------------


class LiveGraph(NamedTuple):
    client: FalkorDB
    graph: Any
    name: str


@pytest_asyncio.fixture
async def episode_read_live_graph():
    """Provision a throwaway graph carrying every production RANGE index, yield it, clean up."""
    name = unique_graph_name('7502_episode_read')
    client = FalkorDB(host=FALKOR_HOST, port=FALKOR_PORT)
    graph = client.select_graph(name)
    try:
        for spec in sorted(expected_index_set()):
            if spec[3] == 'RANGE':
                await graph.query(range_create_statement(spec))
        await await_index_operational(graph)
        yield LiveGraph(client=client, graph=graph, name=name)
    finally:
        with contextlib.suppress(Exception):
            await graph.delete()
        with contextlib.suppress(Exception):
            await client.aclose()


_BASE = datetime(2026, 10, 1, tzinfo=UTC)


async def _create_episode(graph, uuid: str, group_id: str, created_at: str) -> None:
    await graph.query(
        'CREATE (:Episodic {uuid: $uuid, name: $uuid, group_id: $group_id, '
        'created_at: $created_at, valid_at: $created_at, source: $source, '
        "source_description: '', content: $uuid, entity_edges: []})",
        {'uuid': uuid, 'group_id': group_id, 'created_at': created_at, 'source': 'text'},
    )


async def _seed(live: LiveGraph) -> list[str]:
    """Seed 30 in-group episodes and 3 newer other-group ones; return the in-group uuids newest first."""
    in_group: dict[str, str] = {}
    for i in range(30):
        uuid = f'0000{i:04x}-0000-4000-8000-000000000000'
        in_group[uuid] = (_BASE + timedelta(minutes=(7 * i) % 30)).isoformat()
        await _create_episode(live.graph, uuid, live.name, in_group[uuid])
    for j in range(3):
        await _create_episode(
            live.graph,
            f'ffff{j:04x}-0000-4000-8000-000000000000',
            'other_project',
            (_BASE + timedelta(hours=1, minutes=j)).isoformat(),
        )
    return sorted(in_group, key=in_group.__getitem__, reverse=True)


def _live_backend(mock_config, live: LiveGraph) -> GraphitiBackend:
    backend = GraphitiBackend(mock_config, registered_graph_ids=())
    backend._driver = _MultiTenantFalkorDriver(falkor_db=live.client, database=live.name)
    return backend


@falkor_skipif()
@pytest.mark.timeout(60)
@pytest.mark.integration
class TestRetrieveEpisodesLiveFalkorDB:
    @pytest.mark.asyncio
    async def test_returns_the_newest_episodes_newest_first(
        self, mock_config, episode_read_live_graph,
    ):
        live = episode_read_live_graph
        newest_first = await _seed(live)

        episodes = await _live_backend(mock_config, live).retrieve_episodes(
            group_ids=[live.name], last_n=5,
        )

        assert [episode.uuid for episode in episodes] == newest_first[:5]

    @pytest.mark.asyncio
    async def test_returns_the_whole_group_and_nothing_outside_it(
        self, mock_config, episode_read_live_graph,
    ):
        live = episode_read_live_graph
        newest_first = await _seed(live)

        episodes = await _live_backend(mock_config, live).retrieve_episodes(
            group_ids=[live.name], last_n=50,
        )

        assert [episode.uuid for episode in episodes] == newest_first
        assert {episode.group_id for episode in episodes} == {live.name}
