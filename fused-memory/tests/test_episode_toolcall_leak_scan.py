"""Tests for the Graphiti-side tool-call leak discovery read (task 3233).

``fused_memory.backends.episode_toolcall_leak_scan.read_episode_toolcall_leaks``
finds which Episodic nodes in one group carry a leaked serialized tool-call
fragment: a server-side CONTAINS prefilter narrows the transfer, the shared
detector gives the verdict, and the read is paged and census-proven.

The fake raw graph below models only what the read issues. One live test pins
the FalkorDB CONTAINS/any() prefilter semantics the fake cannot.

SENTINEL-LITERAL HAZARD (task 3083): every tool-call sentinel here is spelled
with the ``\\x3c`` escape for ``<``. A verbatim one would truncate the authoring
tool call that writes this file.
"""
from __future__ import annotations

import contextlib
import dataclasses
import re
from typing import Any

import pytest
import pytest_asyncio
from _fm_falkor_live import FALKOR_HOST, FALKOR_PORT, falkor_skipif, unique_graph_name
from _graph_access_fake import FakeGraphAccess, HangingRawGraph
from falkordb.asyncio import FalkorDB

from fused_memory.backends.episode_toolcall_leak_scan import (
    EpisodeToolcallLeak,
    EpisodeToolcallLeakScan,
    read_episode_toolcall_leaks,
)
from fused_memory.backends.graph_access import BudgetedGraph
from fused_memory.backends.paged_read import INCOMPLETE_SHORT_READ
from fused_memory.utils.toolcall_xml_leak import PREFILTER_NEEDLES, find_toolcall_xml_leak

GROUP = 'dark_factory'

_CLOSE_CONTENT = '\x3c/content>'
_CLOSE_INVOKE = '\x3c/invoke>'


def _open_param(name: str) -> str:
    return '\x3cparameter name="' + name + '">'


def _tail_leak(body: str) -> str:
    """The shape of live episodes 4d88a40f/72c293b3/1f588c77 (measured 2026-11-25)."""
    return body + _CLOSE_CONTENT + '\n' + _CLOSE_INVOKE


def _near_miss(body: str) -> str:
    """A closing content tag followed by prose: no continuation, so no leak."""
    return body + _CLOSE_CONTENT + ' and then the prose simply carries on.'


def _row(uuid: str, content: str) -> list[Any]:
    return [uuid, f'episode {uuid}', f'2026-07-2{len(uuid) % 10}T00:00:00Z', content]


class _RawResult:
    def __init__(self, rows: list[list[Any]]) -> None:
        self.result_set = rows


class EpisodeGraphDouble:
    """A raw falkordb-shaped graph serving one group's Episodic population.

    ``population`` answers the plain ``count(*)``, ``candidate_census`` the
    CONTAINS ``count(*)``; page queries are sliced out of ``rows`` by the
    ``SKIP n LIMIT m`` in their text. A None count answers with no row.
    """

    def __init__(
        self,
        *,
        rows: list[list[Any]],
        population: int | None,
        candidate_census: int | None = None,
    ) -> None:
        self.rows = rows
        self.population = population
        self.candidate_census = len(rows) if candidate_census is None else candidate_census
        self.calls: list[tuple[str, dict[str, Any] | None]] = []

    async def ro_query(self, cypher: str, params: dict[str, Any] | None = None) -> _RawResult:
        self.calls.append((cypher, params))
        if 'count(*)' in cypher:
            value = self.candidate_census if 'CONTAINS' in cypher else self.population
            return _RawResult([] if value is None else [[value]])
        bounds = re.search(r'SKIP (\d+) LIMIT (\d+)', cypher)
        assert bounds is not None, f'unrecognised query: {cypher!r}'
        skip, limit = int(bounds[1]), int(bounds[2])
        return _RawResult([list(row) for row in self.rows[skip:skip + limit]])

    def candidate_calls(self) -> list[tuple[str, dict[str, Any] | None]]:
        return [call for call in self.calls if 'CONTAINS' in call[0]]

    def page_calls(self) -> list[tuple[str, dict[str, Any] | None]]:
        return [call for call in self.calls if 'count(*)' not in call[0]]


def _handle(raw: Any, **budgets: float) -> Any:
    return FakeGraphAccess({GROUP: raw}, **budgets).graph(GROUP)


async def _scan(raw: Any, **kwargs: Any) -> EpisodeToolcallLeakScan:
    return await read_episode_toolcall_leaks(_handle(raw), group_id=GROUP, **kwargs)


class TestOnlyConfirmedLeaksComeBack:
    @pytest.mark.asyncio
    async def test_the_shared_detector_is_the_verdict(self) -> None:
        leak_content = _tail_leak('A decision body.')
        double = EpisodeGraphDouble(
            rows=[_row('a-leak', leak_content), _row('b-near', _near_miss('Prose body.'))],
            population=40,
        )

        scan = await _scan(double)

        hit = find_toolcall_xml_leak(leak_content)[0]
        assert scan.leaks == (
            EpisodeToolcallLeak(
                uuid='a-leak',
                name='episode a-leak',
                created_at=_row('a-leak', leak_content)[2],
                content=leak_content,
                fragment=hit.fragment,
                index=hit.index,
            ),
        )


class TestQueryShape:
    @pytest.mark.asyncio
    async def test_needles_and_group_are_bound_never_spelled(self) -> None:
        double = EpisodeGraphDouble(rows=[_row('a', _tail_leak('Body.'))], population=3)

        await _scan(double)

        candidate_calls = double.candidate_calls()
        assert len(candidate_calls) >= 2  # the census and at least one page
        for cypher, params in candidate_calls:
            assert params == {'group_id': GROUP, 'needles': list(PREFILTER_NEEDLES)}
            assert '$needles' in cypher
            assert '$group_id' in cypher
        for cypher, _ in double.calls:
            assert '\x3c' not in cypher
        for cypher, _ in double.page_calls():
            assert 'ORDER BY e.uuid' in cypher


class TestPagingAndCompleteness:
    @pytest.mark.asyncio
    async def test_every_leak_across_pages_is_found(self) -> None:
        rows = [_row(f'u{i}', _tail_leak(f'Body {i}.')) for i in range(5)]
        double = EpisodeGraphDouble(rows=rows, population=50)

        scan = await _scan(double, page_size=2)

        assert [leak.uuid for leak in scan.leaks] == [f'u{i}' for i in range(5)]
        assert scan.candidates.complete is True
        assert len(double.page_calls()) == 3

    @pytest.mark.asyncio
    async def test_a_boundary_row_re_emitted_on_two_pages_yields_one_leak(self) -> None:
        boundary = _row('b', _tail_leak('Boundary body.'))
        rows = [_row('a', _near_miss('A.')), boundary, list(boundary), _row('c', 'clean')]
        double = EpisodeGraphDouble(rows=rows, population=10)

        scan = await _scan(double, page_size=2)

        assert [leak.uuid for leak in scan.leaks] == ['b']

    @pytest.mark.asyncio
    async def test_a_short_read_is_reported_and_its_leaks_still_returned(self) -> None:
        rows = [_row(f'u{i}', _tail_leak(f'Body {i}.')) for i in range(3)]
        double = EpisodeGraphDouble(rows=rows, population=50, candidate_census=10)

        scan = await _scan(double)

        assert scan.candidates.complete is False
        assert scan.candidates.incomplete_kind == INCOMPLETE_SHORT_READ
        assert [leak.uuid for leak in scan.leaks] == ['u0', 'u1', 'u2']


class TestPopulationDenominator:
    @pytest.mark.asyncio
    async def test_episodes_in_group_is_the_population_census(self) -> None:
        double = EpisodeGraphDouble(rows=[], population=14456)

        scan = await _scan(double)

        assert scan.group_id == GROUP
        assert scan.episodes_in_group == 14456
        population_calls = [
            call for call in double.calls
            if 'count(*)' in call[0] and 'CONTAINS' not in call[0]
        ]
        assert population_calls == [(population_calls[0][0], {'group_id': GROUP})]

    @pytest.mark.asyncio
    async def test_an_unusable_population_census_is_none(self) -> None:
        double = EpisodeGraphDouble(rows=[], population=None)

        scan = await _scan(double)

        assert scan.episodes_in_group is None


class TestFailsLoud:
    @pytest.mark.asyncio
    async def test_a_read_that_never_answers_raises_timeout(self) -> None:
        handle = _handle(HangingRawGraph(), read_budget_s=0.05, write_budget_s=0.05)

        with pytest.raises(TimeoutError):
            await read_episode_toolcall_leaks(handle, group_id=GROUP)


class TestImmutability:
    @pytest.mark.asyncio
    async def test_scan_and_leaks_are_frozen(self) -> None:
        double = EpisodeGraphDouble(rows=[_row('a', _tail_leak('Body.'))], population=1)

        scan = await _scan(double)

        with pytest.raises(dataclasses.FrozenInstanceError):
            scan.episodes_in_group = 0  # type: ignore[misc]
        with pytest.raises(dataclasses.FrozenInstanceError):
            scan.leaks[0].content = 'repaired'  # type: ignore[misc]


# ---------------------------------------------------------------------------
# Live semantic pin: FalkorDB's CONTAINS/any() prefilter over a bound list.
# ---------------------------------------------------------------------------


@pytest_asyncio.fixture
async def leak_scan_live_graph():
    """Provision a throwaway, uniquely-named FalkorDB graph, yield it, clean up."""
    graph_name = unique_graph_name('3233_episode_leak_scan')
    client = FalkorDB(host=FALKOR_HOST, port=FALKOR_PORT)
    with contextlib.suppress(Exception):
        stale = client.select_graph(graph_name)
        await stale.delete()
    graph = client.select_graph(graph_name)
    try:
        yield graph_name, graph
    finally:
        with contextlib.suppress(Exception):
            await graph.delete()
        with contextlib.suppress(Exception):
            await client.aclose()


_LIVE_SEED = (
    'UNWIND $episodes AS ep '
    'CREATE (:Episodic {uuid: ep.uuid, name: ep.name, group_id: ep.group_id, '
    'created_at: ep.created_at, content: ep.content})'
)


@falkor_skipif()
@pytest.mark.timeout(60)
@pytest.mark.integration
class TestPrefilterSemanticsLiveFalkorDB:
    @pytest.mark.asyncio
    async def test_only_the_in_group_leak_is_found(self, leak_scan_live_graph) -> None:
        graph_name, graph = leak_scan_live_graph
        episodes = [
            ('in-leak', 'g', _tail_leak('In-group decision body.')),
            ('in-near', 'g', _near_miss('In-group prose body.')),
            ('in-clean', 'g', 'A clean in-group episode.'),
            ('out-leak', 'other', _tail_leak('Foreign-group body.')),
        ]
        await graph.query(_LIVE_SEED, {'episodes': [
            {'uuid': uuid, 'name': uuid, 'group_id': group, 'created_at': '2026-11-25T00:00:00Z',
             'content': content}
            for uuid, group, content in episodes
        ]})
        handle = BudgetedGraph(graph, name=graph_name, read_budget_s=30.0, write_budget_s=30.0)

        scan = await read_episode_toolcall_leaks(handle, group_id='g')

        assert [leak.uuid for leak in scan.leaks] == ['in-leak']
        assert scan.episodes_in_group == 3
        assert scan.candidates.expected_rows == 2
        assert scan.candidates.complete is True
