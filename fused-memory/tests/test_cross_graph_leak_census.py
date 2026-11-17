"""The read-only cross-graph leak census (task 6849).

A node whose ``group_id`` names another graph, or no graph, is task 2115's
cross-graph leak, and every group-scoped read is blind to it. The census counts
those nodes per graph and per label, and says whether its count is whole.

Read-only is asserted structurally, against the raw graph's writable channels,
and the live class pins the real Cypher's NULL semantics and label scoping
against a scratch FalkorDB.
"""
from __future__ import annotations

import contextlib
import logging
import re
from collections import Counter
from collections.abc import Iterable
from dataclasses import FrozenInstanceError, fields
from typing import Any, NamedTuple
from unittest.mock import AsyncMock

import pytest
import pytest_asyncio
from _fm_falkor_live import FALKOR_HOST, FALKOR_PORT, falkor_skipif, unique_graph_name
from _graph_access_fake import FakeGraphAccess
from falkordb.asyncio import FalkorDB

from fused_memory.backends.paged_read import (
    INCOMPLETE_CENSUS_UNAVAILABLE,
    INCOMPLETE_SHORT_READ,
)
from fused_memory.maintenance.cross_graph_leak_census import (
    LEAK_LABELS,
    CrossGraphLeakCensus,
    ForeignGroup,
    GraphLeaks,
    LabelLeak,
)

_CENSUS_LOGGER = 'fused_memory.maintenance.cross_graph_leak_census'
_LABEL_PATTERN = re.compile(r'\(n:(\w+)\)')


class _Result(NamedTuple):
    result_set: list[list[Any]]


def label_in(cypher: str) -> str:
    """The label of the ``(n:<Label>)`` pattern *cypher* matches."""
    found = _LABEL_PATTERN.search(cypher)
    assert found is not None, cypher
    return found.group(1)


class LeakGraph:
    """A raw graph double holding ``(label, group_id-or-None)`` nodes.

    ``ro_query`` answers a ``RETURN count(n)`` projection with the leaked total
    and any other projection with grouped ``[group_id, count]`` rows, for the
    label in the ``(n:<Label>)`` pattern and the key in ``params['graph_key']``.
    ``count_override`` and ``breakdown_override`` replace one label's answer.
    The writable channels are AsyncMocks so a test can assert none was awaited.
    """

    def __init__(
        self,
        nodes: Iterable[tuple[str, str | None]],
        *,
        count_override: dict[str, list[list[Any]]] | None = None,
        breakdown_override: dict[str, list[list[Any]]] | None = None,
    ) -> None:
        self.nodes = list(nodes)
        self.count_override = count_override or {}
        self.breakdown_override = breakdown_override or {}
        self.ro_calls: list[tuple[str, dict[str, Any] | None]] = []
        self.query = AsyncMock()
        self.delete = AsyncMock()
        self.execute_command = AsyncMock()

    async def ro_query(self, cypher: str, params: dict[str, Any] | None = None) -> _Result:
        self.ro_calls.append((cypher, params))
        assert params is not None, cypher
        label = label_in(cypher)
        leaked = Counter(
            group_id for node_label, group_id in self.nodes
            if node_label == label and group_id != params['graph_key']
        )
        if 'RETURN count(n)' in cypher:
            return _Result(self.count_override.get(label, [[sum(leaked.values())]]))
        return _Result(self.breakdown_override.get(
            label, [[group_id, nodes] for group_id, nodes in leaked.items()],
        ))


def nodes_of(label: str, group_id: str | None, count: int) -> list[tuple[str, str | None]]:
    return [(label, group_id)] * count


#: Graph 'home': same-group nodes of both labels, plus foreign ones. The foreign
#: Entity groups arrive smallest-first so only the census's sort can order them.
HOME_NODES = [
    *nodes_of('Entity', 'home', 3),
    *nodes_of('Entity', 'dark-factory', 1),
    *nodes_of('Entity', 'reify', 2),
    *nodes_of('Episodic', 'home', 2),
    *nodes_of('Episodic', 'know_live', 4),
]

CLEAN_NODES = [*nodes_of('Entity', 'home', 5), *nodes_of('Episodic', 'home', 2)]


async def census_of(raw: LeakGraph, group_id: str = 'home') -> GraphLeaks:
    return await CrossGraphLeakCensus(graphs=FakeGraphAccess({group_id: raw})).run(
        group_id=group_id,
    )


def label_of(leaks: GraphLeaks, label: str) -> LabelLeak:
    (found,) = [entry for entry in leaks.labels if entry.label == label]
    return found


class TestCrossGraphLeakCensusRun:
    @pytest.mark.asyncio
    async def test_counts_foreign_nodes_per_label_in_leak_labels_order(self):
        leaks = await census_of(LeakGraph(HOME_NODES))

        assert leaks.group_id == 'home'
        assert LEAK_LABELS == ('Entity', 'Episodic')
        assert [entry.label for entry in leaks.labels] == list(LEAK_LABELS)
        entity = label_of(leaks, 'Entity')
        assert entity.leaked == 3
        assert entity.by_group_id == (
            ForeignGroup(node_group_id='reify', nodes=2),
            ForeignGroup(node_group_id='dark-factory', nodes=1),
        )
        episodic = label_of(leaks, 'Episodic')
        assert episodic.leaked == 4
        assert episodic.by_group_id == (ForeignGroup(node_group_id='know_live', nodes=4),)

    @pytest.mark.asyncio
    async def test_a_node_with_no_group_id_is_leaked_under_none(self):
        raw = LeakGraph([*nodes_of('Entity', 'home', 2), *nodes_of('Entity', None, 1)])

        entity = label_of(await census_of(raw), 'Entity')

        assert entity.leaked == 1
        assert entity.by_group_id == (ForeignGroup(node_group_id=None, nodes=1),)

    @pytest.mark.asyncio
    async def test_ties_order_by_group_id_with_none_last(self):
        raw = LeakGraph([
            *nodes_of('Entity', None, 1),
            *nodes_of('Entity', 'zeta', 1),
            *nodes_of('Entity', 'alpha', 1),
            *nodes_of('Entity', 'big', 2),
        ])

        entity = label_of(await census_of(raw), 'Entity')

        assert [group.node_group_id for group in entity.by_group_id] == [
            'big', 'alpha', 'zeta', None,
        ]

    @pytest.mark.asyncio
    async def test_a_clean_graph_reports_zero_and_complete_for_every_label(self):
        leaks = await census_of(LeakGraph(CLEAN_NODES))

        assert leaks.complete is True
        for entry in leaks.labels:
            assert (entry.leaked, entry.by_group_id) == (0, ())
            assert (entry.complete, entry.incomplete_kind) == (True, None)


class TestCrossGraphLeakCensusCompleteness:
    @pytest.mark.asyncio
    @pytest.mark.parametrize('unusable', [[], [[None]]])
    async def test_an_unusable_count_is_census_unavailable_with_the_breakdown_kept(
        self, unusable,
    ):
        raw = LeakGraph(HOME_NODES, count_override={'Entity': unusable})

        leaks = await census_of(raw)

        entity = label_of(leaks, 'Entity')
        assert entity.leaked is None
        assert entity.complete is False
        assert entity.incomplete_kind == INCOMPLETE_CENSUS_UNAVAILABLE
        assert entity.by_group_id == (
            ForeignGroup(node_group_id='reify', nodes=2),
            ForeignGroup(node_group_id='dark-factory', nodes=1),
        )

    @pytest.mark.asyncio
    async def test_a_breakdown_below_the_proven_total_is_a_short_read(self):
        raw = LeakGraph(HOME_NODES, breakdown_override={'Entity': [['reify', 2]]})

        entity = label_of(await census_of(raw), 'Entity')

        assert entity.leaked == 3
        assert entity.complete is False
        assert entity.incomplete_kind == INCOMPLETE_SHORT_READ

    @pytest.mark.asyncio
    async def test_a_breakdown_above_the_proven_total_is_growth_and_complete(self):
        raw = LeakGraph(
            HOME_NODES, breakdown_override={'Entity': [['reify', 3], ['dark-factory', 1]]},
        )

        entity = label_of(await census_of(raw), 'Entity')

        assert entity.leaked == 3
        assert (entity.complete, entity.incomplete_kind) == (True, None)

    @pytest.mark.asyncio
    async def test_the_graph_is_incomplete_when_any_label_is(self):
        raw = LeakGraph(HOME_NODES, count_override={'Episodic': [[None]]})

        leaks = await census_of(raw)

        assert label_of(leaks, 'Entity').complete is True
        assert label_of(leaks, 'Episodic').complete is False
        assert leaks.complete is False


class TestCrossGraphLeakCensusIsReadOnly:
    @pytest.mark.asyncio
    async def test_every_call_is_a_read_scoped_to_the_graph_key(self):
        raw = LeakGraph(HOME_NODES)

        await census_of(raw)

        raw.query.assert_not_awaited()
        raw.delete.assert_not_awaited()
        raw.execute_command.assert_not_awaited()
        assert raw.ro_calls
        assert all(params == {'graph_key': 'home'} for _, params in raw.ro_calls)
        assert {label_in(cypher) for cypher, _ in raw.ro_calls} == set(LEAK_LABELS)


class TestCrossGraphLeakCensusLogs:
    @pytest.mark.asyncio
    async def test_a_leaking_label_is_a_warning_naming_graph_label_and_count(self, caplog):
        with caplog.at_level(logging.WARNING, logger=_CENSUS_LOGGER):
            await census_of(LeakGraph(HOME_NODES))

        warnings = [r.getMessage() for r in caplog.records if r.levelno == logging.WARNING]
        assert any("'home'" in m and 'Entity' in m and '3' in m for m in warnings)
        assert any("'home'" in m and 'Episodic' in m and '4' in m for m in warnings)

    @pytest.mark.asyncio
    async def test_an_incomplete_label_is_a_coverage_shortfall_error(self, caplog):
        raw = LeakGraph(HOME_NODES, breakdown_override={'Entity': [['reify', 2]]})

        with caplog.at_level(logging.ERROR, logger=_CENSUS_LOGGER):
            await census_of(raw)

        errors = [r.getMessage() for r in caplog.records if r.levelno == logging.ERROR]
        assert len(errors) == 1
        assert 'COVERAGE SHORTFALL' in errors[0]
        assert INCOMPLETE_SHORT_READ in errors[0]


class TestLabelLeakInvariant:
    @pytest.mark.parametrize(
        ('complete', 'incomplete_kind', 'named'),
        [
            (True, INCOMPLETE_SHORT_READ, INCOMPLETE_SHORT_READ),
            (False, None, 'None'),
            (False, 'bogus_kind', 'bogus_kind'),
        ],
    )
    def test_a_contradictory_or_unknown_kind_is_refused(self, complete, incomplete_kind, named):
        with pytest.raises(ValueError) as excinfo:
            LabelLeak(
                label='Entity', leaked=3, by_group_id=(),
                complete=complete, incomplete_kind=incomplete_kind,
            )

        assert named in str(excinfo.value)
        assert str(complete) in str(excinfo.value)

    @pytest.mark.asyncio
    async def test_the_result_is_frozen_all_the_way_down(self):
        leaks = await census_of(LeakGraph(HOME_NODES))
        entity = label_of(leaks, 'Entity')

        for frozen in (leaks, entity, entity.by_group_id[0]):
            for field in fields(frozen):
                with pytest.raises(FrozenInstanceError):
                    setattr(frozen, field.name, None)
        assert isinstance(leaks.labels, tuple)
        assert isinstance(entity.by_group_id, tuple)


# ---------------------------------------------------------------------------
# Live FalkorDB: the real Cypher against a throwaway scratch graph
# ---------------------------------------------------------------------------


class LiveGraph(NamedTuple):
    graph: Any
    name: str


@pytest_asyncio.fixture
async def leak_census_live_graph():
    """A throwaway scratch graph, yielded empty and deleted afterwards."""
    name = unique_graph_name('6849_leak_census')
    client = FalkorDB(host=FALKOR_HOST, port=FALKOR_PORT)
    graph = client.select_graph(name)
    try:
        yield LiveGraph(graph=graph, name=name)
    finally:
        with contextlib.suppress(Exception):
            await graph.delete()
        with contextlib.suppress(Exception):
            await client.aclose()


async def _create(graph, label: str, uuid: str, group_id: str | None) -> None:
    if group_id is None:
        await graph.query(f'CREATE (:{label} {{uuid: $uuid, name: $uuid}})', {'uuid': uuid})
        return
    await graph.query(
        f'CREATE (:{label} {{uuid: $uuid, name: $uuid, group_id: $group_id}})',
        {'uuid': uuid, 'group_id': group_id},
    )


@falkor_skipif()
@pytest.mark.timeout(60)
@pytest.mark.integration
class TestCrossGraphLeakCensusLiveFalkorDB:
    @pytest.mark.asyncio
    async def test_counts_foreign_and_group_less_nodes_per_label(self, leak_census_live_graph):
        live = leak_census_live_graph
        for i, group_id in enumerate([live.name, live.name, 'other_project', 'other_project', None]):
            await _create(live.graph, 'Entity', f'entity-{i}', group_id)
        for i, group_id in enumerate([live.name, 'other_project']):
            await _create(live.graph, 'Episodic', f'episode-{i}', group_id)

        leaks = await CrossGraphLeakCensus(
            graphs=FakeGraphAccess({live.name: live.graph}),
        ).run(group_id=live.name)

        entity = label_of(leaks, 'Entity')
        assert entity.leaked == 3
        assert entity.by_group_id == (
            ForeignGroup(node_group_id='other_project', nodes=2),
            ForeignGroup(node_group_id=None, nodes=1),
        )
        episodic = label_of(leaks, 'Episodic')
        assert episodic.leaked == 1
        assert episodic.by_group_id == (ForeignGroup(node_group_id='other_project', nodes=1),)
        assert leaks.complete is True
        assert all(entry.complete for entry in leaks.labels)
