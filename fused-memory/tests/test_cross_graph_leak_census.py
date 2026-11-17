"""The read-only cross-graph leak census (task 6849).

A node whose ``group_id`` names another graph, or no graph, is task 2115's
cross-graph leak, and every group-scoped read is blind to it. The census counts
those nodes per graph and per label, and says whether its count is whole.

Read-only is asserted structurally, against the raw graph's writable channels,
and the live class pins the real Cypher's NULL semantics and label scoping
against a scratch FalkorDB.
"""
from __future__ import annotations

import asyncio
import contextlib
import json
import logging
import re
from collections import Counter
from collections.abc import Iterable
from dataclasses import FrozenInstanceError, asdict, fields
from typing import Any, NamedTuple
from unittest.mock import AsyncMock, MagicMock, patch

import pytest
import pytest_asyncio
from _fm_falkor_live import FALKOR_HOST, FALKOR_PORT, falkor_skipif, unique_graph_name
from _graph_access_fake import FakeGraphAccess
from falkordb.asyncio import FalkorDB

from fused_memory.backends.paged_read import (
    INCOMPLETE_CENSUS_UNAVAILABLE,
    INCOMPLETE_SHORT_READ,
)
from fused_memory.maintenance._utils import ServiceOpen
from fused_memory.maintenance.cross_graph_leak_census import (
    LEAK_LABELS,
    CrossGraphLeakCensus,
    ForeignGroup,
    GraphLeaks,
    LabelLeak,
    LabelTotal,
    LeakSweep,
    main,
    run_cross_graph_leak_census,
)
from fused_memory.maintenance.graph_sweep import GraphFailure

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


async def census_of(raw: LeakGraph, graph_key: str = 'home') -> GraphLeaks:
    return await CrossGraphLeakCensus(graph_access=FakeGraphAccess({graph_key: raw})).run(
        graph_key=graph_key,
    )


def label_of(leaks: GraphLeaks, label: str) -> LabelLeak:
    (found,) = [entry for entry in leaks.labels if entry.label == label]
    return found


class TestCrossGraphLeakCensusRun:
    @pytest.mark.asyncio
    async def test_counts_foreign_nodes_per_label_in_leak_labels_order(self):
        leaks = await census_of(LeakGraph(HOME_NODES))

        assert leaks.graph_key == 'home'
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
# The sweep: several graphs, best-effort per graph
# ---------------------------------------------------------------------------


class SweepOnlyAccess(FakeGraphAccess):
    """A provider whose listing fails the test: a sweep reads only the keys it is given."""

    async def list_graphs(self) -> list[str]:
        raise AssertionError('sweep consulted list_graphs; it must read only the keys given')


class BrokenGraph(LeakGraph):
    """A raw graph whose every read fails."""

    def __init__(self) -> None:
        super().__init__([])

    async def ro_query(self, cypher: str, params: dict[str, Any] | None = None) -> _Result:
        raise RuntimeError('FalkorDB connection reset')


#: Graph 'other': Entity 2 leaked, Episodic 1 leaked.
OTHER_NODES = [
    *nodes_of('Entity', 'other', 4),
    *nodes_of('Entity', 'reify', 2),
    *nodes_of('Episodic', 'home', 1),
]

#: Graph 'clean': nothing leaked.
CLEAN_GRAPH_NODES = [*nodes_of('Entity', 'clean', 3), *nodes_of('Episodic', 'clean', 1)]


async def sweep_of(graphs: dict[str, LeakGraph], keys: list[str] | None = None) -> LeakSweep:
    census = CrossGraphLeakCensus(graph_access=SweepOnlyAccess(graphs))
    return await census.sweep(list(graphs) if keys is None else keys)


class TestCrossGraphLeakCensusSweep:
    @pytest.mark.asyncio
    async def test_reads_only_the_given_keys_in_the_given_order(self):
        unread = LeakGraph(HOME_NODES)
        graphs = {
            'home': LeakGraph(HOME_NODES),
            'other': LeakGraph(OTHER_NODES),
            'clean': LeakGraph(CLEAN_GRAPH_NODES),
            'unread': unread,
        }

        sweep = await sweep_of(graphs, ['other', 'clean', 'home'])

        assert [graph.graph_key for graph in sweep.graphs] == ['other', 'clean', 'home']
        assert unread.ro_calls == []

    @pytest.mark.asyncio
    async def test_totals_sum_each_label_across_the_graphs_read(self):
        sweep = await sweep_of({
            'home': LeakGraph(HOME_NODES),
            'other': LeakGraph(OTHER_NODES),
            'clean': LeakGraph(CLEAN_GRAPH_NODES),
        })

        assert sweep.totals == (
            LabelTotal(label='Entity', leaked=5),
            LabelTotal(label='Episodic', leaked=5),
        )
        assert [total.label for total in sweep.totals] == list(LEAK_LABELS)

    @pytest.mark.asyncio
    async def test_an_unknown_count_adds_nothing_and_leaves_the_sweep_incomplete(self):
        sweep = await sweep_of({
            'home': LeakGraph(HOME_NODES, count_override={'Entity': [[None]]}),
            'other': LeakGraph(OTHER_NODES),
        })

        assert sweep.totals[0] == LabelTotal(label='Entity', leaked=2)
        assert sweep.complete is False

    @pytest.mark.asyncio
    async def test_a_failing_graph_is_recorded_and_the_others_are_still_read(self):
        sweep = await sweep_of({
            'home': LeakGraph(HOME_NODES),
            'broken': BrokenGraph(),
            'clean': LeakGraph(CLEAN_GRAPH_NODES),
        })

        assert [graph.graph_key for graph in sweep.graphs] == ['home', 'clean']
        assert sweep.failures == (
            GraphFailure(group_id='broken', error='RuntimeError: FalkorDB connection reset'),
        )

    @pytest.mark.asyncio
    async def test_complete_only_when_every_graph_was_read_whole(self):
        whole = await sweep_of({
            'home': LeakGraph(HOME_NODES), 'clean': LeakGraph(CLEAN_GRAPH_NODES),
        })
        truncated = await sweep_of({
            'home': LeakGraph(HOME_NODES, breakdown_override={'Entity': [['reify', 1]]}),
            'clean': LeakGraph(CLEAN_GRAPH_NODES),
        })
        failed = await sweep_of({'home': LeakGraph(HOME_NODES), 'broken': BrokenGraph()})

        assert whole.complete is True
        assert truncated.complete is False
        assert failed.complete is False
        assert all(graph.complete for graph in failed.graphs)

    @pytest.mark.asyncio
    async def test_elapsed_ms_is_a_non_negative_float(self):
        sweep = await sweep_of({'home': LeakGraph(HOME_NODES)})

        assert isinstance(sweep.elapsed_ms, float)
        assert sweep.elapsed_ms >= 0

    @pytest.mark.asyncio
    async def test_the_sweep_and_its_totals_are_frozen(self):
        sweep = await sweep_of({'home': LeakGraph(HOME_NODES)})

        for frozen in (sweep, sweep.totals[0]):
            for field in fields(frozen):
                with pytest.raises(FrozenInstanceError):
                    setattr(frozen, field.name, None)
        assert isinstance(sweep.graphs, tuple)
        assert isinstance(sweep.failures, tuple)
        assert isinstance(sweep.totals, tuple)


# ---------------------------------------------------------------------------
# The entrypoint: which graphs a run sweeps, and how the store is opened
# ---------------------------------------------------------------------------

_ENTRYPOINT_SEAM = 'fused_memory.maintenance.cross_graph_leak_census.maintenance_service'


def leak_service(graphs: dict[str, LeakGraph], listed: list[str]) -> MagicMock:
    """A maintenance service whose graphiti exposes *graphs* and lists *listed*."""
    service = MagicMock()
    service.graphiti.graph_access = FakeGraphAccess(graphs)
    service.graphiti.list_graphs = AsyncMock(return_value=listed)
    return service


class TestRunCrossGraphLeakCensus:
    @pytest.mark.asyncio
    async def test_opens_the_store_for_reads_only(self, make_fake_maintenance_service):
        service = leak_service({'home': LeakGraph(HOME_NODES)}, ['home'])

        with patch(
            _ENTRYPOINT_SEAM, side_effect=make_fake_maintenance_service(MagicMock(), service),
        ) as entered:
            await run_cross_graph_leak_census(config_path='/tmp/config.yaml')

        entered.assert_called_once_with('/tmp/config.yaml', open_as=ServiceOpen.READS)

    @pytest.mark.asyncio
    async def test_without_a_graph_key_sweeps_every_graph_the_backend_lists(
        self, make_fake_maintenance_service,
    ):
        unlisted = LeakGraph(HOME_NODES)
        service = leak_service(
            {'home': LeakGraph(HOME_NODES), 'other': LeakGraph(OTHER_NODES), 'default_db': unlisted},
            ['home', 'other'],
        )

        with patch(
            _ENTRYPOINT_SEAM, side_effect=make_fake_maintenance_service(MagicMock(), service),
        ):
            sweep = await run_cross_graph_leak_census()

        assert [graph.graph_key for graph in sweep.graphs] == ['home', 'other']
        assert unlisted.ro_calls == []

    @pytest.mark.asyncio
    async def test_a_graph_key_reads_only_that_graph_and_lists_nothing(
        self, make_fake_maintenance_service,
    ):
        other = LeakGraph(OTHER_NODES)
        service = leak_service({'home': LeakGraph(HOME_NODES), 'other': other}, ['home', 'other'])

        with patch(
            _ENTRYPOINT_SEAM, side_effect=make_fake_maintenance_service(MagicMock(), service),
        ):
            sweep = await run_cross_graph_leak_census(graph_key='home')

        assert [graph.graph_key for graph in sweep.graphs] == ['home']
        assert other.ro_calls == []
        service.graphiti.list_graphs.assert_not_awaited()


# ---------------------------------------------------------------------------
# The CLI: exit code, --json, the summary, and the flag surface
# ---------------------------------------------------------------------------


def built_sweep(graphs: dict[str, LeakGraph]) -> LeakSweep:
    """The LeakSweep the census really produces over *graphs*; sync because main() runs its own loop."""
    return asyncio.run(sweep_of(graphs))


def run_main(sweep: LeakSweep, argv=()) -> int:
    """``main(argv)`` over a canned *sweep*, returning its exit code."""
    with patch(
        'fused_memory.maintenance.cross_graph_leak_census.run_cross_graph_leak_census',
        AsyncMock(return_value=sweep),
    ):
        return main(list(argv))


def three_graphs() -> dict[str, LeakGraph]:
    return {
        'home': LeakGraph(HOME_NODES),
        'other': LeakGraph(OTHER_NODES),
        'clean': LeakGraph(CLEAN_GRAPH_NODES),
    }


class TestTheExitCodeAnswersWasEveryGraphReadWhole:
    def test_a_whole_sweep_exits_zero_even_with_leaks(self):
        assert run_main(built_sweep(three_graphs())) == 0

    def test_an_incomplete_graph_exits_one(self):
        graphs = {'home': LeakGraph(HOME_NODES, count_override={'Entity': []})}

        assert run_main(built_sweep(graphs)) == 1

    def test_an_unreadable_graph_exits_one(self):
        graphs = {'home': LeakGraph(HOME_NODES), 'broken': BrokenGraph()}

        assert run_main(built_sweep(graphs)) == 1


class TestTheTwoOutputShapes:
    def test_json_round_trips_the_whole_sweep(self, capsys):
        sweep = built_sweep({**three_graphs(), 'broken': BrokenGraph()})

        run_main(sweep, ['--json'])

        emitted = json.loads(capsys.readouterr().out)
        assert emitted == json.loads(json.dumps(asdict(sweep), default=str))
        assert set(emitted) == {'graphs', 'failures', 'totals', 'complete', 'elapsed_ms'}

    def test_the_summary_carries_totals_graph_count_and_only_the_leaking_graphs(self, capsys):
        run_main(built_sweep(three_graphs()))

        out = capsys.readouterr().out
        assert 'Entity=5 Episodic=5' in out
        assert '3 graph(s)' in out
        assert 'home: Entity=3 Episodic=4' in out
        assert 'other: Entity=2 Episodic=1' in out
        assert 'clean' not in out
        assert 'complete=True' in out
        assert 'elapsed_ms=' in out

    def test_the_summary_names_a_graph_with_an_incomplete_label(self, capsys):
        graphs = {'clean': LeakGraph(CLEAN_GRAPH_NODES, count_override={'Entity': []})}

        run_main(built_sweep(graphs))

        out = capsys.readouterr().out
        assert 'clean:' in out
        assert INCOMPLETE_CENSUS_UNAVAILABLE in out
        assert 'complete=False' in out

    def test_the_summary_carries_a_failed_block_naming_every_unread_graph(self, capsys):
        run_main(built_sweep({'home': LeakGraph(HOME_NODES), 'broken': BrokenGraph()}))

        out = capsys.readouterr().out
        assert 'FAILED to read 1 graph(s): broken (RuntimeError: FalkorDB connection reset)' in out

    def test_the_summary_says_so_when_nothing_was_read(self, capsys):
        run_main(built_sweep({}))

        out = capsys.readouterr().out
        assert '0 graph(s)' in out
        assert 'no graphs read' in out


class TestTheCliOffersNoWayToMutate:
    @pytest.mark.parametrize('flag', ['--apply', '--fix'])
    def test_a_mutating_flag_is_rejected_by_argparse(self, flag):
        with pytest.raises(SystemExit) as excinfo:
            run_main(built_sweep({}), [flag])

        assert excinfo.value.code == 2


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
        ).run(graph_key=live.name)

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
