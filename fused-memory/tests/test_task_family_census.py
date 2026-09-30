"""Tests for the read-only task-family census (task 5264, workstream B).

The census reports the RESIDUE workstream A's write-path fix can never reach:
task families that are already split across several nodes and that no future
episode will mention again, so the post-write normalizer never gets a chance to
collapse them.

Every assertion here is about a READ. The module's central promise is that
counting the residue cannot change it, which is why "no writes" is asserted
structurally — against the backend's mutating methods and against the graph
driver itself — rather than by reading the source and trusting it.
"""
from __future__ import annotations

import asyncio
import json
from dataclasses import FrozenInstanceError
from unittest.mock import AsyncMock, MagicMock, patch

import pytest
from _fm_helpers import pydantic_spec

from fused_memory.backends.graphiti_client import PagedRead
from fused_memory.config.schema import FusedMemoryConfig
from fused_memory.maintenance.task_family_census import (
    CensusSweep,
    TaskFamilyCensus,
    _build_parser,
    main,
    run_task_family_census,
)

# Every mutating backend method the census must never reach. Named explicitly
# rather than inferred, so adding one to the backend and quietly calling it from
# here stays a deliberate act.
MUTATING_BACKEND_METHODS = (
    'merge_entities',
    'rename_entity_node',
    'update_node_name',
    'delete_entity_node',
    'ensure_entity_node',
)

#: The fixture graph, as enumerate_entity_nodes returns it. Five shapes in one
#: corpus, because the census's whole job is telling them apart:
#:   - a fragmented family spelled three ways (605);
#:   - a clean single canonical node (700) — NOT residue;
#:   - a lone non-canonically-named node (800) — NOT residue either: it is one
#:     node, and workstream A heals it the next time an episode touches it;
#:   - an exact-name duplicate PAIR (900) — residue with only one spelling;
#:   - noise that a CONTAINS '605' probe would hand back but that is not the
#:     605 family: 'Alice', the foreign 'reify:605', and 'Task 6051'.
FIXTURE_NODES = [
    {'uuid': 'u-605-lower', 'name': 'task 605', 'summary': ''},
    {'uuid': 'u-605-canon', 'name': 'Task 605', 'summary': ''},
    {'uuid': 'u-605-plural', 'name': 'tasks 605', 'summary': ''},
    {'uuid': 'u-700', 'name': 'Task 700', 'summary': ''},
    {'uuid': 'u-800', 'name': 'task 800', 'summary': ''},
    {'uuid': 'u-900-a', 'name': 'Task 900', 'summary': ''},
    {'uuid': 'u-900-b', 'name': 'Task 900', 'summary': ''},
    {'uuid': 'u-alice', 'name': 'Alice', 'summary': ''},
    {'uuid': 'u-foreign', 'name': 'reify:605', 'summary': ''},
    {'uuid': 'u-6051', 'name': 'Task 6051', 'summary': ''},
]

#: What find_entity_nodes_by_name_substring returns per probed number, in the
#: backend's survivor-first order. The '605' probe deliberately includes the
#: foreign 'reify:605' and the unrelated 'Task 6051' — both really do contain
#: the substring, and filtering them out is the caller's job.
FIXTURE_PROBES = {
    '605': [
        {'uuid': 'u-605-lower', 'name': 'task 605', 'created_at': 100,
         'edge_count': 13, 'mentions_count': 1, 'provenance_rank': 14},
        {'uuid': 'u-foreign', 'name': 'reify:605', 'created_at': 20,
         'edge_count': 7, 'mentions_count': 0, 'provenance_rank': 7},
        {'uuid': 'u-605-canon', 'name': 'Task 605', 'created_at': 50,
         'edge_count': 2, 'mentions_count': 1, 'provenance_rank': 3},
        {'uuid': 'u-605-plural', 'name': 'tasks 605', 'created_at': 150,
         'edge_count': 1, 'mentions_count': 0, 'provenance_rank': 1},
        {'uuid': 'u-6051', 'name': 'Task 6051', 'created_at': 9,
         'edge_count': 1, 'mentions_count': 0, 'provenance_rank': 1},
    ],
    '900': [
        {'uuid': 'u-900-a', 'name': 'Task 900', 'created_at': 30,
         'edge_count': 4, 'mentions_count': 0, 'provenance_rank': 4},
        {'uuid': 'u-900-b', 'name': 'Task 900', 'created_at': 40,
         'edge_count': 1, 'mentions_count': 0, 'provenance_rank': 1},
    ],
}

#: A family whose survivor is decided by EPISODES, not edges (task 4986): the
#: edge-poorer 'Task 700' outranks 'task 700' on mentions and so comes back
#: first. Without this the file would merely SURVIVE the new rule while every
#: assertion still described the old one.
MENTIONS_DECIDED_PROBES = {
    '700': [
        {'uuid': 'u-700-canon', 'name': 'Task 700', 'created_at': 80,
         'edge_count': 2, 'mentions_count': 9, 'provenance_rank': 11},
        {'uuid': 'u-700-lower', 'name': 'task 700', 'created_at': 70,
         'edge_count': 8, 'mentions_count': 0, 'provenance_rank': 8},
    ],
}


def complete_read(rows_seen: int) -> PagedRead:
    """A PagedRead that reports a whole enumeration."""
    return PagedRead(
        rows=[], complete=True, rows_seen=rows_seen, expected_rows=rows_seen, reason=None,
    )


def make_census_backend(nodes=None, probes=None, paged=None, graphs=('home',)):
    """A backend stub exposing only what the census is allowed to use.

    Every mutating method is present as an AsyncMock so a test can assert it
    was never awaited — an absent attribute would fail with AttributeError
    instead, which proves nothing about intent.
    """
    backend = MagicMock()
    nodes = FIXTURE_NODES if nodes is None else nodes
    backend.enumerate_entity_nodes = AsyncMock(
        return_value=(nodes, paged if paged is not None else complete_read(len(nodes)))
    )

    probe_rows = FIXTURE_PROBES if probes is None else probes

    async def fake_probe(substring, *, group_id):
        return list(probe_rows.get(substring, []))

    backend.find_entity_nodes_by_name_substring = AsyncMock(side_effect=fake_probe)
    backend.list_graphs = AsyncMock(return_value=list(graphs))
    for method_name in MUTATING_BACKEND_METHODS:
        setattr(backend, method_name, AsyncMock())
    return backend


class TestTaskFamilyCensusRun:
    """TaskFamilyCensus(backend=...).run(group_id=...) — one graph."""

    @pytest.mark.asyncio
    async def test_reports_exactly_the_families_holding_more_than_one_node(self):
        """The >1-node threshold is the honest one.

        'Task 700' is a clean single node and the lone 'task 800' is one node
        with a fixable NAME, not a split family — counting either as residue
        would inflate a number a human is going to act on.
        """
        census = TaskFamilyCensus(backend=make_census_backend())

        result = await census.run(group_id='home')

        assert [family.canonical_name for family in result.families] == [
            'Task 605', 'Task 900',
        ]

    @pytest.mark.asyncio
    async def test_probes_only_the_fragmented_families(self):
        """The edge-count probe is scoped to families the grouping pass already
        flagged, so a clean graph costs exactly one enumeration and nothing
        else."""
        backend = make_census_backend()

        await TaskFamilyCensus(backend=backend).run(group_id='home')

        probed = {
            c.args[0] for c in backend.find_entity_nodes_by_name_substring.await_args_list
        }
        assert probed == {'605', '900'}

    @pytest.mark.asyncio
    async def test_each_family_carries_every_variant_spelling_with_its_edge_count(self):
        """The per-spelling breakdown IS the operator-facing report: a bare
        count says a task is split, the breakdown says which node holds the
        edges and therefore what a collapse would cost."""
        census = TaskFamilyCensus(backend=make_census_backend())

        result = await census.run(group_id='home')

        family_605 = result.families[0]
        assert [(v.name, v.uuid, v.edge_count) for v in family_605.variants] == [
            ('task 605', 'u-605-lower', 13),
            ('Task 605', 'u-605-canon', 2),
            ('tasks 605', 'u-605-plural', 1),
        ]

    @pytest.mark.asyncio
    async def test_the_would_be_survivor_can_be_decided_by_episodes_not_edges(self):
        """variants[0] is the node a collapse would KEEP, and since task 4986
        the backend ranks by provenance_rank (edge_count + mentions_count), not
        edge_count alone.

        The census re-sorts nothing — it inherits the backend's row order — so
        an edge-POORER variant legitimately comes first when it carries more
        episode provenance. Pinning that here is what keeps this file expressing
        the CURRENT survivor rule rather than merely surviving it: every other
        assertion in this class uses fixtures whose edge order and rank order
        happen to agree, so none of them would notice if the census started
        re-deriving the old rule.
        """
        nodes = [
            {'uuid': 'u-700-canon', 'name': 'Task 700', 'summary': ''},
            {'uuid': 'u-700-lower', 'name': 'task 700', 'summary': ''},
        ]
        census = TaskFamilyCensus(
            backend=make_census_backend(nodes=nodes, probes=MENTIONS_DECIDED_PROBES),
        )

        result = await census.run(group_id='home')

        (family_700,) = result.families
        assert [(v.name, v.uuid, v.edge_count) for v in family_700.variants] == [
            ('Task 700', 'u-700-canon', 2),
            ('task 700', 'u-700-lower', 8),
        ], 'the episode-richer variant leads even though it holds FEWER edges'

    @pytest.mark.asyncio
    async def test_a_multi_spelling_family_is_distinguishable_from_an_exact_name_pair(self):
        """Two different residues, two different repairs: 605 is split across
        three SPELLINGS (workstream A's territory), while the 900 pair shares
        one spelling and is the exact-name duplicate _dedup_episode_nodes
        handles. The breakdown has to tell them apart."""
        census = TaskFamilyCensus(backend=make_census_backend())

        result = await census.run(group_id='home')

        family_605, family_900 = result.families
        assert len({v.name for v in family_605.variants}) == 3
        assert len({v.name for v in family_900.variants}) == 1
        assert len(family_900.variants) == 2

    @pytest.mark.asyncio
    async def test_noise_matching_the_substring_never_joins_a_family(self):
        """'reify:605' names ANOTHER project's task and 'Task 6051' a different
        one of ours; both really do contain '605'. Precision comes from
        task_naming's acceptance rule, never from the query."""
        census = TaskFamilyCensus(backend=make_census_backend())

        result = await census.run(group_id='home')

        reported_uuids = {v.uuid for family in result.families for v in family.variants}
        assert 'u-foreign' not in reported_uuids
        assert 'u-6051' not in reported_uuids
        assert reported_uuids == {
            'u-605-lower', 'u-605-canon', 'u-605-plural', 'u-900-a', 'u-900-b',
        }

    @pytest.mark.asyncio
    async def test_records_the_group_id_and_how_many_nodes_it_scanned(self):
        census = TaskFamilyCensus(backend=make_census_backend())

        result = await census.run(group_id='home')

        assert result.group_id == 'home'
        assert result.nodes_scanned == len(FIXTURE_NODES)

    @pytest.mark.asyncio
    async def test_the_result_is_frozen_all_the_way_down(self):
        """A census is evidence an operator may act on destructively, so no
        consumer gets to edit it after the fact — the same reason
        canonical_labels.Referent is frozen.

        The collections are TUPLES, not lists: frozen=True blocks attribute
        REBINDING only, so a list field would leave ``families.append(...)``
        wide open and the frozen-ness would not notice.

        The writes go through ``setattr`` because the two gates disagree here:
        pyright rejects a direct assignment to a frozen dataclass field before
        the test can ever run it, while ruff's B010 rejects ``setattr`` with a
        constant name. The targeted ``noqa`` is this repo's established
        resolution for that standoff (cf. test_project_scope.py,
        test_consolidation_gate.py) — neither gate is weakened.
        """
        census = TaskFamilyCensus(backend=make_census_backend())

        result = await census.run(group_id='home')

        with pytest.raises(FrozenInstanceError):
            setattr(result, 'group_id', 'elsewhere')  # noqa: B010
        with pytest.raises(FrozenInstanceError):
            setattr(result.families[0], 'canonical_name', 'Task 1')  # noqa: B010
        with pytest.raises(FrozenInstanceError):
            setattr(result.families[0].variants[0], 'edge_count', 999)  # noqa: B010
        assert isinstance(result.families, tuple)
        assert isinstance(result.families[0].variants, tuple)


#: A family the ENUMERATION sees whole and the PROBE does not. The two reads
#: are differently scoped — enumerate_entity_nodes matches every :Entity node
#: under the graph KEY, while the probe additionally requires
#: `n.group_id = $group_id` — so a node leaked in by task 2115 counts toward
#: the >1 threshold and is then dropped from the probe.
LEAKED_FAMILY_NODES = [
    {'uuid': 'u-605-canon', 'name': 'Task 605', 'summary': ''},
    {'uuid': 'u-605-leaked', 'name': 'task 605', 'summary': ''},
]

#: What the group-scoped probe returns for that family: the leaked node is
#: correctly invisible to it, leaving one spelling where the enumeration saw two.
LEAKED_FAMILY_PROBES = {
    '605': [
        {'uuid': 'u-605-canon', 'name': 'Task 605', 'created_at': 50,
         'edge_count': 2, 'mentions_count': 1, 'provenance_rank': 3},
    ],
}


class TestAFamilyTheProbeDoesNotConfirm:
    """The enumeration decides membership and the probe supplies the edge
    counts, and those two reads can disagree: they are scoped differently and
    they are not one snapshot. A family that thins below two spellings in
    between is not residue, and must not be counted as any."""

    @pytest.mark.asyncio
    async def test_a_short_probe_is_not_reported_as_a_fragmented_family(self):
        """A FragmentedFamily of one variant would contradict its own meaning —
        'split across more than one node' — while still incrementing the number
        this module exists to produce."""
        backend = make_census_backend(
            nodes=LEAKED_FAMILY_NODES, probes=LEAKED_FAMILY_PROBES,
        )

        result = await TaskFamilyCensus(backend=backend).run(group_id='home')

        assert result.families == ()

    @pytest.mark.asyncio
    async def test_it_is_recorded_rather_than_silently_dropped(self):
        """Dropping it would hide a leak: one is a race, a persistent count is
        a group_id leak worth chasing, and only the two membership numbers tell
        those apart."""
        backend = make_census_backend(
            nodes=LEAKED_FAMILY_NODES, probes=LEAKED_FAMILY_PROBES,
        )

        result = await TaskFamilyCensus(backend=backend).run(group_id='home')

        assert [
            (u.canonical_name, u.enumerated_members, u.probed_members)
            for u in result.unconfirmed
        ] == [('Task 605', 2, 1)]

    @pytest.mark.asyncio
    async def test_a_family_that_vanished_entirely_is_recorded_too(self):
        """The zero case is reachable the same way — every enumerated member
        was a leaked node — and a zero-variant FragmentedFamily would be the
        most misleading shape of all."""
        backend = make_census_backend(nodes=LEAKED_FAMILY_NODES, probes={})

        result = await TaskFamilyCensus(backend=backend).run(group_id='home')

        assert result.families == ()
        assert [u.probed_members for u in result.unconfirmed] == [0]

    @pytest.mark.asyncio
    async def test_a_confirmed_family_alongside_it_is_still_reported(self):
        """The exclusion is per-family: one unconfirmed family must not cost
        the census the residue it did confirm.

        The whole fixture corpus, with only the 605 probe thinned — so 605 goes
        unconfirmed while the 900 pair, probed normally, is still reported.
        """
        backend = make_census_backend(
            probes={**FIXTURE_PROBES, **LEAKED_FAMILY_PROBES},
        )

        result = await TaskFamilyCensus(backend=backend).run(group_id='home')

        assert [family.canonical_name for family in result.families] == ['Task 900']
        assert [u.canonical_name for u in result.unconfirmed] == ['Task 605']

    @pytest.mark.asyncio
    async def test_it_is_logged_loudly_with_both_counts(self, caplog):
        backend = make_census_backend(
            nodes=LEAKED_FAMILY_NODES, probes=LEAKED_FAMILY_PROBES,
        )

        with caplog.at_level('WARNING'):
            await TaskFamilyCensus(backend=backend).run(group_id='home')

        assert 'unconfirmed task family Task 605' in caplog.text
        assert 'NOT counted as residue' in caplog.text

    @pytest.mark.asyncio
    async def test_it_does_not_make_the_graph_incomplete(self):
        """Coverage and confirmation are different questions. The graph WAS
        read in full — that is exactly how the divergence became visible — so
        flipping `complete` would send an operator looking for a truncated read
        that never happened, and would fail the CLI's exit status too."""
        backend = make_census_backend(
            nodes=LEAKED_FAMILY_NODES, probes=LEAKED_FAMILY_PROBES,
        )

        result = await TaskFamilyCensus(backend=backend).run(group_id='home')

        assert result.complete is True
        assert result.incomplete_kind is None

    @pytest.mark.asyncio
    async def test_the_record_is_frozen_like_the_rest_of_the_result(self):
        backend = make_census_backend(
            nodes=LEAKED_FAMILY_NODES, probes=LEAKED_FAMILY_PROBES,
        )

        result = await TaskFamilyCensus(backend=backend).run(group_id='home')

        assert isinstance(result.unconfirmed, tuple)
        with pytest.raises(FrozenInstanceError):
            setattr(result.unconfirmed[0], 'probed_members', 99)  # noqa: B010

class TestTaskFamilyCensusIsReadOnly:
    """No write reaches the graph from the census path — asserted structurally."""

    @pytest.mark.asyncio
    async def test_no_mutating_backend_method_is_ever_awaited(self):
        backend = make_census_backend()

        await TaskFamilyCensus(backend=backend).run(group_id='home')

        for method_name in MUTATING_BACKEND_METHODS:
            getattr(backend, method_name).assert_not_awaited()

    @pytest.mark.asyncio
    async def test_the_real_backend_never_touches_the_writable_query_channel(
        self, mock_config, make_backend,
    ):
        """End-to-end through a REAL GraphitiBackend down to the driver: every
        query the census causes goes through ``ro_query``, never ``query``.

        This is the assertion ``_fm_helpers.assert_ro_query_only`` makes, spelled
        out rather than reused: that helper also pins ``ro_query`` to exactly ONE
        await, and a census legitimately issues several (a count probe, a page,
        then one probe per fragmented family).
        """
        backend = make_backend(mock_config)
        graph = MagicMock()
        graph.query = AsyncMock()

        async def dispatch_ro(cypher, params=None):
            # CONTAINS is tested FIRST: the substring probe's own Cypher carries
            # a `count(DISTINCT e)` edge-count aggregate — and since task 4986 a
            # `count(DISTINCT m)` mentions aggregate as well — so a `count(` test
            # would claim it and answer a six-column read with a one-column
            # census row. The second aggregate makes this ordering more
            # load-bearing, not less.
            result = MagicMock()
            if 'CONTAINS' in cypher:
                result.result_set = [
                    [row['uuid'], row['name'], row['created_at'], row['edge_count'],
                     row['mentions_count'], row['provenance_rank']]
                    for row in FIXTURE_PROBES.get((params or {})['substring'], [])
                ]
            elif 'count(' in cypher:
                result.result_set = [[len(FIXTURE_NODES)]]
            else:
                result.result_set = [
                    [node['uuid'], node['name'], node['summary']] for node in FIXTURE_NODES
                ]
            return result

        graph.ro_query = AsyncMock(side_effect=dispatch_ro)
        backend._driver._get_graph = MagicMock(return_value=graph)

        result = await TaskFamilyCensus(backend=backend).run(group_id='home')

        graph.query.assert_not_awaited()
        assert graph.ro_query.await_count >= 2  # census probe + at least one page
        assert [family.canonical_name for family in result.families] == [
            'Task 605', 'Task 900',
        ]


class TestTaskFamilyCensusCoverageHonesty:
    """A count read off a truncated enumeration is worse than no count at all."""

    @pytest.mark.asyncio
    async def test_an_incomplete_enumeration_is_reported_not_smoothed_over(self):
        """The census's entire output is a number a human will act on, so a
        partial read must never present as a whole one."""
        truncated = PagedRead(
            rows=[],
            complete=False,
            rows_seen=10,
            expected_rows=4321,
            reason='page cap reached after 1000 pages',
            incomplete_kind='page_cap',
        )
        census = TaskFamilyCensus(backend=make_census_backend(paged=truncated))

        result = await census.run(group_id='home')

        assert result.complete is False
        assert result.incomplete_kind == 'page_cap'
        assert result.rows_seen == 10
        assert result.expected_rows == 4321

    @pytest.mark.asyncio
    async def test_a_whole_enumeration_reports_complete_with_no_kind(self):
        census = TaskFamilyCensus(backend=make_census_backend())

        result = await census.run(group_id='home')

        assert result.complete is True
        assert result.incomplete_kind is None

# ---------------------------------------------------------------------------
# step-11: the multi-graph sweep and the CLI-callable entrypoint
# ---------------------------------------------------------------------------

#: A second graph's corpus: the 900 pair and nothing else. Distinct from
#: FIXTURE_NODES on purpose — a sweep that merged graphs into one number would
#: report the wrong residue for both, which is why the breakdown is per-graph.
OTHER_GRAPH_NODES = [
    {'uuid': 'u-900-a', 'name': 'Task 900', 'summary': ''},
    {'uuid': 'u-900-b', 'name': 'Task 900', 'summary': ''},
]

#: A graph with nothing split at all: one clean canonical node.
CLEAN_GRAPH_NODES = [
    {'uuid': 'u-700', 'name': 'Task 700', 'summary': ''},
]


def make_sweep_backend(per_graph, graphs=None):
    """A backend whose ``enumerate_entity_nodes`` answers per group_id.

    ``per_graph`` maps group_id -> either an ``(nodes, PagedRead)`` pair or an
    Exception INSTANCE to raise. Simulating a per-graph failure by raising from
    the backend is what lets the isolation test stay outside the census: it
    never has to reach in and break an internal.
    """
    backend = MagicMock()

    async def fake_enumerate(*, group_id):
        outcome = per_graph[group_id]
        if isinstance(outcome, Exception):
            raise outcome
        return outcome

    backend.enumerate_entity_nodes = AsyncMock(side_effect=fake_enumerate)

    async def fake_probe(substring, *, group_id):
        return list(FIXTURE_PROBES.get(substring, []))

    backend.find_entity_nodes_by_name_substring = AsyncMock(side_effect=fake_probe)
    backend.list_graphs = AsyncMock(
        return_value=list(per_graph if graphs is None else graphs)
    )
    for method_name in MUTATING_BACKEND_METHODS:
        setattr(backend, method_name, AsyncMock())
    return backend


def whole(nodes):
    """``(nodes, PagedRead)`` for a graph that was enumerated in full."""
    return (nodes, complete_read(len(nodes)))


def make_three_graph_backend():
    """home (2 families) / other (1 family) / clean (0) — in list_graphs order."""
    return make_sweep_backend({
        'home': whole(FIXTURE_NODES),
        'other': whole(OTHER_GRAPH_NODES),
        'clean': whole(CLEAN_GRAPH_NODES),
    })


class TestTaskFamilyCensusSweep:
    """sweep(group_id=None) — every graph, or exactly one."""

    @pytest.mark.asyncio
    async def test_no_group_id_censuses_every_graph_and_names_each_one(self):
        """The residue is per-graph, so a whole-store number that cannot be
        attributed to a graph is not actionable: an operator cannot go and look
        at 'three families somewhere'."""
        backend = make_three_graph_backend()

        aggregate = await TaskFamilyCensus(backend=backend).sweep()

        assert [graph.group_id for graph in aggregate.graphs] == [
            'home', 'other', 'clean',
        ]
        assert [len(graph.families) for graph in aggregate.graphs] == [2, 1, 0]

    @pytest.mark.asyncio
    async def test_an_explicit_group_id_reads_only_that_graph_and_never_lists(self):
        """Asking for one graph must not enumerate the store: list_graphs is a
        read an operator scoping a census to one project did not ask for."""
        backend = make_three_graph_backend()

        aggregate = await TaskFamilyCensus(backend=backend).sweep(group_id='other')

        backend.list_graphs.assert_not_awaited()
        assert [graph.group_id for graph in aggregate.graphs] == ['other']
        assert aggregate.total_families == 1

    @pytest.mark.asyncio
    async def test_the_aggregate_totals_the_families_across_graphs(self):
        backend = make_three_graph_backend()

        aggregate = await TaskFamilyCensus(backend=backend).sweep()

        assert aggregate.total_families == 3

    @pytest.mark.asyncio
    async def test_the_aggregate_is_frozen_all_the_way_down(self):
        backend = make_three_graph_backend()

        aggregate = await TaskFamilyCensus(backend=backend).sweep()

        with pytest.raises(FrozenInstanceError):
            setattr(aggregate, 'total_families', 0)  # noqa: B010
        assert isinstance(aggregate.graphs, tuple)
        assert isinstance(aggregate.failures, tuple)

    @pytest.mark.asyncio
    async def test_elapsed_ms_is_recorded_for_the_whole_sweep(self):
        """Mirrors _run_startup_identity_scan's aggregate elapsed_ms: a sweep
        over every graph in the store is the kind of cost that has to be
        observable without guessing."""
        backend = make_three_graph_backend()

        aggregate = await TaskFamilyCensus(backend=backend).sweep()

        assert aggregate.elapsed_ms >= 0


class TestTaskFamilyCensusSweepIsBestEffortPerGraph:
    """One unreachable graph must not cost the census every other graph."""

    @pytest.mark.asyncio
    async def test_a_failing_graph_does_not_stop_the_rest_of_the_sweep(self):
        """Mirrors _run_startup_identity_scan: each graph is processed inside its
        own try/except so a failure on one never aborts the sweep."""
        backend = make_sweep_backend({
            'home': whole(FIXTURE_NODES),
            'broken': RuntimeError('FalkorDB connection reset'),
            'clean': whole(CLEAN_GRAPH_NODES),
        })

        aggregate = await TaskFamilyCensus(backend=backend).sweep()

        assert [graph.group_id for graph in aggregate.graphs] == ['home', 'clean']

    @pytest.mark.asyncio
    async def test_the_failure_is_recorded_naming_the_graph_and_the_error(self):
        """Recorded as STRUCTURE, not folded into prose: the group_id is the
        thing an operator re-runs with, and the exception TYPE is what says
        whether to retry or to go fix something."""
        backend = make_sweep_backend({
            'broken': RuntimeError('FalkorDB connection reset'),
            'clean': whole(CLEAN_GRAPH_NODES),
        })

        aggregate = await TaskFamilyCensus(backend=backend).sweep()

        assert [failure.group_id for failure in aggregate.failures] == ['broken']
        assert 'RuntimeError' in aggregate.failures[0].error
        assert 'FalkorDB connection reset' in aggregate.failures[0].error

    @pytest.mark.asyncio
    async def test_a_failure_is_logged(self, caplog):
        backend = make_sweep_backend({'broken': RuntimeError('FalkorDB connection reset')})

        with caplog.at_level('ERROR'):
            await TaskFamilyCensus(backend=backend).sweep()

        assert 'broken' in caplog.text

    @pytest.mark.asyncio
    async def test_the_sweep_never_returns_a_silently_smaller_number(self):
        """The attempted graphs must all be accounted for — censused or
        recorded as failed. A count that quietly omits an unreachable graph
        reads as 'two graphs have no residue' when the truth is 'one graph was
        never looked at'."""
        backend = make_sweep_backend({
            'home': whole(FIXTURE_NODES),
            'broken': RuntimeError('FalkorDB connection reset'),
            'clean': whole(CLEAN_GRAPH_NODES),
        })

        aggregate = await TaskFamilyCensus(backend=backend).sweep()

        assert len(aggregate.graphs) + len(aggregate.failures) == 3


class TestTaskFamilyCensusSweepCompleteness:
    """One flag answers 'can I trust this number?' for the whole sweep."""

    @pytest.mark.asyncio
    async def test_complete_when_every_graph_was_read_whole(self):
        backend = make_three_graph_backend()

        aggregate = await TaskFamilyCensus(backend=backend).sweep()

        assert aggregate.complete is True
        assert aggregate.failures == ()

    @pytest.mark.asyncio
    async def test_incomplete_when_any_graph_was_truncated(self):
        truncated = PagedRead(
            rows=[], complete=False, rows_seen=2, expected_rows=9999,
            reason='page cap reached', incomplete_kind='page_cap',
        )
        backend = make_sweep_backend({
            'home': whole(FIXTURE_NODES),
            'huge': (OTHER_GRAPH_NODES, truncated),
        })

        aggregate = await TaskFamilyCensus(backend=backend).sweep()

        assert aggregate.complete is False

    @pytest.mark.asyncio
    async def test_unconfirmed_families_total_up_without_going_incomplete(self):
        """A graph read in full that reported a probe divergence stays
        complete: every graph WAS read, and the total is honest precisely
        because those families were excluded from it."""
        backend = make_sweep_backend(
            {'home': whole(LEAKED_FAMILY_NODES), 'other': whole(LEAKED_FAMILY_NODES)},
        )

        async def thin_probe(substring, *, group_id):
            return list(LEAKED_FAMILY_PROBES.get(substring, []))

        backend.find_entity_nodes_by_name_substring = AsyncMock(side_effect=thin_probe)

        aggregate = await TaskFamilyCensus(backend=backend).sweep()

        assert aggregate.total_families == 0
        assert aggregate.total_unconfirmed == 2
        assert aggregate.complete is True

    @pytest.mark.asyncio
    async def test_incomplete_when_any_graph_errored(self):
        """An errored graph is an unknown, not a zero — so the sweep's own
        completeness flag has to go False even though every graph it DID read
        was read in full."""
        backend = make_sweep_backend({
            'home': whole(FIXTURE_NODES),
            'broken': RuntimeError('FalkorDB connection reset'),
        })

        aggregate = await TaskFamilyCensus(backend=backend).sweep()

        assert aggregate.complete is False
        assert all(graph.complete for graph in aggregate.graphs)


class TestRunTaskFamilyCensusDelegation:
    """run_task_family_census() delegates its lifecycle to maintenance_service."""

    @pytest.mark.asyncio
    async def test_delegates_to_maintenance_service_and_uses_service_graphiti(
        self, make_fake_maintenance_service,
    ):
        mock_cfg = MagicMock(spec_set=pydantic_spec(FusedMemoryConfig))
        mock_service = AsyncMock()
        mock_service.graphiti = MagicMock()
        sentinel = object()

        with (
            patch(
                'fused_memory.maintenance.task_family_census.maintenance_service',
                side_effect=make_fake_maintenance_service(mock_cfg, mock_service),
            ),
            patch(
                'fused_memory.maintenance.task_family_census.TaskFamilyCensus'
            ) as mock_census_cls,
        ):
            mock_census = MagicMock()
            mock_census.sweep = AsyncMock(return_value=sentinel)
            mock_census_cls.return_value = mock_census

            result = await run_task_family_census(config_path='/tmp/config.yaml')

        mock_census_cls.assert_called_once_with(backend=mock_service.graphiti)
        assert result is sentinel

    @pytest.mark.asyncio
    async def test_passes_the_group_id_through_to_the_sweep(
        self, make_fake_maintenance_service,
    ):
        mock_cfg = MagicMock(spec_set=pydantic_spec(FusedMemoryConfig))
        mock_service = AsyncMock()
        mock_service.graphiti = MagicMock()

        with (
            patch(
                'fused_memory.maintenance.task_family_census.maintenance_service',
                side_effect=make_fake_maintenance_service(mock_cfg, mock_service),
            ),
            patch(
                'fused_memory.maintenance.task_family_census.TaskFamilyCensus'
            ) as mock_census_cls,
        ):
            mock_census = MagicMock()
            mock_census.sweep = AsyncMock(return_value=MagicMock())
            mock_census_cls.return_value = mock_census

            await run_task_family_census(group_id='know_live')

        mock_census.sweep.assert_awaited_once_with(group_id='know_live')


class TestTheParserExposesNoWayToMutateTheGraph:
    """Read-only is this workstream's central promise, so it is asserted about
    the CLI SURFACE too — not just about the code behind it."""

    #: Every option the census CLI is allowed to have. Exact-set equality, so a
    #: new flag of ANY kind — however innocently spelled — fails this test until
    #: it is added here deliberately. That is the whole surface contract: a
    #: name-scan for mutating verbs was tried alongside it and deleted, because
    #: it caught nothing this misses while passing a flag spelled '--heal'.
    #:
    #: '--dry-run' is absent, and that absence is itself the point:
    #: verify_zombie_edges.py needs one because it can delete, whereas every
    #: census run is already a dry run and offering the flag would imply an
    #: unsafe mode exists.
    EXPECTED_OPTIONS = {'-h', '--help', '--config', '--group-id', '--json'}

    def test_exposes_exactly_config_group_id_and_json(self):
        parser = _build_parser()

        options = {
            option
            for action in parser._actions
            for option in action.option_strings
        }

        assert options == self.EXPECTED_OPTIONS

    def test_json_is_a_bare_flag_and_the_other_two_take_values(self):
        parser = _build_parser()

        args = parser.parse_args([])
        assert args.json is False
        assert args.config is None
        assert args.group_id is None

        args = parser.parse_args(['--json', '--config', '/c.yaml', '--group-id', 'home'])
        assert args.json is True
        assert args.config == '/c.yaml'
        assert args.group_id == 'home'


# ---------------------------------------------------------------------------
# the seam between the sweep and the operator: main(), --json, the summary
# ---------------------------------------------------------------------------

#: An enumeration that stopped short of the graph. The exit code exists for
#: exactly this shape: the count it produced is a lower bound, not a total.
TRUNCATED_READ = PagedRead(
    rows=[], complete=False, rows_seen=2, expected_rows=9999,
    reason='page cap reached', incomplete_kind='page_cap',
)


def sweep_of(backend) -> CensusSweep:
    """The CensusSweep a *backend* really produces.

    Built by running the census rather than by constructing a CensusSweep
    directly, so the tests below are handed exactly the object main() will be
    handed and no summary field is invented by the test. Synchronous because
    main() calls ``asyncio.run`` itself and so cannot be driven from inside an
    already-running loop.
    """
    return asyncio.run(TaskFamilyCensus(backend=backend).sweep())


def run_main(sweep: CensusSweep, argv=()) -> int:
    """Call ``main(argv)`` over a canned *sweep*, returning its exit code.

    ``run_task_family_census`` is stubbed with an AsyncMock so main()'s own
    ``asyncio.run`` still receives a coroutine to drive. What is under test
    here is the seam — exit code and output shape — not the census behind it.
    """
    with patch(
        'fused_memory.maintenance.task_family_census.run_task_family_census',
        AsyncMock(return_value=sweep),
    ):
        return main(list(argv))


class TestTheExitCodeAnswersCanIActOnThisNumber:
    """main()'s exit status is the only part of this module an operator's
    automation reads, and its promise is that a truncated census can never pass
    for a clean one."""

    def test_a_whole_census_exits_zero(self):
        assert run_main(sweep_of(make_three_graph_backend())) == 0

    def test_a_truncated_census_exits_non_zero(self):
        """The number is a lower bound, so a script that treated the exit code
        as 'did it run' would under-count the residue and never know."""
        backend = make_sweep_backend({'huge': (FIXTURE_NODES, TRUNCATED_READ)})

        assert run_main(sweep_of(backend)) == 1

    def test_a_census_that_could_not_read_a_graph_exits_non_zero(self):
        """An unreadable graph is an unknown, not a zero — the same reason the
        sweep's own completeness flag goes False."""
        backend = make_sweep_backend({
            'home': whole(FIXTURE_NODES),
            'broken': RuntimeError('FalkorDB connection reset'),
        })

        assert run_main(sweep_of(backend)) == 1

    def test_an_unconfirmed_family_does_not_move_the_exit_code(self):
        """Coverage and confirmation are different questions. The graph WAS
        read in full, and the count is honest precisely because the unconfirmed
        family was excluded from it — failing the run would send an operator
        looking for a truncated read that never happened."""
        backend = make_sweep_backend({'home': whole(LEAKED_FAMILY_NODES)})

        async def thin_probe(substring, *, group_id):
            return list(LEAKED_FAMILY_PROBES.get(substring, []))

        backend.find_entity_nodes_by_name_substring = AsyncMock(side_effect=thin_probe)
        sweep = sweep_of(backend)

        assert sweep.total_unconfirmed == 1
        assert run_main(sweep) == 0


class TestTheTwoOutputShapes:
    """--json is what a tool consumes and the summary is what a human reads;
    both are produced by main() and neither had an assertion before."""

    def test_json_round_trips_and_carries_the_whole_aggregate(self, capsys):
        """`asdict` over a nested tree of frozen dataclasses with
        ``default=str`` is easy to get subtly wrong — an unserializable field
        would raise at the very end of a long sweep, after the reads are paid
        for."""
        run_main(sweep_of(make_three_graph_backend()), ['--json'])

        emitted = json.loads(capsys.readouterr().out)

        assert set(emitted) == {
            'graphs', 'failures', 'total_families', 'total_unconfirmed',
            'complete', 'elapsed_ms',
        }
        assert emitted['total_families'] == 3
        assert [graph['group_id'] for graph in emitted['graphs']] == [
            'home', 'other', 'clean',
        ]
        assert emitted['graphs'][0]['families'][0]['variants'][0]['edge_count'] == 13

    def test_without_json_the_summary_names_each_graph_and_its_count(self, capsys):
        """'three families somewhere' is not actionable; the breakdown is what
        an operator goes and looks at."""
        run_main(sweep_of(make_three_graph_backend()))

        out = capsys.readouterr().out

        assert '3 fragmented family/families across 3 graph(s)' in out
        assert '[home=2 other=1 clean=0]' in out
        assert 'complete=True' in out

    def test_the_summary_carries_a_failed_block_naming_every_unread_graph(self, capsys):
        """A failure folded into a count reads as 'no residue here' when the
        truth is 'never looked at', so it gets its own line."""
        backend = make_sweep_backend({
            'home': whole(FIXTURE_NODES),
            'broken': RuntimeError('FalkorDB connection reset'),
        })

        run_main(sweep_of(backend))

        out = capsys.readouterr().out

        assert 'FAILED to census 1 graph(s)' in out
        assert 'broken' in out
        assert 'RuntimeError' in out
        assert 'complete=False' in out

    def test_the_summary_reports_unconfirmed_families_even_when_there_are_none(
        self, capsys,
    ):
        """Reported unconditionally: a field that appears only when non-zero is
        a field nobody knows to look for, and its normal value IS zero."""
        run_main(sweep_of(make_three_graph_backend()))

        assert 'unconfirmed=0' in capsys.readouterr().out

    def test_the_summary_survives_a_sweep_that_censused_nothing(self, capsys):
        """The empty store is the one shape a ' '.join breakdown renders as an
        empty bracket, which reads as a bug rather than as no data."""
        run_main(sweep_of(make_sweep_backend({}, graphs=[])))

        assert '(no graphs censused)' in capsys.readouterr().out
