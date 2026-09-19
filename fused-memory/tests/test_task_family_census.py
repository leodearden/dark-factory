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

from dataclasses import FrozenInstanceError
from unittest.mock import AsyncMock, MagicMock

import pytest

from fused_memory.backends.graphiti_client import PagedRead
from fused_memory.maintenance.task_family_census import TaskFamilyCensus

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
        {'uuid': 'u-605-lower', 'name': 'task 605', 'created_at': 100, 'edge_count': 13},
        {'uuid': 'u-foreign', 'name': 'reify:605', 'created_at': 20, 'edge_count': 7},
        {'uuid': 'u-605-canon', 'name': 'Task 605', 'created_at': 50, 'edge_count': 2},
        {'uuid': 'u-605-plural', 'name': 'tasks 605', 'created_at': 150, 'edge_count': 1},
        {'uuid': 'u-6051', 'name': 'Task 6051', 'created_at': 9, 'edge_count': 1},
    ],
    '900': [
        {'uuid': 'u-900-a', 'name': 'Task 900', 'created_at': 30, 'edge_count': 4},
        {'uuid': 'u-900-b', 'name': 'Task 900', 'created_at': 40, 'edge_count': 1},
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
            # a `count(DISTINCT e)` edge-count aggregate, so a `count(` test
            # would claim it and answer a four-column read with a one-column
            # census row.
            result = MagicMock()
            if 'CONTAINS' in cypher:
                result.result_set = [
                    [row['uuid'], row['name'], row['created_at'], row['edge_count']]
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
