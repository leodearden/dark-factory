"""Unit tests for scripts/falkordb_index_activation_probe.py (task 3711, PRD ζ).

The script is the read-only measurement behind
``plans/falkordb-index-activation-run/``. These tests drive its public
functions only: no network, no FalkorDB, no MCP server, and no
``integration`` marker.
"""
from __future__ import annotations

import contextlib
import functools
import json
import types
from pathlib import Path

import anyio
import pytest
from _falkor_index_doubles import rows_for
from _fm_helpers import load_script_module
from test_falkor_indices import LIVE_HEADER

from fused_memory.backends import falkor_indices
from fused_memory.backends.falkor_indices import IndexHeaderShapeError, expected_index_set

SCRIPT_PATH = Path(__file__).parent.parent / 'scripts' / 'falkordb_index_activation_probe.py'


@functools.cache
def _mod() -> types.ModuleType:
    return load_script_module(SCRIPT_PATH, mod_name='falkordb_index_activation_probe')


def _edge(uuid: str, fact: str, *, invalid_at: str | None = None, expired_at: str | None = None):
    return _mod().FulltextEdge(uuid=uuid, fact=fact, invalid_at=invalid_at, expired_at=expired_at)


def _hit(content: str, *, source_store: str = 'graphiti'):
    return _mod().SearchHit(id='r', source_store=source_store, content=content, invalid_at=None)


class TestMentionsTask:
    @pytest.mark.parametrize(
        ('text', 'task_id'),
        [
            ('Task 3127 is set correctly in relation to PRD decision D2', '3127'),
            ('...similar to what was done for tasks 2293 and 2286.', '2286'),
            ("Task 1157's fix requires handling NodeNotFoundError", '1157'),
            ('Task 5627 confirms that an isolated re-run belongs to Task 3600.', '3600'),
            ('Task 5627 confirms that an isolated re-run belongs to Task 3600.', '5627'),
            (
                'AllAccountsCappedException catch sites added in production '
                'orchestrator via task 877 (commit 371d4763)',
                '877',
            ),
            ('Task #4856 landed', '4856'),
        ],
    )
    def test_a_task_reference_mentions_the_task(self, text, task_id):
        assert _mod().mentions_task(text, task_id) is True

    @pytest.mark.parametrize(
        ('text', 'task_id'),
        [
            ('Leo ruled to raise reconciliation.stale_run_recovery_seconds from 1800 to 3600.', '3600'),
            ("The VALUE was left at 3600 pending Leo's ruling.", '3600'),
            (
                'Part 1 measurements show that task leg p50 improved from 4,878 '
                'seconds pre-fast-suite to 670 seconds after.',
                '878',
            ),
            ('Task 31270 shipped', '3127'),
            ('Task 3127 shipped', '312'),
            ('', '877'),
        ],
    )
    def test_a_bare_number_or_a_different_id_does_not(self, text, task_id):
        assert _mod().mentions_task(text, task_id) is False


class TestIsLive:
    def test_an_edge_with_neither_stamp_is_live(self):
        assert _mod().is_live(_edge('u', 'Task 1 shipped')) is True

    def test_an_invalidated_edge_is_not_live(self):
        assert _mod().is_live(_edge('u', 'Task 1 shipped', invalid_at='2026-01-01T00:00:00Z')) is False

    def test_an_expired_edge_is_not_live(self):
        assert _mod().is_live(_edge('u', 'Task 1 shipped', expired_at='2026-01-01T00:00:00Z')) is False


class TestGenuineLiveCount:
    def test_counts_only_live_edges_that_mention_the_task_as_a_task(self):
        edges = [
            _edge('a', 'Task 876 landed the retry fix'),
            _edge('b', 'Task 876 was superseded', invalid_at='2026-01-01T00:00:00Z'),
            _edge('c', 'Task 876 was retired', expired_at='2026-01-01T00:00:00Z'),
            _edge('d', 'The ceiling stays at 876 seconds'),
            _edge('e', 'task 876 and task 12 share a fixture'),
        ]

        assert _mod().genuine_live_count(edges, '876') == 2

    def test_no_edges_count_zero(self):
        assert _mod().genuine_live_count([], '876') == 0


class TestCandidateIds:
    def test_walks_outward_below_then_above(self):
        ids = list(_mod().candidate_ids('877', exclude=(), max_distance=50))

        assert ids[:4] == ['876', '878', '875', '879']

    def test_excluded_ids_are_skipped(self):
        ids = list(_mod().candidate_ids('877', exclude=('876', '879'), max_distance=3))

        assert ids == ['878', '875', '874', '880']

    def test_nothing_beyond_max_distance_is_yielded(self):
        ids = list(_mod().candidate_ids('877', exclude=(), max_distance=2))

        assert ids == ['876', '878', '875', '879']

    def test_nothing_below_one_is_yielded(self):
        ids = list(_mod().candidate_ids('2', exclude=(), max_distance=3))

        assert ids == ['1', '3', '4', '5']


class TestSelectReplacement:
    def _measure(self, table: dict[str, list]):
        calls: list[str] = []

        def measure(task_id: str):
            calls.append(task_id)
            return table.get(task_id, [])

        return measure, calls

    def test_chooses_the_first_outward_candidate_at_the_floor(self):
        table = {
            '876': [_edge('a', 'Task 876 landed')],
            '878': [_edge('b', 'Task 878 landed'), _edge('c', 'Task 878 was reviewed')],
            '875': [_edge('d', 'Task 875 landed'), _edge('e', 'Task 875 was reviewed')],
        }
        measure, _calls = self._measure(table)

        selection = _mod().select_replacement('877', measure, exclude=())

        assert selection.anchor == '877'
        assert selection.chosen == '878'

    def test_examined_holds_every_measured_candidate_in_order_ending_at_the_chosen(self):
        table = {
            '876': [_edge('a', 'Task 876 landed')],
            '878': [_edge('b', 'Task 878 landed'), _edge('c', 'Task 878 was reviewed')],
        }
        measure, calls = self._measure(table)

        selection = _mod().select_replacement('877', measure, exclude=())

        assert [c.task_id for c in selection.examined] == ['876', '878']
        assert [list(c.edges) for c in selection.examined] == [table['876'], table['878']]
        assert calls == ['876', '878']

    def test_dead_matches_and_bare_number_mentions_are_skipped(self):
        table = {
            '876': [
                _edge('a', 'Task 876 landed', invalid_at='2026-01-01T00:00:00Z'),
                _edge('b', 'Task 876 shipped', expired_at='2026-01-01T00:00:00Z'),
            ],
            '878': [_edge('c', 'improved from 4,878 seconds'), _edge('d', 'took 878 seconds')],
            '875': [_edge('e', 'Task 875 landed'), _edge('f', 'tasks 12 and 875 shipped')],
        }
        measure, _calls = self._measure(table)

        selection = _mod().select_replacement('877', measure, exclude=())

        assert selection.chosen == '875'
        assert [c.task_id for c in selection.examined] == ['876', '878', '875']

    def test_excluded_ids_are_never_measured(self):
        table = {
            '878': [_edge('b', 'Task 878 landed'), _edge('c', 'Task 878 was reviewed')],
        }
        measure, calls = self._measure(table)

        selection = _mod().select_replacement('877', measure, exclude=('876',))

        assert selection.chosen == '878'
        assert calls == ['878']

    def test_nothing_within_max_distance_raises_naming_the_anchor_and_distance(self):
        measure, _calls = self._measure({})

        with pytest.raises(_mod().NoReplacementError) as excinfo:
            _mod().select_replacement('877', measure, exclude=(), max_distance=3)

        message = str(excinfo.value)
        assert '877' in message
        assert '3' in message


class TestBriefingHits:
    def test_a_graphiti_result_that_mentions_the_task_counts(self):
        results = [_hit('Task 3127 is set correctly')]

        assert _mod().briefing_hits(results, '3127') == results
        assert _mod().query_hit(results, '3127') is True

    def test_a_mem0_result_that_mentions_the_task_does_not_count(self):
        results = [_hit('Task 3127 is set correctly', source_store='mem0')]

        assert _mod().briefing_hits(results, '3127') == []
        assert _mod().query_hit(results, '3127') is False

    def test_a_graphiti_result_that_names_the_number_only_does_not_count(self):
        results = [_hit('raised the ceiling to 3127 seconds')]

        assert _mod().query_hit(results, '3127') is False


class TestBriefingVerdict:
    @pytest.mark.parametrize(('hit_count', 'passed'), [(5, True), (4, True), (3, False)])
    def test_passes_at_the_floor_and_fails_below_it(self, hit_count, passed):
        ids = ['1', '2', '3', '4', '5']
        hits_by_id = {task_id: index < hit_count for index, task_id in enumerate(ids)}

        verdict = _mod().briefing_verdict(hits_by_id, floor=4)

        assert verdict.passed is passed
        assert verdict.hits == hit_count
        assert verdict.total == 5
        assert verdict.floor == 4


def _records(specs, *, header=LIVE_HEADER):
    return _mod().index_records(header, rows_for(specs))


class TestIndexRecords:
    def test_maps_a_live_header_result_by_name(self):
        expected = expected_index_set()

        records = _records(expected)

        assert records
        for record in records:
            assert set(record) == {'label', 'field', 'type', 'entity_type', 'status'}
            assert record['entity_type'] in {'NODE', 'RELATIONSHIP'}
            assert record['status'] == 'OPERATIONAL'
        assert falkor_indices.normalize_index_records(records) == expected

    def test_a_reordered_header_yields_the_same_records(self):
        rows = rows_for(expected_index_set())
        order = list(reversed(range(len(LIVE_HEADER))))
        header = [LIVE_HEADER[i] for i in order]
        reordered_rows = [[row[i] for i in order] for row in rows]

        assert _mod().index_records(header, reordered_rows) == _mod().index_records(LIVE_HEADER, rows)

    def test_a_header_without_status_raises(self):
        header = [column for column in LIVE_HEADER if column[1] != 'status']

        with pytest.raises(IndexHeaderShapeError):
            _mod().index_records(header, [])


class TestGraphIndexStatus:
    def test_an_absent_graph_is_not_present_and_not_counted_incomplete(self):
        status = _mod().graph_index_status('autotrade', None, expected_index_set())

        assert status.group_id == 'autotrade'
        assert status.present is False
        assert _mod().incomplete_graph_ids([status]) == []

    def test_every_expected_spec_operational_is_complete(self):
        expected = expected_index_set()

        status = _mod().graph_index_status('dark_factory', _records(expected), expected)

        assert status.present is True
        assert status.complete is True
        assert list(status.missing) == []
        assert list(status.unsettled) == []
        assert _mod().incomplete_graph_ids([status]) == []

    def test_a_missing_spec_is_incomplete_and_named(self):
        expected = expected_index_set()
        dropped = sorted(expected)[0]

        status = _mod().graph_index_status('reify', _records(expected - {dropped}), expected)

        assert status.complete is False
        assert dropped in status.missing
        assert _mod().incomplete_graph_ids([status]) == ['reify']

    def test_an_index_under_construction_is_incomplete_and_named(self):
        expected = expected_index_set()
        records = _records(expected)
        building = '[Indexing] 3/9: UNDER CONSTRUCTION'
        records[0] = {**records[0], 'status': building}

        status = _mod().graph_index_status('reify', records, expected)

        assert status.complete is False
        assert (records[0]['label'], building) in status.unsettled

    def test_an_operator_added_index_is_unexpected_but_still_complete(self):
        expected = expected_index_set()
        extra = ('Entity', 'NODE', 'operator_added_field', 'RANGE')
        assert extra not in expected

        status = _mod().graph_index_status('reify', _records(expected | {extra}), expected)

        assert extra in status.unexpected
        assert status.complete is True

    def test_specs_are_recorded_sorted(self):
        expected = expected_index_set()

        status = _mod().graph_index_status('dark_factory', _records(expected), expected)

        assert list(status.actual) == sorted(expected)


class TestTheExpectedSetIsAlphas:
    def test_a_complete_status_reports_the_full_expected_total(self):
        expected = expected_index_set()

        status = _mod().graph_index_status('dark_factory', _records(expected), expected)

        assert status.expected_total == len(expected_index_set())


class TestRequireKnownProjectRoots:
    def test_an_empty_registry_input_raises_naming_the_variable(self):
        with pytest.raises(_mod().RegistryUnavailableError) as excinfo:
            _mod().require_known_project_roots([])

        assert 'DASHBOARD_KNOWN_PROJECT_ROOTS' in str(excinfo.value)

    def test_a_non_empty_list_passes_through(self):
        roots = ['/home/leo/src/dark-factory', '/home/leo/src/reify']

        assert _mod().require_known_project_roots(roots) == roots



# --- The measurement band --------------------------------------------------

MEASURED_AT = '2026-10-08T19:00:00Z'

LIVE = None
DEAD = '2026-09-01T00:00:00+00:00'

# Fulltext rows ``[uuid, fact, invalid_at, expired_at]`` keyed by queried id.
# 877 walks 876 (one genuine) then 878 (two genuine); 3600 walks 3599 (dead
# only) then 3601 (two genuine).
FULLTEXT_ROWS = {
    '876': [['e876a', 'Task 876 landed', LIVE, LIVE]],
    '878': [
        ['e878a', 'Task 878 landed', LIVE, LIVE],
        ['e878b', 'Task 878 was reviewed', LIVE, LIVE],
        ['e878c', 'improved from 4,878 seconds', LIVE, LIVE],
    ],
    '3599': [['e3599a', 'Task 3599 landed', DEAD, LIVE]],
    '3601': [
        ['e3601a', 'Task 3601 landed', LIVE, LIVE],
        ['e3601b', 'tasks 12 and 3601 shipped', LIVE, DEAD],
        ['e3601c', 'Task 3601 was reviewed', LIVE, LIVE],
    ],
    '877': [['e877a', 'via task 877 (commit 371d4763)', LIVE, LIVE]],
    '3600': [['e3600a', 'raised the ceiling to 3600', LIVE, LIVE]],
    '2286': [['e2286a', 'tasks 2293 and 2286 shipped', LIVE, LIVE]],
    '3127': [['e3127a', 'Task 3127 is set correctly', LIVE, LIVE]],
    '1157': [['e1157a', "Task 1157's fix", LIVE, LIVE]],
}

REBASELINED_IDS = ('878', '2286', '3127', '3601', '1157')


def _graphiti(task_id: str, content: str):
    return {'id': f'g-{task_id}', 'source_store': 'graphiti', 'content': content, 'temporal': None}


class FakeReader:
    """Serves canned ``ro_query`` results keyed by graph and query kind; logs every call."""

    def __init__(
        self, graphs, *, indexes, dup_groups=None, population=(3, 6.0), split=None, fulltext=None,
    ):
        self.graphs = list(graphs)
        self.indexes = indexes
        self.fulltext = FULLTEXT_ROWS if fulltext is None else fulltext
        self.dup_groups = dup_groups or {}
        self.population = population
        self.split = split if split is not None else [['ab', ['AB', 'Ab']]]
        self.calls: list[tuple] = []

    def list_graphs(self):
        self.calls.append(('list_graphs',))
        return list(self.graphs)

    def ro_query(self, graph, cypher, params=None):
        self.calls.append((graph, cypher, params))
        mod = _mod()
        if cypher == mod.INDEXES:
            return mod.QueryRows(header=LIVE_HEADER, rows=rows_for(self.indexes[graph]))
        if cypher == mod.FULLTEXT:
            assert graph == 'dark_factory' and params is not None
            return mod.QueryRows(header=[], rows=self.fulltext.get(params['q'], []))
        if cypher == mod.DUP_UUID:
            return mod.QueryRows(header=[], rows=[[self.dup_groups.get(graph, 0)]])
        if cypher == mod.CASE_FOLD_POPULATION:
            return mod.QueryRows(header=[], rows=[list(self.population)])
        if cypher == mod.CASE_FOLD_SPLIT:
            return mod.QueryRows(header=[], rows=self.split)
        raise AssertionError(f'unexpected query: {cypher!r}')

    def queries_for(self, graph):
        return [call[1] for call in self.calls if call[0] == graph]


class FakeSearch:
    """An async search double: canned Graphiti results for four of the five re-baselined ids."""

    def __init__(self, hit_ids=('878', '2286', '3127', '1157'), degraded=False):
        self.hit_ids = set(hit_ids)
        self.degraded = degraded
        self.calls: list[tuple] = []

    async def __call__(self, query, *, stores, limit):
        self.calls.append((query, stores, limit))
        payload: dict = {
            'results': [
                _graphiti(task_id, f'Task {task_id} shipped')
                for task_id in sorted(self.hit_ids)
                if query == _mod().RETIRED_TASK_TEMPLATE.format(task_id=task_id)
            ],
        }
        if self.degraded:
            payload['degraded'] = True
        return _mod().parse_search_payload(payload)


def _reader(**overrides):
    expected = expected_index_set()
    kwargs = {
        'graphs': ['dark_factory', 'reify', 'scratch_probe'],
        'indexes': {'dark_factory': expected, 'reify': expected},
    }
    kwargs.update(overrides)
    return FakeReader(**kwargs)


REGISTRY = frozenset({'dark_factory', 'reify', 'autotrade'})


async def _measure(reader=None, search=None, registry=REGISTRY):
    reader = reader or _reader()
    search = search or FakeSearch()
    record = await _mod().measure(reader, search, registry=registry, measured_at=MEASURED_AT)
    return record, reader, search


class TestMeasureIndexSweep:
    @pytest.mark.asyncio
    async def test_one_status_per_registry_id_sorted(self):
        record, _reader_, _search = await _measure()

        assert [s.group_id for s in record.index_statuses] == sorted(REGISTRY)

    @pytest.mark.asyncio
    async def test_an_unlisted_registry_id_is_absent_and_never_queried(self):
        record, reader, _search = await _measure()

        (autotrade,) = [s for s in record.index_statuses if s.group_id == 'autotrade']
        assert autotrade.present is False
        assert reader.queries_for('autotrade') == []

    @pytest.mark.asyncio
    async def test_the_expected_set_is_recorded_sorted(self):
        record, _reader_, _search = await _measure()

        assert list(record.expected_index_set) == sorted(expected_index_set())


class TestMeasureE1Preflight:
    @pytest.mark.asyncio
    async def test_a_dup_uuid_count_is_recorded_for_every_listed_graph(self):
        record, reader, _search = await _measure()

        counted = {entry.group_id for entry in record.e1_preflight.dup_uuid_groups}
        assert counted == set(reader.graphs)
        assert all(entry.groups == 0 for entry in record.e1_preflight.dup_uuid_groups)

    @pytest.mark.asyncio
    async def test_complete_graphs_and_no_dup_groups_is_a_maintenance_noop(self):
        record, _reader_, _search = await _measure()

        assert record.e1_preflight.maintenance_noop is True

    @pytest.mark.asyncio
    async def test_one_dup_group_is_not_a_noop(self):
        record, _reader_, _search = await _measure(_reader(dup_groups={'scratch_probe': 1}))

        assert record.e1_preflight.maintenance_noop is False

    @pytest.mark.asyncio
    async def test_one_incomplete_graph_is_not_a_noop(self):
        expected = expected_index_set()
        short = expected - {sorted(expected)[0]}

        record, _reader_, _search = await _measure(
            _reader(indexes={'dark_factory': expected, 'reify': short})
        )

        assert record.e1_preflight.maintenance_noop is False
        assert list(record.e1_preflight.incomplete_graphs) == ['reify']


class TestMeasureReplacements:
    @pytest.mark.asyncio
    async def test_one_selection_per_drifted_id_chosen_from_the_fulltext_rows(self):
        record, _reader_, _search = await _measure()

        assert [(s.anchor, s.chosen) for s in record.replacements] == [('877', '878'), ('3600', '3601')]

    @pytest.mark.asyncio
    async def test_every_examined_candidates_raw_edges_are_recorded(self):
        record, _reader_, _search = await _measure()

        (first, second) = record.replacements
        assert [c.task_id for c in first.examined] == ['876', '878']
        assert [c.task_id for c in second.examined] == ['3599', '3601']
        assert [e.uuid for e in first.examined[1].edges] == ['e878a', 'e878b', 'e878c']
        assert second.examined[0].edges[0].invalid_at == DEAD

    def test_original_ids_and_earlier_choices_are_excluded(self):
        mod = _mod()
        qualifying = ('3600', '3601', '3604')
        rows = {
            task_id: [
                [f'{task_id}a', f'Task {task_id} landed', LIVE, LIVE],
                [f'{task_id}b', f'Task {task_id} was reviewed', LIVE, LIVE],
            ]
            for task_id in qualifying
        }

        def measure(task_id):
            return [mod.FulltextEdge(*row) for row in rows.get(task_id, [])]

        first, second = mod.select_replacements(measure, anchors=('3600', '3602'))

        assert first.chosen == '3601'
        assert second.chosen == '3604'
        assert [c.task_id for c in second.examined] == ['3603', '3604']

    @pytest.mark.asyncio
    async def test_fresh_rows_for_every_original_id_are_recorded(self):
        record, _reader_, _search = await _measure()

        assert [c.task_id for c in record.original_id_counts] == list(_mod().ORIGINAL_PROBE_IDS)
        (row_877,) = [c for c in record.original_id_counts if c.task_id == '877']
        assert [e.uuid for e in row_877.edges] == ['e877a']


class TestMeasureProbes:
    @pytest.mark.asyncio
    async def test_the_asserted_probe_fires_the_rebaselined_ids_graphiti_scoped(self):
        record, _reader_, search = await _measure()
        mod = _mod()
        probe = record.asserted_probe

        assert [q.task_id for q in probe.queries] == list(REBASELINED_IDS)
        assert probe.stores == ('graphiti',)
        assert probe.limit == 5
        for task_id in REBASELINED_IDS:
            assert (mod.RETIRED_TASK_TEMPLATE.format(task_id=task_id), ['graphiti'], 5) in search.calls

    @pytest.mark.asyncio
    async def test_the_asserted_verdict_uses_the_floor(self):
        record, _reader_, _search = await _measure()
        verdict = record.asserted_probe.verdict

        assert verdict == _mod().briefing_verdict(
            {task_id: task_id != '3601' for task_id in REBASELINED_IDS}, floor=4
        )
        assert verdict.passed is True

    @pytest.mark.asyncio
    async def test_a_failing_probe_is_recorded_not_raised(self):
        record, _reader_, _search = await _measure(search=FakeSearch(hit_ids=('878', '2286')))

        assert record.asserted_probe.verdict.passed is False
        assert record.asserted_probe.verdict.hits == 2

    @pytest.mark.asyncio
    async def test_the_original_ids_and_unscoped_variants_are_recorded_without_a_verdict(self):
        record, _reader_, search = await _measure()
        mod = _mod()

        assert [q.task_id for q in record.original_ids_probe.queries] == list(mod.ORIGINAL_PROBE_IDS)
        assert record.original_ids_probe.stores == ('graphiti',)
        assert record.original_ids_probe.verdict is None
        assert [q.task_id for q in record.unscoped_probe.queries] == list(REBASELINED_IDS)
        assert record.unscoped_probe.stores is None
        assert record.unscoped_probe.verdict is None
        unscoped_calls = [call for call in search.calls if call[1] is None]
        assert len(unscoped_calls) == len(REBASELINED_IDS)
        assert len(search.calls) == 15

    @pytest.mark.asyncio
    async def test_every_query_records_its_degraded_flag_and_raw_results(self):
        record, _reader_, _search = await _measure(search=FakeSearch(degraded=True))
        (hit_878,) = [q for q in record.asserted_probe.queries if q.task_id == '878']

        assert hit_878.degraded is True
        assert hit_878.query == _mod().RETIRED_TASK_TEMPLATE.format(task_id='878')
        (result,) = hit_878.results
        assert (result.id, result.source_store, result.content, result.invalid_at) == (
            'g-878', 'graphiti', 'Task 878 shipped', None,
        )


class TestMeasureCaseFold:
    @pytest.mark.asyncio
    async def test_the_population_query_runs_on_each_present_registered_graph_only(self):
        record, reader, _search = await _measure()
        mod = _mod()

        for graph in ('dark_factory', 'reify'):
            assert mod.CASE_FOLD_POPULATION in reader.queries_for(graph)
        assert mod.CASE_FOLD_POPULATION not in reader.queries_for('scratch_probe')
        assert [c.group_id for c in record.case_fold] == ['dark_factory', 'reify']

    @pytest.mark.asyncio
    async def test_the_split_query_carries_the_recorded_since_on_each_present_registered_graph(self):
        record, reader, _search = await _measure()
        mod = _mod()

        split_calls = [call for call in reader.calls if len(call) == 3 and call[1] == mod.CASE_FOLD_SPLIT]
        assert {call[0] for call in split_calls} == {'dark_factory', 'reify'}
        assert {c.group_id for c in record.case_fold} == {call[0] for call in split_calls}
        for c in record.case_fold:
            (params,) = [call[2] for call in split_calls if call[0] == c.group_id]
            assert params == {'since': c.split_since}
            assert c.split_since == mod.CASE_FOLD_SPLIT_SINCE

    @pytest.mark.asyncio
    async def test_keys_nodes_and_split_groups_are_recorded_raw(self):
        record, _reader_, _search = await _measure(
            _reader(population=(67, 138.0), split=[['ab', ['AB', 'Ab']], ['cd', ['CD', 'cD']]])
        )
        (dark_factory,) = [c for c in record.case_fold if c.group_id == 'dark_factory']

        assert (dark_factory.keys, dark_factory.nodes) == (67, 138.0)
        assert [(g.key, list(g.spellings)) for g in dark_factory.split_groups] == [
            ('ab', ['AB', 'Ab']), ('cd', ['CD', 'cD']),
        ]


class _ReadOnlyGraph:
    def __init__(self, name, backing: FakeReader, log: list):
        self.name = name
        self.backing = backing
        self.log = log

    def query(self, *args, **kwargs):
        raise AssertionError('the write path GRAPH.QUERY must never be called')

    def ro_query(self, q, params=None, timeout=None):
        self.log.append((q, params, timeout))
        rows = self.backing.ro_query(self.name, q, params)
        return types.SimpleNamespace(header=rows.header, result_set=rows.rows)


class _FalkorLikeClient:
    def __init__(self, backing: FakeReader):
        self.backing = backing
        self.log: list[tuple] = []

    def list_graphs(self):
        return self.backing.list_graphs()

    def select_graph(self, name):
        return _ReadOnlyGraph(name, self.backing, self.log)


class TestFalkorReadOnlyReader:
    @pytest.mark.asyncio
    async def test_measure_over_the_reader_uses_only_ro_query_with_an_explicit_timeout(self):
        client = _FalkorLikeClient(_reader())
        reader = _mod().FalkorReadOnlyReader(client)

        record, _reader_, _search = await _measure(reader=reader)

        assert record.e1_preflight.maintenance_noop is True
        assert client.log
        assert all(
            timeout is not None and timeout == _mod().READ_TIMEOUT_MS for _q, _params, timeout in client.log
        )


class TestRecordToJson:
    @pytest.mark.asyncio
    async def test_the_json_is_deterministic_loadable_and_newline_terminated(self):
        record, _reader_, _search = await _measure()
        mod = _mod()

        first = mod.record_to_json(record)
        second = mod.record_to_json(record)

        assert first == second
        assert first.endswith('\n')
        loaded = json.loads(first)
        assert loaded['measured_at'] == MEASURED_AT

    @pytest.mark.asyncio
    async def test_two_identical_runs_serialise_identically(self):
        first, _r1, _s1 = await _measure()
        second, _r2, _s2 = await _measure()

        assert _mod().record_to_json(first) == _mod().record_to_json(second)


class TestMainRefusesWithoutTheRegistry:
    def test_an_unset_registry_exits_non_zero_before_any_connection(self, monkeypatch, tmp_path):
        mod = _mod()

        def _never(*args, **kwargs):
            raise AssertionError('no adapter may be constructed before the registry guard')

        monkeypatch.delenv('DASHBOARD_KNOWN_PROJECT_ROOTS', raising=False)
        monkeypatch.setattr(mod.FalkorReadOnlyReader, 'connect', _never)
        monkeypatch.setattr(mod, 'open_mcp_search', _never)

        assert mod.main(['--out-dir', str(tmp_path)]) != 0
        assert list(tmp_path.iterdir()) == []


@contextlib.asynccontextmanager
async def _task_group_transport(_url):
    """Stands in for mcp's ``streamablehttp_client``: it yields inside a live anyio task group."""
    async with anyio.create_task_group() as group:
        group.start_soon(anyio.sleep_forever)
        try:
            yield None, None, None
        finally:
            group.cancel_scope.cancel()


def _session_answering(result):
    class _Session:
        def __init__(self, _read, _write):
            pass

        async def __aenter__(self):
            return self

        async def __aexit__(self, *_exc_info):
            return None

        async def initialize(self):
            return None

        async def call_tool(self, _name, _arguments, read_timeout_seconds=None):
            return result

    return _Session


SEARCH_ANSWERS_NOTHING = types.SimpleNamespace(isError=False, content=[], structuredContent={'results': []})
SEARCH_ERRORS = types.SimpleNamespace(isError=True, content=['backend down'], structuredContent=None)


class TestMainReportsARunFailureAsOneLine:
    """A declared run failure raised inside the MCP session ends in one ``error:`` line."""

    def _main(self, monkeypatch, tmp_path, capsys, *, reader, search_result):
        mod = _mod()
        project_root = tmp_path / 'dark_factory'
        project_root.mkdir()
        monkeypatch.setenv('DASHBOARD_KNOWN_PROJECT_ROOTS', str(project_root))
        monkeypatch.setattr(
            mod.FalkorReadOnlyReader, 'connect',
            lambda _uri, _password: mod.FalkorReadOnlyReader(_FalkorLikeClient(reader)),
        )
        monkeypatch.setattr(mod, 'streamablehttp_client', _task_group_transport)
        monkeypatch.setattr(mod, 'ClientSession', _session_answering(search_result))
        out_dir = tmp_path / 'out'

        exit_code = mod.main(['--out-dir', str(out_dir)])

        err = capsys.readouterr().err
        return exit_code, [line for line in err.splitlines() if line.startswith('error: ')], err, out_dir

    def test_a_search_tool_error_exits_run_failed_with_one_error_line(self, monkeypatch, tmp_path, capsys):
        exit_code, error_lines, err, out_dir = self._main(
            monkeypatch, tmp_path, capsys, reader=_reader(), search_result=SEARCH_ERRORS,
        )

        assert exit_code == _mod().EXIT_RUN_FAILED
        (line,) = error_lines
        assert 'backend down' in line
        assert 'Traceback' not in err
        assert not out_dir.exists()

    def test_no_replacement_exits_run_failed_with_one_error_line(self, monkeypatch, tmp_path, capsys):
        exit_code, error_lines, err, out_dir = self._main(
            monkeypatch, tmp_path, capsys,
            reader=_reader(fulltext={}), search_result=SEARCH_ANSWERS_NOTHING,
        )

        assert exit_code == _mod().EXIT_RUN_FAILED
        (line,) = error_lines
        assert '877' in line
        assert 'Traceback' not in err
        assert not out_dir.exists()


class TestParseSearchPayload:
    def test_results_are_normalised_with_invalid_at_from_temporal(self):
        payload = {
            'results': [
                {
                    'id': 'u1', 'source_store': 'graphiti', 'content': 'Task 1 landed',
                    'temporal': {'valid_at': None, 'invalid_at': '2026-01-01T00:00:00Z'},
                    'relevance_score': 0.5,
                },
                {'id': 'm1', 'source_store': 'mem0', 'content': 'a preference'},
            ],
        }

        outcome = _mod().parse_search_payload(payload)

        assert [(r.id, r.source_store, r.content, r.invalid_at) for r in outcome.results] == [
            ('u1', 'graphiti', 'Task 1 landed', '2026-01-01T00:00:00Z'),
            ('m1', 'mem0', 'a preference', None),
        ]
        assert outcome.degraded is False

    @pytest.mark.parametrize(
        'extra', [{'degraded': True}, {'failed_stores': ['mem0']}],
    )
    def test_a_degraded_or_failed_store_marks_the_outcome_degraded(self, extra):
        outcome = _mod().parse_search_payload({'results': [], **extra})

        assert outcome.degraded is True

    def test_a_result_envelope_is_unwrapped(self):
        inner = {'results': [{'id': 'u1', 'source_store': 'graphiti', 'content': 'x'}]}

        assert _mod().parse_search_payload({'result': inner}) == _mod().parse_search_payload(inner)

    @pytest.mark.parametrize(
        'payload',
        [
            {'error': 'Invalid limit', 'error_type': 'ValidationError'},
            {'results': 'not a list'},
            {'results': [{'id': 'u1'}]},
            ['not', 'a', 'mapping'],
            {'results': [{'id': 'u1', 'source_store': 'graphiti', 'content': 'x', 'temporal': 'yesterday'}]},
            {'results': [{'id': 'u1', 'source_store': 'graphiti', 'content': 'x', 'temporal': {'invalid_at': 20260101}}]},
        ],
    )
    def test_an_unrecognised_shape_raises(self, payload):
        with pytest.raises(_mod().SearchPayloadShapeError):
            _mod().parse_search_payload(payload)
