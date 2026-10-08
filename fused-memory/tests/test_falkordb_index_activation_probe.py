"""Unit tests for scripts/falkordb_index_activation_probe.py (task 3711, PRD ζ).

The script is the read-only measurement behind
``plans/falkordb-index-activation-run/``. These tests drive its public
functions only: no network, no FalkorDB, no MCP server, and no
``integration`` marker.
"""
from __future__ import annotations

import functools
import types
from pathlib import Path

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
    def test_the_module_uses_alphas_expected_set_not_a_copy(self):
        assert _mod().expected_index_set is falkor_indices.expected_index_set

    def test_a_complete_status_reports_the_full_expected_total(self):
        expected = expected_index_set()

        status = _mod().graph_index_status('dark_factory', _records(expected), expected)

        assert status.expected_total == len(expected_index_set())


class TestRequireKnownProjectRoots:
    def test_an_empty_registry_input_raises_naming_the_variable_and_its_source(self):
        with pytest.raises(_mod().RegistryUnavailableError) as excinfo:
            _mod().require_known_project_roots([])

        message = str(excinfo.value)
        assert 'DASHBOARD_KNOWN_PROJECT_ROOTS' in message
        assert 'systemctl --user show fused-memory.service -p Environment' in message

    def test_a_non_empty_list_passes_through(self):
        roots = ['/home/leo/src/dark-factory', '/home/leo/src/reify']

        assert _mod().require_known_project_roots(roots) == roots
