"""Tests for scripts/census_memory_metadata.py.

Loaded via importlib so the script (not on PYTHONPATH) can be tested without
sys.path pollution -- mirrors the pattern in test_clear_malformed_empty_memory.py
and test_consolidate_namespace_families.py.

No live services: the Qdrant client and Mem0Backend are AsyncMock/MagicMock
stand-ins throughout.
"""
from __future__ import annotations

import importlib.util
import json
import sys
import types
from pathlib import Path

from fused_memory.models.enums import GRAPHITI_PRIMARY, MEM0_PRIMARY, MemoryCategory

SCRIPT_PATH = Path(__file__).parent.parent / 'scripts' / 'census_memory_metadata.py'


def _load_module() -> types.ModuleType:
    """Load census_memory_metadata.py from its file path.

    The module is registered in sys.modules under its name so that
    reflection-based decorators work correctly.
    """
    mod_name = 'census_memory_metadata'
    spec = importlib.util.spec_from_file_location(mod_name, SCRIPT_PATH)
    if spec is None or spec.loader is None:
        raise ImportError(f'Cannot load {SCRIPT_PATH}')
    module = importlib.util.module_from_spec(spec)
    sys.modules[mod_name] = module
    try:
        spec.loader.exec_module(module)  # type: ignore[union-attr]
    except Exception:
        sys.modules.pop(mod_name, None)
        raise
    return module


_mod = _load_module()


# ===========================================================================
# Tests: CENSUS_CATEGORIES
# ===========================================================================

class TestCensusCategories:
    """The census iterates all six MemoryCategory values, Mem0-primary first.

    Measured against live Qdrant during planning: sum(three Mem0-primary
    categories) falls 80 points short of the two collections' point totals,
    because dual-write records carry a Graphiti-primary category.  A
    three-category scroll therefore does NOT cover the corpus.
    """

    def test_covers_all_six_categories(self):
        assert set(_mod.CENSUS_CATEGORIES) == set(MemoryCategory)
        assert len(_mod.CENSUS_CATEGORIES) == 6

    def test_mem0_primary_categories_come_first(self):
        first_three = set(_mod.CENSUS_CATEGORIES[:3])
        last_three = set(_mod.CENSUS_CATEGORIES[3:])
        assert first_three == set(MEM0_PRIMARY)
        assert last_three == set(GRAPHITI_PRIMARY)

    def test_members_are_the_shared_enum_not_restated_strings(self):
        # INV-5 no-lockstep-duplication: the category list is derived from the
        # shared enum, never a hardcoded literal list in the script.
        for cat in _mod.CENSUS_CATEGORIES:
            assert isinstance(cat, MemoryCategory)
        assert _mod.MemoryCategory is MemoryCategory

    def test_ordering_is_deterministic(self):
        # Sorted within each band, so a re-run produces a byte-identical artifact.
        assert list(_mod.CENSUS_CATEGORIES[:3]) == sorted(MEM0_PRIMARY)
        assert list(_mod.CENSUS_CATEGORIES[3:]) == sorted(GRAPHITI_PRIMARY)


# ===========================================================================
# Tests: classify_supersedes
# ===========================================================================

class TestClassifySupersedes:
    """Shape census for the ``supersedes`` metadata key.

    PRD V1 says beta rejects scalar supersedes; the census must report the
    populations of each shape so beta can size its grandfather list.
    """

    def test_absent_when_key_missing(self):
        assert _mod.classify_supersedes({}) == 'absent'
        assert _mod.classify_supersedes({'kind': 'cycle_summary'}) == 'absent'

    def test_null_when_value_is_none(self):
        assert _mod.classify_supersedes({'supersedes': None}) == 'null'

    def test_scalar_when_value_is_a_string(self):
        uuid = '8f14e45f-ceea-467a-9b2c-64e1a2b3c4d5'
        assert _mod.classify_supersedes({'supersedes': uuid}) == 'scalar'
        assert _mod.classify_supersedes({'supersedes': ''}) == 'scalar'

    def test_list_when_value_is_a_list(self):
        assert _mod.classify_supersedes({'supersedes': ['a', 'b']}) == 'list'

    def test_empty_list_is_still_list(self):
        assert _mod.classify_supersedes({'supersedes': []}) == 'list'

    def test_other_for_int_dict_and_bool(self):
        assert _mod.classify_supersedes({'supersedes': 3}) == 'other'
        assert _mod.classify_supersedes({'supersedes': {'a': 1}}) == 'other'
        assert _mod.classify_supersedes({'supersedes': True}) == 'other'

    def test_does_not_mutate_payload(self):
        payload = {'supersedes': ['a']}
        _mod.classify_supersedes(payload)
        assert payload == {'supersedes': ['a']}


# ===========================================================================
# Tests: classify_uuid_member
# ===========================================================================

class TestClassifyUuidMember:
    """Per-member shape of a ``supersedes`` pointer."""

    def test_full_uuid(self):
        assert _mod.classify_uuid_member('8f14e45f-ceea-467a-9b2c-64e1a2b3c4d5') == 'full_uuid'
        # Uppercase is still a canonical UUID rendering.
        assert _mod.classify_uuid_member('8F14E45F-CEEA-467A-9B2C-64E1A2B3C4D5') == 'full_uuid'

    def test_short_hex(self):
        # The malformed member shape PRD V1 says beta must reject.
        assert _mod.classify_uuid_member('8f14e45f') == 'short_hex'
        assert _mod.classify_uuid_member('abc123') == 'short_hex'
        assert _mod.classify_uuid_member('8f14e45fceea467a9b2c') == 'short_hex'

    def test_other_for_non_str(self):
        assert _mod.classify_uuid_member(None) == 'other'
        assert _mod.classify_uuid_member(3) == 'other'
        assert _mod.classify_uuid_member(['a']) == 'other'
        assert _mod.classify_uuid_member({'a': 1}) == 'other'

    def test_other_for_unrecognised_string(self):
        assert _mod.classify_uuid_member('') == 'other'
        assert _mod.classify_uuid_member('not-a-uuid-at-all') == 'other'
        assert _mod.classify_uuid_member('cycle_summary') == 'other'
        # 32 hex chars unhyphenated is not the canonical rendering, and is not
        # "shorter than 32 chars" either -- it falls to 'other'.
        assert _mod.classify_uuid_member('8f14e45fceea467a9b2c64e1a2b3c4d5') == 'other'


# ===========================================================================
# Tests: CategoryCensus
# ===========================================================================

UUID_A = '8f14e45f-ceea-467a-9b2c-64e1a2b3c4d5'
UUID_B = '1c383cd3-0b7c-298a-b7ad-9f1f2a3b4c5d'


class TestCategoryCensusRecords:
    """Record and key population."""

    def test_records_counts_payloads(self):
        c = _mod.CategoryCensus()
        c.add({'kind': 'cycle_summary'})
        c.add({})
        assert c.records == 2

    def test_empty_payload_counted_without_inventing_keys(self):
        c = _mod.CategoryCensus()
        c.add({})
        assert c.records == 1
        assert dict(c.key_counts) == {}
        assert c.kind_missing == 1

    def test_key_counts_counts_each_top_level_key_once_per_payload(self):
        c = _mod.CategoryCensus()
        c.add({'kind': 'x', 'data': 'text', 'hash': 'abc'})
        c.add({'kind': 'y', 'data': 'more'})
        assert c.key_counts['kind'] == 2
        assert c.key_counts['data'] == 2
        assert c.key_counts['hash'] == 1

    def test_key_counts_includes_mem0_managed_keys_no_allowlist_filtering(self):
        # INV-5: the script does NOT copy _MEM0_MANAGED_METADATA_KEYS; it
        # reports the raw population and lets beta subtract with its own
        # single-home constant.
        c = _mod.CategoryCensus()
        c.add({'data': 'text', 'hash': 'h', 'created_at': 'ts', 'user_id': 'u'})
        for key in ('data', 'hash', 'created_at', 'user_id'):
            assert c.key_counts[key] == 1

    def test_add_does_not_mutate_payload(self):
        payload = {'kind': 'x', 'supersedes': [UUID_A], 'canonical': True}
        before = {'kind': 'x', 'supersedes': [UUID_A], 'canonical': True}
        c = _mod.CategoryCensus()
        c.add(payload)
        assert payload == before


class TestCategoryCensusKindAndSource:
    """kind / source vocabulary axes, plus the source-set-but-kind-missing drift."""

    def test_kind_counts_and_kind_missing(self):
        c = _mod.CategoryCensus()
        c.add({'kind': 'cycle_summary'})
        c.add({'kind': 'cycle_summary'})
        c.add({'kind': 'stage1_flag_marker'})
        c.add({'source': 'x'})
        assert c.kind_counts['cycle_summary'] == 2
        assert c.kind_counts['stage1_flag_marker'] == 1
        assert c.kind_missing == 1

    def test_source_counts(self):
        c = _mod.CategoryCensus()
        c.add({'source': 'stage1_flag_marker', 'kind': 'k'})
        c.add({'source': 'stage1_flag_marker'})
        c.add({'source': 'recon'})
        assert c.source_counts['stage1_flag_marker'] == 2
        assert c.source_counts['recon'] == 1

    def test_source_without_kind_is_a_per_source_breakdown(self):
        # tools.py:1595-1597 drift: source set, kind absent.
        c = _mod.CategoryCensus()
        c.add({'source': 'stage1_flag_marker'})
        c.add({'source': 'stage1_flag_marker'})
        c.add({'source': 'recon'})
        assert c.source_without_kind['stage1_flag_marker'] == 2
        assert c.source_without_kind['recon'] == 1

    def test_source_without_kind_zero_when_both_present(self):
        c = _mod.CategoryCensus()
        c.add({'source': 'stage1_flag_marker', 'kind': 'stage1_flag_marker'})
        assert c.source_without_kind['stage1_flag_marker'] == 0
        assert sum(c.source_without_kind.values()) == 0

    def test_kind_missing_not_counted_when_source_absent_too(self):
        c = _mod.CategoryCensus()
        c.add({'data': 'text'})
        assert c.kind_missing == 1
        assert sum(c.source_without_kind.values()) == 0


class TestCategoryCensusOccurrenceAxes:
    """topic / canonical / parent_id."""

    def test_topic_and_parent_id_presence(self):
        c = _mod.CategoryCensus()
        c.add({'topic': 'merge_lane', 'parent_id': 'p1'})
        c.add({'topic': 'merge_lane'})
        c.add({})
        assert c.topic_present == 2
        assert c.parent_id_present == 1
        assert c.topic_values['merge_lane'] == 2

    def test_canonical_splits_true_false_and_non_bool(self):
        c = _mod.CategoryCensus()
        c.add({'canonical': True})
        c.add({'canonical': True})
        c.add({'canonical': False})
        c.add({'canonical': 'true'})
        c.add({'canonical': 1})
        c.add({})
        assert c.canonical_true == 2
        assert c.canonical_false == 1
        # A string 'true' and an int 1 are both non-bool -- exactly the
        # coercion drift beta needs to see.
        assert c.canonical_non_bool == 2


class TestCategoryCensusSupersedes:
    """supersedes shape / member-shape / list-length census."""

    def test_shapes_tallied(self):
        c = _mod.CategoryCensus()
        c.add({})
        c.add({'supersedes': None})
        c.add({'supersedes': UUID_A})
        c.add({'supersedes': [UUID_A, UUID_B]})
        c.add({'supersedes': 7})
        assert c.supersedes_shapes['absent'] == 1
        assert c.supersedes_shapes['null'] == 1
        assert c.supersedes_shapes['scalar'] == 1
        assert c.supersedes_shapes['list'] == 1
        assert c.supersedes_shapes['other'] == 1

    def test_member_shapes_across_list_members(self):
        c = _mod.CategoryCensus()
        c.add({'supersedes': [UUID_A, 'abc123', 7]})
        assert c.supersedes_member_shapes['full_uuid'] == 1
        assert c.supersedes_member_shapes['short_hex'] == 1
        assert c.supersedes_member_shapes['other'] == 1

    def test_scalar_value_counted_as_a_lone_member(self):
        c = _mod.CategoryCensus()
        c.add({'supersedes': UUID_A})
        assert c.supersedes_member_shapes['full_uuid'] == 1

    def test_list_lengths_tallied(self):
        c = _mod.CategoryCensus()
        c.add({'supersedes': []})
        c.add({'supersedes': [UUID_A]})
        c.add({'supersedes': [UUID_A, UUID_B]})
        c.add({'supersedes': [UUID_A, UUID_B]})
        assert c.supersedes_list_lengths[0] == 1
        assert c.supersedes_list_lengths[1] == 1
        assert c.supersedes_list_lengths[2] == 2

    def test_absent_and_null_contribute_no_members_or_lengths(self):
        c = _mod.CategoryCensus()
        c.add({})
        c.add({'supersedes': None})
        assert sum(c.supersedes_member_shapes.values()) == 0
        assert sum(c.supersedes_list_lengths.values()) == 0


class TestCategoryCensusMerge:
    """Rollups for per-project and grand-total aggregation."""

    def test_merge_sums_every_counter_and_scalar(self):
        a = _mod.CategoryCensus()
        a.add({'kind': 'k1', 'source': 's', 'topic': 't', 'canonical': True})
        a.add({'supersedes': [UUID_A]})

        b = _mod.CategoryCensus()
        b.add({'kind': 'k1', 'parent_id': 'p'})
        b.add({'source': 's2', 'canonical': False})

        a.merge(b)
        assert a.records == 4
        assert a.kind_counts['k1'] == 2
        assert a.kind_missing == 2
        assert a.source_counts['s'] == 1
        assert a.source_counts['s2'] == 1
        assert a.source_without_kind['s2'] == 1
        assert a.topic_present == 1
        assert a.parent_id_present == 1
        assert a.canonical_true == 1
        assert a.canonical_false == 1
        assert a.supersedes_list_lengths[1] == 1
        assert a.key_counts['kind'] == 2

    def test_merge_does_not_mutate_the_other_census(self):
        a = _mod.CategoryCensus()
        a.add({'kind': 'k'})
        b = _mod.CategoryCensus()
        b.add({'kind': 'k'})
        a.merge(b)
        assert b.records == 1
        assert b.kind_counts['k'] == 1


# ===========================================================================
# Tests: build_report
# ===========================================================================

OBS = MemoryCategory.observations_and_summaries.value
PROC = MemoryCategory.procedural_knowledge.value
PREF = MemoryCategory.preferences_and_norms.value
DEC = MemoryCategory.decisions_and_rationale.value


def _census(payloads: list[dict]) -> object:
    c = _mod.CategoryCensus()
    for p in payloads:
        c.add(p)
    return c


def _coverage(
    collection: str,
    collection_points: int,
    categories: dict[str, tuple[int, int]],
) -> dict:
    """Build a per-project coverage record: {category: (expected, scrolled)}."""
    return {
        'collection': collection,
        'collection_points': collection_points,
        'categories': {
            cat: {'expected': exp, 'scrolled': scr}
            for cat, (exp, scr) in categories.items()
        },
    }


def _entries(table: dict) -> list[tuple]:
    return [(e['value'], e['count']) for e in table['entries']]


class TestBuildReportShape:
    """Serialisability, cells, and the generation-parameter block."""

    def test_json_round_trips(self):
        cells = {'dark_factory': {OBS: _census([{'kind': 'cycle_summary'}])}}
        cov = {'dark_factory': _coverage('fused_dark_factory', 1, {OBS: (1, 1)})}
        report = _mod.build_report(cells, cov, top_n=50)
        assert json.loads(json.dumps(report)) == report

    def test_has_schema_version_and_params(self):
        cells = {'dark_factory': {OBS: _census([{}])}}
        cov = {'dark_factory': _coverage('fused_dark_factory', 1, {OBS: (1, 1)})}
        report = _mod.build_report(cells, cov, top_n=7, page_size=250)
        assert report['schema_version'] >= 1
        assert report['params']['top_n'] == 7
        assert report['params']['page_size'] == 250
        assert report['params']['projects'] == ['dark_factory']
        assert report['params']['categories'] == [OBS]

    def test_cell_per_project_and_category(self):
        cells = {
            'dark_factory': {OBS: _census([{}, {}]), PROC: _census([{}])},
            'reify': {OBS: _census([{}, {}, {}])},
        }
        cov = {
            'dark_factory': _coverage('fused_dark_factory', 3, {OBS: (2, 2), PROC: (1, 1)}),
            'reify': _coverage('fused_reify', 3, {OBS: (3, 3)}),
        }
        report = _mod.build_report(cells, cov, top_n=50)
        assert report['projects']['dark_factory']['categories'][OBS]['records'] == 2
        assert report['projects']['dark_factory']['categories'][PROC]['records'] == 1
        assert report['projects']['reify']['categories'][OBS]['records'] == 3


class TestBuildReportRollups:
    """Per-project and grand-total counters equal the sum of their cells."""

    def test_project_rollup_sums_its_cells(self):
        cells = {
            'dark_factory': {
                OBS: _census([{'kind': 'cycle_summary'}, {'kind': 'cycle_summary'}]),
                PROC: _census([{'kind': 'gotcha'}, {}]),
            },
        }
        cov = {'dark_factory': _coverage('fused_dark_factory', 4, {OBS: (2, 2), PROC: (2, 2)})}
        report = _mod.build_report(cells, cov, top_n=50)
        total = report['projects']['dark_factory']['total']
        assert total['records'] == 4
        assert dict(_entries(total['kind'])) == {'cycle_summary': 2, 'gotcha': 1}
        assert total['kind_missing'] == 1

    def test_grand_total_sums_across_projects(self):
        cells = {
            'dark_factory': {OBS: _census([{'kind': 'k'}])},
            'reify': {OBS: _census([{'kind': 'k'}, {'kind': 'other'}])},
        }
        cov = {
            'dark_factory': _coverage('fused_dark_factory', 1, {OBS: (1, 1)}),
            'reify': _coverage('fused_reify', 2, {OBS: (2, 2)}),
        }
        report = _mod.build_report(cells, cov, top_n=50)
        assert report['grand_total']['records'] == 3
        assert dict(_entries(report['grand_total']['kind'])) == {'k': 2, 'other': 1}


class TestBuildReportDeterminism:
    """A re-run must produce a byte-identical artifact."""

    def test_value_tables_sorted_by_count_desc_then_value_asc(self):
        cells = {
            'dark_factory': {
                OBS: _census(
                    [{'kind': 'b'}] * 3 + [{'kind': 'a'}] * 3 + [{'kind': 'c'}] * 5,
                ),
            },
        }
        cov = {'dark_factory': _coverage('fused_dark_factory', 11, {OBS: (11, 11)})}
        report = _mod.build_report(cells, cov, top_n=50)
        table = report['projects']['dark_factory']['categories'][OBS]['kind']
        assert _entries(table) == [('c', 5), ('a', 3), ('b', 3)]

    def test_same_input_renders_identical_json_twice(self):
        cells = {'dark_factory': {OBS: _census([{'kind': 'a'}, {'kind': 'b'}, {'topic': 't'}])}}
        cov = {'dark_factory': _coverage('fused_dark_factory', 3, {OBS: (3, 3)})}
        first = json.dumps(_mod.build_report(cells, cov, top_n=50), sort_keys=False)
        second = json.dumps(_mod.build_report(cells, cov, top_n=50), sort_keys=False)
        assert first == second


class TestBuildReportTruncation:
    """Capped value tables disclose the cap; a long tail is never mistaken
    for a complete one."""

    def test_long_table_truncated_with_disclosure(self):
        payloads = [{'kind': f'k{i:02d}'} for i in range(10)]
        cells = {'dark_factory': {OBS: _census(payloads)}}
        cov = {'dark_factory': _coverage('fused_dark_factory', 10, {OBS: (10, 10)})}
        report = _mod.build_report(cells, cov, top_n=3)
        table = report['projects']['dark_factory']['categories'][OBS]['kind']
        assert len(table['entries']) == 3
        assert table['distinct_total'] == 10
        assert table['truncated_values'] is True

    def test_short_table_not_flagged_truncated(self):
        cells = {'dark_factory': {OBS: _census([{'kind': 'a'}, {'kind': 'b'}])}}
        cov = {'dark_factory': _coverage('fused_dark_factory', 2, {OBS: (2, 2)})}
        report = _mod.build_report(cells, cov, top_n=50)
        table = report['projects']['dark_factory']['categories'][OBS]['kind']
        assert table['distinct_total'] == 2
        assert table['truncated_values'] is False
        assert len(table['entries']) == 2

    def test_table_exactly_at_top_n_is_not_flagged(self):
        cells = {'dark_factory': {OBS: _census([{'kind': 'a'}, {'kind': 'b'}])}}
        cov = {'dark_factory': _coverage('fused_dark_factory', 2, {OBS: (2, 2)})}
        report = _mod.build_report(cells, cov, top_n=2)
        table = report['projects']['dark_factory']['categories'][OBS]['kind']
        assert table['truncated_values'] is False

    def test_key_table_also_capped_and_disclosed(self):
        payloads = [{f'key{i:02d}': 1 for i in range(10)}]
        cells = {'dark_factory': {OBS: _census(payloads)}}
        cov = {'dark_factory': _coverage('fused_dark_factory', 1, {OBS: (1, 1)})}
        report = _mod.build_report(cells, cov, top_n=4)
        table = report['projects']['dark_factory']['categories'][OBS]['keys']
        assert len(table['entries']) == 4
        assert table['distinct_total'] == 10
        assert table['truncated_values'] is True


class TestBuildReportCoverage:
    """INV-2 no-silent-fail: every enumeration shortfall is named, never swallowed."""

    def test_complete_when_every_cell_agrees_and_nothing_uncovered(self):
        cells = {'dark_factory': {OBS: _census([{}, {}]), PROC: _census([{}])}}
        cov = {'dark_factory': _coverage('fused_dark_factory', 3, {OBS: (2, 2), PROC: (1, 1)})}
        report = _mod.build_report(cells, cov, top_n=50)
        assert report['coverage']['complete'] is True
        assert report['coverage']['deltas'] == []
        assert report['coverage']['projects']['dark_factory']['uncovered_points'] == 0

    def test_per_cell_expected_vs_scrolled_recorded(self):
        cells = {'dark_factory': {OBS: _census([{}, {}])}}
        cov = {'dark_factory': _coverage('fused_dark_factory', 2, {OBS: (2, 2)})}
        report = _mod.build_report(cells, cov, top_n=50)
        cell_cov = report['coverage']['projects']['dark_factory']['categories'][OBS]
        assert cell_cov['expected'] == 2
        assert cell_cov['scrolled'] == 2
        assert cell_cov['delta'] == 0
        assert cell_cov['complete'] is True

    def test_under_enumerated_cell_marks_incomplete_with_named_delta(self):
        cells = {'dark_factory': {OBS: _census([{}, {}])}}
        cov = {'dark_factory': _coverage('fused_dark_factory', 5, {OBS: (5, 2)})}
        report = _mod.build_report(cells, cov, top_n=50)
        coverage = report['coverage']
        assert coverage['complete'] is False
        cell_cov = coverage['projects']['dark_factory']['categories'][OBS]
        assert cell_cov['expected'] == 5
        assert cell_cov['scrolled'] == 2
        assert cell_cov['delta'] == -3
        assert cell_cov['complete'] is False
        named = [d for d in coverage['deltas'] if d['kind'] == 'category_shortfall']
        assert len(named) == 1
        assert named[0]['project_id'] == 'dark_factory'
        assert named[0]['category'] == OBS
        assert named[0]['delta'] == -3

    def test_uncovered_points_surfaced_for_the_measured_live_shape(self):
        # Measured 2026-07-29: sum(three Mem0-primary categories) = 29,872 vs
        # 29,951 points in fused_reify -- an 80-point dual-write residue that
        # must be surfaced, not swallowed.
        cells = {
            'reify': {
                OBS: _census([]),
                PROC: _census([]),
                PREF: _census([]),
            },
        }
        cov = {
            'reify': _coverage(
                'fused_reify',
                29951,
                {OBS: (24408, 24408), PROC: (3981, 3981), PREF: (1483, 1483)},
            ),
        }
        report = _mod.build_report(cells, cov, top_n=50)
        coverage = report['coverage']
        proj = coverage['projects']['reify']
        assert proj['counted'] == 29872
        assert proj['collection_points'] == 29951
        assert proj['uncovered_points'] == 79
        assert proj['complete'] is False
        assert coverage['complete'] is False
        named = [d for d in coverage['deltas'] if d['kind'] == 'uncovered_points']
        assert len(named) == 1
        assert named[0]['project_id'] == 'reify'
        assert named[0]['delta'] == 79

    def test_all_six_categories_close_the_residue(self):
        # Adding the dual-write categories accounts for the remainder, and
        # coverage goes complete.
        cells = {'reify': {OBS: _census([]), PROC: _census([]), PREF: _census([]), DEC: _census([])}}
        cov = {
            'reify': _coverage(
                'fused_reify',
                100,
                {OBS: (60, 60), PROC: (20, 20), PREF: (10, 10), DEC: (10, 10)},
            ),
        }
        report = _mod.build_report(cells, cov, top_n=50)
        assert report['coverage']['projects']['reify']['uncovered_points'] == 0
        assert report['coverage']['complete'] is True


# ===========================================================================
# Tests: render_markdown
# ===========================================================================

def _rich_report(top_n: int = 50, *, complete: bool = True) -> dict:
    """A report exercising every section render_markdown must emit."""
    cells = {
        'dark_factory': {
            OBS: _census([
                {'kind': 'cycle_summary', 'topic': 'merge_lane', 'canonical': True},
                {'kind': 'cycle_summary', 'supersedes': [UUID_A, 'abc123']},
                {'source': 'stage1_flag_marker', 'parent_id': 'p1'},
                {'source': 'stage1_flag_marker', 'supersedes': UUID_B},
                {'canonical': 'yes'},
            ]),
            PROC: _census([{'kind': 'gotcha'}, {'supersedes': None}]),
        },
    }
    scrolled_obs = 5 if complete else 2
    cov = {
        'dark_factory': _coverage(
            'fused_dark_factory', 7, {OBS: (5, scrolled_obs), PROC: (2, 2)},
        ),
    }
    return _mod.build_report(cells, cov, top_n=top_n, page_size=1000)


class TestRenderMarkdownSections:
    """Every section the artifact must carry, driven off the report dict."""

    def test_header_names_the_prd_leaf(self):
        md = _mod.render_markdown(_rich_report())
        assert md.startswith('#')
        assert 'memory-metadata-vocabulary' in md

    def test_record_counts_per_project_and_category(self):
        md = _mod.render_markdown(_rich_report())
        assert 'dark_factory' in md
        assert OBS in md
        assert PROC in md

    def test_key_population_table(self):
        md = _mod.render_markdown(_rich_report())
        assert 'key' in md.lower()
        assert 'supersedes' in md

    def test_kind_table_and_missing_count(self):
        md = _mod.render_markdown(_rich_report())
        assert 'cycle_summary' in md
        assert 'gotcha' in md
        assert 'kind' in md.lower()
        assert 'missing' in md.lower()

    def test_supersedes_shape_table_with_member_and_length_breakdowns(self):
        md = _mod.render_markdown(_rich_report())
        for label in ('absent', 'null', 'scalar', 'list'):
            assert label in md
        assert 'full_uuid' in md
        assert 'short_hex' in md
        assert 'length' in md.lower()

    def test_topic_canonical_parent_id_occurrences(self):
        md = _mod.render_markdown(_rich_report())
        assert 'topic' in md.lower()
        assert 'canonical' in md.lower()
        assert 'parent_id' in md

    def test_source_set_but_kind_missing_section_lists_offending_sources(self):
        md = _mod.render_markdown(_rich_report())
        assert 'stage1_flag_marker' in md
        lowered = md.lower()
        assert 'source' in lowered
        # The drift section must name the condition, not just the value.
        assert 'without' in lowered or 'missing' in lowered


class TestRenderMarkdownDisclosure:
    """Truncation and incomplete coverage must be visible in the markdown twin."""

    def test_truncated_table_renders_its_disclosure(self):
        md = _mod.render_markdown(_rich_report(top_n=1))
        lowered = md.lower()
        assert 'truncated' in lowered
        assert 'distinct' in lowered

    def test_untruncated_report_has_no_truncation_note(self):
        md = _mod.render_markdown(_rich_report(top_n=500))
        assert 'truncated' not in md.lower()

    def test_incomplete_coverage_renders_a_warning_naming_the_deltas(self):
        md = _mod.render_markdown(_rich_report(complete=False))
        lowered = md.lower()
        assert 'incomplete' in lowered or 'warning' in lowered
        # The named delta itself, not just a flag.
        assert 'category_shortfall' in md or '-3' in md
        assert OBS in md

    def test_complete_coverage_renders_no_warning(self):
        md = _mod.render_markdown(_rich_report(complete=True))
        assert 'category_shortfall' not in md
        assert 'INCOMPLETE' not in md

    def test_uncovered_points_residue_rendered(self):
        cells = {'reify': {OBS: _census([])}}
        cov = {'reify': _coverage('fused_reify', 100, {OBS: (20, 20)})}
        md = _mod.render_markdown(_mod.build_report(cells, cov, top_n=50))
        assert 'uncovered' in md.lower()
        assert '80' in md


class TestRenderMarkdownDeterminism:
    def test_same_report_renders_identical_string_twice(self):
        report = _rich_report()
        assert _mod.render_markdown(report) == _mod.render_markdown(report)

    def test_returns_a_string(self):
        assert isinstance(_mod.render_markdown(_rich_report()), str)
