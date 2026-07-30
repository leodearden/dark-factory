"""Tests for scripts/census_memory_metadata.py.

Loaded via importlib so the script (not on PYTHONPATH) can be tested without
sys.path pollution -- mirrors the pattern in test_clear_malformed_empty_memory.py
and test_consolidate_namespace_families.py.

No live services: the Qdrant client and Mem0Backend are AsyncMock/MagicMock
stand-ins throughout.
"""
from __future__ import annotations

import asyncio
import importlib.util
import json
import sys
import types
from pathlib import Path
from unittest.mock import AsyncMock, MagicMock

import pytest

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

    def test_hex32_uuid_is_not_lumped_in_with_garbage(self):
        # uuid.UUID.hex / str(u).replace('-','') -- non-canonical but VALID.
        # A consumer told to reject 'other' must not have to guess whether
        # those members are malformed pointers or this rendering.
        assert _mod.classify_uuid_member('8f14e45fceea467a9b2c64e1a2b3c4d5') == 'hex32_uuid'
        assert _mod.classify_uuid_member('8F14E45FCEEA467A9B2C64E1A2B3C4D5') == 'hex32_uuid'

    def test_hex32_is_checked_before_the_short_hex_branch(self):
        # 32 hex chars must never fall through to short_hex (or to 'other').
        thirty_two = 'a' * 32
        assert _mod.classify_uuid_member(thirty_two) == 'hex32_uuid'
        assert _mod.classify_uuid_member('a' * 31) == 'short_hex'
        # 33+ hex chars is neither rendering.
        assert _mod.classify_uuid_member('a' * 33) == 'other'

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
        # Hex-looking but not a UUID rendering: a hyphenated non-UUID.
        assert _mod.classify_uuid_member('8f14e45f-ceea-467a-9b2c') == 'other'


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
    categories: dict[str, tuple[int, int] | tuple[int, int, int]],
) -> dict:
    """Build a per-project coverage record.

    ``{category: (expected, scrolled)}`` -- or ``(expected, scrolled, recount)``
    to model a corpus that changed under the scan. Omitted, *recount* equals
    *expected* (a stable corpus), which is what census_project records when
    nothing was written mid-census.
    """
    return {
        'collection': collection,
        'collection_points': collection_points,
        'categories': {
            cat: {
                'expected': counts[0],
                'scrolled': counts[1],
                'recount': counts[2] if len(counts) > 2 else counts[0],
            }
            for cat, counts in categories.items()
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


class TestBuildReportNeverTruncatesTheJson:
    """The JSON artifact carries the WHOLE population, whatever --top-n says.

    ``top_n`` is a markdown readability knob only. A capped JSON table would
    orphan every in-use value in its tail (leaf beta's kind registry reads
    this file) while ``coverage.complete`` still read ``true`` -- a silent
    under-report in a different coordinate from a short scroll, but the same
    failure (INV-2).
    """

    def test_long_kind_table_is_complete_despite_a_small_top_n(self):
        payloads = [{'kind': f'k{i:02d}'} for i in range(10)]
        cells = {'dark_factory': {OBS: _census(payloads)}}
        cov = {'dark_factory': _coverage('fused_dark_factory', 10, {OBS: (10, 10)})}
        report = _mod.build_report(cells, cov, top_n=3)
        table = report['projects']['dark_factory']['categories'][OBS]['kind']
        assert len(table['entries']) == 10
        assert table['distinct_total'] == 10
        assert {e['value'] for e in table['entries']} == {f'k{i:02d}' for i in range(10)}

    def test_key_table_is_complete_too(self):
        payloads = [{f'key{i:02d}': 1 for i in range(10)}]
        cells = {'dark_factory': {OBS: _census(payloads)}}
        cov = {'dark_factory': _coverage('fused_dark_factory', 1, {OBS: (1, 1)})}
        report = _mod.build_report(cells, cov, top_n=4)
        table = report['projects']['dark_factory']['categories'][OBS]['keys']
        assert len(table['entries']) == 10
        assert table['distinct_total'] == 10

    def test_distinct_total_always_equals_the_entry_count(self):
        payloads = [{'kind': f'k{i:02d}', 'source': f's{i:02d}'} for i in range(10)]
        cells = {'dark_factory': {OBS: _census(payloads)}}
        cov = {'dark_factory': _coverage('fused_dark_factory', 10, {OBS: (10, 10)})}
        report = _mod.build_report(cells, cov, top_n=1)
        cell = report['projects']['dark_factory']['categories'][OBS]
        for axis in ('keys', 'kind', 'source', 'source_without_kind', 'topic',
                     'supersedes_shapes', 'supersedes_member_shapes',
                     'supersedes_list_lengths'):
            table = cell[axis]
            assert len(table['entries']) == table['distinct_total'], axis

    def test_a_small_top_n_never_flips_coverage_incomplete(self):
        # The bug this replaces: --top-n 50 against 327 distinct kinds left
        # coverage.complete true while the artifact held 50 of them.
        payloads = [{'kind': f'k{i:03d}'} for i in range(200)]
        cells = {'dark_factory': {OBS: _census(payloads)}}
        cov = {'dark_factory': _coverage('fused_dark_factory', 200, {OBS: (200, 200)})}
        report = _mod.build_report(cells, cov, top_n=50)
        assert report['coverage']['complete'] is True
        assert report['projects']['dark_factory']['categories'][OBS]['kind'][
            'distinct_total'] == 200
        assert len(report['projects']['dark_factory']['categories'][OBS]['kind'][
            'entries']) == 200

    def test_top_n_is_recorded_in_params_for_the_markdown_render(self):
        cells = {'dark_factory': {OBS: _census([{'kind': 'a'}])}}
        cov = {'dark_factory': _coverage('fused_dark_factory', 1, {OBS: (1, 1)})}
        report = _mod.build_report(cells, cov, top_n=9)
        assert report['params']['top_n'] == 9


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

    def test_surplus_is_named_a_surplus_not_a_shortfall(self):
        # Scrolling MORE than either count expected is a different diagnosis
        # than scrolling fewer: nothing is missing. Calling it a shortfall
        # sends a reader hunting for data that is not gone.
        cells = {'dark_factory': {OBS: _census([])}}
        cov = {'dark_factory': _coverage('fused_dark_factory', 100, {OBS: (100, 105, 100)})}
        coverage = _mod.build_report(cells, cov, top_n=50)['coverage']
        assert coverage['complete'] is False
        kinds = [d['kind'] for d in coverage['deltas']]
        assert 'category_surplus' in kinds
        assert 'category_shortfall' not in kinds
        surplus = next(d for d in coverage['deltas'] if d['kind'] == 'category_surplus')
        assert surplus['delta'] == 5
        assert surplus['scrolled'] == 105

    def test_uncovered_points_surfaced_for_the_measured_live_shape(self):
        # Measured 2026-07-29: sum(three Mem0-primary categories) = 29,872 vs
        # 29,951 points in fused_reify -- a 79-point dual-write residue in
        # THAT collection (80 across both, the extra point being in
        # fused_dark_factory) that must be surfaced, not swallowed.
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


class TestBuildReportChurn:
    """A LIVE corpus moving under the scan is drift, not a truncated scroll.

    project_root is written continuously by orchestrators, so count and
    scroll are two unsynchronised reads. Treating every delta as a shortfall
    made a healthy census intermittently report itself broken and exit 1.
    """

    def test_scroll_matching_the_post_scan_recount_is_complete(self):
        # A write landed between the count and the scroll: expected 100,
        # scrolled 101, and the re-count agrees at 101. Nothing is missing.
        cells = {'dark_factory': {OBS: _census([])}}
        cov = {'dark_factory': _coverage('fused_dark_factory', 101, {OBS: (100, 101, 101)})}
        coverage = _mod.build_report(cells, cov, top_n=50)['coverage']
        cell = coverage['projects']['dark_factory']['categories'][OBS]
        assert cell['complete'] is True
        assert cell['churn'] is True
        assert cell['recount'] == 101
        assert [d for d in coverage['deltas'] if d.get('category') == OBS] == []
        # The cell-level filter above is structurally blind to a
        # collection-level delta (which carries no 'category' key), which is
        # exactly how the report-level defect hid behind a green cell. Assert
        # the REPORT verdict too, so that gap cannot reopen.
        assert coverage['complete'] is True
        assert coverage['deltas'] == []

    def test_churn_is_recorded_so_drift_is_distinguishable_from_truncation(self):
        cells = {'dark_factory': {OBS: _census([])}}
        cov = {'dark_factory': _coverage('fused_dark_factory', 101, {OBS: (100, 101, 101)})}
        churn = _mod.build_report(cells, cov, top_n=50)['coverage']['churn']
        # The same write also moves the COLLECTION total, which is recorded as
        # its own churn entry -- so select the per-cell one rather than
        # pinning the list length.
        cell_churn = [c for c in churn if c.get('category') == OBS]
        assert len(cell_churn) == 1
        assert (cell_churn[0]['expected'], cell_churn[0]['recount'],
                cell_churn[0]['scrolled']) == (100, 101, 101)

    def test_scroll_matching_the_pre_scan_count_is_also_complete(self):
        # The write landed after the scroll passed that page: scrolled agrees
        # with the pre-scan count even though the corpus has since grown.
        cells = {'dark_factory': {OBS: _census([])}}
        cov = {'dark_factory': _coverage('fused_dark_factory', 101, {OBS: (100, 100, 101)})}
        cell = _mod.build_report(cells, cov, top_n=50)['coverage'][
            'projects']['dark_factory']['categories'][OBS]
        assert cell['complete'] is True
        assert cell['churn'] is True

    def test_scroll_matching_NEITHER_count_is_still_a_shortfall(self):
        # Churn tolerance must not swallow a genuinely truncated scan.
        cells = {'dark_factory': {OBS: _census([])}}
        cov = {'dark_factory': _coverage('fused_dark_factory', 100, {OBS: (100, 40, 101)})}
        coverage = _mod.build_report(cells, cov, top_n=50)['coverage']
        assert coverage['complete'] is False
        named = [d for d in coverage['deltas'] if d['kind'] == 'category_shortfall']
        assert len(named) == 1
        assert named[0]['recount'] == 101

    def test_stable_corpus_reports_no_churn(self):
        cells = {'dark_factory': {OBS: _census([])}}
        cov = {'dark_factory': _coverage('fused_dark_factory', 100, {OBS: (100, 100, 100)})}
        coverage = _mod.build_report(cells, cov, top_n=50)['coverage']
        assert coverage['churn'] == []
        assert coverage['projects']['dark_factory']['categories'][OBS]['churn'] is False

    def test_missing_recount_falls_back_to_strict_equality(self):
        # A coverage record without a recount (an older/partial caller) must
        # not silently gain churn tolerance.
        cov = {
            'dark_factory': {
                'collection': 'fused_dark_factory',
                'collection_points': 100,
                'categories': {OBS: {'expected': 100, 'scrolled': 90}},
            },
        }
        coverage = _mod.build_report({'dark_factory': {}}, cov, top_n=50)['coverage']
        cell = coverage['projects']['dark_factory']['categories'][OBS]
        assert cell['recount'] is None
        assert cell['complete'] is False
        assert cell['churn'] is False

    # -- collection-level reconciliation ------------------------------------
    #
    # The per-cell churn tolerance above left the level ABOVE it unfixed:
    # the project residue compared sum(expected) -- the PRE-scroll counts --
    # against a `collection_points` read AFTER every scroll, i.e. two
    # unsynchronised reads of a live corpus treated as simultaneous. The
    # counted total is therefore an interval [min, max] over
    # (sum expected, sum recount), the collection total an interval over its
    # bracketing pair, and the project is covered when they INTERSECT.

    def test_collection_total_moving_under_the_scan_is_not_an_uncovered_residue(self):
        # sum(expected)=100, sum(recount)=101, collection_points=101: the one
        # extra point is the SAME write the cell already recorded as churn.
        # Charging it again as an uncovered residue made a healthy census
        # emit an artifact whose own banner declares it unusable.
        cells = {'dark_factory': {OBS: _census([])}}
        cov = {'dark_factory': _coverage('fused_dark_factory', 101, {OBS: (100, 101, 101)})}
        coverage = _mod.build_report(cells, cov, top_n=50)['coverage']
        assert coverage['complete'] is True
        assert [d for d in coverage['deltas'] if d['kind'] == 'uncovered_points'] == []
        proj = coverage['projects']['dark_factory']
        assert proj['complete'] is True
        # The residue a reader sees is the one that drove `complete`.
        assert proj['uncovered_points'] == 0
        # ...and the raw reads stay alongside it so the call is auditable.
        assert proj['collection_points'] == 101
        assert proj['counted'] == 100
        assert proj['counted_interval'] == [100, 101]
        assert proj['collection_points_interval'] == [101, 101]

    def test_collection_total_movement_is_recorded_as_churn_not_swallowed(self):
        cells = {'dark_factory': {OBS: _census([])}}
        cov = {'dark_factory': _coverage('fused_dark_factory', 101, {OBS: (100, 101, 101)})}
        coverage = _mod.build_report(cells, cov, top_n=50)['coverage']
        moved = [c for c in coverage['churn'] if c.get('kind') == 'collection_total_moved']
        assert len(moved) == 1
        assert moved[0]['project_id'] == 'dark_factory'
        assert moved[0]['collection'] == 'fused_dark_factory'
        assert moved[0]['counted_interval'] == [100, 101]
        assert moved[0]['collection_points_interval'] == [101, 101]

    def test_negative_within_window_residue_is_churn_not_a_shortfall(self):
        # A deletion landed mid-census: expected 100, scroll and re-count both
        # agree at 99, and the collection now holds 99 points. The counted
        # interval [99, 100] still contains 99, so nothing is unaccounted.
        cells = {'fused_p1': {OBS: _census([])}}
        cov = {'fused_p1': _coverage('fused_p1', 99, {OBS: (100, 99, 99)})}
        coverage = _mod.build_report(cells, cov, top_n=50)['coverage']
        assert coverage['complete'] is True
        assert coverage['deltas'] == []
        assert coverage['projects']['fused_p1']['uncovered_points'] == 0

    def test_residue_outside_the_churn_window_is_still_named(self):
        # Regression guard for the measured-live shape: a STABLE corpus
        # (expected == recount on every cell) collapses both intervals to a
        # point, so the 79-point dual-write residue must survive the churn
        # tolerance exactly as before.
        cells = {'reify': {OBS: _census([]), PROC: _census([]), PREF: _census([])}}
        cov = {
            'reify': _coverage(
                'fused_reify',
                29951,
                {OBS: (24408, 24408), PROC: (3981, 3981), PREF: (1483, 1483)},
            ),
        }
        coverage = _mod.build_report(cells, cov, top_n=50)['coverage']
        assert coverage['complete'] is False
        assert coverage['projects']['reify']['uncovered_points'] == 79
        named = [d for d in coverage['deltas'] if d['kind'] == 'uncovered_points']
        assert len(named) == 1
        assert named[0]['delta'] == 79

    def test_surviving_negative_residue_is_a_surplus_not_uncovered_points(self):
        # The counted interval sits ENTIRELY above the collection total, far
        # beyond any churn window: the categories counted more points than the
        # collection holds. That is deletions or double-counting -- sending a
        # reader hunting for an unenumerated category is a wrong diagnosis.
        cells = {'fused_p1': {OBS: _census([])}}
        cov = {'fused_p1': _coverage('fused_p1', 90, {OBS: (100, 100, 100)})}
        coverage = _mod.build_report(cells, cov, top_n=50)['coverage']
        assert coverage['complete'] is False
        kinds = [d['kind'] for d in coverage['deltas']]
        assert 'surplus_counted' in kinds
        assert 'uncovered_points' not in kinds
        surplus = next(d for d in coverage['deltas'] if d['kind'] == 'surplus_counted')
        assert surplus['delta'] == -10
        assert surplus['project_id'] == 'fused_p1'
        assert coverage['projects']['fused_p1']['uncovered_points'] == -10


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
    """Every section the artifact must carry, driven off the report dict.

    Asserted on RENDERED STRUCTURE -- literal section headings and concrete
    table rows -- not on bare words. A substring check like `'kind' in
    md.lower()` passes on essentially any census markdown ('kind' is in the
    preamble, 'key' in every header row), so it would stay green while the
    section leaf beta reads was deleted outright.
    """

    def test_header_names_the_prd_leaf(self):
        md = _mod.render_markdown(_rich_report())
        assert md.startswith('# Mem0 metadata corpus census')
        assert 'memory-metadata-vocabulary' in md

    def test_record_counts_per_project_and_category(self):
        md = _mod.render_markdown(_rich_report())
        assert '## Record counts' in md
        assert '| project | category | records |' in md
        assert f'| `dark_factory` | `{OBS}` | 5 |' in md
        assert f'| `dark_factory` | `{PROC}` | 2 |' in md
        assert '| `dark_factory` | **(all)** | **7** |' in md

    def test_key_population_table(self):
        md = _mod.render_markdown(_rich_report())
        assert '#### Top-level metadata key population' in md
        assert '| key | count |' in md
        # A concrete measured row: 'kind' appears on 3 of the 7 payloads.
        assert '| `kind` | 3 |' in md
        assert '| `supersedes` | 3 |' in md

    def test_kind_table_and_missing_count(self):
        md = _mod.render_markdown(_rich_report())
        assert '#### `kind` values' in md
        assert '| `cycle_summary` | 2 |' in md
        assert '| `gotcha` | 1 |' in md
        assert '`kind` missing: **3** record(s).' in md

    def test_supersedes_shape_table_with_member_and_length_breakdowns(self):
        md = _mod.render_markdown(_rich_report())
        assert '#### `supersedes` shapes' in md
        assert '#### `supersedes` member shapes' in md
        assert '#### `supersedes` list lengths' in md
        for label in ('absent', 'null', 'scalar', 'list'):
            assert f'| `{label}` |' in md
        assert '| `full_uuid` | 2 |' in md
        assert '| `short_hex` | 1 |' in md
        # list-length distribution: one list of 2 members.
        assert '| length | count |' in md
        assert '| `2` | 1 |' in md

    def test_topic_canonical_parent_id_occurrences(self):
        md = _mod.render_markdown(_rich_report())
        assert '#### Occurrence axes' in md
        assert '| `topic` present | 1 |' in md
        assert '| `parent_id` present | 1 |' in md
        assert '| `canonical` true | 1 |' in md
        assert '| `canonical` false | 0 |' in md
        assert '| `canonical` non-bool | 1 |' in md
        assert '#### `topic` values' in md
        assert '| `merge_lane` | 1 |' in md

    def test_source_set_but_kind_missing_section_lists_offending_sources(self):
        md = _mod.render_markdown(_rich_report())
        assert '#### `source` values' in md
        assert '#### `source` set but `kind` missing' in md
        # Both stage1_flag_marker records lack a kind -- the tools.py:1595-1597
        # drift, rendered as its own row rather than merely mentioned.
        assert '| `stage1_flag_marker` | 2 |' in md


class TestRenderMarkdownDisclosure:
    """Truncation and incomplete coverage must be visible in the markdown twin."""

    def test_truncated_table_renders_its_disclosure(self):
        md = _mod.render_markdown(_rich_report(top_n=1))
        assert '_Showing top 1 of ' in md
        assert 'distinct values' in md
        assert '**truncated**' in md
        # And it points at the copy that is NOT cut.
        assert 'JSON artifact carries the full population' in md

    def test_untruncated_report_has_no_truncation_note(self):
        md = _mod.render_markdown(_rich_report(top_n=500))
        assert '_Showing top ' not in md
        assert '**truncated**' not in md

    def test_markdown_cut_never_drops_a_value_from_the_json(self):
        # The markdown is the readable twin; the JSON it renders from stays
        # complete, so a cut row is always recoverable.
        report = _rich_report(top_n=1)
        md = _mod.render_markdown(report)
        kind_table = report['projects']['dark_factory']['categories'][OBS]['kind']
        assert kind_table['distinct_total'] == len(kind_table['entries'])
        assert md.count('| `cycle_summary` | 2 |') >= 1

    def test_top_n_zero_or_negative_renders_the_full_population(self):
        md = _mod.render_markdown(_rich_report(top_n=0))
        assert '_Showing top ' not in md
        assert '| `cycle_summary` | 2 |' in md

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


# ===========================================================================
# Tests: scroll_all_payloads
# ===========================================================================

def _record(payload: dict | None) -> MagicMock:
    """A Qdrant scroll record stand-in (payload only -- id/vector unused)."""
    record = MagicMock()
    record.payload = payload
    return record


def _paging_client(pages: list[tuple[list, object]]) -> AsyncMock:
    """AsyncMock Qdrant client whose scroll() replays a scripted sequence of
    ``(records, next_offset)`` pairs."""
    client = AsyncMock()
    client.scroll = AsyncMock(side_effect=list(pages))
    return client


async def _drain(agen) -> list[dict]:
    return [payload async for payload in agen]


class TestScrollAllPayloads:
    """Real pagination -- the whole point of this script.

    ``Mem0Backend.scroll_by_metadata`` discards ``next_offset``
    (mem0_client.py:395) and so re-reads the same head forever; these tests
    fail any implementation that does the same.
    """

    @pytest.mark.asyncio
    async def test_yields_every_payload_across_all_pages_in_order(self):
        client = _paging_client([
            ([_record({'n': 1}), _record({'n': 2})], 'off-1'),
            ([_record({'n': 3})], 'off-2'),
            ([_record({'n': 4})], None),
        ])
        got = await _drain(_mod.scroll_all_payloads(
            client, 'fused_dark_factory', {'category': OBS}, page_size=2,
        ))
        assert got == [{'n': 1}, {'n': 2}, {'n': 3}, {'n': 4}]

    @pytest.mark.asyncio
    async def test_stops_when_next_offset_is_none(self):
        client = _paging_client([([_record({'n': 1})], None)])
        await _drain(_mod.scroll_all_payloads(client, 'c', {'category': OBS}, page_size=10))
        assert client.scroll.await_count == 1

    @pytest.mark.asyncio
    async def test_passes_previous_next_offset_on_each_subsequent_call(self):
        client = _paging_client([
            ([_record({'n': 1})], 'off-1'),
            ([_record({'n': 2})], 'off-2'),
            ([_record({'n': 3})], None),
        ])
        await _drain(_mod.scroll_all_payloads(client, 'c', {'category': OBS}, page_size=1))
        offsets = [call.kwargs.get('offset') for call in client.scroll.await_args_list]
        # A non-paging implementation would send offset=None three times and
        # re-read the same head.
        assert offsets == [None, 'off-1', 'off-2']

    @pytest.mark.asyncio
    async def test_requests_payload_and_never_vectors(self):
        client = _paging_client([([_record({'n': 1})], None)])
        await _drain(_mod.scroll_all_payloads(client, 'c', {'category': OBS}, page_size=10))
        kwargs = client.scroll.await_args_list[0].kwargs
        assert kwargs['with_payload'] is True
        assert kwargs.get('with_vectors', False) is False
        assert kwargs['collection_name'] == 'c'
        assert kwargs['limit'] == 10

    @pytest.mark.asyncio
    async def test_filter_is_built_from_the_filters_mapping(self):
        client = _paging_client([([], None)])
        await _drain(_mod.scroll_all_payloads(client, 'c', {'category': OBS}, page_size=10))
        scroll_filter = client.scroll.await_args_list[0].kwargs['scroll_filter']
        # Same Filter/FieldCondition/MatchValue construction count_by_metadata
        # uses (mem0_client.py:386-394), so the cross-check count and the
        # scroll select identically.
        assert len(scroll_filter.must) == 1
        assert scroll_filter.must[0].key == 'category'
        assert scroll_filter.must[0].match.value == OBS

    @pytest.mark.asyncio
    async def test_none_payload_yields_an_empty_dict(self):
        client = _paging_client([([_record(None), _record({'n': 1})], None)])
        got = await _drain(_mod.scroll_all_payloads(client, 'c', {'category': OBS}, page_size=10))
        assert got == [{}, {'n': 1}]

    @pytest.mark.asyncio
    async def test_empty_collection_yields_nothing(self):
        client = _paging_client([([], None)])
        got = await _drain(_mod.scroll_all_payloads(client, 'c', {'category': OBS}, page_size=10))
        assert got == []

    @pytest.mark.asyncio
    async def test_payloads_are_copies_not_live_references(self):
        payload = {'n': 1}
        client = _paging_client([([_record(payload)], None)])
        got = await _drain(_mod.scroll_all_payloads(client, 'c', {'category': OBS}, page_size=10))
        got[0]['n'] = 999
        assert payload == {'n': 1}


class TestScrollAllPayloadsBudget:
    """A truncated stream must raise, never return short (INV-2)."""

    @pytest.mark.asyncio
    async def test_raises_when_page_budget_exhausted_with_live_next_offset(self):
        client = _paging_client([
            ([_record({'n': 1})], 'off-1'),
            ([_record({'n': 2})], 'off-2'),
        ])
        with pytest.raises(_mod.CensusScanIncomplete) as excinfo:
            await _drain(_mod.scroll_all_payloads(
                client, 'fused_reify', {'category': OBS}, page_size=1, max_pages=2,
            ))
        message = str(excinfo.value)
        assert 'fused_reify' in message
        assert 'category' in message
        assert '2' in message

    @pytest.mark.asyncio
    async def test_does_not_raise_when_budget_is_exactly_enough(self):
        client = _paging_client([
            ([_record({'n': 1})], 'off-1'),
            ([_record({'n': 2})], None),
        ])
        got = await _drain(_mod.scroll_all_payloads(
            client, 'c', {'category': OBS}, page_size=1, max_pages=2,
        ))
        assert got == [{'n': 1}, {'n': 2}]

    @pytest.mark.asyncio
    async def test_census_scan_incomplete_is_an_exception(self):
        assert issubclass(_mod.CensusScanIncomplete, Exception)


class TestScrollAllPayloadsTimeout:
    """A wedged connection must fail loudly, not hang the scan.

    Every other Qdrant read on this path bounds itself with the backend's
    ``_read_timeout`` (mem0_client.py:287, 331, 395). A ~30-round-trip census
    is the most exposed caller of all: unbounded, a stalled socket produces
    no artifact and no diagnostic, forever.
    """

    @pytest.mark.asyncio
    async def test_a_hung_page_request_raises_instead_of_hanging(self):
        async def _hang(**kwargs):
            await asyncio.sleep(3600)

        client = AsyncMock()
        client.scroll = AsyncMock(side_effect=_hang)
        with pytest.raises(TimeoutError):
            await _drain(_mod.scroll_all_payloads(
                client, 'c', {'category': OBS}, page_size=10, read_timeout=0.01,
            ))

    @pytest.mark.asyncio
    async def test_timeout_applies_per_page_not_per_scan(self):
        # Two quick pages must both complete under a short per-page bound.
        client = _paging_client([
            ([_record({'n': 1})], 'off-1'),
            ([_record({'n': 2})], None),
        ])
        got = await _drain(_mod.scroll_all_payloads(
            client, 'c', {'category': OBS}, page_size=1, read_timeout=5,
        ))
        assert got == [{'n': 1}, {'n': 2}]

    @pytest.mark.asyncio
    async def test_none_disables_the_bound(self):
        client = _paging_client([([_record({'n': 1})], None)])
        got = await _drain(_mod.scroll_all_payloads(
            client, 'c', {'category': OBS}, page_size=1, read_timeout=None,
        ))
        assert got == [{'n': 1}]


# ===========================================================================
# Tests: census_project
# ===========================================================================

ALL_SIX = [c.value for c in _mod.CENSUS_CATEGORIES]


def _backend(
    payloads_by_category: dict[str, list[dict]],
    *,
    counts_by_category: dict[str, int] | None = None,
    collection_points: int | None = None,
    page_size_pages: bool = False,
    call_log: list | None = None,
) -> AsyncMock:
    """Mem0Backend stand-in: count_by_metadata / count / _get_async_qdrant.

    The fake Qdrant client dispatches on the scroll filter's ``category``
    value, returning that category's payloads (one page, or one page per
    payload when *page_size_pages* is set, to exercise the paging path).
    """
    log = call_log if call_log is not None else []
    counts = counts_by_category or {c: len(p) for c, p in payloads_by_category.items()}

    async def _scroll(**kwargs):
        category = kwargs['scroll_filter'].must[0].match.value
        log.append(('scroll', category))
        payloads = payloads_by_category.get(category, [])
        if not page_size_pages or len(payloads) <= 1:
            return ([_record(p) for p in payloads], None)
        offset = kwargs.get('offset') or 0
        index = int(offset)
        next_offset = index + 1 if index + 1 < len(payloads) else None
        return ([_record(payloads[index])], next_offset)

    client = AsyncMock()
    client.scroll = AsyncMock(side_effect=_scroll)

    async def _count_by_metadata(scope, filters):
        category = filters['category']
        log.append(('count', category))
        return counts.get(category, 0)

    total = (
        collection_points if collection_points is not None
        else sum(counts.get(c, 0) for c in payloads_by_category)
    )

    backend = AsyncMock()
    backend.config = MagicMock()
    backend.config.mem0.collection_prefix = 'fused'
    backend.count_by_metadata = AsyncMock(side_effect=_count_by_metadata)
    backend.count = AsyncMock(return_value=total)
    backend._get_async_qdrant = AsyncMock(return_value=client)
    backend._fake_client = client
    return backend


class TestCensusProjectCells:
    @pytest.mark.asyncio
    async def test_returns_one_cell_per_category_keyed_by_category_value(self):
        backend = _backend({OBS: [{'kind': 'a'}], PROC: [{'kind': 'b'}, {}]})
        cells, _ = await _mod.census_project(backend, 'dark_factory', [OBS, PROC])
        assert set(cells) == {OBS, PROC}
        assert cells[OBS].records == 1
        assert cells[PROC].records == 2
        assert cells[PROC].kind_counts['b'] == 1
        assert cells[PROC].kind_missing == 1

    @pytest.mark.asyncio
    async def test_covers_all_six_categories_including_empty_dual_write_ones(self):
        # A dual-write-only category must still produce a (zero) cell, not be
        # dropped -- otherwise the residue it explains disappears silently.
        backend = _backend({OBS: [{'kind': 'a'}]}, counts_by_category={OBS: 1})
        cells, coverage = await _mod.census_project(backend, 'dark_factory', ALL_SIX)
        assert set(cells) == set(ALL_SIX)
        assert cells[DEC].records == 0
        assert coverage['categories'][DEC]['expected'] == 0

    @pytest.mark.asyncio
    async def test_defaults_to_all_six_categories(self):
        backend = _backend({OBS: [{'kind': 'a'}]}, counts_by_category={OBS: 1})
        cells, _ = await _mod.census_project(backend, 'dark_factory')
        assert set(cells) == set(ALL_SIX)

    @pytest.mark.asyncio
    async def test_payloads_stream_through_pagination(self):
        backend = _backend(
            {OBS: [{'kind': 'a'}, {'kind': 'b'}, {'kind': 'c'}]},
            counts_by_category={OBS: 3},
            page_size_pages=True,
        )
        cells, coverage = await _mod.census_project(backend, 'dark_factory', [OBS], page_size=1)
        assert cells[OBS].records == 3
        assert coverage['categories'][OBS]['scrolled'] == 3
        assert backend._fake_client.scroll.await_count == 3


class TestCensusProjectCorroboration:
    """INV-3: count first, then scroll, then reconcile the two."""

    @pytest.mark.asyncio
    async def test_counts_each_category_before_scrolling_it(self):
        log: list = []
        backend = _backend({OBS: [{}], PROC: [{}]}, call_log=log)
        await _mod.census_project(backend, 'dark_factory', [OBS, PROC])
        assert ('count', OBS) in log
        assert log.index(('count', OBS)) < log.index(('scroll', OBS))
        assert log.index(('count', PROC)) < log.index(('scroll', PROC))

    @pytest.mark.asyncio
    async def test_count_by_metadata_brackets_each_category_scroll(self):
        # Counted before AND after: the second read is what tells corpus
        # churn apart from a truncated scan on a live store.
        backend = _backend({OBS: [{}], PROC: [{}]})
        await _mod.census_project(backend, 'dark_factory', [OBS, PROC])
        assert backend.count_by_metadata.await_count == 4
        filters = [call.args[1] for call in backend.count_by_metadata.await_args_list]
        assert filters == [
            {'category': OBS}, {'category': OBS},
            {'category': PROC}, {'category': PROC},
        ]

    @pytest.mark.asyncio
    async def test_the_second_count_happens_after_the_scroll(self):
        log: list = []
        backend = _backend({OBS: [{}]}, call_log=log)
        await _mod.census_project(backend, 'dark_factory', [OBS])
        assert log == [('count', OBS), ('scroll', OBS), ('count', OBS)]

    @pytest.mark.asyncio
    async def test_expected_recount_and_scrolled_recorded_per_category(self):
        backend = _backend({OBS: [{}, {}]}, counts_by_category={OBS: 2})
        _, coverage = await _mod.census_project(backend, 'dark_factory', [OBS])
        assert coverage['categories'][OBS]['expected'] == 2
        assert coverage['categories'][OBS]['recount'] == 2
        assert coverage['categories'][OBS]['scrolled'] == 2

    @pytest.mark.asyncio
    async def test_churn_between_the_two_counts_is_not_called_a_shortfall(self):
        # count says 2, the scroll yields 3, and the re-count agrees at 3 --
        # a write landed mid-census. Nothing is missing.
        counts = iter([2, 3])
        backend = _backend({OBS: [{}, {}, {}]})
        backend.count_by_metadata = AsyncMock(side_effect=lambda scope, f: next(counts))
        _, coverage = await _mod.census_project(backend, 'dark_factory', [OBS])
        cell = coverage['categories'][OBS]
        assert (cell['expected'], cell['recount'], cell['scrolled']) == (2, 3, 3)
        assert cell['complete'] is True

    @pytest.mark.asyncio
    async def test_under_enumeration_marks_the_cell_incomplete_with_the_delta(self):
        # count says 5, scroll yields 2 -- the shortfall must be named.
        backend = _backend({OBS: [{}, {}]}, counts_by_category={OBS: 5})
        _, coverage = await _mod.census_project(backend, 'dark_factory', [OBS])
        cell = coverage['categories'][OBS]
        assert cell['expected'] == 5
        assert cell['scrolled'] == 2
        assert cell['delta'] == -3
        assert cell['complete'] is False


class TestCensusProjectCoverage:
    @pytest.mark.asyncio
    async def test_resolves_collection_name_from_scope_and_config_prefix(self):
        backend = _backend({OBS: [{}]}, counts_by_category={OBS: 1})
        _, coverage = await _mod.census_project(backend, 'dark_factory', [OBS])
        assert coverage['collection'] == 'fused_dark_factory'

    @pytest.mark.asyncio
    async def test_backend_count_brackets_the_whole_category_loop(self):
        # The collection total is read BEFORE and AFTER the scan, exactly as
        # each category's count is. A single read cannot be reconciled with a
        # counted total gathered over a ~1-minute scan of a live corpus:
        # _build_coverage would charge ordinary drift as missing data.
        backend = _backend({OBS: [{}]}, counts_by_category={OBS: 1}, collection_points=19464)
        _, coverage = await _mod.census_project(backend, 'dark_factory', [OBS])
        assert coverage['collection_points'] == 19464
        assert coverage['collection_points_before'] == 19464
        assert backend.count.await_count == 2

    @pytest.mark.asyncio
    async def test_both_bracket_reads_are_recorded_when_they_disagree(self):
        # A single-read implementation cannot record two different values.
        backend = _backend({OBS: [{}]}, counts_by_category={OBS: 1})
        backend.count = AsyncMock(side_effect=[100, 101])
        _, coverage = await _mod.census_project(backend, 'dark_factory', [OBS])
        assert coverage['collection_points_before'] == 100
        assert coverage['collection_points'] == 101

    @pytest.mark.asyncio
    async def test_the_pre_loop_count_happens_before_the_first_category_count(self):
        # Order matters: a bracket read taken after the category counts
        # brackets nothing.
        log: list = []
        backend = _backend({OBS: [{}], PROC: [{}]}, call_log=log)

        async def _count(scope):
            log.append(('collection_count', None))
            return 2

        backend.count = AsyncMock(side_effect=_count)
        await _mod.census_project(backend, 'dark_factory', [OBS, PROC])
        assert log[0] == ('collection_count', None)
        assert log[-1] == ('collection_count', None)
        assert log.index(('count', OBS)) > 0

    @pytest.mark.asyncio
    async def test_uncovered_points_is_collection_total_minus_summed_categories(self):
        backend = _backend(
            {OBS: [{}], PROC: [{}]},
            counts_by_category={OBS: 24408, PROC: 3981},
            collection_points=29951,
        )
        _, coverage = await _mod.census_project(backend, 'dark_factory', [OBS, PROC])
        report = _mod.build_report({'dark_factory': {}}, {'dark_factory': coverage}, top_n=50)
        assert report['coverage']['projects']['dark_factory']['uncovered_points'] == 1562

    @pytest.mark.asyncio
    async def test_passes_the_backends_read_timeout_to_every_scroll(self, monkeypatch):
        seen: list = []

        async def _fake_scroll(client, collection, filters, page_size=1000,
                               max_pages=200, read_timeout=None):
            seen.append(read_timeout)
            return
            yield  # pragma: no cover -- makes this an async generator

        monkeypatch.setattr(_mod, 'scroll_all_payloads', _fake_scroll)
        backend = _backend({OBS: []}, counts_by_category={OBS: 0})
        backend._read_timeout = 7.5
        await _mod.census_project(backend, 'dark_factory', [OBS])
        assert seen == [7.5]

    @pytest.mark.asyncio
    async def test_non_numeric_read_timeout_degrades_to_unbounded(self, monkeypatch):
        # A backend stand-in auto-creates _read_timeout as a Mock; passing
        # that into asyncio.wait_for would TypeError mid-scan.
        seen: list = []

        async def _fake_scroll(client, collection, filters, page_size=1000,
                               max_pages=200, read_timeout=None):
            seen.append(read_timeout)
            return
            yield  # pragma: no cover -- makes this an async generator

        monkeypatch.setattr(_mod, 'scroll_all_payloads', _fake_scroll)
        backend = _backend({OBS: []}, counts_by_category={OBS: 0})
        await _mod.census_project(backend, 'dark_factory', [OBS])
        assert seen == [None]

    @pytest.mark.asyncio
    async def test_coverage_feeds_build_report_unchanged(self):
        backend = _backend({OBS: [{'kind': 'a'}]}, counts_by_category={OBS: 1})
        cells, coverage = await _mod.census_project(backend, 'dark_factory', [OBS])
        report = _mod.build_report({'dark_factory': cells}, {'dark_factory': coverage}, top_n=50)
        assert report['coverage']['complete'] is True
        assert report['projects']['dark_factory']['categories'][OBS]['records'] == 1


# ===========================================================================
# Tests: CLI band (_run / main)
# ===========================================================================

def _args(tmp_path, **overrides):
    """argparse.Namespace matching the CLI's parser defaults."""
    import argparse

    defaults = {
        'project_id': ['dark_factory'],
        'page_size': 1000,
        'top_n': 50,
        'max_pages': _mod.DEFAULT_MAX_PAGES,
        'json_out': str(tmp_path / 'census.json'),
        'md_out': str(tmp_path / 'census.md'),
        'config': None,
    }
    defaults.update(overrides)
    return argparse.Namespace(**defaults)


def _census_project_stub(*, complete: bool = True):
    """Stand-in for census_project returning one populated cell."""
    async def _stub(backend, project_id, categories=None, page_size=1000, max_pages=200):
        cells = {OBS: _census([{'kind': 'cycle_summary'}])}
        coverage = _coverage('fused_x', 1 if complete else 9, {OBS: (1, 1)})
        return cells, coverage
    return _stub


class TestCliArtifacts:
    @pytest.mark.asyncio
    async def test_writes_both_artifacts_to_the_given_paths(self, tmp_path, monkeypatch):
        monkeypatch.setattr(_mod, 'census_project', _census_project_stub())
        monkeypatch.setattr(_mod, '_build_backend', lambda cfg: AsyncMock())
        rc = await _mod._run(_args(tmp_path))
        assert rc == 0
        written = json.loads((tmp_path / 'census.json').read_text())
        assert written['projects']['dark_factory']['categories'][OBS]['records'] == 1
        assert (tmp_path / 'census.md').read_text().strip()

    @pytest.mark.asyncio
    async def test_creates_missing_parent_directories(self, tmp_path, monkeypatch):
        monkeypatch.setattr(_mod, 'census_project', _census_project_stub())
        monkeypatch.setattr(_mod, '_build_backend', lambda cfg: AsyncMock())
        nested = tmp_path / 'a' / 'b'
        rc = await _mod._run(_args(
            tmp_path, json_out=str(nested / 'c.json'), md_out=str(nested / 'c.md'),
        ))
        assert rc == 0
        assert (nested / 'c.json').exists()
        assert (nested / 'c.md').exists()

    @pytest.mark.asyncio
    async def test_honours_repeated_project_id(self, tmp_path, monkeypatch):
        seen: list[str] = []

        async def _stub(backend, project_id, categories=None, page_size=1000, max_pages=200):
            seen.append(project_id)
            return {OBS: _census([{}])}, _coverage('fused_x', 1, {OBS: (1, 1)})

        monkeypatch.setattr(_mod, 'census_project', _stub)
        monkeypatch.setattr(_mod, '_build_backend', lambda cfg: AsyncMock())
        await _mod._run(_args(tmp_path, project_id=['dark_factory', 'reify']))
        assert seen == ['dark_factory', 'reify']


class TestCliExitCodes:
    @pytest.mark.asyncio
    async def test_zero_when_coverage_complete(self, tmp_path, monkeypatch):
        monkeypatch.setattr(_mod, 'census_project', _census_project_stub(complete=True))
        monkeypatch.setattr(_mod, '_build_backend', lambda cfg: AsyncMock())
        assert await _mod._run(_args(tmp_path)) == 0

    @pytest.mark.asyncio
    async def test_nonzero_when_coverage_incomplete(self, tmp_path, monkeypatch):
        monkeypatch.setattr(_mod, 'census_project', _census_project_stub(complete=False))
        monkeypatch.setattr(_mod, '_build_backend', lambda cfg: AsyncMock())
        assert await _mod._run(_args(tmp_path)) != 0

    @pytest.mark.asyncio
    async def test_incomplete_run_still_writes_the_artifacts(self, tmp_path, monkeypatch):
        # The evidence of the shortfall is IN the artifact -- a non-zero exit
        # must not also suppress it.
        monkeypatch.setattr(_mod, 'census_project', _census_project_stub(complete=False))
        monkeypatch.setattr(_mod, '_build_backend', lambda cfg: AsyncMock())
        await _mod._run(_args(tmp_path))
        report = json.loads((tmp_path / 'census.json').read_text())
        assert report['coverage']['complete'] is False
        assert report['coverage']['deltas']

    @pytest.mark.asyncio
    async def test_nonzero_when_census_scan_incomplete_propagates(self, tmp_path, monkeypatch):
        async def _boom(*a, **kw):
            raise _mod.CensusScanIncomplete('truncated')

        monkeypatch.setattr(_mod, 'census_project', _boom)
        monkeypatch.setattr(_mod, '_build_backend', lambda cfg: AsyncMock())
        assert await _mod._run(_args(tmp_path)) != 0


class TestCliCleanup:
    @pytest.mark.asyncio
    async def test_closes_the_backend(self, tmp_path, monkeypatch):
        backend = AsyncMock()
        monkeypatch.setattr(_mod, 'census_project', _census_project_stub())
        monkeypatch.setattr(_mod, '_build_backend', lambda cfg: backend)
        await _mod._run(_args(tmp_path))
        backend.close.assert_awaited_once()

    @pytest.mark.asyncio
    async def test_closes_the_backend_even_when_census_raises(self, tmp_path, monkeypatch):
        backend = AsyncMock()

        async def _boom(*a, **kw):
            raise _mod.CensusScanIncomplete('truncated')

        monkeypatch.setattr(_mod, 'census_project', _boom)
        monkeypatch.setattr(_mod, '_build_backend', lambda cfg: backend)
        await _mod._run(_args(tmp_path))
        backend.close.assert_awaited_once()


class TestCliParser:
    def test_exposes_the_documented_flags(self):
        parser = _mod._build_parser()
        args = parser.parse_args([])
        assert args.page_size == 1000
        assert args.max_pages == _mod.DEFAULT_MAX_PAGES
        assert isinstance(args.top_n, int)
        assert args.config is None

    def test_defaults_to_both_projects(self):
        args = _mod._build_parser().parse_args([])
        assert args.project_id == ['dark_factory', 'reify']

    def test_default_artifact_paths_are_the_committed_ones(self):
        args = _mod._build_parser().parse_args([])
        assert args.json_out.endswith('plans/memory-metadata-census-report.json')
        assert args.md_out.endswith('plans/memory-metadata-census-report.md')

    def test_project_id_is_repeatable(self):
        args = _mod._build_parser().parse_args(['--project-id', 'a', '--project-id', 'b'])
        assert args.project_id == ['a', 'b']

    def test_overrides_parse(self):
        args = _mod._build_parser().parse_args([
            '--page-size', '250', '--top-n', '7', '--max-pages', '3',
            '--json-out', '/tmp/j.json', '--md-out', '/tmp/m.md', '--config', '/tmp/c.yaml',
        ])
        assert (args.page_size, args.top_n, args.max_pages) == (250, 7, 3)
        assert args.json_out == '/tmp/j.json'
        assert args.md_out == '/tmp/m.md'
        assert args.config == '/tmp/c.yaml'
