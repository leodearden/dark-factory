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
from typing import Any
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
        # INV-5: the script does NOT copy MEM0_MANAGED_METADATA_KEYS; it
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


class TestCategoryCensusCanonicalByTopic:
    """The topic x canonical CROSS-TAB (task 4006).

    ``topic_values`` and ``canonical_true`` are INDEPENDENT counters, so the
    three numbers the coverage ask needs -- topics with exactly one canonical,
    with zero, with more than one -- cannot be derived from either.  This
    accumulator joins them at fold time, keyed by topic, mirroring 3198's
    write-time probe ``count_memories_by_metadata(project_id,
    {'topic': T, 'canonical': True})``
    (fused-memory/src/fused_memory/services/memory_service.py::_check_canonical_uniqueness).
    """

    def test_canonical_true_keyed_by_its_topic(self):
        c = _mod.CategoryCensus()
        c.add({'topic': 'merge-lane', 'canonical': True})
        c.add({'topic': 'merge-lane', 'canonical': True})
        c.add({'topic': 'census', 'canonical': True})
        assert c.canonical_true_by_topic['merge-lane'] == 2
        assert c.canonical_true_by_topic['census'] == 1

    def test_topic_without_canonical_true_is_not_in_the_cross_tab(self):
        c = _mod.CategoryCensus()
        c.add({'topic': 'merge-lane'})
        c.add({'topic': 'merge-lane', 'canonical': False})
        assert c.topic_values['merge-lane'] == 2
        assert c.canonical_true_by_topic['merge-lane'] == 0
        assert 'merge-lane' not in c.canonical_true_by_topic

    def test_canonical_true_without_a_topic_is_named_not_dropped(self):
        # Reachable today: the ``canonical_without_topic`` shape rule is
        # warn-mode under the shipped ``memory_metadata.enforce=false``, so
        # such records CAN land.  Counting them in a distinct scalar means
        # canonical_true == sum(by_topic) + without_topic holds exactly, and
        # no canonical vanishes silently in the join.
        c = _mod.CategoryCensus()
        c.add({'canonical': True})
        c.add({'topic': None, 'canonical': True})
        c.add({'topic': 'census', 'canonical': True})
        assert c.canonical_true == 3
        assert c.canonical_true_without_topic == 2
        assert sum(c.canonical_true_by_topic.values()) == 1
        assert c.canonical_true == sum(c.canonical_true_by_topic.values()) + \
            c.canonical_true_without_topic

    def test_non_bool_canonical_lands_in_neither_counter(self):
        # Measured live: 1 such record in reify.  Matches the existing
        # canonical_non_bool treatment -- a string 'true' is drift, not a
        # canonical, and must not be scored as one.
        c = _mod.CategoryCensus()
        c.add({'topic': 'census', 'canonical': 'true'})
        c.add({'topic': 'census', 'canonical': 1})
        c.add({'canonical': 'true'})
        assert c.canonical_non_bool == 3
        assert sum(c.canonical_true_by_topic.values()) == 0
        assert c.canonical_true_without_topic == 0

    def test_merge_folds_both_the_counter_and_the_scalar(self):
        a = _mod.CategoryCensus()
        a.add({'topic': 'merge-lane', 'canonical': True})
        a.add({'canonical': True})

        b = _mod.CategoryCensus()
        b.add({'topic': 'merge-lane', 'canonical': True})
        b.add({'topic': 'census', 'canonical': True})
        b.add({'canonical': True})

        a.merge(b)
        assert a.canonical_true_by_topic['merge-lane'] == 2
        assert a.canonical_true_by_topic['census'] == 1
        assert a.canonical_true_without_topic == 2
        assert a.canonical_true == 5
        # The rollup identity survives the merge.
        assert a.canonical_true == sum(a.canonical_true_by_topic.values()) + \
            a.canonical_true_without_topic

    def test_merge_does_not_mutate_the_other_cross_tab(self):
        a = _mod.CategoryCensus()
        a.add({'topic': 't', 'canonical': True})
        b = _mod.CategoryCensus()
        b.add({'topic': 't', 'canonical': True})
        b.add({'canonical': True})
        a.merge(b)
        assert b.canonical_true_by_topic['t'] == 1
        assert b.canonical_true_without_topic == 1

    def test_cross_tab_is_emitted_through_census_to_dict(self):
        c = _mod.CategoryCensus()
        c.add({'topic': 'merge-lane', 'canonical': True})
        c.add({'topic': 'merge-lane', 'canonical': True})
        c.add({'topic': 'census', 'canonical': True})
        c.add({'canonical': True})
        rendered = _mod._census_to_dict(c)
        assert rendered['canonical_true_without_topic'] == 1
        # Through _table(), so ordering is the deterministic count-desc /
        # value-asc total order and the JSON table is never capped.
        table = rendered['canonical_true_by_topic']
        assert table['distinct_total'] == 2
        assert [(e['value'], e['count']) for e in table['entries']] == [
            ('merge-lane', 2),
            ('census', 1),
        ]

    def test_add_does_not_mutate_a_canonical_topic_payload(self):
        payload = {'topic': 'merge-lane', 'canonical': True}
        before = dict(payload)
        c = _mod.CategoryCensus()
        c.add(payload)
        assert payload == before


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


def _census(payloads: list[dict]) -> Any:
    # `Any`, not `object`: `_mod` is loaded from a path at runtime, so
    # `CategoryCensus` has no statically-known type to name here -- and
    # callers legitimately read `.records` off the result, which `object`
    # forbids.
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


class TestTopicCoverageBlock:
    """The per-project ``topic_coverage`` block (task 4006).

    The existing report counts topic_present and canonical_true as raw
    integers.  Coverage is a RATE plus the three-way canonical partition,
    and neither is derivable from those counts.
    """

    def test_block_carries_the_rate_and_the_counts(self):
        cells = {
            'dark_factory': {
                OBS: _census(
                    [{'topic': 'merge-lane'}, {'topic': 'census'}] + [{}] * 2,
                ),
            },
        }
        cov = {'dark_factory': _coverage('fused_dark_factory', 4, {OBS: (4, 4)})}
        report = _mod.build_report(cells, cov, top_n=50)
        block = report['projects']['dark_factory']['topic_coverage']
        assert block['records'] == 4
        assert block['topic_present'] == 2
        assert block['topic_coverage_pct'] == 50.0
        assert block['distinct_topics'] == 2

    def test_empty_project_reports_a_null_rate_not_a_fabricated_zero(self):
        # A 0.0 rate says "measured, and nothing is stamped"; null says
        # "there was nothing to measure".  Conflating them is the same
        # fabricated-baseline failure task 3291 hardened extract_done_count
        # against -- and a ZeroDivisionError would lose the census entirely.
        cells = {'dark_factory': {OBS: _census([])}}
        cov = {'dark_factory': _coverage('fused_dark_factory', 0, {OBS: (0, 0)})}
        report = _mod.build_report(cells, cov, top_n=50)
        block = report['projects']['dark_factory']['topic_coverage']
        assert block['records'] == 0
        assert block['topic_coverage_pct'] is None

    def test_three_way_canonical_partition(self):
        cells = {
            'dark_factory': {
                OBS: _census([
                    {'topic': 'exactly-one', 'canonical': True},
                    {'topic': 'exactly-one'},
                    {'topic': 'none-at-all'},
                    {'topic': 'too-many', 'canonical': True},
                    {'topic': 'too-many', 'canonical': True},
                ]),
            },
        }
        cov = {'dark_factory': _coverage('fused_dark_factory', 5, {OBS: (5, 5)})}
        report = _mod.build_report(cells, cov, top_n=50)
        block = report['projects']['dark_factory']['topic_coverage']
        assert block['topics_with_one_canonical'] == 1
        assert block['topics_with_zero_canonical'] == 1
        assert block['topics_with_multiple_canonical'] == 1

    def test_partition_sums_to_distinct_topics(self):
        cells = {
            'dark_factory': {
                OBS: _census([
                    {'topic': 'a', 'canonical': True},
                    {'topic': 'b'},
                    {'topic': 'c', 'canonical': True},
                    {'topic': 'c', 'canonical': True},
                    {'topic': 'd', 'canonical': False},
                ]),
            },
        }
        cov = {'dark_factory': _coverage('fused_dark_factory', 5, {OBS: (5, 5)})}
        report = _mod.build_report(cells, cov, top_n=50)
        block = report['projects']['dark_factory']['topic_coverage']
        assert (
            block['topics_with_one_canonical']
            + block['topics_with_zero_canonical']
            + block['topics_with_multiple_canonical']
        ) == block['distinct_topics'] == 4

    def test_partition_is_computed_at_project_grain_across_all_categories(self):
        # THE load-bearing assertion.  3198's write-time probe is
        # count_memories_by_metadata(project_id, {'topic': T, 'canonical':
        # True}) -- scope-wide, NOT category-filtered.  A topic whose peers
        # sit in procedural_knowledge while its lone canonical sits in
        # observations_and_summaries has EXACTLY ONE canonical.  Partitioning
        # per-(project, category) would score it as one zero-canonical topic
        # plus one separate exactly-one topic, manufacturing a false gap.
        cells = {
            'dark_factory': {
                PROC: _census([{'topic': 'split-topic'}, {'topic': 'split-topic'}]),
                OBS: _census([{'topic': 'split-topic', 'canonical': True}]),
            },
        }
        cov = {
            'dark_factory': _coverage(
                'fused_dark_factory', 3, {PROC: (2, 2), OBS: (1, 1)},
            ),
        }
        report = _mod.build_report(cells, cov, top_n=50)
        block = report['projects']['dark_factory']['topic_coverage']
        assert block['distinct_topics'] == 1
        assert block['topics_with_one_canonical'] == 1
        assert block['topics_with_zero_canonical'] == 0

    def test_a_canonical_split_across_categories_is_still_a_violation(self):
        # The same grain in the other direction: two canonicals for one topic
        # in DIFFERENT categories is exactly the duplicate 3198's probe would
        # have caught, so the census must see it as one violation, not as two
        # well-formed cells.
        cells = {
            'dark_factory': {
                PROC: _census([{'topic': 'dup', 'canonical': True}]),
                OBS: _census([{'topic': 'dup', 'canonical': True}]),
            },
        }
        cov = {
            'dark_factory': _coverage(
                'fused_dark_factory', 2, {PROC: (1, 1), OBS: (1, 1)},
            ),
        }
        report = _mod.build_report(cells, cov, top_n=50)
        block = report['projects']['dark_factory']['topic_coverage']
        assert block['topics_with_multiple_canonical'] == 1
        assert block['topics_with_one_canonical'] == 0

    def test_projects_are_scored_independently(self):
        cells = {
            'dark_factory': {OBS: _census([{'topic': 'shared', 'canonical': True}])},
            'reify': {OBS: _census([{'topic': 'shared'}])},
        }
        cov = {
            'dark_factory': _coverage('fused_dark_factory', 1, {OBS: (1, 1)}),
            'reify': _coverage('fused_reify', 1, {OBS: (1, 1)}),
        }
        report = _mod.build_report(cells, cov, top_n=50)
        df = report['projects']['dark_factory']['topic_coverage']
        rf = report['projects']['reify']['topic_coverage']
        assert df['topics_with_one_canonical'] == 1
        assert rf['topics_with_one_canonical'] == 0
        assert rf['topics_with_zero_canonical'] == 1

    def test_grand_total_block_is_emitted(self):
        cells = {
            'dark_factory': {OBS: _census([{'topic': 'a', 'canonical': True}])},
            'reify': {OBS: _census([{'topic': 'b'}, {}])},
        }
        cov = {
            'dark_factory': _coverage('fused_dark_factory', 1, {OBS: (1, 1)}),
            'reify': _coverage('fused_reify', 2, {OBS: (2, 2)}),
        }
        report = _mod.build_report(cells, cov, top_n=50)
        block = report['topic_coverage']
        assert block['records'] == 3
        assert block['topic_present'] == 2
        assert block['distinct_topics'] == 2

    def test_canonical_without_a_topic_is_disclosed_in_the_block(self):
        cells = {
            'dark_factory': {
                OBS: _census([{'topic': 'a', 'canonical': True}, {'canonical': True}]),
            },
        }
        cov = {'dark_factory': _coverage('fused_dark_factory', 2, {OBS: (2, 2)})}
        report = _mod.build_report(cells, cov, top_n=50)
        block = report['projects']['dark_factory']['topic_coverage']
        assert block['canonical_true'] == 2
        assert block['canonical_true_without_topic'] == 1
        # The join drops it from every per-topic tally, so naming it is what
        # keeps the block reconcilable against the canonical_true axis.
        assert block['topics_with_one_canonical'] == 1

    def test_schema_version_is_bumped_for_the_new_block(self):
        cells = {'dark_factory': {OBS: _census([{}])}}
        cov = {'dark_factory': _coverage('fused_dark_factory', 1, {OBS: (1, 1)})}
        report = _mod.build_report(cells, cov, top_n=50)
        assert report['schema_version'] >= 4

    def test_block_is_json_serialisable_and_deterministic(self):
        cells = {
            'dark_factory': {
                OBS: _census([
                    {'topic': 'a', 'canonical': True},
                    {'topic': 'b'},
                ]),
            },
        }
        cov = {'dark_factory': _coverage('fused_dark_factory', 2, {OBS: (2, 2)})}
        first = _mod.build_report(cells, cov, top_n=50)
        second = _mod.build_report(cells, cov, top_n=50)
        assert json.dumps(first, sort_keys=False) == json.dumps(second, sort_keys=False)


class TestUniquenessViolationsAreNamedAndReadable:
    """>1 canonical per topic: NAMED, and readable against the live regime.

    An operator cannot act on an integer, and a bare count is also
    MISREADABLE: 3198 ships warn-mode-first behind ``memory_metadata.enforce``
    (default False,
    ``fused-memory/src/fused_memory/config/schema.py::MemoryMetadataConfig.enforce``),
    so a duplicate can land by an ordinary write and a non-zero count is
    partly backlog, not proof the guard broke.  The block therefore carries the violators, the live flag,
    and -- per Leo's 2026-08-12 ruling (Option B on esc-4006-3) -- what still
    stands between here and flipping that flag.
    """

    @staticmethod
    def _violators_report(**kwargs):
        cells = {
            'dark_factory': {
                OBS: _census([
                    {'topic': 'three-way', 'canonical': True},
                    {'topic': 'three-way', 'canonical': True},
                    {'topic': 'three-way', 'canonical': True},
                    {'topic': 'a-pair', 'canonical': True},
                    {'topic': 'a-pair', 'canonical': True},
                    {'topic': 'b-pair', 'canonical': True},
                    {'topic': 'b-pair', 'canonical': True},
                    {'topic': 'fine', 'canonical': True},
                ]),
            },
        }
        cov = {'dark_factory': _coverage('fused_dark_factory', 8, {OBS: (8, 8)})}
        return _mod.build_report(cells, cov, top_n=50, **kwargs)

    def test_violators_are_named_not_merely_counted(self):
        block = self._violators_report()['projects']['dark_factory']['topic_coverage']
        assert block['topics_with_multiple_canonical'] == 3
        rows = block['multiple_canonical_topics']
        assert [(r['topic'], r['canonical_count']) for r in rows] == [
            ('three-way', 3),
            ('a-pair', 2),
            ('b-pair', 2),
        ]

    def test_violator_rows_follow_the_house_total_order(self):
        # count desc, then topic asc -- the same total order _table() uses,
        # so the artifact stays byte-stable across re-runs.
        rows = self._violators_report()['projects']['dark_factory'][
            'topic_coverage'
        ]['multiple_canonical_topics']
        assert rows == sorted(rows, key=lambda r: (-r['canonical_count'], r['topic']))

    def test_a_clean_corpus_reports_an_empty_violator_list(self):
        cells = {'dark_factory': {OBS: _census([{'topic': 'fine', 'canonical': True}])}}
        cov = {'dark_factory': _coverage('fused_dark_factory', 1, {OBS: (1, 1)})}
        report = _mod.build_report(cells, cov, top_n=50)
        block = report['projects']['dark_factory']['topic_coverage']
        assert block['multiple_canonical_topics'] == []

    def test_enforce_state_is_disclosed_beside_the_count(self):
        for state in (True, False):
            block = self._violators_report(canonical_uniqueness_enforced=state)[
                'projects'
            ]['dark_factory']['topic_coverage']
            assert block['canonical_uniqueness_enforced'] is state

    def test_enforce_state_is_three_valued_never_defaulted_to_false(self):
        # null means "could not read the config", which is a DIFFERENT claim
        # from "the guard is off".  Fabricating False here would assert the
        # wrong defect class confidently: it would tell a reader the
        # violations are expected backlog when in fact nothing is known.
        block = self._violators_report(canonical_uniqueness_enforced=None)[
            'projects'
        ]['dark_factory']['topic_coverage']
        assert block['canonical_uniqueness_enforced'] is None

    def test_enforce_reader_returns_a_bool_from_the_live_config(self):
        assert _mod.read_canonical_uniqueness_enforced() in (True, False)

    def test_enforce_reader_degrades_to_none_rather_than_raising(self, monkeypatch):
        # The census must still produce its MEASUREMENT when config loading
        # is broken -- losing the whole census to a config fault would be a
        # strictly worse outcome than an undisclosed flag.
        import fused_memory.config.schema as schema_mod

        class _Exploding:
            def __init__(self, *a, **kw):
                raise RuntimeError('config is unloadable')

        monkeypatch.setattr(schema_mod, 'FusedMemoryConfig', _Exploding)
        assert _mod.read_canonical_uniqueness_enforced() is None

    # ── the corpus-wide block is graded at (project, topic) grain ──────────
    #
    # Caught on the FIRST live regeneration (step-22): the report headlined
    # "Topics carrying more than one live canonical: true: 1" directly above a
    # named-violator table reading "(none)".  Both were right about their own
    # scope and the report was still wrong, which is the worst shape a measured
    # artifact can take -- a reader either distrusts the number or hunts for a
    # violation that does not exist.
    #
    # Cause: the corpus-wide block was built from a census merged ACROSS
    # projects, which merges the two topic NAMESPACES.  3198's invariant is
    # per-(project, topic) -- its probe is
    # count_memories_by_metadata(project_id, {'topic': T, 'canonical': True}),
    # bound to one project -- so a topic correctly carrying one canonical in
    # dark_factory and one in reify is two compliant cells, not one violation.
    # Live instance: `merge-request-bare-task-id-branch-arg`, canonical in both.

    def test_one_canonical_in_each_of_two_projects_is_not_a_violation(self):
        """The exact live shape that produced the contradictory artifact."""
        shared = 'merge-request-bare-task-id-branch-arg'
        cells = {
            'dark_factory': {OBS: _census([{'topic': shared, 'canonical': True}])},
            'reify': {OBS: _census([{'topic': shared, 'canonical': True}])},
        }
        cov = {
            'dark_factory': _coverage('fused_dark_factory', 1, {OBS: (1, 1)}),
            'reify': _coverage('fused_reify', 1, {OBS: (1, 1)}),
        }
        report = _mod.build_report(cells, cov, top_n=50)

        for project_id in ('dark_factory', 'reify'):
            block = report['projects'][project_id]['topic_coverage']
            assert block['topics_with_multiple_canonical'] == 0, project_id
            assert block['topics_with_one_canonical'] == 1, project_id

        grand = report['topic_coverage']
        assert grand['topics_with_multiple_canonical'] == 0
        assert grand['multiple_canonical_topics'] == []
        # Two compliant CELLS, not one merged topic: the partition is counted
        # over (project, topic) pairs, the grain the invariant is stated at.
        assert grand['topics_with_one_canonical'] == 2

    def test_the_corpus_wide_count_and_its_named_list_can_never_disagree(self):
        """The count IS the length of the named list, at every scope.

        Pinned as an identity rather than as two numbers, because the artifact
        that shipped had a count and a list sourced from different grains.
        """
        cells = {
            'dark_factory': {
                OBS: _census([
                    # A real per-project violation: two canonicals, one topic.
                    {'topic': 'dupe', 'canonical': True},
                    {'topic': 'dupe', 'canonical': True},
                    {'topic': 'shared', 'canonical': True},
                ]),
            },
            'reify': {OBS: _census([{'topic': 'shared', 'canonical': True}])},
        }
        cov = {
            'dark_factory': _coverage('fused_dark_factory', 3, {OBS: (3, 3)}),
            'reify': _coverage('fused_reify', 1, {OBS: (1, 1)}),
        }
        report = _mod.build_report(cells, cov, top_n=50)

        for block in (
            report['projects']['dark_factory']['topic_coverage'],
            report['projects']['reify']['topic_coverage'],
            report['topic_coverage'],
        ):
            assert (block['topics_with_multiple_canonical']
                    == len(block['multiple_canonical_topics']))

        grand = report['topic_coverage']
        # The genuine violation survives the regrade; the cross-project pair
        # does not become one.
        assert grand['topics_with_multiple_canonical'] == 1
        assert [r['topic'] for r in grand['multiple_canonical_topics']] == ['dupe']
        # A corpus-wide violator must name its project -- "dupe is duplicated"
        # is not actionable without knowing where.
        assert grand['multiple_canonical_topics'][0]['project_id'] == 'dark_factory'

    def test_the_partition_still_sums_to_the_cell_population(self):
        """one + zero + multiple == the number of (project, topic) cells.

        The per-project blocks each partition their own topic_values, so the
        corpus-wide sum must equal the sum of the per-project distinct counts
        -- NOT the corpus-wide distinct-VALUE count, which de-duplicates a
        topic shared by two projects.
        """
        cells = {
            'dark_factory': {
                OBS: _census([{'topic': 'shared', 'canonical': True}, {'topic': 'a'}]),
            },
            'reify': {OBS: _census([{'topic': 'shared'}, {'topic': 'b'}])},
        }
        cov = {
            'dark_factory': _coverage('fused_dark_factory', 2, {OBS: (2, 2)}),
            'reify': _coverage('fused_reify', 2, {OBS: (2, 2)}),
        }
        report = _mod.build_report(cells, cov, top_n=50)
        grand = report['topic_coverage']
        cells_total = sum(
            report['projects'][p]['topic_coverage']['distinct_topics']
            for p in ('dark_factory', 'reify')
        )
        assert cells_total == 4
        assert (grand['topics_with_one_canonical']
                + grand['topics_with_zero_canonical']
                + grand['topics_with_multiple_canonical']) == cells_total
        # And the distinct-VALUE axis is untouched: gate 3626's item-3 history
        # (98 -> 103) is a distinct-value series, so de-duplicating `shared`
        # here is correct and must NOT be regraded to the cell count.
        assert grand['distinct_topics'] == 3

    def test_slug_conformance_stays_on_the_distinct_value_axis(self):
        """The regrade touches the canonical partition ONLY.

        Item 3 counts non-conforming distinct topic VALUES corpus-wide; a
        legacy spelling present in both projects is one non-conforming topic,
        not two, or the series this census continues would step on re-grain.
        """
        legacy = 'legacy_snake_case_topic'
        cells = {
            'dark_factory': {OBS: _census([{'topic': legacy}])},
            'reify': {OBS: _census([{'topic': legacy}])},
        }
        cov = {
            'dark_factory': _coverage('fused_dark_factory', 1, {OBS: (1, 1)}),
            'reify': _coverage('fused_reify', 1, {OBS: (1, 1)}),
        }
        report = _mod.build_report(cells, cov, top_n=50)
        grand = report['topic_coverage']
        assert grand['slug_non_conforming'] == 1
        assert [r['topic'] for r in grand['non_conforming_topics']] == [legacy]

    def test_the_regrade_touches_exactly_the_fields_the_constant_names(self):
        """`_UNIQUENESS_GRAIN_FIELDS` is authoritative, in BOTH directions.

        The constant documents itself as "exactly those fields whose meaning
        is fixed by 3198's per-(project, topic) invariant", and the regrade's
        docstring says its scope is "only :data:`_UNIQUENESS_GRAIN_FIELDS`".
        Neither claim was checked, and the implementation carried its own
        duplicate literal of the scalar columns -- so the constant could grow
        a field that nothing re-graded (the corpus-wide block would silently
        headline a merged-across-projects number again, the exact defect the
        regrade exists to fix), or the implementation could grow a re-grade
        the constant does not name.

        Asserted as a SET EQUALITY over the keys whose values actually change,
        so it fails on either drift. The reverse half is load-bearing on its
        own: `distinct_topics` and the slug axes are distinct-VALUE series
        that gate 3626's recipe item 3 tracks across runs (98 -> 103), and
        re-graining one of them to a cell count would step that series
        discontinuously while every existing assertion stayed green.
        """
        # Sentinels on EVERY key, grain-fixed or not: a key that still holds
        # its sentinel was left alone, one that does not was re-graded.
        grand_block = {
            'topics_with_one_canonical': -101,
            'topics_with_zero_canonical': -102,
            'topics_with_multiple_canonical': -103,
            'multiple_canonical_topics': [{'topic': 'sentinel',
                                           'canonical_count': -104}],
            'distinct_topics': -105,
            'slug_conforming': -106,
            'slug_non_conforming': -107,
            'non_conforming_topics': [{'topic': 'sentinel-slug'}],
            'records': -108,
            'topic_present': -109,
            'topic_coverage_pct': -110.0,
        }
        projects = {
            'dark_factory': {'topic_coverage': {
                'topics_with_one_canonical': 2,
                'topics_with_zero_canonical': 3,
                'topics_with_multiple_canonical': 1,
                'multiple_canonical_topics': [{'topic': 'dupe',
                                               'canonical_count': 2}],
            }},
            'reify': {'topic_coverage': {
                'topics_with_one_canonical': 5,
                'topics_with_zero_canonical': 7,
                'topics_with_multiple_canonical': 0,
                'multiple_canonical_topics': [],
            }},
        }

        regraded = _mod._regrade_uniqueness_at_project_grain(grand_block, projects)

        changed = {k for k, v in regraded.items() if v != grand_block[k]}
        assert changed == set(_mod._UNIQUENESS_GRAIN_FIELDS)
        # And the re-grade is the per-project sum, not a passthrough that
        # merely happens to differ from a sentinel.
        assert regraded['topics_with_one_canonical'] == 7
        assert regraded['topics_with_zero_canonical'] == 10
        assert regraded['topics_with_multiple_canonical'] == 1
        assert [r['topic'] for r in regraded['multiple_canonical_topics']] == ['dupe']
        # PURE: the caller's dict is not mutated.
        assert grand_block['topics_with_one_canonical'] == -101


class TestEnforceFlipPreconditionCitations:
    """The report says what stands between here and flipping ``enforce``.

    Leo's 2026-08-12 ruling (Option B on esc-4006-3): emitting the flag tells
    a reader its CURRENT state but not what remains.  This is a REPORTING
    addition -- the report cites the preconditions; it does not chase them.
    """

    @staticmethod
    def _block():
        cells = {'dark_factory': {OBS: _census([{'topic': 't', 'canonical': True}])}}
        cov = {'dark_factory': _coverage('fused_dark_factory', 1, {OBS: (1, 1)})}
        report = _mod.build_report(cells, cov, top_n=50)
        return report['projects']['dark_factory']['topic_coverage']

    def test_exactly_three_citations_each_fully_shaped(self):
        entries = self._block()['enforce_flip_preconditions']
        assert len(entries) == 3
        for entry in entries:
            assert set(entry) >= {'id', 'what', 'status', 'source'}
            assert entry['what'] and entry['source']

    def test_the_three_cited_ids(self):
        ids = [e['id'] for e in self._block()['enforce_flip_preconditions']]
        assert ids == ['legacy_topic_spelling_remains', '3202', '3626']

    def test_legacy_spelling_bucket_is_not_cited_as_flatly_open(self):
        # PRD §159's 2026-08-04 amendment records this half DISCHARGED, so
        # citing it as flatly open would be as wrong as omitting it.  Asserted
        # as "not open" rather than against the date literal: WHICH discharge
        # date is recorded is editorial and will be superseded, but the
        # distinction between a discharged half and an open one is the thing a
        # reader acts on.
        entry = self._block()['enforce_flip_preconditions'][0]
        assert entry['status'] != 'open'
        assert entry['source'], 'a citation with no source is not checkable'

    def test_citations_are_single_homed_in_a_module_constant(self):
        # One home for the three citations, so a future edit cannot leave the
        # JSON and the markdown disagreeing about what blocks the flip.
        ids = [e['id'] for e in _mod.ENFORCE_FLIP_PRECONDITIONS]
        assert ids == [e['id'] for e in self._block()['enforce_flip_preconditions']]

    def test_the_emitted_block_does_not_alias_the_constant(self):
        # Rendering must not hand out the module constant itself: a consumer
        # mutating the report would silently rewrite every later run's
        # citations in the same process (the nightly wrapper censuses two
        # projects in one invocation).
        # Compared against the constant's own captured value rather than a
        # date literal, so this stays a pure aliasing test and does not
        # re-pin the editorial status text the review flagged.
        original = _mod.ENFORCE_FLIP_PRECONDITIONS[0]['status']
        block = self._block()
        block['enforce_flip_preconditions'][0]['status'] = 'MUTATED'
        assert _mod.ENFORCE_FLIP_PRECONDITIONS[0]['status'] == original

    def test_citations_are_citation_only_carrying_no_live_task_probe(self):
        # The ruling's scope note is explicit: cite the preconditions, do not
        # chase them.  A live status probe here would silently expand this
        # task into 3202/3626's territory and could go stale against the task
        # store without any test noticing.
        for entry in self._block()['enforce_flip_preconditions']:
            assert 'task_status' not in entry
            assert 'probed_at' not in entry

    def test_citations_are_emitted_for_every_project_and_the_grand_total(self):
        cells = {
            'dark_factory': {OBS: _census([{'topic': 'a'}])},
            'reify': {OBS: _census([{'topic': 'b'}])},
        }
        cov = {
            'dark_factory': _coverage('fused_dark_factory', 1, {OBS: (1, 1)}),
            'reify': _coverage('fused_reify', 1, {OBS: (1, 1)}),
        }
        report = _mod.build_report(cells, cov, top_n=50)
        for block in (
            report['projects']['dark_factory']['topic_coverage'],
            report['projects']['reify']['topic_coverage'],
            report['topic_coverage'],
        ):
            assert len(block['enforce_flip_preconditions']) == 3


class TestTopicSlugConformance:
    """Two conformance partitions, because the gate asks two questions.

    Gate 3626's re-measurement recipe asks item 3 -- "the non-conforming
    distinct-topic count" -- and item 4 -- "confirm every live canonical:true
    record's topic passes the slug regex, in BOTH projects" -- as SEPARATE
    checks, and PRD §159's discharge claim is stated over the canonical-scoped
    predicate only.  A single merged column would report the exact state §159
    records as of 2026-08-04 (legacy spellings present among ordinary records,
    none among canonicals) as an undifferentiated failure.
    """

    # Live-measured pair: the corpus carries the snake_case spelling (8
    # records) while the committed registry carries the hyphenated one.
    LEGACY = 'architect_report_task_already_done_main_reachability'
    CONFORMING = 'architect-report-task-already-done-main-reachability'

    @staticmethod
    def _block(payloads, project='dark_factory'):
        cells = {project: {OBS: _census(payloads)}}
        collection = 'fused_dark_factory' if project == 'dark_factory' else 'fused_reify'
        cov = {
            project: _coverage(
                collection, len(payloads), {OBS: (len(payloads), len(payloads))}
            )
        }
        report = _mod.build_report(cells, cov, top_n=50)
        return report['projects'][project]['topic_coverage']

    def test_predicate_is_the_shared_one_pinned_by_identity(self):
        # INV-5: pinned by ``is``, matching tests/test_topic_slug_namespace.py,
        # so no second copy of the regex can drift from the rule that
        # metadata.topic and ProceduralTopicCluster.topic_id are both
        # validated against.
        import fused_memory.topic_slug as ts

        assert _mod.is_valid_topic_slug is ts.is_valid_topic_slug

    def test_distinct_topics_are_partitioned_by_conformance(self):
        block = self._block([
            {'topic': self.LEGACY},
            {'topic': self.LEGACY},
            {'topic': self.CONFORMING},
            {'topic': 'merge-lane'},
        ])
        assert block['distinct_topics'] == 3
        assert block['slug_conforming'] == 2
        assert block['slug_non_conforming'] == 1

    def test_the_partition_counts_DISTINCT_topics_not_records(self):
        # Recipe item 3 asks for the distinct-topic count (its 98 -> 103
        # history is a distinct-value series), so a topic stamped on 8
        # records is ONE non-conforming topic, not eight.
        block = self._block([{'topic': self.LEGACY}] * 8)
        assert block['slug_non_conforming'] == 1

    def test_non_conforming_topics_are_named(self):
        block = self._block([
            {'topic': self.LEGACY},
            {'topic': 'Also Bad'},
            {'topic': 'merge-lane'},
        ])
        assert [r['topic'] for r in block['non_conforming_topics']] == [
            'Also Bad', self.LEGACY,
        ]
        assert all('count' in r for r in block['non_conforming_topics'])

    def test_the_legacy_and_conforming_spellings_coexist_as_two_topics(self):
        block = self._block([{'topic': self.LEGACY}, {'topic': self.CONFORMING}])
        assert block['distinct_topics'] == 2
        assert block['slug_non_conforming'] == 1
        assert [r['topic'] for r in block['non_conforming_topics']] == [self.LEGACY]

    def test_conformance_partition_sums_to_distinct_topics(self):
        block = self._block([
            {'topic': self.LEGACY}, {'topic': 'ok-one'}, {'topic': 'ok-two'},
        ])
        assert block['slug_conforming'] + block['slug_non_conforming'] == \
            block['distinct_topics'] == 3

    def test_canonical_scoped_partition_is_separate_and_narrower(self):
        block = self._block([
            {'topic': self.LEGACY, 'canonical': True},
            {'topic': 'merge-lane', 'canonical': True},
            {'topic': 'census'},
        ])
        assert block['slug_non_conforming'] == 1
        assert block['canonical_slug_conforming'] == 1
        assert block['canonical_slug_non_conforming'] == 1
        assert [r['topic'] for r in block['canonical_slug_non_conforming_topics']] == [
            self.LEGACY,
        ]

    def test_the_2026_08_04_recorded_state_reports_as_two_DIFFERENT_numbers(self):
        # THE discriminating case.  PRD §159 records legacy spellings present
        # among ordinary records but NONE among canonicals -- item 3 non-zero,
        # item 4 zero.  A single merged column would collapse this into one
        # undifferentiated failure and mis-report the discharge.
        block = self._block([
            {'topic': self.LEGACY},
            {'topic': self.LEGACY},
            {'topic': self.CONFORMING, 'canonical': True},
            {'topic': 'merge-lane', 'canonical': True},
        ])
        assert block['slug_non_conforming'] == 1
        assert block['canonical_slug_non_conforming'] == 0
        assert block['canonical_slug_non_conforming_topics'] == []
        assert block['canonical_slug_conforming'] == 2

    def test_canonical_scoped_partition_ignores_non_canonical_records(self):
        block = self._block([{'topic': self.LEGACY}, {'topic': self.LEGACY}])
        assert block['canonical_slug_conforming'] == 0
        assert block['canonical_slug_non_conforming'] == 0

    def test_both_partitions_are_at_project_grain(self):
        cells = {
            'dark_factory': {
                PROC: _census([{'topic': self.LEGACY}]),
                OBS: _census([{'topic': self.LEGACY, 'canonical': True}]),
            },
            'reify': {OBS: _census([{'topic': 'clean-slug', 'canonical': True}])},
        }
        cov = {
            'dark_factory': _coverage(
                'fused_dark_factory', 2, {PROC: (1, 1), OBS: (1, 1)},
            ),
            'reify': _coverage('fused_reify', 1, {OBS: (1, 1)}),
        }
        report = _mod.build_report(cells, cov, top_n=50)
        df = report['projects']['dark_factory']['topic_coverage']
        rf = report['projects']['reify']['topic_coverage']
        # One distinct topic in dark_factory across two categories, not two.
        assert df['slug_non_conforming'] == 1
        assert df['canonical_slug_non_conforming'] == 1
        assert rf['slug_non_conforming'] == 0
        assert rf['canonical_slug_non_conforming'] == 0

    def test_named_lists_follow_the_house_total_order(self):
        block = self._block([
            {'topic': 'Bad One'}, {'topic': 'Bad One'}, {'topic': 'Bad One'},
            {'topic': 'Zed Bad'}, {'topic': 'Zed Bad'},
            {'topic': 'Aaa Bad'}, {'topic': 'Aaa Bad'},
        ])
        rows = block['non_conforming_topics']
        assert [(r['topic'], r['count']) for r in rows] == [
            ('Bad One', 3), ('Aaa Bad', 2), ('Zed Bad', 2),
        ]

    def test_the_legacy_bucket_citation_carries_this_runs_re_measurement(self):
        # A bucket recorded empty on 2026-08-04 can REGROW -- the leaf-alpha
        # census's own 98 -> 103 history proves it does.  So the citation must
        # carry a live number, not only a recorded status.
        block = self._block([
            {'topic': self.LEGACY},
            {'topic': self.CONFORMING, 'canonical': True},
        ])
        entry = block['enforce_flip_preconditions'][0]
        assert entry['id'] == 'legacy_topic_spelling_remains'
        live = entry['live_re_measurement']
        assert live['slug_non_conforming'] == 1
        assert live['canonical_slug_non_conforming'] == 0

    def test_the_other_two_citations_carry_no_re_measurement(self):
        # 3202 and 3626 are cited, not measured -- the ruling's scope note.
        block = self._block([{'topic': 'ok'}])
        for entry in block['enforce_flip_preconditions'][1:]:
            assert 'live_re_measurement' not in entry


class _FakeEntry:
    """Minimal stand-in for the probe's RegistryEntry (topic + project_id)."""

    def __init__(self, topic: str, project_id: str):
        self.topic = topic
        self.project_id = project_id


class _FakeRegistry:
    def __init__(self, entries):
        self.entries = entries


class TestRegistryCoverageGauge:
    """The TARGET: every committed registry topic carries exactly one canonical.

    A corpus-wide stamping percentage over 49.6k records has no owner and no
    bounded worklist.  The committed 32-entry registry is a bounded, named,
    checkable set -- and it is precisely the set E1's retrieval-health run
    scores 32/32 failing.

    NOT a duplicate of E1's METRIC_TOPIC_CANONICAL_PRESENT (INV-5): that one
    is computed from _tripwire_items(observations, TRIPWIRE_K) -- from SEARCH
    observations -- so it cannot distinguish "the canonical does not exist"
    from "it exists but ranks below K".  Only a deterministic metadata count
    can establish the former, which is this task's central claim.
    """

    @staticmethod
    def _report(cells, registry, **kwargs):
        cov = {
            pid: _coverage(
                f'fused_{pid}',
                sum(c.records for c in cats.values()),
                {cat: (c.records, c.records) for cat, c in cats.items()},
            )
            for pid, cats in cells.items()
        }
        return _mod.build_report(cells, cov, top_n=50, registry=registry, **kwargs)

    def test_loader_is_the_probes_pinned_by_identity(self):
        # Imported through the same _load_probe_module importlib idiom
        # fused-memory/scripts/retro_stamp_topics.py::_load_probe_module and
        # ::load_topic_registry already established, so the schema
        # check, the zero-entry rejection and the RegistryError type stay
        # single-homed rather than re-parsed here.
        import importlib.util as _ilu
        import sys as _sys
        from pathlib import Path as _Path

        assert _mod.__file__ is not None
        probe_path = _Path(_mod.__file__).parent / 'memory_eval_retrieval_probe.py'
        cached = _sys.modules.get('memory_eval_retrieval_probe')
        if cached is None:
            spec = _ilu.spec_from_file_location('memory_eval_retrieval_probe', probe_path)
            assert spec is not None and spec.loader is not None
            cached = _ilu.module_from_spec(spec)
            _sys.modules['memory_eval_retrieval_probe'] = cached
            spec.loader.exec_module(cached)
        # Through the ACCESSOR, not a module attribute: the loader is bound
        # lazily so an unloadable probe degrades to registry_error instead of
        # killing the import (and with it the whole census). The reuse is
        # still pinned by identity.
        assert _mod.topic_registry_loader() is cached.load_topic_registry

    # DELIBERATELY NOT TESTED HERE (review of 2026-08-16): that the probe is
    # not loaded AT IMPORT TIME, via `inspect.getsource` +
    # `'_load_probe_module()' in src` and an AST sweep for top-level call
    # nodes.  That test made a RUNTIME claim its own body never observed at
    # runtime: a correct lazy rewrite reaching the helper through an alias or
    # attribute failed it, while an eager load reached via `getattr` passed
    # it.  The property that actually matters -- an unloadable probe degrades
    # to `registry_error` instead of killing the census import -- is asserted
    # behaviourally by the test directly below, and the loader's lazy binding
    # and reuse are pinned by identity in the test directly above.  Do not
    # reinstate this as a tightened regex or AST pin.

    def test_an_unloadable_probe_is_DISCLOSED_not_fatal(self, monkeypatch):
        # The registry gauge is one optional block. A probe that cannot be
        # loaded must take the same registry_error path every other registry
        # failure takes -- not abort the measurement.
        monkeypatch.setattr(_mod, '_TOPIC_REGISTRY_LOADER', None)

        def _boom():
            raise ImportError('memory_eval_retrieval_probe moved')

        monkeypatch.setattr(_mod, '_load_probe_module', _boom)
        registry, error = _mod.load_registry_or_error(_mod.DEFAULT_REGISTRY_PATH)
        assert registry is None
        assert error is not None
        assert error.startswith('ImportError:')

    def test_default_registry_path_is_the_committed_fixture(self):
        assert _mod.DEFAULT_REGISTRY_PATH.endswith(
            'tests/fixtures/memory_eval_topic_registry.json',
        )
        assert Path(_mod.DEFAULT_REGISTRY_PATH).is_file()

    def test_per_topic_record_and_canonical_counts(self):
        cells = {
            'dark_factory': {
                OBS: _census([
                    {'topic': 'alpha', 'canonical': True},
                    {'topic': 'alpha'},
                    {'topic': 'alpha'},
                    {'topic': 'beta'},
                ]),
            },
        }
        registry = _FakeRegistry([
            _FakeEntry('alpha', 'dark_factory'),
            _FakeEntry('beta', 'dark_factory'),
            _FakeEntry('gamma', 'dark_factory'),
        ])
        block = self._report(cells, registry)['registry_coverage']
        rows = {r['topic']: r for r in block['topics']}
        assert rows['alpha']['records'] == 3
        assert rows['alpha']['canonical_count'] == 1
        assert rows['beta']['records'] == 1
        assert rows['beta']['canonical_count'] == 0
        # Absent from the corpus entirely -- measured zero, not omitted.
        assert rows['gamma']['records'] == 0
        assert rows['gamma']['canonical_count'] == 0

    def test_rollup_partitions_the_registry(self):
        cells = {
            'dark_factory': {
                OBS: _census([
                    {'topic': 'one', 'canonical': True},
                    {'topic': 'many', 'canonical': True},
                    {'topic': 'many', 'canonical': True},
                    {'topic': 'zero'},
                ]),
            },
        }
        registry = _FakeRegistry([
            _FakeEntry(t, 'dark_factory') for t in ('one', 'many', 'zero', 'absent')
        ])
        block = self._report(cells, registry)['registry_coverage']
        assert block['registry_topics_total'] == 4
        assert block['registry_topics_with_exactly_one_canonical'] == 1
        assert block['registry_topics_with_zero_canonical'] == 2
        assert block['registry_topics_with_multiple_canonical'] == 1

    def test_zero_canonical_topics_are_named_as_the_worklist(self):
        cells = {
            'dark_factory': {
                OBS: _census([{'topic': 'has-one', 'canonical': True}, {'topic': 'bare'}]),
            },
        }
        registry = _FakeRegistry([
            _FakeEntry('has-one', 'dark_factory'),
            _FakeEntry('bare', 'dark_factory'),
            _FakeEntry('missing', 'dark_factory'),
        ])
        block = self._report(cells, registry)['registry_coverage']
        # The actionable list -- exactly what 3201's retro sweep would stamp.
        assert [r['topic'] for r in block['zero_canonical_topics']] == [
            'bare', 'missing',
        ]

    def test_registry_is_matched_within_its_own_project_not_globally(self):
        # A canonical stamped in reify does NOT discharge a dark_factory
        # registry entry: 3198's invariant is per-(project, topic), and a
        # global match would report the target met while the dark_factory
        # corpus stayed bare.
        cells = {
            'dark_factory': {OBS: _census([{'topic': 'shared'}])},
            'reify': {OBS: _census([{'topic': 'shared', 'canonical': True}])},
        }
        registry = _FakeRegistry([_FakeEntry('shared', 'dark_factory')])
        block = self._report(cells, registry)['registry_coverage']
        assert block['registry_topics_with_zero_canonical'] == 1
        assert block['registry_topics_with_exactly_one_canonical'] == 0

    def test_an_entry_for_an_uncensused_project_is_measured_as_zero(self):
        cells = {'dark_factory': {OBS: _census([{'topic': 'a', 'canonical': True}])}}
        registry = _FakeRegistry([
            _FakeEntry('a', 'dark_factory'), _FakeEntry('b', 'reify'),
        ])
        block = self._report(cells, registry)['registry_coverage']
        rows = {r['topic']: r for r in block['topics']}
        assert rows['b']['records'] == 0
        assert rows['b']['project_id'] == 'reify'
        assert block['registry_topics_with_zero_canonical'] == 1

    def test_rows_carry_their_project_and_are_deterministically_ordered(self):
        cells = {'dark_factory': {OBS: _census([])}}
        registry = _FakeRegistry([
            _FakeEntry('zulu', 'reify'),
            _FakeEntry('alpha', 'reify'),
            _FakeEntry('mike', 'dark_factory'),
        ])
        block = self._report(cells, registry)['registry_coverage']
        assert [(r['project_id'], r['topic']) for r in block['topics']] == [
            ('dark_factory', 'mike'), ('reify', 'alpha'), ('reify', 'zulu'),
        ]

    def test_a_missing_registry_is_disclosed_not_silently_all_zero(self):
        # A silent all-zero gauge reads as "measured, and everything is
        # fine" -- indistinguishable from a real clean result. The whole
        # point of the gauge is that its zero is meaningful.
        cells = {'dark_factory': {OBS: _census([{'topic': 'a'}])}}
        report = self._report(cells, None, registry_error='cannot read registry: nope')
        assert report['registry_coverage'] is None
        assert 'cannot read registry' in report['registry_error']

    def test_a_zero_entry_registry_is_disclosed_as_an_error_not_a_perfect_score(self):
        cells = {'dark_factory': {OBS: _census([{'topic': 'a'}])}}
        report = self._report(cells, _FakeRegistry([]))
        assert report['registry_coverage'] is None
        assert report['registry_error']

    def test_registry_error_is_absent_when_the_gauge_loaded(self):
        cells = {'dark_factory': {OBS: _census([{'topic': 'a', 'canonical': True}])}}
        report = self._report(cells, _FakeRegistry([_FakeEntry('a', 'dark_factory')]))
        assert report['registry_error'] is None
        assert report['registry_coverage']['registry_topics_total'] == 1

    def test_a_registry_of_the_wrong_SHAPE_is_disclosed_not_fatal(self):
        """A loaded-but-misshapen registry degrades like every other failure.

        The load path is already careful -- `topic_registry_loader` is lazily
        memoized and `load_registry_or_error` catches everything -- precisely
        so the optional gauge cannot take the measurement down with it. But
        SCORING happens after the load, reaching `entry.project_id` /
        `entry.topic` by attribute on objects owned by ANOTHER script's
        dataclass. A `RegistryEntry` field rename (or a loader that starts
        returning dicts) therefore raised straight out of `build_report`:
        no report, no markdown, no trend row, from a block the design says
        is optional.

        This is not hypothetical reach: the registry, its loader and its
        models all live in `memory_eval_retrieval_probe.py`, a file this task
        does not own and does not test, and the census reuses them by import
        exactly so they stay single-homed. Reuse across that seam is right;
        inheriting an abort from it is not.
        """
        class _ShapelessEntry:
            project_id = 'dark_factory'
            # No `.topic` -- the shape a field rename in the probe produces.

        cells = {'dark_factory': {OBS: _census([{'topic': 'a', 'canonical': True}])}}
        report = self._report(cells, _FakeRegistry([_ShapelessEntry()]))

        assert report['registry_coverage'] is None
        assert 'AttributeError' in report['registry_error']
        # The MEASUREMENT survived intact -- that is the whole claim.
        assert report['projects']['dark_factory']['topic_coverage'][
            'topics_with_one_canonical'] == 1

    def test_a_registry_that_is_not_iterable_at_all_is_disclosed_not_fatal(self):
        """The same guard, reached through a different failure shape."""
        class _HostileRegistry:
            @property
            def entries(self):
                raise RuntimeError('probe module changed under us')

        cells = {'dark_factory': {OBS: _census([{'topic': 'a'}])}}
        report = self._report(cells, _HostileRegistry())

        assert report['registry_coverage'] is None
        assert 'RuntimeError' in report['registry_error']
        assert report['projects']['dark_factory']['topic_coverage']['records'] == 1

    def test_gauge_is_json_serialisable_and_deterministic(self):
        cells = {'dark_factory': {OBS: _census([{'topic': 'a', 'canonical': True}])}}
        registry = _FakeRegistry([
            _FakeEntry('a', 'dark_factory'), _FakeEntry('b', 'reify'),
        ])
        first = self._report(cells, registry)
        second = self._report(cells, registry)
        assert json.dumps(first) == json.dumps(second)

    def test_the_committed_registry_still_holds_the_32_named_targets(self):
        # The accountable set this task names as the TARGET. If the fixture
        # grows or shrinks, the target moved and the docs must say so.
        registry = _mod.topic_registry_loader()(_mod.DEFAULT_REGISTRY_PATH)
        assert len(registry.entries) == 32


def _history_report(registry=None, **kwargs) -> dict:
    """A two-project report whose topic_coverage blocks carry known scalars.

    dark_factory: 5 records, 3 topic-stamped, distinct {alpha, legacy_spelling},
    alpha has exactly one canonical, legacy_spelling none, plus one canonical
    carrying NO topic. reify: 3 records, all stamped 'beta', which holds TWO
    canonicals (a uniqueness violation, reachable under the warn-mode default).
    """
    cells = {
        'dark_factory': {
            OBS: _census([
                {'topic': 'alpha', 'canonical': True},
                {'topic': 'alpha'},
                {'topic': 'legacy_spelling'},
                {'canonical': True},
                {},
            ]),
        },
        'reify': {
            OBS: _census([
                {'topic': 'beta'},
                {'topic': 'beta', 'canonical': True},
                {'topic': 'beta', 'canonical': True},
            ]),
        },
    }
    cov = {
        pid: _coverage(
            f'fused_{pid}',
            sum(c.records for c in cats.values()),
            {cat: (c.records, c.records) for cat, c in cats.items()},
        )
        for pid, cats in cells.items()
    }
    return _mod.build_report(cells, cov, top_n=50, registry=registry, **kwargs)


def _walk(obj):
    """Yield every nested value in a JSON-ish structure, including the root."""
    yield obj
    if isinstance(obj, dict):
        for v in obj.values():
            yield from _walk(v)
    elif isinstance(obj, list):
        for v in obj:
            yield from _walk(v)


class TestCoverageHistory:
    """The TREND substrate: a committed, append-only history of headline scalars.

    The ask names trending as the point -- a number measured once is not
    owned.  Diffing consecutive runs needs durable prior state, and
    ``docs/legibility/census-state.json`` is the in-repo precedent for a
    committed one.

    But this file is written by a NIGHTLY timer, so the row must stay
    compact: persisting the per-topic tables (276 distinct topics in reify
    alone) would grow the committed repo without bound on every run.  The
    full tables stay in the regenerated report artifact, which is rewritten
    rather than appended.

    An absent history is an empty history; a MALFORMED one is an error.
    Losing the trend must be loud -- restarting silently from empty is how
    ``census_trigger.extract_done_count`` fabricated a ``0`` baseline twice
    and armed a threshold with it (task 3291).
    """

    # ---- the compact row -------------------------------------------------

    def test_appends_exactly_one_row_per_project(self):
        report = _history_report()
        history = _mod.append_coverage_run(
            _mod.empty_coverage_history(), report, stamp='2026-08-16T05:00:00Z',
        )
        assert len(history['runs']) == 1
        run = history['runs'][0]
        assert set(run['projects']) == {'dark_factory', 'reify'}

    def test_row_carries_the_headline_scalars(self):
        report = _history_report()
        history = _mod.append_coverage_run(
            _mod.empty_coverage_history(), report, stamp='2026-08-16T05:00:00Z',
        )
        row = history['runs'][0]['projects']['dark_factory']
        assert row['records'] == 5
        assert row['topic_present'] == 3
        assert row['topic_coverage_pct'] == 60.0
        assert row['distinct_topics'] == 2
        assert row['topics_with_one_canonical'] == 1
        assert row['topics_with_zero_canonical'] == 1
        assert row['topics_with_multiple_canonical'] == 0
        assert row['canonical_true'] == 2
        assert row['canonical_true_without_topic'] == 1
        assert row['slug_conforming'] == 1
        assert row['slug_non_conforming'] == 1
        assert row['canonical_slug_conforming'] == 1
        assert row['canonical_slug_non_conforming'] == 0

        reify = history['runs'][0]['projects']['reify']
        assert reify['topics_with_multiple_canonical'] == 1

    def test_row_holds_scalars_only_never_the_per_topic_tables(self):
        # The bound on the committed file's growth is STRUCTURAL, not a
        # promise: nothing that scales with the corpus may enter a row.
        report = _history_report()
        history = _mod.append_coverage_run(
            _mod.empty_coverage_history(), report, stamp='2026-08-16T05:00:00Z',
        )
        row = history['runs'][0]['projects']['dark_factory']
        for value in row.values():
            assert isinstance(value, (bool, int, float, str, type(None))), row
        for key in (
            'topic', 'keys', 'kind', 'canonical_true_by_topic',
            'multiple_canonical_topics', 'non_conforming_topics',
            'canonical_slug_non_conforming_topics', 'enforce_flip_preconditions',
        ):
            assert key not in row

    def test_no_uncapped_value_table_survives_anywhere_in_a_run(self):
        # _table()'s {'entries': [...], 'distinct_total': n} shape is the
        # unbounded thing; assert none of it reached the persisted run.
        report = _history_report()
        run = _mod.append_coverage_run(
            _mod.empty_coverage_history(), report, stamp='2026-08-16T05:00:00Z',
        )['runs'][0]
        for node in _walk(run):
            if isinstance(node, dict):
                assert 'distinct_total' not in node

    def test_run_records_the_enforce_regime_the_counts_were_measured_under(self):
        report = _history_report(canonical_uniqueness_enforced=False)
        run = _mod.append_coverage_run(
            _mod.empty_coverage_history(), report, stamp='2026-08-16T05:00:00Z',
        )['runs'][0]
        # A violation count trended across a flag flip is two different
        # series; without the flag beside it the trend is unreadable.
        assert run['canonical_uniqueness_enforced'] is False

    # ---- the registry digest: bounded by the TARGET SET, not the corpus ---

    def test_registry_scalars_land_per_project(self):
        registry = _FakeRegistry([
            _FakeEntry('alpha', 'dark_factory'),
            _FakeEntry('legacy_spelling', 'dark_factory'),
            _FakeEntry('beta', 'reify'),
        ])
        report = _history_report(registry=registry)
        run = _mod.append_coverage_run(
            _mod.empty_coverage_history(), report, stamp='2026-08-16T05:00:00Z',
        )['runs'][0]
        dark = run['projects']['dark_factory']
        assert dark['registry_topics_total'] == 2
        assert dark['registry_topics_with_exactly_one_canonical'] == 1
        assert dark['registry_topics_with_zero_canonical'] == 1
        assert dark['registry_topics_with_multiple_canonical'] == 0
        assert run['projects']['reify']['registry_topics_with_multiple_canonical'] == 1

    def test_registry_scalars_are_null_not_zero_when_the_gauge_did_not_run(self):
        # Three-valued, like every other unread axis here: a 0 would read as
        # "measured, and no registry topic is stamped" -- a different claim
        # from "the gauge did not run", and the one that fabricates a trend.
        report = _history_report()
        run = _mod.append_coverage_run(
            _mod.empty_coverage_history(), report, stamp='2026-08-16T05:00:00Z',
        )['runs'][0]
        assert run['projects']['dark_factory']['registry_topics_total'] is None
        assert run['registry_error']

    def test_registry_digest_is_bounded_by_the_registry_not_the_corpus(self):
        # The ONE per-topic structure a run may carry, and only because the
        # committed registry bounds it (32 entries). It is what makes
        # "this registry topic LOST its canonical" nameable in the diff --
        # the actionable regression -- without persisting 353 live topics.
        registry = _FakeRegistry([
            _FakeEntry('alpha', 'dark_factory'),
            _FakeEntry('legacy_spelling', 'dark_factory'),
            _FakeEntry('beta', 'reify'),
        ])
        report = _history_report(registry=registry)
        run = _mod.append_coverage_run(
            _mod.empty_coverage_history(), report, stamp='2026-08-16T05:00:00Z',
        )['runs'][0]
        digest = run['registry_topics']
        assert len(digest) == 3
        assert digest['dark_factory::alpha'] == {'records': 2, 'canonical': 1}
        assert digest['dark_factory::legacy_spelling'] == {'records': 1, 'canonical': 0}
        # Keyed by project too: 3198's invariant is per-(project, topic), so
        # a canonical in reify must never discharge a dark_factory entry.
        assert digest['reify::beta'] == {'records': 3, 'canonical': 2}
        # No live topic outside the registry may appear -- that set is
        # unbounded and is exactly what must stay in the report artifact.
        assert not [k for k in digest if k.endswith('::gamma')]

    def test_registry_digest_is_absent_when_the_gauge_did_not_run(self):
        run = _mod.append_coverage_run(
            _mod.empty_coverage_history(), _history_report(),
            stamp='2026-08-16T05:00:00Z',
        )['runs'][0]
        assert run['registry_topics'] == {}

    # ---- append-only ordering, purity, retention -------------------------

    def test_history_is_append_only_and_ordered_oldest_first(self):
        report = _history_report()
        history = _mod.empty_coverage_history()
        for stamp in ('2026-08-14T05:00:00Z', '2026-08-15T05:00:00Z', '2026-08-16T05:00:00Z'):
            history = _mod.append_coverage_run(history, report, stamp=stamp)
        assert [r['stamp'] for r in history['runs']] == [
            '2026-08-14T05:00:00Z', '2026-08-15T05:00:00Z', '2026-08-16T05:00:00Z',
        ]

    def test_append_does_not_mutate_the_history_it_was_given(self):
        history = _mod.empty_coverage_history()
        before = json.dumps(history)
        _mod.append_coverage_run(history, _history_report(), stamp='2026-08-16T05:00:00Z')
        assert json.dumps(history) == before

    def test_stamp_is_injected_never_read_from_the_clock(self):
        # Injected so the row is deterministic and testable; a datetime.now()
        # inside would make the artifact unreproducible and the test racy.
        with pytest.raises(TypeError):
            _mod.append_coverage_run(_mod.empty_coverage_history(), _history_report())

    def test_retention_drops_the_oldest_and_DISCLOSES_the_drop(self):
        report = _history_report()
        history = _mod.empty_coverage_history()
        for day in range(1, 6):
            history = _mod.append_coverage_run(
                history, report, stamp=f'2026-08-{day:02d}T05:00:00Z', max_runs=3,
            )
        assert [r['stamp'] for r in history['runs']] == [
            '2026-08-03T05:00:00Z', '2026-08-04T05:00:00Z', '2026-08-05T05:00:00Z',
        ]
        # Silently truncated history would look like a corpus that only ever
        # had three runs -- the same no-silent-truncation rule the JSON
        # tables already follow.
        assert history['retention']['max_runs'] == 3
        assert history['retention']['dropped_runs'] == 2

    # ---- load / save -----------------------------------------------------

    def test_absent_history_file_loads_as_an_empty_history(self, tmp_path):
        history = _mod.load_coverage_history(str(tmp_path / 'nope.json'))
        assert history['runs'] == []
        assert history['schema_version'] == _mod.COVERAGE_HISTORY_SCHEMA_VERSION

    def test_malformed_history_raises_rather_than_restarting_from_empty(self, tmp_path):
        path = tmp_path / 'history.json'
        path.write_text('{not json', encoding='utf-8')
        with pytest.raises(_mod.CoverageHistoryError) as exc:
            _mod.load_coverage_history(str(path))
        assert 'history.json' in str(exc.value)

    def test_a_history_without_a_runs_list_raises(self, tmp_path):
        path = tmp_path / 'history.json'
        path.write_text(json.dumps({'schema_version': 1, 'runs': {}}), encoding='utf-8')
        with pytest.raises(_mod.CoverageHistoryError):
            _mod.load_coverage_history(str(path))

    def test_an_unknown_schema_version_raises_rather_than_being_misread(self, tmp_path):
        path = tmp_path / 'history.json'
        path.write_text(json.dumps({'schema_version': 99, 'runs': []}), encoding='utf-8')
        with pytest.raises(_mod.CoverageHistoryError) as exc:
            _mod.load_coverage_history(str(path))
        assert '99' in str(exc.value)

    def test_round_trip_write_then_load_is_stable(self, tmp_path):
        path = str(tmp_path / 'history.json')
        history = _mod.append_coverage_run(
            _mod.empty_coverage_history(),
            _history_report(registry=_FakeRegistry([_FakeEntry('alpha', 'dark_factory')])),
            stamp='2026-08-16T05:00:00Z',
        )
        _mod.save_coverage_history(history, path)
        assert _mod.load_coverage_history(path) == history

    def test_save_creates_missing_parent_directories(self, tmp_path):
        path = str(tmp_path / 'nested' / 'dir' / 'history.json')
        _mod.save_coverage_history(_mod.empty_coverage_history(), path)
        assert Path(path).is_file()

    def test_default_history_path_is_committed_beside_the_report(self):
        assert _mod.DEFAULT_HISTORY_OUT.endswith(
            'plans/memory-metadata-coverage-history.json',
        )


def _topic_report(payloads_by_project: dict, registry=None, **kwargs) -> dict:
    """A report over ``{project_id: [payload, ...]}``, all in one category."""
    cells = {pid: {OBS: _census(p)} for pid, p in payloads_by_project.items()}
    cov = {
        pid: _coverage(
            f'fused_{pid}',
            sum(c.records for c in cats.values()),
            {cat: (c.records, c.records) for cat, c in cats.items()},
        )
        for pid, cats in cells.items()
    }
    return _mod.build_report(cells, cov, top_n=50, registry=registry, **kwargs)


class TestCoverageDiff:
    """Trending: what MOVED since the most recent prior run.

    The first-ever run is the load-bearing case. It reports ``baseline:
    null`` with a stated reason and emits NO deltas at all -- a table of
    zeros would read as "coverage is flat" when the truth is "there is
    nothing to compare against". That fabricated-zero-baseline failure is
    exactly what task 3291 hardened ``census_trigger.extract_done_count``
    against, where a fabricated 0 was persisted twice and armed a threshold.
    """

    # A pair of runs that exercises all three regrowth classes at once:
    # alpha grows 2 -> 3 records while LOSING its canonical, and beta is a
    # newly registered target that arrives already stamped.
    _PRIOR = {'dark_factory': [{'topic': 'alpha', 'canonical': True}, {'topic': 'alpha'}]}
    _NOW = {
        'dark_factory': [
            {'topic': 'alpha'}, {'topic': 'alpha'}, {'topic': 'alpha'},
            {'topic': 'beta', 'canonical': True},
        ],
    }

    @classmethod
    def _history_with_prior(cls) -> dict:
        prior = _topic_report(
            cls._PRIOR, registry=_FakeRegistry([_FakeEntry('alpha', 'dark_factory')]),
        )
        return _mod.append_coverage_run(
            _mod.empty_coverage_history(), prior, stamp='2026-08-15T05:00:00Z',
        )

    @classmethod
    def _current(cls) -> dict:
        return _topic_report(
            cls._NOW,
            registry=_FakeRegistry([
                _FakeEntry('alpha', 'dark_factory'), _FakeEntry('beta', 'dark_factory'),
            ]),
        )

    # ---- the no-baseline case -------------------------------------------

    def test_first_ever_run_reports_a_null_baseline_with_a_reason(self):
        diff = _mod.build_coverage_diff(_topic_report(self._NOW), _mod.empty_coverage_history())
        assert diff['baseline'] is None
        assert diff['baseline_reason']
        assert diff['projects'] == {}

    def test_first_run_emits_no_fabricated_zero_anywhere(self):
        diff = _mod.build_coverage_diff(_topic_report(self._NOW), _mod.empty_coverage_history())
        zeros = [
            n for n in _walk(diff)
            if isinstance(n, (int, float)) and not isinstance(n, bool) and n == 0
        ]
        assert zeros == []

    def test_an_absent_history_argument_is_disclosed_distinctly(self):
        # "no history was supplied" is a different claim from "the history
        # is empty", and both are different from "coverage is flat".
        diff = _mod.build_coverage_diff(_topic_report(self._NOW), None)
        assert diff['baseline'] is None
        assert diff['baseline_reason'] != _mod.build_coverage_diff(
            _topic_report(self._NOW), _mod.empty_coverage_history(),
        )['baseline_reason']

    # ---- per-project deltas ---------------------------------------------

    def test_baseline_is_the_most_recent_prior_run(self):
        history = self._history_with_prior()
        history = _mod.append_coverage_run(
            history, _topic_report(self._PRIOR), stamp='2026-08-16T05:00:00Z',
        )
        diff = _mod.build_coverage_diff(self._current(), history)
        assert diff['baseline'] == '2026-08-16T05:00:00Z'

    def test_every_headline_column_is_trended(self):
        diff = _mod.build_coverage_diff(self._current(), self._history_with_prior())
        columns = set(diff['projects']['dark_factory']['columns'])
        assert columns == set(
            _mod._HISTORY_COVERAGE_COLUMNS + _mod._HISTORY_REGISTRY_COLUMNS,
        )

    def test_each_column_carries_before_after_and_delta(self):
        diff = _mod.build_coverage_diff(self._current(), self._history_with_prior())
        cols = diff['projects']['dark_factory']['columns']
        assert cols['records'] == {'before': 2, 'after': 4, 'delta': 2}
        assert cols['distinct_topics'] == {'before': 1, 'after': 2, 'delta': 1}
        assert cols['topics_with_zero_canonical'] == {'before': 0, 'after': 1, 'delta': 1}
        # Unchanged columns keep a real 0 -- there IS a baseline here, so 0
        # is a measurement, not a fabrication.
        assert cols['canonical_true']['delta'] == 0

    def test_an_unmeasured_axis_diffs_to_null_not_zero(self):
        # The registry gauge ran now but not before: "we cannot compare" is
        # not "nothing changed".
        history = _mod.append_coverage_run(
            _mod.empty_coverage_history(), _topic_report(self._PRIOR),
            stamp='2026-08-15T05:00:00Z',
        )
        diff = _mod.build_coverage_diff(self._current(), history)
        column = diff['projects']['dark_factory']['columns']['registry_topics_total']
        assert column['before'] is None
        assert column['delta'] is None
        assert column['after'] == 2

    def test_a_project_absent_from_the_prior_run_is_new_not_a_delta_from_zero(self):
        current = _topic_report({**self._NOW, 'reify': [{'topic': 'gamma'}]})
        diff = _mod.build_coverage_diff(current, self._history_with_prior())
        assert diff['projects']['reify']['status'] == 'new_project'
        assert 'columns' not in diff['projects']['reify']

    def test_a_project_absent_from_this_run_is_disclosed_not_dropped(self):
        history = _mod.append_coverage_run(
            _mod.empty_coverage_history(),
            _topic_report({**self._PRIOR, 'reify': [{'topic': 'gamma'}]}),
            stamp='2026-08-15T05:00:00Z',
        )
        diff = _mod.build_coverage_diff(_topic_report(self._NOW), history)
        assert diff['projects']['reify']['status'] == 'absent_from_this_run'

    # ---- per-topic regrowth ---------------------------------------------

    def test_regrowth_names_newly_registered_topics(self):
        regrowth = _mod.build_coverage_diff(
            self._current(), self._history_with_prior(),
        )['topic_regrowth']
        assert regrowth['new_topics'] == [
            {'project_id': 'dark_factory', 'topic': 'beta', 'records': 1, 'canonical': 1},
        ]

    def test_regrowth_names_topics_whose_member_count_grew(self):
        regrowth = _mod.build_coverage_diff(
            self._current(), self._history_with_prior(),
        )['topic_regrowth']
        assert regrowth['grown_topics'] == [
            {
                'project_id': 'dark_factory', 'topic': 'alpha',
                'records_before': 2, 'records_after': 3,
            },
        ]

    def test_regrowth_names_topics_that_LOST_their_canonical(self):
        # The actionable regression: a target that WAS met and no longer is.
        regrowth = _mod.build_coverage_diff(
            self._current(), self._history_with_prior(),
        )['topic_regrowth']
        assert regrowth['lost_canonical_topics'] == [
            {
                'project_id': 'dark_factory', 'topic': 'alpha',
                'canonical_before': 1, 'canonical_after': 0,
            },
        ]

    def test_regrowth_scope_is_disclosed_as_registry_bounded(self):
        # The signal is scoped to the committed registry because that is
        # what the history may bound-safely carry. Disclosed, never silently
        # narrowed -- a reader must not take it for a corpus-wide sweep.
        regrowth = _mod.build_coverage_diff(
            self._current(), self._history_with_prior(),
        )['topic_regrowth']
        assert regrowth['available'] is True
        assert regrowth['scope'] == 'registry'
        assert 'registry' in regrowth['scope_note']

    def test_regrowth_is_unavailable_rather_than_all_new_when_the_baseline_lacks_a_digest(self):
        # Without this, every registry topic would be reported as "newly
        # appearing" the first time the gauge runs -- a fabricated signal
        # indistinguishable from a genuine 32-topic regression.
        history = _mod.append_coverage_run(
            _mod.empty_coverage_history(), _topic_report(self._PRIOR),
            stamp='2026-08-15T05:00:00Z',
        )
        regrowth = _mod.build_coverage_diff(self._current(), history)['topic_regrowth']
        assert regrowth['available'] is False
        assert regrowth['reason']
        assert regrowth['new_topics'] == []

    # ---- wiring + determinism -------------------------------------------

    def test_report_carries_the_trend_under_coverage_trend(self):
        report = _topic_report(
            self._NOW,
            registry=_FakeRegistry([
                _FakeEntry('alpha', 'dark_factory'), _FakeEntry('beta', 'dark_factory'),
            ]),
            history=self._history_with_prior(),
        )
        assert report['coverage_trend']['baseline'] == '2026-08-15T05:00:00Z'
        assert report['coverage_trend']['projects']['dark_factory']['status'] == 'compared'

    def test_a_report_built_without_history_still_carries_a_disclosed_trend(self):
        report = _topic_report(self._NOW)
        assert report['coverage_trend']['baseline'] is None
        assert report['coverage_trend']['baseline_reason']

    def test_diff_is_pure_and_deterministic(self):
        history = self._history_with_prior()
        before = json.dumps(history)
        first = json.dumps(_mod.build_coverage_diff(self._current(), history))
        second = json.dumps(_mod.build_coverage_diff(self._current(), history))
        assert first == second
        assert json.dumps(history) == before


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
    """INV-10 no-silent-fail-soft: every enumeration shortfall is named, never swallowed."""

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
        # The banner itself, not 'incomplete' OR 'warning' anywhere: the
        # disjunction could be satisfied by either word appearing in unrelated
        # prose, and the complete-coverage twin below already pins its
        # absence by this exact spelling.
        assert '**WARNING — COVERAGE INCOMPLETE.**' in md
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
        # Anchored to the line that REPORTS the residue: a bare `'80' in md`
        # matches any of the hundreds of numbers in a ~200 KiB document.
        residue = next(
            line for line in md.split('\n')
            if 'carry a category outside the censused set' in line
        )
        assert '80' in residue, residue


def _both_default_projects_report(*, top_n: int = 50, page_size: int = 1000) -> dict:
    """A report covering BOTH default project ids (dark_factory + reify) --
    the only shape whose params match a fully-bare-default CLI invocation.
    """
    cells = {
        'dark_factory': {OBS: _census([{'kind': 'gotcha'}])},
        'reify': {OBS: _census([{'kind': 'gotcha'}])},
    }
    cov = {
        'dark_factory': _coverage('fused_dark_factory', 1, {OBS: (1, 1)}),
        'reify': _coverage('fused_reify', 1, {OBS: (1, 1)}),
    }
    return _mod.build_report(cells, cov, top_n=top_n, page_size=page_size)


#: The bare regen command every artifact header starts from: repo-root
#: relative, `--frozen`, and identical to what the nightly wrapper runs and
#: what OPERATIONS.md §12 documents. Single-homed here so the flag-
#: reconstruction tests below assert what they are about -- which FLAGS are
#: emitted -- while the spelling itself is pinned exactly once, by
#: TestRegenCommandCoversTheNewFlags::
#: test_the_command_is_the_invocation_the_nightly_wrapper_makes.
_BARE_REGEN = (
    'uv run --frozen --project fused-memory python '
    'fused-memory/scripts/census_memory_metadata.py'
)


class TestRegenCommand:
    """The header's regen command must always reproduce the artifact it is
    embedded in (task 3507): a bare-default invocation silently truncated
    the markdown ~4,800 rows shorter than an artifact generated with
    --top-n 400, while both runs exited 0 and reported coverage.complete.
    """

    def test_all_default_params_render_the_bare_command(self):
        md = _mod.render_markdown(_both_default_projects_report())
        assert (
            f'Regenerate with `{_BARE_REGEN}`.'
            in md
        )

    def test_non_default_top_n_is_named_in_the_regen_command(self):
        md = _mod.render_markdown(_both_default_projects_report(top_n=400))
        assert (
            f'Regenerate with `{_BARE_REGEN} --top-n 400`.' in md
        )
        # And the bare command -- which would silently under-render -- must
        # NOT be what's documented.
        assert (
            f'Regenerate with `{_BARE_REGEN}`.'
            not in md
        )

    def test_non_default_page_size_is_named_in_the_regen_command(self):
        md = _mod.render_markdown(_both_default_projects_report(page_size=250))
        assert '--page-size 250' in md

    def test_non_default_project_ids_are_named_in_the_regen_command(self):
        cells = {'reify': {OBS: _census([{'kind': 'gotcha'}])}}
        cov = {'reify': _coverage('fused_reify', 1, {OBS: (1, 1)})}
        report = _mod.build_report(cells, cov, top_n=50)
        md = _mod.render_markdown(report)
        assert '--project-id reify' in md
        # Both default project ids together never render an explicit
        # --project-id flag.
        assert (
            '--project-id'
            not in _mod.render_markdown(_both_default_projects_report())
        )

    def test_surplus_counted_residue_is_not_rendered_as_uncovered_points(self):
        # The collection holds FEWER points than the categories counted:
        # deletions or double-counting. Rendering that through the
        # uncovered-points sentence produced "-10 of 90 points carry a
        # category outside the censused set", which describes nothing and
        # sends a reader hunting for data that is not missing.
        cells = {'fused_p1': {OBS: _census([])}}
        cov = {'fused_p1': _coverage('fused_p1', 90, {OBS: (100, 100, 100)})}
        md = _mod.render_markdown(_mod.build_report(cells, cov, top_n=50))
        assert 'surplus_counted' in md
        assert 'carry a category outside the censused set' not in md
        assert 'deletion' in md.lower() or 'double-count' in md.lower()
        # Not a trace of the uncovered-points diagnosis anywhere -- including
        # the coverage table's column heading, which labels the same signed
        # residue and so mislabels a negative one exactly as the sentence did.
        assert 'uncovered' not in md.lower()

    def test_churn_only_report_renders_no_incomplete_banner(self):
        # The collection total moved under the scan but the intervals
        # intersect: nothing is missing, so none of the alarm machinery fires.
        cells = {'dark_factory': {OBS: _census([])}}
        cov = {'dark_factory': _coverage('fused_dark_factory', 101, {OBS: (100, 101, 101)})}
        md = _mod.render_markdown(_mod.build_report(cells, cov, top_n=50))
        assert '**WARNING — COVERAGE INCOMPLETE.**' not in md
        assert 'Named shortfalls' not in md
        assert 'carry a category outside the censused set' not in md
        # ...but the movement is still visible, with both intervals named.
        assert '### Corpus churn during the scan' in md
        churn = _md_section(md, '### Corpus churn during the scan')
        assert 'collection total' in churn.lower()
        # Both interval endpoints inside the CHURN section -- unanchored,
        # '100'/'101' matched the record counts rendered elsewhere.
        assert '100' in churn and '101' in churn, churn

    def test_churn_render_tolerates_a_collection_level_entry(self):
        # A 'collection_total_moved' entry has no 'category'/'expected'/
        # 'recount' keys; a loop that formats those unconditionally renders
        # a line of Nones for it.
        cells = {'dark_factory': {OBS: _census([])}}
        cov = {'dark_factory': _coverage('fused_dark_factory', 101, {OBS: (100, 101, 101)})}
        md = _mod.render_markdown(_mod.build_report(cells, cov, top_n=50, page_size=1000))
        section = md.split('### Corpus churn during the scan')[1].split('\n#')[0]
        assert 'None' not in section
        # Both the per-cell and the collection-level entry render.
        assert len([ln for ln in section.splitlines() if ln.startswith('- ')]) == 2

    def test_churn_only_report_renders_identically_twice(self):
        cells = {'dark_factory': {OBS: _census([])}}
        cov = {'dark_factory': _coverage('fused_dark_factory', 101, {OBS: (100, 101, 101)})}
        report = _mod.build_report(cells, cov, top_n=50)
        assert _mod.render_markdown(report) == _mod.render_markdown(report)


def _md_section(md: str, heading: str, stop: str = '#') -> str:
    """The body of *heading*, up to the next line starting with *stop*.

    Section-scoped assertions cannot be satisfied by a match somewhere else
    in a ~200KB document -- which is how three trend assertions came to pass
    against lists that were empty ('lost' matched its own heading;
    '`alpha`' and 'registry' matched unrelated sections). Asserts the
    heading is unique, so a slice can never silently address the wrong one.
    """
    assert md.count(heading) == 1, f'{heading!r} occurs {md.count(heading)} times'
    start = md.index(heading) + len(heading)
    body: list[str] = []
    for line in md[start:].split('\n'):
        if line.startswith(stop):
            break
        body.append(line)
    return '\n'.join(body)


def _coverage_md_report(top_n: int = 50, *, history=..., registry=True, **kwargs) -> dict:
    """A report exercising every new coverage/trend section of the markdown.

    dark_factory carries two legacy-spelled topics, a topic with exactly one
    canonical, and a DUPLICATE-canonical topic; reify carries a second
    duplicate. Two rows in every named list, so a ``--top-n 1`` render has
    something to cut.

    Defaults to an EMPTY history rather than none, modelling the real first
    nightly run: the timer always loads the history file, which on day one
    simply has no runs in it.
    """
    if history is ...:
        history = _mod.empty_coverage_history()
    payloads = {
        'dark_factory': [
            {'topic': 'alpha', 'canonical': True},
            {'topic': 'alpha'},
            {'topic': 'legacy_spelling'},
            {'topic': 'another_legacy'},
            {'topic': 'dup', 'canonical': True},
            {'topic': 'dup', 'canonical': True},
            {},
        ],
        'reify': [
            {'topic': 'beta', 'canonical': True},
            {'topic': 'beta', 'canonical': True},
        ],
    }
    cells = {pid: {OBS: _census(p)} for pid, p in payloads.items()}
    cov = {
        pid: _coverage(
            f'fused_{pid}',
            sum(c.records for c in cats.values()),
            {cat: (c.records, c.records) for cat, c in cats.items()},
        )
        for pid, cats in cells.items()
    }
    return _mod.build_report(
        cells, cov, top_n=top_n, page_size=1000,
        registry=_FakeRegistry([
            _FakeEntry('alpha', 'dark_factory'),
            _FakeEntry('gamma', 'dark_factory'),
            _FakeEntry('delta', 'dark_factory'),
            _FakeEntry('beta', 'reify'),
        ]) if registry else None,
        history=history,
        **kwargs,
    )


class TestRenderMarkdownTopicCoverage:
    """The `## Topic & canonical coverage` section.

    The markdown is the twin a human actually reads. A coverage number that
    lands only in the JSON is a number nobody is accountable to -- which is
    the failure mode the ask names outright.
    """

    def test_section_renders_the_rate_and_the_three_way_partition(self):
        md = _mod.render_markdown(_coverage_md_report())
        assert '## Topic & canonical coverage' in md
        # The RATE is the thing the raw counts cannot express.
        assert '| `dark_factory` | 7 | 6 | 85.7143% | 4 | 1 | 2 | 1 |' in md
        assert '| `reify` | 2 | 2 | 100.0% | 1 | 0 | 0 | 1 |' in md

    def test_partition_is_rendered_for_the_grand_total_too(self):
        md = _mod.render_markdown(_coverage_md_report())
        assert '| **(all)** | 9 | 8 |' in md

    def test_uniqueness_violators_are_NAMED_with_their_project(self):
        md = _mod.render_markdown(_coverage_md_report())
        assert '| `dark_factory` | `dup` | 2 |' in md
        assert '| `reify` | `beta` | 2 |' in md

    def test_enforce_state_is_rendered_inline_as_the_caveat(self):
        md = _mod.render_markdown(_coverage_md_report(canonical_uniqueness_enforced=False))
        # Without the regime beside the count a reader cannot tell "the
        # guard is warn-only, so this is backlog" from "the guard broke".
        assert '`memory_metadata.enforce`' in md
        assert 'warn' in md.lower()

    def test_an_unreadable_enforce_flag_renders_as_unknown_not_false(self):
        md = _mod.render_markdown(_coverage_md_report(canonical_uniqueness_enforced=None))
        assert 'unknown' in md.lower()
        assert '**false**' not in md.lower()

    def test_both_slug_conformance_partitions_render_distinctly_labelled(self):
        md = _mod.render_markdown(_coverage_md_report())
        # Two columns, two questions -- gate 3626's recipe items 3 and 4.
        # A reader must not mistake one for the other.
        assert 'item 3' in md
        assert 'item 4' in md
        assert '`canonical: true`' in md
        assert '| `legacy_spelling` | 1 |' in md
        assert '| `another_legacy` | 1 |' in md

    def test_registry_gauge_renders_with_its_actionable_worklist(self):
        md = _mod.render_markdown(_coverage_md_report())
        registry = _md_section(
            md, '### Registry coverage — the accountable target', stop='### ',
        )
        # registry_topics_total, inside the registry section. Unanchored,
        # '**4**' matched any bolded 4 anywhere in the document.
        assert '**4**' in registry, registry
        # The worklist: the exact set 3201's retro sweep would stamp.
        assert '| `dark_factory` | `gamma` |' in md
        assert '| `dark_factory` | `delta` |' in md

    def test_a_registry_that_did_not_load_is_disclosed_not_omitted(self):
        md = _mod.render_markdown(_coverage_md_report(registry=False))
        assert 'no topic registry supplied' in md


class TestRenderMarkdownEnforcePreconditions:
    """The flip-precondition citations, rendered as prose beneath the count.

    Leo's 2026-08-12 ruling (Option B on esc-4006-3): a reader must learn
    the number, the regime that produced it, AND what remains -- in one
    place, without reconstructing the last part from the PRD.
    """

    def test_all_three_citations_render(self):
        md = _mod.render_markdown(_coverage_md_report())
        # Default stop='#': the citation block ONLY, cut at the very next
        # heading of any level -- the tightest slice available. Unanchored,
        # a task id matched anywhere in a ~200 KiB document, so the ruling's
        # requirement was not guarded where it applies.
        block = _md_section(
            md, '#### What still blocks flipping `memory_metadata.enforce`',
        )
        # Driven off the SINGLE-HOMED constant rather than three literals, so
        # what this catches is the RENDERER dropping an entry the report still
        # claims to cite -- the regression Leo's 2026-08-12 ruling is actually
        # exposed to. WHICH three ids are cited is pinned on the JSON side by
        # test_the_three_cited_ids and is deliberately not restated here.
        cited = [entry['id'] for entry in _mod.ENFORCE_FLIP_PRECONDITIONS]
        assert len(cited) == 3, cited
        for precondition_id in cited:
            # The rendered ROW, not the bare id: the block's own intro prose
            # reads "task 3626 decides", so `'3626' in block` survived the
            # renderer dropping 3626's entry entirely. Verified by mutation.
            assert f'- **`{precondition_id}`**' in block, (precondition_id, block)

    # DELIBERATELY NOT TESTED HERE (review of 2026-08-16): the rendered
    # WORDING, the citation ORDER relative to the count/flag, and the
    # 'discharged_2026_08_04' date literal.  Three tests pinning those were
    # removed: `'re-measur' in md.lower()`, a positional
    # `md.index(...) < md.index(...)` ordering assertion, and `'task 3626' in
    # md`.  None could catch a defect in what the census MEASURES, and each
    # turned a cosmetic re-wording -- or the ordinary event of 3202 landing
    # and a status flipping -- into a suite failure.  What the ruling actually
    # requires is that the citations REACH the reader and are never truncated;
    # that is what the two surviving tests assert, keyed on stable IDs rather
    # than prose.  The live re-measurement is asserted as COMPUTED COUNTS in
    # `test_the_legacy_bucket_citation_carries_this_runs_re_measurement`,
    # which is a measurement rather than a substring.

    def test_citations_are_NEVER_subject_to_the_top_n_cut(self):
        # A truncated precondition list silently under-reports what blocks
        # the flip -- the one list where a cut changes the conclusion.
        md = _mod.render_markdown(_coverage_md_report(top_n=1))
        for cited in ('legacy_topic_spelling_remains', '3202', '3626'):
            assert cited in md


class TestRenderMarkdownCoverageTrend:
    """The `## Coverage trend` section -- and its no-baseline case."""

    @staticmethod
    def _history():
        prior = _topic_report(
            {'dark_factory': [{'topic': 'alpha', 'canonical': True}, {'topic': 'alpha'}]},
            registry=_FakeRegistry([_FakeEntry('alpha', 'dark_factory')]),
        )
        return _mod.append_coverage_run(
            _mod.empty_coverage_history(), prior, stamp='2026-08-15T05:00:00Z',
        )

    def test_first_run_prints_an_explicit_no_baseline_line(self):
        md = _mod.render_markdown(_coverage_md_report())
        assert '## Coverage trend' in md
        assert 'No baseline' in md
        assert 'first recorded run' in md

    def test_first_run_renders_no_delta_table_of_zeros(self):
        md = _mod.render_markdown(_coverage_md_report())
        assert '| before | after | delta |' not in md

    def test_baseline_stamp_and_deltas_render(self):
        md = _mod.render_markdown(_coverage_md_report(history=self._history()))
        assert '2026-08-15T05:00:00Z' in md
        assert '| `dark_factory` | `records` | 2 | 7 | +5 |' in md

    def test_an_uncomparable_column_renders_as_not_measured_not_zero(self):
        md = _mod.render_markdown(_coverage_md_report(history=self._history()))
        # reify is new_project this run: no deltas, disclosed as such.
        assert '`reify`' in md
        assert 'new_project' in md

    @staticmethod
    def _history_with_movement():
        """A prior run the current fixture actually MOVES against.

        ``_history()`` above happens to be static in every regrowth class --
        alpha sits at 2 records / 1 canonical on both sides -- so a test
        driven off it renders `_(none)_` under every heading and can only be
        satisfied by matching the headings themselves. Here alpha GREW
        (1 -> 2 records) and gamma LOST its canonical (1 -> 0), so both
        actionable classes produce a real row to pin.
        """
        prior = _topic_report(
            {'dark_factory': [
                {'topic': 'alpha', 'canonical': True},
                {'topic': 'gamma', 'canonical': True},
            ]},
            registry=_FakeRegistry([
                _FakeEntry('alpha', 'dark_factory'),
                _FakeEntry('gamma', 'dark_factory'),
            ]),
        )
        return _mod.append_coverage_run(
            _mod.empty_coverage_history(), prior, stamp='2026-08-15T05:00:00Z',
        )

    def test_regrowth_classes_render_named(self):
        # Pinned as whole ROWS under their own headings. `'lost' in
        # md.lower()` matched the heading itself and `'`alpha`' in md`
        # matched any of ~200KB of unrelated document, so both passed
        # against a fixture in which every regrowth list was empty -- the
        # green-by-construction class this suite's own
        # test_the_forbidden_git_patterns_can_actually_fire exists to catch.
        md = _mod.render_markdown(
            _coverage_md_report(history=self._history_with_movement()),
        )
        lost = _md_section(md, '#### Lost their `canonical: true`')
        assert '| `dark_factory` | `gamma` | 1 | 0 |' in lost
        grew = _md_section(md, '#### Grew in member records')
        assert '| `dark_factory` | `alpha` | 1 | 2 |' in grew

    def test_a_class_with_nothing_to_report_says_so_rather_than_vanishing(self):
        # The mirror image, and why the row assertions above are keyed to
        # their own section: an empty class must render an explicit "(none)"
        # under its heading, not disappear and leave the heading to carry a
        # substring match for a class that was never measured.
        md = _mod.render_markdown(_coverage_md_report(history=self._history()))
        assert _md_section(md, '#### Lost their `canonical: true`').strip() == '_(none)_'
        assert _md_section(md, '#### Grew in member records').strip() == '_(none)_'

    def test_regrowth_scope_is_disclosed_in_the_markdown_too(self):
        # `'registry' in md.lower()` could not fail: the document always
        # carries a '### Registry coverage' section. The claim is that the
        # NARROWING is disclosed where the signal is rendered, so assert the
        # scope note itself, inside the regrowth section, verbatim from the
        # single-homed constant.
        md = _mod.render_markdown(_coverage_md_report(history=self._history()))
        regrowth = _md_section(md, '### Registry topic regrowth', stop='#### ')
        assert _mod._REGROWTH_SCOPE_NOTE in regrowth
        assert 'Scoped to the committed topic registry' in regrowth


class TestRenderMarkdownCoverageCutsDiscloseThemselves:
    """The new named lists obey the same cut-and-disclose rule as the old."""

    def test_named_lists_are_cut_with_a_disclosure(self):
        md = _mod.render_markdown(_coverage_md_report(top_n=1))
        # The cut lands on the NEW lists too: one violator row shown, the
        # second cut -- and the cut says so and says where the rest is.
        #
        # WHICH row survives is severity-major, not project-major: the list is
        # ordered by canonical count desc, then (topic, project) asc, matching
        # _table's total order. Under a project-major order a `--top-n 1` cut
        # would keep dark_factory's 2-canonical topic and drop a hypothetical
        # 9-canonical one in reify -- a cut that changes the conclusion. Both
        # fixture rows tie at 2, so the tiebreak is the topic: `beta` < `dup`.
        assert '| `reify` | `beta` | 2 |' in md
        assert '| `dark_factory` | `dup` | 2 |' not in md
        assert '**truncated**' in md
        assert 'JSON artifact carries the full population' in md

    def test_the_violator_cut_keeps_the_worst_offender(self):
        """Severity-major ordering, stated over rows that do NOT tie.

        The tie in the fixture above makes the assertion above readable but
        not load-bearing; this one fails if the order ever reverts to
        project-major, which would silently cut the worse violation.
        """
        payloads = {
            'dark_factory': [
                {'topic': 'mild', 'canonical': True},
                {'topic': 'mild', 'canonical': True},
            ],
            'reify': [
                {'topic': 'severe', 'canonical': True},
                {'topic': 'severe', 'canonical': True},
                {'topic': 'severe', 'canonical': True},
            ],
        }
        cells = {pid: {OBS: _census(p)} for pid, p in payloads.items()}
        cov = {
            pid: _coverage(
                f'fused_{pid}',
                sum(c.records for c in cats.values()),
                {cat: (c.records, c.records) for cat, c in cats.items()},
            )
            for pid, cats in cells.items()
        }
        report = _mod.build_report(cells, cov, top_n=1, page_size=1000)
        md = _mod.render_markdown(report)
        assert '| `reify` | `severe` | 3 |' in md
        assert '| `dark_factory` | `mild` | 2 |' not in md

    def test_markdown_cut_never_drops_a_value_from_the_json(self):
        report = _coverage_md_report(top_n=1)
        md = _mod.render_markdown(report)
        cov = report['projects']['dark_factory']['topic_coverage']
        # The JSON keeps every row whatever the markdown shows.
        assert len(cov['non_conforming_topics']) == 2
        assert len(report['registry_coverage']['zero_canonical_topics']) == 2
        # ...and the cut view points at the copy that is not cut.
        assert md.count('JSON artifact carries the full population') >= 1

    def test_a_wide_top_n_renders_every_named_row(self):
        md = _mod.render_markdown(_coverage_md_report(top_n=500))
        assert '| `dark_factory` | `dup` | 2 |' in md
        assert '| `reify` | `beta` | 2 |' in md
        assert '| `legacy_spelling` | 1 |' in md
        assert '| `another_legacy` | 1 |' in md


class TestRenderMarkdownDeterminism:
    def test_same_report_renders_identical_string_twice(self):
        report = _rich_report()
        assert _mod.render_markdown(report) == _mod.render_markdown(report)

    def test_returns_a_string(self):
        assert isinstance(_mod.render_markdown(_rich_report()), str)


# ===========================================================================
# Tests: the census's scroll path
#
# The paging loop itself no longer lives here. Task 3225 moved it into
# ``Mem0Backend.scroll_collection_pages`` / ``scroll_all_by_metadata``, and
# with it the whole contract this section used to assert: multi-page
# ordering, the exact offset sequence, stop-on-None, page-budget exhaustion
# and the per-page read bound. Those tests moved to
# ``tests/test_mem0_client.py::TestMem0BackendScrollCollectionPages`` /
# ``::TestMem0BackendScrollAllByMetadata`` — coverage follows the code
# rather than being asserted in two places (INV-5). What remains here is
# the census's own concern: that it DRIVES that backend generator, with the
# right scope and filters, and folds what it yields.
# ===========================================================================

# ===========================================================================
# Tests: census_project
# ===========================================================================

ALL_SIX = [c.value for c in _mod.CENSUS_CATEGORIES]


def _backend(
    payloads_by_category: dict[str, list[dict]],
    *,
    counts_by_category: dict[str, int] | None = None,
    collection_points: int | None = None,
    call_log: list | None = None,
    scroll_calls: list | None = None,
) -> AsyncMock:
    """Mem0Backend stand-in: count_by_metadata / count / scroll_all_by_metadata.

    ``scroll_all_by_metadata`` is an async generator dispatching on the
    ``category`` filter and yielding that category's payloads as normalised
    records.  The offset/next_offset walk this stand-in used to simulate now
    lives inside the backend (task 3225), so the census sees exactly ONE
    generator per category no matter how many Qdrant pages back it.

    *scroll_calls* collects ``(scope, filters, kwargs)`` per scroll so tests
    can assert what the census forwards.
    """
    log = call_log if call_log is not None else []
    scrolls = scroll_calls if scroll_calls is not None else []
    counts = counts_by_category or {c: len(p) for c, p in payloads_by_category.items()}

    async def _scroll_all_by_metadata(scope, filters, **kwargs):
        category = filters['category']
        log.append(('scroll', category))
        scrolls.append((scope, dict(filters), dict(kwargs)))
        for index, payload in enumerate(payloads_by_category.get(category, [])):
            yield {
                'id': f'{category}-{index}',
                'created_at': payload.get('created_at'),
                'metadata': dict(payload),
            }

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
    backend.scroll_all_by_metadata = _scroll_all_by_metadata
    backend._scroll_calls = scrolls
    return backend


class TestCensusUsesTheBackendPager:
    """The census DRIVES ``Mem0Backend.scroll_all_by_metadata``; it owns no loop.

    Task 3225: the script used to re-implement the payload-filter
    construction and the offset/next_offset walk against a raw async Qdrant
    client, because the backend structurally could not page.  It can now, so
    the duplicate is gone.  That matters beyond tidiness — the census
    reconciles its SCROLL against ``count_by_metadata``'s COUNT to decide
    ``coverage.complete``, and two independent filter constructions could
    drift into comparing two different point sets while still reporting
    complete coverage.
    """

    @pytest.mark.asyncio
    async def test_calls_the_backend_generator_once_per_category(self):
        scrolls: list = []
        backend = _backend({OBS: [{}], PROC: [{}]}, scroll_calls=scrolls)
        await _mod.census_project(backend, 'dark_factory', [OBS, PROC])
        assert [filters for _scope, filters, _kw in scrolls] == [
            {'category': OBS}, {'category': PROC},
        ]

    @pytest.mark.asyncio
    async def test_forwards_page_size_and_max_pages(self):
        scrolls: list = []
        backend = _backend({OBS: [{}]}, scroll_calls=scrolls)
        await _mod.census_project(backend, 'dark_factory', [OBS], page_size=250, max_pages=7)
        _scope, _filters, kwargs = scrolls[0]
        assert kwargs['page_size'] == 250
        assert kwargs['max_pages'] == 7

    @pytest.mark.asyncio
    async def test_does_not_replumb_a_read_timeout(self):
        """The per-page read bound is the backend's own concern now.

        Contract moved to test_mem0_client.py::
        TestMem0BackendScrollCollectionPages::{test_a_hung_page_request_
        raises_instead_of_hanging, test_timeout_applies_per_page_not_per_scan}.
        """
        scrolls: list = []
        backend = _backend({OBS: [{}]}, scroll_calls=scrolls)
        backend._read_timeout = 7.5
        await _mod.census_project(backend, 'dark_factory', [OBS])
        assert 'read_timeout' not in scrolls[0][2]

    @pytest.mark.asyncio
    async def test_scroll_and_count_are_given_the_same_scope(self):
        """Otherwise the reconciliation compares two different collections."""
        scrolls: list = []
        backend = _backend({OBS: [{}]}, scroll_calls=scrolls)
        await _mod.census_project(backend, 'dark_factory', [OBS])
        scroll_scope = scrolls[0][0]
        count_scopes = [call.args[0] for call in backend.count_by_metadata.await_args_list]
        assert count_scopes, 'count_by_metadata must be called with a positional scope'
        for count_scope in count_scopes:
            assert count_scope is scroll_scope

    @pytest.mark.asyncio
    async def test_folds_record_metadata_into_the_category_census(self):
        """The census reads ``record['metadata']`` — an empty one still counts."""
        backend = _backend({OBS: [{'kind': 'a'}, {}]}, counts_by_category={OBS: 2})
        cells, coverage = await _mod.census_project(backend, 'dark_factory', [OBS])
        assert cells[OBS].records == 2, 'a record with empty metadata is still a record'
        assert cells[OBS].kind_counts['a'] == 1
        assert cells[OBS].kind_missing == 1
        assert coverage['categories'][OBS]['scrolled'] == 2

    @pytest.mark.asyncio
    async def test_census_does_not_open_its_own_qdrant_client(self):
        """No ``backend._get_async_qdrant()`` for scrolling: it goes through the API."""
        backend = _backend({OBS: [{}]})
        await _mod.census_project(backend, 'dark_factory', [OBS])
        assert backend._get_async_qdrant.await_count == 0


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
    async def test_every_streamed_record_is_folded_however_many_pages_backed_it(self):
        """The census consumes the generator to exhaustion, not just its head.

        How many Qdrant pages back that stream is the backend's business now
        (test_mem0_client.py::TestMem0BackendScrollAllByMetadata::
        test_records_stream_across_page_boundaries_in_order); from here it is
        one generator, and every record it yields must land in the census.
        """
        scrolls: list = []
        backend = _backend(
            {OBS: [{'kind': 'a'}, {'kind': 'b'}, {'kind': 'c'}]},
            counts_by_category={OBS: 3},
            scroll_calls=scrolls,
        )
        cells, coverage = await _mod.census_project(backend, 'dark_factory', [OBS], page_size=1)
        assert cells[OBS].records == 3
        assert coverage['categories'][OBS]['scrolled'] == 3
        assert len(scrolls) == 1, 'one generator per category — the paging is inside it'
        assert scrolls[0][2]['page_size'] == 1


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

    # The read-timeout plumbing tests that stood here are gone with the
    # plumbing: the per-page bound is applied inside
    # Mem0Backend.scroll_collection_pages, which reads its own
    # ``_read_timeout``, so the census neither derives nor forwards one.
    # Contract moved to test_mem0_client.py::
    # TestMem0BackendScrollCollectionPages::{test_a_hung_page_request_raises_
    # instead_of_hanging, test_timeout_applies_per_page_not_per_scan}; that
    # the census does not re-plumb it is asserted by
    # TestCensusUsesTheBackendPager::test_does_not_replumb_a_read_timeout.

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
        'registry': _mod.DEFAULT_REGISTRY_PATH,
        'history_out': str(tmp_path / 'coverage-history.json'),
        'no_history': False,
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


class TestCensusScanIncompleteIsTheBackendException:
    """CensusScanIncomplete must BE the backend's exception, not merely
    resemble it.

    _run's ``except CensusScanIncomplete`` has to name exactly what
    ``backend.scroll_all_by_metadata`` raises. The name is an ALIAS of
    ``ScrollPageBudgetExhausted`` -- but every existing test that exercises
    the handler raises ``_mod.CensusScanIncomplete`` itself, so the identity
    is satisfied tautologically: re-declaring it as
    ``class CensusScanIncomplete(RuntimeError)`` would keep all of them green
    while the real exception escaped as an uncaught traceback instead of a
    clean exit 1. These two guards are what close that gap. Both PASS against
    today's correct code; their value is regression pressure.
    """

    def test_is_the_backend_exception_itself_not_a_lookalike(self):
        """The one-line identity the handler depends on.

        A subclass would not catch its parent; a sibling re-declaration would
        not catch it either. Only identity works.
        """
        from fused_memory.backends.mem0_client import ScrollPageBudgetExhausted

        assert _mod.CensusScanIncomplete is ScrollPageBudgetExhausted

    @pytest.mark.asyncio
    async def test_the_real_backend_exception_exits_nonzero_through_census_project(
        self, tmp_path, monkeypatch,
    ):
        """End-to-end: the REAL backend exception, raised from a real scroll,
        travelling through the REAL census_project into _run's handler.

        Deliberately imports ScrollPageBudgetExhausted from the backend
        rather than referencing _mod.CensusScanIncomplete -- naming the alias
        here would rebuild the same tautology this class exists to break.
        census_project is NOT stubbed, so the exception crosses the same code
        path it crosses in production.
        """
        from fused_memory.backends.mem0_client import ScrollPageBudgetExhausted

        async def _scroll_all_by_metadata(scope, filters, **kwargs):
            raise ScrollPageBudgetExhausted('page budget exhausted')
            yield  # pragma: no cover - makes this an async generator

        backend = AsyncMock()
        backend.config = MagicMock()
        backend.config.mem0.collection_prefix = 'fused'
        backend.count_by_metadata = AsyncMock(return_value=1)
        backend.count = AsyncMock(return_value=1)
        backend.scroll_all_by_metadata = _scroll_all_by_metadata
        monkeypatch.setattr(_mod, '_build_backend', lambda cfg: backend)

        exit_code = await _mod._run(_args(tmp_path))

        assert exit_code != 0, (
            'a truncated scan must abort with a non-zero exit, not surface as '
            'an uncaught traceback'
        )
        # the finally: close() still runs even on the abort path
        backend.close.assert_awaited_once()


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


class TestCliHistoryFlags:
    """`--history-out` / `--no-history` / `--registry` (task 4006).

    The trend file is committed and written by a nightly timer, so an
    operator's ad-hoc run must be able to MEASURE without polluting it --
    otherwise the only safe way to look at coverage is not to look.
    """

    def test_parser_defaults_for_the_new_flags(self):
        args = _mod._build_parser().parse_args([])
        assert args.history_out == _mod.DEFAULT_HISTORY_OUT
        assert args.no_history is False
        assert args.registry == _mod.DEFAULT_REGISTRY_PATH

    def test_new_flags_parse(self):
        args = _mod._build_parser().parse_args([
            '--history-out', '/tmp/h.json', '--no-history', '--registry', '/tmp/r.json',
        ])
        assert args.history_out == '/tmp/h.json'
        assert args.no_history is True
        assert args.registry == '/tmp/r.json'

    @pytest.mark.asyncio
    async def test_run_appends_a_run_to_the_history(self, tmp_path, monkeypatch):
        monkeypatch.setattr(_mod, 'census_project', _census_project_stub())
        monkeypatch.setattr(_mod, '_build_backend', lambda cfg: AsyncMock())
        history_out = str(tmp_path / 'history.json')
        rc = await _mod._run(_args(tmp_path, history_out=history_out))
        assert rc == 0
        history = _mod.load_coverage_history(history_out)
        assert len(history['runs']) == 1
        assert history['runs'][0]['stamp']
        assert 'dark_factory' in history['runs'][0]['projects']

    @pytest.mark.asyncio
    async def test_consecutive_runs_append_rather_than_overwrite(self, tmp_path, monkeypatch):
        monkeypatch.setattr(_mod, 'census_project', _census_project_stub())
        monkeypatch.setattr(_mod, '_build_backend', lambda cfg: AsyncMock())
        history_out = str(tmp_path / 'history.json')
        await _mod._run(_args(tmp_path, history_out=history_out))
        await _mod._run(_args(tmp_path, history_out=history_out))
        assert len(_mod.load_coverage_history(history_out)['runs']) == 2

    @pytest.mark.asyncio
    async def test_no_history_leaves_the_file_byte_unchanged(self, tmp_path, monkeypatch):
        monkeypatch.setattr(_mod, 'census_project', _census_project_stub())
        monkeypatch.setattr(_mod, '_build_backend', lambda cfg: AsyncMock())
        history_out = tmp_path / 'history.json'
        await _mod._run(_args(tmp_path, history_out=str(history_out)))
        before = history_out.read_bytes()
        rc = await _mod._run(_args(tmp_path, history_out=str(history_out), no_history=True))
        assert rc == 0
        assert history_out.read_bytes() == before

    @pytest.mark.asyncio
    async def test_no_history_still_REPORTS_the_trend_it_did_not_append(
        self, tmp_path, monkeypatch,
    ):
        # Measure without polluting: the ad-hoc run still reads the history
        # and renders the diff, it just does not extend it.
        monkeypatch.setattr(_mod, 'census_project', _census_project_stub())
        monkeypatch.setattr(_mod, '_build_backend', lambda cfg: AsyncMock())
        history_out = str(tmp_path / 'history.json')
        await _mod._run(_args(tmp_path, history_out=history_out))
        await _mod._run(_args(tmp_path, history_out=history_out, no_history=True))
        written = json.loads((tmp_path / 'census.json').read_text())

        # The baseline must be the PRIOR RUN's stamp, not merely non-None:
        # the behaviour under test is that the ad-hoc run trends against the
        # recorded series it declined to extend. `is not None` also passes on
        # a wrong stamp or an empty-ish placeholder, which is the one way
        # this could plausibly break.
        recorded = _mod.load_coverage_history(history_out)['runs']
        assert len(recorded) == 1, 'the --no-history run must not have appended'
        assert written['coverage_trend']['baseline'] == recorded[-1]['stamp']

    @pytest.mark.asyncio
    async def test_a_malformed_history_fails_the_run_loudly(self, tmp_path, monkeypatch):
        # Losing the trend must be loud: a run that silently restarted the
        # history from empty would report "first recorded run" forever.
        monkeypatch.setattr(_mod, 'census_project', _census_project_stub())
        monkeypatch.setattr(_mod, '_build_backend', lambda cfg: AsyncMock())
        history_out = tmp_path / 'history.json'
        history_out.write_text('{not json', encoding='utf-8')
        rc = await _mod._run(_args(tmp_path, history_out=str(history_out)))
        assert rc == 1
        assert history_out.read_text(encoding='utf-8') == '{not json'

    @pytest.mark.asyncio
    async def test_a_malformed_history_fails_BEFORE_the_scroll(self, tmp_path, monkeypatch):
        # The fault costs milliseconds to detect and the scroll costs minutes
        # over ~54k live payloads. Validating after it would discard a
        # completed measurement over an unrelated file -- and the nightly job
        # would then emit nothing at all, every night, until someone noticed.
        scrolled: list[str] = []

        async def _stub(backend, project_id, categories=None, page_size=1000, max_pages=200):
            scrolled.append(project_id)
            return {OBS: _census([{'kind': 'cycle_summary'}])}, _coverage(
                'fused_x', 1, {OBS: (1, 1)},
            )

        monkeypatch.setattr(_mod, 'census_project', _stub)
        monkeypatch.setattr(_mod, '_build_backend', lambda cfg: AsyncMock())
        history_out = tmp_path / 'history.json'
        history_out.write_text('{not json', encoding='utf-8')
        rc = await _mod._run(_args(tmp_path, history_out=str(history_out)))
        assert rc == 1
        assert scrolled == []
        # And the artifacts do not half-land: nothing was measured, so
        # nothing claims to have been.
        assert not (tmp_path / 'census.json').exists()
        assert not (tmp_path / 'census.md').exists()

    @pytest.mark.asyncio
    async def test_an_incomplete_census_still_lands_its_artifacts_and_exits_1(
        self, tmp_path, monkeypatch,
    ):
        # The pre-existing exit-code contract survives the new wiring.
        monkeypatch.setattr(_mod, 'census_project', _census_project_stub(complete=False))
        monkeypatch.setattr(_mod, '_build_backend', lambda cfg: AsyncMock())
        rc = await _mod._run(_args(tmp_path, history_out=str(tmp_path / 'h.json')))
        assert rc == 1
        assert (tmp_path / 'census.json').exists()


class TestRegenCommandCoversTheNewFlags:
    """Task 3507's drift, re-armed for 4006's flags: a documented regen
    command that no longer reproduces the artifact is worse than none."""

    def test_default_new_flags_keep_the_bare_command(self):
        md = _mod.render_markdown(_both_default_projects_report())
        assert (
            f'Regenerate with `{_BARE_REGEN}`.'
            in md
        )

    def test_every_path_in_the_command_resolves_from_ONE_working_directory(self):
        """The script path and the flag paths must share a cwd.

        The emitted prefix named the script relative to ``fused-memory/``,
        while ``--registry`` / ``--history-out`` carry the values recorded by
        ``_repo_relative`` -- relative to the REPO ROOT. A run that used
        either non-default path therefore printed a command whose script and
        whose flags could not both resolve from any single directory, which
        is precisely the drift ``_repo_relative``'s own docstring cites task
        3507 for: "a documented regen command that no longer reproduces the
        artifact is worse than none".

        Asserted by RESOLVING every path in the command against one base,
        not by pinning its spelling -- so it stays true through a rename and
        fails the moment the two bases diverge again.
        """
        # Two real, non-default, repo-root-relative paths, so both flags are
        # emitted and both can be resolved.
        params = {
            'projects': ['dark_factory', 'reify'],
            'registry': 'fused-memory/tests/fixtures/census-grandfather-oracle.json',
            'history_out': 'plans/memory-metadata-census-report.json',
        }
        command = _mod._regen_command(params)
        tokens = command.split()
        repo_root = Path(_mod._REPO_ROOT)

        script = next(t for t in tokens if t.endswith('census_memory_metadata.py'))
        paths = {'script': script}
        for flag in ('--registry', '--history-out'):
            assert flag in tokens, command
            paths[flag] = tokens[tokens.index(flag) + 1]

        unresolvable = {
            name: value for name, value in paths.items()
            if not (repo_root / value).is_file()
        }
        assert not unresolvable, (
            f'{command!r} mixes path bases; these do not resolve from the '
            f'repo root: {unresolvable}'
        )

    def test_the_command_is_the_invocation_the_nightly_wrapper_makes(self):
        """One spelling, shared by the wrapper, OPERATIONS.md and this header.

        The wrapper runs `uv run --frozen --project "$FM" python
        "$FM/scripts/census_memory_metadata.py"`; OPERATIONS.md §12 documents
        that same form and claims it is "what the report's own regen line
        reproduces". That claim was false while this function emitted a
        different, cwd-incompatible spelling -- so a reader following the
        artifact header and a reader following the runbook ran two different
        commands.
        """
        command = _mod._regen_command({'projects': ['dark_factory', 'reify']})
        assert command == (
            'uv run --frozen --project fused-memory python '
            'fused-memory/scripts/census_memory_metadata.py'
        )
        # --frozen matters on its own: an unpinned resolve could regenerate a
        # committed artifact under a different dependency set than the run
        # that produced it.
        assert '--frozen' in command

    def test_non_default_registry_is_named(self):
        params = {'projects': ['dark_factory', 'reify'], 'registry': '/tmp/r.json'}
        assert '--registry /tmp/r.json' in _mod._regen_command(params)

    def test_non_default_history_out_is_named(self):
        params = {'projects': ['dark_factory', 'reify'], 'history_out': '/tmp/h.json'}
        assert '--history-out /tmp/h.json' in _mod._regen_command(params)

    def test_no_history_is_named(self):
        params = {'projects': ['dark_factory', 'reify'], 'no_history': True}
        assert '--no-history' in _mod._regen_command(params)

    def test_the_defaults_are_never_restated_in_the_command(self):
        params = {
            'projects': ['dark_factory', 'reify'],
            'registry': _mod.DEFAULT_REGISTRY_PATH,
            'history_out': _mod.DEFAULT_HISTORY_OUT,
            'no_history': False,
        }
        assert _mod._regen_command(params) == _BARE_REGEN

    @pytest.mark.asyncio
    async def test_run_records_the_new_flags_in_params(self, tmp_path, monkeypatch):
        monkeypatch.setattr(_mod, 'census_project', _census_project_stub())
        monkeypatch.setattr(_mod, '_build_backend', lambda cfg: AsyncMock())
        await _mod._run(_args(
            tmp_path, history_out='/tmp/h.json', no_history=True, registry='/tmp/r.json',
        ))
        written = json.loads((tmp_path / 'census.json').read_text())
        # Out-of-tree paths stay absolute: they are already
        # checkout-independent, and shortening them is not _repo_relative's job.
        assert written['params']['history_out'] == '/tmp/h.json'
        assert written['params']['no_history'] is True
        assert written['params']['registry'] == '/tmp/r.json'


class TestCommittedParamsAreCheckoutIndependent:
    """The artifact is COMMITTED, so a param that names the checkout which
    happened to generate it churns the diff, points a reader at a worktree
    that will be reclaimed, and makes _regen_command print a path that does
    not exist where the artifact is read (task 3507, re-armed)."""

    def test_an_in_repo_path_is_recorded_relative_to_the_repo_root(self):
        assert _mod._repo_relative(_mod.DEFAULT_REGISTRY_PATH) == (
            'fused-memory/tests/fixtures/memory_eval_topic_registry.json'
        )
        assert _mod._repo_relative(_mod.DEFAULT_HISTORY_OUT) == (
            'plans/memory-metadata-coverage-history.json'
        )

    def test_an_out_of_tree_path_stays_absolute(self):
        assert _mod._repo_relative('/tmp/elsewhere/r.json') == '/tmp/elsewhere/r.json'

    def test_a_recorded_relative_value_is_re_anchored_at_the_repo_root_not_the_cwd(self):
        # The reader's CWD is not the repo root (this suite runs from
        # fused-memory/, the wrapper from the repo root). Anchoring a
        # recorded value at the CWD would make an artifact compare unequal to
        # its own default depending on where it was read from.
        assert _mod._resolved_repo_path(
            'plans/memory-metadata-coverage-history.json',
        ) == _mod._resolved_repo_path(_mod.DEFAULT_HISTORY_OUT)
        assert _mod._resolved_repo_path('plans/x.json').startswith(str(_mod._REPO_ROOT))

    @pytest.mark.asyncio
    async def test_run_records_the_default_paths_repo_relative(self, tmp_path, monkeypatch):
        monkeypatch.setattr(_mod, 'census_project', _census_project_stub())
        monkeypatch.setattr(_mod, '_build_backend', lambda cfg: AsyncMock())
        await _mod._run(_args(
            tmp_path, registry=_mod.DEFAULT_REGISTRY_PATH,
            history_out=_mod.DEFAULT_HISTORY_OUT, no_history=True,
        ))
        params = json.loads((tmp_path / 'census.json').read_text())['params']
        assert params['registry'] == (
            'fused-memory/tests/fixtures/memory_eval_topic_registry.json'
        )
        assert params['history_out'] == 'plans/memory-metadata-coverage-history.json'
        # The load-bearing half: no absolute prefix, so no worktree lane can
        # be baked into a committed artifact.
        assert not params['registry'].startswith('/')
        assert '.worktrees' not in params['history_out']

    def test_a_relative_default_still_reproduces_the_bare_regen_command(self):
        # An artifact generated in one checkout and read in another must not
        # emit a --registry flag naming a path that no longer exists.
        params = {
            'projects': ['dark_factory', 'reify'],
            'registry': 'fused-memory/tests/fixtures/memory_eval_topic_registry.json',
            'history_out': 'plans/memory-metadata-coverage-history.json',
            'no_history': False,
        }
        assert _mod._regen_command(params) == _BARE_REGEN

    def test_a_pre_4006_absolute_default_is_normalised_too(self):
        # Artifacts already committed carry the absolute spelling; reading
        # them must not manufacture a spurious flag either.
        params = {
            'projects': ['dark_factory', 'reify'],
            'registry': _mod.DEFAULT_REGISTRY_PATH,
            'history_out': _mod.DEFAULT_HISTORY_OUT,
        }
        assert _mod._regen_command(params) == _BARE_REGEN


class TestTheDisclosedRegimeIsTheOneTheRunUsed:
    @pytest.mark.asyncio
    async def test_the_enforce_flag_is_read_from_the_censuss_own_config(
        self, tmp_path, monkeypatch,
    ):
        # Two independently-constructed FusedMemoryConfig()s can resolve
        # different CONFIG_PATHs, so the report could disclose an `enforce`
        # regime belonging to a config the census never operated under.
        seen: dict[str, object] = {}

        def _backend_of(cfg):
            seen['backend'] = cfg
            return AsyncMock()

        def _enforce_of(config=None):
            seen['enforce'] = config
            return True

        monkeypatch.setattr(_mod, 'census_project', _census_project_stub())
        monkeypatch.setattr(_mod, '_build_backend', _backend_of)
        monkeypatch.setattr(_mod, 'read_canonical_uniqueness_enforced', _enforce_of)
        await _mod._run(_args(tmp_path))
        assert seen['enforce'] is seen['backend']
        assert seen['enforce'] is not None

    def test_an_injected_config_is_read_without_constructing_a_second(self):
        class _Cfg:
            memory_metadata = types.SimpleNamespace(enforce=True)

        assert _mod.read_canonical_uniqueness_enforced(_Cfg()) is True

    def test_an_unreadable_injected_config_still_degrades_to_null(self):
        class _Broken:
            @property
            def memory_metadata(self):
                raise RuntimeError('config is a wreck')

        # Never raises: the measurement does not depend on the config, and
        # losing the whole census to a config fault is strictly worse than an
        # undisclosed flag.
        assert _mod.read_canonical_uniqueness_enforced(_Broken()) is None
