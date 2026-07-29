"""Tests for scripts/census_memory_metadata.py.

Loaded via importlib so the script (not on PYTHONPATH) can be tested without
sys.path pollution -- mirrors the pattern in test_clear_malformed_empty_memory.py
and test_consolidate_namespace_families.py.

No live services: the Qdrant client and Mem0Backend are AsyncMock/MagicMock
stand-ins throughout.
"""
from __future__ import annotations

import importlib.util
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
