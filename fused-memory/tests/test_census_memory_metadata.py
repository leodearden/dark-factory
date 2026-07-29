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
