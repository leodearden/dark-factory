"""Tests for :mod:`fused_memory.canonical_marker` — the ONE strict reading
of the ``canonical`` marker (task 6215).

Every module that asks "does this record claim canonical?" calls the same
predicate.  These tests pin the predicate's verdicts, its identity through
the registry re-export, and that the leaf stays import-light enough for the
stdlib-only ``reconciliation/consolidation_auto.py`` to call it.
"""

import subprocess
import sys
import types

import pytest

from fused_memory.canonical_marker import is_canonical

_CANONICAL_CASES: list[tuple[object, bool, str]] = [
    ({'canonical': True}, True, 'the one canonical shape'),
    ({'canonical': True, 'topic': 'a-topic'}, True, 'other keys do not matter'),
    (types.MappingProxyType({'canonical': True}), True, 'any Mapping, not only dict'),
    ({'canonical': False}, False, 'explicit False'),
    ({'canonical': 1}, False, 'int 1 == True, but is not True'),
    ({'canonical': 1.0}, False, 'float 1.0 == True, but is not True'),
    ({'canonical': 'true'}, False, 'a string is not the marker'),
    ({'canonical': 'True'}, False, 'nor is its capitalised spelling'),
    ({'canonical': None}, False, 'null value'),
    ({'canonical': [True]}, False, 'a truthy container'),
    ({}, False, 'key absent'),
    (None, False, 'no metadata at all'),
    (True, False, 'a bare bool is not a metadata mapping'),
    ('canonical', False, 'a string is not a metadata mapping'),
    ([('canonical', True)], False, 'pairs are not a Mapping'),
    (1, False, 'an int is not a metadata mapping'),
    (object(), False, 'an arbitrary object'),
]


class TestCanonicalMarkerPredicate:
    @pytest.mark.parametrize(('meta', 'expected', 'why'), _CANONICAL_CASES)
    def test_predicate_verdicts(self, meta, expected, why):
        assert is_canonical(meta) is expected, f'{meta!r}: {why}'


class TestOnePredicateOneObject:
    """INV-5: the registry re-exports the leaf's predicate, not a copy."""

    def test_memory_metadata_re_exports_the_same_object(self):
        import fused_memory.canonical_marker as cm
        import fused_memory.memory_metadata as mm

        assert mm.is_canonical is cm.is_canonical

    def test_registry_advertises_the_name(self):
        import fused_memory.memory_metadata as mm

        assert 'is_canonical' in mm.__all__

    def test_module_is_import_light(self):
        """Probed in a FRESH interpreter: this process already loaded the
        heavy modules through the registry import above, so an in-process
        ``sys.modules`` check would pass vacuously.
        """
        probe = (
            'import sys; import fused_memory.canonical_marker as cm; '
            "heavy = ('mem0', 'pydantic', 'yaml', 'fused_memory.config.schema', "
            "'fused_memory.memory_metadata'); "
            'loaded = [m for m in heavy if m in sys.modules]; '
            'assert not loaded, loaded; '
            "assert cm.is_canonical({'canonical': True}) is True"
        )
        result = subprocess.run(
            [sys.executable, '-c', probe], capture_output=True, text=True, timeout=120
        )
        assert result.returncode == 0, result.stderr
