"""Tests for scripts/investigate_cross_graph_duplication.py.

Loaded via importlib so the script (not on PYTHONPATH) can be tested
without sys.path pollution -- mirrors the pattern in
test_purge_knowlive_namespace.py.
"""
from __future__ import annotations

import importlib.util
import sys
import types
from pathlib import Path

import pytest

SCRIPT_PATH = Path(__file__).parent.parent / 'scripts' / 'investigate_cross_graph_duplication.py'


def _load_module() -> types.ModuleType:
    """Load investigate_cross_graph_duplication.py from its file path.

    The module is registered in sys.modules under its name so that
    reflection-based decorators work correctly.
    """
    mod_name = 'investigate_cross_graph_duplication'
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
# Tests: is_path_shaped_name
# ===========================================================================

class TestIsPathShapedName:
    """Tests for is_path_shaped_name(name) -> bool."""

    @pytest.mark.parametrize('name', [
        '/home/leo/src/dark-factory',
        '-home-leo-src-dark-factory',
        '-abs-path-proj',
    ])
    def test_path_shaped_names_are_true(self, name):
        """Filesystem-path-shaped graph names are detected as True."""
        assert _mod.is_path_shaped_name(name) is True

    @pytest.mark.parametrize('name', [
        'dark_factory',
        'dark-factory',
        'know_live',
        'know-live',
        'knowlive',
        'reify',
    ])
    def test_clean_project_keys_are_false(self, name):
        """Clean project keys (no path shape) are detected as False."""
        assert _mod.is_path_shaped_name(name) is False


# ===========================================================================
# Tests: detect_collision_groups
# ===========================================================================

class TestDetectCollisionGroups:
    """Tests for detect_collision_groups(graph_names) -> dict."""

    def test_two_variants_collide_into_one_group(self):
        """dark_factory/dark-factory collide into one canonical group; the
        lone 'reify' does not, and there are no path leaks."""
        result = _mod.detect_collision_groups(['dark_factory', 'dark-factory', 'reify'])

        assert result['collisions'] == [
            {'canonical': 'dark_factory', 'variants': ['dark-factory', 'dark_factory'], 'count': 2},
        ]
        assert result['suspected_path_leaks'] == []

    def test_know_live_variants_collide(self):
        """know_live/know-live collide the same way as dark_factory/dark-factory."""
        result = _mod.detect_collision_groups(['know_live', 'know-live'])

        assert result['collisions'] == [
            {'canonical': 'know_live', 'variants': ['know-live', 'know_live'], 'count': 2},
        ]

    def test_lone_name_has_no_collision(self):
        """A single, unambiguous graph name yields no collision and no path leak."""
        result = _mod.detect_collision_groups(['reify'])

        assert result['collisions'] == []
        assert result['suspected_path_leaks'] == []

    def test_path_shaped_name_is_flagged_not_merged(self):
        """The mangled path name is reported under suspected_path_leaks and is
        NOT folded into the dark_factory collision group's variants."""
        result = _mod.detect_collision_groups(
            ['dark_factory', 'dark-factory', '-home-leo-src-dark-factory'],
        )

        assert result['suspected_path_leaks'] == ['-home-leo-src-dark-factory']
        collision = next(c for c in result['collisions'] if c['canonical'] == 'dark_factory')
        assert '-home-leo-src-dark-factory' not in collision['variants']
        assert collision['count'] == 2

    def test_legacy_knowlive_not_merged_into_know_live(self):
        """Legacy no-separator 'knowlive' stays distinct from the know_live
        collision group (task 515's re-key was cancelled; not our call to
        reverse here)."""
        result = _mod.detect_collision_groups(['know_live', 'know-live', 'knowlive'])

        collision = next(c for c in result['collisions'] if c['canonical'] == 'know_live')
        assert 'knowlive' not in collision['variants']
        assert collision['count'] == 2
        assert result['suspected_path_leaks'] == []

    def test_full_task_scenario_matches_expected_shape(self):
        """The real GRAPH.LIST from task 2116: two collision families plus one
        path leak plus two untouched clean singletons (reify, knowlive)."""
        graph_names = [
            'reify', 'dark_factory', 'dark-factory',
            '-home-leo-src-dark-factory', 'know_live', 'know-live', 'knowlive',
        ]

        result = _mod.detect_collision_groups(graph_names)

        assert result['collisions'] == [
            {'canonical': 'dark_factory', 'variants': ['dark-factory', 'dark_factory'], 'count': 2},
            {'canonical': 'know_live', 'variants': ['know-live', 'know_live'], 'count': 2},
        ]
        assert result['suspected_path_leaks'] == ['-home-leo-src-dark-factory']

    def test_deterministic_ordering(self):
        """Output list ordering does not depend on input ordering."""
        names = ['reify', 'dark-factory', 'dark_factory', 'know-live', 'know_live']

        r1 = _mod.detect_collision_groups(names)
        r2 = _mod.detect_collision_groups(list(reversed(names)))

        assert r1 == r2
