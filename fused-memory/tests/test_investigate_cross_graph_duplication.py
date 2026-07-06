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
