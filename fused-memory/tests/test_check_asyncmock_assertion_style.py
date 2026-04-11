"""Tests for check_asyncmock_assertion_style.py lint checker.

Tests for the AST-based lint check that flags assert_not_called() when the same function
also contains assert_not_awaited(). See task 673 (lint guard replacing task-571 meta-test).
"""
from __future__ import annotations

import importlib.util
import types
from pathlib import Path

# Load the checker script via importlib to avoid sys.path pollution.
# fused-memory/scripts/ is not on PYTHONPATH per pyproject.toml (pythonpath=['src']).
SCRIPT_PATH = Path(__file__).parent.parent / 'scripts' / 'check_asyncmock_assertion_style.py'


def _load_checker() -> types.ModuleType:
    """Load the checker module from its script path."""
    spec = importlib.util.spec_from_file_location('check_asyncmock_assertion_style', SCRIPT_PATH)
    if spec is None or spec.loader is None:
        raise ImportError(f'Cannot load {SCRIPT_PATH}')
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)  # type: ignore[union-attr]
    return module


_checker = _load_checker()
find_violations = _checker.find_violations


class TestFindViolationsBasicDetection:
    """Core detection: flag assert_not_called in functions that also use assert_not_awaited."""

    def test_flags_assert_not_called_when_same_function_has_assert_not_awaited(self):
        """Single assert_not_called alongside assert_not_awaited in the same function."""
        source = '''\
class TestMyClass:
    async def test_mixed_assertions(self):
        backend = object()
        backend.foo.assert_not_called()
        backend.bar.assert_not_awaited()
'''
        violations = find_violations(source, 'test_example.py')
        assert len(violations) == 1
        v = violations[0]
        assert v.filename == 'test_example.py'
        assert v.lineno == 4  # line of assert_not_called
        assert 'assert_not_awaited' in v.message


class TestFindViolationsNegativeCases:
    """Precision tests: no violations when both styles do NOT coexist in the same function."""

    def test_ignores_assert_not_called_when_function_has_no_assert_not_awaited(self):
        """Function with only assert_not_called — no violation."""
        source = '''\
def test_only_not_called():
    mock_a.method.assert_not_called()
    mock_b.other.assert_not_called()
'''
        violations = find_violations(source, 'test_only_not_called.py')
        assert violations == []

    def test_ignores_assert_not_awaited_when_function_has_no_assert_not_called(self):
        """Function with only assert_not_awaited — no violation."""
        source = '''\
async def test_only_not_awaited():
    backend.get_all_valid_edges.assert_not_awaited()
    backend.search.assert_not_awaited()
'''
        violations = find_violations(source, 'test_only_not_awaited.py')
        assert violations == []
