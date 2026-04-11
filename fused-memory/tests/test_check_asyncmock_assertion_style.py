"""Tests for check_asyncmock_assertion_style.py lint checker.

Tests for the AST-based lint check that flags assert_not_called() when the same function
also contains assert_not_awaited(). See task 673 (lint guard replacing task-571 meta-test).
"""
from __future__ import annotations

import importlib.util
import subprocess
import sys
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


class TestFindViolationsPerFunctionScoping:
    """Rule is per-function, not per-class or per-module."""

    def test_scopes_per_function_not_per_class_or_module(self):
        """Two methods in the same class — each style in a different method — no violation."""
        source = '''\
class TestMyClass:
    def method_a(self):
        backend.get_edges.assert_not_awaited()

    def method_b(self):
        backend.get_nodes.assert_not_called()
'''
        violations = find_violations(source, 'test_scoped.py')
        assert violations == []

    def test_module_level_functions_scoped_independently(self):
        """Module-level assert_not_awaited in func_a + assert_not_called in func_b — no violation."""
        source = '''\
def func_a():
    mock.foo.assert_not_awaited()

def func_b():
    mock.bar.assert_not_called()
'''
        violations = find_violations(source, 'test_module_scoped.py')
        assert violations == []


class TestFindViolationsNestedFunctions:
    """Inner function assertions must not leak into the outer function's tally."""

    def test_inner_assert_not_awaited_does_not_pollute_outer_scope(self):
        """Outer has assert_not_called only; inner nested has assert_not_awaited only — no violation."""
        source = '''\
async def outer():
    mock.a.assert_not_called()

    async def inner():
        mock.b.assert_not_awaited()
'''
        violations = find_violations(source, 'test_nested.py')
        assert violations == []

    def test_inner_assert_not_called_does_not_pollute_outer_scope(self):
        """Outer has assert_not_awaited only; inner nested has assert_not_called only — no violation."""
        source = '''\
async def outer():
    mock.a.assert_not_awaited()

    async def inner():
        mock.b.assert_not_called()
'''
        violations = find_violations(source, 'test_nested_flip.py')
        assert violations == []

    def test_nested_function_itself_can_violate(self):
        """Inner function mixing both styles produces a violation scoped to the inner function."""
        source = '''\
def outer():
    mock.a.assert_not_awaited()

    def inner():
        mock.b.assert_not_called()
        mock.c.assert_not_awaited()
'''
        violations = find_violations(source, 'test_inner_violation.py')
        # inner() mixes both styles; outer() only has assert_not_awaited — outer is clean
        assert len(violations) == 1
        assert violations[0].lineno == 5  # assert_not_called in inner()


class TestFindViolationsMultipleCallsPerFunction:
    """Every assert_not_called in a mixed function is flagged, not just the first."""

    def test_reports_multiple_assert_not_called_violations_in_same_function(self):
        """Three assert_not_called calls in one function with assert_not_awaited → 3 violations."""
        source = '''\
async def test_three_not_called():
    backend.get_all_valid_edges.assert_not_awaited()
    backend.a.assert_not_called()
    backend.b.assert_not_called()
    backend.c.assert_not_called()
'''
        violations = find_violations(source, 'test_multi.py')
        assert len(violations) == 3
        line_numbers = [v.lineno for v in violations]
        assert 3 in line_numbers  # backend.a.assert_not_called()
        assert 4 in line_numbers  # backend.b.assert_not_called()
        assert 5 in line_numbers  # backend.c.assert_not_called()
        for v in violations:
            assert v.filename == 'test_multi.py'
            assert 'assert_not_awaited' in v.message


# ---------------------------------------------------------------------------
# CLI integration tests (steps 11-16)
# ---------------------------------------------------------------------------

_MIXED_STYLE_SOURCE = '''\
async def test_mixed():
    backend.get_all_valid_edges.assert_not_awaited()
    backend.foo.assert_not_called()
'''

_CLEAN_SOURCE_DIFFERENT_FUNCS = '''\
def func_a():
    mock.a.assert_not_awaited()

def func_b():
    mock.b.assert_not_called()
'''


class TestCliExitCodes:
    """CLI exit-code contract: 1 on violations, 0 on clean input."""

    def test_cli_main_exits_nonzero_and_prints_violations_on_bad_file(self, tmp_path: Path):
        """Violations file → returncode 1, stdout contains path/lineno/keywords."""
        bad_file = tmp_path / 'test_bad.py'
        bad_file.write_text(_MIXED_STYLE_SOURCE)
        result = subprocess.run(
            [sys.executable, str(SCRIPT_PATH), str(bad_file)],
            capture_output=True,
            text=True,
        )
        assert result.returncode == 1
        output = result.stdout
        assert str(bad_file) in output
        # assert_not_called is on line 3 of _MIXED_STYLE_SOURCE
        assert ':3:' in output
        assert 'assert_not_awaited' in output
        assert 'assert_not_called' in output

    def test_cli_main_exits_zero_on_clean_file(self, tmp_path: Path):
        """Clean file (styles in different functions) → returncode 0, stdout empty."""
        clean_file = tmp_path / 'test_clean.py'
        clean_file.write_text(_CLEAN_SOURCE_DIFFERENT_FUNCS)
        result = subprocess.run(
            [sys.executable, str(SCRIPT_PATH), str(clean_file)],
            capture_output=True,
            text=True,
        )
        assert result.returncode == 0
        assert result.stdout == ''


class TestCliDirectoryScan:
    """Directory-mode scans test_*.py and conftest.py recursively, ignores other .py files."""

    def test_cli_main_recursively_scans_directory_for_test_files_and_conftest(
        self, tmp_path: Path
    ):
        """Dir scan: test_example.py + conftest.py flagged; other_file.py ignored."""
        subdir = tmp_path / 'sub'
        subdir.mkdir()

        # Should be scanned and violate
        (subdir / 'test_example.py').write_text(_MIXED_STYLE_SOURCE)
        (subdir / 'conftest.py').write_text(_MIXED_STYLE_SOURCE)
        # Should NOT be scanned (not a test_*.py or conftest.py)
        (subdir / 'other_file.py').write_text(_MIXED_STYLE_SOURCE)

        result = subprocess.run(
            [sys.executable, str(SCRIPT_PATH), str(tmp_path)],
            capture_output=True,
            text=True,
        )
        assert result.returncode == 1
        output = result.stdout
        assert 'test_example.py' in output
        assert 'conftest.py' in output
        assert 'other_file.py' not in output
