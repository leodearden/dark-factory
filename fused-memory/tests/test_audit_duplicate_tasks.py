"""Tests for audit_duplicate_tasks.py.

Loaded via importlib so the script (not on PYTHONPATH) can be tested without
sys.path pollution — mirrors the pattern in test_check_asyncmock_assertion_style.py.
"""
from __future__ import annotations

import importlib.util
import types
from pathlib import Path

import pytest

SCRIPT_PATH = Path(__file__).parent.parent / 'scripts' / 'audit_duplicate_tasks.py'


def _load_module() -> types.ModuleType:
    """Load audit_duplicate_tasks.py from its file path."""
    spec = importlib.util.spec_from_file_location('audit_duplicate_tasks', SCRIPT_PATH)
    if spec is None or spec.loader is None:
        raise ImportError(f'Cannot load {SCRIPT_PATH}')
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)  # type: ignore[union-attr]
    return module


_mod = _load_module()
find_exact_duplicate_groups = _mod.find_exact_duplicate_groups


# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------

def _task(
    id: str,
    title: str,
    status: str = 'pending',
    priority: str = 'medium',
    dependencies: list[str] | None = None,
) -> dict:
    t: dict = {'id': id, 'title': title, 'status': status, 'priority': priority}
    if dependencies is not None:
        t['dependencies'] = dependencies
    return t


# ===========================================================================
# Step-1: find_exact_duplicate_groups
# ===========================================================================

class TestFindExactDuplicateGroupsEmptyAndUnique:
    """Baseline: empty / all-unique inputs produce no groups."""

    def test_empty_list_returns_empty(self):
        """No tasks → no duplicate groups."""
        result = find_exact_duplicate_groups([])
        assert result == []

    def test_single_task_returns_empty(self):
        """A single task can't form a duplicate group."""
        result = find_exact_duplicate_groups([_task('1', 'Build feature A')])
        assert result == []

    def test_all_unique_titles_returns_empty(self):
        """Three tasks with distinct titles → no groups."""
        tasks = [
            _task('1', 'Build feature A'),
            _task('2', 'Deploy to staging'),
            _task('3', 'Write integration tests'),
        ]
        result = find_exact_duplicate_groups(tasks)
        assert result == []


class TestFindExactDuplicateGroupsIdenticalTitles:
    """Core grouping: tasks with identical (normalised) titles form groups."""

    def test_two_tasks_identical_title_form_group_of_two(self):
        """Two tasks with the same title → one group containing both."""
        tasks = [
            _task('10', 'Refactor pipeline'),
            _task('11', 'Refactor pipeline'),
        ]
        result = find_exact_duplicate_groups(tasks)
        assert len(result) == 1
        group = result[0]
        assert len(group) == 2
        ids = {t['id'] for t in group}
        assert ids == {'10', '11'}

    def test_three_identical_plus_one_unrelated(self):
        """Three tasks with identical title + one unrelated → one group of 3."""
        tasks = [
            _task('20', 'Run migration script'),
            _task('21', 'Run migration script'),
            _task('22', 'Run migration script'),
            _task('23', 'Update documentation'),
        ]
        result = find_exact_duplicate_groups(tasks)
        assert len(result) == 1
        group = result[0]
        assert len(group) == 3
        ids = {t['id'] for t in group}
        assert ids == {'20', '21', '22'}
        # Unrelated task must NOT appear in any group.
        all_grouped_ids = {t['id'] for g in result for t in g}
        assert '23' not in all_grouped_ids

    def test_two_independent_duplicate_pairs(self):
        """Two separate duplicate pairs → two groups, each of size 2."""
        tasks = [
            _task('30', 'Task Alpha'),
            _task('31', 'Task Beta'),
            _task('32', 'Task Alpha'),
            _task('33', 'Task Beta'),
        ]
        result = find_exact_duplicate_groups(tasks)
        assert len(result) == 2
        sizes = sorted(len(g) for g in result)
        assert sizes == [2, 2]


class TestFindExactDuplicateGroupsNormalisation:
    """Normalisation: titles differing only in whitespace / case are grouped."""

    def test_case_insensitive_grouping(self):
        """'Fix Bug', 'fix bug', 'FIX BUG' are all identical after .lower()."""
        tasks = [
            _task('40', 'Fix Bug'),
            _task('41', 'fix bug'),
            _task('42', 'FIX BUG'),
        ]
        result = find_exact_duplicate_groups(tasks)
        assert len(result) == 1
        assert len(result[0]) == 3

    def test_leading_trailing_whitespace_stripped(self):
        """Leading/trailing spaces are stripped before comparison."""
        tasks = [
            _task('50', '  Deploy service  '),
            _task('51', 'Deploy service'),
        ]
        result = find_exact_duplicate_groups(tasks)
        assert len(result) == 1
        assert len(result[0]) == 2

    def test_whitespace_and_case_combined(self):
        """Both case and whitespace normalisation applied together."""
        tasks = [
            _task('60', '  Write Tests  '),
            _task('61', 'write tests'),
            _task('62', 'WRITE TESTS'),
        ]
        result = find_exact_duplicate_groups(tasks)
        assert len(result) == 1
        assert len(result[0]) == 3


class TestFindExactDuplicateGroupsNoStatusFiltering:
    """The function does NOT filter by status — that is the caller's job."""

    def test_does_not_filter_cancelled_tasks(self):
        """Cancelled tasks are still grouped if their titles match."""
        tasks = [
            _task('70', 'Archive logs', status='pending'),
            _task('71', 'Archive logs', status='cancelled'),
        ]
        result = find_exact_duplicate_groups(tasks)
        assert len(result) == 1
        ids = {t['id'] for t in result[0]}
        assert ids == {'70', '71'}

    def test_does_not_filter_done_tasks(self):
        """Done / in-progress tasks are still included when titles match."""
        tasks = [
            _task('80', 'Cleanup temp files', status='done'),
            _task('81', 'Cleanup temp files', status='in-progress'),
        ]
        result = find_exact_duplicate_groups(tasks)
        assert len(result) == 1
        assert len(result[0]) == 2

    def test_none_title_does_not_crash(self):
        """Tasks with a None title are normalised to '' and grouped together."""
        tasks = [
            {'id': '90', 'title': None, 'status': 'pending'},
            {'id': '91', 'title': None, 'status': 'pending'},
        ]
        result = find_exact_duplicate_groups(tasks)
        assert len(result) == 1
        ids = {t['id'] for t in result[0]}
        assert ids == {'90', '91'}
