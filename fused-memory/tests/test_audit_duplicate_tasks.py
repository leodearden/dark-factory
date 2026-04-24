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
    """Load audit_duplicate_tasks.py from its file path.

    The module is registered in sys.modules under its name so that
    @dataclass and other reflection-based decorators work correctly
    (they call sys.modules.get(cls.__module__)).
    """
    import sys  # noqa: PLC0415

    mod_name = 'audit_duplicate_tasks'
    spec = importlib.util.spec_from_file_location(mod_name, SCRIPT_PATH)
    if spec is None or spec.loader is None:
        raise ImportError(f'Cannot load {SCRIPT_PATH}')
    module = importlib.util.module_from_spec(spec)
    sys.modules[mod_name] = module  # required for @dataclass __module__ lookup
    try:
        spec.loader.exec_module(module)  # type: ignore[union-attr]
    except Exception:
        sys.modules.pop(mod_name, None)
        raise
    return module


_mod = _load_module()
find_exact_duplicate_groups = _mod.find_exact_duplicate_groups
find_near_duplicate_groups = _mod.find_near_duplicate_groups


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


# ===========================================================================
# Step-3: find_near_duplicate_groups
# ===========================================================================

class TestFindNearDuplicateGroupsPairwise:
    """Pairwise near-duplicate detection at different thresholds."""

    def test_similar_titles_grouped_at_lower_threshold(self):
        """'Refactor task pipeline' and 'Refactor the task pipeline' are near-dup at 0.85."""
        tasks = [
            _task('100', 'Refactor task pipeline'),
            _task('101', 'Refactor the task pipeline'),
        ]
        result = find_near_duplicate_groups(tasks, threshold=0.85)
        assert len(result) == 1
        ids = {t['id'] for t in result[0]}
        assert ids == {'100', '101'}

    def test_same_titles_not_grouped_at_very_high_threshold(self):
        """Those two titles are NOT grouped at threshold=0.99 (ratio is < 0.99)."""
        tasks = [
            _task('110', 'Refactor task pipeline'),
            _task('111', 'Refactor the task pipeline'),
        ]
        result = find_near_duplicate_groups(tasks, threshold=0.99)
        assert result == []

    def test_completely_unrelated_titles_not_grouped(self):
        """Titles with very low similarity are never grouped."""
        tasks = [
            _task('120', 'Fix authentication bug'),
            _task('121', 'Deploy new database index'),
        ]
        result = find_near_duplicate_groups(tasks, threshold=0.90)
        assert result == []


class TestFindNearDuplicateGroupsTransitive:
    """Transitive closure via union-find merges chains into a single group."""

    def test_three_transitively_similar_titles_merge_into_one_group(self):
        """'Foo bar', 'Foo baz', 'Foo bar baz' — all near each other → one merged group."""
        tasks = [
            _task('130', 'Foo bar'),
            _task('131', 'Foo baz'),
            _task('132', 'Foo bar baz'),
        ]
        # At threshold 0.60 all three are near-similar transitively.
        result = find_near_duplicate_groups(tasks, threshold=0.60)
        assert len(result) == 1
        group = result[0]
        assert len(group) == 3
        ids = {t['id'] for t in group}
        assert ids == {'130', '131', '132'}


class TestFindNearDuplicateGroupsExcludeIds:
    """Tasks in exclude_ids are dropped before pairwise comparison."""

    def test_excluded_ids_not_present_in_any_group(self):
        """If task '140' is in exclude_ids, it must not appear in any near-dup group."""
        tasks = [
            _task('140', 'Build widget'),
            _task('141', 'Build widget v2'),
            _task('142', 'Build widget v3'),
        ]
        result = find_near_duplicate_groups(tasks, threshold=0.75, exclude_ids={'140'})
        # Group for 141 + 142 may still form, but 140 must not be present.
        all_ids = {t['id'] for g in result for t in g}
        assert '140' not in all_ids

    def test_all_candidates_excluded_returns_empty(self):
        """Excluding all tasks leaves nothing to compare → []."""
        tasks = [
            _task('150', 'Setup CI'),
            _task('151', 'Setup CI pipeline'),
        ]
        result = find_near_duplicate_groups(tasks, threshold=0.80, exclude_ids={'150', '151'})
        assert result == []


class TestFindNearDuplicateGroupsEdgeCases:
    """Edge cases: empty input, single task, no groups for singletons."""

    def test_empty_list_returns_empty(self):
        result = find_near_duplicate_groups([], threshold=0.90)
        assert result == []

    def test_single_task_returns_empty(self):
        result = find_near_duplicate_groups([_task('160', 'Only task')], threshold=0.90)
        assert result == []

    def test_result_is_deterministic(self):
        """Shuffling input order should yield the same groups (sorted by min ID)."""
        import random  # noqa: PLC0415
        tasks = [
            _task('170', 'Deploy backend service'),
            _task('171', 'Deploy backend services'),
            _task('172', 'Deploy frontend service'),
            _task('173', 'Some unrelated task here'),
        ]
        result_a = find_near_duplicate_groups(list(tasks), threshold=0.80)
        shuffled = list(tasks)
        random.shuffle(shuffled)
        result_b = find_near_duplicate_groups(shuffled, threshold=0.80)
        # Groups should contain the same IDs regardless of input order.
        ids_a = sorted(sorted(t['id'] for t in g) for g in result_a)
        ids_b = sorted(sorted(t['id'] for t in g) for g in result_b)
        assert ids_a == ids_b
