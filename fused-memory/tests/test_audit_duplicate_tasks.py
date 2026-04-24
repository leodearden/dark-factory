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
pick_survivor = _mod.pick_survivor
compute_dependency_updates = _mod.compute_dependency_updates
DependencyUpdate = _mod.DependencyUpdate


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


# ===========================================================================
# Step-5: pick_survivor
# ===========================================================================

class TestPickSurvivorPriority:
    """Survivor = highest priority; ties broken by lowest ID."""

    def test_high_priority_wins_over_medium(self):
        """Task with 'high' priority beats one with 'medium' regardless of ID order."""
        group = [
            _task('200', 'Do thing', priority='medium'),
            _task('201', 'Do thing', priority='high'),
        ]
        survivor, losers = pick_survivor(group)
        assert survivor['id'] == '201'
        assert len(losers) == 1
        assert losers[0]['id'] == '200'

    def test_critical_wins_over_high(self):
        """'critical' outranks 'high'."""
        group = [
            _task('210', 'Do thing', priority='high'),
            _task('211', 'Do thing', priority='critical'),
        ]
        survivor, losers = pick_survivor(group)
        assert survivor['id'] == '211'

    def test_all_same_priority_lowest_id_wins(self):
        """All tasks same priority → lowest ID wins (earliest created)."""
        group = [
            _task('220', 'Do thing', priority='medium'),
            _task('221', 'Do thing', priority='medium'),
            _task('222', 'Do thing', priority='medium'),
        ]
        survivor, losers = pick_survivor(group)
        assert survivor['id'] == '220'
        assert len(losers) == 2
        loser_ids = {t['id'] for t in losers}
        assert loser_ids == {'221', '222'}


class TestPickSurvivorMissingPriority:
    """Tasks with absent / unknown priority rank below any explicit priority."""

    def test_absent_priority_ranks_below_low(self):
        """Task with priority='low' beats one with no priority field."""
        group = [
            {'id': '230', 'title': 'Task X'},                   # no priority key
            _task('231', 'Task X', priority='low'),
        ]
        survivor, losers = pick_survivor(group)
        assert survivor['id'] == '231'

    def test_none_priority_ranks_below_low(self):
        """priority=None is treated the same as absent (rank=0)."""
        group = [
            {'id': '240', 'title': 'Task Y', 'priority': None},
            _task('241', 'Task Y', priority='low'),
        ]
        survivor, losers = pick_survivor(group)
        assert survivor['id'] == '241'

    def test_two_absent_priority_tasks_lowest_id_wins(self):
        """Two tasks with no priority → lowest ID wins."""
        group = [
            {'id': '250', 'title': 'Task Z'},
            {'id': '251', 'title': 'Task Z'},
        ]
        survivor, losers = pick_survivor(group)
        assert survivor['id'] == '250'


class TestPickSurvivorEdgeCases:
    """Degenerate input and determinism guarantees."""

    def test_single_task_raises_value_error(self):
        """A group of 1 task is degenerate — ValueError must be raised."""
        with pytest.raises(ValueError):
            pick_survivor([_task('260', 'Solo task')])

    def test_empty_group_raises_value_error(self):
        """Empty group is also degenerate."""
        with pytest.raises(ValueError):
            pick_survivor([])

    def test_deterministic_with_shuffled_input(self):
        """Shuffling input order always yields the same survivor."""
        import random  # noqa: PLC0415
        group = [
            _task('270', 'Work item', priority='high'),
            _task('271', 'Work item', priority='medium'),
            _task('272', 'Work item', priority='high'),    # same prio as 270, higher ID
            _task('273', 'Work item', priority='low'),
        ]
        expected_survivor_id = '270'  # high priority, lowest ID among high-prio tasks
        for _ in range(10):
            shuffled = list(group)
            random.shuffle(shuffled)
            survivor, _ = pick_survivor(shuffled)
            assert survivor['id'] == expected_survivor_id


# ===========================================================================
# Step-7: compute_dependency_updates
# ===========================================================================

class TestComputeDependencyUpdatesBasic:
    """Core: remapping cancelled deps to the survivor."""

    def test_no_task_references_cancelled_id_returns_empty(self):
        """No task has a cancelled task in its deps → empty list."""
        all_tasks = [
            _task('300', 'Survivor'),
            _task('301', 'Other task', dependencies=['300']),  # depends on survivor, not cancelled
        ]
        result = compute_dependency_updates(
            survivor_id='300',
            cancelled_ids={'999'},
            all_tasks=all_tasks,
        )
        assert result == []

    def test_single_task_with_cancelled_dep(self):
        """Task A depends on cancelled X → emit (remove=X, add=survivor)."""
        all_tasks = [
            _task('310', 'Survivor'),
            _task('311', 'Task A', dependencies=['399']),
        ]
        result = compute_dependency_updates(
            survivor_id='310',
            cancelled_ids={'399'},
            all_tasks=all_tasks,
        )
        assert len(result) == 1
        u = result[0]
        assert u.dependent_id == '311'
        assert u.remove_dep == '399'
        assert u.add_dep == '310'

    def test_task_already_depends_on_survivor_only_remove(self):
        """Task B depends on [cancelled X, survivor] → remove X, do NOT add survivor again."""
        all_tasks = [
            _task('320', 'Survivor'),
            _task('321', 'Task B', dependencies=['399', '320']),
        ]
        result = compute_dependency_updates(
            survivor_id='320',
            cancelled_ids={'399'},
            all_tasks=all_tasks,
        )
        assert len(result) == 1
        u = result[0]
        assert u.dependent_id == '321'
        assert u.remove_dep == '399'
        assert u.add_dep is None  # survivor already present


class TestComputeDependencyUpdatesMultipleCancelledDeps:
    """One task depending on two different cancelled IDs → two remove ops, one add."""

    def test_two_cancelled_deps_two_removes_one_add(self):
        """Task C depends on cancelled_X and cancelled_Y (both → same survivor).

        Expect: two DependencyUpdates — first has add_dep=survivor, second has add_dep=None.
        """
        all_tasks = [
            _task('330', 'Survivor'),
            _task('331', 'Task C', dependencies=['397', '398']),
        ]
        result = compute_dependency_updates(
            survivor_id='330',
            cancelled_ids={'397', '398'},
            all_tasks=all_tasks,
        )
        assert len(result) == 2
        remove_deps = {u.remove_dep for u in result}
        assert remove_deps == {'397', '398'}
        # Survivor should be added exactly once across all updates for task C.
        add_deps = [u.add_dep for u in result if u.add_dep is not None]
        assert len(add_deps) == 1
        assert add_deps[0] == '330'


class TestComputeDependencyUpdatesCancelledTaskSkipped:
    """Cancelled tasks themselves are NOT included as dependents."""

    def test_cancelled_task_not_remapped(self):
        """If a task is in cancelled_ids, its own outgoing deps are not remapped."""
        all_tasks = [
            _task('340', 'Survivor'),
            _task('341', 'Cancelled loser', dependencies=['399']),  # being cancelled
            _task('342', 'Active task', dependencies=['341']),
        ]
        result = compute_dependency_updates(
            survivor_id='340',
            cancelled_ids={'341'},
            all_tasks=all_tasks,
        )
        # Task 341 is being cancelled — it must NOT appear as a dependent.
        dependent_ids = {u.dependent_id for u in result}
        assert '341' not in dependent_ids
        # Task 342 depends on 341 (cancelled) → should be remapped.
        assert '342' in dependent_ids


class TestComputeDependencyUpdatesIntDependencies:
    """Dependencies stored as int OR str → both shapes handled."""

    def test_int_dependency_ids_normalised(self):
        """Dependencies given as integers (not strings) are normalised correctly."""
        all_tasks = [
            _task('350', 'Survivor'),
            {'id': '351', 'title': 'Task D', 'status': 'pending', 'dependencies': [399]},  # int
        ]
        result = compute_dependency_updates(
            survivor_id='350',
            cancelled_ids={'399'},
            all_tasks=all_tasks,
        )
        assert len(result) == 1
        u = result[0]
        assert u.dependent_id == '351'
        assert u.remove_dep == '399'
        assert u.add_dep == '350'
