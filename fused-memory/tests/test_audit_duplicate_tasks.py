"""Tests for audit_duplicate_tasks.py.

Loaded via importlib so the script (not on PYTHONPATH) can be tested without
sys.path pollution — mirrors the pattern in test_check_asyncmock_assertion_style.py.
"""
from __future__ import annotations

import importlib.util
import logging
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
apply_changes = _mod.apply_changes
build_audit_plan = _mod.build_audit_plan
_extract_tasks = _mod._extract_tasks


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
# Deterministic ordering: find_exact_duplicate_groups
# ===========================================================================

class TestFindExactDuplicateGroupsDeterministicOrdering:
    """find_exact_duplicate_groups sorts members by ID and groups by min ID."""

    def test_members_within_group_sorted_by_id(self):
        """Members of the same duplicate group are returned sorted by numeric ID."""
        # Deliberately insert in non-ascending order: 1003, 1001, 1002
        tasks = [
            _task('1003', 'Sync database'),
            _task('1001', 'Sync database'),
            _task('1002', 'Sync database'),
        ]
        result = find_exact_duplicate_groups(tasks)
        assert len(result) == 1
        assert [t['id'] for t in result[0]] == ['1001', '1002', '1003']

    def test_groups_sorted_by_min_id(self):
        """Groups are returned sorted by the minimum (first) ID within each group."""
        # Interleave two duplicate pairs: 'Beta' group (min 2001) first in input,
        # 'Alpha' group (min 1001) second — expected output reverses that order.
        tasks = [
            _task('2001', 'Beta task'),
            _task('2002', 'Beta task'),
            _task('1001', 'Alpha task'),
            _task('1002', 'Alpha task'),
        ]
        result = find_exact_duplicate_groups(tasks)
        assert len(result) == 2
        assert result[0][0]['id'] == '1001'
        assert result[1][0]['id'] == '2001'

    def test_non_numeric_ids_do_not_raise(self):
        """Non-numeric/dotted IDs (e.g. '1.2') are handled via _id_as_int fallback=0.

        Regression guard: if int() were used directly instead of _id_as_int, sorting
        would raise ValueError on dotted subtask IDs. This test locks in:
        - No exception is raised during sorting.
        - The dotted-ID group (both members map to fallback=0) sorts before the
          numeric group (min ID=1001 > 0), consistent with the documented ordering.
        """
        tasks = [
            _task('1001', 'Numeric task'),
            _task('1002', 'Numeric task'),
            _task('1.2', 'Dotted task'),
            _task('1.3', 'Dotted task'),
        ]
        # Must not raise (int('1.2') would raise ValueError).
        result = find_exact_duplicate_groups(tasks)
        assert len(result) == 2
        # Dotted-ID group: _id_as_int → 0 for both members, so g[0] min = 0 < 1001.
        dotted_ids = {t['id'] for t in result[0]}
        numeric_ids = {t['id'] for t in result[1]}
        assert dotted_ids == {'1.2', '1.3'}
        assert numeric_ids == {'1001', '1002'}


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


# ===========================================================================
# Step-9: apply_changes
# ===========================================================================

# apply_changes signature:
#   apply_changes(backend, project_root, plan, tag=None) -> None (async)
#
# plan dict shape:
#   {'cancellations': [task_id, ...], 'dependency_updates': [dict, ...]}
#
# Uses assert_awaited_* (not assert_called_*) per check_asyncmock_assertion_style.py lint rule.



@pytest.mark.asyncio
class TestApplyChangesCancellations:
    """Cancellation calls: one set_task_status per ID in plan['cancellations']."""

    async def test_set_task_status_called_once_per_cancellation(self):
        """Each cancellation ID triggers exactly one set_task_status('cancelled') call."""
        from unittest.mock import AsyncMock, MagicMock  # noqa: PLC0415

        backend = MagicMock()
        backend.set_task_status = AsyncMock(return_value={})
        backend.remove_dependency = AsyncMock(return_value={})
        backend.add_dependency = AsyncMock(return_value={})

        plan = {'cancellations': ['401', '402'], 'dependency_updates': []}
        await apply_changes(backend, '/project', plan, tag='master')

        assert backend.set_task_status.await_count == 2
        # Both calls should pass 'cancelled' as the status and the correct project_root.
        calls = backend.set_task_status.await_args_list
        called_ids = {c.args[0] for c in calls}
        called_statuses = {c.args[1] for c in calls}
        assert called_ids == {'401', '402'}
        assert called_statuses == {'cancelled'}
        # project_root and tag must be passed through.
        for c in calls:
            assert c.args[2] == '/project'
            assert c.args[3] == 'master'

    async def test_no_cancellations_set_task_status_not_called(self):
        """Empty cancellations list → set_task_status never awaited."""
        from unittest.mock import AsyncMock, MagicMock  # noqa: PLC0415

        backend = MagicMock()
        backend.set_task_status = AsyncMock(return_value={})
        backend.remove_dependency = AsyncMock(return_value={})
        backend.add_dependency = AsyncMock(return_value={})

        plan = {'cancellations': [], 'dependency_updates': []}
        await apply_changes(backend, '/project', plan)

        backend.set_task_status.assert_not_awaited()


@pytest.mark.asyncio
class TestApplyChangesDependencyUpdates:
    """Dependency remap calls: remove + optional add per DependencyUpdate."""

    async def test_remove_dependency_called_per_update(self):
        """Each dep update triggers a remove_dependency call."""
        from unittest.mock import AsyncMock, MagicMock  # noqa: PLC0415

        backend = MagicMock()
        backend.set_task_status = AsyncMock(return_value={})
        backend.remove_dependency = AsyncMock(return_value={})
        backend.add_dependency = AsyncMock(return_value={})

        plan = {
            'cancellations': [],
            'dependency_updates': [
                {'dependent_id': '410', 'remove_dep': '499', 'add_dep': '400'},
            ],
        }
        await apply_changes(backend, '/project', plan)

        backend.remove_dependency.assert_awaited_once_with('410', '499', '/project', None)
        backend.add_dependency.assert_awaited_once_with('410', '400', '/project', None)

    async def test_add_dependency_not_called_when_add_dep_is_none(self):
        """When add_dep is None, add_dependency must not be called."""
        from unittest.mock import AsyncMock, MagicMock  # noqa: PLC0415

        backend = MagicMock()
        backend.set_task_status = AsyncMock(return_value={})
        backend.remove_dependency = AsyncMock(return_value={})
        backend.add_dependency = AsyncMock(return_value={})

        plan = {
            'cancellations': [],
            'dependency_updates': [
                {'dependent_id': '420', 'remove_dep': '499', 'add_dep': None},
            ],
        }
        await apply_changes(backend, '/project', plan)

        backend.remove_dependency.assert_awaited_once_with('420', '499', '/project', None)
        backend.add_dependency.assert_not_awaited()

    async def test_cancellations_precede_dependency_updates(self):
        """ALL cancellations complete before ANY dep operation — verified with 2+2 ops.

        Uses 2 cancellations and 2 dep updates so a single index check can't
        mask future interleaving: asserts max(cancel_indices) < min(dep_indices).
        """
        from unittest.mock import AsyncMock, MagicMock  # noqa: PLC0415

        call_order: list[str] = []

        backend = MagicMock()

        async def _cancel(tid, status, pr, tag=None):
            call_order.append(f'cancel:{tid}')
            return {}

        async def _remove(dep, rem, pr, tag=None):
            call_order.append(f'remove:{dep}->{rem}')
            return {}

        async def _add(dep, add, pr, tag=None):
            call_order.append(f'add:{dep}->{add}')
            return {}

        backend.set_task_status = AsyncMock(side_effect=_cancel)
        backend.remove_dependency = AsyncMock(side_effect=_remove)
        backend.add_dependency = AsyncMock(side_effect=_add)

        plan = {
            'cancellations': ['430', '431'],
            'dependency_updates': [
                {'dependent_id': '432', 'remove_dep': '430', 'add_dep': '429'},
                {'dependent_id': '433', 'remove_dep': '431', 'add_dep': '429'},
            ],
        }
        await apply_changes(backend, '/project', plan)

        # All cancel: entries must appear before any remove:/add: entry.
        cancel_indices = [i for i, e in enumerate(call_order) if e.startswith('cancel:')]
        dep_indices = [i for i, e in enumerate(call_order) if not e.startswith('cancel:')]
        assert len(cancel_indices) == 2, f'Expected 2 cancel ops, got: {call_order}'
        assert len(dep_indices) >= 2, f'Expected ≥2 dep ops, got: {call_order}'
        assert max(cancel_indices) < min(dep_indices), (
            f'Some cancellation came after a dep op: {call_order}'
        )

    async def test_partial_failure_does_not_abort_remaining_ops(self):
        """A failing set_task_status should not prevent subsequent operations."""
        from unittest.mock import AsyncMock, MagicMock  # noqa: PLC0415

        backend = MagicMock()
        backend.set_task_status = AsyncMock(side_effect=RuntimeError('Taskmaster offline'))
        backend.remove_dependency = AsyncMock(return_value={})
        backend.add_dependency = AsyncMock(return_value={})

        plan = {
            'cancellations': ['440'],
            'dependency_updates': [
                {'dependent_id': '441', 'remove_dep': '440', 'add_dep': '439'},
            ],
        }
        # Should NOT raise; partial failure is tolerated.
        result = await apply_changes(backend, '/project', plan)

        # Dep updates still proceed despite cancel failure.
        backend.remove_dependency.assert_awaited_once()
        backend.add_dependency.assert_awaited_once()

        # Return counters must reflect: 1 cancel error, 1 dep update applied, 0 dep errors.
        assert result['cancelled'] == 0
        assert result['cancel_errors'] == 1
        assert result['dep_updates_applied'] == 1
        assert result['dep_update_errors'] == 0

    async def test_remove_failure_skips_add_and_counts_one_error(self):
        """When remove_dependency raises, add_dependency must NOT be called and the error is counted once."""
        from unittest.mock import AsyncMock, MagicMock  # noqa: PLC0415

        backend = MagicMock()
        backend.set_task_status = AsyncMock(return_value={})
        backend.remove_dependency = AsyncMock(side_effect=RuntimeError('backend offline'))
        backend.add_dependency = AsyncMock(return_value={})

        plan = {
            'cancellations': [],
            'dependency_updates': [
                {'dependent_id': '450', 'remove_dep': '499', 'add_dep': '400'},
            ],
        }
        result = await apply_changes(backend, '/project', plan)

        # remove was attempted
        backend.remove_dependency.assert_awaited_once_with('450', '499', '/project', None)
        # add must be skipped entirely when remove fails
        backend.add_dependency.assert_not_awaited()
        # failure counted exactly once, not as applied
        assert result == {
            'cancelled': 0,
            'cancel_errors': 0,
            'dep_updates_applied': 0,
            'dep_update_errors': 1,
        }


# ===========================================================================
# Step-11: build_audit_plan
# ===========================================================================

class TestBuildAuditPlanFiltering:
    """Tasks with status done/cancelled or id < min_id are filtered out."""

    def test_done_and_cancelled_tasks_excluded(self):
        """Only pending and in-progress tasks are scanned."""
        tasks = [
            _task('1000', 'Duplicate title'),
            _task('1001', 'Duplicate title', status='done'),
            _task('1002', 'Duplicate title', status='cancelled'),
        ]
        plan = build_audit_plan(tasks, min_id=1000)
        # Only tasks 1000 is left as a candidate — no duplicate group.
        assert plan['candidates_total'] == 1
        assert plan['auto_cancel'] == []

    def test_tasks_below_min_id_excluded(self):
        """Tasks with id < min_id are not scanned."""
        tasks = [
            _task('999', 'Duplicate title'),
            _task('1000', 'Duplicate title'),
        ]
        plan = build_audit_plan(tasks, min_id=1000)
        # Task 999 is below the cutoff — only 1000 is a candidate.
        assert plan['candidates_total'] == 1
        assert plan['auto_cancel'] == []


class TestBuildAuditPlanExactGroups:
    """Exact duplicate handling: pending losers auto-cancelled, in-progress → human review."""

    def test_pending_and_inprogress_both_in_same_exact_group(self):
        """Both pending and in-progress duplicates of the same title form one exact group."""
        tasks = [
            _task('1010', 'Analyse telemetry', status='pending', priority='medium'),
            _task('1011', 'Analyse telemetry', status='in-progress', priority='medium'),
        ]
        plan = build_audit_plan(tasks, min_id=1000)
        # Should have one exact group.
        assert len(plan['exact_groups']) == 1

    def test_pending_loser_auto_cancelled(self):
        """Pending loser (lower priority / higher ID) goes into auto_cancel."""
        tasks = [
            _task('1020', 'Build cache layer', status='pending', priority='high'),
            _task('1021', 'Build cache layer', status='pending', priority='medium'),
        ]
        plan = build_audit_plan(tasks, min_id=1000)
        # 1020 = survivor (high prio); 1021 = loser (medium prio) → auto_cancel.
        assert '1021' in plan['auto_cancel']
        assert '1020' not in plan['auto_cancel']

    def test_inprogress_loser_goes_to_human_review(self):
        """In-progress loser is reported under needs_human_review, NOT auto-cancelled."""
        tasks = [
            _task('1030', 'Migrate schema', status='pending', priority='high'),
            _task('1031', 'Migrate schema', status='in-progress', priority='low'),
        ]
        plan = build_audit_plan(tasks, min_id=1000)
        # 1031 is in-progress loser → human review, NOT auto_cancel.
        assert '1031' not in plan['auto_cancel']
        human_ids = {r['id'] for r in plan['needs_human_review']}
        assert '1031' in human_ids

    def test_dependency_updates_only_for_auto_cancelled_ids(self):
        """Dependency remaps are computed only for auto-cancelled (pending) losers."""
        tasks = [
            _task('1040', 'Core feature', status='pending', priority='high'),
            _task('1041', 'Core feature', status='pending', priority='low'),
            # Task that depends on the loser.
            _task('1042', 'Follow-up task', dependencies=['1041']),
        ]
        plan = build_audit_plan(tasks, min_id=1000)
        assert '1041' in plan['auto_cancel']
        # Dependency remap: 1042 depends on loser 1041 → remap to survivor 1040.
        dep_updates = plan['dependency_updates']
        assert len(dep_updates) >= 1
        remap = next(u for u in dep_updates if u['dependent_id'] == '1042')
        assert remap['remove_dep'] == '1041'
        assert remap['add_dep'] == '1040'


class TestBuildAuditPlanNearDuplicates:
    """Near-duplicate groups are reported separately — no auto-cancel."""

    def test_near_duplicates_in_separate_bucket_no_auto_cancel(self):
        """Near-dups appear in near_duplicate_groups and are NOT auto-cancelled."""
        tasks = [
            _task('1050', 'Setup CI pipeline'),
            _task('1051', 'Setup the CI pipeline'),
        ]
        plan = build_audit_plan(tasks, threshold=0.80, min_id=1000)
        # No exact match, so auto_cancel should be empty.
        assert plan['auto_cancel'] == []
        # Near-dup group should be reported.
        assert len(plan['near_duplicate_groups']) >= 1

    def test_exact_match_ids_not_in_near_duplicate_groups(self):
        """Tasks already in exact groups must be excluded from near-dup search."""
        tasks = [
            _task('1060', 'Deploy service'),
            _task('1061', 'Deploy service'),          # exact duplicate of 1060
            _task('1062', 'Deploy service layer'),     # near-dup of 1060/1061
        ]
        plan = build_audit_plan(tasks, threshold=0.85, min_id=1000)
        # 1060 and 1061 are exact duplicates → excluded from near-dup groups.
        # 1062 should not be grouped with them (excluded IDs).
        near_ids = {t['id'] for g in plan['near_duplicate_groups'] for t in g['tasks']}
        # Exact match tasks must not leak into near_duplicate_groups.
        assert '1060' not in near_ids
        assert '1061' not in near_ids


class TestBuildAuditPlanPlanShape:
    """Return value always has the expected keys."""

    def test_plan_has_required_keys(self):
        """build_audit_plan always returns a dict with all required keys."""
        plan = build_audit_plan([], min_id=1000)
        required = {
            'candidates_total', 'exact_groups', 'near_duplicate_groups',
            'auto_cancel', 'needs_human_review', 'dependency_updates',
        }
        assert required <= set(plan.keys())

    def test_empty_input_produces_empty_plan(self):
        """No tasks → all lists/counts empty."""
        plan = build_audit_plan([], min_id=1000)
        assert plan['candidates_total'] == 0
        assert plan['exact_groups'] == []
        assert plan['near_duplicate_groups'] == []
        assert plan['auto_cancel'] == []
        assert plan['needs_human_review'] == []
        assert plan['dependency_updates'] == []


# ===========================================================================
# _extract_tasks: Taskmaster response unwrapping
# ===========================================================================


class TestExtractTasksValidShapes:
    """Valid input shapes return the task list; no WARNING is emitted."""

    def test_documented_shape_data_tasks(self, caplog):
        """{'data': {'tasks': [...]}} (documented 2026-04-25 shape) → inner list; no warning."""
        tasks = [{'id': '1', 'title': 'First'}, {'id': '2', 'title': 'Second'}]
        raw = {
            'data': {'tasks': tasks, 'filter': None, 'stats': {}},
            'version': {},
            'tag': 'master',
        }
        with caplog.at_level(logging.WARNING, logger='audit_duplicate_tasks'):
            result = _extract_tasks(raw)
        assert result == tasks
        assert not any(r.levelno >= logging.WARNING for r in caplog.records)

    def test_legacy_top_level_tasks(self, caplog):
        """{'tasks': [...]} (legacy shape without 'data' wrapper) → the list; no warning."""
        tasks = [{'id': '10', 'title': 'Legacy task'}]
        raw = {'tasks': tasks}
        with caplog.at_level(logging.WARNING, logger='audit_duplicate_tasks'):
            result = _extract_tasks(raw)
        assert result == tasks
        assert not any(r.levelno >= logging.WARNING for r in caplog.records)

    def test_legacy_data_as_list(self, caplog):
        """{'data': [...]} (data key holds a list directly) → the list; no warning."""
        tasks = [{'id': '20', 'title': 'Data list task'}]
        raw = {'data': tasks}
        with caplog.at_level(logging.WARNING, logger='audit_duplicate_tasks'):
            result = _extract_tasks(raw)
        assert result == tasks
        assert not any(r.levelno >= logging.WARNING for r in caplog.records)

    def test_bare_list(self, caplog):
        """A bare list → returns the same list; no warning."""
        tasks = [{'id': '30', 'title': 'Bare list task'}]
        with caplog.at_level(logging.WARNING, logger='audit_duplicate_tasks'):
            result = _extract_tasks(tasks)
        assert result == tasks
        assert not any(r.levelno >= logging.WARNING for r in caplog.records)

    def test_documented_shape_empty_tasks(self, caplog):
        """{'data': {'tasks': []}} (empty database) → [] with NO warning (recognised shape)."""
        raw = {
            'data': {'tasks': [], 'filter': None, 'stats': {}},
            'version': {},
            'tag': 'master',
        }
        with caplog.at_level(logging.WARNING, logger='audit_duplicate_tasks'):
            result = _extract_tasks(raw)
        assert result == []
        assert not any(r.levelno >= logging.WARNING for r in caplog.records)


class TestExtractTasksCorruptionShapes:
    """Non-recognisable but truthy inputs return [] and emit a WARNING with shape hint."""

    @pytest.mark.parametrize('raw', [
        {'data': {'items': [{'id': '1'}]}},
        {'unexpected_key': 'some_value'},
        42,
        'non_list_string',
    ])
    def test_returns_empty_list(self, raw):
        """Result is always [] for corruption shapes."""
        result = _extract_tasks(raw)
        assert result == []

    @pytest.mark.parametrize('raw,expected_hint_fragment', [
        (
            # dict data with no 'tasks' key — unrecognised inner shape
            {'data': {'items': [{'id': '1'}]}},
            "['data']",
        ),
        (
            # no 'data' or 'tasks' key at top level
            {'unexpected_key': 'some_value'},
            "['unexpected_key']",
        ),
        (
            # non-dict, non-list truthy value (int)
            42,
            'int',
        ),
        (
            # non-dict, non-list truthy value (str)
            'non_list_string',
            'str',
        ),
    ])
    def test_warning_logged_with_shape_hint(self, raw, expected_hint_fragment, caplog):
        """A WARNING identifying the mismatch and the top-level keys/type is emitted."""
        with caplog.at_level(logging.WARNING, logger='audit_duplicate_tasks'):
            _extract_tasks(raw)
        warnings = [
            r for r in caplog.records
            if r.levelno >= logging.WARNING and r.name == 'audit_duplicate_tasks'
        ]
        assert len(warnings) >= 1
        msg = warnings[0].getMessage()
        assert 'response-shape mismatch' in msg
        assert expected_hint_fragment in msg


@pytest.mark.parametrize('raw', [
    {},
    [],
    None,
])
class TestExtractTasksFalsyInputs:
    """Falsy inputs return [] and must NOT trigger any WARNING."""

    def test_returns_empty_list(self, raw):
        """[] for any falsy input."""
        assert _extract_tasks(raw) == []

    def test_no_warning_logged(self, raw, caplog):
        """No WARNING emitted for falsy inputs (they represent 'nothing returned')."""
        with caplog.at_level(logging.WARNING, logger='audit_duplicate_tasks'):
            _extract_tasks(raw)
        warnings = [
            r for r in caplog.records
            if r.levelno >= logging.WARNING and r.name == 'audit_duplicate_tasks'
        ]
        assert warnings == []
