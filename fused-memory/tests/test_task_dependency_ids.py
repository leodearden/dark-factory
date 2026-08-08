"""Tests for the shared CSV-tolerant task-dependency-id parser (task 3615)."""

from __future__ import annotations

from fused_memory.utils.task_dependency_ids import task_dependency_ids


class TestTaskDependencyIds:
    def test_list_of_str(self):
        assert task_dependency_ids({'dependencies': ['1', '2', '3']}) == ['1', '2', '3']

    def test_list_of_int_coerced_to_str(self):
        # get_tasks returns dependencies as INTS while task ids are STRINGS —
        # this is the coercion that must never be dropped.
        assert task_dependency_ids({'dependencies': [1, 2, 3]}) == ['1', '2', '3']

    def test_mixed_list(self):
        assert task_dependency_ids({'dependencies': [1, '2', 3]}) == ['1', '2', '3']

    def test_csv_string_fallback(self):
        assert task_dependency_ids({'dependencies': '1, 2,3'}) == ['1', '2', '3']

    def test_csv_string_with_empty_pieces_dropped(self):
        assert task_dependency_ids({'dependencies': '1,,2, ,3'}) == ['1', '2', '3']

    def test_missing_key(self):
        assert task_dependency_ids({}) == []

    def test_none_value(self):
        assert task_dependency_ids({'dependencies': None}) == []

    def test_empty_list(self):
        assert task_dependency_ids({'dependencies': []}) == []

    def test_empty_string(self):
        assert task_dependency_ids({'dependencies': ''}) == []

    def test_falsy_entries_dropped_from_list(self):
        assert task_dependency_ids({'dependencies': [1, 0, None, 2]}) == ['1', '2']

    def test_tuple_takes_the_list_branch(self):
        assert task_dependency_ids({'dependencies': (1, '2')}) == ['1', '2']

    def test_bare_scalar_does_not_raise(self):
        # A single-dep row stored unwrapped. Iterating an int raises
        # TypeError, which every caller's surrounding `except Exception`
        # would swallow — skipping the whole sweep, not just this row.
        assert task_dependency_ids({'dependencies': 5}) == ['5']

    def test_unknown_shape_is_stringified_whole_not_iterated(self):
        # A dict is iterable, so the naive parse would silently yield its
        # KEYS as dependency ids. Stringifying whole yields an id that
        # matches nothing (fails closed) instead.
        deps = {'a': 1}
        assert task_dependency_ids({'dependencies': deps}) == [str(deps)]
        assert task_dependency_ids({'dependencies': deps}) != ['a']

    def test_empty_unknown_shape(self):
        assert task_dependency_ids({'dependencies': {}}) == []


class TestCallersShareTheOneParser:
    """The consolidation itself, pinned — not just the parse table.

    If someone reintroduces a local copy of this parse in a caller, every
    behavioural case above still passes while the drift this task exists to
    remove is silently back. These identity assertions are what catch that.
    """

    def test_task_curator_reexports_the_shared_parser(self):
        from fused_memory.middleware import task_curator

        assert task_curator._task_dependencies is task_dependency_ids

    def test_targeted_reconciler_imports_the_shared_parser(self):
        from fused_memory.reconciliation import targeted

        assert targeted.task_dependency_ids is task_dependency_ids
