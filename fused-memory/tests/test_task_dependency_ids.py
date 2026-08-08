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
