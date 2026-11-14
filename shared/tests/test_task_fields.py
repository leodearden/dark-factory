"""Tests for the shared task-field readers (S17 α0, contract C11)."""

from __future__ import annotations

import pytest

from shared.task_fields import dependency_ids, is_deterministic, train_id


class TestDependencyIds:
    def test_list_of_str(self):
        assert dependency_ids({'dependencies': ['1', '2', '3']}) == ['1', '2', '3']

    def test_list_of_int_coerced_to_str(self):
        assert dependency_ids({'dependencies': [1, 2, 3]}) == ['1', '2', '3']

    def test_mixed_list(self):
        assert dependency_ids({'dependencies': [1, '2', 3]}) == ['1', '2', '3']

    def test_dict_entry_with_id_is_unwrapped(self):
        assert dependency_ids({'dependencies': [{'id': 7}, {'id': '6'}, 8, '9']}) == ['7', '6', '8', '9']

    def test_set_takes_the_list_branch(self):
        assert sorted(dependency_ids({'dependencies': {1, '2', 0}})) == ['1', '2']

    def test_dict_entry_without_id_is_stringified_whole(self):
        entry = {'name': 'x'}
        assert dependency_ids({'dependencies': [entry]}) == [str(entry)]

    def test_csv_string_fallback(self):
        assert dependency_ids({'dependencies': '1, 2,3'}) == ['1', '2', '3']

    def test_csv_string_with_empty_pieces_dropped(self):
        assert dependency_ids({'dependencies': '1,,2, ,3'}) == ['1', '2', '3']

    @pytest.mark.parametrize('task', [{}, {'dependencies': None}, {'dependencies': []}, {'dependencies': ''}])
    def test_missing_or_empty_is_empty(self, task):
        assert dependency_ids(task) == []

    def test_falsy_entries_dropped_from_list(self):
        assert dependency_ids({'dependencies': [1, 0, None, 2]}) == ['1', '2']

    def test_tuple_takes_the_list_branch(self):
        assert dependency_ids({'dependencies': (1, '2')}) == ['1', '2']

    def test_bare_int_does_not_raise(self):
        assert dependency_ids({'dependencies': 5}) == ['5']

    def test_dict_as_whole_value_is_stringified_not_iterated(self):
        deps = {'a': 1}
        assert dependency_ids({'dependencies': deps}) == [str(deps)]

    def test_empty_dict_value(self):
        assert dependency_ids({'dependencies': {}}) == []

    def test_returns_a_fresh_list(self):
        source = ['1', '2']
        result = dependency_ids({'dependencies': source})
        result.append('3')
        assert source == ['1', '2']


class TestTrainId:
    def test_non_empty_id(self):
        assert train_id({'metadata': {'train': {'id': 'tr-1'}}}) == 'tr-1'

    def test_int_id_is_stringified(self):
        assert train_id({'metadata': {'train': {'id': 5}}}) == '5'

    def test_json_string_metadata(self):
        assert train_id({'metadata': '{"train": {"id": "tr-1"}}'}) == 'tr-1'

    @pytest.mark.parametrize(
        'task',
        [
            {},
            {'metadata': None},
            {'metadata': {}},
            {'metadata': {'train': None}},
            {'metadata': {'train': {}}},
            {'metadata': {'train': {'id': None}}},
            {'metadata': {'train': {'id': ''}}},
            {'metadata': {'train': {'id': '   '}}},
            {'metadata': {'train': {'id': 0}}},
            {'metadata': {'train': {'id': False}}},
            {'metadata': {'train': {'id': []}}},
            {'metadata': {'train': {'id': {}}}},
            {'metadata': {'train': 'tr-1'}},
            {'metadata': 'not-a-dict'},
        ],
    )
    def test_absent_blank_or_malformed_is_none(self, task):
        assert train_id(task) is None


class TestIsDeterministic:
    def test_dict_metadata(self):
        assert is_deterministic({'metadata': {'task_kind': 'deterministic'}})

    def test_json_string_metadata(self):
        assert is_deterministic({'metadata': '{"task_kind": "deterministic"}'})

    @pytest.mark.parametrize(
        'task',
        [
            {},
            {'metadata': None},
            {'metadata': {}},
            {'metadata': {'task_kind': 'normal'}},
            {'metadata': 'not json'},
            {'metadata': '[1, 2]'},
            {'metadata': 5},
        ],
    )
    def test_everything_else_is_false(self, task):
        assert not is_deterministic(task)
