"""Tests for fused_memory.reconciliation.flag_shape — the leaf flag-text matcher."""

from __future__ import annotations

import pytest

from fused_memory.reconciliation.flag_shape import collect_str_values, contains_any_casefolded


class TestContainsAnyCasefolded:
    def test_member_found_inside_text(self):
        assert contains_any_casefolded('task_STRANDED_x', ('strand',)) is True

    def test_members_are_casefolded_too(self):
        assert contains_any_casefolded('stranded', ('STRAND',)) is True

    def test_no_member_found(self):
        assert contains_any_casefolded('abc', ('x',)) is False

    @pytest.mark.parametrize('text', [None, '', 5, b'strand', ['strand']])
    def test_total_over_malformed_text(self, text):
        assert contains_any_casefolded(text, ('strand',)) is False

    def test_empty_family_never_matches(self):
        assert contains_any_casefolded('anything', ()) is False


class TestCollectStrValues:
    def test_every_nested_str_value_in_traversal_order(self):
        flag = {
            'flag_type': 'topic_conflation',
            'description': 'entity a is too big',
            'evidence': {'note': 'see b'},
            'items': [{'ref': 'cited c'}, 'plain text'],
            'pair': ('left', {'deep': ['right']}),
        }
        assert collect_str_values(flag) == [
            'topic_conflation',
            'entity a is too big',
            'see b',
            'cited c',
            'plain text',
            'left',
            'right',
        ]

    def test_dict_keys_are_never_collected(self):
        assert collect_str_values({'key_only': None, 'k': 'value'}) == ['value']

    @pytest.mark.parametrize('value', [777, 0.5, True, None, b'bytes'])
    def test_non_str_scalars_are_ignored_not_stringified(self, value):
        assert collect_str_values({'edge_count': value}) == []

    def test_a_bare_str_is_its_own_only_value(self):
        assert collect_str_values('plain') == ['plain']

    @pytest.mark.parametrize('value', [{}, [], None, {'a': 1, 'b': None, 'c': [2, 3]}])
    def test_nothing_to_collect_is_an_empty_list(self, value):
        assert collect_str_values(value) == []

    def test_each_call_returns_a_fresh_list(self):
        first = collect_str_values({'a': 'x'})
        first.append('mutated')
        assert collect_str_values({'a': 'x'}) == ['x']
