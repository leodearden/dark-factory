"""Tests for ``shared.posix_ere``: a POSIX ERE read the way ``git grep -E``
reads it, asserted on the BEHAVIOUR of the compiled translation."""

from __future__ import annotations

import re
import warnings

import pytest

from shared.posix_ere import to_python_regex, top_level_arms


def _compiled(pattern: str) -> re.Pattern[str]:
    translated = to_python_regex(pattern)
    assert translated is not None
    return re.compile(translated)


class TestToPythonRegex:
    @pytest.mark.parametrize('pattern', [r'test_foo\.py', 'a|b', '^x(y|z)+$'])
    def test_an_ere_with_no_bracket_expression_is_unchanged(self, pattern):
        assert to_python_regex(pattern) == pattern

    def test_the_measured_space_class_specimen(self):
        matcher = _compiled('a[[:space:]]b')

        assert matcher.search('a b')
        assert matcher.search('a\tb')
        assert not matcher.search('a:]b')

    def test_a_class_beside_a_literal_fullmatches_a_snake_name(self):
        assert _compiled('[[:alpha:]_]+').fullmatch('check_ready')

    def test_a_negated_class(self):
        matcher = _compiled('[^[:digit:]]')

        assert matcher.fullmatch('x')
        assert not matcher.fullmatch('5')

    @pytest.mark.parametrize(
        ('pattern', 'member', 'non_member'),
        [
            ('[[:alnum:]]', 'q', '_'),
            ('[[:upper:]]', 'Q', 'q'),
            ('[[:xdigit:]]', 'f', 'g'),
            ('[[:punct:]]', '!', 'a'),
        ],
    )
    def test_each_class_admits_its_members_only(self, pattern, member, non_member):
        matcher = _compiled(pattern)

        assert matcher.fullmatch(member)
        assert not matcher.fullmatch(non_member)

    def test_a_backslash_is_literal_inside_a_bracket(self):
        matcher = _compiled('[\\.]')

        assert matcher.fullmatch('\\')
        assert matcher.fullmatch('.')

    def test_a_leading_close_bracket_is_literal(self):
        assert _compiled('[]a]').fullmatch(']')

    def test_a_range(self):
        matcher = _compiled('[a-c]')

        assert matcher.fullmatch('b')
        assert not matcher.fullmatch('d')

    @pytest.mark.parametrize('pattern', ['[-a]', '[a-]'])
    def test_a_dash_first_or_last_is_literal(self, pattern):
        assert _compiled(pattern).fullmatch('-')

    _SET_OPERATOR_SPECIMENS = [
        ('[a&&b]', '&'),
        ('[a||b]', '|'),
        ('[a~~b]', '~'),
        ('[[]', '['),
    ]

    @pytest.mark.parametrize(('pattern', 'punctuation'), _SET_OPERATOR_SPECIMENS)
    def test_python_set_operators_are_literal(self, pattern, punctuation):
        assert _compiled(pattern).fullmatch(punctuation)

    def test_no_translation_trips_pythons_set_syntax_warning(self):
        with warnings.catch_warnings(record=True) as caught:
            warnings.simplefilter('always')
            for pattern, _ in self._SET_OPERATOR_SPECIMENS:
                _compiled(pattern)

        assert [w for w in caught if issubclass(w.category, FutureWarning)] == []

    @pytest.mark.parametrize(
        'pattern',
        [
            pytest.param('[[:nope:]]', id='unknown-class'),
            pytest.param('[abc', id='unterminated-bracket'),
            pytest.param('[[.space.]]', id='collating-element'),
        ],
    )
    def test_untranslatable_input_is_none(self, pattern):
        assert to_python_regex(pattern) is None


class TestTopLevelArms:
    @pytest.mark.parametrize(
        ('pattern', 'arms'),
        [
            ('a|b', ['a', 'b']),
            ('(a|b)', ['a', 'b']),
            ('x[d|]', ['x[d|]']),
            (r'a\|b', [r'a\|b']),
        ],
    )
    def test_arms(self, pattern, arms):
        assert top_level_arms(pattern) == arms

    @pytest.mark.parametrize('pattern', ['(a', 'a)'])
    def test_an_unbalanced_pattern_is_none(self, pattern):
        assert top_level_arms(pattern) is None
