"""Tests for shared.task_complexity — the one "author declared simple" predicate."""
from __future__ import annotations

import pytest

from shared.task_complexity import is_simple_complexity


@pytest.mark.parametrize(
    'value',
    ['simple', 'Simple', 'SIMPLE', ' simple ', '\tsimple\n'],
    ids=['canonical', 'capitalised', 'upper', 'padded', 'tab-newline'],
)
def test_simple_spellings_declare_the_fast_path(value):
    assert is_simple_complexity(value) is True


@pytest.mark.parametrize(
    'value',
    [None, '', '   ', 'full', 'simpler', 'not simple', 0, False, ['simple'], {'simple': True}],
    ids=[
        'none',
        'empty',
        'blank',
        'full',
        'simpler',
        'not-simple',
        'zero',
        'false',
        'list-of-simple',
        'dict-keyed-simple',
    ],
)
def test_everything_else_does_not(value):
    assert is_simple_complexity(value) is False


def test_stringified_list_is_not_simple():
    assert str(['simple']) != 'simple'
    assert is_simple_complexity(['simple']) is False
