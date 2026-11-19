"""The reading rule of ``shared.declared_checks.declared_checks``, pinned through its public surface."""

from __future__ import annotations

import dataclasses
from typing import Any

import pytest

from shared.declared_checks import DeclaredChecks, declared_checks

_A = {'name': 'cap-a', 'kind': 'grep', 'pattern': 'A', 'expect': 'present'}
_B = {'name': 'cap-b', 'kind': 'grep', 'pattern': 'B', 'expect': 'absent'}


def _read(value: Any) -> DeclaredChecks:
    return declared_checks({'delivered_checks': value})


@pytest.mark.parametrize(
    'metadata',
    [
        None,
        {},
        {'files': ['a.py']},
        {'delivered_checks': None},
        {'delivered_checks': []},
        {'delivered_checks': ''},
        {'delivered_checks': {}},
        {'delivered_checks': 0},
    ],
    ids=['none-metadata', 'empty-metadata', 'missing-key', 'none', 'empty-list', 'empty-str', 'empty-dict', 'zero'],
)
def test_absent_or_falsy_declares_nothing(metadata: dict[str, Any] | None) -> None:
    declared = declared_checks(metadata)
    assert declared.is_empty is True
    assert declared.checks == ()
    assert declared.unreadable == ()


def test_a_list_of_dicts_is_read_in_order() -> None:
    declared = _read([_A, _B])
    assert declared.checks == (_A, _B)
    assert declared.unreadable == ()
    assert declared.is_empty is False


def test_a_mixed_list_splits_element_wise_keeping_order() -> None:
    declared = _read(['x', _A, 7, _B])
    assert declared.checks == (_A, _B)
    assert declared.unreadable == ('x', 7)
    assert declared.is_empty is False


def test_a_bare_dict_value_is_one_unreadable_entry_never_iterated_by_key() -> None:
    declared = _read(_A)
    assert declared.checks == ()
    assert declared.unreadable == (_A,)
    assert declared.is_empty is False


@pytest.mark.parametrize('value', ['cap-one', 7], ids=['str', 'int'])
def test_a_truthy_scalar_value_is_one_unreadable_entry(value: object) -> None:
    declared = _read(value)
    assert declared.checks == ()
    assert declared.unreadable == (value,)
    assert declared.is_empty is False


def test_a_tuple_value_is_read_like_a_list() -> None:
    declared = _read(('x', _A, _B))
    assert declared.checks == (_A, _B)
    assert declared.unreadable == ('x',)


def test_the_rule_is_shape_only_an_empty_mapping_is_still_a_check() -> None:
    declared = _read([{}])
    assert declared.checks == ({},)
    assert declared.unreadable == ()
    assert declared.is_empty is False


def test_declared_checks_is_frozen() -> None:
    mutable: Any = _read([_A])
    with pytest.raises(dataclasses.FrozenInstanceError):
        mutable.checks = ()


def test_a_list_of_only_unreadable_entries_is_not_empty() -> None:
    declared = _read(['x', None, 3])
    assert declared.checks == ()
    assert declared.unreadable == ('x', None, 3)
    assert declared.is_empty is False
