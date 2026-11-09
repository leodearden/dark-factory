"""shared.json_safe: a refusal echoes the caller's value, or its repr when JSON cannot carry it."""

from __future__ import annotations

import json

import pytest

from shared.json_safe import json_safe


@pytest.mark.parametrize(
    'value',
    [None, 0, 'text', [1, 'a'], {'k': [1.5]}],
    ids=['none', 'int', 'str', 'list', 'dict'],
)
def test_a_value_json_carries_is_returned_unchanged(value):
    assert json_safe(value) is value


@pytest.mark.parametrize(
    'value',
    [{1, 2}, object(), b'bytes', {'k': {3}}],
    ids=['set', 'object', 'bytes', 'nested-set'],
)
def test_a_value_json_cannot_carry_becomes_its_repr(value):
    safe = json_safe(value)

    assert safe == repr(value)
    assert json.dumps(safe)


def test_a_circular_value_becomes_its_repr():
    looped: list[object] = []
    looped.append(looped)

    assert json_safe(looped) == repr(looped)
