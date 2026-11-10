"""Tests for ``shared.mcp_unknown_arguments`` — the undeclared-argument refusal and its reader."""
from __future__ import annotations

import json

from shared.mcp_unknown_arguments import (
    UNKNOWN_ARGUMENT_CODE,
    refused_arguments,
    unknown_argument_refusal,
)

_DECLARED = ('id', 'project_root', 'prompt', 'metadata', 'append', 'metadata_mode', 'title')


def test_every_declared_name_is_no_refusal() -> None:
    assert unknown_argument_refusal('update_task', _DECLARED, {'id': '1', 'metadata': {}}) is None
    assert unknown_argument_refusal('update_task', _DECLARED, {}) is None


def test_a_misnamed_argument_is_refused_naming_it_and_its_closest_declared_name() -> None:
    arguments = {'id': '7', 'project_root': '/p', 'metadata_patch': {'k': 'v'}}

    refusal = unknown_argument_refusal('update_task', _DECLARED, arguments)

    assert refusal is not None
    assert refusal['code'] == UNKNOWN_ARGUMENT_CODE
    assert refusal['tool'] == 'update_task'
    assert refusal['unknown'] == [{'name': 'metadata_patch', 'closest_declared': 'metadata'}]
    assert refusal['declared'] == list(_DECLARED)
    assert refusal['provided'] == ['id', 'project_root', 'metadata_patch']
    assert 'repaired_call' not in refusal
    assert 'metadata_patch' in refusal['error']
    assert 'nothing was executed' in refusal['hint']
    json.dumps(refusal)


def test_a_name_with_no_close_declared_name_has_no_closest_declared() -> None:
    refusal = unknown_argument_refusal('update_task', _DECLARED, {'id': '7', 'zzzzzz': 1})

    assert refusal is not None
    assert refusal['unknown'] == [{'name': 'zzzzzz', 'closest_declared': None}]


def test_the_reader_recovers_the_refused_names_from_the_wire_text() -> None:
    refusal = unknown_argument_refusal('find', ('key', 'value'), {'key': 'k', 'include_rows': True})

    assert refused_arguments(json.dumps(refusal)) == frozenset({'include_rows'})


def test_the_reader_finds_nothing_in_any_other_failure_text() -> None:
    assert refused_arguments('include_rows\n  Extra inputs are not permitted') == frozenset()
    assert refused_arguments(json.dumps({'code': 'missing_required_argument'})) == frozenset()
    assert refused_arguments(json.dumps(['unknown_argument'])) == frozenset()
    assert refused_arguments(json.dumps({'code': UNKNOWN_ARGUMENT_CODE, 'unknown': 'x'})) == frozenset()
