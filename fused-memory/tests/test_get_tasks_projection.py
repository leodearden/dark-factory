"""get_tasks(fields=...): the typed row projection of plans/dashboard-prd-lanes-prd.md §5.1.

Three faces of one projection: the pure ``task_projection`` module, the
in-process ``TaskInterceptor.get_tasks`` seam over a real SQLite store, and
the MCP tool.
"""

from __future__ import annotations

from typing import Any

import pytest

from fused_memory.backends.task_projection import (
    ALLOWED_FIELDS,
    TASK_ROW_KEYS,
    InvalidTaskProjectionField,
    TaskProjection,
    parse_task_projection,
)

_FULL_ROW: dict[str, Any] = {
    'id': '7',
    'title': 'a task',
    'description': 'what it does',
    'details': 'how',
    'testStrategy': 'check it',
    'status': 'pending',
    'dependencies': [3],
    'priority': 'medium',
    'subtasks': [],
    'updatedAt': '2026-10-22T00:00:00Z',
    'metadata': {'prd_path': 'plans/x.md', 'other': 1},
    'claimant_run_id': None,
    'heartbeat_at': None,
    'candidate_key': None,
}


def _project(fields: object, row: dict[str, Any] = _FULL_ROW) -> dict[str, Any]:
    projection = parse_task_projection(fields)
    assert isinstance(projection, TaskProjection)
    return projection.apply(row)


# ── Pure projection contract ────────────────────────────────────────────


def test_the_literal_row_carries_exactly_the_allow_listed_keys():
    assert tuple(_FULL_ROW) == TASK_ROW_KEYS


def test_none_means_no_projection():
    assert parse_task_projection(None) is None


def test_requested_metadata_keys_present_on_the_task_are_kept_and_absent_ones_omitted():
    projected = _project(['id', 'metadata.prd_path', 'metadata.spawned_from'])

    assert projected == {'id': '7', 'metadata': {'prd_path': 'plans/x.md'}}


def test_id_is_always_returned():
    assert _project(['status']) == {'id': '7', 'status': 'pending'}
    assert _project([]) == {'id': '7'}


def test_no_metadata_entry_means_no_metadata_key():
    assert 'metadata' not in _project(['title'])


def test_a_coerced_empty_metadata_projects_an_empty_dict():
    row = {**_FULL_ROW, 'metadata': {}}

    assert _project(['metadata.prd_path'], row) == {'id': '7', 'metadata': {}}


def test_whole_metadata_subsumes_its_sub_keys():
    projected = _project(['metadata', 'metadata.prd_path'])

    assert projected == {'id': '7', 'metadata': _FULL_ROW['metadata']}


def test_duplicate_entries_collapse():
    duplicated = _project(['id', 'id', 'metadata.prd_path', 'metadata.prd_path'])

    assert duplicated == _project(['id', 'metadata.prd_path'])


def test_every_row_key_is_accepted_from_a_list_or_a_tuple():
    assert _project(list(TASK_ROW_KEYS)) == _FULL_ROW
    assert _project(tuple(TASK_ROW_KEYS)) == _FULL_ROW


@pytest.mark.parametrize(
    'entry',
    ['foo', '', 'metadata.', 'metadata.a-b', 'metadata.a.b', 'metadata.1a', 5],
)
def test_an_invalid_entry_is_named_verbatim_with_the_allowed_set(entry):
    with pytest.raises(InvalidTaskProjectionField) as caught:
        parse_task_projection(['id', entry])

    assert caught.value.field == entry
    assert caught.value.allowed == ALLOWED_FIELDS
    assert repr(entry) in str(caught.value)


@pytest.mark.parametrize('argument', ['id', {'id': 1}])
def test_a_non_list_argument_is_named_whole(argument):
    with pytest.raises(InvalidTaskProjectionField) as caught:
        parse_task_projection(argument)

    assert caught.value.field == argument
    assert caught.value.allowed == ALLOWED_FIELDS


def test_the_allowed_set_is_every_row_key_plus_the_metadata_form():
    assert ALLOWED_FIELDS == (*TASK_ROW_KEYS, 'metadata.<key>')
