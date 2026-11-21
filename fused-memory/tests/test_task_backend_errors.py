"""Unit tests for the canonical write-authority rejection shapes + typed errors.

Pure functions/classes — no async fixtures needed. These pin the exact wire
shape that ``task_interceptor.py`` and ``server/tools.py`` historically
duplicated, so C2 can delete those copies without breaking callers that branch
on ``error == '...'``. The status hint is pinned verbatim; the done_provenance
hint is pinned by its load-bearing phrases, since it is prose that may grow.
"""

from __future__ import annotations

import pytest

from fused_memory.backends.task_backend_errors import (
    DependencyCycle,
    DoneProvenanceWriteAuthorityError,
    MetadataNotAnObjectError,
    StatusWriteAuthorityError,
    TaskmasterError,
    TaskNotFoundError,
    done_provenance_via_update_task_error,
    metadata_not_an_object_error,
    status_via_update_task_error,
)

_STATUS_HINT = (
    'update_task is metadata-only. Use '
    'set_task_status(status=…, done_provenance={...} when '
    'status="done") to change status — it enforces the '
    'terminal-exit, phantom-done, and done-provenance gates.'
)

_SET_TASK_STATUS_GUIDANCE = 'set_task_status(status="done", done_provenance={...})'


def _assert_carries_done_provenance_guidance(text: str) -> None:
    assert _SET_TASK_STATUS_GUIDANCE in text
    assert "metadata_mode='replace'" in text
    assert 'verbatim' in text


# ── Canonical dict shapes ───────────────────────────────────────────


def test_status_via_update_task_error_shape():
    assert status_via_update_task_error('7', 'done') == {
        'success': False,
        'error': 'status_via_update_task',
        'task_id': '7',
        'status': 'done',
        'hint': _STATUS_HINT,
    }


def test_done_provenance_via_update_task_error_shape():
    """The wire contract callers branch on is unchanged; only the prose hint
    grew, and it must tell a whole-blob replace caller what to do."""
    error = done_provenance_via_update_task_error('7')
    assert {key: value for key, value in error.items() if key != 'hint'} == {
        'success': False,
        'error': 'done_provenance_via_update_task',
        'task_id': '7',
    }
    _assert_carries_done_provenance_guidance(error['hint'])


def test_done_provenance_error_shape_has_no_status_key():
    assert 'status' not in done_provenance_via_update_task_error('7')


# ── Typed errors — to_error_dict() round-trips to the canonical shape ──


def test_status_write_authority_error_to_error_dict_matches_canonical():
    err = StatusWriteAuthorityError('7', 'done')
    assert err.to_error_dict() == status_via_update_task_error('7', 'done')


def test_done_provenance_write_authority_error_to_error_dict_matches_canonical():
    err = DoneProvenanceWriteAuthorityError('7')
    assert err.to_error_dict() == done_provenance_via_update_task_error('7')
    _assert_carries_done_provenance_guidance(err.to_error_dict()['hint'])
    _assert_carries_done_provenance_guidance(err.message)


# ── Typed errors — backward-compat: IS-A TaskmasterError, same code/message ──


def test_status_write_authority_error_is_taskmaster_error():
    err = StatusWriteAuthorityError('7', 'done')
    assert isinstance(err, TaskmasterError)
    assert err.code == 'TASKMASTER_TOOL_ERROR'
    assert 'set_task_status' in err.message


def test_done_provenance_write_authority_error_is_taskmaster_error():
    err = DoneProvenanceWriteAuthorityError('7')
    assert isinstance(err, TaskmasterError)
    assert err.code == 'TASKMASTER_TOOL_ERROR'
    assert 'set_task_status' in err.message


# ── MetadataNotAnObjectError — non-object metadata payload (task 6130) ──


def test_metadata_not_an_object_error_shape():
    error = metadata_not_an_object_error('7', 'array')
    assert {key: value for key, value in error.items() if key != 'hint'} == {
        'success': False,
        'error': 'metadata_not_an_object',
        'task_id': '7',
        'received': 'array',
    }
    assert isinstance(error['hint'], str)


def test_metadata_not_an_object_error_is_taskmaster_error():
    err = MetadataNotAnObjectError('7', 'null')
    assert isinstance(err, TaskmasterError)
    assert err.code == 'TASKMASTER_TOOL_ERROR'
    assert err.task_id == '7'
    assert err.received == 'null'
    assert 'null' in err.message


def test_metadata_not_an_object_error_to_error_dict_matches_canonical():
    err = MetadataNotAnObjectError('7', 'invalid_json')
    assert err.to_error_dict() == metadata_not_an_object_error('7', 'invalid_json')


def test_metadata_not_an_object_error_is_not_a_done_provenance_error():
    """A distinct refusal with a distinct error code, not a variant of the
    done_provenance floor: it applies to rows that hold no done_provenance."""
    assert not isinstance(
        MetadataNotAnObjectError('7', 'array'), DoneProvenanceWriteAuthorityError,
    )


# ── TaskNotFoundError — definitive zero-row absence (task 2521 RC2) ──────


def test_task_not_found_error_is_taskmaster_error():
    """isinstance(err, TaskmasterError) so existing `except TaskmasterError`
    sites keep catching it unchanged — discrimination is by type only."""
    err = TaskNotFoundError('7', tag='master')
    assert isinstance(err, TaskmasterError)


def test_task_not_found_error_code_and_message_unchanged():
    """code and message stay byte-identical to the generic not-found raise
    so code-branch and message-phrase consumers keep working."""
    err = TaskNotFoundError('7', tag='master')
    assert err.code == 'TASKMASTER_TOOL_ERROR'
    assert 'No tasks found for ID(s): 7' in err.message
    assert 'No tasks found for ID(s): 7' in str(err)


def test_task_not_found_error_attributes():
    """Carries task_id and tag so the (id, tag) absence scope is self-documenting."""
    err = TaskNotFoundError('7', tag='master')
    assert err.task_id == '7'
    assert err.tag == 'master'


def test_task_not_found_error_tag_defaults_to_none():
    err = TaskNotFoundError('7')
    assert err.task_id == '7'
    assert err.tag is None


# ── DependencyCycle — a refused cycle-closing edge (task 9490) ──────────────

_CYCLE_MESSAGE = (
    'add_dependency: 9301 → 9288 would close a cycle 9301 → 9288 → 9290 → 9301; '
    'remove one of those edges first'
)


def test_dependency_cycle_is_a_taskmaster_error_naming_the_path():
    err = DependencyCycle('add_dependency', (9301, 9288, 9290, 9301))

    assert isinstance(err, TaskmasterError)
    assert err.code == 'DEPENDENCY_CYCLE'
    assert err.cycle == (9301, 9288, 9290, 9301)
    assert err.message == _CYCLE_MESSAGE


def test_dependency_cycle_wire_shape_is_json_native():
    err = DependencyCycle('add_dependency', [9301, 9288, 9290, 9301])

    assert err.to_error_dict() == {
        'success': False,
        'error': 'dependency_cycle',
        'error_type': 'DependencyCycle',
        'operation': 'add_dependency',
        'task_id': 9301,
        'depends_on': 9288,
        'cycle': [9301, 9288, 9290, 9301],
        'hint': _CYCLE_MESSAGE,
    }


@pytest.mark.parametrize(
    ('cycle', 'invariant'),
    [
        pytest.param((), 'at least two', id='empty'),
        pytest.param((9301,), 'at least two', id='one-entry'),
        pytest.param((9301, 9288, 9290), 'start and end', id='open-path'),
    ],
)
def test_dependency_cycle_refuses_a_malformed_path(cycle: tuple[int, ...], invariant: str):
    with pytest.raises(ValueError, match=invariant) as caught:
        DependencyCycle('update_task', cycle)

    assert repr(cycle) in str(caught.value)
