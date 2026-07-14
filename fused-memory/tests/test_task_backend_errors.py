"""Unit tests for the canonical write-authority rejection shapes + typed errors.

Pure functions/classes — no async fixtures needed. These pin the exact wire
shape (including verbatim hint strings) that ``task_interceptor.py`` and
``server/tools.py`` historically duplicated, so C2 can delete those copies
without breaking callers that branch on ``error == '...'``.
"""

from __future__ import annotations

from fused_memory.backends.task_backend_errors import (
    DoneProvenanceWriteAuthorityError,
    StatusWriteAuthorityError,
    TaskmasterError,
    TaskNotFoundError,
    done_provenance_via_update_task_error,
    status_via_update_task_error,
)

_STATUS_HINT = (
    'update_task is metadata-only. Use '
    'set_task_status(status=…, done_provenance={...} when '
    'status="done") to change status — it enforces the '
    'terminal-exit, phantom-done, and done-provenance gates.'
)

_DONE_PROVENANCE_HINT = (
    'update_task cannot write metadata.done_provenance. Use '
    'set_task_status(status="done", done_provenance={...}) instead — '
    'it validates the kind/commit/note schema and runs an ancestor '
    'backstop on the merge sha.'
)


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
    assert done_provenance_via_update_task_error('7') == {
        'success': False,
        'error': 'done_provenance_via_update_task',
        'task_id': '7',
        'hint': _DONE_PROVENANCE_HINT,
    }


def test_done_provenance_error_shape_has_no_status_key():
    assert 'status' not in done_provenance_via_update_task_error('7')


# ── Typed errors — to_error_dict() round-trips to the canonical shape ──


def test_status_write_authority_error_to_error_dict_matches_canonical():
    err = StatusWriteAuthorityError('7', 'done')
    assert err.to_error_dict() == status_via_update_task_error('7', 'done')


def test_done_provenance_write_authority_error_to_error_dict_matches_canonical():
    err = DoneProvenanceWriteAuthorityError('7')
    assert err.to_error_dict() == done_provenance_via_update_task_error('7')


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
