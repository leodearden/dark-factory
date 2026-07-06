"""Error type + adapter-level error codes raised by every task backend.

Extracted from the now-deleted ``taskmaster_types`` module so the SQLite
backend (and any future task backend) can keep raising the same public
exception type without depending on the legacy Taskmaster MCP wrapper.

The two adapter-level codes are kept here so callers and tests that
``raise TaskmasterError(TASKMASTER_TOOL_ERROR, ...)`` keep their wire
contract intact across the cutover.
"""

from __future__ import annotations

from typing import Any

TASKMASTER_TOOL_ERROR: str = 'TASKMASTER_TOOL_ERROR'
TASKMASTER_UNAVAILABLE: str = 'TASKMASTER_UNAVAILABLE'


class TaskmasterError(Exception):
    """Raised when a task backend call fails or returns an unexpected shape.

    ``code`` is one of the adapter-level codes
    (``TASKMASTER_TOOL_ERROR`` / ``TASKMASTER_UNAVAILABLE``) or a
    backend-specific code propagated unchanged.

    ``raw`` preserves the underlying response for post-mortem diagnosis.
    """

    def __init__(self, code: str, message: str, raw: Any = None) -> None:
        super().__init__(f'{code}: {message}')
        self.code = code
        self.message = message
        self.raw = raw


class DuplicateCandidateKeyError(TaskmasterError):
    """Raised by ``add_task`` when the partial UNIQUE index on
    ``(tag, candidate_key)`` rejects an insert (fm-task-dedup W8 task A2).

    Names the surviving non-cancelled row so callers can resolve the
    collision as a combine/dedup rather than a hard failure: the
    interceptor's create-dispatch and planning-mode paths catch this
    explicitly and return a ``'combined'``-style result pointing at
    ``existing_id``.

    Attributes:
        existing_id: The surviving row's task id (``int``), or ``None`` if
            the post-collision lookup somehow found no matching row.
        existing_status: The surviving row's current status, or ``None``
            under the same fallback condition.
        tag: The tag the collision occurred under.
        candidate_key: The colliding ``candidate_key`` value.
    """

    def __init__(
        self,
        existing_id: int | None,
        existing_status: str | None = None,
        tag: str | None = None,
        candidate_key: str | None = None,
    ) -> None:
        super().__init__(
            'DUPLICATE_CANDIDATE_KEY',
            f'A task with the same normalized (title, files) already exists: '
            f'tag={tag!r} candidate_key={candidate_key!r} '
            f'existing_id={existing_id!r} existing_status={existing_status!r}',
        )
        self.existing_id = existing_id
        self.existing_status = existing_status
        self.tag = tag
        self.candidate_key = candidate_key


def status_via_update_task_error(task_id: str, status: object) -> dict[str, Any]:
    """Canonical rejection shape for ``update_task(status=…)`` calls.

    ``set_task_status`` is the only sanctioned writer for task status — it
    enforces the terminal-exit, phantom-done, and done-provenance gates.
    This dict is byte-identical to the historical ``success: False`` variant
    produced by ``task_interceptor.py``'s ``_reject_status_in_update_task``
    (NOT the divergent ``error_type: 'ValidationError'`` variant that lived
    in ``server/tools.py``), so callers branching on
    ``error == 'status_via_update_task'`` keep working across the cutover.
    """
    return {
        'success': False,
        'error': 'status_via_update_task',
        'task_id': task_id,
        'status': status,
        'hint': (
            'update_task is metadata-only. Use '
            'set_task_status(status=…, done_provenance={...} when '
            'status="done") to change status — it enforces the '
            'terminal-exit, phantom-done, and done-provenance gates.'
        ),
    }


def done_provenance_via_update_task_error(task_id: str) -> dict[str, Any]:
    """Canonical rejection shape for ``update_task`` calls writing ``metadata.done_provenance``.

    ``set_task_status`` is the only sanctioned writer for ``done_provenance``
    — it validates the kind/commit/note schema and runs an ancestor backstop
    on the merge sha. This dict is byte-identical to the historical
    ``success: False`` variant produced by ``task_interceptor.py``'s
    ``_reject_done_provenance_in_update_metadata``, so callers branching on
    ``error == 'done_provenance_via_update_task'`` keep working across the
    cutover.
    """
    return {
        'success': False,
        'error': 'done_provenance_via_update_task',
        'task_id': task_id,
        'hint': (
            'update_task cannot write metadata.done_provenance. Use '
            'set_task_status(status="done", done_provenance={...}) instead — '
            'it validates the kind/commit/note schema and runs an ancestor '
            'backstop on the merge sha.'
        ),
    }


class StatusWriteAuthorityError(TaskmasterError):
    """Raised when ``update_task`` is asked to write a non-None ``status``.

    ``update_task`` is metadata-only — ``set_task_status`` is the sole
    sanctioned writer for status (it enforces the terminal-exit,
    phantom-done, and done-provenance gates). Subclasses
    :class:`TaskmasterError` with the ``TASKMASTER_TOOL_ERROR`` code and a
    ``set_task_status``-mentioning message so existing ``TaskmasterError``
    catchers/assertions keep working unchanged; call :meth:`to_error_dict`
    for the canonical wire shape.
    """

    def __init__(self, task_id: str, status: object) -> None:
        self.task_id = task_id
        self.status = status
        super().__init__(
            'TASKMASTER_TOOL_ERROR',
            'update_task is metadata-only and cannot write status. '
            'Use set_task_status(status=…) instead — it enforces the '
            'terminal-exit, phantom-done, and done-provenance gates.',
        )

    def to_error_dict(self) -> dict[str, Any]:
        return status_via_update_task_error(self.task_id, self.status)


class DoneProvenanceWriteAuthorityError(TaskmasterError):
    """Raised when ``update_task`` is asked to write ``metadata.done_provenance``.

    ``set_task_status`` is the sole sanctioned writer for ``done_provenance``
    — it validates the kind/commit/note schema and runs an ancestor backstop
    on the merge sha. Subclasses :class:`TaskmasterError` with the
    ``TASKMASTER_TOOL_ERROR`` code and a ``set_task_status``-mentioning
    message so existing ``TaskmasterError`` catchers/assertions keep working
    unchanged; call :meth:`to_error_dict` for the canonical wire shape.
    """

    def __init__(self, task_id: str) -> None:
        self.task_id = task_id
        super().__init__(
            'TASKMASTER_TOOL_ERROR',
            'update_task cannot write metadata.done_provenance. Use '
            'set_task_status(status="done", done_provenance={...}) instead — '
            'it validates the kind/commit/note schema and runs an ancestor '
            'backstop on the merge sha.',
        )

    def to_error_dict(self) -> dict[str, Any]:
        return done_provenance_via_update_task_error(self.task_id)


__all__ = [
    'TASKMASTER_TOOL_ERROR',
    'TASKMASTER_UNAVAILABLE',
    'DuplicateCandidateKeyError',
    'TaskmasterError',
    'StatusWriteAuthorityError',
    'DoneProvenanceWriteAuthorityError',
    'status_via_update_task_error',
    'done_provenance_via_update_task_error',
]
