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


class TaskNotFoundError(TaskmasterError):
    """Raised by ``get_task`` when a successful zero-row query proves absence.

    Marks a DEFINITIVE (project, tag, id) not-found — the query executed
    successfully and returned no row — as distinct from a connect/execute
    outage (``TASKMASTER_UNAVAILABLE`` / a raw driver exception raised before
    the query even runs). Callers that need to tell "task was deleted" apart
    from "backend is having a moment" can branch on
    ``isinstance(err, TaskNotFoundError)``.

    Keeps ``code='TASKMASTER_TOOL_ERROR'`` and the verbatim
    ``'No tasks found for ID(s): {task_id}'`` message byte-identical to the
    generic raise it replaces, so discrimination is by exception TYPE only —
    every existing ``except TaskmasterError`` site, ``err.code ==
    'TASKMASTER_TOOL_ERROR'`` branch, and not-found message-phrase matcher
    (e.g. ``flag_dedup.confirm_task_absent``) keeps working unchanged.

    Attributes:
        task_id: The id that was queried and not found.
        tag: The tag the lookup was scoped to (``None`` if not supplied),
            documenting the (project, tag) absence scope.
    """

    def __init__(self, task_id: str, tag: str | None = None, raw: Any = None) -> None:
        self.task_id = task_id
        self.tag = tag
        super().__init__(
            'TASKMASTER_TOOL_ERROR', f'No tasks found for ID(s): {task_id}', raw=raw,
        )


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


class AppendUnsupportedFieldError(TaskmasterError):
    """Raised when ``update_task`` combines ``append=True`` with a REPLACE-ONLY column.

    ``title``, ``description`` and ``priority`` can only ever be REPLACED —
    ``append`` has never governed them; it governs only the ``details`` /
    ``prompt`` concatenation and the metadata merge mode. Before task 4039
    the pair was accepted silently and the incoming text OVERWROTE the
    column, destroying multi-KB authored prose with no error and no warning.

    This mirrors the loud-over-silent guards
    ``sqlite_task_backend.py::_resolve_metadata_mode`` already applies to the
    two destructive metadata flag combinations (the task-2180 metadata-wipe
    and the task-3581 nested-metadata clobber) — same method, same shape of
    destructive combination, same principle — for the text columns that
    cannot append at all.

    Keeps ``code='TASKMASTER_TOOL_ERROR'`` so every existing ``except
    TaskmasterError`` site and ``err.code == 'TASKMASTER_TOOL_ERROR'`` branch
    keeps working unchanged (the same compatibility trick
    :class:`TaskNotFoundError` uses); the subclass exists for ``isinstance``
    discrimination and so the MCP boundary reports a specific
    ``error_type='AppendUnsupportedFieldError'``. Deliberately has no
    ``to_error_dict()`` — unlike the two write-authority errors above it has
    no historical ``success: False`` wire shape to reproduce, and it reaches
    callers by propagating raw to ``server/tool_errors.py::mcp_tool_errors``.

    Attributes:
        fields: The offending replace-only field name(s), in
            ``sqlite_task_backend.py::_REPLACE_ONLY_FIELDS`` order.
        task_id: The task the rejected write targeted (``None`` if not supplied).
    """

    def __init__(self, fields: tuple[str, ...], task_id: str | None = None) -> None:
        self.fields = fields
        self.task_id = task_id
        named = ', '.join(fields)
        plural = 's' if len(fields) > 1 else ''
        super().__init__(
            'TASKMASTER_TOOL_ERROR',
            f'Refusing an append=True write to {named}: append has NEVER applied '
            f'to the {named} column{plural} and never will — these columns are '
            'REPLACE-ONLY, so this write would OVERWRITE the current value, not '
            'concatenate onto it. Accepting it silently destroyed authored prose '
            'in four recorded live repros, including reify task 6586 and reify '
            'task 5791, the latter wiping ~17KB of a human-ratified decomposition '
            'record that existed nowhere else. To EXTEND the field, read-modify-'
            'write: call get_task to read the current text, concatenate locally, '
            f'then resend the COMPLETE new {named} with append omitted. To '
            'REPLACE it deliberately, drop append (or pass append=False) to '
            'confirm the overwrite. If the append=True was meant for details/'
            'prompt/metadata, split it into a separate update_task call.',
        )


__all__ = [
    'TASKMASTER_TOOL_ERROR',
    'TASKMASTER_UNAVAILABLE',
    'DuplicateCandidateKeyError',
    'TaskmasterError',
    'TaskNotFoundError',
    'StatusWriteAuthorityError',
    'DoneProvenanceWriteAuthorityError',
    'AppendUnsupportedFieldError',
    'status_via_update_task_error',
    'done_provenance_via_update_task_error',
]
