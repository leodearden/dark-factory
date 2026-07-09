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


__all__ = [
    'TASKMASTER_TOOL_ERROR',
    'TASKMASTER_UNAVAILABLE',
    'DuplicateCandidateKeyError',
    'TaskmasterError',
]
