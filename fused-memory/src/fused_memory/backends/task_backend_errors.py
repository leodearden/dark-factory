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


__all__ = [
    'TASKMASTER_TOOL_ERROR',
    'TASKMASTER_UNAVAILABLE',
    'TaskmasterError',
]
