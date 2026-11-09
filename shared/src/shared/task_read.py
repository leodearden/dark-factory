"""The typed outcome of one fused-memory ``get_task`` read.

A caller that must tell "no such task" from "could not read the task" branches
on a type instead of on a ``None``:

* :class:`TaskFound` — the server returned the task row, exactly as sent.
* :class:`TaskAbsent` — the server answered that no such task exists.
* :class:`TaskReadFailed` — any other error reply, or a response that could
  not be read; ``reason`` carries the server's own text when it gave one.

Lives in ``shared`` because ``escalation`` consumes it and must not import
``orchestrator`` statically.
"""

from __future__ import annotations

from dataclasses import dataclass
from typing import Any

from shared.mcp_envelope import parse_tool_result

__all__ = [
    'TASK_NOT_FOUND_ERROR_TYPE',
    'TaskAbsent',
    'TaskFound',
    'TaskRead',
    'TaskReadFailed',
    'parse_get_task_result',
]

# fused-memory/src/fused_memory/backends/task_backend_errors.py::TaskNotFoundError:
# a successful zero-row query, the one error reply that is an answer.
TASK_NOT_FOUND_ERROR_TYPE = 'TaskNotFoundError'


@dataclass(frozen=True)
class TaskFound:
    task: dict[str, Any]


@dataclass(frozen=True)
class TaskAbsent:
    pass


@dataclass(frozen=True)
class TaskReadFailed:
    reason: str


TaskRead = TaskFound | TaskAbsent | TaskReadFailed


def parse_get_task_result(result: Any) -> TaskRead:
    """Classify a ``dispatch_tool('get_task', ...)`` response.  Never raises.

    *result* is the JSON-RPC body the transport returns; the payload is read
    through ``shared/src/shared/mcp_envelope.py::parse_tool_result`` in
    whole-inner-dict mode, so an ``isError`` reply is never a value.
    """
    inner, err = parse_tool_result(result, None, dict)
    if err is not None:
        return TaskReadFailed(f'unreadable get_task response ({err.shape}): {err.payload_prefix}')
    assert inner is not None  # parse_tool_result invariant: (None, err) | (value, None)
    if 'error' in inner:
        if inner.get('error_type') == TASK_NOT_FOUND_ERROR_TYPE:
            return TaskAbsent()
        return TaskReadFailed(
            f'get_task error={inner.get("error")!r}, error_type={inner.get("error_type")!r}'
        )
    return TaskFound(inner)
