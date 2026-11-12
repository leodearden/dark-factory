"""The ``get_tasks(fields=...)`` row projection, in one place.

A projection names top-level task keys (``TASK_ROW_KEYS``, the wire shape
``fused_memory/backends/sqlite_task_backend.py::_row_to_task`` emits, in its
spelling and order) and ``metadata.<key>`` entries. ``id`` is always kept. A
requested metadata key absent on a task, or a requested
``SCHEMA_OPTIONAL_ROW_KEYS`` entry a store whose v6 ALTER never landed cannot
emit, is omitted rather than nulled; every other row key is always emitted, so
its absence is a ``KeyError``.
``SqliteTaskBackend.get_tasks`` parses before it reads and applies after
``_row_to_task``'s coercion; the MCP tool parses the same argument at its
boundary for the INV-2 error payload. The contract is
``plans/dashboard-prd-lanes-prd.md`` §5.1.
"""

from __future__ import annotations

import re
from collections.abc import Mapping
from dataclasses import dataclass
from typing import Any

TASK_ROW_KEYS: tuple[str, ...] = (
    'id',
    'title',
    'description',
    'details',
    'testStrategy',
    'status',
    'dependencies',
    'priority',
    'subtasks',
    'updatedAt',
    'metadata',
    'claimant_run_id',
    'heartbeat_at',
    'candidate_key',
    'created_at',
    'created_at_source',
)
SCHEMA_OPTIONAL_ROW_KEYS: tuple[str, ...] = ('created_at', 'created_at_source')
METADATA_FIELD_PREFIX = 'metadata.'
PROJECTION_METADATA_KEY_PATTERN = re.compile(r'[A-Za-z_][A-Za-z0-9_]*')
ALLOWED_FIELDS: tuple[str, ...] = (*TASK_ROW_KEYS, f'{METADATA_FIELD_PREFIX}<key>')


class InvalidTaskProjectionField(ValueError):
    """A ``fields`` entry, or the whole argument, names nothing a task row can project."""

    def __init__(self, field: object) -> None:
        super().__init__(
            f'cannot project {field!r}: fields must be a list of entries drawn from '
            f'{list(ALLOWED_FIELDS)}, where <key> matches '
            f'{PROJECTION_METADATA_KEY_PATTERN.pattern}',
        )
        self.field = field
        self.allowed = ALLOWED_FIELDS


@dataclass(frozen=True)
class TaskProjection:
    row_keys: tuple[str, ...]
    metadata_keys: tuple[str, ...]

    def apply(self, task: Mapping[str, Any]) -> dict[str, Any]:
        projected = {
            key: task[key]
            for key in self.row_keys
            if key in task or key not in SCHEMA_OPTIONAL_ROW_KEYS
        }
        if self.metadata_keys:
            metadata = task['metadata']
            projected['metadata'] = {
                key: metadata[key] for key in self.metadata_keys if key in metadata
            }
        return projected


def _metadata_key(entry: str) -> str | None:
    if not entry.startswith(METADATA_FIELD_PREFIX):
        return None
    key = entry.removeprefix(METADATA_FIELD_PREFIX)
    return key if PROJECTION_METADATA_KEY_PATTERN.fullmatch(key) else None


def parse_task_projection(fields: object) -> TaskProjection | None:
    """``None`` for no projection; otherwise the projection ``fields`` names.

    Raises:
        InvalidTaskProjectionField: ``fields`` is not a list or tuple, or one
            of its entries is neither a row key nor ``metadata.<key>``.
    """
    if fields is None:
        return None
    if not isinstance(fields, (list, tuple)):
        raise InvalidTaskProjectionField(fields)
    requested_rows = {'id'}
    requested_metadata: dict[str, None] = {}
    for entry in fields:
        if not isinstance(entry, str):
            raise InvalidTaskProjectionField(entry)
        if entry in TASK_ROW_KEYS:
            requested_rows.add(entry)
        elif (key := _metadata_key(entry)) is not None:
            requested_metadata[key] = None
        else:
            raise InvalidTaskProjectionField(entry)
    whole_metadata = 'metadata' in requested_rows
    return TaskProjection(
        row_keys=tuple(key for key in TASK_ROW_KEYS if key in requested_rows),
        metadata_keys=() if whole_metadata else tuple(requested_metadata),
    )
