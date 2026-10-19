"""recut_ruling — a human's "re-cut fresh off main" ruling, enforced at provisioning.

A re-cut ruling says that a task's existing branch, lane and session state
predate a human decision and must not be resumed: the next dispatch parks
that prior incarnation aside and starts on a branch freshly cut from main.
The marker lives in the TASK record's metadata under ``recut_ruling``, since
a ruling is a human-issued, task-lifetime fact. It is satisfied only by the
explicit ``recut_executed`` record naming the same ruling id. Provenance:
esc-2896-8, task 3154.

Dependency-light (stdlib + ``shared.task_metadata_wire``), like
``cross_repo_gate``: git_ops, harness and workflow import this module, never
the reverse.
"""

from __future__ import annotations

import logging
from dataclasses import dataclass
from datetime import UTC, datetime
from enum import Enum
from typing import Any

from shared.task_metadata_wire import coerce_task_metadata

logger = logging.getLogger(__name__)

RECUT_RULING_KEY = 'recut_ruling'
RECUT_EXECUTED_KEY = 'recut_executed'


@dataclass(frozen=True)
class RecutRuling:
    ruling_id: str
    ruled_at: datetime


@dataclass(frozen=True)
class ParkedIncarnation:
    """What parking moved aside; a field is None or empty when nothing existed."""

    parked_branch: str | None
    parked_tip: str | None
    quarantined_worktree: str | None
    relocated_meta_roots: tuple[str, ...]


_NOTHING_PARKED = ParkedIncarnation(None, None, None, ())


class RecutState(Enum):
    ABSENT = 'absent'
    SKIP = 'skip'
    PENDING = 'pending'
    SATISFIED = 'satisfied'
    MALFORMED = 'malformed'


@dataclass(frozen=True)
class RecutReading:
    state: RecutState
    ruling: RecutRuling | None = None
    problem: str = ''


@dataclass(frozen=True)
class _Malformed:
    problem: str


class RecutRulingUnreadable(RuntimeError):
    """A ``recut_ruling`` or ``recut_executed`` value that cannot be parsed."""


class RecutRulingViolation(RuntimeError):
    """A pending ruling met evidence of a prior incarnation."""

    def __init__(self, ruling: RecutRuling, evidence: tuple[str, ...]) -> None:
        self.ruling = ruling
        self.evidence = evidence
        super().__init__(
            f're-cut ruling {ruling.ruling_id} (ruled {ruling.ruled_at.isoformat()}) '
            f'is pending but a prior incarnation remains: {"; ".join(evidence)}'
        )


def carries_recut_ruling(task: dict[str, Any]) -> bool:
    """Key presence, not validity: a malformed or unreadable marker must
    enter the gate and fail loudly there, not skip it."""
    metadata = coerce_task_metadata(task.get('metadata'))
    return metadata is None or RECUT_RULING_KEY in metadata


def read_recut_ruling(task: dict[str, Any]) -> RecutReading:
    metadata = coerce_task_metadata(task.get('metadata'))
    if metadata is None:
        logger.warning(
            'recut_ruling: task %s metadata is unreadable; no ruling was read, '
            'so this is not a clean bill',
            task.get('id'),
        )
        return RecutReading(RecutState.SKIP)
    if RECUT_RULING_KEY not in metadata:
        return RecutReading(RecutState.ABSENT)
    ruling = _parse_ruling(metadata[RECUT_RULING_KEY])
    if isinstance(ruling, str):
        return RecutReading(RecutState.MALFORMED, problem=ruling)
    executed_id = _executed_ruling_id(metadata)
    if isinstance(executed_id, _Malformed):
        return RecutReading(RecutState.MALFORMED, ruling, executed_id.problem)
    if executed_id == ruling.ruling_id:
        return RecutReading(RecutState.SATISFIED, ruling)
    return RecutReading(RecutState.PENDING, ruling)


def ruling_to_enforce(task: dict[str, Any]) -> RecutRuling | None:
    """The ruling still to execute, or None; raises when it cannot be read."""
    reading = read_recut_ruling(task)
    if reading.state is RecutState.MALFORMED:
        raise RecutRulingUnreadable(f'task {task.get("id")}: {reading.problem}')
    return reading.ruling if reading.state is RecutState.PENDING else None


def execution_record(
    ruling: RecutRuling,
    parked: ParkedIncarnation | None,
    executed_at: datetime,
) -> dict[str, Any]:
    """The ``recut_executed`` value that marks *ruling* satisfied."""
    parked = parked or _NOTHING_PARKED
    return {
        'ruling_id': ruling.ruling_id,
        'ruled_at': ruling.ruled_at.isoformat(),
        'executed_at': executed_at.isoformat(),
        'parked_branch': parked.parked_branch,
        'parked_tip': parked.parked_tip,
        'quarantined_worktree': parked.quarantined_worktree,
        'relocated_meta_roots': list(parked.relocated_meta_roots),
    }


def _parse_ruling(value: object) -> RecutRuling | str:
    if not isinstance(value, dict):
        return f'{RECUT_RULING_KEY} must be an object; got {value!r}'
    ruling_id = value.get('ruling_id')
    if not isinstance(ruling_id, str) or not ruling_id.strip():
        return f'{RECUT_RULING_KEY}.ruling_id must be a non-blank string; got {ruling_id!r}'
    ruled_at = _parse_ruled_at(value.get('ruled_at'))
    if ruled_at is None:
        return (
            f'{RECUT_RULING_KEY}.ruled_at must be an ISO-8601 date or timestamp; '
            f'got {value.get("ruled_at")!r}'
        )
    return RecutRuling(ruling_id=ruling_id, ruled_at=ruled_at)


def _parse_ruled_at(raw: object) -> datetime | None:
    if not isinstance(raw, str):
        return None
    try:
        parsed = datetime.fromisoformat(raw)
    except ValueError:
        return None
    return parsed if parsed.tzinfo is not None else parsed.replace(tzinfo=UTC)


def _executed_ruling_id(metadata: dict[str, Any]) -> str | None | _Malformed:
    if RECUT_EXECUTED_KEY not in metadata:
        return None
    record = metadata[RECUT_EXECUTED_KEY]
    ruling_id = record.get('ruling_id') if isinstance(record, dict) else None
    if not isinstance(ruling_id, str) or not ruling_id.strip():
        return _Malformed(
            f'{RECUT_EXECUTED_KEY} must be an object with a non-blank string '
            f'ruling_id; got {record!r}'
        )
    return ruling_id
