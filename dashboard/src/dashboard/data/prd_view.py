"""The per-PRD view of one root: segments, readiness and attribution, composed.

PURE (PRD ``plans/dashboard-prd-lanes-prd.md`` §4.4–§4.5, §5.5): no I/O and
no clock read but the injected *now*. It reads one root's status map, its raw
active rows and its lineage index, handed in by the caller, and never imports
the modules that acquire them.
"""

from __future__ import annotations

import enum
from collections.abc import Mapping
from datetime import datetime
from types import MappingProxyType
from typing import Any

from shared.task_metadata import Milestone
from shared.task_statuses import TaskStatus

from dashboard.data.census import VIEWS, TaskView


class Segment(enum.StrEnum):
    """Where a task sits in its PRD's bar; declaration order is display order."""

    DONE = 'done'
    IN_PROGRESS = 'in_progress'
    BLOCKED = 'blocked'
    DEFERRED = 'deferred'
    READY = 'ready'
    WAITING = 'waiting'


SEGMENT_TONES: MappingProxyType[Segment, str] = MappingProxyType({
    Segment.DONE: 'ok',
    Segment.IN_PROGRESS: 'accent',
    Segment.BLOCKED: 'bad',
    Segment.DEFERRED: 'fg3',
    Segment.READY: 'warn',
    Segment.WAITING: 'info',
})

SEGMENT_LABELS: MappingProxyType[Segment, str] = MappingProxyType({
    Segment.DONE: 'done',
    Segment.IN_PROGRESS: 'in progress',
    Segment.BLOCKED: 'blocked',
    Segment.DEFERRED: 'deferred',
    Segment.READY: 'ready',
    Segment.WAITING: 'waiting',
})

_IN_PROGRESS_STATUSES = VIEWS[TaskView.IN_FLIGHT] - {TaskStatus.BLOCKED}
_STATUS_SEGMENTS: Mapping[str, Segment] = MappingProxyType({
    TaskStatus.DONE: Segment.DONE,
    TaskStatus.BLOCKED: Segment.BLOCKED,
    TaskStatus.DEFERRED: Segment.DEFERRED,
})
_SATISFIED_DEPENDENCY_STATUSES = frozenset({TaskStatus.DONE, TaskStatus.CANCELLED})
_MILESTONE_ANCHOR_KEY = 'milestone_deps_satisfied_at'


def is_human_gate(metadata: Mapping[str, Any]) -> bool:
    """A deterministic task that always escalates: it waits on a human, never a dispatch."""
    return metadata.get('task_kind') == 'deterministic' and metadata.get('always_escalates') is True


def is_ready(
    task_id: int,
    row: Mapping[str, Any] | None,
    status_map: Mapping[int, str],
    *,
    now: datetime,
) -> bool:
    """Whether pending *task_id* could be dispatched at *now*, as far as its row says.

    Ready means pending with an active row, every dependency known and done or
    cancelled, no external dependency, no unfired milestone, and not a human
    gate. It does not evaluate delivered checks on dependencies, pins, parks,
    requeue cool-downs or external statuses, so it is an upper bound on what
    the scheduler would dispatch.
    """
    if status_map.get(task_id) != TaskStatus.PENDING or row is None:
        return False
    metadata = row.get('metadata') or {}
    dependencies = row.get('dependencies') or ()
    external_deps = metadata.get('external_deps')
    return (
        all(status_map.get(dep) in _SATISFIED_DEPENDENCY_STATUSES for dep in dependencies)
        and not (isinstance(external_deps, list) and external_deps)
        and not _milestone_unfired(metadata, now)
        and not is_human_gate(metadata)
    )


def segment_of(
    task_id: int,
    status: str,
    row: Mapping[str, Any] | None,
    status_map: Mapping[int, str],
    *,
    now: datetime,
) -> Segment | None:
    """*task_id*'s segment; ``None`` for a cancelled task, which no bar counts."""
    if status == TaskStatus.PENDING:
        return Segment.READY if is_ready(task_id, row, status_map, now=now) else Segment.WAITING
    if status in _IN_PROGRESS_STATUSES:
        return Segment.IN_PROGRESS
    return _STATUS_SEGMENTS.get(status)


def _milestone_unfired(metadata: Mapping[str, Any], now: datetime) -> bool:
    """A milestone that has not fired, where a malformed one never fires."""
    spec = metadata.get('milestone')
    if not spec:
        return False
    anchor = metadata.get(_MILESTONE_ANCHOR_KEY)
    if anchor is not None and not isinstance(anchor, str):
        return True
    try:
        return not Milestone.model_validate(spec).fired(anchor, now)
    except ValueError:
        return True
