"""Facts read from the orchestrator's runs.db: a batched cursor over events.id, and stateless whole-store readers.

Each payload is built inside SQLite from its model's field names, so the `data` blob is never selected whole.
"""

from __future__ import annotations

import math
import re
from collections.abc import Callable, Iterable, Mapping
from dataclasses import dataclass
from datetime import UTC, datetime
from enum import StrEnum
from pathlib import Path
from types import MappingProxyType
from typing import Any, Final, Self, TypeVar

from pydantic import (
    BaseModel,
    ConfigDict,
    JsonValue,
    StrictBool,
    ValidationError,
    model_validator,
)
from shared.project_layout import runs_db

from client_feed.contract import (
    NOT_RECORDED,
    CommitSha,
    NonNegativeInt,
    NotRecorded,
    ReviewCounts,
    SchedulerState,
    Source,
    VerifyKind,
)
from client_feed.sources.store import read_only


class StoreEventType(StrEnum):
    """The producer's spellings this module reads: orchestrator/src/orchestrator/event_store.py::EventType."""

    MERGE_FINALIZED = 'merge_finalized'
    TRAIN_MERGED = 'train_merged'
    MERGE_VERIFY = 'merge_verify'
    WORKFLOW_VERIFY = 'workflow_verify'
    PHASE_ENTER = 'phase_enter'
    PHASE_EXIT = 'phase_exit'
    SCHEDULER_PAUSED = 'scheduler_paused'
    SCHEDULER_RESUMED = 'scheduler_resumed'
    SCHEDULER_PAUSE_RESTORED = 'scheduler_pause_restored'
    ESCALATION_CREATED = 'escalation_created'
    ESCALATION_RESOLVED = 'escalation_resolved'
    REVIEW_COMPLETED = 'review_completed'


class SkipReason(StrEnum):
    MALFORMED_PAYLOAD = 'malformed_payload'
    BAD_TIMESTAMP = 'bad_timestamp'


@dataclass(frozen=True)
class LandedCommit:
    event_id: int
    at: datetime
    commit: str
    task_ids: tuple[int, ...]


@dataclass(frozen=True)
class VerifyVerdict:
    event_id: int
    at: datetime
    kind: VerifyKind
    passed: bool
    duration_ms: int | None
    commit: str | None
    task_id: int | None


@dataclass(frozen=True)
class PhaseChange:
    event_id: int
    at: datetime
    task_id: int
    phase: str
    entered: bool


StreamEvent = LandedCommit | VerifyVerdict | PhaseChange


@dataclass(frozen=True)
class SkippedRow:
    """A row of a mapped type that could not be projected; reported so it neither vanishes nor stalls the cursor."""

    event_id: int
    event_type: StoreEventType
    reason: SkipReason


@dataclass(frozen=True)
class EventBatch:
    """The mapped rows after a cursor; `cursor` is the last row read, `has_more` whether the limit was reached."""

    events: tuple[StreamEvent, ...]
    skipped: tuple[SkippedRow, ...]
    cursor: int
    has_more: bool


class _Payload(BaseModel):
    """The allowlist of one event type's `data` keys; an absent key arrives as null."""

    model_config = ConfigDict(frozen=True, extra='forbid')


_LANDED_STATE: Final = 'done'


class _MergeFinalized(_Payload):
    state: str | None
    merge_sha: CommitSha | None

    @model_validator(mode='after')
    def _a_landing_names_its_commit(self) -> Self:
        if self.state == _LANDED_STATE and self.merge_sha is None:
            raise ValueError('a done merge_finalized carries no merge_sha')
        return self


class _TrainMerged(_Payload):
    member_task_ids: tuple[str, ...]
    merge_commit_sha: CommitSha


class _MergeVerify(_Payload):
    passed: StrictBool
    duration_ms: NonNegativeInt | None
    merge_sha: CommitSha | None


class _WorkflowVerify(_Payload):
    passed: StrictBool
    tip_sha: CommitSha | None


def _payload_sql(model: type[_Payload]) -> str:
    """The SQL building `model`'s keys from `data`, or NULL when `data` is not JSON; field names are identifiers."""
    pairs = ', '.join(f"'{name}', json(data -> '$.{name}')" for name in model.model_fields)
    return f'CASE WHEN json_valid(data) THEN json_object({pairs}) END'


_TASK_ID: Final = re.compile(r'[0-9]+')


def _task_id(text: str | None) -> int | None:
    """The positive integer a TEXT task_id spells in decimal digits, else None."""
    if text is None or not _TASK_ID.fullmatch(text):
        return None
    value = int(text)
    return value if value >= 1 else None


def _instant(text: str) -> datetime | None:
    """The UTC instant an ISO timestamp names, or None when it is unparseable or carries no offset."""
    try:
        value = datetime.fromisoformat(text)
    except (TypeError, ValueError):
        return None
    return None if value.utcoffset() is None else value.astimezone(UTC)


@dataclass(frozen=True)
class _Row:
    event_id: int
    at: datetime
    task_id: int | None
    phase: str | None


def _merge_finalized(row: _Row, payload: _MergeFinalized) -> LandedCommit | None:
    if payload.state != _LANDED_STATE or payload.merge_sha is None:
        return None
    task_ids = () if row.task_id is None else (row.task_id,)
    return LandedCommit(event_id=row.event_id, at=row.at, commit=payload.merge_sha, task_ids=task_ids)


def _train_merged(row: _Row, payload: _TrainMerged) -> LandedCommit:
    members = (_task_id(member) for member in payload.member_task_ids)
    return LandedCommit(
        event_id=row.event_id,
        at=row.at,
        commit=payload.merge_commit_sha,
        task_ids=tuple(member for member in members if member is not None),
    )


def _merge_verify(row: _Row, payload: _MergeVerify) -> VerifyVerdict:
    return VerifyVerdict(
        event_id=row.event_id,
        at=row.at,
        kind=VerifyKind.MERGE,
        passed=payload.passed,
        duration_ms=payload.duration_ms,
        commit=payload.merge_sha,
        task_id=row.task_id,
    )


def _workflow_verify(row: _Row, payload: _WorkflowVerify) -> VerifyVerdict:
    return VerifyVerdict(
        event_id=row.event_id,
        at=row.at,
        kind=VerifyKind.TASK,
        passed=payload.passed,
        duration_ms=None,
        commit=payload.tip_sha,
        task_id=row.task_id,
    )


def _phase_change(*, entered: bool) -> Callable[[_Row, None], PhaseChange | None]:
    def read(row: _Row, _payload: None) -> PhaseChange | None:
        if row.task_id is None or not row.phase:
            return None
        return PhaseChange(
            event_id=row.event_id, at=row.at, task_id=row.task_id, phase=row.phase, entered=entered
        )

    return read


@dataclass(frozen=True)
class _StreamType:
    payload: type[_Payload] | None
    read: Callable[[_Row, Any], StreamEvent | None]


_STREAM: Final[Mapping[StoreEventType, _StreamType]] = MappingProxyType({
    StoreEventType.MERGE_FINALIZED: _StreamType(_MergeFinalized, _merge_finalized),
    StoreEventType.TRAIN_MERGED: _StreamType(_TrainMerged, _train_merged),
    StoreEventType.MERGE_VERIFY: _StreamType(_MergeVerify, _merge_verify),
    StoreEventType.WORKFLOW_VERIFY: _StreamType(_WorkflowVerify, _workflow_verify),
    StoreEventType.PHASE_ENTER: _StreamType(None, _phase_change(entered=True)),
    StoreEventType.PHASE_EXIT: _StreamType(None, _phase_change(entered=False)),
})


def _placeholders(count: int) -> str:
    return ', '.join('?' * count)


@dataclass(frozen=True)
class _Query:
    sql: str
    params: tuple[Any, ...]


_Payloads = Mapping[StoreEventType, type[_Payload] | None]


def _rows_of_types(columns: str, payloads: _Payloads, *, after_id: int = 0, limit: int = -1) -> _Query:
    """`columns` plus each row's payload (NULL for a type without a model), in id order; -1 is no limit."""
    with_payload = [(kind, model) for kind, model in payloads.items() if model is not None]
    cases = ' '.join(f'WHEN ? THEN {_payload_sql(model)}' for _, model in with_payload)
    payload = f'CASE event_type {cases} END' if with_payload else 'NULL'
    return _Query(
        f'SELECT {columns}, {payload} FROM events'
        f' WHERE id > ? AND event_type IN ({_placeholders(len(payloads))}) ORDER BY id LIMIT ?',
        (*(kind.value for kind, _ in with_payload), after_id, *(kind.value for kind in payloads), limit),
    )


_ModelT = TypeVar('_ModelT', bound=_Payload)


def _validated(model: type[_ModelT], payload: str | None) -> _ModelT | None:
    if payload is None:
        return None
    try:
        return model.model_validate_json(payload)
    except ValidationError:
        return None


def _read(root: Path, *queries: _Query) -> tuple[list[tuple[Any, ...]], ...]:
    """Every query's rows, all read from one snapshot of runs.db."""
    with read_only(root, runs_db(root), source=Source.EVENTS) as conn:
        return tuple(conn.execute(query.sql, query.params).fetchall() for query in queries)


def _projected(raw: tuple[Any, ...]) -> StreamEvent | SkippedRow | None:
    event_id, timestamp, event_type, task_id, phase, payload = raw
    kind = StoreEventType(event_type)
    at = _instant(timestamp)
    if at is None:
        return SkippedRow(event_id=event_id, event_type=kind, reason=SkipReason.BAD_TIMESTAMP)
    spec = _STREAM[kind]
    parsed = None
    if spec.payload is not None:
        parsed = _validated(spec.payload, payload)
        if parsed is None:
            return SkippedRow(event_id=event_id, event_type=kind, reason=SkipReason.MALFORMED_PAYLOAD)
    return spec.read(_Row(event_id=event_id, at=at, task_id=_task_id(task_id), phase=phase), parsed)


_STREAM_PAYLOADS: Final[_Payloads] = MappingProxyType(
    {kind: spec.payload for kind, spec in _STREAM.items()}
)


def read_events(root: Path, *, after_id: int, limit: int) -> EventBatch:
    """Up to `limit` rows of the mapped types after `after_id`, projected in id order."""
    if limit < 1:
        raise ValueError(f'limit must be at least 1, got {limit}')
    (rows,) = _read(
        root,
        _rows_of_types(
            'id, timestamp, event_type, task_id, phase', _STREAM_PAYLOADS,
            after_id=after_id, limit=limit,
        ),
    )
    projected = [_projected(row) for row in rows]
    return EventBatch(
        events=tuple(item for item in projected if isinstance(item, StreamEvent)),
        skipped=tuple(item for item in projected if isinstance(item, SkippedRow)),
        cursor=rows[-1][0] if rows else after_id,
        has_more=len(rows) == limit,
    )


@dataclass(frozen=True)
class FactoryStatus:
    """Whether the scheduler runs and how long runs.db has been quiet; named as contract.FactoryState's fields."""

    scheduler: SchedulerState
    since: datetime | None
    last_event_at: datetime | None
    idle_seconds: int | None


class _PauseRestored(_Payload):
    reasserted: StrictBool | None


_SCHEDULER_PAYLOADS: Final[_Payloads] = MappingProxyType({
    StoreEventType.SCHEDULER_PAUSED: None,
    StoreEventType.SCHEDULER_RESUMED: None,
    StoreEventType.SCHEDULER_PAUSE_RESTORED: _PauseRestored,
})

_NEWEST_TIMESTAMP: Final = _Query('SELECT timestamp FROM events ORDER BY id DESC LIMIT 1', ())


def _scheduler_after(kind: StoreEventType, payload: str | None) -> SchedulerState:
    """A restored pause ends only when the restart recorded that it declined to reassert it."""
    if kind is StoreEventType.SCHEDULER_RESUMED:
        return SchedulerState.RUNNING
    if kind is StoreEventType.SCHEDULER_PAUSE_RESTORED:
        restored = _validated(_PauseRestored, payload)
        if restored is not None and restored.reasserted is False:
            return SchedulerState.RUNNING
    return SchedulerState.PAUSED


def _scheduler_since(rows: list[tuple[Any, ...]]) -> tuple[SchedulerState, datetime | None]:
    scheduler, since = SchedulerState.RUNNING, None
    for event_type, timestamp, payload in rows:
        after = _scheduler_after(StoreEventType(event_type), payload)
        if after is not scheduler:
            scheduler, since = after, _instant(timestamp)
    return scheduler, since


def read_factory_status(root: Path, *, now: datetime) -> FactoryStatus:
    """The scheduler state folded from every pause and resume, and the quiet time since the newest row."""
    if now.utcoffset() is None:
        raise ValueError(f'now must be timezone-aware, got {now.isoformat()}')
    changes, newest = _read(
        root, _rows_of_types('event_type, timestamp', _SCHEDULER_PAYLOADS), _NEWEST_TIMESTAMP
    )
    scheduler, since = _scheduler_since(changes)
    last_event_at = _instant(newest[0][0]) if newest else None
    idle_seconds = (
        None
        if last_event_at is None
        else max(0, math.floor((now - last_event_at).total_seconds()))
    )
    return FactoryStatus(
        scheduler=scheduler, since=since, last_event_at=last_event_at, idle_seconds=idle_seconds
    )


@dataclass(frozen=True)
class OpenEscalation:
    escalation_id: str
    level: int | None


@dataclass(frozen=True)
class OpenEscalations:
    """Every escalation runs.db records as created and not yet resolved, by task."""

    by_task: Mapping[int, tuple[OpenEscalation, ...]]

    def levels(self, task_id: int) -> tuple[int | None, ...]:
        """The recorded level of each of the task's open escalations, None where none is recorded."""
        return tuple(escalation.level for escalation in self.by_task.get(task_id, ()))


class _EscalationCreated(_Payload):
    escalation_id: str | None
    level: JsonValue


class _EscalationResolved(_Payload):
    escalation_id: str | None
    escalation_ids: tuple[str, ...] | None


_ESCALATION_PAYLOADS: Final[_Payloads] = MappingProxyType({
    StoreEventType.ESCALATION_CREATED: _EscalationCreated,
    StoreEventType.ESCALATION_RESOLVED: _EscalationResolved,
})


def _recorded_level(value: JsonValue) -> int | None:
    return value if isinstance(value, int) and not isinstance(value, bool) else None


_Opened = dict[str, tuple[int, OpenEscalation]]


def _closed_ids(resolved: _EscalationResolved, task_id: int | None, opened: _Opened) -> tuple[str, ...]:
    """By id, else by the listed ids, else every open escalation of the resolving task."""
    if resolved.escalation_id is not None:
        return (resolved.escalation_id,)
    if resolved.escalation_ids is not None:
        return resolved.escalation_ids
    return tuple(key for key, (owner, _) in opened.items() if task_id is not None and owner == task_id)


def _apply_escalation(opened: _Opened, event_type: str, task_id: int | None, payload: str | None) -> None:
    if event_type == StoreEventType.ESCALATION_CREATED:
        created = _validated(_EscalationCreated, payload)
        if created is not None and created.escalation_id is not None and task_id is not None:
            level = _recorded_level(created.level)
            opened[created.escalation_id] = (task_id, OpenEscalation(created.escalation_id, level))
        return
    resolved = _validated(_EscalationResolved, payload)
    if resolved is not None:
        for escalation_id in _closed_ids(resolved, task_id, opened):
            opened.pop(escalation_id, None)


def read_open_escalations(root: Path) -> OpenEscalations:
    """The open escalations left by replaying every created and resolved row in id order."""
    (rows,) = _read(root, _rows_of_types('event_type, task_id', _ESCALATION_PAYLOADS))
    opened: _Opened = {}
    for event_type, task_id, payload in rows:
        _apply_escalation(opened, event_type, _task_id(task_id), payload)
    by_task: dict[int, tuple[OpenEscalation, ...]] = {}
    for owner, escalation in opened.values():
        by_task[owner] = (*by_task.get(owner, ()), escalation)
    return OpenEscalations(by_task=MappingProxyType(by_task))


class _ReviewCompleted(_Payload):
    cycle: NonNegativeInt
    blocking_count: NonNegativeInt
    suggestion_count: NonNegativeInt
    by_category: dict[str, NonNegativeInt]


def _review_queries(task_ids: tuple[str, ...]) -> tuple[_Query, _Query]:
    marks = _placeholders(len(task_ids))
    reviews = _Query(
        f'SELECT task_id, {_payload_sql(_ReviewCompleted)} FROM events'
        f' WHERE event_type = ? AND task_id IN ({marks}) ORDER BY id',
        (StoreEventType.REVIEW_COMPLETED.value, *task_ids),
    )
    cycles = _Query(
        f'SELECT task_id, SUM(review_cycles) FROM task_results'
        f' WHERE task_id IN ({marks}) GROUP BY task_id',
        task_ids,
    )
    return reviews, cycles


def _review_counts(newest: _ReviewCompleted | None, cycles_total: int) -> ReviewCounts | NotRecorded:
    """The newest valid review's counts; its cycle never under-reports the runs' lifetime total."""
    if newest is None:
        return NOT_RECORDED
    return ReviewCounts(
        cycles=max(newest.cycle, cycles_total),
        blocking_count=newest.blocking_count,
        suggestion_count=newest.suggestion_count,
        by_category=newest.by_category,
    )


def read_review_counts(
    root: Path, task_ids: Iterable[int]
) -> dict[int, ReviewCounts | NotRecorded]:
    """Each requested task's last recorded review counts, or not_recorded when runs.db holds none."""
    requested = sorted(set(task_ids))
    if not requested:
        return {}
    reviews, cycles = _read(root, *_review_queries(tuple(str(task_id) for task_id in requested)))
    newest: dict[int, _ReviewCompleted] = {}
    for task_id, payload in reviews:
        parsed = _validated(_ReviewCompleted, payload)
        if parsed is not None:
            newest[int(task_id)] = parsed
    totals = {int(task_id): total or 0 for task_id, total in cycles}
    return {
        task_id: _review_counts(newest.get(task_id), totals.get(task_id, 0)) for task_id in requested
    }
