"""Facts read from the orchestrator's runs.db: a batched cursor over events.id, and stateless whole-store readers.

Each payload is built inside SQLite from its model's field names, so the `data` blob is never selected whole.
"""

from __future__ import annotations

import re
from collections.abc import Callable, Mapping
from dataclasses import dataclass
from datetime import UTC, datetime
from enum import StrEnum
from pathlib import Path
from types import MappingProxyType
from typing import Any, Final, Self

from pydantic import BaseModel, ConfigDict, StrictBool, ValidationError, model_validator
from shared.project_layout import runs_db

from client_feed.contract import CommitSha, NonNegativeInt, Source, VerifyKind
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


def _stream_query() -> tuple[str, tuple[str, ...]]:
    """The stream SELECT and its leading CASE parameters; `?` for the cursor, the types and the limit follow."""
    with_payload = [(kind, spec.payload) for kind, spec in _STREAM.items() if spec.payload]
    cases = ' '.join(f'WHEN ? THEN {_payload_sql(model)}' for _, model in with_payload)
    sql = (
        f'SELECT id, timestamp, event_type, task_id, phase, CASE event_type {cases} END'
        f' FROM events WHERE id > ? AND event_type IN ({_placeholders(len(_STREAM))})'
        ' ORDER BY id LIMIT ?'
    )
    return sql, tuple(kind.value for kind, _ in with_payload)


_STREAM_SQL, _STREAM_CASE_TYPES = _stream_query()


def _validated(model: type[_Payload], payload: str | None) -> _Payload | None:
    if payload is None:
        return None
    try:
        return model.model_validate_json(payload)
    except ValidationError:
        return None


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


def _read(root: Path, sql: str, params: tuple[Any, ...]) -> list[tuple[Any, ...]]:
    with read_only(root, runs_db(root), source=Source.EVENTS) as conn:
        return conn.execute(sql, params).fetchall()


def read_events(root: Path, *, after_id: int, limit: int) -> EventBatch:
    """Up to `limit` rows of the mapped types after `after_id`, projected in id order."""
    if limit < 1:
        raise ValueError(f'limit must be at least 1, got {limit}')
    rows = _read(
        root, _STREAM_SQL, (*_STREAM_CASE_TYPES, after_id, *(kind.value for kind in _STREAM), limit)
    )
    projected = [_projected(row) for row in rows]
    return EventBatch(
        events=tuple(item for item in projected if isinstance(item, StreamEvent)),
        skipped=tuple(item for item in projected if isinstance(item, SkippedRow)),
        cursor=rows[-1][0] if rows else after_id,
        has_more=len(rows) == limit,
    )

