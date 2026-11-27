"""The scheduler state snapshot: one typed shape for its producer and every reader.

The orchestrator scheduler builds a ``SchedulerStateSnapshot`` and writes its
``to_json()`` form to ``scheduler_state.json``; fused-memory and the dashboard
parse that form back with ``from_json``. Unknown keys from a newer producer are
ignored and keys an older producer lacks take their defaults, so a field is
added in one place: here. Each process re-serialises through its own deployed
copy of this model, so fused-memory's ``get_scheduler_state`` forwards only the
fields that copy defines: a new field reaches the dashboard only once the
orchestrator, fused-memory and the dashboard all run code that defines it.
``snapshot_at is None`` means no producer has written a snapshot.
Contract: plans/review-all-s14-shared-leaves-prd.md (decision 12, invariant I4,
INV-13).

``read_scheduler_state_checked`` is the one cross-checked reader of the pause
fact: it serves the snapshot as a cache over the runs.db ``scheduler_state``
row, which is that fact's home.
"""

from __future__ import annotations

import json
import logging
import sqlite3
from collections.abc import Mapping
from dataclasses import dataclass
from datetime import datetime
from pathlib import Path
from typing import Any, Self

from pydantic import (
    AwareDatetime,
    BaseModel,
    ConfigDict,
    Field,
    StrictInt,
    ValidationError,
    field_serializer,
)

from shared.project_layout import StrPath, runs_db, scheduler_state_json

__all__ = [
    'CheckedSchedulerState',
    'MalformedSnapshotError',
    'OverrideRecord',
    'ParkRecord',
    'PauseRow',
    'PinQueueEntry',
    'RequeueCooldown',
    'SchedulerStateSnapshot',
    'read_scheduler_state_checked',
]

logger = logging.getLogger(__name__)

_RECORD_CONFIG = ConfigDict(frozen=True, extra='ignore')

_EXCLUSIVE = 1


class MalformedSnapshotError(ValueError):
    """The input to ``SchedulerStateSnapshot.from_json`` or ``from_bytes`` is not a snapshot's wire form."""


class ParkRecord(BaseModel):
    """The modules one owner holds parked, and when the park was installed ('' if unknown)."""

    model_config = _RECORD_CONFIG

    modules: tuple[str, ...] = ()
    installed_at: str = ''


class PinQueueEntry(BaseModel):
    model_config = _RECORD_CONFIG

    task_id: str
    order: int | None = None


class OverrideRecord(BaseModel):
    """An operator override; the all-default record means no override."""

    model_config = _RECORD_CONFIG

    boost_tier: str | None = None
    pinned: bool = False
    reserve_now: bool = False
    ttl_until: str | None = None


class RequeueCooldown(BaseModel):
    """The requeue cooldown gating a task's re-dispatch, as armed.

    Every value is fixed at arming, never derived from now, because the
    producer content-dedups snapshot writes. ``envelope_secs`` is the
    un-jittered ceiling ``armed_secs`` was drawn from (armed lies in
    [envelope/2, envelope]); both grow across successive transient requeues
    and stay flat for a genuine one. ``armed_at`` and ``expires_at`` are
    ISO-8601 wall-clock strings. An expired entry can linger until the
    producer's next tick, so compare ``expires_at`` against now rather than
    treating presence as proof the task is still cooling.
    """

    model_config = _RECORD_CONFIG

    armed_secs: float
    envelope_secs: float
    transient: bool
    transient_count: int
    armed_at: str
    expires_at: str


class SchedulerStateSnapshot(BaseModel):
    """``parks`` maps each owner to the modules it tops; ``park_stacks`` maps each
    module to its whole stack, bottom to top, as open entry records; ``lock_depth``
    is the depth the producer normalises lock keys to. ``pause_class`` is the wire
    value of ``orchestrator/src/orchestrator/pause_policy.py::PauseClass`` (shared
    cannot import orchestrator), ``None`` when not paused. ``held_by_task`` maps
    each holding task to its sorted lock keys and names every holder of a shared
    key, while ``current_holders`` names one holder per key, for display.
    ``holder_limits`` maps each held key to the producer lock table's holder
    limit (a very large int for a key taking unlimited holders), so a reader
    compares ``len(holders) >= limit``. ``successor_of`` maps each early-started
    dependent to its predecessor; both appear as holders of their shared keys in
    ``held_by_task``.

    Frozen means no field can be reassigned. The mappings are plain dicts that
    validation copies from its input, so a snapshot never aliases its producer's
    state, but nothing stops a reader mutating one in place.
    """

    model_config = _RECORD_CONFIG

    skip_counts: Mapping[str, int] = Field(default_factory=dict)
    parks: Mapping[str, ParkRecord] = Field(default_factory=dict)
    park_stacks: Mapping[str, tuple[Mapping[str, Any], ...]] = Field(default_factory=dict)
    pin_reservations: Mapping[str, ParkRecord] = Field(default_factory=dict)
    effective_priorities: Mapping[str, str] = Field(default_factory=dict)
    pin_queue: tuple[PinQueueEntry, ...] = ()
    overrides: Mapping[str, OverrideRecord] = Field(default_factory=dict)
    current_holders: Mapping[str, str] = Field(default_factory=dict)
    held_by_task: Mapping[str, tuple[str, ...]] = Field(default_factory=dict)
    holder_limits: Mapping[str, int] = Field(default_factory=dict)
    successor_of: Mapping[str, str] = Field(default_factory=dict)
    lock_depth: StrictInt | None = None
    is_paused: bool = False
    pause_reason: str | None = None
    pause_class: str | None = None
    requeue_cooldowns: Mapping[str, RequeueCooldown] = Field(default_factory=dict)
    snapshot_at: AwareDatetime | None = None

    @field_serializer('snapshot_at')
    def _render_snapshot_at(self, value: datetime | None) -> str | None:
        return value.isoformat() if value is not None else None

    @classmethod
    def empty(cls) -> SchedulerStateSnapshot:
        return cls()

    @classmethod
    def from_json(cls, data: Mapping[str, Any]) -> Self:
        """Parse a snapshot's wire form; unknown keys are ignored.

        Raises MalformedSnapshotError when *data* does not have the snapshot's shape.
        """
        try:
            return cls.model_validate(data)
        except ValidationError as exc:
            raise MalformedSnapshotError(str(exc)) from exc

    @classmethod
    def from_bytes(cls, raw: bytes) -> Self:
        """Parse the bytes of ``scheduler_state.json``.

        Raises MalformedSnapshotError when *raw* is not JSON text holding a
        snapshot's wire form.
        """
        try:
            data = json.loads(raw)
        except (json.JSONDecodeError, UnicodeDecodeError) as exc:
            raise MalformedSnapshotError(str(exc)) from exc
        return cls.from_json(data)

    def to_json(self) -> dict[str, Any]:
        return self.model_dump(mode='json')

    def holder_task_ids(self) -> frozenset[str]:
        """Every task id the snapshot names as holding a lock key.

        ``held_by_task`` names every holder of a shared key. ``current_holders``
        names one per key, and is all an older producer writes; the union
        reads either producer.
        """
        return frozenset(self.held_by_task) | frozenset(self.current_holders.values())

    def holders_by_key(self) -> dict[str, frozenset[str]]:
        """Every task holding each held lock key.

        Inverts ``held_by_task`` and unions ``current_holders``, the same
        fallback ``holder_task_ids`` reads: an older producer writes only
        ``current_holders``, and a newer one derives it from ``held_by_task``.
        """
        holders: dict[str, set[str]] = {}
        for task_id, keys in self.held_by_task.items():
            for key in keys:
                holders.setdefault(key, set()).add(task_id)
        for key, task_id in self.current_holders.items():
            holders.setdefault(key, set()).add(task_id)
        return {key: frozenset(ids) for key, ids in holders.items()}

    def holder_limit(self, key: str) -> int:
        """The holder limit published for *key*, else exclusive.

        Exclusive is how every reader treated a held key before limits were
        published, so an older producer reads as it always did.
        """
        return self.holder_limits.get(key, _EXCLUSIVE)


class PauseRow(BaseModel):
    """The runs.db ``scheduler_state`` row, as its raw stored strings."""

    model_config = _RECORD_CONFIG

    reason: str
    pause_at: str
    pause_class: str | None = None


class CheckedSchedulerState(SchedulerStateSnapshot):
    """The snapshot cross-checked against the runs.db pause row.

    ``is_paused``, ``pause_reason`` and ``pause_class`` are the SERVED verdict:
    the pause fact's home is the runs.db row, and the JSON snapshot is a cache
    of it. ``pause_row`` is the row as read, ``None`` when there is none or it
    is unreadable. Contract: plans/instruments-honesty-prd.md (decision 5,
    §Contract "Scheduler state reader").
    """

    snapshot_age_s: float | None = None
    pause_row: PauseRow | None = None
    pause_disagreement: bool = False


@dataclass(frozen=True)
class _PauseRowRead:
    readable: bool
    row: PauseRow | None = None


_LATEST_PAUSE_ROW = (
    'SELECT pause_reason, pause_at, pause_class FROM scheduler_state '
    'ORDER BY pause_at DESC LIMIT 1'
)


def read_scheduler_state_checked(root: StrPath, *, now: datetime) -> CheckedSchedulerState:
    """The scheduler state of the project at *root*, its pause verdict cross-checked.

    The snapshot is read from ``scheduler_state.json`` and the pause row from
    runs.db, opened read-only so the reader never creates or writes it. When
    the row is readable and the snapshot is unreadable or disagrees with it,
    the row's verdict is served. Unreadable inputs are logged, never raised.

    Raises:
        ValueError: *now* is naive.
    """
    if now.tzinfo is None:
        raise ValueError(f'now must be timezone-aware; got naive {now.isoformat()!r}')
    snapshot = _read_snapshot(scheduler_state_json(root))
    row_read = _read_pause_row(runs_db(root))
    served = snapshot if snapshot is not None else CheckedSchedulerState()
    disagreement = (
        snapshot is not None
        and row_read.readable
        and snapshot.is_paused != (row_read.row is not None)
    )
    update: dict[str, Any] = {
        'snapshot_age_s': (
            (now - served.snapshot_at).total_seconds() if served.snapshot_at is not None else None
        ),
        'pause_row': row_read.row,
        'pause_disagreement': disagreement,
    }
    if row_read.readable and (snapshot is None or disagreement):
        row = row_read.row
        update['is_paused'] = row is not None
        update['pause_reason'] = row.reason if row is not None else None
        update['pause_class'] = row.pause_class if row is not None else None
    return served.model_copy(update=update)


def _read_snapshot(path: Path) -> CheckedSchedulerState | None:
    try:
        raw = path.read_bytes()
    except FileNotFoundError:
        return None
    except OSError as exc:
        logger.warning('scheduler snapshot %s is unreadable: %s', path, exc)
        return None
    try:
        return CheckedSchedulerState.from_bytes(raw)
    except MalformedSnapshotError as exc:
        logger.warning('scheduler snapshot %s is not a snapshot: %s', path, exc)
        return None


def _read_pause_row(path: Path) -> _PauseRowRead:
    if not path.exists():
        return _PauseRowRead(readable=False)
    try:
        conn = sqlite3.connect(f'{path.resolve().as_uri()}?mode=ro', uri=True)
        try:
            found = conn.execute(_LATEST_PAUSE_ROW).fetchone()
        finally:
            conn.close()
    except sqlite3.Error as exc:
        logger.warning('runs.db %s pause row is unreadable: %s', path, exc)
        return _PauseRowRead(readable=False)
    if found is None:
        return _PauseRowRead(readable=True)
    reason, pause_at, pause_class = found
    try:
        row = PauseRow(reason=reason, pause_at=pause_at, pause_class=pause_class)
    except ValidationError as exc:
        logger.warning('runs.db %s pause row is malformed: %s', path, exc)
        return _PauseRowRead(readable=False)
    return _PauseRowRead(readable=True, row=row)
