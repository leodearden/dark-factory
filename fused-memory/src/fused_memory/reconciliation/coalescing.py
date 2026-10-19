"""The rule for which buffered task events fold together and how their payloads merge.

Recon lever 3 (task 4278): a task that changes status or is edited many times
before the next drain needs only one buffered row, because reconciliation reads
the task's live state when it assembles.  ``coalesce_key`` names the events that
may share a row; ``event_buffer.EventBuffer.push`` performs the fold.

Dependency operations never fold.  Each ``add_dependency`` / ``remove_dependency``
event's ``depends_on`` is a distinct fact: on 2026-10-19, 113 live such events
covered 111 distinct edges.

The key is a pure function of the event, so it is stored even when coalescing is
switched off; turning it back on then folds into rows written meanwhile.
Rationale and sizing: ``plans/recon-throughput-capacity.md`` (Lever 3).
"""

from __future__ import annotations

from collections.abc import Mapping
from dataclasses import dataclass
from datetime import UTC, datetime
from typing import Any

from fused_memory.models.reconciliation import EventType, ReconciliationEvent

__all__ = [
    'COALESCED_COUNT_KEY',
    'COALESCED_LAST_ID_KEY',
    'COALESCED_LAST_TIMESTAMP_KEY',
    'COALESCED_STATUS_PATH_KEY',
    'COALESCED_STATUS_PATH_TRUNCATED_KEY',
    'CoalescedFold',
    'coalesce_key',
    'coalesced_count',
    'merge_coalesced_payload',
    'task_id_key',
]

_COALESCIBLE_TASK_MODIFIED_OPERATIONS = frozenset({'update_task'})

# Bookkeeping keys a fold writes into the merged payload.  Every other key is a
# producer key, owned by whichever constituent event is newest.
COALESCED_COUNT_KEY = '_coalesced_count'
COALESCED_LAST_ID_KEY = '_coalesced_last_id'
COALESCED_LAST_TIMESTAMP_KEY = '_coalesced_last_timestamp'
COALESCED_STATUS_PATH_KEY = '_coalesced_status_path'
COALESCED_STATUS_PATH_TRUNCATED_KEY = '_coalesced_status_path_truncated'
_BOOKKEEPING_KEYS = frozenset({
    COALESCED_COUNT_KEY,
    COALESCED_LAST_ID_KEY,
    COALESCED_LAST_TIMESTAMP_KEY,
    COALESCED_STATUS_PATH_KEY,
    COALESCED_STATUS_PATH_TRUNCATED_KEY,
})

# Bounds the payload of a task that flaps for hours between drains.
_MAX_COALESCED_STATUS_PATH = 20


def task_id_key(value: object) -> str | None:
    """A payload ``task_id`` as its canonical string, or None when it names no task.

    Writers emit int or str ids, so ``7`` and ``'7'`` are one task.  Anything
    else is rejected rather than stringified: ``str()`` of a bool or a float
    would mint a key (``'True'``, ``'7.0'``) that never matches the real id.
    """
    if isinstance(value, bool) or not isinstance(value, int | str):
        return None
    key = str(value)
    return key or None


def coalesce_key(event: ReconciliationEvent) -> str | None:
    """The key ``event`` folds under in the buffer, or None when it never folds."""
    task_id = task_id_key(event.payload.get('task_id'))
    if task_id is None:
        return None
    if event.type is EventType.task_status_changed:
        return f'{event.type.value}:{task_id}'
    operation = event.payload.get('operation')
    if (
        event.type is EventType.task_modified
        and operation in _COALESCIBLE_TASK_MODIFIED_OPERATIONS
    ):
        return f'{event.type.value}:{task_id}:{operation}'
    return None


@dataclass(frozen=True)
class CoalescedFold:
    """The outcome of folding one event into a buffered row.

    ``row_timestamp`` is the row's timestamp after the fold: unchanged, or
    moved earlier when an older event arrives late.  ``folded_timestamp`` is
    the timestamp of the constituent that does NOT own the row timestamp, i.e.
    the event the fold made disappear, which is the one to count per hour.
    """

    payload: dict[str, Any]
    row_timestamp: datetime
    folded_timestamp: datetime


def coalesced_count(payload: Mapping[str, Any]) -> int:
    """How many pushed events a stored payload stands for (1 when never folded)."""
    count = payload.get(COALESCED_COUNT_KEY)
    if isinstance(count, bool) or not isinstance(count, int) or count < 1:
        return 1
    return count


def merge_coalesced_payload(
    existing: Mapping[str, Any],
    incoming: Mapping[str, Any],
    *,
    event_type: EventType,
    incoming_id: str,
    incoming_ts: datetime,
    row_ts: datetime,
) -> CoalescedFold | None:
    """Fold ``incoming`` into a buffered row's ``existing`` payload.

    Returns None when ``incoming_id`` is the last event already folded in: a
    redelivery, which must not be counted twice.

    Timestamps are minted when task_interceptor builds the event but committed
    later through EventQueue, by several MCP processes sharing one database, so
    pushes can arrive out of timestamp order.  The arrival is classified
    against the row's span: OLDER than the row, NEWER than the newest folded
    constituent, or in the MIDDLE.  That makes ``old_status`` (earliest
    constituent), ``new_status`` and every producer key (newest constituent)
    and the count independent of arrival order.  The status path is
    best-effort: a middle arrival only bumps the count.
    """
    if existing.get(COALESCED_LAST_ID_KEY) == incoming_id:
        return None
    row_ts = _as_utc(row_ts)
    incoming_ts = _as_utc(incoming_ts)
    newest_ts = _newest_timestamp(existing, default=row_ts)
    tracks_status = event_type is EventType.task_status_changed
    if incoming_ts < row_ts:
        return CoalescedFold(
            payload=_merge_older(existing, incoming, newest_ts, tracks_status=tracks_status),
            row_timestamp=incoming_ts,
            folded_timestamp=row_ts,
        )
    if incoming_ts >= newest_ts:
        return CoalescedFold(
            payload=_merge_newer(
                existing, incoming, incoming_id, incoming_ts, tracks_status=tracks_status,
            ),
            row_timestamp=row_ts,
            folded_timestamp=incoming_ts,
        )
    return CoalescedFold(
        payload=_counted(existing), row_timestamp=row_ts, folded_timestamp=incoming_ts,
    )


def _as_utc(instant: datetime) -> datetime:
    return instant.replace(tzinfo=UTC) if instant.tzinfo is None else instant


def _newest_timestamp(payload: Mapping[str, Any], *, default: datetime) -> datetime:
    """The newest folded constituent's timestamp; ``default`` before any fold."""
    stamped = payload.get(COALESCED_LAST_TIMESTAMP_KEY)
    if not isinstance(stamped, str):
        return default
    try:
        return _as_utc(datetime.fromisoformat(stamped))
    except ValueError:
        return default


def _counted(existing: Mapping[str, Any]) -> dict[str, Any]:
    return {**existing, COALESCED_COUNT_KEY: coalesced_count(existing) + 1}


def _merge_newer(
    existing: Mapping[str, Any],
    incoming: Mapping[str, Any],
    incoming_id: str,
    incoming_ts: datetime,
    *,
    tracks_status: bool,
) -> dict[str, Any]:
    """Incoming is the newest constituent: its producer keys replace the row's."""
    merged = {key: value for key, value in incoming.items() if key not in _BOOKKEEPING_KEYS}
    merged.update(
        (key, value) for key, value in existing.items() if key in _BOOKKEEPING_KEYS
    )
    merged[COALESCED_COUNT_KEY] = coalesced_count(existing) + 1
    merged[COALESCED_LAST_ID_KEY] = incoming_id
    merged[COALESCED_LAST_TIMESTAMP_KEY] = incoming_ts.isoformat()
    if tracks_status:
        if 'old_status' in existing:
            merged['old_status'] = existing['old_status']
        merged.update(_grown_path(existing, incoming.get('new_status'), at_start=False))
    return merged


def _merge_older(
    existing: Mapping[str, Any],
    incoming: Mapping[str, Any],
    newest_ts: datetime,
    *,
    tracks_status: bool,
) -> dict[str, Any]:
    """Incoming predates the row: only the chain's start rewinds to it."""
    merged = _counted(existing)
    # The row timestamp is about to move earlier, so the newest constituent's
    # timestamp must now be stated explicitly.
    merged[COALESCED_LAST_TIMESTAMP_KEY] = newest_ts.isoformat()
    if tracks_status:
        if 'old_status' in incoming:
            merged['old_status'] = incoming['old_status']
        merged.update(_grown_path(existing, incoming.get('old_status'), at_start=True))
    return merged


def _grown_path(existing: Mapping[str, Any], entry: Any, *, at_start: bool) -> dict[str, Any]:
    """The status-path keys after adding ``entry`` at one end, within the cap."""
    path = _status_path(existing)
    truncated = existing.get(COALESCED_STATUS_PATH_TRUNCATED_KEY) is True
    if len(path) >= _MAX_COALESCED_STATUS_PATH:
        truncated = True
    else:
        path = [entry, *path] if at_start else [*path, entry]
    grown: dict[str, Any] = {COALESCED_STATUS_PATH_KEY: path}
    if truncated:
        grown[COALESCED_STATUS_PATH_TRUNCATED_KEY] = True
    return grown


def _status_path(payload: Mapping[str, Any]) -> list[Any]:
    path = payload.get(COALESCED_STATUS_PATH_KEY)
    if isinstance(path, list):
        return list(path)
    return [payload[key] for key in ('old_status', 'new_status') if key in payload]
