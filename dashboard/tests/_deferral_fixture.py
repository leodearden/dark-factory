"""Task rows, one per deferral hold the Tasks tab must word, built with the shared deferral model.

The rows are plain data and carry no assertions. ``test_deferral_summary.py``,
``test_task_snapshot.py``, ``test_active_tasks.py`` and ``test_tab_tasks_deferral.py``
all read this one population. :func:`expected_needs_human` computes each
deferred row's verdict with ``shared.task_deferral.needs_human``. That is the
one shared definition, applied to the population the census counts.
"""

from __future__ import annotations

import dataclasses
from collections.abc import Iterable, Mapping
from dataclasses import dataclass
from datetime import UTC, datetime, timedelta
from typing import Any, Final

from _canned_mcp import _raw_row
from shared.process_identity import Liveness, ProcessIdentity, ProcessIdentityUnreadable, liveness
from shared.task_deferral import (
    CarriedByDeferral,
    Deferral,
    HeldBySessionDeferral,
    LegacyUnknownDeferral,
    NeedsHumanReason,
    PlanningDeferral,
    UntilConditionDeferral,
    needs_human,
    parse_record,
    record_to_json,
)
from shared.task_statuses import TaskStatus

NOW: Final = datetime(2026, 10, 21, 12, 0, tzinfo=UTC)

CARRIER_DONE: Final = 900
CARRIER_CANCELLED: Final = 901
CARRIER_LIVE: Final = 902
CARRIER_ABSENT: Final = 903
CARRIER_STATUSES: Final[Mapping[int, TaskStatus]] = {
    CARRIER_DONE: TaskStatus.DONE,
    CARRIER_CANCELLED: TaskStatus.CANCELLED,
    CARRIER_LIVE: TaskStatus.PENDING,
}

CLOSABLE: Final = 101
ORPHANED: Final = 102
CARRIED_LIVE: Final = 103
CARRIER_GONE: Final = 104
CONDITION_EXPIRING: Final = 111
CONDITION_STALE: Final = 112
HELD_LIVE: Final = 121
HELD_DEAD: Final = 122
HELD_OTHER_HOST: Final = 123
PLANNING_FRESH: Final = 131
PLANNING_STALE: Final = 132
LEGACY: Final = 141
INVALID: Final = 151
UNRECORDED: Final = 161
STRAY: Final = 171
PLAIN: Final = 181

OTHER_HOST: Final = 'elsewhere.invalid'
FORGED_BOOT_ID: Final = 'forged-boot-id-a-reboot-ago'
CONDITION: Final = 'the upstream 2.0 release ships'
STALE_CONDITION: Final = 'the vendor answers the support ticket'
CARRIED_REASON: Final = 'folded into the carrier'
STAMPED_BY: Final = 'claude-task-6527-fixture'
INVALID_RECORD_JSON: Final[Mapping[str, Any]] = {'kind': 'carried_by'}


@dataclass(frozen=True)
class FixtureRow:
    task_id: int
    status: TaskStatus
    title: str
    record_json: Mapping[str, Any] | None


def _deferred(task_id: int, title: str, record: Deferral) -> FixtureRow:
    return FixtureRow(task_id, TaskStatus.DEFERRED, title, record_to_json(record))


def _stamp(at: datetime, *, stamped_by: str | None = STAMPED_BY) -> dict[str, Any]:
    return {'deferred_at': at, 'stamped_at': at, 'stamped_by': stamped_by}


def _carried(task_id: int, title: str, carrier: int, now: datetime) -> FixtureRow:
    record = CarriedByDeferral(
        carrier_task_id=carrier, reason=CARRIED_REASON, **_stamp(now - timedelta(hours=2))
    )
    return _deferred(task_id, title, record)


def _held(task_id: int, title: str, holder: ProcessIdentity, now: datetime) -> FixtureRow:
    record = HeldBySessionDeferral(holder=holder, **_stamp(now - timedelta(minutes=10)))
    return _deferred(task_id, title, record)


def _carried_by_rows(now: datetime) -> tuple[FixtureRow, ...]:
    return (
        _carried(CLOSABLE, 'hold: carrier done', CARRIER_DONE, now),
        _carried(ORPHANED, 'hold: carrier cancelled', CARRIER_CANCELLED, now),
        _carried(CARRIED_LIVE, 'hold: carrier pending', CARRIER_LIVE, now),
        _carried(CARRIER_GONE, 'hold: carrier absent', CARRIER_ABSENT, now),
    )


def _until_condition_rows(now: datetime) -> tuple[FixtureRow, ...]:
    expiring = UntilConditionDeferral(
        condition=CONDITION, expires_at=now + timedelta(days=3), **_stamp(now - timedelta(days=1))
    )
    stale = UntilConditionDeferral(
        condition=STALE_CONDITION, **_stamp(now - timedelta(days=31), stamped_by=None)
    )
    return (
        _deferred(CONDITION_EXPIRING, 'hold: condition with expiry', expiring),
        _deferred(CONDITION_STALE, 'hold: condition gone stale', stale),
    )


def _held_by_session_rows(now: datetime, live_holder: ProcessIdentity) -> tuple[FixtureRow, ...]:
    rebooted = dataclasses.replace(live_holder, boot_id=FORGED_BOOT_ID)
    elsewhere = dataclasses.replace(live_holder, host=OTHER_HOST)
    return (
        _held(HELD_LIVE, 'hold: live session', live_holder, now),
        _held(HELD_DEAD, 'hold: session gone', rebooted, now),
        _held(HELD_OTHER_HOST, 'hold: session elsewhere', elsewhere, now),
    )


def _unattended_rows(now: datetime) -> tuple[FixtureRow, ...]:
    fresh = PlanningDeferral(**_stamp(now - timedelta(hours=1)))
    stale = PlanningDeferral(**_stamp(now - timedelta(hours=25)))
    legacy = LegacyUnknownDeferral(
        deferred_at=None, stamped_at=now - timedelta(days=5), stamped_by='deferral-migration'
    )
    return (
        _deferred(PLANNING_FRESH, 'hold: planning batch', fresh),
        _deferred(PLANNING_STALE, 'hold: planning batch gone stale', stale),
        _deferred(LEGACY, 'hold: legacy', legacy),
        FixtureRow(INVALID, TaskStatus.DEFERRED, 'hold: invalid record', INVALID_RECORD_JSON),
        FixtureRow(UNRECORDED, TaskStatus.DEFERRED, 'hold: no record', None),
    )


def _not_deferred_rows(now: datetime) -> tuple[FixtureRow, ...]:
    stray = LegacyUnknownDeferral(deferred_at=None, stamped_at=now - timedelta(days=5))
    return (
        FixtureRow(STRAY, TaskStatus.PENDING, 'pending with a stray record', record_to_json(stray)),
        FixtureRow(PLAIN, TaskStatus.PENDING, 'plain pending', None),
    )


def deferral_fixture(*, now: datetime, live_holder: ProcessIdentity) -> tuple[FixtureRow, ...]:
    """Every hold kind and verdict, the unrecorded deferred row, a stray record and a plain row."""
    return (
        *_carried_by_rows(now),
        *_until_condition_rows(now),
        *_held_by_session_rows(now, live_holder),
        *_unattended_rows(now),
        *_not_deferred_rows(now),
    )


def raw_rows(fixture: Iterable[FixtureRow]) -> list[dict[str, Any]]:
    """The rows as ``CannedMCP`` serves them from ``get_tasks``."""
    return [
        _raw_row(
            row.task_id,
            row.status.value,
            title=row.title,
            metadata={} if row.record_json is None else {'deferral': dict(row.record_json)},
        )
        for row in fixture
    ]


def status_map(fixture: Iterable[FixtureRow]) -> dict[int, str]:
    """``{id: status}`` for every fixture row and the three carriers the store holds."""
    statuses = {**CARRIER_STATUSES, **{row.task_id: row.status for row in fixture}}
    return {task_id: status.value for task_id, status in statuses.items()}


def _carrier_status(record: object, statuses: Mapping[int, str]) -> TaskStatus | None:
    if not isinstance(record, CarriedByDeferral) or record.carrier_task_id not in statuses:
        return None
    return TaskStatus(statuses[record.carrier_task_id])


def _holder_liveness(record: object) -> Liveness | None:
    if not isinstance(record, HeldBySessionDeferral):
        return None
    try:
        return liveness(record.holder)
    except ProcessIdentityUnreadable:
        return None


def expected_needs_human(
    fixture: Iterable[FixtureRow], *, now: datetime
) -> dict[int, NeedsHumanReason | None]:
    """``task_deferral.needs_human`` for each DEFERRED row, read against the fixture's store."""
    rows = tuple(fixture)
    statuses = status_map(rows)
    verdicts: dict[int, NeedsHumanReason | None] = {}
    for row in rows:
        if row.status != TaskStatus.DEFERRED:
            continue
        record = parse_record(row.record_json)
        verdicts[row.task_id] = needs_human(
            record, now, _carrier_status(record, statuses), _holder_liveness(record)
        )
    return verdicts
