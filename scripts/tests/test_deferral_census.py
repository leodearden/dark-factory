"""Tests for ``scripts/deferral_census.py``: every deferred row classified with
the shared deferral model.

The classifier section is pure, over ``deferral_store.TaskRow.from_wire`` rows
and a fixed NOW. Every record is built with ``shared.task_deferral``'s own
models and ``record_to_json``, never hand-written JSON.
"""

from __future__ import annotations

import json
import os
from collections.abc import Mapping
from datetime import UTC, datetime, timedelta
from pathlib import Path
from typing import Any

import deferral_census
import pytest
from deferral_census import RowVerdict, classify_deferred, fails_check, referents
from deferral_store import TaskRow
from shared.process_identity import Liveness, ProcessIdentity, capture
from shared.task_deferral import (
    LAPSE_OVERDUE_AFTER_SECS,
    CarriedByDeferral,
    Deferral,
    HeldBySessionDeferral,
    LegacyUnknownDeferral,
    NeedsHumanReason,
    PlanningDeferral,
    UntilConditionDeferral,
    record_to_json,
)
from shared.task_statuses import TaskStatus

NOW = datetime(2026, 10, 19, 12, 0, tzinfo=UTC)
DAY = timedelta(days=1)
HOLDER = ProcessIdentity(pid=4242, start_ticks=99, boot_id='boot-a', host='factory-host')
CARRIER_STATUSES: Mapping[int, TaskStatus] = {
    900: TaskStatus.DONE,
    901: TaskStatus.CANCELLED,
    902: TaskStatus.PENDING,
}


def _stamps(stamped_at: datetime = NOW - DAY) -> dict[str, Any]:
    return {'deferred_at': stamped_at, 'stamped_at': stamped_at, 'stamped_by': 'claude-task-1'}


def carried_by(carrier: int) -> Deferral:
    return CarriedByDeferral(carrier_task_id=carrier, **_stamps())


def until_condition(*, expires_at: datetime | None, stamped_at: datetime = NOW - DAY) -> Deferral:
    return UntilConditionDeferral(condition='upstream lands', expires_at=expires_at, **_stamps(stamped_at))


def held_by_session(holder: ProcessIdentity = HOLDER) -> Deferral:
    return HeldBySessionDeferral(holder=holder, **_stamps())


def planning(*, stamped_at: datetime) -> Deferral:
    return PlanningDeferral(**_stamps(stamped_at))


def legacy_unknown() -> Deferral:
    return LegacyUnknownDeferral(deferred_at=None, stamped_at=NOW - DAY, stamped_by='migrate-deferrals')


def wire_row(
    task_id: int,
    *,
    record: Deferral | None = None,
    status: str = 'deferred',
    metadata: object = None,
    **extra: Any,
) -> dict[str, Any]:
    if metadata is None:
        metadata = dict(extra)
        if record is not None:
            metadata['deferral'] = record_to_json(record)
    return {'id': str(task_id), 'title': f'task {task_id}', 'status': status, 'metadata': metadata}


def row(task_id: int, record: Deferral | None = None, **kwargs: Any) -> TaskRow:
    return TaskRow.from_wire(wire_row(task_id, record=record, **kwargs))


def classify_one(
    task_row: TaskRow, *, liveness: Mapping[int, Liveness | None] | None = None
) -> RowVerdict:
    (verdict,) = classify_deferred(
        [task_row], now=NOW, carrier_statuses=CARRIER_STATUSES, holder_liveness=liveness or {}
    )
    return verdict


class TestClassifyDeferred:
    @pytest.mark.parametrize(
        ('carrier', 'expected'),
        [
            (900, NeedsHumanReason.CLOSABLE),
            (901, NeedsHumanReason.ORPHANED),
            (902, None),
            (903, NeedsHumanReason.CARRIER_MISSING),
        ],
    )
    def test_carried_by_reads_its_carriers_status(
        self, carrier: int, expected: NeedsHumanReason | None
    ) -> None:
        verdict = classify_one(row(10, carried_by(carrier)))

        assert verdict.held == 'carried_by'
        assert verdict.needs_human is expected

    @pytest.mark.parametrize(
        ('record', 'expected'),
        [
            (until_condition(expires_at=NOW + DAY), None),
            (
                until_condition(expires_at=NOW - timedelta(seconds=LAPSE_OVERDUE_AFTER_SECS + 1)),
                NeedsHumanReason.EXPIRY_OVERDUE,
            ),
            (until_condition(expires_at=None, stamped_at=NOW - 31 * DAY), NeedsHumanReason.STALE_CONDITION),
        ],
        ids=['future-expiry', 'expiry-overdue', 'stale-condition'],
    )
    def test_until_condition(self, record: Deferral, expected: NeedsHumanReason | None) -> None:
        verdict = classify_one(row(11, record))

        assert verdict.held == 'until_condition'
        assert verdict.needs_human is expected

    @pytest.mark.parametrize(
        ('liveness', 'expected'),
        [
            ({12: Liveness.ALIVE}, None),
            ({12: Liveness.DEAD}, NeedsHumanReason.HOLDER_DEAD),
            ({12: Liveness.OTHER_HOST}, NeedsHumanReason.HOLDER_OTHER_HOST),
            ({12: None}, NeedsHumanReason.LIVENESS_UNKNOWN),
            ({}, NeedsHumanReason.LIVENESS_UNKNOWN),
        ],
        ids=['alive', 'dead', 'other-host', 'unread', 'absent'],
    )
    def test_held_by_session_reads_its_holders_liveness(
        self, liveness: Mapping[int, Liveness | None], expected: NeedsHumanReason | None
    ) -> None:
        verdict = classify_one(row(12, held_by_session()), liveness=liveness)

        assert verdict.held == 'held_by_session'
        assert verdict.needs_human is expected

    @pytest.mark.parametrize(
        ('age', 'expected'),
        [(timedelta(minutes=5), None), (timedelta(hours=25), NeedsHumanReason.STALE_PLANNING)],
        ids=['fresh', 'stale'],
    )
    def test_planning(self, age: timedelta, expected: NeedsHumanReason | None) -> None:
        verdict = classify_one(row(13, planning(stamped_at=NOW - age)))

        assert verdict.held == 'planning'
        assert verdict.needs_human is expected

    def test_legacy_unknown_needs_a_human(self) -> None:
        verdict = classify_one(row(14, legacy_unknown()))

        assert verdict.held == 'legacy_unknown'
        assert verdict.needs_human is NeedsHumanReason.LEGACY_UNKNOWN
        assert verdict.deflake_owner is False

    def test_a_deflake_owner_is_marked(self) -> None:
        verdict = classify_one(row(15, legacy_unknown(), spawn_context=deferral_census.DEFLAKE_SPAWN_CONTEXT))

        assert verdict.needs_human is NeedsHumanReason.LEGACY_UNKNOWN
        assert verdict.deflake_owner is True

    def test_no_record_is_missing(self) -> None:
        verdict = classify_one(row(16, x_coalesced_into=5255))

        assert verdict.held == 'no record'
        assert verdict.needs_human is NeedsHumanReason.MISSING_RECORD

    def test_a_malformed_record_is_invalid(self) -> None:
        malformed = TaskRow.from_wire(
            wire_row(17, metadata={'deferral': {'kind': 'carried_by', 'carrier_task_id': 'soon'}})
        )

        verdict = classify_one(malformed)

        assert verdict.held == 'invalid record'
        assert verdict.needs_human is NeedsHumanReason.INVALID_RECORD

    @pytest.mark.parametrize('metadata', ['[1, 2]', '"text"', '{not json'])
    def test_unreadable_wire_metadata_is_an_invalid_record(self, metadata: str) -> None:
        verdict = classify_one(TaskRow.from_wire(wire_row(18, metadata=metadata)))

        assert verdict.held == 'invalid record'
        assert verdict.needs_human is NeedsHumanReason.INVALID_RECORD
        assert verdict.deflake_owner is False

    def test_a_json_object_string_is_read_as_metadata(self) -> None:
        text = json.dumps({'deferral': record_to_json(legacy_unknown())})

        verdict = classify_one(TaskRow.from_wire(wire_row(19, metadata=text)))

        assert verdict.held == 'legacy_unknown'

    def test_one_verdict_per_row_in_order_carrying_id_and_title(self) -> None:
        rows = [row(21, carried_by(900)), row(20), row(22, legacy_unknown())]

        verdicts = classify_deferred(rows, now=NOW, carrier_statuses=CARRIER_STATUSES, holder_liveness={})

        assert [(v.task_id, v.title) for v in verdicts] == [
            (21, 'task 21'),
            (20, 'task 20'),
            (22, 'task 22'),
        ]


class TestFailsCheck:
    FAILING = frozenset(
        {NeedsHumanReason.MISSING_RECORD, NeedsHumanReason.INVALID_RECORD, NeedsHumanReason.EXPIRY_OVERDUE}
    )

    @pytest.mark.parametrize('reason', [*NeedsHumanReason, None])
    def test_only_missing_invalid_and_overdue_fail(self, reason: NeedsHumanReason | None) -> None:
        verdict = RowVerdict(task_id=1, title='t', held='x', needs_human=reason, deflake_owner=False)

        assert fails_check(verdict) is (reason in self.FAILING)


class TestReferents:
    def test_carriers_and_holders_of_the_recorded_rows(self) -> None:
        rows = [
            row(30, carried_by(900)),
            row(31, carried_by(901)),
            row(32, carried_by(900)),
            row(33, held_by_session()),
            row(34, until_condition(expires_at=NOW + DAY)),
            row(35),
            TaskRow.from_wire(wire_row(36, metadata='[]')),
        ]

        found = referents(rows)

        assert found.carrier_ids == frozenset({900, 901})
        assert dict(found.holders) == {33: HOLDER}


class TestHolderLiveness:
    def test_an_unreadable_proc_is_unknown_never_dead(self, tmp_path: Path) -> None:
        identity = capture(os.getpid())
        assert identity is not None
        empty_proc = tmp_path / 'proc'
        empty_proc.mkdir()

        assert deferral_census.holder_liveness(identity, proc_root=empty_proc) is None

    def test_this_process_is_alive(self) -> None:
        identity = capture(os.getpid())
        assert identity is not None

        assert deferral_census.holder_liveness(identity) is Liveness.ALIVE


class TestTaskRowFromWire:
    def test_parses_id_status_and_metadata(self) -> None:
        parsed = TaskRow.from_wire({'id': 7, 'title': 'seven', 'status': 'pending', 'metadata': {'a': 1}})

        assert (parsed.task_id, parsed.title, parsed.status) == (7, 'seven', TaskStatus.PENDING)
        assert parsed.metadata == {'a': 1}
        assert parsed.deferral is None

    @pytest.mark.parametrize(
        ('field', 'value'),
        [('id', 'abc'), ('id', '1.5'), ('id', True), ('status', 'parked'), ('title', None)],
    )
    def test_an_unusable_field_is_a_value_error_naming_it(self, field: str, value: object) -> None:
        wire = wire_row(1)
        wire[field] = value

        with pytest.raises(ValueError, match=field):
            TaskRow.from_wire(wire)
