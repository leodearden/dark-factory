"""Tests for ``scripts/exception_owner_liveness.py`` — is an INV-12 debt owner alive.

Under test, through public names only: the immutable task-row reads from a
real temp tasks.db (``make_tasks_db`` under ``<root>/.taskmaster/tasks``), the
successor links a row carries, the parse of ``resolve_ticket``'s wire result,
and ``resolve_owner``'s verdicts.  The wire values are the ones measured live
on 2026-10-16 (plan analysis of task 5608).
"""

from __future__ import annotations

import dataclasses
import datetime
from pathlib import Path

import _task_db_scan
import pytest
from exception_owner_liveness import (
    TaskRow,
    TicketOutcome,
    TicketState,
    TicketUnreadable,
    read_tasks,
    read_tasks_carrying,
    ticket_outcome,
)
from shared.task_deferral import CarriedByDeferral, PlanningDeferral, record_to_json
from shared.task_statuses import OwnerLiveness, owner_liveness

STAMPED_AT = datetime.datetime(2026, 10, 16, 6, 0, tzinfo=datetime.UTC)
FIELD = 'x_inv12_finding'


@pytest.fixture
def store_dir(tmp_path: Path) -> Path:
    directory = tmp_path / '.taskmaster' / 'tasks'
    directory.mkdir(parents=True)
    return directory


def _carried_by(carrier: int) -> dict:
    return record_to_json(
        CarriedByDeferral(carrier_task_id=carrier, deferred_at=STAMPED_AT, stamped_at=STAMPED_AT)
    )


def _row(status: str = 'pending', **metadata: object) -> TaskRow:
    return TaskRow(id=10, status=status, title='t', metadata=metadata)


# ---------------------------------------------------------------------------
# (1) read_tasks


def test_read_tasks_returns_only_the_requested_master_rows(make_tasks_db, store_dir):
    db = make_tasks_db(
        [
            {'id': 1, 'status': 'pending', 'title': 'one', 'metadata': {'a': 1}},
            {'id': 2, 'status': 'done'},
            {'id': 3, 'status': 'pending', 'tag': 'other'},
        ],
        directory=store_dir,
    )
    rows = read_tasks(db, [1, 3, 99])
    assert set(rows) == {1}
    assert rows[1] == TaskRow(id=1, status='pending', title='one', metadata={'a': 1})


def test_a_task_row_is_frozen_and_its_metadata_read_only(make_tasks_db, store_dir):
    db = make_tasks_db([{'id': 1, 'metadata': {'a': 1}}], directory=store_dir)
    row = read_tasks(db, [1])[1]
    with pytest.raises(dataclasses.FrozenInstanceError):
        setattr(row, 'status', 'done')
    with pytest.raises(TypeError):
        row.metadata['a'] = 2
    assert isinstance(row.id, int)


def test_malformed_metadata_reads_as_empty(make_tasks_db, store_dir):
    db = make_tasks_db([{'id': 1, 'metadata': '{"truncated": '}], directory=store_dir)
    assert dict(read_tasks(db, [1])[1].metadata) == {}


def test_no_ids_reads_nothing_and_opens_nothing(tmp_path: Path):
    assert read_tasks(tmp_path / 'absent.db', []) == {}


# ---------------------------------------------------------------------------
# (2) read_tasks_carrying


def test_read_tasks_carrying_keeps_exactly_the_rows_holding_a_value(make_tasks_db, store_dir):
    db = make_tasks_db(
        [
            {'id': 7, 'metadata': {FIELD: {'kind': 'tighten'}}},
            {'id': 3, 'metadata': {FIELD: 'anything'}},
            {'id': 4, 'metadata': {FIELD: None}},
            {'id': 5, 'metadata': {'note': f'mentions "{FIELD}" in prose'}},
            {'id': 6, 'metadata': '{"' + FIELD + '": '},
            {'id': 8, 'tag': 'other', 'metadata': {FIELD: 'x'}},
            {'id': 9},
        ],
        directory=store_dir,
    )
    carriers = read_tasks_carrying(db, FIELD)
    assert [row.id for row in carriers] == [3, 7]
    assert carriers[1].metadata[FIELD] == {'kind': 'tighten'}


# ---------------------------------------------------------------------------
# (3) Store refusal propagates.


def _absent(directory: Path) -> Path:
    return directory / 'tasks.db'


def _empty(directory: Path) -> Path:
    path = directory / 'tasks.db'
    path.write_bytes(b'')
    return path


def _a_directory(directory: Path) -> Path:
    path = directory / 'tasks.db'
    path.mkdir()
    return path


@pytest.mark.parametrize('make_path', [_absent, _empty, _a_directory])
def test_an_unreadable_store_propagates_as_an_unreadable_store_error(store_dir, make_path):
    path = make_path(store_dir)
    with pytest.raises(_task_db_scan.UNREADABLE_STORE_ERRORS):
        read_tasks(path, [1])
    with pytest.raises(_task_db_scan.UNREADABLE_STORE_ERRORS):
        read_tasks_carrying(path, FIELD)


# ---------------------------------------------------------------------------
# (4) TaskRow.successor


def test_a_carried_by_deferral_names_the_successor():
    assert _row('deferred', deferral=_carried_by(5400)).successor() == 5400


@pytest.mark.parametrize('value', [5311, '5311'])
def test_x_coalesced_into_names_the_successor(value):
    assert _row('done', x_coalesced_into=value).successor() == 5311


def test_the_deferral_record_wins_over_x_coalesced_into():
    row = _row('done', deferral=_carried_by(5400), x_coalesced_into=5311)
    assert row.successor() == 5400


def test_a_malformed_deferral_falls_back_to_x_coalesced_into():
    row = _row('done', deferral={'kind': 'carried_by'}, x_coalesced_into=5311)
    assert row.successor() == 5311


def test_a_deferral_that_is_not_carried_by_names_no_carrier():
    planning = record_to_json(PlanningDeferral(deferred_at=STAMPED_AT, stamped_at=STAMPED_AT))
    assert _row('deferred', deferral=planning).successor() is None


@pytest.mark.parametrize('value', ['abc', True, 0, -3, '0', 5311.0, None])
def test_no_usable_link_is_no_successor(value):
    assert _row('done', x_coalesced_into=value).successor() is None
    assert _row('done').successor() is None


# ---------------------------------------------------------------------------
# (5) TaskRow.liveness


@pytest.mark.parametrize('status', ['pending', 'in-progress', 'blocked', 'merge-deferred'])
def test_open_statuses_are_live(status):
    assert _row(status).liveness() is OwnerLiveness.OPEN


@pytest.mark.parametrize('status', ['done', 'cancelled', 'deferred', ''])
def test_terminal_deferred_and_empty_statuses_are_dead(status):
    verdict = _row(status).liveness()
    assert verdict is not OwnerLiveness.OPEN
    assert verdict is owner_liveness(status)


# ---------------------------------------------------------------------------
# (6) ticket_outcome


@pytest.mark.parametrize(
    ('result', 'expected'),
    [
        (
            {'status': 'created', 'task_id': '7001'},
            TicketOutcome(TicketState.RESOLVED_TO_TASK, 'created', 7001, None),
        ),
        (
            {'status': 'combined', 'task_id': '7002'},
            TicketOutcome(TicketState.RESOLVED_TO_TASK, 'combined', 7002, None),
        ),
        (
            {'status': 'created', 'task_id': 7003},
            TicketOutcome(TicketState.RESOLVED_TO_TASK, 'created', 7003, None),
        ),
        (
            {'status': 'failed', 'reason': 'timeout', 'task_id': None},
            TicketOutcome(TicketState.PENDING, 'failed', None, 'timeout'),
        ),
        ({'status': 'pending'}, TicketOutcome(TicketState.PENDING, 'pending', None, None)),
        (
            {'status': 'failed', 'reason': 'unknown_ticket', 'task_id': None},
            TicketOutcome(TicketState.DEAD, 'failed', None, 'unknown_ticket'),
        ),
        (
            {'status': 'failed', 'reason': 'worker_dead'},
            TicketOutcome(TicketState.DEAD, 'failed', None, 'worker_dead'),
        ),
        (
            {'status': 'refused', 'reason': 'premise_cancelled'},
            TicketOutcome(TicketState.DEAD, 'refused', None, 'premise_cancelled'),
        ),
        (
            {'status': 'cancelled', 'reason': 'user_cancelled'},
            TicketOutcome(TicketState.DEAD, 'cancelled', None, 'user_cancelled'),
        ),
    ],
)
def test_ticket_outcome_parses_each_wire_shape(result, expected):
    assert ticket_outcome(result) == expected


@pytest.mark.parametrize(
    'result',
    [
        {'error': 'connection refused', 'error_type': 'ConnectionError'},
        {'status': 'failed', 'reason': 'server_closed', 'task_id': None},
        {'status': 'failed', 'reason': 'ticket_store not configured', 'task_id': None},
        {'status': 'created'},
        {'status': 'created', 'task_id': 'abc'},
        {'status': 'exploded'},
        'not a dict',
    ],
)
def test_ticket_outcome_refuses_what_it_cannot_read(result):
    """An unreadable answer is never a verdict: INV-11 forbids reading an outage as death."""
    with pytest.raises(TicketUnreadable) as excinfo:
        ticket_outcome(result)
    assert excinfo.value.result == result


def test_a_ticket_outcome_is_frozen():
    outcome = ticket_outcome({'status': 'pending'})
    with pytest.raises(dataclasses.FrozenInstanceError):
        setattr(outcome, 'status', 'created')
