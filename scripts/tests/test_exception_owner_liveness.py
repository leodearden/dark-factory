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
import functools
from collections.abc import Callable, Mapping, MutableMapping
from pathlib import Path

import _task_db_scan
import pytest
from exception_owner_liveness import (
    MAX_SUCCESSOR_HOPS,
    DeathCause,
    Liveness,
    OwnerVerdict,
    TaskRow,
    TicketOutcome,
    TicketState,
    TicketUnreadable,
    read_tasks,
    read_tasks_carrying,
    resolve_owner,
    ticket_outcome,
)
from shared.governed_exceptions import TaskRef, TicketRef
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
    field = 'status'
    with pytest.raises(dataclasses.FrozenInstanceError):
        setattr(row, field, 'done')
    assert not isinstance(row.metadata, MutableMapping)
    assert isinstance(row.id, int)


def test_a_task_row_does_not_share_the_callers_metadata():
    metadata: dict[str, object] = {'a': 1}
    row = TaskRow(id=1, status='pending', title='t', metadata=metadata)
    metadata['a'] = 2
    assert row.metadata['a'] == 1


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
    field = 'status'
    with pytest.raises(dataclasses.FrozenInstanceError):
        setattr(outcome, field, 'created')


# ---------------------------------------------------------------------------
# resolve_owner: a real temp store plus a dict-backed resolve_ticket.

TICKET_PENDING = TicketRef('tkt_01')
TICKET_UNKNOWN = TicketRef('tkt_02')
TICKET_FAILED = TicketRef('tkt_03')
TICKET_REFUSED = TicketRef('tkt_04')
TICKET_CREATED = TicketRef('tkt_05')
TICKET_COMBINED = TicketRef('tkt_06')
TICKET_CANCELLED = TicketRef('tkt_07')

_TICKET_WIRE: Mapping[TicketRef, dict] = {
    TICKET_PENDING: {'status': 'failed', 'reason': 'timeout', 'task_id': None},
    TICKET_UNKNOWN: {'status': 'failed', 'reason': 'unknown_ticket', 'task_id': None},
    TICKET_FAILED: {'status': 'failed', 'reason': 'worker_dead'},
    TICKET_REFUSED: {'status': 'refused', 'reason': 'premise_cancelled'},
    TICKET_CREATED: {'status': 'created', 'task_id': '20'},
    TICKET_COMBINED: {'status': 'combined', 'task_id': '20'},
    TICKET_CANCELLED: {'status': 'cancelled', 'reason': 'user_cancelled'},
}


def _resolve_ticket(ref: TicketRef) -> TicketOutcome:
    return ticket_outcome(_TICKET_WIRE[ref])


def _resolver(make_tasks_db, store_dir: Path, rows: list[dict]) -> Callable[..., OwnerVerdict]:
    db = make_tasks_db(rows, directory=store_dir)
    return functools.partial(
        resolve_owner,
        read_tasks=functools.partial(read_tasks, db),
        resolve_ticket=_resolve_ticket,
    )


def test_a_pending_task_owner_is_live(make_tasks_db, store_dir):
    verdict = _resolver(make_tasks_db, store_dir, [{'id': 10, 'status': 'pending'}])(TaskRef(10))
    assert verdict.owner == TaskRef(10)
    assert verdict.liveness is Liveness.LIVE
    assert verdict.owning_task == 10
    assert verdict.path == (10,)
    assert verdict.cause is None


def test_an_absent_task_owner_is_dead(make_tasks_db, store_dir):
    """Scenario 17: an owner the store does not hold."""
    verdict = _resolver(make_tasks_db, store_dir, [{'id': 10, 'status': 'pending'}])(TaskRef(99))
    assert verdict.liveness is Liveness.DEAD
    assert verdict.cause is DeathCause.OWNER_ABSENT
    assert verdict.owning_task is None


@pytest.mark.parametrize('status', ['done', 'cancelled', 'deferred'])
def test_a_closed_owner_with_no_successor_is_dead_and_says_how(make_tasks_db, store_dir, status):
    verdict = _resolver(make_tasks_db, store_dir, [{'id': 10, 'status': status}])(TaskRef(10))
    assert verdict.liveness is Liveness.DEAD
    assert verdict.cause is DeathCause.OWNER_CLOSED
    assert verdict.closed_status == status
    assert verdict.owning_task is None


def test_a_closed_owner_lives_through_a_coalesced_successor(make_tasks_db, store_dir):
    """Scenario 18."""
    resolve = _resolver(
        make_tasks_db,
        store_dir,
        [
            {'id': 10, 'status': 'done', 'metadata': {'x_coalesced_into': 11}},
            {'id': 11, 'status': 'pending'},
        ],
    )
    verdict = resolve(TaskRef(10))
    assert verdict.liveness is Liveness.LIVE_VIA_SUCCESSOR
    assert verdict.owning_task == 11
    assert verdict.path == (10, 11)
    assert verdict.cause is None


def test_a_deferred_owner_lives_through_its_carrier(make_tasks_db, store_dir):
    resolve = _resolver(
        make_tasks_db,
        store_dir,
        [
            {'id': 10, 'status': 'deferred', 'metadata': {'deferral': _carried_by(11)}},
            {'id': 11, 'status': 'in-progress'},
        ],
    )
    verdict = resolve(TaskRef(10))
    assert verdict.liveness is Liveness.LIVE_VIA_SUCCESSOR
    assert verdict.owning_task == 11


def test_a_successor_chain_is_followed_to_its_first_live_task(make_tasks_db, store_dir):
    resolve = _resolver(
        make_tasks_db,
        store_dir,
        [
            {'id': 10, 'status': 'done', 'metadata': {'x_coalesced_into': 11}},
            {'id': 11, 'status': 'cancelled', 'metadata': {'x_coalesced_into': '12'}},
            {'id': 12, 'status': 'pending'},
        ],
    )
    verdict = resolve(TaskRef(10))
    assert verdict.liveness is Liveness.LIVE_VIA_SUCCESSOR
    assert verdict.owning_task == 12
    assert verdict.path == (10, 11, 12)


def test_a_successor_cycle_terminates_dead(make_tasks_db, store_dir):
    resolve = _resolver(
        make_tasks_db,
        store_dir,
        [
            {'id': 10, 'status': 'done', 'metadata': {'x_coalesced_into': 11}},
            {'id': 11, 'status': 'done', 'metadata': {'x_coalesced_into': 10}},
        ],
    )
    verdict = resolve(TaskRef(10))
    assert verdict.liveness is Liveness.DEAD
    assert verdict.cause is DeathCause.SUCCESSOR_UNRESOLVED
    assert verdict.path == (10, 11)


def test_a_successor_the_store_does_not_hold_is_unresolved(make_tasks_db, store_dir):
    resolve = _resolver(
        make_tasks_db,
        store_dir,
        [{'id': 10, 'status': 'done', 'metadata': {'x_coalesced_into': 11}}],
    )
    verdict = resolve(TaskRef(10))
    assert verdict.cause is DeathCause.SUCCESSOR_UNRESOLVED
    assert verdict.closed_status == 'done'


def _chain(hops: int) -> list[dict]:
    """Task 100 handing on through *hops* successor links to a live last task."""
    closed = [
        {'id': 100 + index, 'status': 'done', 'metadata': {'x_coalesced_into': 101 + index}}
        for index in range(hops)
    ]
    return [*closed, {'id': 100 + hops, 'status': 'pending'}]


def test_a_chain_of_exactly_the_bound_is_followed(make_tasks_db, store_dir):
    resolve = _resolver(make_tasks_db, store_dir, _chain(MAX_SUCCESSOR_HOPS))
    verdict = resolve(TaskRef(100))
    assert verdict.liveness is Liveness.LIVE_VIA_SUCCESSOR
    assert verdict.owning_task == 100 + MAX_SUCCESSOR_HOPS


def test_a_chain_longer_than_the_bound_is_unresolved(make_tasks_db, store_dir):
    resolve = _resolver(make_tasks_db, store_dir, _chain(MAX_SUCCESSOR_HOPS + 1))
    verdict = resolve(TaskRef(100))
    assert verdict.liveness is Liveness.DEAD
    assert verdict.cause is DeathCause.SUCCESSOR_UNRESOLVED


def test_a_pending_ticket_owner_is_live_with_no_task_yet(make_tasks_db, store_dir):
    verdict = _resolver(make_tasks_db, store_dir, [])(TICKET_PENDING)
    assert verdict.liveness is Liveness.LIVE
    assert verdict.owning_task is None
    assert verdict.path == ()
    assert verdict.ticket == _resolve_ticket(TICKET_PENDING)


@pytest.mark.parametrize(
    ('ticket', 'cause'),
    [
        (TICKET_UNKNOWN, DeathCause.TICKET_ABSENT),
        (TICKET_FAILED, DeathCause.TICKET_FAILED),
        (TICKET_REFUSED, DeathCause.TICKET_REFUSED),
        (TICKET_CANCELLED, DeathCause.TICKET_CANCELLED),
    ],
)
def test_a_ticket_that_yields_no_task_is_a_dead_owner(make_tasks_db, store_dir, ticket, cause):
    """Scenario 17's ticket owners."""
    verdict = _resolver(make_tasks_db, store_dir, [])(ticket)
    assert verdict.liveness is Liveness.DEAD
    assert verdict.cause is cause
    assert verdict.ticket == _resolve_ticket(ticket)


def test_a_created_ticket_is_owned_by_its_task(make_tasks_db, store_dir):
    verdict = _resolver(make_tasks_db, store_dir, [{'id': 20, 'status': 'pending'}])(TICKET_CREATED)
    assert verdict.liveness is Liveness.LIVE
    assert verdict.owning_task == 20
    assert verdict.path == (20,)


def test_a_ticket_whose_task_was_cancelled_is_a_dead_owner(make_tasks_db, store_dir):
    """Scenario 17: a TicketRef resolving to a cancelled task."""
    verdict = _resolver(make_tasks_db, store_dir, [{'id': 20, 'status': 'cancelled'}])(TICKET_CREATED)
    assert verdict.liveness is Liveness.DEAD
    assert verdict.cause is DeathCause.OWNER_CLOSED
    assert verdict.closed_status == 'cancelled'


def test_a_combined_ticket_follows_its_task_to_a_successor(make_tasks_db, store_dir):
    resolve = _resolver(
        make_tasks_db,
        store_dir,
        [
            {'id': 20, 'status': 'done', 'metadata': {'x_coalesced_into': 21}},
            {'id': 21, 'status': 'pending'},
        ],
    )
    verdict = resolve(TICKET_COMBINED)
    assert verdict.liveness is Liveness.LIVE_VIA_SUCCESSOR
    assert verdict.owning_task == 21
    assert verdict.path == (20, 21)


def test_an_unreadable_ticket_propagates_and_is_never_a_verdict(make_tasks_db, store_dir):
    """INV-11: a server that cannot answer must not mass-produce dead owners."""
    db = make_tasks_db([], directory=store_dir)

    def unreadable(ref: TicketRef) -> TicketOutcome:
        return ticket_outcome({'status': 'failed', 'reason': 'server_closed', 'task_id': None})

    with pytest.raises(TicketUnreadable):
        resolve_owner(
            TICKET_CREATED,
            read_tasks=functools.partial(read_tasks, db),
            resolve_ticket=unreadable,
        )


@pytest.mark.parametrize(
    ('liveness', 'cause'),
    [
        (Liveness.DEAD, None),
        (Liveness.LIVE, DeathCause.OWNER_CLOSED),
        (Liveness.LIVE_VIA_SUCCESSOR, DeathCause.OWNER_ABSENT),
    ],
)
def test_a_verdict_has_a_cause_exactly_when_it_is_dead(liveness, cause):
    with pytest.raises(ValueError) as excinfo:
        OwnerVerdict(
            owner=TaskRef(10),
            liveness=liveness,
            owning_task=None,
            path=(10,),
            cause=cause,
            closed_status=None,
            ticket=None,
        )
    assert str(liveness) in str(excinfo.value)
    assert str(cause) in str(excinfo.value)
