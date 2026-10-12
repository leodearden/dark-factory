"""The recurrence chain-state gauge (docs/prds/recurring-deterministic-tasks.md R-D5/C-4).

One state per chain, read from the chain's current link: blocked is paused, done
is broken, cancelled is ended, and every other status is judged by its fire time.
"""

from __future__ import annotations

import json
from datetime import UTC, datetime, timedelta
from pathlib import Path
from typing import Any

import pytest
from _tasks_db_fixture import task_row, write_tasks_db
from shared.task_statuses import TaskStatus
from shared.task_tags import DEFAULT_TAG

from dashboard.data.db import DbPool
from dashboard.data.escalation_corpus import PendingEscalations
from dashboard.data.recurrence_gauge import ChainEntry, ChainState, chain_state, classify_chains
from dashboard.data.root_invariants import scan_root_invariants

NOW = datetime(2026, 10, 12, 12, 0, tzinfo=UTC)
HOUR = timedelta(hours=1)


def link(
    link_id: int,
    status: str,
    at: datetime | str = NOW + HOUR,
    interval_secs: int | None = 3600,
    key: str = 'nightly-job',
    *,
    tag: str = 'master',
) -> dict[str, Any]:
    """One decoded chain link whose dated milestone fires at *at*."""
    recurrence: dict[str, Any] = {'key': key}
    if interval_secs is not None:
        recurrence['interval_secs'] = interval_secs
    at_text = at if isinstance(at, str) else at.isoformat()
    return {
        'tag': tag,
        'id': link_id,
        'status': status,
        'metadata': {'recurrence': recurrence, 'milestone': {'mode': 'dated', 'at': at_text}},
    }


def _with_milestone(task: dict[str, Any], milestone: dict[str, Any] | None) -> dict[str, Any]:
    metadata = {k: v for k, v in task['metadata'].items() if k != 'milestone'}
    if milestone is not None:
        metadata['milestone'] = milestone
    return {**task, 'metadata': metadata}


@pytest.mark.parametrize(
    ('status', 'expected'),
    [('blocked', ChainState.PAUSED), ('done', ChainState.BROKEN), ('cancelled', ChainState.ENDED)],
)
@pytest.mark.parametrize('at', [NOW + HOUR, NOW - 48 * HOUR])
def test_status_decides_regardless_of_fire_time(status, expected, at):
    assert chain_state(link(1, status, at), now=NOW) is expected


@pytest.mark.parametrize(
    ('at', 'expected'),
    [
        (NOW + HOUR, ChainState.SCHEDULED),
        (NOW - timedelta(minutes=30), ChainState.SCHEDULED),
        (NOW - HOUR, ChainState.SCHEDULED),
        (NOW - HOUR - timedelta(seconds=1), ChainState.OVERDUE),
    ],
)
def test_pending_link_is_overdue_only_past_fire_time_plus_one_interval(at, expected):
    assert chain_state(link(1, 'pending', at, interval_secs=3600), now=NOW) is expected


@pytest.mark.parametrize(
    ('at', 'expected'),
    [(NOW - 25 * HOUR, ChainState.OVERDUE), (NOW - 23 * HOUR, ChainState.SCHEDULED)],
)
def test_grace_is_capped_at_24_hours(at, expected):
    assert chain_state(link(1, 'pending', at, interval_secs=7 * 86400), now=NOW) is expected


@pytest.mark.parametrize(
    ('at', 'expected'),
    [(NOW - 25 * HOUR, ChainState.OVERDUE), (NOW - 23 * HOUR, ChainState.SCHEDULED)],
)
def test_an_invalid_recurrence_takes_the_24_hour_cap(at, expected):
    assert chain_state(link(1, 'pending', at, interval_secs=None), now=NOW) is expected


@pytest.mark.parametrize(
    ('at', 'expected'),
    [(NOW - 2 * HOUR, ChainState.OVERDUE), (NOW + HOUR, ChainState.SCHEDULED)],
)
def test_a_naive_fire_time_is_read_as_utc(at, expected):
    naive = at.replace(tzinfo=None).isoformat()
    assert chain_state(link(1, 'pending', naive, interval_secs=3600), now=NOW) is expected


@pytest.mark.parametrize(
    'milestone',
    [None, {'mode': 'delayed', 'after_secs': 60}, {'mode': 'dated', 'at': 'not-a-date'}],
    ids=['absent', 'delayed', 'unparseable'],
)
def test_an_unreadable_fire_time_is_overdue(milestone):
    task = _with_milestone(link(1, 'pending'), milestone)
    assert chain_state(task, now=NOW) is ChainState.OVERDUE


@pytest.mark.parametrize('status', ['in-progress', 'deferred'])
@pytest.mark.parametrize(
    ('at', 'expected'),
    [(NOW - 2 * HOUR, ChainState.OVERDUE), (NOW + HOUR, ChainState.SCHEDULED)],
)
def test_other_live_statuses_follow_the_fire_time_rule(status, at, expected):
    assert chain_state(link(1, status, at, interval_secs=3600), now=NOW) is expected


@pytest.mark.parametrize('status', [*TaskStatus, 'some-status-no-one-defined'])
@pytest.mark.parametrize('at', [NOW + HOUR, NOW - 48 * HOUR])
def test_every_status_resolves_to_exactly_one_state(status, at):
    assert isinstance(chain_state(link(1, str(status), at), now=NOW), ChainState)


@pytest.mark.parametrize(
    ('at', 'interval_secs', 'expected'),
    [
        ('9999-12-31T23:59:59+00:00', 3600, ChainState.SCHEDULED),
        (NOW - 25 * HOUR, 10**20, ChainState.OVERDUE),
    ],
)
def test_extreme_fire_times_and_intervals_still_resolve(at, interval_secs, expected):
    assert chain_state(link(1, 'pending', at, interval_secs=interval_secs), now=NOW) is expected


def raw(task: dict[str, Any]) -> dict[str, Any]:
    """*task* as sqlite hands it back: metadata is JSON text."""
    return {**task, 'metadata': json.dumps(task['metadata'])}


def _classify(*tasks: dict[str, Any], pending_escalations=None) -> tuple[ChainEntry, ...]:
    return classify_chains(
        [raw(task) for task in tasks], now=NOW, pending_escalations=pending_escalations
    )


def _states(entries: tuple[ChainEntry, ...]) -> list[tuple[str, str, ChainState, int]]:
    return [(entry.tag, entry.key, entry.state, entry.link_id) for entry in entries]


def test_a_renewed_chain_is_read_from_its_live_successor_not_its_done_link():
    entries = _classify(link(1, 'done', NOW - 48 * HOUR), link(2, 'pending', NOW + HOUR))
    assert _states(entries) == [('master', 'nightly-job', ChainState.SCHEDULED, 2)]


def test_a_done_link_with_no_successor_is_broken():
    entries = _classify(link(3, 'done'))
    assert _states(entries) == [('master', 'nightly-job', ChainState.BROKEN, 3)]


def test_the_newest_terminal_link_decides_a_chain_with_no_live_link():
    entries = _classify(
        link(4, 'done', key='stopped-job'),
        link(5, 'cancelled', key='stopped-job'),
        link(6, 'cancelled', key='crashed-job'),
        link(7, 'done', key='crashed-job'),
    )
    assert _states(entries) == [
        ('master', 'crashed-job', ChainState.BROKEN, 7),
        ('master', 'stopped-job', ChainState.ENDED, 5),
    ]


def test_each_tag_is_classified_from_its_own_rows():
    entries = _classify(link(5, 'pending', tag='master'), link(9, 'done', tag='feature'))
    assert _states(entries) == [
        ('feature', 'nightly-job', ChainState.BROKEN, 9),
        ('master', 'nightly-job', ChainState.SCHEDULED, 5),
    ]


def test_rows_that_carry_no_chain_key_are_ignored():
    keyed = link(1, 'pending')
    no_key = {**keyed, 'id': 2, 'metadata': {'recurrence': {'interval_secs': 3600}}}
    blank_key = {**keyed, 'id': 3, 'metadata': {'recurrence': {'key': '', 'interval_secs': 60}}}
    not_a_mapping = {**keyed, 'id': 4, 'metadata': {'recurrence': 'nightly-job'}}
    rows = [raw(task) for task in (keyed, no_key, blank_key, not_a_mapping)]
    rows += [
        {**keyed, 'id': 5, 'status': 'done', 'metadata': '{not json'},
        {**keyed, 'id': 6, 'status': 'done', 'metadata': None},
    ]
    entries = classify_chains(rows, now=NOW, pending_escalations=None)
    assert _states(entries) == [('master', 'nightly-job', ChainState.SCHEDULED, 1)]


def test_entries_are_sorted_by_tag_then_key():
    entries = _classify(
        link(1, 'pending', key='zeta-job', tag='master'),
        link(2, 'pending', key='alpha-job', tag='master'),
        link(3, 'pending', key='zeta-job', tag='feature'),
        link(4, 'pending', key='alpha-job', tag='feature'),
    )
    assert [(entry.tag, entry.key) for entry in entries] == [
        ('feature', 'alpha-job'),
        ('feature', 'zeta-job'),
        ('master', 'alpha-job'),
        ('master', 'zeta-job'),
    ]


def test_each_entry_names_its_links_pending_escalations():
    entries = _classify(
        link(4, 'blocked', key='failing-job'),
        link(8, 'pending', key='nightly-job'),
        pending_escalations=PendingEscalations({'4': ('esc-4-1',)}),
    )
    assert {entry.key: entry.pending_escalation_ids for entry in entries} == {
        'failing-job': ('esc-4-1',),
        'nightly-job': (),
    }


def test_a_chain_outside_the_queue_tag_has_an_unknown_pinning_escalation():
    entries = _classify(
        link(4, 'blocked', tag='feature'),
        link(4, 'blocked', tag=DEFAULT_TAG),
        pending_escalations=PendingEscalations({'4': ('esc-4-1',)}),
    )
    assert {entry.tag: entry.pending_escalation_ids for entry in entries} == {
        'feature': None,
        DEFAULT_TAG: ('esc-4-1',),
    }


def test_an_unknown_escalation_lookup_is_none_on_every_entry():
    entries = _classify(link(4, 'blocked', key='failing-job'), link(8, 'pending'))
    assert [entry.pending_escalation_ids for entry in entries] == [None, None]


def test_an_entry_on_the_wire():
    at = NOW - 3 * HOUR
    (entry,) = _classify(
        link(4, 'blocked', at), pending_escalations=PendingEscalations({'4': ('esc-4-1',)})
    )
    assert entry.to_wire() == {
        'tag': 'master',
        'key': 'nightly-job',
        'state': 'paused',
        'link_id': 4,
        'link_status': 'blocked',
        'at': at.isoformat(),
        'pending_escalation_ids': ['esc-4-1'],
    }


def test_an_entry_without_a_fire_time_or_escalation_lookup_on_the_wire():
    task = _with_milestone(link(4, 'pending'), None)
    (entry,) = _classify(task)
    wire = entry.to_wire()
    assert wire['state'] == 'overdue'
    assert wire['at'] is None
    assert wire['pending_escalation_ids'] is None


def _db_path(root: Path) -> Path:
    return root / '.taskmaster' / 'tasks' / 'tasks.db'


def _row(task: dict[str, Any]) -> dict[str, Any]:
    return task_row(task['id'], task['status'], tag=task['tag'], metadata=task['metadata'])


async def _scan_root(root: Path, rows: list[dict[str, Any]]) -> dict[str, Any]:
    db_path = write_tasks_db(_db_path(root), rows)
    pool = DbPool()
    try:
        conn = await pool.get(db_path)
        record = await scan_root_invariants(
            root, db_path, conn, now=NOW, pending_escalations=PendingEscalations({})
        )
    finally:
        await pool.close_all()
    return record.to_wire()


async def test_an_ok_root_lists_its_chains_with_their_states(tmp_path):
    rows = [
        _row(link(1, 'pending', NOW + HOUR, key='future-job')),
        _row(link(2, 'pending', NOW - 3 * HOUR, key='late-job')),
        _row(link(3, 'done', key='crashed-mint')),
        task_row(4, 'pending'),
    ]
    wire = await _scan_root(tmp_path / 'proj', rows)
    assert wire['status'] == 'ok'
    assert [(c['key'], c['state'], c['link_id']) for c in wire['chains']] == [
        ('crashed-mint', 'broken', 3),
        ('future-job', 'scheduled', 1),
        ('late-job', 'overdue', 2),
    ]
    assert all(c['pending_escalation_ids'] == [] for c in wire['chains'])


async def test_a_root_with_no_chain_carriers_measures_none(tmp_path):
    wire = await _scan_root(tmp_path / 'proj', [task_row(1, 'pending'), task_row(2, 'done')])
    assert wire['status'] == 'ok'
    assert wire['chains'] == []


async def test_metadata_that_only_mentions_recurrence_is_not_a_chain(tmp_path):
    row = task_row(1, 'done', metadata={'note': 'add recurrence later', 'recurrence_key': 'x'})
    wire = await _scan_root(tmp_path / 'proj', [row])
    assert wire['chains'] == []


async def test_an_unopened_root_makes_no_chain_claim(tmp_path):
    root = tmp_path / 'proj'
    record = await scan_root_invariants(
        root, _db_path(root), None, now=NOW, pending_escalations=PendingEscalations({})
    )
    assert record.to_wire()['chains'] is None


async def test_an_unchanged_database_is_scanned_once_and_classified_at_each_now(tmp_path):
    root = tmp_path / 'proj'
    db_path = write_tasks_db(_db_path(root), [_row(link(1, 'pending', NOW + HOUR))])
    pool = DbPool()
    try:
        conn = await pool.get(db_path)
        assert conn is not None
        statements: list[str] = []
        await conn.set_trace_callback(statements.append)
        first = await scan_root_invariants(root, db_path, conn, now=NOW, pending_escalations=None)
        later = await scan_root_invariants(
            root, db_path, conn, now=NOW + 3 * HOUR, pending_escalations=None
        )
    finally:
        await pool.close_all()
    assert [chain.state for chain in first.chains or ()] == [ChainState.SCHEDULED]
    assert [chain.state for chain in later.chains or ()] == [ChainState.OVERDUE]
    assert sum('recurrence' in sql for sql in statements) == 1


async def test_a_commit_from_another_connection_reaches_the_next_scan(tmp_path):
    root = tmp_path / 'proj'
    db_path = write_tasks_db(_db_path(root), [_row(link(1, 'pending', key='old-job'))])
    pool = DbPool()
    try:
        conn = await pool.get(db_path)
        first = await scan_root_invariants(root, db_path, conn, now=NOW, pending_escalations=None)
        write_tasks_db(db_path, [_row(link(2, 'done', key='new-job'))])
        second = await scan_root_invariants(root, db_path, conn, now=NOW, pending_escalations=None)
    finally:
        await pool.close_all()
    assert _states(first.chains or ()) == [('master', 'old-job', ChainState.SCHEDULED, 1)]
    assert _states(second.chains or ()) == [
        ('master', 'new-job', ChainState.BROKEN, 2),
        ('master', 'old-job', ChainState.SCHEDULED, 1),
    ]
