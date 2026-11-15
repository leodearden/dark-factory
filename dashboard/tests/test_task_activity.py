"""A project's task activity, read from its runs.db: who started lately, who finished, since when."""

from __future__ import annotations

import sqlite3
from collections.abc import AsyncIterator
from contextlib import asynccontextmanager, closing
from datetime import UTC, datetime, timedelta
from pathlib import Path

import aiosqlite
import pytest
from _runs_db_fixture import runs_db_path, write_runs_db

import dashboard.data.task_activity as task_activity
from dashboard.data.datum import DatumState
from dashboard.data.prd_view import MOVING_WINDOW_DAYS
from dashboard.data.task_activity import ACTIVITY_QUERIES, acquire_activity, clear_activity_state

NOW = datetime(2026, 11, 5, 9, 0, tzinfo=UTC)


def ago(days: float) -> datetime:
    return NOW - timedelta(days=days)


EVENTS = [
    (ago(50), '1', 'phase_enter', {}),
    (ago(13), '1', 'task_started', {}),
    (ago(15), '2', 'task_started', {}),
    (ago(10), '3', 'task_started', {}),
    (ago(2), '3', 'task_started', {}),
    (ago(1), 'abc', 'task_started', {}),
    (ago(3), '4', 'task_completed', {'outcome': 'done'}),
    (ago(2), '5', 'task_completed', {'outcome': 'done'}),
    (ago(2), '6', 'task_completed', {'outcome': 'blocked'}),
    (ago(2), '7', 'task_completed', 'not json'),
    (ago(40), '9', 'task_completed', {'outcome': 'done'}),
    (ago(1), 'abc', 'task_completed', {'outcome': 'done'}),
]
TASK_RESULTS = [
    ('4', 'done', ago(1)),
    ('5', 'done', ago(5)),
    ('8', 'blocked', ago(1)),
    ('10', 'done', None),
]


@pytest.fixture(autouse=True)
def _forget_activity():
    clear_activity_state()
    yield
    clear_activity_state()


@asynccontextmanager
async def opened(store: Path) -> AsyncIterator[aiosqlite.Connection]:
    """The store as the dashboard's DbPool opens it: read-only, rows by name."""
    async with aiosqlite.connect(f'{store.as_uri()}?mode=ro', uri=True) as conn:
        conn.row_factory = aiosqlite.Row
        yield conn


async def activity_of(root: Path, **written):
    async with opened(write_runs_db(root, **written)) as conn:
        return await acquire_activity(conn, root, now=NOW)


async def test_started_holds_each_tasks_latest_start_inside_the_moving_window(tmp_path):
    datum = await activity_of(tmp_path, events=EVENTS, task_results=TASK_RESULTS)

    assert MOVING_WINDOW_DAYS == 14
    assert datum.state is DatumState.FRESH
    assert datum.as_of == NOW
    assert datum.value is not None
    assert dict(datum.value.started) == {1: ago(13), 3: ago(2)}


async def test_completed_holds_each_tasks_latest_done_completion_from_events_or_results(tmp_path):
    datum = await activity_of(tmp_path, events=EVENTS, task_results=TASK_RESULTS)

    assert datum.value is not None
    assert dict(datum.value.completed) == {4: ago(1), 5: ago(2), 9: ago(40)}


async def test_every_started_task_and_the_first_event_are_known(tmp_path):
    datum = await activity_of(tmp_path, events=EVENTS, task_results=TASK_RESULTS)

    assert datum.value is not None
    assert datum.value.ever_started == frozenset({1, 2, 3})
    assert datum.value.first_event_at == ago(50)


async def test_an_empty_runs_db_has_no_first_event(tmp_path):
    datum = await activity_of(tmp_path)

    assert datum.state is DatumState.FRESH
    assert datum.value is not None
    assert datum.value.first_event_at is None
    assert datum.value.ever_started == frozenset()


async def test_no_runs_db_is_unknown_naming_where_it_was_looked_for(tmp_path):
    datum = await acquire_activity(None, tmp_path, now=NOW)

    assert datum.state is DatumState.UNKNOWN
    assert datum.value is None
    assert str(runs_db_path(tmp_path)) in (datum.reason or '')


WALK_TIMED_OUT = 'the request walk of run ledgers did not finish within 2.0s'


async def test_no_connection_for_a_known_reason_is_unknown_with_that_reason(tmp_path):
    datum = await acquire_activity(None, tmp_path, now=NOW, absent_reason=WALK_TIMED_OUT)

    assert datum.state is DatumState.UNKNOWN
    assert datum.reason == WALK_TIMED_OUT


async def test_no_connection_after_a_good_read_serves_the_last_good_activity(
    tmp_path, monkeypatch,
):
    good = await activity_of(tmp_path, events=EVENTS, task_results=TASK_RESULTS)
    monkeypatch.setattr(task_activity, 'ACTIVITY_TTL_SECONDS', 0.0)

    datum = await acquire_activity(
        None, tmp_path, now=NOW + timedelta(seconds=1), absent_reason=WALK_TIMED_OUT,
    )

    assert datum.state is DatumState.STALE
    assert datum.reason == WALK_TIMED_OUT
    assert datum.value == good.value
    assert datum.as_of == NOW


async def test_a_runs_db_without_task_results_is_a_lower_bound_naming_the_table(tmp_path):
    datum = await activity_of(tmp_path, events=EVENTS, with_task_results_table=False)

    assert datum.state is DatumState.LOWER_BOUND
    assert 'task_results' in (datum.reason or '')
    assert datum.value is not None
    assert dict(datum.value.completed) == {4: ago(3), 5: ago(2), 9: ago(40)}


def test_every_events_query_is_bounded_by_an_index(tmp_path):
    store = write_runs_db(tmp_path, events=EVENTS, task_results=TASK_RESULTS)
    events_queries = {name: sql for name, sql in ACTIVITY_QUERIES.items() if 'events' in sql}

    with closing(sqlite3.connect(store)) as conn:
        plans = {
            name: [row[-1] for row in conn.execute(f'EXPLAIN QUERY PLAN {sql}', (None,) * sql.count('?'))]
            for name, sql in events_queries.items()
        }

    assert events_queries
    for name, details in plans.items():
        touching = [detail for detail in details if 'events' in detail]
        assert touching, (name, details)
        assert all('idx_events_type' in d or 'idx_events_ts' in d for d in touching), (name, details)
        assert 'SCAN events' not in details, (name, details)
