"""The recurrence chain-state gauge (docs/prds/recurring-deterministic-tasks.md R-D5/C-4).

One state per chain, read from the chain's current link: blocked is paused, done
is broken, cancelled is ended, and every other status is judged by its fire time.
"""

from __future__ import annotations

from datetime import UTC, datetime, timedelta
from typing import Any

import pytest
from shared.task_statuses import TaskStatus

from dashboard.data.recurrence_gauge import ChainState, chain_state

NOW = datetime(2026, 10, 12, 12, 0, tzinfo=UTC)
HOUR = timedelta(hours=1)


def link(
    link_id: int,
    status: str,
    at: datetime | str = NOW + HOUR,
    interval_secs: int | None = 3600,
    key: str = 'nightly-job',
) -> dict[str, Any]:
    """One decoded chain link whose dated milestone fires at *at*."""
    recurrence: dict[str, Any] = {'key': key}
    if interval_secs is not None:
        recurrence['interval_secs'] = interval_secs
    at_text = at if isinstance(at, str) else at.isoformat()
    return {
        'tag': 'master',
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
