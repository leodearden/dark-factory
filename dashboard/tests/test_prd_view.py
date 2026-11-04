"""Tests for ``dashboard.data.prd_view``: the pure per-PRD view over one root's
census, active rows and lineage (PRD dashboard-prd-lanes §4.4–§4.5, §5.5)."""

from __future__ import annotations

from datetime import UTC, datetime, timedelta
from typing import Any

import pytest

from dashboard.data.prd_view import (
    SEGMENT_TONES,
    Segment,
    is_human_gate,
    is_ready,
    segment_of,
)

NOW = datetime(2026, 11, 4, 12, 0, tzinfo=UTC)
DONE_DEP = 1
CANCELLED_DEP = 2
PENDING_DEP = 3
ABSENT_DEP = 999


def _row(task_id: int, *, deps: tuple[int, ...] = (), **metadata: Any) -> dict:
    """One raw active row, as ``task_snapshot`` holds it."""
    return {
        'id': task_id,
        'status': 'pending',
        'priority': 'medium',
        'dependencies': list(deps),
        'metadata': metadata,
    }


def _status_map(task_id: int) -> dict[int, str]:
    return {DONE_DEP: 'done', CANCELLED_DEP: 'cancelled', PENDING_DEP: 'pending', task_id: 'pending'}


# ---------------------------------------------------------------------------
# segments: one per status, pending split by readiness
# ---------------------------------------------------------------------------


@pytest.mark.parametrize(
    ('status', 'expected'),
    [
        ('done', Segment.DONE),
        ('in-progress', Segment.IN_PROGRESS),
        ('review', Segment.IN_PROGRESS),
        ('merge-deferred', Segment.IN_PROGRESS),
        ('infra-hold', Segment.IN_PROGRESS),
        ('blocked', Segment.BLOCKED),
        ('deferred', Segment.DEFERRED),
        ('cancelled', None),
    ],
)
def test_segment_of_a_non_pending_status(status, expected):
    status_map = {10: status}
    row = {**_row(10), 'status': status}
    assert segment_of(10, status, row, status_map, now=NOW) == expected


_WAITING: dict[str, dict | None] = {
    'pending_dep': _row(10, deps=(DONE_DEP, PENDING_DEP)),
    'dep_absent_from_status_map': _row(10, deps=(ABSENT_DEP,)),
    'external_deps': _row(10, external_deps=['reify:1']),
    'unfired_dated_milestone': _row(
        10, milestone={'mode': 'dated', 'at': (NOW + timedelta(days=1)).isoformat()},
    ),
    'delayed_milestone_without_anchor': _row(10, milestone={'mode': 'delayed', 'after_secs': 3600}),
    'malformed_milestone': _row(10, milestone={'mode': 'dated'}),
    'non_string_anchor': _row(
        10,
        milestone={'mode': 'delayed', 'after_secs': 3600},
        milestone_deps_satisfied_at=12345,
    ),
    'human_gate': _row(10, task_kind='deterministic', always_escalates=True),
    'no_active_row': None,
}

_READY: dict[str, dict] = {
    'deps_done_or_cancelled': _row(10, deps=(DONE_DEP, CANCELLED_DEP)),
    'fired_dated_milestone': _row(
        10, milestone={'mode': 'dated', 'at': (NOW - timedelta(days=1)).isoformat()},
    ),
    'fired_dated_milestone_naive_at': _row(
        10, milestone={'mode': 'dated', 'at': '2026-11-03T12:00:00'},
    ),
    'fired_delayed_milestone': _row(
        10,
        milestone={'mode': 'delayed', 'after_secs': 3600},
        milestone_deps_satisfied_at=(NOW - timedelta(hours=2)).isoformat(),
    ),
    'clean': _row(10),
    'empty_external_deps': _row(10, external_deps=[]),
}


@pytest.mark.parametrize('row', list(_WAITING.values()), ids=list(_WAITING))
def test_a_pending_task_that_cannot_dispatch_waits(row):
    assert is_ready(10, row, _status_map(10), now=NOW) is False
    assert segment_of(10, 'pending', row, _status_map(10), now=NOW) is Segment.WAITING


@pytest.mark.parametrize('row', list(_READY.values()), ids=list(_READY))
def test_a_pending_task_that_could_dispatch_is_ready(row):
    assert is_ready(10, row, _status_map(10), now=NOW) is True
    assert segment_of(10, 'pending', row, _status_map(10), now=NOW) is Segment.READY


def test_a_task_not_pending_in_the_status_map_is_never_ready():
    status_map = {**_status_map(10), 10: 'in-progress'}
    assert is_ready(10, _row(10), status_map, now=NOW) is False


@pytest.mark.parametrize(
    ('metadata', 'expected'),
    [
        ({'task_kind': 'deterministic', 'always_escalates': True}, True),
        ({'task_kind': 'deterministic', 'always_escalates': 1}, False),
        ({'task_kind': 'deterministic', 'always_escalates': 'yes'}, False),
        ({'task_kind': 'deterministic'}, False),
        ({'task_kind': 'llm', 'always_escalates': True}, False),
        ({}, False),
    ],
    ids=['gate', 'truthy_int', 'truthy_str', 'no_flag', 'not_deterministic', 'empty'],
)
def test_is_human_gate(metadata, expected):
    assert is_human_gate(metadata) is expected


def test_segments_are_declared_in_display_order():
    assert [segment.value for segment in Segment] == [
        'done', 'in_progress', 'blocked', 'deferred', 'ready', 'waiting',
    ]


def test_segment_tones_are_the_palette_keys():
    assert dict(SEGMENT_TONES) == {
        Segment.DONE: 'ok',
        Segment.IN_PROGRESS: 'accent',
        Segment.BLOCKED: 'bad',
        Segment.DEFERRED: 'fg3',
        Segment.READY: 'warn',
        Segment.WAITING: 'info',
    }
