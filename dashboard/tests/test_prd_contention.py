"""The contention line: which held lock keys the ready set wants, and who holds them (PRD §4.9).

Pure: built over a fixture scheduler snapshot and fixture rows naming
``metadata.files``, with no patching (boundary row 19, core half).
"""

from __future__ import annotations

import dataclasses
from collections.abc import Mapping
from datetime import UTC, datetime, timedelta
from types import MappingProxyType
from typing import Any

import pytest
from shared.scheduler_state import SchedulerStateSnapshot

from dashboard.data.census import build_census
from dashboard.data.datum import Datum, DatumState, unknown_datum
from dashboard.data.prd_contention import (
    CONTENTION_RULES,
    CONTENTION_TOP_N,
    Contention,
    ContentionContractError,
    ContentionFigure,
    ContentionInputs,
    ContentionInvariant,
    Holder,
    LockKey,
    build_contention,
    validate_contention,
)
from dashboard.data.prd_view import TaskActivity, ViewInput

NOW = datetime(2026, 11, 5, 12, 0, tzinfo=UTC)
ROOT = '/srv/contention'
AS_OF = NOW - timedelta(seconds=5)
BOUND = 60
SNAPSHOT_AT = NOW - timedelta(hours=3)
FIRST_EVENT_AT = NOW - timedelta(days=200)

PENDING_HOLDER = 50
BUSY_HOLDER = 51
CURRENT_HOLDERS = {'orch/core': str(PENDING_HOLDER), 'web/ui': str(BUSY_HOLDER)}

_GATE = {'task_kind': 'deterministic', 'always_escalates': True}


def _task(
    status: str,
    priority: str,
    files: list[Any] | None = None,
    dependencies: tuple[int, ...] = (),
    **metadata: Any,
) -> dict[str, Any]:
    if files is not None:
        metadata['files'] = files
    return {
        'status': status, 'priority': priority, 'dependencies': list(dependencies),
        'metadata': metadata,
    }


TASKS: Mapping[int, dict[str, Any]] = MappingProxyType({
    1: _task('pending', 'high', ['orch/core/a.py', 'orch/core/b.py']),
    2: _task('pending', 'critical', ['orch/core/c.py', 'docs/guide/x.md']),
    3: _task('pending', 'medium', ['web/ui/app.js']),
    4: _task('pending', 'low', ['docs/guide/y.md']),
    5: _task('pending', 'high', ['docs/guide/z.md']),
    6: _task('pending', 'medium'),
    7: _task('pending', 'high', ['orch/core/x.py'], dependencies=(BUSY_HOLDER,)),
    8: _task('in-progress', 'high', ['orch/core/y.py']),
    9: _task('pending', 'high', ['web/ui/gate.md'], **_GATE),
    PENDING_HOLDER: _task('pending', 'medium', ['orch/core/z.py'], dependencies=(BUSY_HOLDER,)),
    BUSY_HOLDER: _task('in-progress', 'medium', ['web/ui/b.js']),
})
"""Ready: 1, 2, 3, 4, 5, 6. Not ready: 7 (a dependency in progress), 8 (in progress),
9 (a human gate), the pending holder (a dependency in progress), the busy holder."""

EVER_STARTED = frozenset({1, 99})


def _fresh(value: Any, as_of: datetime = AS_OF) -> Datum[Any]:
    return Datum(value, as_of, DatumState.FRESH, None, BOUND)


def _snapshot(**overrides: Any) -> SchedulerStateSnapshot:
    fields: dict[str, Any] = {
        'current_holders': CURRENT_HOLDERS, 'lock_depth': 2, 'snapshot_at': SNAPSHOT_AT,
    }
    return SchedulerStateSnapshot(**{**fields, **overrides})


def _activity(ever_started: frozenset[int] = EVER_STARTED) -> TaskActivity:
    return TaskActivity(
        started=MappingProxyType({}),
        completed=MappingProxyType({}),
        ever_started=ever_started,
        first_event_at=FIRST_EVENT_AT,
    )


def _inputs(
    tasks: Mapping[int, dict[str, Any]] = TASKS,
    *,
    scheduler: Datum[SchedulerStateSnapshot] | None = None,
    activity: Datum[TaskActivity] | None = None,
) -> ContentionInputs:
    status_map = {task_id: task['status'] for task_id, task in tasks.items()}
    rows = [{'id': task_id, **task} for task_id, task in tasks.items()]
    return ContentionInputs(
        project_root=ROOT,
        status_map=status_map,
        census=_fresh(build_census(status_map)),
        rows=_fresh(rows),
        activity=activity or _fresh(_activity()),
        scheduler=scheduler or _fresh(_snapshot()),
    )


def _built(inputs: ContentionInputs) -> Contention:
    datum = build_contention(inputs, now=NOW)
    assert datum.value is not None, datum.reason
    return datum.value


@pytest.fixture(scope='module')
def contention() -> Contention:
    return _built(_inputs())


@pytest.fixture(scope='module')
def wire(contention) -> dict[str, Any]:
    return contention.to_wire()


# ---------------------------------------------------------------------------
# The top keys: the ready set's wanted locks and their holders
# ---------------------------------------------------------------------------


def test_top_counts_ready_tasks_once_per_key_most_wanted_first(wire):
    assert [(entry['key'], entry['wanted_by_ready']) for entry in wire['top']] == [
        ('docs/guide', 3), ('orch/core', 2), ('web/ui', 1),
    ]


def test_each_holder_carries_its_status_in_the_store_pending_served_as_pending(wire):
    holders = {entry['key']: entry['holder'] for entry in wire['top']}
    assert holders == {
        'docs/guide': None,
        'orch/core': {'task_id': str(PENDING_HOLDER), 'status': 'pending'},
        'web/ui': {'task_id': str(BUSY_HOLDER), 'status': 'in-progress'},
    }


def test_a_holder_absent_from_the_store_has_no_status():
    inputs = _inputs(scheduler=_fresh(_snapshot(current_holders={'docs/guide': '777'})))
    [docs] = [entry for entry in _built(inputs).top if entry.key == 'docs/guide']
    assert docs.holder == Holder(task_id='777', status=None)


def test_top_keeps_at_most_n_keys_ties_broken_by_key():
    tasks = {
        task_id: _task('pending', 'medium', [f'mod{task_id}/sub/file.py'])
        for task_id in range(1, CONTENTION_TOP_N + 3)
    }
    top = _built(_inputs(tasks)).top
    assert [entry.key for entry in top] == sorted(f'mod{n}/sub' for n in tasks)[:CONTENTION_TOP_N]
    assert all(entry.wanted_by_ready == 1 for entry in top)


def test_files_that_are_not_a_list_of_paths_want_no_key():
    tasks = {
        1: {'status': 'pending', 'priority': 'high', 'dependencies': [],
            'metadata': {'files': 'orch/core/a.py'}},
        2: _task('pending', 'low', [7, '', 'web/ui/app.js']),
    }
    assert [entry.key for entry in _built(_inputs(tasks)).top] == ['web/ui']


def test_snapshot_at_is_the_files_own_stamp_not_the_read_instant(contention, wire):
    assert contention.snapshot_at == SNAPSHOT_AT
    assert wire['snapshot_at'] == SNAPSHOT_AT.isoformat()
    assert wire['lock_depth'] == 2


# ---------------------------------------------------------------------------
# held_share and never_started
# ---------------------------------------------------------------------------


def test_held_share_counts_each_band_of_the_ready_set(wire):
    assert wire['held_share'] == {
        'critical': {'ready': 1, 'naming_held_key': 1},
        'high': {'ready': 2, 'naming_held_key': 1},
        'medium': {'ready': 2, 'naming_held_key': 1},
        'low': {'ready': 1, 'naming_held_key': 0},
    }


def test_never_started_lists_ready_high_tasks_with_no_task_started(wire):
    assert wire['never_started'] == {
        'count': 2, 'of': 3, 'ids': [2, 5], 'window_start': FIRST_EVENT_AT.isoformat(),
    }


def test_never_started_on_an_empty_ledger_has_no_window_start():
    empty = TaskActivity(
        started=MappingProxyType({}), completed=MappingProxyType({}),
        ever_started=frozenset(), first_event_at=None,
    )
    never_started = _built(_inputs(activity=_fresh(empty))).to_wire()['never_started']
    assert never_started == {'count': 3, 'of': 3, 'ids': [1, 2, 5], 'window_start': None}


# ---------------------------------------------------------------------------
# Degradation and provenance
# ---------------------------------------------------------------------------


def test_an_unknown_scheduler_state_makes_contention_unknown_with_its_reason():
    failed = unknown_datum('get_scheduler_state failed on every URL: boom', 30)
    datum = build_contention(_inputs(scheduler=failed), now=NOW)
    assert datum.state is DatumState.UNKNOWN
    assert datum.value is None
    assert 'get_scheduler_state failed on every URL: boom' in (datum.reason or '')


def test_a_snapshot_never_written_makes_contention_unknown():
    datum = build_contention(_inputs(scheduler=_fresh(_snapshot(snapshot_at=None))), now=NOW)
    assert datum.state is DatumState.UNKNOWN
    assert 'no scheduler snapshot written' in (datum.reason or '')


def test_a_snapshot_with_no_lock_depth_makes_contention_unknown_naming_it():
    datum = build_contention(_inputs(scheduler=_fresh(_snapshot(lock_depth=None))), now=NOW)
    assert datum.state is DatumState.UNKNOWN
    assert 'lock_depth' in (datum.reason or '')


def test_an_unknown_activity_makes_contention_unknown_naming_it():
    failed = unknown_datum('runs.db unreadable', BOUND)
    datum = build_contention(_inputs(activity=failed), now=NOW)
    assert datum.state is DatumState.UNKNOWN
    assert 'activity' in (datum.reason or '')
    assert 'runs.db unreadable' in (datum.reason or '')


def test_inputs_are_listed_each_with_its_own_provenance():
    stale_as_of = NOW - timedelta(seconds=BOUND * 2)
    datum = build_contention(_inputs(scheduler=_fresh(_snapshot(), as_of=stale_as_of)), now=NOW)
    assert datum.value is not None, datum.reason
    assert datum.state is DatumState.STALE
    assert ViewInput.SCHEDULER_STATE.value in (datum.reason or '')
    listed = {state.name: state for state in datum.value.inputs}
    assert list(listed) == [
        ViewInput.STATUS_MAP, ViewInput.ACTIVE_ROWS, ViewInput.ACTIVITY, ViewInput.SCHEDULER_STATE,
    ]
    assert listed[ViewInput.SCHEDULER_STATE].state is DatumState.STALE
    assert listed[ViewInput.SCHEDULER_STATE].as_of == stale_as_of
    assert {listed[name].state for name in list(listed)[:3]} == {DatumState.FRESH}
    assert datum.as_of == stale_as_of


def test_a_fresh_build_is_fresh_as_of_its_oldest_input():
    datum = build_contention(_inputs(), now=NOW)
    assert datum.state is DatumState.FRESH
    assert datum.as_of == AS_OF
    assert datum.reason is None


# ---------------------------------------------------------------------------
# Rules and validation
# ---------------------------------------------------------------------------


def test_every_figure_has_rule_text_and_every_wire_figure_names_one(wire):
    assert set(CONTENTION_RULES) == set(ContentionFigure)
    assert all(text.strip() for text in CONTENTION_RULES.values())
    assert wire['rules'] == {figure.value: text for figure, text in CONTENTION_RULES.items()}
    figures = set(wire) - {'inputs', 'rules'}
    assert figures == {figure.value for figure in ContentionFigure}


def _breach(contention: Contention, **changes: Any) -> ContentionInvariant:
    with pytest.raises(ContentionContractError) as caught:
        validate_contention(dataclasses.replace(contention, **changes))
    return caught.value.rule


def test_validate_refuses_a_top_longer_than_n(contention):
    top = tuple(LockKey(f'k{n:02d}', 1, None) for n in range(CONTENTION_TOP_N + 1))
    assert _breach(contention, top=top) is ContentionInvariant.TOP_LENGTH


def test_validate_refuses_an_unsorted_top(contention):
    top = (LockKey('a', 1, None), LockKey('b', 2, None))
    assert _breach(contention, top=top) is ContentionInvariant.TOP_ORDER
    tied = (LockKey('b', 1, None), LockKey('a', 1, None))
    assert _breach(contention, top=tied) is ContentionInvariant.TOP_ORDER


def test_validate_refuses_a_never_started_count_that_is_not_its_ids(contention):
    never_started = dataclasses.replace(contention.never_started, count=5)
    assert _breach(contention, never_started=never_started) is (
        ContentionInvariant.NEVER_STARTED_COUNT
    )


def test_validate_accepts_the_built_contention(contention):
    validate_contention(contention)
