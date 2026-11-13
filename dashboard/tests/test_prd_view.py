"""Tests for ``dashboard.data.prd_view``: the pure per-PRD view over one root's
census, active rows and lineage (PRD dashboard-prd-lanes §4.4–§4.5, §5.5)."""

from __future__ import annotations

import dataclasses
from collections.abc import Callable, Mapping
from datetime import UTC, datetime, timedelta
from types import MappingProxyType
from typing import Any

import pytest
from _prd_tree import PRD_A, PRD_B, prd_tree_tasks
from shared.task_statuses import TaskStatus

from dashboard.data import prd_view
from dashboard.data.census import CensusVocabularyError, TaskCensus, build_census
from dashboard.data.datum import (
    Datum,
    DatumContractError,
    DatumInvariant,
    DatumState,
    unknown_datum,
    validate_datum,
)
from dashboard.data.prd_attribution import LineageEntry, UnresolvedReason, lineage_entry
from dashboard.data.prd_view import (
    FIGURE_RULES,
    SEGMENT_LABELS,
    SEGMENT_TONES,
    FigureId,
    PrdView,
    PrdViewContractError,
    PrdViewInputs,
    PrdViewInvariant,
    Segment,
    TaskActivity,
    build_prd_view,
    is_human_gate,
    is_ready,
    open_work_keys,
    segment_of,
    validate_prd_view,
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


# ---------------------------------------------------------------------------
# build_prd_view: the census join over one root (boundary rows 8, 9 and 12)
# ---------------------------------------------------------------------------

ROOT = '/home/leo/src/dark-factory'
SNAPSHOT_BOUND = 30
LINEAGE_BOUND = 900
SNAPSHOT_AS_OF = NOW - timedelta(seconds=5)
LINEAGE_AS_OF = NOW - timedelta(seconds=120)
ESCALATIONS_AS_OF = NOW - timedelta(seconds=10)
ESCALATIONS_BOUND = 60
ACTIVITY_AS_OF = NOW - timedelta(seconds=20)
ACTIVITY_BOUND = 180
DOCUMENTS_AS_OF = NOW - timedelta(seconds=30)
DOCUMENTS_BOUND = 90
NO_ACTIVITY = TaskActivity(
    started=MappingProxyType({}), completed=MappingProxyType({}),
    ever_started=frozenset(), first_event_at=None,
)


def _pending_since(days: float) -> str:
    return (NOW - timedelta(days=days)).isoformat()


_Tasks = Mapping[int, tuple[str, str | None, Mapping[str, Any]]]

_TASKS: _Tasks = prd_tree_tasks(NOW)
STATUS_MAP: dict[int, str] = {task_id: status for task_id, (status, _, _) in _TASKS.items()}


def _active_row(task_id: int, status: str, priority: str | None, metadata: Mapping) -> dict:
    row: dict[str, Any] = {
        'id': task_id, 'status': status, 'dependencies': [], 'metadata': metadata,
    }
    if priority is not None:
        row['priority'] = priority
    return row


def _rows(tasks: _Tasks) -> list[dict]:
    """The open tasks' raw rows, as the snapshot unit holds its active rows."""
    return [
        _active_row(task_id, status, priority, metadata)
        for task_id, (status, priority, metadata) in tasks.items()
        if TaskStatus(status) not in {TaskStatus.DONE, TaskStatus.CANCELLED}
    ]


def _lineage(tasks: _Tasks) -> dict[int, LineageEntry]:
    return {task_id: lineage_entry(metadata, ROOT) for task_id, (_, _, metadata) in tasks.items()}


def _fresh(value: Any, as_of: datetime, bound: int) -> Datum[Any]:
    return Datum(value, as_of, DatumState.FRESH, None, bound)


def _inputs(
    tasks: _Tasks = _TASKS,
    *,
    status_map: dict[int, str] | None = None,
    census: Datum[TaskCensus] | None = None,
    rows: Datum[list[dict]] | None = None,
    lineage: Datum[Any] | None = None,
    escalations: Datum[Any] | None = None,
    activity: Datum[TaskActivity] | None = None,
    documents: Datum[Any] | None = None,
    liveness: Datum[Any] | None = None,
) -> PrdViewInputs:
    status_map = (
        {task_id: status for task_id, (status, _, _) in tasks.items()}
        if status_map is None else status_map
    )
    return PrdViewInputs(
        project_root=ROOT,
        status_map=status_map,
        census=census or _fresh(build_census(status_map), SNAPSHOT_AS_OF, SNAPSHOT_BOUND),
        rows=rows or _fresh(_rows(tasks), SNAPSHOT_AS_OF, SNAPSHOT_BOUND),
        lineage=lineage or _fresh(MappingProxyType(_lineage(tasks)), LINEAGE_AS_OF, LINEAGE_BOUND),
        escalations=escalations or _fresh(
            MappingProxyType({}), ESCALATIONS_AS_OF, ESCALATIONS_BOUND,
        ),
        activity=activity or _fresh(NO_ACTIVITY, ACTIVITY_AS_OF, ACTIVITY_BOUND),
        documents=documents or _fresh(MappingProxyType({}), DOCUMENTS_AS_OF, DOCUMENTS_BOUND),
        liveness=liveness or _fresh(MappingProxyType({}), SNAPSHOT_AS_OF, SNAPSHOT_BOUND),
    )


def _view(inputs: PrdViewInputs | None = None) -> PrdView:
    view = build_prd_view(inputs or _inputs(), now=NOW).value
    assert view is not None
    return view


def _wire(inputs: PrdViewInputs | None = None) -> dict[str, Any]:
    return _view(inputs).to_wire()


def test_a_prd_row_counts_its_members_by_segment():
    [row] = _wire()['prds']
    assert row.pop('title') == PRD_A
    assert row.pop('signal')['mark'] == 'unknown'
    assert row.pop('lane') == 'moving'
    assert row.pop('activity') == {'direct': 0, 'followup': 0}
    assert row.pop('state_line') == (
        'moving: 0 direct, 0 follow-ups started or finished in 14 d; 1 in progress'
    )
    assert row.pop('next_action') == {'kind': 'none', 'target': None, 'text': ''}
    assert row.pop('detail') == {
        'blocker_refs': [], 'ruling_refs': [], 'oldest_ready_high_ids': [],
        'deferred_unrecorded_ids': [], 'needs_human_refs': [], 'high_open_ids': [],
    }
    assert row == {
        'key': PRD_A,
        'segments': {
            'done': 2, 'in_progress': 1, 'blocked': 0, 'deferred': 0, 'ready': 1, 'waiting': 0,
        },
        'total': 4,
        'direct_open_ids': [2],
        'followup_open': [{'task_id': 3, 'hops': 1}],
        'ready': {
            'count': 1, 'oldest_task_id': 2, 'oldest_pending_days': 40,
            'oldest_is_lower_bound': False,
        },
    }


def test_a_prd_with_no_open_member_is_closed_not_a_row():
    wire = _wire()
    assert PRD_B not in [row['key'] for row in wire['prds']]
    assert wire['closed'] == {'prds': 1, 'tasks': 1}


def test_done_tasks_no_prd_owns_close_the_sum_undrawn():
    assert _wire()['unattributed_done'] == 1


def test_multi_link_counts_tasks_carrying_two_link_keys():
    assert _wire()['multi_link'] == 1


def test_dark_matter_groups_open_unattributed_tasks():
    dark_matter = _wire()['dark_matter']
    assert dark_matter['total'] == 3
    assert dark_matter['members'] == {
        'no_link': [7], 'cross_project_link': [9], 'foreign_prd': [10],
    }
    assert dark_matter['by_reason'] == {
        reason.value: (1 if reason.value in {'no_link', 'cross_project_link', 'foreign_prd'} else 0)
        for reason in UnresolvedReason
    }
    assert dark_matter['by_status'] == {
        status.value: {'pending': 2, 'blocked': 1}.get(status.value, 0) for status in TaskStatus
    }
    assert dark_matter['by_priority'] == {'medium': 1, 'high': 1, 'unrecorded': 1}
    assert dark_matter['by_source'] == {'review': 1, 'unrecorded': 2}
    assert dark_matter['ready_high_ids'] == [9]


def test_dark_matter_sources_are_the_top_five_plus_other():
    sources = ['a', 'a', 'a', 'b', 'b', 'c', 'd', 'e', 'f', 'g']
    tasks = {
        task_id: ('pending', 'medium', {'source': source})
        for task_id, source in enumerate(sources, start=1)
    }
    assert _wire(_inputs(tasks))['dark_matter']['by_source'] == {
        'a': 3, 'b': 2, 'c': 1, 'd': 1, 'e': 1, 'other': 2,
    }


def test_the_census_identity_holds_against_the_status_map():
    wire = _wire()
    census = build_census(STATUS_MAP)
    drawn = sum(row['total'] for row in wire['prds'])
    assert (
        drawn + wire['closed']['tasks'] + wire['unattributed_done'] + wire['dark_matter']['total']
        == census.total - census.counts[TaskStatus.CANCELLED]
    )


def _with_metadata(task_id: int, **metadata: Any) -> _Tasks:
    status, priority, base = _TASKS[task_id]
    return {**_TASKS, task_id: (status, priority, {**base, **metadata})}


def test_a_backfilled_anchor_makes_the_ready_age_a_lower_bound():
    tasks = _with_metadata(2, pending_since_backfilled=True)
    [row] = _wire(_inputs(tasks))['prds']
    assert row['ready'] == {
        'count': 1, 'oldest_task_id': 2, 'oldest_pending_days': 40,
        'oldest_is_lower_bound': True,
    }


def test_a_younger_backfilled_anchor_still_makes_the_ready_age_a_lower_bound():
    tasks = {
        **_TASKS,
        11: ('pending', 'medium', {
            'prd_path': PRD_A, 'pending_since': _pending_since(3), 'pending_since_backfilled': True,
        }),
    }
    [row] = _wire(_inputs(tasks))['prds']
    assert row['ready'] == {
        'count': 2, 'oldest_task_id': 2, 'oldest_pending_days': 40,
        'oldest_is_lower_bound': True,
    }


@pytest.mark.parametrize(
    'anchor', [{}, {'pending_since': 'not a timestamp'}, {'pending_since': 12345}],
    ids=['absent', 'unparseable', 'not_a_string'],
)
def test_a_ready_member_without_a_parseable_anchor_makes_the_age_a_lower_bound(anchor):
    tasks = {**_TASKS, 11: ('pending', 'medium', {'prd_path': PRD_A, **anchor})}
    [row] = _wire(_inputs(tasks))['prds']
    assert row['ready'] == {
        'count': 2, 'oldest_task_id': 2, 'oldest_pending_days': 40,
        'oldest_is_lower_bound': True,
    }


def test_no_ready_member_has_no_ready_age():
    tasks = _with_metadata(2, external_deps=['reify:1'])
    [row] = _wire(_inputs(tasks))['prds']
    assert row['ready'] == {
        'count': 0, 'oldest_task_id': None, 'oldest_pending_days': None,
        'oldest_is_lower_bound': False,
    }


# --- row 8: the active rows overlay a lagging lineage index ---------------


def test_an_active_rows_own_prd_overrides_its_lineage_entry():
    lineage = {**_lineage(_TASKS), 2: lineage_entry({}, ROOT)}
    inputs = _inputs(lineage=_fresh(MappingProxyType(lineage), LINEAGE_AS_OF, LINEAGE_BOUND))
    wire = _wire(inputs)
    assert wire['prds'][0]['direct_open_ids'] == [2]
    assert 2 not in [task_id for ids in wire['dark_matter']['members'].values() for task_id in ids]


def test_a_status_map_id_unknown_to_lineage_and_rows_is_lineage_pending():
    status_map = {**STATUS_MAP, 11: 'pending', 12: 'done'}
    wire = _wire(_inputs(status_map=status_map))
    assert wire['dark_matter']['members']['lineage_pending'] == [11]
    assert wire['unattributed_done'] == 2
    census = build_census(status_map)
    drawn = sum(row['total'] for row in wire['prds'])
    assert (
        drawn + wire['closed']['tasks'] + wire['unattributed_done'] + wire['dark_matter']['total']
        == census.total - census.counts[TaskStatus.CANCELLED]
    )


# --- the attribution is reused for the same rows and lineage values ------


def test_one_rows_and_lineage_value_is_attributed_once_while_now_moves(monkeypatch):
    walks: list[int] = []
    walk = prd_view.attribute

    def _counted(entries):
        walks.append(len(entries))
        return walk(entries)

    monkeypatch.setattr(prd_view, 'attribute', _counted)
    inputs = _inputs()
    first = _view(inputs)
    later = build_prd_view(inputs, now=NOW + timedelta(days=2)).value
    assert later is not None
    assert len(walks) == 1
    assert first.prds[0].ready.oldest_pending_days == 40
    assert later.prds[0].ready.oldest_pending_days == 42


def test_a_new_rows_value_beside_the_same_lineage_is_attributed_afresh():
    lineage = _fresh(MappingProxyType(_lineage(_TASKS)), LINEAGE_AS_OF, LINEAGE_BOUND)
    before = _wire(_inputs(lineage=lineage))
    after = _wire(_inputs(_with_metadata(7, prd_path=PRD_B), lineage=lineage))
    assert before['dark_matter']['members']['no_link'] == [7]
    assert 'no_link' not in after['dark_matter']['members']
    assert [row['direct_open_ids'] for row in after['prds'] if row['key'] == PRD_B] == [[7]]


# --- composite provenance -------------------------------------------------


def test_a_view_over_fresh_inputs_is_fresh_as_of_its_oldest_input():
    datum = build_prd_view(_inputs(), now=NOW)
    assert datum.state is DatumState.FRESH
    assert datum.reason is None
    assert datum.as_of == LINEAGE_AS_OF
    assert datum.freshness_bound_seconds == LINEAGE_BOUND
    validate_datum(datum, NOW)


def test_the_view_lists_each_input_with_its_own_provenance():
    def fresh_at(name: str, as_of: datetime) -> dict[str, Any]:
        return {
            'name': name, 'as_of': as_of.astimezone(UTC).isoformat(),
            'state': 'fresh', 'reason': None,
        }

    assert _wire()['inputs'] == [
        fresh_at('status_map', SNAPSHOT_AS_OF),
        fresh_at('active_rows', SNAPSHOT_AS_OF),
        fresh_at('lineage', LINEAGE_AS_OF),
        fresh_at('escalations', ESCALATIONS_AS_OF),
        fresh_at('activity', ACTIVITY_AS_OF),
        fresh_at('signals', DOCUMENTS_AS_OF),
        fresh_at('liveness', SNAPSHOT_AS_OF),
    ]


def test_stale_rows_make_a_stale_view_naming_the_rows_reason():
    rows = Datum(_rows(_TASKS), NOW - timedelta(seconds=60), DatumState.STALE, 'rows read failed: boom', 30)
    datum = build_prd_view(_inputs(rows=rows), now=NOW)
    assert datum.state is DatumState.STALE
    assert datum.reason == 'active_rows: rows read failed: boom'
    assert datum.value is not None
    validate_datum(datum, NOW)


def test_an_input_past_its_own_bound_at_now_is_judged_stale():
    rows = _fresh(_rows(_TASKS), NOW - timedelta(seconds=31), SNAPSHOT_BOUND)
    datum = build_prd_view(_inputs(rows=rows), now=NOW)
    assert datum.state is DatumState.STALE
    assert datum.reason == 'active_rows: measured 31s before it was served, past the 30s freshness bound'
    assert datum.value is not None
    assert [state.state for state in datum.value.inputs] == [
        DatumState.FRESH, DatumState.STALE, *[DatumState.FRESH] * 5,
    ]


def test_every_non_fresh_input_is_named_in_the_reason():
    rows = Datum(_rows(_TASKS), SNAPSHOT_AS_OF, DatumState.STALE, 'rows lag', SNAPSHOT_BOUND)
    lineage = Datum(
        MappingProxyType(_lineage(_TASKS)), LINEAGE_AS_OF, DatumState.STALE, 'lineage lag', LINEAGE_BOUND,
    )
    datum = build_prd_view(_inputs(rows=rows, lineage=lineage), now=NOW)
    assert datum.state is DatumState.STALE
    assert datum.reason == 'active_rows: rows lag; lineage: lineage lag'


def test_stale_activity_makes_a_stale_view_naming_the_activity_reason():
    activity = Datum(
        NO_ACTIVITY, ACTIVITY_AS_OF, DatumState.STALE, 'read still in flight after 4.0s',
        ACTIVITY_BOUND,
    )
    datum = build_prd_view(_inputs(activity=activity), now=NOW)
    assert datum.state is DatumState.STALE
    assert datum.reason == 'activity: read still in flight after 4.0s'
    assert datum.value is not None


def test_unknown_documents_make_an_unknown_view_naming_the_signals_input():
    documents = unknown_datum('project root /gone is not a readable directory', DOCUMENTS_BOUND)
    datum = build_prd_view(_inputs(documents=documents), now=NOW)
    assert datum.state is DatumState.UNKNOWN
    assert datum.value is None
    assert datum.reason == 'signals: project root /gone is not a readable directory'
    validate_datum(datum, NOW)


@pytest.mark.parametrize('name', ['escalations', 'activity', 'liveness'])
def test_an_unknown_escalations_activity_or_liveness_input_makes_an_unknown_view(name):
    inputs = dataclasses.replace(_inputs(), **{name: unknown_datum('boom', 60)})
    datum = build_prd_view(inputs, now=NOW)
    assert datum.state is DatumState.UNKNOWN
    assert datum.reason == f'{name}: boom'


def test_open_work_keys_are_the_keys_of_the_rows_the_view_draws():
    inputs = _inputs()
    assert inputs.rows.value is not None and inputs.lineage.value is not None
    keys = open_work_keys(ROOT, inputs.status_map, inputs.rows.value, inputs.lineage.value)
    assert keys == tuple(row.key for row in _view(inputs).prds) == (PRD_A,)


def test_unknown_lineage_makes_an_unknown_view_naming_the_lineage_reason():
    lineage = unknown_datum('first lineage read still in flight after 5.0s', LINEAGE_BOUND)
    datum = build_prd_view(_inputs(lineage=lineage), now=NOW)
    assert datum.state is DatumState.UNKNOWN
    assert datum.value is None
    assert datum.reason == 'lineage: first lineage read still in flight after 5.0s'
    assert datum.freshness_bound_seconds == LINEAGE_BOUND
    validate_datum(datum, NOW)


def test_an_unknown_census_makes_an_unknown_view_with_the_census_reason():
    census = unknown_datum('status map read failed: connection refused', SNAPSHOT_BOUND)
    datum = build_prd_view(_inputs(census=census), now=NOW)
    assert datum.state is DatumState.UNKNOWN
    assert datum.reason == 'status map read failed: connection refused'


def test_a_failed_map_half_beside_a_last_good_census_is_unknown_with_the_maps_reason():
    census = Datum(
        build_census(STATUS_MAP), NOW - timedelta(seconds=40), DatumState.STALE,
        'status map read failed: timeout', SNAPSHOT_BOUND,
    )
    datum = build_prd_view(_inputs(status_map={}, census=census), now=NOW)
    assert datum.state is DatumState.UNKNOWN
    assert datum.reason == 'status map read failed: timeout'


def test_an_empty_map_beside_a_fresh_census_is_unknown():
    census = _fresh(build_census({}), SNAPSHOT_AS_OF, SNAPSHOT_BOUND)
    datum = build_prd_view(_inputs(status_map={}, census=census), now=NOW)
    assert datum.state is DatumState.UNKNOWN
    assert datum.reason == 'the status map measured no tasks'


def test_an_off_vocabulary_status_makes_an_unknown_view_with_the_census_error():
    status_map = {**STATUS_MAP, 99: 'bogus'}
    with pytest.raises(CensusVocabularyError) as drift:
        build_census(status_map)
    census = _fresh(build_census(STATUS_MAP), SNAPSHOT_AS_OF, SNAPSHOT_BOUND)
    datum = build_prd_view(_inputs(status_map=status_map, census=census), now=NOW)
    assert datum.state is DatumState.UNKNOWN
    assert datum.reason == str(drift.value)


# --- row 12: every figure carries its rule --------------------------------

_STRUCTURAL_VIEW_KEYS = {'prds', 'inputs', 'rules', 'segment_legend', 'lane_precedence'}
_STRUCTURAL_ROW_KEYS = {'key'}


def test_every_figure_id_has_rule_text_shipped_in_the_view():
    assert set(FIGURE_RULES) == set(FigureId)
    assert all(isinstance(text, str) and text.strip() for text in FIGURE_RULES.values())
    assert _wire()['rules'] == {figure.value: text for figure, text in FIGURE_RULES.items()}


def test_every_figure_on_the_wire_names_a_figure_id():
    wire = _wire()
    figure_ids = {figure.value for figure in FigureId}
    view_figures = set(wire) - _STRUCTURAL_VIEW_KEYS
    row_figures = {key for row in wire['prds'] for key in row} - _STRUCTURAL_ROW_KEYS
    nested_figures = {FigureId.HIDDEN_HIGH.value}
    assert nested_figures <= set(wire['dark_matter'])
    assert view_figures <= figure_ids
    assert row_figures <= figure_ids
    assert view_figures | row_figures | nested_figures == figure_ids


def test_the_segment_legend_is_served_in_display_order():
    assert _wire()['segment_legend'] == [
        {'segment': segment.value, 'label': SEGMENT_LABELS[segment], 'tone': SEGMENT_TONES[segment]}
        for segment in Segment
    ]


def test_every_open_id_appears_exactly_once():
    wire = _wire(_inputs(status_map={**STATUS_MAP, 11: 'pending'}))
    listed = [
        *(task_id for row in wire['prds'] for task_id in row['direct_open_ids']),
        *(followup['task_id'] for row in wire['prds'] for followup in row['followup_open']),
        *(task_id for ids in wire['dark_matter']['members'].values() for task_id in ids),
    ]
    open_ids = {
        task_id for task_id, status in {**STATUS_MAP, 11: 'pending'}.items()
        if status not in {'done', 'cancelled'}
    }
    assert sorted(listed) == sorted(open_ids)


# --- validate_prd_view: each declared rule, breached by hand --------------


def _replace_row(view: PrdView, **changes: Any) -> PrdView:
    row, *rest = view.prds
    return dataclasses.replace(view, prds=(dataclasses.replace(row, **changes), *rest))


def _segments_off_by_one(view: PrdView) -> PrdView:
    segments = view.prds[0].segments
    return _replace_row(
        view, segments=MappingProxyType({**segments, Segment.DONE: segments[Segment.DONE] + 1}),
    )


def _no_open_member(view: PrdView) -> PrdView:
    total = view.prds[0].total
    segments = {segment: (total if segment is Segment.DONE else 0) for segment in Segment}
    return _replace_row(view, segments=MappingProxyType(segments))


def _broken_identity(view: PrdView) -> PrdView:
    return dataclasses.replace(view, unattributed_done=view.unattributed_done + 1)


def _dark_matter_off_by_one(view: PrdView) -> PrdView:
    dark_matter = view.dark_matter
    return dataclasses.replace(
        view, dark_matter=dataclasses.replace(dark_matter, total=dark_matter.total + 1),
    )


def _open_id_twice(view: PrdView) -> PrdView:
    return _replace_row(view, direct_open_ids=(*view.prds[0].direct_open_ids, 7))


def _open_id_missing(view: PrdView) -> PrdView:
    return _replace_row(view, followup_open=())


@dataclasses.dataclass(frozen=True, slots=True)
class _ViewServingAnUnruledFigure(PrdView):
    def to_wire(self) -> dict[str, object]:
        return {**PrdView.to_wire(self), 'unruled_figure': 0}


def _unruled_figure(view: PrdView) -> PrdView:
    return _ViewServingAnUnruledFigure(
        **{field.name: getattr(view, field.name) for field in dataclasses.fields(view)},
    )


_BREACHES: dict[str, tuple[Callable[[PrdView], PrdView], PrdViewInvariant, tuple[str, ...]]] = {
    'segments_sum': (_segments_off_by_one, PrdViewInvariant.SEGMENTS_SUM, (PRD_A, 'total=4')),
    'open_member': (_no_open_member, PrdViewInvariant.OPEN_MEMBER, (PRD_A,)),
    'census_identity': (_broken_identity, PrdViewInvariant.CENSUS_IDENTITY, ('= 10', '= 9')),
    'dark_matter_total': (_dark_matter_off_by_one, PrdViewInvariant.DARK_MATTER_TOTAL, ('total=4',)),
    'open_id_twice': (_open_id_twice, PrdViewInvariant.OPEN_ID_ONCE, ('[7]',)),
    'open_id_missing': (_open_id_missing, PrdViewInvariant.OPEN_ID_ONCE, ('[3]',)),
    'unruled_figure': (_unruled_figure, PrdViewInvariant.FIGURE_RULE, ('unruled_figure',)),
}


def test_a_built_view_passes_its_own_validator():
    validate_prd_view(_view(), STATUS_MAP)


@pytest.mark.parametrize('breach', list(_BREACHES.values()), ids=list(_BREACHES))
def test_validate_prd_view_refuses_each_breach(breach):
    tamper, rule, offending = breach
    with pytest.raises(PrdViewContractError) as refused:
        validate_prd_view(tamper(_view()), STATUS_MAP)
    assert isinstance(refused.value, DatumContractError)
    assert refused.value.invariant is DatumInvariant.VALUE_CONTRACT
    assert refused.value.rule is rule
    message = str(refused.value)
    assert rule.value in message
    for value in offending:
        assert value in message


def test_a_breach_lists_at_most_five_offending_ids():
    tasks = {**_TASKS, **{task_id: ('pending', 'medium', {}) for task_id in range(11, 19)}}
    status_map = {task_id: status for task_id, (status, _, _) in tasks.items()}
    view = _view(_inputs(tasks))
    duplicated = _replace_row(view, direct_open_ids=(2, *range(11, 19)))
    with pytest.raises(PrdViewContractError) as refused:
        validate_prd_view(duplicated, status_map)
    assert '[11, 12, 13, 14, 15] (+3 more)' in str(refused.value)
