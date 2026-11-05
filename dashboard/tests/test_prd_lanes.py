"""A PRD's lane: what it needs, assigned by first match in precedence order (PRD §4.8).

Pure: the view is built over ``_prd_lanes_tree.py``'s scenarios, one PRD
each, with the holders of its session-held deferrals real processes whose
liveness is read from /proc (boundary rows 13, 14, 16, 17 and 18).
"""

from __future__ import annotations

import subprocess
import sys
from collections import Counter
from collections.abc import Iterator, Mapping
from datetime import UTC, datetime, timedelta
from types import MappingProxyType
from typing import Any

import pytest
from _prd_lanes_tree import Scenario, lanes_tree, sidecar, tree_tasks
from shared.capability_manifest import parse_capability_manifest
from shared.process_identity import Liveness, ProcessIdentity, capture
from shared.task_statuses import TERMINAL

from dashboard.data.census import build_census
from dashboard.data.datum import Datum, DatumState
from dashboard.data.deferral_summary import read_liveness, session_holders
from dashboard.data.escalation_corpus import HUMAN_LEVEL
from dashboard.data.prd_attribution import lineage_entry
from dashboard.data.prd_view import (
    LANE_DISPLAY_ORDER,
    LANE_LABELS,
    LANE_PRECEDENCE,
    LANE_RULES,
    MOVING_WINDOW_DAYS,
    FigureId,
    Lane,
    MilestoneStatus,
    PrdDocument,
    PrdViewInputs,
    TaskActivity,
    build_prd_view,
    milestone_status,
)

NOW = datetime(2026, 11, 5, 12, 0, tzinfo=UTC)
ROOT = '/srv/lanes'
AS_OF = NOW - timedelta(seconds=5)
BOUND = 60


def _blocked_child() -> subprocess.Popen[bytes]:
    """A real process that runs until its stdin is closed."""
    return subprocess.Popen(
        [sys.executable, '-c', 'import sys; sys.stdin.read()'], stdin=subprocess.PIPE,
    )


def _identity(child: subprocess.Popen[bytes]) -> ProcessIdentity:
    holder = capture(child.pid)
    assert holder is not None, f'could not identify spawned child {child.pid}'
    return holder


def _reap(child: subprocess.Popen[bytes]) -> None:
    assert child.stdin is not None
    child.stdin.close()
    child.wait(timeout=10)


@pytest.fixture(scope='module')
def live_holder() -> Iterator[ProcessIdentity]:
    child = _blocked_child()
    try:
        yield _identity(child)
    finally:
        _reap(child)


@pytest.fixture(scope='module')
def dead_holder() -> ProcessIdentity:
    child = _blocked_child()
    try:
        return _identity(child)
    finally:
        _reap(child)


@pytest.fixture(scope='module')
def tree(live_holder, dead_holder) -> Mapping[str, Scenario]:
    return lanes_tree(NOW, live_holder=live_holder, dead_holder=dead_holder)


def _fresh(value: Any) -> Datum[Any]:
    return Datum(value, AS_OF, DatumState.FRESH, None, BOUND)


def lanes_inputs(tree: Mapping[str, Scenario]) -> PrdViewInputs:
    """The view's inputs over *tree*, each read the way its access module reads it."""
    tasks = tree_tasks(tree)
    status_map = {task_id: task.status for task_id, task in tasks.items()}
    rows = [
        {
            'id': task_id, 'status': task.status, 'priority': task.priority,
            'dependencies': list(task.dependencies), 'metadata': dict(task.metadata),
        }
        for task_id, task in tasks.items()
        if task.status not in TERMINAL
    ]
    escalations: dict[int, tuple[str, ...]] = {}
    started: dict[int, datetime] = {}
    completed: dict[int, datetime] = {}
    documents: dict[str, PrdDocument] = {}
    for scenario in tree.values():
        for filed in scenario.escalations:
            if filed.level == HUMAN_LEVEL:
                escalations[filed.task_id] = (*escalations.get(filed.task_id, ()), filed.escalation_id)
        started.update(scenario.started)
        completed.update(scenario.completed)
        if (raw := sidecar(scenario)) is not None:
            documents[scenario.key] = PrdDocument(None, parse_capability_manifest(raw))
    activity = TaskActivity(
        started=MappingProxyType(started),
        completed=MappingProxyType(completed),
        ever_started=frozenset(started),
        first_event_at=min([*started.values(), *completed.values()]),
    )
    return PrdViewInputs(
        project_root=ROOT,
        status_map=status_map,
        census=_fresh(build_census(status_map)),
        rows=_fresh(rows),
        lineage=_fresh(MappingProxyType({
            task_id: lineage_entry(task.metadata, ROOT) for task_id, task in tasks.items()
        })),
        escalations=_fresh(MappingProxyType(escalations)),
        activity=_fresh(activity),
        documents=_fresh(MappingProxyType(documents)),
        liveness=_fresh(MappingProxyType(read_liveness(session_holders(rows)))),
    )


@pytest.fixture(scope='module')
def wire(tree) -> dict[str, Any]:
    datum = build_prd_view(lanes_inputs(tree), now=NOW)
    assert datum.value is not None, datum.reason
    return datum.value.to_wire()


def row(wire: Mapping[str, Any], tree: Mapping[str, Scenario], name: str) -> dict[str, Any]:
    [found] = [row for row in wire['prds'] if row['key'] == tree[name].key]
    return found


def lanes_of(wire: Mapping[str, Any], tree: Mapping[str, Scenario], *names: str) -> dict[str, str]:
    return {name: row(wire, tree, name)['lane'] for name in names}


# ---------------------------------------------------------------------------
# Row 13: precedence, and the lanes as a partition
# ---------------------------------------------------------------------------


def test_each_precedence_pair_lands_in_its_first_matching_lane(wire, tree):
    assert lanes_of(
        wire, tree, 'ruling_over_tail', 'tail_over_moving', 'parked_over_moving', 'moving_over_stuck',
    ) == {
        'ruling_over_tail': 'ruling',
        'tail_over_moving': 'tail',
        'parked_over_moving': 'parked',
        'moving_over_stuck': 'moving',
    }


def test_every_drawn_prd_appears_in_exactly_one_lane_equal_to_its_rows_lane(wire):
    listed = Counter(key for entry in wire['lanes'] for key in entry['prd_keys'])
    assert listed == Counter(row['key'] for row in wire['prds'])
    by_key = {key: entry['lane'] for entry in wire['lanes'] for key in entry['prd_keys']}
    assert by_key == {row['key']: row['lane'] for row in wire['prds']}


def test_every_lane_is_served_in_display_order_with_its_label_and_rule(wire):
    assert [entry['lane'] for entry in wire['lanes']] == [
        'ruling', 'moving', 'stuck', 'parked', 'tail',
    ]
    assert [Lane(entry['lane']) for entry in wire['lanes']] == list(LANE_DISPLAY_ORDER)
    for entry in wire['lanes']:
        lane = Lane(entry['lane'])
        assert entry['label'] == LANE_LABELS[lane]
        assert entry['rule'] == LANE_RULES[lane]


def test_the_precedence_is_served_beside_the_lanes(wire):
    assert wire['lane_precedence'] == ['ruling', 'tail', 'parked', 'moving', 'stuck']
    assert list(LANE_PRECEDENCE) == [Lane(lane) for lane in wire['lane_precedence']]


def test_every_lane_has_one_rule_and_the_moving_rule_names_its_window():
    assert isinstance(LANE_RULES, MappingProxyType)
    assert set(LANE_RULES) == set(Lane)
    assert all(text.strip() for text in LANE_RULES.values())
    assert f'{MOVING_WINDOW_DAYS} days' in LANE_RULES[Lane.MOVING]


def test_the_lane_figures_carry_rule_text(wire):
    for figure in (FigureId.LANE, FigureId.LANES, FigureId.ACTIVITY):
        assert wire['rules'][figure.value].strip()


# ---------------------------------------------------------------------------
# Row 14: what puts a PRD in ruling
# ---------------------------------------------------------------------------


def test_only_actionable_gates_and_level2_escalations_put_a_prd_in_ruling(wire, tree):
    assert lanes_of(
        wire, tree,
        'gate_deps_met', 'gate_blocked', 'gate_deps_unmet', 'gate_milestone_unfired',
        'level2_escalation', 'level1_escalation',
    ) == {
        'gate_deps_met': 'ruling',
        'gate_blocked': 'ruling',
        'gate_deps_unmet': 'stuck',
        'gate_milestone_unfired': 'parked',
        'level2_escalation': 'ruling',
        'level1_escalation': 'stuck',
    }


# ---------------------------------------------------------------------------
# Row 16: what moves a PRD
# ---------------------------------------------------------------------------


def test_moving_reads_starts_and_completions_within_the_window(wire, tree):
    assert lanes_of(
        wire, tree,
        'started_13d', 'started_15d', 'done_completed_in_window', 'done_stamped_in_window',
        'done_stamped_outside_window', 'done_unstamped', 'done_completed_before_stamp',
        'in_progress_no_event',
    ) == {
        'started_13d': 'moving',
        'started_15d': 'stuck',
        'done_completed_in_window': 'moving',
        'done_stamped_in_window': 'moving',
        'done_stamped_outside_window': 'stuck',
        'done_unstamped': 'stuck',
        'done_completed_before_stamp': 'stuck',
        'in_progress_no_event': 'moving',
    }


def test_a_rows_activity_counts_recent_members_split_direct_from_followup(wire, tree):
    assert row(wire, tree, 'tail_over_moving')['activity'] == {'direct': 0, 'followup': 1}
    assert row(wire, tree, 'moving_over_stuck')['activity'] == {'direct': 1, 'followup': 0}
    assert row(wire, tree, 'done_stamped_in_window')['activity'] == {'direct': 1, 'followup': 0}
    assert row(wire, tree, 'started_15d')['activity'] == {'direct': 0, 'followup': 0}


# ---------------------------------------------------------------------------
# Row 17: what parks a PRD
# ---------------------------------------------------------------------------


def test_the_session_holders_are_a_live_and_a_reaped_process(tree):
    liveness = lanes_inputs(tree).liveness.value
    assert liveness == {283: Liveness.ALIVE, 311: Liveness.DEAD}


def test_parked_needs_every_open_member_held_by_a_recorded_trigger(wire, tree):
    assert lanes_of(
        wire, tree,
        'parked_caller_kinds', 'parked_one_unrecorded', 'all_planning', 'held_by_dead_session',
        'milestone_unfired', 'milestone_fired', 'milestone_malformed',
    ) == {
        'parked_caller_kinds': 'parked',
        'parked_one_unrecorded': 'stuck',
        'all_planning': 'stuck',
        'held_by_dead_session': 'stuck',
        'milestone_unfired': 'parked',
        'milestone_fired': 'stuck',
        'milestone_malformed': 'stuck',
    }


_DELAYED = {'mode': 'delayed', 'after_secs': 3600}


@pytest.mark.parametrize(
    ('metadata', 'expected'),
    [
        ({}, MilestoneStatus.NONE),
        ({'milestone': {'mode': 'dated', 'at': (NOW + timedelta(days=1)).isoformat()}},
         MilestoneStatus.UNFIRED),
        ({'milestone': {'mode': 'dated', 'at': (NOW - timedelta(days=1)).isoformat()}},
         MilestoneStatus.FIRED),
        ({'milestone': _DELAYED}, MilestoneStatus.UNFIRED),
        ({'milestone': _DELAYED,
          'milestone_deps_satisfied_at': (NOW - timedelta(hours=2)).isoformat()},
         MilestoneStatus.FIRED),
        ({'milestone': {'mode': 'dated'}}, MilestoneStatus.MALFORMED),
        ({'milestone': _DELAYED, 'milestone_deps_satisfied_at': 12345}, MilestoneStatus.MALFORMED),
        ({'milestone': _DELAYED, 'milestone_deps_satisfied_at': 'not a time'},
         MilestoneStatus.MALFORMED),
    ],
    ids=[
        'none', 'dated_future', 'dated_past', 'delayed_unanchored', 'delayed_elapsed',
        'dated_without_at', 'non_string_anchor', 'unparseable_anchor',
    ],
)
def test_milestone_status(metadata, expected):
    assert milestone_status(metadata, NOW) is expected


# ---------------------------------------------------------------------------
# Row 18: a stuck row's ready age
# ---------------------------------------------------------------------------


def test_a_stuck_rows_ready_age_runs_from_its_oldest_anchor_as_a_lower_bound(wire, tree):
    stuck = row(wire, tree, 'stuck_ready_ages')
    assert stuck['lane'] == 'stuck'
    assert stuck['ready'] == {
        'count': 2, 'oldest_task_id': 351, 'oldest_pending_days': 43,
        'oldest_is_lower_bound': True,
    }
