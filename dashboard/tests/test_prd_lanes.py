"""A PRD's lane: what it needs, assigned by first match in precedence order (PRD §4.8).

Pure: the view is built over ``_prd_lanes_tree.py``'s scenarios, one PRD
each, with the holders of its session-held deferrals real processes whose
liveness is read from /proc (boundary rows 13, 14, 16, 17 and 18).
"""

from __future__ import annotations

import dataclasses
from collections import Counter
from collections.abc import Callable, Iterator, Mapping
from datetime import UTC, datetime, timedelta
from types import MappingProxyType
from typing import Any

import pytest
from _prd_lanes_tree import Scenario, holder_processes, lanes_tree, sidecar, tree_tasks
from shared.capability_manifest import parse_capability_manifest
from shared.process_identity import Liveness
from shared.task_deferral import NeedsHumanReason
from shared.task_statuses import TERMINAL

from dashboard.data.census import build_census
from dashboard.data.datum import Datum, DatumState
from dashboard.data.deferral_summary import read_liveness, session_holders
from dashboard.data.escalation_corpus import HUMAN_LEVEL
from dashboard.data.prd_attribution import lineage_entry
from dashboard.data.prd_view import (
    FIGURE_RULES,
    LANE_DISPLAY_ORDER,
    LANE_LABELS,
    LANE_PRECEDENCE,
    LANE_RULES,
    NEEDS_HUMAN_TEXT,
    FigureId,
    Lane,
    MilestoneStatus,
    PrdDocument,
    PrdView,
    PrdViewContractError,
    PrdViewInputs,
    PrdViewInvariant,
    TaskActivity,
    build_prd_view,
    milestone_status,
    validate_prd_view,
)

NOW = datetime(2026, 11, 5, 12, 0, tzinfo=UTC)
ROOT = '/srv/lanes'
AS_OF = NOW - timedelta(seconds=5)
BOUND = 60


@pytest.fixture(scope='module')
def tree() -> Iterator[Mapping[str, Scenario]]:
    with holder_processes() as (live, dead):
        yield lanes_tree(NOW, live_holder=live, dead_holder=dead)


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
def view(tree) -> PrdView:
    datum = build_prd_view(lanes_inputs(tree), now=NOW)
    assert datum.value is not None, datum.reason
    return datum.value


@pytest.fixture(scope='module')
def wire(view) -> dict[str, Any]:
    return view.to_wire()


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


def test_every_lane_has_one_rule():
    assert isinstance(LANE_RULES, MappingProxyType)
    assert set(LANE_RULES) == set(Lane)
    assert all(text.strip() for text in LANE_RULES.values())


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
        'gate_external_deps', 'level2_escalation', 'level1_escalation',
    ) == {
        'gate_deps_met': 'ruling',
        'gate_blocked': 'ruling',
        'gate_deps_unmet': 'stuck',
        'gate_milestone_unfired': 'parked',
        'gate_external_deps': 'stuck',
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


_NEEDING_A_HUMAN: Mapping[str, tuple[int, NeedsHumanReason]] = MappingProxyType({
    'carried_by_done_carrier': (312, NeedsHumanReason.CLOSABLE),
    'carried_by_cancelled_carrier': (313, NeedsHumanReason.ORPHANED),
    'carried_by_missing_carrier': (314, NeedsHumanReason.CARRIER_MISSING),
    'condition_stale': (315, NeedsHumanReason.STALE_CONDITION),
    'condition_expiry_overdue': (316, NeedsHumanReason.EXPIRY_OVERDUE),
    'invalid_record': (317, NeedsHumanReason.INVALID_RECORD),
    'legacy_record': (318, NeedsHumanReason.LEGACY_UNKNOWN),
})
"""Each scenario whose one deferral needs a human: that member and why."""


def test_the_session_holders_are_a_live_and_a_reaped_process(tree):
    liveness = lanes_inputs(tree).liveness.value
    assert liveness == {283: Liveness.ALIVE, 311: Liveness.DEAD}


def test_parked_needs_every_open_member_held_by_a_recorded_trigger(wire, tree):
    assert lanes_of(
        wire, tree,
        'parked_caller_kinds', 'parked_one_unrecorded', 'all_planning', 'held_by_dead_session',
        *_NEEDING_A_HUMAN, 'milestone_unfired', 'milestone_fired', 'milestone_malformed',
    ) == {
        'parked_caller_kinds': 'parked',
        'parked_one_unrecorded': 'stuck',
        'all_planning': 'stuck',
        'held_by_dead_session': 'stuck',
        **{name: 'stuck' for name in _NEEDING_A_HUMAN},
        'milestone_unfired': 'parked',
        'milestone_fired': 'stuck',
        'milestone_malformed': 'stuck',
    }


def test_a_session_hold_whose_liveness_was_not_read_needs_a_human(tree):
    inputs = lanes_inputs(tree)
    unread = dataclasses.replace(inputs, liveness=_fresh(MappingProxyType({311: Liveness.DEAD})))

    datum = build_prd_view(unread, now=NOW)

    assert datum.value is not None, datum.reason
    parked = row(datum.value.to_wire(), tree, 'parked_caller_kinds')
    assert parked['lane'] == 'stuck'
    assert parked['state_line'] == _needs_human_line(283, NeedsHumanReason.LIVENESS_UNKNOWN)
    assert parked['detail']['needs_human_refs'] == [{'task_id': 283, 'reason': 'liveness_unknown'}]


def test_every_needs_human_reason_has_its_text():
    assert set(NEEDS_HUMAN_TEXT) == set(NeedsHumanReason)
    assert all(text.strip() for text in NEEDS_HUMAN_TEXT.values())


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


# ---------------------------------------------------------------------------
# Rows 12, 17, 18: each row's state line, next action and detail
# ---------------------------------------------------------------------------


def lines_of(wire: Mapping[str, Any], tree: Mapping[str, Scenario], *names: str) -> dict[str, str]:
    return {name: row(wire, tree, name)['state_line'] for name in names}


def _needs_human_line(task_id: int, reason: NeedsHumanReason) -> str:
    return f'1 deferral needs a human: #{task_id} ({NEEDS_HUMAN_TEXT[reason]})'


def test_a_ruling_line_names_the_escalation_else_the_gate(wire, tree):
    assert lines_of(wire, tree, 'ruling_over_tail', 'gate_deps_met', 'gate_blocked') == {
        'ruling_over_tail': 'awaits ruling on esc-102-1 (#102)',
        'gate_deps_met': 'awaits ruling on gate #142',
        'gate_blocked': 'awaits ruling on gate #151',
    }


def test_a_tail_line_counts_the_open_follow_ups(wire, tree):
    assert row(wire, tree, 'tail_over_moving')['state_line'] == 'signal delivered; 2 follow-ups open'


def test_a_moving_line_always_splits_direct_from_follow_up_activity(wire, tree):
    assert lines_of(wire, tree, 'moving_over_stuck', 'started_13d', 'in_progress_no_event') == {
        'moving_over_stuck': 'moving: 1 direct, 0 follow-ups started or finished in 14 d',
        'started_13d': 'moving: 1 direct, 0 follow-ups started or finished in 14 d',
        'in_progress_no_event': (
            'moving: 0 direct, 0 follow-ups started or finished in 14 d; 1 in progress'
        ),
    }


def test_a_parked_line_counts_what_holds_it(wire, tree):
    assert lines_of(wire, tree, 'parked_caller_kinds', 'milestone_unfired') == {
        'parked_caller_kinds': 'held: 3 deferred with a recorded trigger',
        'milestone_unfired': 'held: 1 pending until a milestone fires',
    }


def test_a_stuck_line_with_ready_work_shows_its_age_bounded_only_when_lower(wire, tree):
    assert lines_of(wire, tree, 'stuck_ready_ages', 'started_15d', 'level1_escalation') == {
        'stuck_ready_ages': '2 ready, oldest pending ≥43 d; nothing started in 14 d',
        'started_15d': '1 ready, oldest pending 20 d; nothing started in 14 d',
        'level1_escalation': '1 ready; nothing started in 14 d',
    }


def test_a_stuck_line_with_nothing_ready_says_why(wire, tree):
    assert lines_of(
        wire, tree,
        'gate_deps_unmet', 'blocked_by_other_prd', 'parked_one_unrecorded', 'all_planning',
        'milestone_malformed', 'held_by_dead_session',
    ) == {
        'gate_deps_unmet': 'nothing ready: waiting on #901 (no PRD)',
        'blocked_by_other_prd': 'nothing ready: waiting on #131 (plans/moving-over-stuck-prd.md)',
        'parked_one_unrecorded': '1 deferred with no recorded reason',
        'all_planning': '2 planning births not yet committed',
        'milestone_malformed': 'malformed milestone on #341',
        'held_by_dead_session': _needs_human_line(311, NeedsHumanReason.HOLDER_DEAD),
    }


def test_a_stuck_line_names_each_deferral_that_needs_a_human_and_why(wire, tree):
    assert lines_of(wire, tree, *_NEEDING_A_HUMAN) == {
        name: _needs_human_line(task_id, reason)
        for name, (task_id, reason) in _NEEDING_A_HUMAN.items()
    }


def _action(kind: str, text: str, task_id: int | None = None, escalation_id: str | None = None,
            prd: str | None = None) -> dict[str, Any]:
    target = None if task_id is None else {
        'task_id': task_id, 'escalation_id': escalation_id, 'prd': prd,
    }
    return {'kind': kind, 'target': target, 'text': text}


def test_each_row_names_its_next_action(wire, tree):
    actions = {
        name: row(wire, tree, name)['next_action']
        for name in (
            'ruling_over_tail', 'gate_deps_met', 'stuck_ready_ages', 'parked_one_unrecorded',
            'gate_deps_unmet', 'blocked_by_other_prd', 'all_planning',
            'moving_over_stuck', 'tail_over_moving', 'parked_caller_kinds',
            'invalid_record', 'legacy_record', 'held_by_dead_session', 'carried_by_done_carrier',
        )
    }
    assert actions == {
        'ruling_over_tail': _action('rule_on', 'rule on esc-102-1 (#102)', 102, 'esc-102-1'),
        'gate_deps_met': _action('rule_on', 'rule on gate #142', 142),
        'stuck_ready_ages': _action('dispatch', 'dispatch #351', 351),
        'parked_one_unrecorded': _action('record_deferral', 'record why #292 is deferred', 292),
        'gate_deps_unmet': _action('unblock_dependency', 'unblock #901 (no PRD)', 901),
        'blocked_by_other_prd': _action(
            'unblock_dependency', 'unblock #131 (plans/moving-over-stuck-prd.md)', 131,
            prd='plans/moving-over-stuck-prd.md',
        ),
        'all_planning': _action('none', ''),
        'moving_over_stuck': _action('none', ''),
        'tail_over_moving': _action('none', ''),
        'parked_caller_kinds': _action('none', ''),
        'invalid_record': _action('record_deferral', 'record why #317 is deferred', 317),
        'legacy_record': _action('record_deferral', 'record why #318 is deferred', 318),
        'held_by_dead_session': _action('none', ''),
        'carried_by_done_carrier': _action('none', ''),
    }


def test_a_rows_detail_lists_the_ids_behind_its_figures(wire, tree):
    assert row(wire, tree, 'blocked_by_other_prd')['detail']['blocker_refs'] == [
        {'task_id': 131, 'prd': 'plans/moving-over-stuck-prd.md'},
        {'task_id': 901, 'prd': None},
    ]
    assert row(wire, tree, 'ruling_over_tail')['detail']['ruling_refs'] == [
        {'task_id': 102, 'escalation_id': 'esc-102-1'},
    ]
    assert row(wire, tree, 'gate_deps_met')['detail']['ruling_refs'] == [
        {'task_id': 142, 'escalation_id': None},
    ]
    assert row(wire, tree, 'parked_one_unrecorded')['detail']['deferred_unrecorded_ids'] == [292]
    assert row(wire, tree, 'held_by_dead_session')['detail']['needs_human_refs'] == [
        {'task_id': 311, 'reason': 'holder_dead'},
    ]
    assert row(wire, tree, 'tail_over_moving')['detail']['high_open_ids'] == [113]
    assert row(wire, tree, 'stuck_high')['detail']['oldest_ready_high_ids'] == [371]
    assert row(wire, tree, 'started_15d')['detail'] == {
        'blocker_refs': [], 'ruling_refs': [], 'oldest_ready_high_ids': [],
        'deferred_unrecorded_ids': [], 'needs_human_refs': [], 'high_open_ids': [],
    }


def _attention(row: Mapping[str, Any]) -> tuple[int, int, int, str]:
    age = row['ready']['oldest_pending_days']
    return (
        -len(row['detail']['high_open_ids']), -(-1 if age is None else age),
        -row['ready']['count'], row['key'],
    )


def test_rows_within_a_lane_are_ordered_by_attention(wire, tree):
    rows = {row['key']: row for row in wire['prds']}
    for entry in wire['lanes']:
        assert entry['prd_keys'] == sorted(entry['prd_keys'], key=lambda key: _attention(rows[key]))
    [stuck] = [entry for entry in wire['lanes'] if entry['lane'] == 'stuck']
    assert stuck['prd_keys'][:3] == [
        tree['stuck_high'].key, tree['stuck_ready_ages'].key, tree['started_15d'].key,
    ]


def test_the_rows_are_served_lane_by_lane_in_display_order(wire):
    assert [row['key'] for row in wire['prds']] == [
        key for entry in wire['lanes'] for key in entry['prd_keys']
    ]


def test_high_work_under_a_tail_prd_is_listed_in_dark_matter(wire, tree):
    assert wire['dark_matter']['hidden_high'] == {
        'count': 1, 'refs': [{'task_id': 113, 'prd': tree['tail_over_moving'].key}],
    }


# ---------------------------------------------------------------------------
# Row 12 extended: the γ2 figures are present and ruled; the lanes partition
# ---------------------------------------------------------------------------

_ROW_FIGURES = ('title', 'lane', 'state_line', 'signal', 'next_action', 'activity', 'detail')
_VIEW_FIGURES = ('lanes', 'signal_coverage')


def test_every_row_and_the_view_carry_every_gamma_2_figure(wire):
    for served in wire['prds']:
        assert set(_ROW_FIGURES) <= set(served)
    assert set(_VIEW_FIGURES) <= set(wire)
    assert 'hidden_high' in wire['dark_matter']


def test_the_gamma_2_figures_carry_rule_text():
    for figure in (
        FigureId.STATE_LINE, FigureId.NEXT_ACTION, FigureId.DETAIL, FigureId.HIDDEN_HIGH,
    ):
        assert FIGURE_RULES[figure].strip()


def _replace_lanes(view: PrdView, change: Callable[[list[tuple[Lane, tuple[str, ...]]]], None]):
    entries = [(entry.lane, entry.prd_keys) for entry in view.lanes]
    change(entries)
    return dataclasses.replace(view, lanes=tuple(
        dataclasses.replace(view.lanes[0], lane=lane, prd_keys=keys) for lane, keys in entries
    ))


def _omit_a_key(view: PrdView) -> PrdView:
    def change(entries):
        lane, keys = entries[0]
        entries[0] = (lane, keys[1:])
    return _replace_lanes(view, change)


def _duplicate_a_key(view: PrdView) -> PrdView:
    def change(entries):
        lane, keys = entries[1]
        entries[1] = (lane, (*keys, entries[0][1][0]))
    return _replace_lanes(view, change)


def _disagree_with_the_row(view: PrdView) -> PrdView:
    def change(entries):
        (first, first_keys), (second, second_keys) = entries[0], entries[1]
        entries[0] = (first, first_keys[1:])
        entries[1] = (second, (*second_keys, first_keys[0]))
    return _replace_lanes(view, change)


def _without(wire: Mapping[str, Any], path: tuple[str, ...]) -> dict[str, Any]:
    """*wire* with the figure at *path* removed; a list step descends into its first item."""
    head, *tail = path
    if not tail:
        return {key: value for key, value in wire.items() if key != head}
    value = wire[head]
    if isinstance(value, list):
        first, *rest = value
        return {**wire, head: [_without(first, tuple(tail)), *rest]}
    return {**wire, head: _without(value, tuple(tail))}


@dataclasses.dataclass(frozen=True, slots=True)
class _ViewDroppingAFigure(PrdView):
    path: tuple[str, ...] = ()

    def to_wire(self) -> dict[str, object]:
        return _without(PrdView.to_wire(self), self.path)


def _dropping(*path: str) -> Callable[[PrdView], PrdView]:
    def tamper(view: PrdView) -> PrdView:
        fields = {field.name: getattr(view, field.name) for field in dataclasses.fields(view)}
        return _ViewDroppingAFigure(**fields, path=path)
    return tamper


_LANE_BREACHES = {
    'omits_a_key': (_omit_a_key, 'lanes_partition'),
    'duplicates_a_key': (_duplicate_a_key, 'lanes_partition'),
    'disagrees_with_the_row': (_disagree_with_the_row, 'lanes_partition'),
    'row_missing_state_line': (_dropping('prds', 'state_line'), 'figure_present'),
    'view_missing_lanes': (_dropping('lanes'), 'figure_present'),
    'dark_matter_missing_hidden_high': (
        _dropping('dark_matter', 'hidden_high'), 'figure_present',
    ),
}


@pytest.mark.parametrize('breach', list(_LANE_BREACHES.values()), ids=list(_LANE_BREACHES))
def test_validate_prd_view_refuses_each_lane_breach(breach, view, tree):
    tamper, rule = breach
    status_map = dict(lanes_inputs(tree).status_map)
    validate_prd_view(view, status_map)
    with pytest.raises(PrdViewContractError) as refused:
        validate_prd_view(tamper(view), status_map)
    assert refused.value.rule is PrdViewInvariant(rule)
