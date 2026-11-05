"""The PRD lanes fixture tree: one PRD per lane scenario, the one source both lane suites read.

``test_prd_lanes.py`` builds the pure view's inputs from :func:`lanes_tree`,
and the endpoint suite serves the same tree through CannedMCP, a fixture
runs.db, an escalation queue and sidecar files, so neither can drift from the
other (PRD ``plans/dashboard-prd-lanes-prd.md`` §4.8, boundary rows 13–18).

Each :class:`Scenario` is one PRD: its tasks, the run-ledger events and
escalations naming them, and its sidecar's signal leaves when it has one.
:data:`OUTSIDE_TASKS` are the tasks no PRD owns that scenarios point at: a
pending carrier and a pending outside dependency. Task ids are unique across
the whole tree, so every scenario can be drawn from one status map.
"""

from __future__ import annotations

from collections.abc import Mapping
from datetime import datetime, timedelta
from types import MappingProxyType
from typing import Any, NamedTuple

from shared.process_identity import ProcessIdentity
from shared.task_deferral import (
    CarriedByDeferral,
    Deferral,
    HeldBySessionDeferral,
    PlanningDeferral,
    UntilConditionDeferral,
    record_to_json,
)

CARRIER = 900
"""A pending task carrying held work, owned by no PRD."""
OUTSIDE_DEPENDENCY = 901
"""A pending task scenarios depend on, owned by no PRD."""
STAMPED_BY = 'claude-task-6584-fixture'
CONDITION = 'the upstream 2.0 release ships'
_NO_EVENTS: Mapping[int, datetime] = MappingProxyType({})


class LaneTask(NamedTuple):
    status: str
    metadata: Mapping[str, Any]
    priority: str = 'medium'
    dependencies: tuple[int, ...] = ()


class FiledEscalation(NamedTuple):
    escalation_id: str
    task_id: int
    level: int


class Scenario(NamedTuple):
    """One PRD: its tasks, their run-ledger starts and done completions, escalations, leaves.

    ``signal_leaves`` maps each sidecar label to the task it is bound to; a
    scenario without one has no sidecar. ``key`` is the PRD's key, set from
    the scenario's name by :func:`lanes_tree`.
    """

    tasks: Mapping[int, LaneTask]
    started: Mapping[int, datetime] = _NO_EVENTS
    completed: Mapping[int, datetime] = _NO_EVENTS
    escalations: tuple[FiledEscalation, ...] = ()
    signal_leaves: Mapping[str, int] = MappingProxyType({})
    key: str = ''


OUTSIDE_TASKS: Mapping[int, LaneTask] = MappingProxyType({
    CARRIER: LaneTask('pending', {}),
    OUTSIDE_DEPENDENCY: LaneTask('pending', {}),
})


def prd_key(name: str) -> str:
    """Scenario *name*'s PRD key."""
    return f"plans/{name.replace('_', '-')}-prd.md"


def lanes_tree(
    at: datetime, *, live_holder: ProcessIdentity, dead_holder: ProcessIdentity,
) -> Mapping[str, Scenario]:
    """Every scenario at *at*, by name; the two holders identify real processes.

    A scenario's task with no ``spawned_from`` link names the scenario's PRD
    in its own ``prd_path``; a linked task reaches it as a follow-up.
    """

    def ago(days: float) -> datetime:
        return at - timedelta(days=days)

    def task(status: str, **metadata: Any) -> LaneTask:
        return LaneTask(status, metadata)

    def deferred(record: Deferral) -> LaneTask:
        return task('deferred', deferral=record_to_json(record))

    def stamped(days: float) -> dict[str, Any]:
        return {'deferred_at': ago(days), 'stamped_at': ago(days), 'stamped_by': STAMPED_BY}

    def done_stamped(days: float) -> LaneTask:
        return task('done', done_provenance={'kind': 'merged', 'stamped_at': ago(days).isoformat()})

    def gate(status: str, *dependencies: int, **metadata: Any) -> LaneTask:
        return LaneTask(
            status,
            {'task_kind': 'deterministic', 'always_escalates': True, **metadata},
            dependencies=dependencies,
        )

    future = {'mode': 'dated', 'at': ago(-3).isoformat()}
    past = {'mode': 'dated', 'at': ago(3).isoformat()}
    scenarios = {
        # Row 13: one PRD per precedence pair.
        'ruling_over_tail': Scenario(
            {101: task('done'), 102: task('pending')},
            escalations=(FiledEscalation('esc-102-1', 102, 2),),
            signal_leaves={'gate': 101},
        ),
        'tail_over_moving': Scenario(
            {
                111: task('done'),
                112: task('in-progress', spawned_from=111),
                113: LaneTask('pending', {'spawned_from': 111}, priority='high'),
            },
            started={112: ago(1)},
            signal_leaves={'gate': 111},
        ),
        'parked_over_moving': Scenario(
            {
                121: deferred(UntilConditionDeferral(condition=CONDITION, **stamped(1))),
                122: deferred(CarriedByDeferral(carrier_task_id=CARRIER, **stamped(1))),
            },
            started={121: ago(3)},
        ),
        'moving_over_stuck': Scenario(
            {131: task('pending'), 132: task('done')}, completed={132: ago(2)},
        ),
        # Row 14: the ruling inputs.
        'gate_deps_met': Scenario({141: task('done'), 142: gate('pending', 141)}),
        'gate_blocked': Scenario({151: gate('blocked', OUTSIDE_DEPENDENCY)}),
        'gate_deps_unmet': Scenario({161: gate('pending', OUTSIDE_DEPENDENCY)}),
        'gate_milestone_unfired': Scenario(
            {171: task('done'), 172: gate('pending', 171, milestone=future)},
        ),
        'level2_escalation': Scenario(
            {181: task('pending')}, escalations=(FiledEscalation('esc-181-2', 181, 2),),
        ),
        'level1_escalation': Scenario(
            {191: task('pending')}, escalations=(FiledEscalation('esc-191-1', 191, 1),),
        ),
        # Row 16: the moving inputs.
        'started_13d': Scenario({201: task('pending')}, started={201: ago(13)}),
        'started_15d': Scenario(
            {211: task('pending', pending_since=ago(20).isoformat())}, started={211: ago(15)},
        ),
        'done_completed_in_window': Scenario(
            {221: task('done'), 222: task('pending')}, completed={221: ago(5)},
        ),
        'done_stamped_in_window': Scenario({231: done_stamped(5), 232: task('pending')}),
        'done_stamped_outside_window': Scenario({241: done_stamped(20), 242: task('pending')}),
        'done_unstamped': Scenario({251: task('done'), 252: task('pending')}),
        'done_completed_before_stamp': Scenario(
            {261: done_stamped(5), 262: task('pending')}, completed={261: ago(20)},
        ),
        'in_progress_no_event': Scenario({271: task('in-progress')}),
        # Row 17: parked.
        'parked_caller_kinds': Scenario({
            281: deferred(CarriedByDeferral(carrier_task_id=CARRIER, **stamped(2))),
            282: deferred(UntilConditionDeferral(condition=CONDITION, **stamped(2))),
            283: deferred(HeldBySessionDeferral(holder=live_holder, **stamped(2))),
        }),
        'parked_one_unrecorded': Scenario({
            291: deferred(CarriedByDeferral(carrier_task_id=CARRIER, **stamped(2))),
            292: task('deferred'),
        }),
        'all_planning': Scenario({
            301: deferred(PlanningDeferral(**stamped(0.1))),
            302: deferred(PlanningDeferral(**stamped(0.1))),
        }),
        'held_by_dead_session': Scenario(
            {311: deferred(HeldBySessionDeferral(holder=dead_holder, **stamped(2)))},
        ),
        'milestone_unfired': Scenario({321: task('pending', milestone=future)}),
        'milestone_fired': Scenario({331: task('pending', milestone=past)}),
        'milestone_malformed': Scenario({341: task('pending', milestone={'mode': 'dated'})}),
        # Row 18: the stuck row's ready age.
        'stuck_ready_ages': Scenario({
            351: task(
                'pending', pending_since=ago(43).isoformat(), pending_since_backfilled=True,
            ),
            352: task('pending', pending_since=ago(10).isoformat()),
        }),
        # The stuck row's prose, next action and attention order.
        'blocked_by_other_prd': Scenario({
            361: LaneTask('pending', {}, dependencies=(OUTSIDE_DEPENDENCY, 131)),
            362: LaneTask('pending', {}, dependencies=(361,)),
        }),
        'stuck_high': Scenario({
            371: LaneTask('pending', {'pending_since': ago(5).isoformat()}, priority='high'),
        }),
    }
    return MappingProxyType({
        name: _owned(name, scenario) for name, scenario in scenarios.items()
    })


def _owned(name: str, scenario: Scenario) -> Scenario:
    key = prd_key(name)
    tasks = {
        task_id: task if 'spawned_from' in task.metadata
        else task._replace(metadata={'prd_path': key, **task.metadata})
        for task_id, task in scenario.tasks.items()
    }
    return scenario._replace(key=key, tasks=MappingProxyType(tasks))


def tree_tasks(tree: Mapping[str, Scenario]) -> Mapping[int, LaneTask]:
    """Every task of *tree* and the outside tasks, by id."""
    tasks = dict(OUTSIDE_TASKS)
    for scenario in tree.values():
        tasks.update(scenario.tasks)
    return MappingProxyType(tasks)


def sidecar(scenario: Scenario) -> dict[str, Any] | None:
    """*scenario*'s capability-manifest sidecar as raw data; ``None`` when it has none."""
    if not scenario.signal_leaves:
        return None
    return {
        'prd': scenario.key,
        'schema_version': 1,
        'tasks': [
            {'label': label, 'task_id': task_id, 'capabilities': []}
            for label, task_id in scenario.signal_leaves.items()
        ],
        'signal_leaves': list(scenario.signal_leaves),
    }
