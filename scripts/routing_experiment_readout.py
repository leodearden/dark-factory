#!/usr/bin/env python3
"""Read a routing split experiment out of every project ledger, and check an A/A split.

Contract: plans/routing-experiment-substrate-prd.md §"The readout (β)". Strictly
read-only: every store is opened with a ``mode=ro`` URI.

Exit codes:
  0  report printed
  1  --aa: a headline metric's |observed arm difference| exceeds its null p99
  2  usage error (argparse)
  3  nothing was read: no root resolved, or no root's runs.db was readable
  4  unproven: an --expect-rule id is absent from the window, --subset has no
     plan-shape producer row, or an --aa headline metric has too few eligible
     tasks in an arm
"""
from __future__ import annotations

import argparse
import itertools
import json
import math
import os
import random
import shlex
import sqlite3
import statistics
import subprocess
import sys
from collections import Counter, defaultdict
from collections.abc import Callable, Iterable, Mapping, Sequence
from contextlib import closing
from dataclasses import asdict, dataclass, field
from datetime import UTC, datetime, timedelta
from enum import Enum
from operator import itemgetter
from pathlib import Path
from statistics import NormalDist
from typing import Any

import _checkout as _checkout
from _task_db_scan import UNREADABLE_STORE_ERRORS, decode_metadata
from shared.project_layout import KNOWN_PROJECT_ROOTS_ENV, known_project_roots, runs_db, tasks_db
from shared.routing_split import split_bucket
from shared.sqlite_readonly import connect_readonly
from shared.tasks_db_readonly import TaskDbUnreadable, connect_ro
from systemd_unit_environment import parse_assignments

EXIT_OK = 0  # report printed
EXIT_AA_SKEWED = 1  # --aa: a headline metric's |diff| exceeds its null p99
EXIT_NOTHING_READ = 3  # no root resolved, or no runs.db readable
EXIT_UNPROVEN = 4  # a producer or eligibility precondition of the claim is unmet

DEFAULT_SEED = 20261103  # seat N's prototype seed
DEFAULT_RESAMPLES = 4000  # the PRD's A/A null size

FUSED_MEMORY_UNIT = 'fused-memory.service'
_UNIT_ENVIRONMENT_ARGV = (
    'systemctl', '--user', 'show', FUSED_MEMORY_UNIT, '-p', 'Environment', '--value',
)
_UNIT_READ_TIMEOUT_SECS = 10


class RootSource(Enum):
    """Which source named the roots a report was computed over."""

    CLI = '--roots'
    ENVIRONMENT = KNOWN_PROJECT_ROOTS_ENV
    UNIT = f'{FUSED_MEMORY_UNIT} Environment'
    NONE = 'no source'


@dataclass(frozen=True)
class RootSelection:
    roots: tuple[Path, ...]
    source: RootSource


class KeyKind(Enum):
    """The stable key a task's arm is computed from."""

    TASK_ID = 'task_id'
    CANDIDATE_KEY = 'candidate_key'


class Subset(Enum):
    ALL = 'all'
    EXACT_3_MODULES = 'exact-3-modules'


@dataclass(frozen=True)
class Window:
    """The half-open interval [start, end) a report reads, both bounds in UTC."""

    start: datetime
    end: datetime

    @classmethod
    def trailing(cls, end: datetime, days: int) -> Window:
        return cls(end - timedelta(days=days), end)

    def sql_bounds(self) -> tuple[str, str]:
        """The bounds as the ISO-8601 UTC text the ledgers store, for TEXT comparison."""
        return self.start.isoformat(), self.end.isoformat()


def _roots_listed(text: str) -> tuple[Path, ...]:
    return known_project_roots({KNOWN_PROJECT_ROOTS_ENV: text})


def _unit_roots(unit_text: str | None) -> tuple[Path, ...]:
    if not unit_text:
        return ()
    try:
        entries = shlex.split(unit_text)
    except ValueError:
        return ()
    return known_project_roots(parse_assignments(entries))


def resolve_roots(
    cli_roots: str | None,
    environ: Mapping[str, str],
    unit_environment: Callable[[], str | None],
) -> RootSelection:
    """The roots to read: --roots, else the environment, else the installed fused-memory unit.

    There is deliberately no default-project-root fallback: reading dark-factory
    alone would certify only the null-contrast project.
    """
    if cli_roots is not None:
        return RootSelection(_roots_listed(cli_roots), RootSource.CLI)
    from_environment = known_project_roots(environ)
    if from_environment:
        return RootSelection(from_environment, RootSource.ENVIRONMENT)
    from_unit = _unit_roots(unit_environment())
    if from_unit:
        return RootSelection(from_unit, RootSource.UNIT)
    return RootSelection((), RootSource.NONE)


def read_fused_memory_unit_environment() -> str | None:
    """The installed fused-memory unit's Environment= text, or None when systemd does not answer."""
    try:
        completed = subprocess.run(
            _UNIT_ENVIRONMENT_ARGV,
            capture_output=True,
            text=True,
            timeout=_UNIT_READ_TIMEOUT_SECS,
            check=False,
        )
    except (OSError, subprocess.TimeoutExpired):
        return None
    return completed.stdout if completed.returncode == 0 else None


# The large-plan rule ids: orchestrator/src/orchestrator/defaults.yaml routing.rules,
# plus experiment 1's large-plan-modules-4 (plans/routing-experiment-substrate-prd.md
# §"Experiment 1"). Matched by membership, never by prefix.
LARGE_PLAN_RULE_IDS = frozenset({'large-plan-steps', 'large-plan-modules', 'large-plan-modules-4'})
BLOCKED_OUTCOMES = frozenset({'blocked', 'escalated'})
IMPLEMENTER_ROLE = 'implementer'
CURATOR_HINT_KEY = 'x_curator_complexity_hint'
_MS_PER_MINUTE = 60_000

_ROUTING_DECISION = 'routing_decision'
_WORKFLOW_VERIFY = 'workflow_verify'
_MERGE_VERIFY = 'merge_verify'
_REVIEW_COMPLETED = 'review_completed'
_READ_EVENT_TYPES = (_ROUTING_DECISION, _WORKFLOW_VERIFY, _MERGE_VERIFY, _REVIEW_COMPLETED)

NO_CANDIDATE_KEY = 'no candidate_key'


@dataclass(frozen=True)
class TaskRecord:
    """One task of the population: its arm and every per-task metric input."""

    project: str
    task_id: str
    key: str
    bucket: int
    implementer_models: frozenset[str]
    plan_steps: int | None
    plan_modules: int | None
    large_plan_rule_fired: bool
    complexity: str | None
    curator_complexity_hint: str | None
    final_outcome: str
    blocked_or_escalated: bool
    max_review_cycles: int
    max_execute_iterations: int
    cost_usd: float
    workflow_verify_minutes: float | None
    merge_verify_minutes: float | None
    reviewer_blocking_count: float | None


@dataclass(frozen=True)
class ProducerEvidence:
    """Whether the producers the metrics depend on wrote anything in the window."""

    rule_id_counts: dict[str, int] = field(default_factory=dict)
    plan_shape_decisions: int = 0
    hinted_tasks: int = 0

    @classmethod
    def combined(cls, parts: Iterable[ProducerEvidence]) -> ProducerEvidence:
        parts = tuple(parts)
        rule_ids: Counter[str] = Counter()
        for part in parts:
            rule_ids.update(part.rule_id_counts)
        return cls(
            dict(sorted(rule_ids.items())),
            sum(part.plan_shape_decisions for part in parts),
            sum(part.hinted_tasks for part in parts),
        )


@dataclass(frozen=True)
class RootStatus:
    """What happened to one root's stores; a problem is a human-readable reason, None when read."""

    root: Path
    runs_db_problem: str | None = None
    tasks_db_problem: str | None = None

    @property
    def project(self) -> str:
        return self.root.name

    @property
    def read(self) -> bool:
        return self.runs_db_problem is None


@dataclass(frozen=True)
class FleetLoad:
    records: tuple[TaskRecord, ...]
    roots: tuple[RootStatus, ...]
    evidence: ProducerEvidence
    exclusions: dict[str, int] = field(default_factory=dict)

    @classmethod
    def combined(cls, parts: Iterable[FleetLoad]) -> FleetLoad:
        parts = tuple(parts)
        exclusions: Counter[str] = Counter()
        for part in parts:
            exclusions.update(part.exclusions)
        return cls(
            tuple(record for part in parts for record in part.records),
            tuple(status for part in parts for status in part.roots),
            ProducerEvidence.combined(part.evidence for part in parts),
            dict(sorted(exclusions.items())),
        )


@dataclass(frozen=True)
class _Result:
    outcome: str
    review_cycles: int
    execute_iterations: int


@dataclass(frozen=True)
class _Invocation:
    role: str
    model: str
    cost_usd: float


@dataclass(frozen=True)
class _Event:
    event_type: str
    role: str | None
    data: dict


@dataclass(frozen=True)
class _TaskRow:
    metadata: dict
    candidate_key: str | None


_NO_TASK_ROW = _TaskRow({}, None)


@dataclass(frozen=True)
class _TaskLedger:
    """One population task's in-window rows from runs.db."""

    results: tuple[_Result, ...]
    invocations: tuple[_Invocation, ...]
    events: tuple[_Event, ...]

    def payloads(self, event_type: str) -> list[dict]:
        return [event.data for event in self.events if event.event_type == event_type]

    def decisions(self) -> list[_Event]:
        return [event for event in self.events if event.event_type == _ROUTING_DECISION]


def _text(value: object) -> str | None:
    return value if isinstance(value, str) else None


def _integer(value: object) -> int | None:
    return value if isinstance(value, int) and not isinstance(value, bool) else None


def _population(conn: sqlite3.Connection, window: Window) -> dict[str, list[_Result]]:
    population: dict[str, list[_Result]] = defaultdict(list)
    rows = conn.execute(
        'SELECT task_id, outcome, review_cycles, execute_iterations FROM task_results '
        'WHERE completed_at >= ? AND completed_at < ? ORDER BY task_id, completed_at, rowid',
        window.sql_bounds(),
    )
    for task_id, outcome, review_cycles, execute_iterations in rows:
        population[task_id].append(_Result(outcome, review_cycles or 0, execute_iterations or 0))
    return population


def _invocations(conn: sqlite3.Connection, window: Window) -> dict[str, list[_Invocation]]:
    invocations: dict[str, list[_Invocation]] = defaultdict(list)
    rows = conn.execute(
        'SELECT task_id, role, model, cost_usd FROM invocations '
        'WHERE started_at >= ? AND started_at < ? AND task_id IS NOT NULL',
        window.sql_bounds(),
    )
    for task_id, role, model, cost_usd in rows:
        invocations[task_id].append(_Invocation(role, model, cost_usd or 0.0))
    return invocations


def _events(conn: sqlite3.Connection, window: Window) -> dict[str, list[_Event]]:
    events: dict[str, list[_Event]] = defaultdict(list)
    placeholders = ', '.join('?' for _ in _READ_EVENT_TYPES)
    rows = conn.execute(
        'SELECT task_id, event_type, role, data FROM events '
        f'WHERE timestamp >= ? AND timestamp < ? AND event_type IN ({placeholders}) '
        'AND task_id IS NOT NULL ORDER BY id',
        (*window.sql_bounds(), *_READ_EVENT_TYPES),
    )
    for task_id, event_type, role, data in rows:
        events[task_id].append(_Event(event_type, role, decode_metadata(data)))
    return events


def _task_rows(path: Path) -> dict[str, _TaskRow]:
    with closing(connect_ro(path)) as conn:
        rows = conn.execute("SELECT id, metadata, candidate_key FROM tasks WHERE tag = 'master'")
        return {
            str(task_id): _TaskRow(decode_metadata(metadata), candidate_key)
            for task_id, metadata, candidate_key in rows
        }


def _store_problem(store: str, exc: Exception) -> str:
    if isinstance(exc, TaskDbUnreadable):
        return f'{store} {exc.reason.value}'
    return f'{store} unreadable: {exc}'


def _summed_payload_field(payloads: Sequence[dict], name: str, divisor: float = 1.0) -> float | None:
    """The sum of *name* over *payloads* divided by *divisor*; 0.0 for none, None if any lacks it.

    The one field-presence rule: an event written before its producer carried
    the field makes the metric unknown for that task, rather than silently low.
    """
    numbers: list[float] = []
    for payload in payloads:
        value = payload.get(name)
        if isinstance(value, bool) or not isinstance(value, (int, float)):
            return None
        numbers.append(value)
    return math.fsum(numbers) / divisor


def _carries_plan_shape(decision: _Event) -> bool:
    return (
        decision.role == IMPLEMENTER_ROLE
        and _integer(decision.data.get('plan_steps')) is not None
        and _integer(decision.data.get('plan_modules')) is not None
    )


def _plan_shape(decisions: Sequence[_Event]) -> tuple[int | None, int | None]:
    """The earliest implementer decision's plan shape: the one the resolver chose its model by."""
    for decision in decisions:
        if _carries_plan_shape(decision):
            return _integer(decision.data['plan_steps']), _integer(decision.data['plan_modules'])
    return None, None


def _record(
    project: str, task_id: str, key: str, bucket: int, ledger: _TaskLedger, task_row: _TaskRow,
) -> TaskRecord:
    decisions = ledger.decisions()
    plan_steps, plan_modules = _plan_shape(decisions)
    return TaskRecord(
        project=project,
        task_id=task_id,
        key=key,
        bucket=bucket,
        implementer_models=frozenset(
            invocation.model for invocation in ledger.invocations
            if invocation.role == IMPLEMENTER_ROLE
        ),
        plan_steps=plan_steps,
        plan_modules=plan_modules,
        large_plan_rule_fired=any(
            _text(decision.data.get('rule_id')) in LARGE_PLAN_RULE_IDS for decision in decisions
        ),
        complexity=_text(task_row.metadata.get('complexity')),
        curator_complexity_hint=_text(task_row.metadata.get(CURATOR_HINT_KEY)),
        final_outcome=ledger.results[-1].outcome,
        blocked_or_escalated=any(result.outcome in BLOCKED_OUTCOMES for result in ledger.results),
        max_review_cycles=max(result.review_cycles for result in ledger.results),
        max_execute_iterations=max(result.execute_iterations for result in ledger.results),
        cost_usd=math.fsum(invocation.cost_usd for invocation in ledger.invocations),
        workflow_verify_minutes=_summed_payload_field(
            ledger.payloads(_WORKFLOW_VERIFY), 'duration_ms', _MS_PER_MINUTE
        ),
        merge_verify_minutes=_summed_payload_field(
            ledger.payloads(_MERGE_VERIFY), 'duration_ms', _MS_PER_MINUTE
        ),
        reviewer_blocking_count=_summed_payload_field(
            ledger.payloads(_REVIEW_COMPLETED), 'blocking_count'
        ),
    )


def _evidence(
    events: Mapping[str, Sequence[_Event]], records: Sequence[TaskRecord],
) -> ProducerEvidence:
    decisions = [
        event for task_events in events.values() for event in task_events
        if event.event_type == _ROUTING_DECISION
    ]
    rule_ids = Counter(
        rule_id for decision in decisions
        if (rule_id := _text(decision.data.get('rule_id'))) is not None
    )
    return ProducerEvidence(
        rule_id_counts=dict(sorted(rule_ids.items())),
        plan_shape_decisions=sum(1 for decision in decisions if _carries_plan_shape(decision)),
        hinted_tasks=sum(1 for record in records if record.curator_complexity_hint is not None),
    )


def _read_task_rows(root: Path) -> tuple[dict[str, _TaskRow], str | None]:
    try:
        return _task_rows(tasks_db(root)), None
    except UNREADABLE_STORE_ERRORS as exc:
        return {}, _store_problem('tasks.db', exc)


def _read_runs_db(
    path: Path, window: Window,
) -> tuple[dict[str, _TaskLedger], dict[str, list[_Event]]]:
    """Each population task's ledger, and every in-window event by task (for producer evidence)."""
    with closing(connect_readonly(path)) as conn:
        population = _population(conn, window)
        invocations = _invocations(conn, window)
        events = _events(conn, window)
    ledgers = {
        task_id: _TaskLedger(
            tuple(results), tuple(invocations.get(task_id, ())), tuple(events.get(task_id, ()))
        )
        for task_id, results in population.items()
    }
    return ledgers, events


def _split_key(task_id: str, task_row: _TaskRow, kind: KeyKind) -> str | None:
    if kind is KeyKind.TASK_ID:
        return task_id
    return task_row.candidate_key


def _keyed_records(
    project: str,
    ledgers: Mapping[str, _TaskLedger],
    task_rows: Mapping[str, _TaskRow],
    salt: str,
    buckets: int,
    kind: KeyKind,
) -> tuple[tuple[TaskRecord, ...], dict[str, int]]:
    records: list[TaskRecord] = []
    exclusions: Counter[str] = Counter()
    for task_id, ledger in ledgers.items():
        task_row = task_rows.get(task_id, _NO_TASK_ROW)
        key = _split_key(task_id, task_row, kind)
        if key is None:
            exclusions[NO_CANDIDATE_KEY] += 1
            continue
        bucket = split_bucket(salt, key, buckets)
        records.append(_record(project, task_id, key, bucket, ledger, task_row))
    return tuple(records), dict(exclusions)


def _skipped(root: Path, problem: str) -> FleetLoad:
    return FleetLoad((), (RootStatus(root, runs_db_problem=problem),), ProducerEvidence())


def load_project(
    root: Path, window: Window, salt: str, buckets: int, key: KeyKind,
) -> FleetLoad:
    """One root's population records, read status, producer evidence and exclusions."""
    runs_path = runs_db(root)
    if not runs_path.is_file():
        return _skipped(root, 'runs.db absent')
    try:
        ledgers, events = _read_runs_db(runs_path, window)
    except UNREADABLE_STORE_ERRORS as exc:
        return _skipped(root, _store_problem('runs.db', exc))
    task_rows, tasks_db_problem = _read_task_rows(root)
    records, exclusions = _keyed_records(root.name, ledgers, task_rows, salt, buckets, key)
    return FleetLoad(
        records,
        (RootStatus(root, tasks_db_problem=tasks_db_problem),),
        _evidence(events, records),
        exclusions,
    )


def load_fleet(
    roots: Sequence[Path], window: Window, salt: str, buckets: int, key: KeyKind,
) -> FleetLoad:
    return FleetLoad.combined(load_project(root, window, salt, buckets, key) for root in roots)


MIN_ELIGIBLE_PER_ARM = 20  # seat N's prototype floor: below it an arm gets no null
_MDE_Z = NormalDist().inv_cdf(0.95) + NormalDist().inv_cdf(0.80)


def _rate(flag: bool) -> float:
    return 1.0 if flag else 0.0


def _curator_hint_simple(record: TaskRecord) -> float | None:
    hint = record.curator_complexity_hint
    return None if hint is None else _rate(hint == 'simple')


@dataclass(frozen=True)
class Metric:
    """One per-task metric; *value* is None when the task is not eligible for it."""

    name: str
    headline: bool
    value: Callable[[TaskRecord], float | None]


METRICS = (
    Metric('blocked_or_escalated', True, lambda record: _rate(record.blocked_or_escalated)),
    Metric('review_cycles_ge1', True, lambda record: _rate(record.max_review_cycles >= 1)),
    Metric('cost_usd', True, lambda record: record.cost_usd),
    Metric('workflow_verify_minutes', True, lambda record: record.workflow_verify_minutes),
    Metric('max_execute_iterations', False, lambda record: float(record.max_execute_iterations)),
    Metric('max_review_cycles', False, lambda record: float(record.max_review_cycles)),
    Metric('merge_verify_minutes', False, lambda record: record.merge_verify_minutes),
    Metric('reviewer_blocking_count', False, lambda record: record.reviewer_blocking_count),
    Metric('large_plan_rule_fired', False, lambda record: _rate(record.large_plan_rule_fired)),
    Metric('declared_simple', False, lambda record: _rate(record.complexity == 'simple')),
    Metric('curator_hint_simple', False, _curator_hint_simple),
)


@dataclass(frozen=True)
class NullBand:
    """The A/A null of one difference: percentiles of its magnitude, and the MDE."""

    p95: float
    p99: float
    mde: float


@dataclass(frozen=True)
class MetricRow:
    metric: str
    headline: bool
    base_n: int
    arm_n: int
    base_mean: float | None
    arm_mean: float | None
    diff: float | None
    null: NullBand | None


@dataclass(frozen=True)
class Comparison:
    """Arm *arm* against arm 0, one row per metric."""

    arm: int
    rows: tuple[MetricRow, ...]


_ArmSums = tuple[int, float]  # an arm's eligible task count and its value total
_Picker = Callable[[Sequence[Any]], tuple[Any, ...]]


def _picker(members: Sequence[int]) -> _Picker:
    """Select *members* from a column as a tuple, whatever their count."""
    if not members:
        return lambda column: ()
    if len(members) == 1:
        (only,) = members
        return lambda column: (column[only],)
    return itemgetter(*members)


@dataclass(frozen=True)
class _Column:
    """A metric's values over every task (None read as 0.0) and whether each task is eligible."""

    values: tuple[float, ...]
    eligible: tuple[int, ...]
    fully_eligible: bool

    @classmethod
    def of(cls, metric: Metric, records: Sequence[TaskRecord]) -> _Column:
        raw = [metric.value(record) for record in records]
        return cls(
            tuple(0.0 if value is None else value for value in raw),
            tuple(0 if value is None else 1 for value in raw),
            all(value is not None for value in raw),
        )

    def totals(self) -> _ArmSums:
        return sum(self.eligible), math.fsum(self.values)

    def arm_sums(self, members: Sequence[int]) -> _ArmSums:
        pick = _picker(members)
        eligible_n = len(members) if self.fully_eligible else sum(pick(self.eligible))
        return eligible_n, math.fsum(pick(self.values))


def _difference(base: _ArmSums, arm: _ArmSums) -> float | None:
    (base_n, base_total), (arm_n, arm_total) = base, arm
    if not (base_n and arm_n):
        return None
    return arm_total / arm_n - base_total / base_n


def _resampled_arm_sums(
    column: _Column, totals: _ArmSums, drawn_arms: Sequence[Sequence[int]],
) -> list[_ArmSums]:
    """Every arm's sums, the last arm taken as the complement of the drawn ones."""
    sums = [column.arm_sums(members) for members in drawn_arms]
    last_n = totals[0] - sum(n for n, _ in sums)
    last_total = totals[1] - math.fsum(total for _, total in sums)
    return [*sums, (last_n, last_total)]


def _null_diffs(
    columns: Sequence[_Column],
    sizes: Sequence[int],
    testable: Sequence[tuple[int, int]],
    resamples: int,
    rng: random.Random,
) -> dict[tuple[int, int], list[float]]:
    """Signed differences per (arm, metric index) under random relabelling of the pooled tasks.

    Each resample draws one random ordering of all but the last arm's worth of
    tasks; every metric reads that same draw, sliced into consecutive arms of
    the observed sizes, with the last arm as its complement. That is a
    permutation of the observed labels at half a shuffle's cost. A resample in
    which an arm has no eligible task is skipped for that metric.
    """
    diffs: dict[tuple[int, int], list[float]] = {pair: [] for pair in testable}
    if not testable:
        return diffs
    population = range(sum(sizes))
    bounds = list(itertools.accumulate(sizes[:-1], initial=0))
    indices = sorted({index for _, index in testable})
    totals = {index: columns[index].totals() for index in indices}
    for _ in range(resamples):
        order = rng.sample(population, bounds[-1])
        drawn_arms = [order[low:high] for low, high in itertools.pairwise(bounds)]
        sums = {
            index: _resampled_arm_sums(columns[index], totals[index], drawn_arms)
            for index in indices
        }
        for arm, index in testable:
            diff = _difference(sums[index][0], sums[index][arm])
            if diff is not None:
                diffs[(arm, index)].append(diff)
    return diffs


def _null_band(diffs: Sequence[float]) -> NullBand | None:
    if len(diffs) < 2:
        return None
    cuts = statistics.quantiles([abs(diff) for diff in diffs], n=100, method='inclusive')
    return NullBand(p95=cuts[94], p99=cuts[98], mde=_MDE_Z * statistics.stdev(diffs))


def _metric_row(
    metric: Metric,
    base: _ArmSums,
    arm: _ArmSums,
    null_diffs: Sequence[float] | None,
) -> MetricRow:
    (base_n, base_total), (arm_n, arm_total) = base, arm
    return MetricRow(
        metric=metric.name,
        headline=metric.headline,
        base_n=base_n,
        arm_n=arm_n,
        base_mean=base_total / base_n if base_n else None,
        arm_mean=arm_total / arm_n if arm_n else None,
        diff=_difference(base, arm),
        null=None if null_diffs is None else _null_band(null_diffs),
    )


def compare_arms(
    records: Sequence[TaskRecord],
    metrics: Sequence[Metric],
    buckets: int,
    resamples: int,
    rng: random.Random,
) -> tuple[Comparison, ...]:
    """Every arm k >= 1 against arm 0, each difference beside its permutation null.

    A metric gets a null only when both arms hold at least
    MIN_ELIGIBLE_PER_ARM eligible tasks.
    """
    columns = [_Column.of(metric, records) for metric in metrics]
    members = [[i for i, record in enumerate(records) if record.bucket == arm] for arm in range(buckets)]
    observed = [[column.arm_sums(arm) for arm in members] for column in columns]
    testable = [
        (arm, index)
        for arm in range(1, buckets)
        for index in range(len(metrics))
        if min(observed[index][0][0], observed[index][arm][0]) >= MIN_ELIGIBLE_PER_ARM
    ]
    null_diffs = _null_diffs(columns, [len(arm) for arm in members], testable, resamples, rng)
    return tuple(
        Comparison(
            arm,
            tuple(
                _metric_row(metric, observed[index][0], observed[index][arm], null_diffs.get((arm, index)))
                for index, metric in enumerate(metrics)
            ),
        )
        for arm in range(1, buckets)
    )


def _bucket_count(text: str) -> int:
    count = int(text)
    try:
        split_bucket('', '', count)
    except ValueError as exc:
        raise argparse.ArgumentTypeError(str(exc)) from exc
    return count


def _positive_int(text: str) -> int:
    value = int(text)
    if value < 1:
        raise argparse.ArgumentTypeError(f'must be >= 1, got {value}')
    return value


def _utc_instant(text: str) -> datetime:
    try:
        instant = datetime.fromisoformat(text)
    except ValueError as exc:
        raise argparse.ArgumentTypeError(f'{text!r} is not ISO-8601') from exc
    if instant.utcoffset() is None:
        raise argparse.ArgumentTypeError(
            f'{text!r} carries no UTC offset; write e.g. 2026-11-20T00:00:00+00:00'
        )
    return instant.astimezone(UTC)


def _choices_metavar(choices: type[Enum]) -> str:
    return '{' + ','.join(member.value for member in choices) + '}'


def _parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(
        description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter
    )
    parser.add_argument(
        '--roots',
        help=f'comma-separated project roots; default {KNOWN_PROJECT_ROOTS_ENV}, '
        f'then the installed {FUSED_MEMORY_UNIT} Environment',
    )
    parser.add_argument('--salt', required=True, help="the experiment's split salt")
    parser.add_argument('--buckets', type=_bucket_count, default=2)
    parser.add_argument('--days', type=_positive_int, default=30)
    parser.add_argument(
        '--until',
        type=_utc_instant,
        default=None,
        help='window end, ISO-8601 with a UTC offset (default: now)',
    )
    parser.add_argument(
        '--key', type=KeyKind, choices=list(KeyKind), default=KeyKind.TASK_ID,
        metavar=_choices_metavar(KeyKind),
    )
    parser.add_argument(
        '--subset', type=Subset, choices=list(Subset), default=Subset.ALL,
        metavar=_choices_metavar(Subset),
    )
    parser.add_argument('--aa', action='store_true', help='exit 1 on a skewed split')
    parser.add_argument('--resamples', type=_positive_int, default=DEFAULT_RESAMPLES)
    parser.add_argument('--seed', type=int, default=DEFAULT_SEED)
    parser.add_argument(
        '--expect-rule',
        action='append',
        default=[],
        help='a routing rule id that must appear in the window (repeatable)',
    )
    parser.add_argument('--json', action='store_true')
    return parser


@dataclass(frozen=True)
class Settings:
    """The command-line choices a report was computed under."""

    window: Window
    salt: str
    buckets: int
    key: KeyKind
    subset: Subset
    resamples: int
    seed: int


@dataclass(frozen=True)
class Report:
    settings: Settings
    source: RootSource
    roots: tuple[RootStatus, ...]
    evidence: ProducerEvidence
    exclusions: dict[str, int]
    tasks: tuple[TaskRecord, ...]
    comparisons: tuple[Comparison, ...]


def _listing(items: Sequence[str]) -> str:
    return ', '.join(items) or 'none'


def _roots_clause(source: RootSource, statuses: Sequence[RootStatus]) -> str:
    read = [status.project for status in statuses if status.read]
    skipped = [f'{status.project} ({status.runs_db_problem})' for status in statuses if not status.read]
    tasks_unread = [
        f'{status.project} ({status.tasks_db_problem})'
        for status in statuses if status.read and status.tasks_db_problem is not None
    ]
    clause = f'roots from {source.value}: read {_listing(read)}'
    if skipped:
        clause += f'; skipped {_listing(skipped)}'
    if tasks_unread:
        clause += f'; tasks.db unread {_listing(tasks_unread)}'
    return clause


def _rule_line(report: Report) -> str:
    settings = report.settings
    start, end = settings.window.sql_bounds()
    excluded = [f'{count} {reason}' for reason, count in report.exclusions.items()]
    return 'rule: ' + '; '.join([
        f'window [{start}, {end})',
        f'population = the {len(report.tasks)} tasks with a task_results row completed in the window',
        _roots_clause(report.source, report.roots),
        f'key {settings.key.value}',
        f'split salt {settings.salt!r} buckets {settings.buckets}',
        f'subset {settings.subset.value}',
        f'excluded {_listing(excluded)}',
        f'null = {settings.resamples} permutations of the observed arm labels (seed '
        f'{settings.seed}), arm sizes kept, one permutation shared by every metric',
        'diff = mean(arm k) - mean(arm 0) in metric units over eligible tasks',
        'null p95/p99 = percentiles of |diff|',
        'mde = (z0.95 + z0.80) x sd(diff)',
        f'no null below {MIN_ELIGIBLE_PER_ARM} eligible tasks in an arm',
    ])


def _producers_line(evidence: ProducerEvidence) -> str:
    rule_ids = [f'{rule_id}={count}' for rule_id, count in evidence.rule_id_counts.items()]
    return 'producers: ' + '; '.join([
        f'routing_decision rule ids {_listing(rule_ids)}',
        f'implementer decisions carrying plan shape {evidence.plan_shape_decisions}',
        f'tasks carrying {CURATOR_HINT_KEY} {evidence.hinted_tasks}',
    ])


def _arm_sizes_line(report: Report) -> str:
    sizes = Counter(record.bucket for record in report.tasks)
    return '; '.join(f'arm {arm}: {sizes[arm]} tasks' for arm in range(report.settings.buckets))


_TABLE_COLUMNS = '{marker} {metric:<26}{n0:>6}{n1:>6}{mean0:>10}{mean1:>10}{diff:>10}{p95:>10}{p99:>10}{mde:>10}'


def _number(value: float | None, sign: str = '') -> str:
    return '-' if value is None else f'{value:{sign}.3f}'


def _table_row(row: MetricRow) -> str:
    null = row.null
    return _TABLE_COLUMNS.format(
        marker='*' if row.headline else '-',
        metric=row.metric,
        n0=row.base_n,
        n1=row.arm_n,
        mean0=_number(row.base_mean),
        mean1=_number(row.arm_mean),
        diff=_number(row.diff, '+'),
        p95=_number(null and null.p95),
        p99=_number(null and null.p99),
        mde=_number(null and null.mde),
    )


def _comparison_table(comparison: Comparison) -> list[str]:
    arm = comparison.arm
    header = _TABLE_COLUMNS.format(
        marker=' ', metric='metric', n0='n0', n1=f'n{arm}', mean0='mean0', mean1=f'mean{arm}',
        diff='diff', p95='null_p95', p99='null_p99', mde='mde',
    )
    return [
        '',
        f'arm {arm} vs arm 0 (* headline metric, - secondary)',
        header,
        *(_table_row(row) for row in comparison.rows),
    ]


def render_text(report: Report) -> str:
    lines = [
        _rule_line(report),
        _producers_line(report.evidence),
        _arm_sizes_line(report),
        *(line for comparison in report.comparisons for line in _comparison_table(comparison)),
    ]
    return '\n'.join(lines) + '\n'


def _json_value(value: object) -> object:
    if isinstance(value, (set, frozenset)):
        return sorted(value)
    if isinstance(value, Enum):
        return value.value
    if isinstance(value, datetime):
        return value.isoformat()
    if isinstance(value, Path):
        return str(value)
    raise TypeError(f'{type(value).__name__} is not JSON serialisable')


def render_json(report: Report) -> str:
    return json.dumps(asdict(report), default=_json_value, indent=2) + '\n'


def _warn(message: str) -> None:
    print(f'routing_experiment_readout: {message}', file=sys.stderr)


def _warn_unread_stores(statuses: Iterable[RootStatus]) -> None:
    for status in statuses:
        if status.runs_db_problem is not None:
            _warn(f'skipped {status.root}: {status.runs_db_problem}')
        elif status.tasks_db_problem is not None:
            _warn(
                f'{status.root}: {status.tasks_db_problem}; complexity, curator hint and '
                'candidate_key read as absent'
            )


def main(
    argv: Sequence[str] | None = None,
    *,
    environ: Mapping[str, str] = os.environ,
    unit_environment: Callable[[], str | None] = read_fused_memory_unit_environment,
) -> int:
    args = _parser().parse_args(argv)
    selection = resolve_roots(args.roots, environ, unit_environment)
    if not selection.roots:
        _warn(
            f'no project root resolved: pass --roots, set {KNOWN_PROJECT_ROOTS_ENV}, or '
            f'install {FUSED_MEMORY_UNIT} with it in its Environment='
        )
        return EXIT_NOTHING_READ
    settings = Settings(
        Window.trailing(args.until or datetime.now(UTC), args.days),
        args.salt,
        args.buckets,
        args.key,
        args.subset,
        args.resamples,
        args.seed,
    )
    load = load_fleet(selection.roots, settings.window, args.salt, args.buckets, args.key)
    _warn_unread_stores(load.roots)
    if not any(status.read for status in load.roots):
        _warn(
            f'no runs.db was readable under the {len(load.roots)} root(s) from '
            f'{selection.source.value}'
        )
        return EXIT_NOTHING_READ
    comparisons = compare_arms(
        load.records, METRICS, args.buckets, args.resamples, random.Random(args.seed)
    )
    report = Report(
        settings,
        selection.source,
        load.roots,
        load.evidence,
        load.exclusions,
        load.records,
        comparisons,
    )
    sys.stdout.write(render_json(report) if args.json else render_text(report))
    return EXIT_OK


if __name__ == '__main__':
    sys.exit(main())
