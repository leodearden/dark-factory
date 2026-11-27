"""Landed-per-dollar by arm over a routing experiment's first-dispatch cohort.

The library behind ``routing_experiment_readout.py --throughput`` and the
weekly look, ``effort_experiment_look.py``. It parses no command line:
:class:`ThroughputOptions` carries every input.

The cohort is every task whose FIRST routing_decision in one of the options'
roles falls in [since, cohort_end), where cohort_end is since + cohort_days
capped at read_at (read_at when cohort_days is None). A task's arm is
``split_bucket(salt, task_id, buckets)``: treated is the treated bucket,
control every other bucket, by intent to treat. A task has landed when the
landing datum (shared.quality_series.landings.recorded_landings) records it
before read_at, and its cost is every invocation of any role started before
read_at. R is (landed_t / cost_t) / (landed_c / cost_c).

R's interval resamples tasks with replacement within each arm, as a percentile
interval at the options' level. Its A/A null permutes the observed arm labels
at the observed sizes and reports the p95 and p99 of max(R, 1/R). One
``random.Random(seed)`` draws the bootstrap first, then the null.

Reads runs.db read-only; never imports the readout or the look.
"""
from __future__ import annotations

import math
import random
from collections import defaultdict
from collections.abc import Iterable, Mapping, Sequence
from contextlib import closing
from dataclasses import dataclass
from datetime import datetime, timedelta
from enum import Enum
from pathlib import Path
from sqlite3 import Connection
from types import MappingProxyType

import _checkout as _checkout
from _task_db_scan import UNREADABLE_STORE_ERRORS
from shared.project_layout import runs_db
from shared.quality_series.landings import UnrecognisedTrainEvent, parse_utc, recorded_landings
from shared.routing_split import split_bucket
from shared.sqlite_readonly import connect_readonly
from split_resampling import bootstrap_arms, null_cuts, percentile_interval, permuted_arms

__all__ = [
    'Arm',
    'ArmTotals',
    'BLOCKED_OUTCOMES',
    'CohortRecord',
    'IntervalSide',
    'ThroughputOptions',
    'ThroughputReport',
    'ThroughputUnavailable',
    'Unavailable',
    'resolve_since',
    'side_of',
    'throughput_report',
]

BLOCKED_OUTCOMES = frozenset({'blocked', 'escalated'})

_ROUTING_DECISION = 'routing_decision'


class Arm(Enum):
    TREATED = 'treated'
    CONTROL = 'control'


class Unavailable(Enum):
    """Why no report could be made."""

    NO_TREATED_DISPATCH = 'no routing_decision carries the rule prefix'
    STORE_UNREADABLE = 'a runs.db or its landing datum is unreadable'
    HISTORY_UNREADABLE = "main's git history is unreadable"


class IntervalSide(Enum):
    """Where R's interval lies against 1."""

    ABOVE_1 = 'wholly above 1'
    BELOW_1 = 'wholly below 1'
    INCLUDES_1 = 'includes 1'
    UNDEFINED = 'undefined'


def side_of(interval: tuple[float, float] | None) -> IntervalSide:
    if interval is None:
        return IntervalSide.UNDEFINED
    low, high = interval
    if low > 1:
        return IntervalSide.ABOVE_1
    if high < 1:
        return IntervalSide.BELOW_1
    return IntervalSide.INCLUDES_1


class ThroughputUnavailable(Exception):
    """No report: *kind* says why, *detail* names the root, prefix or store."""

    def __init__(self, kind: Unavailable, detail: str) -> None:
        self.kind = kind
        self.detail = detail
        super().__init__(f'{kind.value}: {detail}')


@dataclass(frozen=True)
class ThroughputOptions:
    """Every input of one throughput report.

    *roles* restrict which routing_decision rows make a cohort's first
    dispatch. *rule_prefix* marks a treated routing rule; with *since* None it
    also dates the experiment's start (:func:`resolve_since`). *treated_effort*
    and *control_effort* are each arm's expected effort. *read_at* bounds every
    read: nothing at or after it counts. *seed* and *resamples* drive the
    interval and the A/A null; *interval_level* is the interval's two-sided level.
    """

    salt: str
    buckets: int
    treated_bucket: int
    roles: tuple[str, ...]
    rule_prefix: str
    treated_effort: str
    control_effort: str
    since: datetime | None
    cohort_days: int | None
    read_at: datetime
    roots: tuple[Path, ...]
    interval_level: float
    seed: int
    resamples: int

    def __post_init__(self) -> None:
        split_bucket(self.salt, '', self.buckets)
        if not 0 <= self.treated_bucket < self.buckets:
            raise ValueError(
                f'treated_bucket lies in range(buckets={self.buckets}), got {self.treated_bucket!r}'
            )
        if not 0 < self.interval_level < 1:
            raise ValueError(
                f'interval_level lies strictly between 0 and 1, got {self.interval_level!r}'
            )
        if self.resamples < 1:
            raise ValueError(f'resamples is at least 1, got {self.resamples!r}')
        if not self.roles:
            raise ValueError(f'roles names at least one role, got {self.roles!r}')
        if self.cohort_days is not None and self.cohort_days < 1:
            raise ValueError(f'cohort_days is at least 1 or None, got {self.cohort_days!r}')
        for name, at in (('since', self.since), ('read_at', self.read_at)):
            if at is not None and at.utcoffset() is None:
                raise ValueError(f'{name} carries its UTC offset, got {at!r}')

    def arm_of(self, task_id: str) -> Arm:
        treated = split_bucket(self.salt, task_id, self.buckets) == self.treated_bucket
        return Arm.TREATED if treated else Arm.CONTROL

    def cohort_end(self, since: datetime) -> datetime:
        if self.cohort_days is None:
            return self.read_at
        return min(since + timedelta(days=self.cohort_days), self.read_at)


@dataclass(frozen=True)
class CohortRecord:
    """One cohort task: its arm, first dispatch, landing (None when none before read_at) and cost."""

    project: str
    task_id: str
    arm: Arm
    first_dispatch_at: datetime
    landed_at: datetime | None
    merge_sha: str | None
    cost_usd: float
    cost_by_role: Mapping[str, float]


@dataclass(frozen=True)
class ArmTotals:
    n: int
    landed: int
    cost_usd: float


@dataclass(frozen=True)
class ThroughputReport:
    """The cohort's records, each arm's totals, R, its interval and its A/A null.

    ratio and interval are None when undefined; null is the (p95, p99) of
    max(R, 1/R) under relabelling, None when fewer than 2 relabellings define R.
    """

    options: ThroughputOptions
    since: datetime
    cohort_end: datetime
    records: tuple[CohortRecord, ...]
    totals: Mapping[Arm, ArmTotals]
    ratio: float | None
    interval: tuple[float, float] | None
    interval_side: IntervalSide
    null: tuple[float, float] | None


def _placeholders(values: Sequence[object]) -> str:
    return ', '.join('?' for _ in values)


def _unreadable(root: Path, exc: Exception) -> ThroughputUnavailable:
    return ThroughputUnavailable(Unavailable.STORE_UNREADABLE, f'{root}: runs.db unreadable: {exc}')


def _open_runs_db(root: Path) -> Connection:
    path = runs_db(root)
    if not path.is_file():
        raise ThroughputUnavailable(Unavailable.STORE_UNREADABLE, f'{root}: no runs.db at {path}')
    try:
        return connect_readonly(path)
    except UNREADABLE_STORE_ERRORS as exc:
        raise _unreadable(root, exc) from exc


_PREFIXED_DISPATCH_SQL = f"""
SELECT timestamp FROM events
 WHERE event_type = '{_ROUTING_DECISION}'
   AND substr(json_extract(data, '$.rule_id'), 1, length(?)) = ?
"""


def resolve_since(options: ThroughputOptions) -> datetime:
    """options.since, else the first routing_decision of any role whose rule_id starts with the prefix."""
    if options.since is not None:
        return options.since
    found: list[datetime] = []
    for root in options.roots:
        with closing(_open_runs_db(root)) as conn:
            try:
                rows = conn.execute(
                    _PREFIXED_DISPATCH_SQL, (options.rule_prefix, options.rule_prefix)
                ).fetchall()
            except UNREADABLE_STORE_ERRORS as exc:
                raise _unreadable(root, exc) from exc
        found.extend(parse_utc(timestamp) for (timestamp,) in rows)
    if not found:
        raise ThroughputUnavailable(
            Unavailable.NO_TREATED_DISPATCH,
            f'no routing_decision carries the rule prefix {options.rule_prefix!r} in '
            f'{", ".join(str(root) for root in options.roots)}',
        )
    return min(found)


@dataclass(frozen=True)
class _Ledger:
    """One root's rows for its cohort tasks, every instant aware and before read_at."""

    first_dispatch: Mapping[str, datetime]
    costs: Mapping[str, Mapping[str, float]]
    landings: Mapping[str, tuple[datetime, str]]


def _first_dispatches(
    conn: Connection, options: ThroughputOptions, since: datetime, cohort_end: datetime,
) -> dict[str, datetime]:
    earliest: dict[str, datetime] = {}
    rows = conn.execute(
        f"SELECT task_id, timestamp FROM events WHERE event_type = '{_ROUTING_DECISION}' "
        f'AND task_id IS NOT NULL AND role IN ({_placeholders(options.roles)})',
        options.roles,
    )
    for task_id, timestamp in rows:
        at = parse_utc(timestamp)
        if task_id not in earliest or at < earliest[task_id]:
            earliest[str(task_id)] = at
    return {task_id: at for task_id, at in earliest.items() if since <= at < cohort_end}


def _costs(
    conn: Connection, cohort: Iterable[str], read_at: datetime,
) -> dict[str, dict[str, float]]:
    wanted = set(cohort)
    spent: dict[str, dict[str, list[float]]] = defaultdict(lambda: defaultdict(list))
    rows = conn.execute(
        'SELECT task_id, role, cost_usd, started_at FROM invocations WHERE task_id IS NOT NULL'
    )
    for task_id, role, cost_usd, started_at in rows:
        if str(task_id) in wanted and parse_utc(started_at) < read_at:
            spent[str(task_id)][role].append(cost_usd)
    return {
        task_id: {role: math.fsum(costs) for role, costs in sorted(by_role.items())}
        for task_id, by_role in spent.items()
    }


def _landings(conn: Connection, cohort: Iterable[str], read_at: datetime) -> dict[str, tuple[datetime, str]]:
    wanted = set(cohort)
    first: dict[str, tuple[datetime, str]] = {}
    for landing in recorded_landings(conn):
        if landing.task_id in wanted and landing.landed_at < read_at:
            first.setdefault(landing.task_id, (landing.landed_at, landing.merge_sha))
    return first


def _read_ledger(
    root: Path, options: ThroughputOptions, since: datetime, cohort_end: datetime,
) -> _Ledger:
    with closing(_open_runs_db(root)) as conn:
        try:
            first_dispatch = _first_dispatches(conn, options, since, cohort_end)
            return _Ledger(
                first_dispatch=first_dispatch,
                costs=_costs(conn, first_dispatch, options.read_at),
                landings=_landings(conn, first_dispatch, options.read_at),
            )
        except UnrecognisedTrainEvent as exc:
            raise _unreadable(root, exc) from exc
        except UNREADABLE_STORE_ERRORS as exc:
            raise _unreadable(root, exc) from exc


def _records(root: Path, ledger: _Ledger, options: ThroughputOptions) -> list[CohortRecord]:
    records: list[CohortRecord] = []
    for task_id, first_dispatch_at in ledger.first_dispatch.items():
        cost_by_role = ledger.costs.get(task_id, {})
        landed_at, merge_sha = ledger.landings.get(task_id, (None, None))
        records.append(CohortRecord(
            project=root.name,
            task_id=task_id,
            arm=options.arm_of(task_id),
            first_dispatch_at=first_dispatch_at,
            landed_at=landed_at,
            merge_sha=merge_sha,
            cost_usd=math.fsum(cost_by_role.values()),
            cost_by_role=MappingProxyType(dict(cost_by_role)),
        ))
    return records


def _ratio(landed_t: int, cost_t: float, landed_c: int, cost_c: float) -> float | None:
    """R = (landed_t / cost_t) / (landed_c / cost_c); None when a cost or the control landed count is 0."""
    if not (cost_t and cost_c and landed_c):
        return None
    return (landed_t / cost_t) / (landed_c / cost_c)


def _arm_totals(records: Sequence[CohortRecord], arm: Arm) -> ArmTotals:
    members = [record for record in records if record.arm is arm]
    return ArmTotals(
        n=len(members),
        landed=sum(1 for record in members if record.landed_at is not None),
        cost_usd=math.fsum(record.cost_usd for record in members),
    )


@dataclass(frozen=True)
class _Columns:
    """Each record's landed flag and cost, by record index."""

    landed: tuple[int, ...]
    cost: tuple[float, ...]

    @classmethod
    def of(cls, records: Sequence[CohortRecord]) -> _Columns:
        return cls(
            tuple(int(record.landed_at is not None) for record in records),
            tuple(record.cost_usd for record in records),
        )

    def sums(self, members: Iterable[int]) -> tuple[int, float]:
        picked = list(members)
        return sum(self.landed[i] for i in picked), math.fsum(self.cost[i] for i in picked)


def _bootstrap_interval(
    columns: _Columns, treated: Sequence[int], control: Sequence[int],
    options: ThroughputOptions, rng: random.Random,
) -> tuple[float, float] | None:
    draws = [
        _ratio(*columns.sums(drawn_treated), *columns.sums(drawn_control))
        for drawn_treated, drawn_control in bootstrap_arms([treated, control], options.resamples, rng)
    ]
    defined = [draw for draw in draws if draw is not None]
    if len(defined) < 2:
        return None
    return percentile_interval(defined, options.interval_level)


def _magnitude(ratio: float | None) -> float | None:
    if not ratio:
        return None
    return max(ratio, 1 / ratio)


def _aa_null(
    columns: _Columns, n_treated: int, n_control: int, resamples: int, rng: random.Random,
) -> tuple[float, float] | None:
    total_landed, total_cost = columns.sums(range(n_treated + n_control))
    magnitudes: list[float] = []
    for (drawn_control,) in permuted_arms([n_control, n_treated], resamples, rng):
        landed_c, cost_c = columns.sums(drawn_control)
        magnitude = _magnitude(_ratio(total_landed - landed_c, total_cost - cost_c, landed_c, cost_c))
        if magnitude is not None:
            magnitudes.append(magnitude)
    if len(magnitudes) < 2:
        return None
    return null_cuts(magnitudes)


def throughput_report(options: ThroughputOptions, since: datetime) -> ThroughputReport:
    """The cohort dispatched from *since* (see :func:`resolve_since`), read across options.roots.

    Raises ThroughputUnavailable(STORE_UNREADABLE) naming the root whose runs.db
    or landing datum cannot be read.
    """
    cohort_end = options.cohort_end(since)
    records = sorted(
        (
            record
            for root in options.roots
            for record in _records(root, _read_ledger(root, options, since, cohort_end), options)
        ),
        key=lambda record: (record.project, record.first_dispatch_at, record.task_id),
    )
    totals = {arm: _arm_totals(records, arm) for arm in Arm}
    treated, control = totals[Arm.TREATED], totals[Arm.CONTROL]
    columns = _Columns.of(records)
    treated_members = [i for i, record in enumerate(records) if record.arm is Arm.TREATED]
    control_members = [i for i, record in enumerate(records) if record.arm is Arm.CONTROL]
    rng = random.Random(options.seed)
    interval = _bootstrap_interval(columns, treated_members, control_members, options, rng)
    return ThroughputReport(
        options=options,
        since=since,
        cohort_end=cohort_end,
        records=tuple(records),
        totals=MappingProxyType(totals),
        ratio=_ratio(treated.landed, treated.cost_usd, control.landed, control.cost_usd),
        interval=interval,
        interval_side=side_of(interval),
        null=_aa_null(columns, treated.n, control.n, options.resamples, rng),
    )
