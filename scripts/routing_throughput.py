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

Secondary measures, per arm: verify attempts (task_results.verify_attempts
plus the task's own verify_failed merge_attempt rows), review cycles and
failures after merge, each per landing; the share of tasks with a blocked or
escalated outcome; and cost per task by role. A failure after merge is an
integration_skew merge_attempt implicating a commit the merge brought
(``git rev-list sha^1..sha``), or a main commit saying ``This reverts commit
<sha>``, inside [landed_at, min(landed_at + AFTER_MERGE_WINDOW, read_at)).

Compliance reads every cohort task's routing_decision in --roles inside
[since, read_at): a row whose effort is not its arm's expected effort is a
deviation, a treated row whose rule_id lacks the prefix is treated without the
rule, and a treated row at the control effort is a COMPLIANCE FAILURE.

R's interval resamples tasks with replacement within each arm, as a percentile
interval at the options' level. Its A/A null permutes the observed arm labels
at the observed sizes and reports the p95 and p99 of max(R, 1/R). One
``random.Random(seed)`` draws the bootstrap first, then the null.

:func:`render_text` prints the report as three ruled tables (R, the secondary
measures, compliance), and :func:`summary_record` is its one-line JSON summary.

Reads runs.db read-only; never imports the readout or the look.
"""
from __future__ import annotations

import math
import os
import random
import re
import subprocess
from collections import defaultdict
from collections.abc import Iterable, Mapping, Sequence
from contextlib import closing
from dataclasses import dataclass
from datetime import UTC, datetime, timedelta
from enum import Enum
from pathlib import Path
from sqlite3 import Connection
from types import MappingProxyType

import _checkout as _checkout
from _task_db_scan import UNREADABLE_STORE_ERRORS, decode_metadata
from git_env import scrubbed_git_env
from shared.project_layout import runs_db
from shared.quality_series.landings import UnrecognisedTrainEvent, parse_utc, recorded_landings
from shared.routing_split import split_bucket
from shared.sqlite_readonly import connect_readonly
from split_resampling import bootstrap_arms, null_cuts, percentile_interval, permuted_arms

__all__ = [
    'AFTER_MERGE_WINDOW',
    'Arm',
    'ArmSecondary',
    'ArmTotals',
    'BLOCKED_OUTCOMES',
    'CohortRecord',
    'Compliance',
    'ComplianceFailure',
    'IntervalSide',
    'ThroughputOptions',
    'ThroughputReport',
    'ThroughputUnavailable',
    'RoleCompliance',
    'Unavailable',
    'render_text',
    'resolve_since',
    'side_of',
    'summary_record',
    'throughput_report',
]

BLOCKED_OUTCOMES = frozenset({'blocked', 'escalated'})
AFTER_MERGE_WINDOW = timedelta(days=7)

_ROUTING_DECISION = 'routing_decision'
_MERGE_ATTEMPT = 'merge_attempt'
_VERIFY_FAILED = 'verify_failed'
_INTEGRATION_SKEW = 'integration_skew'
_GIT_TIMEOUT_S = 60
_REVERT_MARKER = 'This reverts commit'
_REVERTED_SHA = re.compile(rf'{_REVERT_MARKER} ([0-9a-f]{{40}})')


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
    """One cohort task: its arm, first dispatch, landing (None when none before read_at), cost and secondaries."""

    project: str
    task_id: str
    arm: Arm
    first_dispatch_at: datetime
    landed_at: datetime | None
    merge_sha: str | None
    cost_usd: float
    cost_by_role: Mapping[str, float]
    verify_attempts: int
    review_cycles: int
    failures_after_merge: int
    blocked_or_escalated: bool


@dataclass(frozen=True)
class ArmSecondary:
    """One arm's secondary measures; a per-landing figure or a share is None when its denominator is 0."""

    n: int
    verify_attempts_per_landing: float | None
    review_cycles_per_landing: float | None
    failures_after_merge_per_landing: float | None
    blocked_or_escalated_share: float | None
    cost_per_task_by_role: Mapping[str, float]


@dataclass(frozen=True)
class ArmTotals:
    n: int
    landed: int
    cost_usd: float


@dataclass(frozen=True)
class RoleCompliance:
    dispatches: int
    effort_deviations: int
    treated_without_rule: int


@dataclass(frozen=True)
class ComplianceFailure:
    """A treated task dispatched at the control effort."""

    project: str
    task_id: str
    role: str
    at: datetime
    effort: str


@dataclass(frozen=True)
class Compliance:
    """Per-role counts, keyed by every role in options.roles, and each failure oldest first."""

    roles: Mapping[str, RoleCompliance]
    failures: tuple[ComplianceFailure, ...]


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
    secondary: Mapping[Arm, ArmSecondary]
    compliance: Compliance


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
class _Result:
    outcome: str
    verify_attempts: int
    review_cycles: int


@dataclass(frozen=True)
class _Skew:
    at: datetime
    implicated: frozenset[str]


@dataclass(frozen=True)
class _Dispatch:
    task_id: str
    at: datetime
    role: str
    data: Mapping[str, object]


@dataclass(frozen=True)
class _Ledger:
    """One root's rows for its cohort tasks, every instant aware and before read_at.

    *dispatches* are the cohort's routing_decisions in the options' roles from
    since; *skews* are every task's integration_skew merge_attempts that
    implicate a commit.
    """

    first_dispatch: Mapping[str, datetime]
    dispatches: tuple[_Dispatch, ...]
    costs: Mapping[str, Mapping[str, float]]
    landings: Mapping[str, tuple[datetime, str]]
    results: Mapping[str, tuple[_Result, ...]]
    own_verify_failures: Mapping[str, int]
    skews: tuple[_Skew, ...]


def _role_dispatches(conn: Connection, roles: Sequence[str]) -> list[tuple[str, datetime, str, object]]:
    rows = conn.execute(
        f"SELECT task_id, timestamp, role, data FROM events WHERE event_type = '{_ROUTING_DECISION}' "
        f'AND task_id IS NOT NULL AND role IN ({_placeholders(roles)})',
        tuple(roles),
    )
    return [(str(task_id), parse_utc(timestamp), role, raw) for task_id, timestamp, role, raw in rows]


def _cohort_dispatches(
    conn: Connection, options: ThroughputOptions, since: datetime, cohort_end: datetime,
) -> tuple[dict[str, datetime], tuple[_Dispatch, ...]]:
    """Each cohort task's first dispatch, and every cohort dispatch in [since, read_at)."""
    rows = _role_dispatches(conn, options.roles)
    earliest: dict[str, datetime] = {}
    for task_id, at, _, _ in rows:
        if task_id not in earliest or at < earliest[task_id]:
            earliest[task_id] = at
    cohort = {task_id: at for task_id, at in earliest.items() if since <= at < cohort_end}
    dispatches = tuple(
        _Dispatch(task_id, at, role, decode_metadata(raw))
        for task_id, at, role, raw in rows
        if task_id in cohort and since <= at < options.read_at
    )
    return cohort, dispatches


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


def _results(
    conn: Connection, cohort: Iterable[str], read_at: datetime,
) -> dict[str, tuple[_Result, ...]]:
    wanted = set(cohort)
    found: dict[str, list[_Result]] = defaultdict(list)
    rows = conn.execute(
        'SELECT task_id, outcome, verify_attempts, review_cycles, completed_at FROM task_results '
        'WHERE completed_at IS NOT NULL'
    )
    for task_id, outcome, verify_attempts, review_cycles, completed_at in rows:
        if str(task_id) in wanted and parse_utc(completed_at) < read_at:
            found[str(task_id)].append(_Result(outcome, verify_attempts or 0, review_cycles or 0))
    return {task_id: tuple(results) for task_id, results in found.items()}


def _merge_attempts(
    conn: Connection, cohort: Iterable[str], read_at: datetime,
) -> tuple[dict[str, int], tuple[_Skew, ...]]:
    """Each cohort task's own verify_failed attempts, and every task's skews implicating a commit."""
    wanted = set(cohort)
    own_failures: dict[str, int] = defaultdict(int)
    skews: list[_Skew] = []
    rows = conn.execute(
        f"SELECT timestamp, task_id, data FROM events WHERE event_type = '{_MERGE_ATTEMPT}'"
    )
    for timestamp, task_id, raw in rows:
        at = parse_utc(timestamp)
        if at >= read_at:
            continue
        data = decode_metadata(raw)
        if str(task_id) in wanted and data.get('outcome') == _VERIFY_FAILED:
            own_failures[str(task_id)] += 1
        implicated = data.get('implicated_commits')
        if data.get('disposition') == _INTEGRATION_SKEW and isinstance(implicated, list) and implicated:
            skews.append(_Skew(at, frozenset(sha for sha in implicated if isinstance(sha, str))))
    return dict(own_failures), tuple(skews)


def _read_ledger(
    root: Path, options: ThroughputOptions, since: datetime, cohort_end: datetime,
) -> _Ledger:
    with closing(_open_runs_db(root)) as conn:
        try:
            first_dispatch, dispatches = _cohort_dispatches(conn, options, since, cohort_end)
            own_verify_failures, skews = _merge_attempts(conn, first_dispatch, options.read_at)
            return _Ledger(
                first_dispatch=first_dispatch,
                dispatches=dispatches,
                costs=_costs(conn, first_dispatch, options.read_at),
                landings=_landings(conn, first_dispatch, options.read_at),
                results=_results(conn, first_dispatch, options.read_at),
                own_verify_failures=own_verify_failures,
                skews=skews,
            )
        except UnrecognisedTrainEvent as exc:
            raise _unreadable(root, exc) from exc
        except UNREADABLE_STORE_ERRORS as exc:
            raise _unreadable(root, exc) from exc


@dataclass(frozen=True)
class _Revert:
    committed_at: datetime
    reverted: frozenset[str]


def _git(root: Path, args: Sequence[str]) -> str:
    argv = ['git', '-C', str(root), *args]
    try:
        completed = subprocess.run(
            argv, capture_output=True, text=True, timeout=_GIT_TIMEOUT_S,
            env=scrubbed_git_env(os.environ),
        )
    except (OSError, subprocess.TimeoutExpired) as exc:
        raise ThroughputUnavailable(Unavailable.HISTORY_UNREADABLE, f'{root}: {exc}') from exc
    if completed.returncode != 0:
        raise ThroughputUnavailable(
            Unavailable.HISTORY_UNREADABLE,
            f'{root}: {" ".join(argv[3:])} exited {completed.returncode}: {completed.stderr.strip()}',
        )
    return completed.stdout


def _parse_revert(record: str) -> _Revert:
    _, committed, body = record.split('\x1f', 2)
    return _Revert(
        datetime.fromtimestamp(int(committed), UTC), frozenset(_REVERTED_SHA.findall(body)),
    )


@dataclass(frozen=True)
class MainHistory:
    """main's revert commits in one root, read once; a merge's own commits are read on demand."""

    root: Path
    reverts: tuple[_Revert, ...]

    @classmethod
    def read(cls, root: Path) -> MainHistory:
        stdout = _git(root, [
            'log', 'main', '-F', f'--grep={_REVERT_MARKER}', '--format=%H%x1f%ct%x1f%B%x1e',
        ])
        records = (record.strip() for record in stdout.split('\x1e'))
        return cls(root, tuple(_parse_revert(record) for record in records if record))

    def revert_count(self, merge_sha: str, after: datetime, until: datetime) -> int:
        return sum(
            1 for revert in self.reverts
            if after <= revert.committed_at < until and merge_sha in revert.reverted
        )

    def commits_of_merge(self, merge_sha: str) -> frozenset[str]:
        return frozenset(_git(self.root, ['rev-list', f'{merge_sha}^1..{merge_sha}']).split())


def _failures_after_merge(
    landed_at: datetime, merge_sha: str, ledger: _Ledger, history: MainHistory, read_at: datetime,
) -> int:
    until = min(landed_at + AFTER_MERGE_WINDOW, read_at)
    skews = [skew for skew in ledger.skews if landed_at <= skew.at < until]
    skew_count = 0
    if skews:
        brought = history.commits_of_merge(merge_sha)
        skew_count = sum(1 for skew in skews if skew.implicated & brought)
    return skew_count + history.revert_count(merge_sha, landed_at, until)


def _records(root: Path, ledger: _Ledger, options: ThroughputOptions) -> list[CohortRecord]:
    history = MainHistory.read(root) if ledger.landings else None
    records: list[CohortRecord] = []
    for task_id, first_dispatch_at in ledger.first_dispatch.items():
        cost_by_role = ledger.costs.get(task_id, {})
        landing = ledger.landings.get(task_id)
        results = ledger.results.get(task_id, ())
        failures = 0
        if landing is not None and history is not None:
            failures = _failures_after_merge(*landing, ledger, history, options.read_at)
        records.append(CohortRecord(
            project=root.name,
            task_id=task_id,
            arm=options.arm_of(task_id),
            first_dispatch_at=first_dispatch_at,
            landed_at=None if landing is None else landing[0],
            merge_sha=None if landing is None else landing[1],
            cost_usd=math.fsum(cost_by_role.values()),
            cost_by_role=MappingProxyType(dict(cost_by_role)),
            verify_attempts=sum(result.verify_attempts for result in results)
            + ledger.own_verify_failures.get(task_id, 0),
            review_cycles=sum(result.review_cycles for result in results),
            failures_after_merge=failures,
            blocked_or_escalated=any(result.outcome in BLOCKED_OUTCOMES for result in results),
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


def _expected_effort(arm: Arm, options: ThroughputOptions) -> str:
    return options.treated_effort if arm is Arm.TREATED else options.control_effort


@dataclass(frozen=True)
class _DispatchCheck:
    role: str
    deviation: bool
    without_rule: bool
    failure: ComplianceFailure | None


def _dispatch_checks(root: Path, ledger: _Ledger, options: ThroughputOptions) -> list[_DispatchCheck]:
    checks: list[_DispatchCheck] = []
    for dispatch in ledger.dispatches:
        arm = options.arm_of(dispatch.task_id)
        effort = dispatch.data.get('effort')
        rule_id = dispatch.data.get('rule_id')
        treated = arm is Arm.TREATED
        checks.append(_DispatchCheck(
            role=dispatch.role,
            deviation=effort != _expected_effort(arm, options),
            without_rule=treated and not (
                isinstance(rule_id, str) and rule_id.startswith(options.rule_prefix)
            ),
            failure=ComplianceFailure(
                root.name, dispatch.task_id, dispatch.role, dispatch.at, options.control_effort,
            ) if treated and effort == options.control_effort else None,
        ))
    return checks


def _compliance(checks: Sequence[_DispatchCheck], roles: Sequence[str]) -> Compliance:
    by_role = {
        role: RoleCompliance(
            dispatches=sum(1 for check in checks if check.role == role),
            effort_deviations=sum(1 for check in checks if check.role == role and check.deviation),
            treated_without_rule=sum(1 for check in checks if check.role == role and check.without_rule),
        )
        for role in roles
    }
    failures = sorted(
        (check.failure for check in checks if check.failure is not None),
        key=lambda failure: (failure.at, failure.project, failure.task_id),
    )
    return Compliance(MappingProxyType(by_role), tuple(failures))


def _per(total: float, denominator: int) -> float | None:
    return total / denominator if denominator else None


def _arm_secondary(records: Sequence[CohortRecord], arm: Arm) -> ArmSecondary:
    members = [record for record in records if record.arm is arm]
    landed = sum(1 for record in members if record.landed_at is not None)
    by_role: dict[str, list[float]] = defaultdict(list)
    for record in members:
        for role, cost in record.cost_by_role.items():
            by_role[role].append(cost)
    return ArmSecondary(
        n=len(members),
        verify_attempts_per_landing=_per(sum(r.verify_attempts for r in members), landed),
        review_cycles_per_landing=_per(sum(r.review_cycles for r in members), landed),
        failures_after_merge_per_landing=_per(sum(r.failures_after_merge for r in members), landed),
        blocked_or_escalated_share=_per(sum(r.blocked_or_escalated for r in members), len(members)),
        cost_per_task_by_role=MappingProxyType({
            role: math.fsum(costs) / len(members) for role, costs in sorted(by_role.items())
        }),
    )


def throughput_report(options: ThroughputOptions, since: datetime) -> ThroughputReport:
    """The cohort dispatched from *since* (see :func:`resolve_since`), read across options.roots.

    Raises ThroughputUnavailable naming the root: STORE_UNREADABLE when its
    runs.db or landing datum cannot be read, HISTORY_UNREADABLE when main's git
    history is needed for a landing and cannot be read.
    """
    cohort_end = options.cohort_end(since)
    unordered: list[CohortRecord] = []
    checks: list[_DispatchCheck] = []
    for root in options.roots:
        ledger = _read_ledger(root, options, since, cohort_end)
        unordered.extend(_records(root, ledger, options))
        checks.extend(_dispatch_checks(root, ledger, options))
    records = sorted(
        unordered, key=lambda record: (record.project, record.first_dispatch_at, record.task_id),
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
        secondary=MappingProxyType({arm: _arm_secondary(records, arm) for arm in Arm}),
        compliance=_compliance(checks, options.roles),
    )


def _instant(at: datetime) -> str:
    return at.astimezone(UTC).isoformat()


def _figure(value: float | None, digits: int = 3) -> str:
    return '-' if value is None else f'{value:.{digits}f}'


def _level(level: float) -> str:
    return f'{level * 100:g}%'


def _table(header: Sequence[str], rows: Iterable[Sequence[str]]) -> list[str]:
    """*header* and *rows* as aligned columns: the first left-justified, the rest right."""
    grid = [list(header), *(list(row) for row in rows)]
    widths = [max(len(row[column]) for row in grid) for column in range(len(header))]
    return [
        '  '.join(
            cell.ljust(width) if column == 0 else cell.rjust(width)
            for column, (cell, width) in enumerate(zip(row, widths, strict=True))
        ).rstrip()
        for row in grid
    ]


def _rule_line(report: ThroughputReport, table: str, excluded: str, defined: Sequence[str]) -> str:
    options = report.options
    treated, control = report.totals[Arm.TREATED], report.totals[Arm.CONTROL]
    return 'rule: ' + '; '.join([
        f'{table} table',
        f'window [{_instant(report.since)}, {_instant(report.cohort_end)}) of each task\'s first '
        f'routing_decision in roles {", ".join(options.roles)}',
        f'read_at {_instant(options.read_at)}: nothing at or after it counts',
        f'population = the {treated.n + control.n} cohort tasks ({treated.n} treated, '
        f'{control.n} control)',
        f'roots read {", ".join(str(root) for root in options.roots)}',
        f'arm = split_bucket({options.salt!r}, task_id, {options.buckets}): treated is bucket '
        f'{options.treated_bucket}, control every other bucket, by intent to treat',
        f'excluded {excluded}',
        *defined,
    ])


_OUTSIDE_COHORT = (
    'tasks first dispatched in those roles before the window or from its end, and tasks never '
    'dispatched in them'
)


def _ratio_section(report: ThroughputReport) -> list[str]:
    options = report.options
    rule = _rule_line(report, 'R', _OUTSIDE_COHORT, [
        'landed = the earliest recorded landing before read_at (merge_finalized done, a '
        'coalesced train member or an attributed solo passer)',
        'cost = invocations.cost_usd of every role started before read_at',
        'R = (landed_t / cost_t) / (landed_c / cost_c)',
        f'interval = {options.resamples} bootstrap redraws of tasks within each arm (seed '
        f'{options.seed}), type-7 percentile interval at {_level(options.interval_level)}',
        f'null = {options.resamples} permutations of the observed arm labels, arm sizes kept, '
        'drawn after the bootstrap from the same seed',
        'null p95/p99 = percentiles of max(R, 1/R)',
    ])
    rows = [
        (arm.value, str(totals.n), str(totals.landed), _figure(totals.cost_usd),
         _figure(totals.landed / totals.cost_usd if totals.cost_usd else None, 4))
        for arm, totals in ((arm, report.totals[arm]) for arm in Arm)
    ]
    if report.interval is None:
        interval = f'{_level(options.interval_level)} interval {report.interval_side.value}'
    else:
        low, high = report.interval
        interval = (
            f'{_level(options.interval_level)} interval [{_figure(low)}, {_figure(high)}] '
            f'{report.interval_side.value}'
        )
    if report.null is None:
        null = 'A/A null undefined: fewer than 2 relabellings define R'
    else:
        p95, p99 = report.null
        null = f'A/A null of max(R, 1/R): null p95 {_figure(p95)}, p99 {_figure(p99)}'
    return [
        rule,
        *_table(('arm', 'n', 'landed', 'cost_usd', 'landed_per_dollar'), rows),
        f'R {_figure(report.ratio)}; {interval}',
        null,
    ]


_SECONDARY_COLUMNS = (
    'verify_attempts_per_landing',
    'review_cycles_per_landing',
    'failures_after_merge_per_landing',
    'blocked_or_escalated_share',
)


def _secondary_section(report: ThroughputReport) -> list[str]:
    rule = _rule_line(report, 'secondary', f'{_OUTSIDE_COHORT}, and every row at or after read_at', [
        'verify attempts = task_results.verify_attempts plus the task\'s own verify_failed '
        'merge_attempt rows',
        'review cycles = task_results.review_cycles',
        'failures after merge = integration_skew merge_attempts implicating a commit the merge '
        'brought (git rev-list sha^1..sha), plus main commits saying "This reverts commit <sha>", '
        f'inside [landed_at, min(landed_at + {AFTER_MERGE_WINDOW.days}d, read_at))',
        '*_per_landing = arm total / arm landed tasks',
        f'blocked_or_escalated_share = arm tasks with any outcome in '
        f'{", ".join(sorted(BLOCKED_OUTCOMES))} / arm n',
        'cost per task by role = the role\'s arm cost / arm n',
    ])
    rows = []
    costs = []
    for arm in Arm:
        secondary = report.secondary[arm]
        rows.append((
            arm.value, str(secondary.n),
            *(_figure(getattr(secondary, column)) for column in _SECONDARY_COLUMNS),
        ))
        by_role = ', '.join(
            f'{role} {_figure(cost)}' for role, cost in secondary.cost_per_task_by_role.items()
        )
        costs.append(f'cost per task by role, {arm.value}: {by_role or "none"}')
    return [rule, *_table(('arm', 'n', *_SECONDARY_COLUMNS), rows), *costs]


def _compliance_section(report: ThroughputReport) -> list[str]:
    options = report.options
    rule = _rule_line(report, 'compliance', (
        'dispatches before the window\'s start or at or after read_at, of other roles, or of '
        'tasks outside the cohort'
    ), [
        f'expected effort = {options.treated_effort} treated, {options.control_effort} control',
        'effort_deviations = dispatches not at their arm\'s expected effort',
        f'treated_without_rule = treated dispatches whose rule_id lacks the prefix '
        f'{options.rule_prefix!r}',
        'COMPLIANCE FAILURE = a treated dispatch at the control effort',
    ])
    rows = [
        (role, str(counts.dispatches), str(counts.effort_deviations),
         str(counts.treated_without_rule))
        for role, counts in report.compliance.roles.items()
    ]
    failures = [
        f'COMPLIANCE FAILURE: {failure.project} task {failure.task_id} {failure.role} dispatched '
        f'at {_instant(failure.at)} at effort {failure.effort}, treated arm expects '
        f'{options.treated_effort}'
        for failure in report.compliance.failures
    ]
    return [
        rule,
        *_table(('role', 'dispatches', 'effort_deviations', 'treated_without_rule'), rows),
        *failures,
    ]


def render_text(report: ThroughputReport) -> str:
    """The R, secondary and compliance tables, each opened by its ``rule:`` line."""
    sections = (_ratio_section(report), _secondary_section(report), _compliance_section(report))
    return '\n\n'.join('\n'.join(section) for section in sections) + '\n'


def summary_record(report: ThroughputReport) -> dict[str, object]:
    """The report's one-line summary: ISO instants, and None for an undefined ratio or interval bound."""
    treated, control = report.totals[Arm.TREATED], report.totals[Arm.CONTROL]
    low, high = report.interval if report.interval is not None else (None, None)
    return {
        'salt': report.options.salt,
        'since': _instant(report.since),
        'cohort_days': report.options.cohort_days,
        'read_at': _instant(report.options.read_at),
        'n_treated': treated.n,
        'landed_treated': treated.landed,
        'cost_treated': treated.cost_usd,
        'n_control': control.n,
        'landed_control': control.landed,
        'cost_control': control.cost_usd,
        'ratio': report.ratio,
        'interval_level': report.options.interval_level,
        'interval_low': low,
        'interval_high': high,
        'compliance_failures': len(report.compliance.failures),
    }
