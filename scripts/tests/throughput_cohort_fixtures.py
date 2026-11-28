"""Builders for the effort experiment's throughput tests: cohort ledgers and real main lines.

A plain module rather than conftest fixtures, the convention of
``verify_host_trial_fixtures``: conftest appends this directory to ``sys.path``,
so the throughput and weekly-look suites import it by bare name, and neither
test file imports the other.

A :class:`CohortTask` states one task's ledger rows as values; :func:`seed_tasks`
writes them through the conftest ``SeedingConnection``. Arm membership is never
hard-coded: :func:`arm_task_ids` picks ids through ``split_bucket``, the same
function the resolver and the readout assign arms with. Timestamps are aware
datetimes, stored as ISO text with their offsets, as the live store holds them.
"""
from __future__ import annotations

import hashlib
import os
import subprocess
from collections.abc import Callable, Iterable, Mapping, Sequence
from dataclasses import dataclass
from datetime import UTC, datetime, timedelta
from pathlib import Path
from typing import Any, Protocol

import _checkout as _checkout
from git_env import scrubbed_git_env
from shared.project_layout import runs_db
from shared.routing_split import split_bucket

__all__ = [
    'BUCKETS',
    'CONTROL_EFFORT',
    'CohortTask',
    'Dispatch',
    'RunsDbSeeder',
    'TREATED_BUCKET',
    'TREATED_EFFORT',
    'TREATED_RULE_ID',
    'arm_task_ids',
    'build_cohort_root',
    'init_main_repo',
    'land_branch',
    'planted_cohort',
    'revert_merge',
    'seed_tasks',
]

BUCKETS = 2
TREATED_BUCKET = 1
TREATED_EFFORT = 'high'
CONTROL_EFFORT = 'xhigh'
TREATED_RULE_ID = 'effort-impl-high-implementer'

_MODEL = 'claude-opus-5-5'
_GIT_TIMEOUT_S = 30
_INITIAL_COMMIT_DATE = datetime(2026, 1, 1, tzinfo=UTC)
_IDENTITY = ('-c', 'user.name=fixture', '-c', 'user.email=fixture@example.invalid',
             '-c', 'commit.gpgsign=false')


@dataclass(frozen=True, kw_only=True)
class Dispatch:
    """One routing_decision row: ``events.role`` is *role*, and data carries role, effort and rule_id."""

    at: datetime
    role: str = 'implementer'
    effort: str
    rule_id: str | None = None


@dataclass(frozen=True, kw_only=True)
class CohortTask:
    """One task's ledger rows.

    *invocations* are (role, cost_usd, started_at); *results* are task_results
    rows (outcome, completed_at, verify_attempts, review_cycles), each under its
    own run; *merge_attempts* are (at, data). A landing is a merge_finalized
    ``done`` row at *landed_at* naming *merge_sha*, a stand-in sha derived from
    the task id when none is given.
    """

    task_id: int | str
    dispatches: Sequence[Dispatch]
    invocations: Sequence[tuple[str, float, datetime]] = ()
    landed_at: datetime | None = None
    merge_sha: str | None = None
    results: Sequence[tuple[str, datetime, int, int]] = ()
    merge_attempts: Sequence[tuple[datetime, Mapping[str, Any]]] = ()


class RunsDbSeeder(Protocol):
    """The slice of the conftest ``SeedingConnection`` these builders and their tests write through."""

    def execute(self, sql: str, /) -> object: ...

    def seed_event(
        self, timestamp: str, event_type: str, *,
        task_id: str | None = None, role: str | None = None, data: Mapping[str, Any] | None = None,
    ) -> None: ...

    def seed_invocation(
        self, *, model: str, role: str, started_at: str, completed_at: str,
        task_id: str | None = None, cost_usd: float = 0.0,
    ) -> None: ...

    def seed_task_result(
        self, *, task_id: str, outcome: str, completed_at: str,
        run_id: str = 'run-1', review_cycles: int = 0, verify_attempts: int = 0,
    ) -> None: ...


def _iso(at: datetime) -> str:
    if at.tzinfo is None:
        raise ValueError(f'fixture timestamps are aware, got naive {at!r}')
    return at.isoformat()


def _stand_in_sha(task_id: int | str) -> str:
    return hashlib.sha1(f'landing-of-task-{task_id}'.encode()).hexdigest()


def arm_task_ids(
    salt: str, buckets: int, treated_bucket: int, n_per_arm: int, start: int = 1,
) -> tuple[list[int], list[int]]:
    """(treated_ids, control_ids), *n_per_arm* each, scanning integers up from *start*.

    Control is every bucket except *treated_bucket*, as the readout defines it.
    """
    treated: list[int] = []
    control: list[int] = []
    candidate = start
    while len(treated) < n_per_arm or len(control) < n_per_arm:
        arm = treated if split_bucket(salt, str(candidate), buckets) == treated_bucket else control
        if len(arm) < n_per_arm:
            arm.append(candidate)
        candidate += 1
    return treated, control


def seed_tasks(conn: RunsDbSeeder, tasks: Iterable[CohortTask]) -> None:
    """Write every row of *tasks* through *conn*, a conftest ``SeedingConnection``."""
    for task in tasks:
        task_id = str(task.task_id)
        for dispatch in task.dispatches:
            conn.seed_event(
                _iso(dispatch.at), 'routing_decision', task_id=task_id, role=dispatch.role,
                data={'role': dispatch.role, 'effort': dispatch.effort, 'rule_id': dispatch.rule_id},
            )
        for role, cost_usd, started_at in task.invocations:
            conn.seed_invocation(
                model=_MODEL, role=role, task_id=task_id, cost_usd=cost_usd,
                started_at=_iso(started_at), completed_at=_iso(started_at + timedelta(minutes=5)),
            )
        for index, (outcome, completed_at, verify_attempts, review_cycles) in enumerate(task.results):
            conn.seed_task_result(
                task_id=task_id, outcome=outcome, completed_at=_iso(completed_at),
                run_id=f'run-{task_id}-{index}', verify_attempts=verify_attempts,
                review_cycles=review_cycles,
            )
        for at, data in task.merge_attempts:
            conn.seed_event(_iso(at), 'merge_attempt', task_id=task_id, data=dict(data))
        if task.landed_at is not None:
            sha = task.merge_sha if task.merge_sha is not None else _stand_in_sha(task.task_id)
            conn.seed_event(
                _iso(task.landed_at), 'merge_finalized', task_id=task_id,
                data={'state': 'done', 'merge_sha': sha},
            )


def build_cohort_root(
    root: Path, seeding_runs_db: Callable[[Path], RunsDbSeeder], tasks: Iterable[CohortTask], *,
    git: bool = True,
) -> RunsDbSeeder:
    """Lay a project root: ``<root>/data/orchestrator/runs.db`` seeded with *tasks*.

    *seeding_runs_db* is the conftest ``make_seeding_runs_db`` factory. With
    *git*, *root* is also a repo whose ``main`` holds one empty commit. Returns
    the open seeding connection, so a test can add rows after the cohort.

    Every seed_* commits its row, and a synthetic ledger needs no crash
    durability, so the connection runs ``PRAGMA synchronous = OFF``: without
    it, each commit's fsync makes seeding the slowest part of a test on a
    loaded host. Readers on other connections see every committed row either way.
    """
    directory = runs_db(root).parent
    directory.mkdir(parents=True, exist_ok=True)
    if git:
        init_main_repo(root)
    conn = seeding_runs_db(directory)
    conn.execute('PRAGMA synchronous = OFF')
    seed_tasks(conn, tasks)
    return conn


def _git(root: Path, *args: str, at: datetime | None = None) -> str:
    env = scrubbed_git_env(os.environ)
    if at is not None:
        env['GIT_AUTHOR_DATE'] = env['GIT_COMMITTER_DATE'] = _iso(at)
    completed = subprocess.run(
        ['git', '-C', str(root), *_IDENTITY, *args],
        check=True, capture_output=True, text=True, timeout=_GIT_TIMEOUT_S, env=env,
    )
    return completed.stdout.strip()


def init_main_repo(root: Path) -> None:
    """``git init -b main`` *root* with one empty commit at a fixed date."""
    root.mkdir(parents=True, exist_ok=True)
    _git(root, 'init', '-b', 'main')
    _git(root, 'commit', '--allow-empty', '-m', 'initial', at=_INITIAL_COMMIT_DATE)


def land_branch(root: Path, task_id: int | str, at: datetime) -> tuple[str, str]:
    """Land a one-commit ``task/<id>`` branch on main with ``--no-ff``, committed at *at*.

    The merge's subject is ``Merge task/<id> into main``. Landing the same task
    again re-cuts its branch from main, and each landing adds its own file, so
    any one of them reverts cleanly. Returns (merge_sha, branch_commit_sha).
    """
    branch = f'task/{task_id}'
    _git(root, 'checkout', '-q', '-B', branch, 'main')
    work = root / f'task-{task_id}-{int(at.timestamp())}.txt'
    work.write_text(f'landed at {_iso(at)}\n', encoding='utf-8')
    _git(root, 'add', work.name)
    _git(root, 'commit', '-q', '-m', f'work for {branch}', at=at)
    branch_commit = _git(root, 'rev-parse', 'HEAD')
    _git(root, 'checkout', '-q', 'main')
    _git(root, 'merge', '-q', '--no-ff', '-m', f'Merge {branch} into main', branch, at=at)
    return _git(root, 'rev-parse', 'HEAD'), branch_commit


def revert_merge(root: Path, merge_sha: str, at: datetime) -> str:
    """Revert *merge_sha* on main (``git revert -m 1 --no-edit``), committed at *at*; return its sha."""
    _git(root, 'checkout', '-q', 'main')
    _git(root, 'revert', '-m', '1', '--no-edit', merge_sha, at=at)
    return _git(root, 'rev-parse', 'HEAD')


def planted_cohort(
    salt: str, since: datetime, *, n_per_arm: int = 40, treated_cost_factor: float,
) -> list[CohortTask]:
    """Two arms of *n_per_arm* tasks with one shared per-position pattern, under ``BUCKETS``/``TREATED_BUCKET``.

    Position *i* is dispatched once, to the implementer, inside [since, since + 6d)
    (treated at TREATED_EFFORT under TREATED_RULE_ID, control at CONTROL_EFFORT
    with no rule). It lands within 3 days of that dispatch unless ``i % 5 == 0``,
    and its implementer cost is ``(1 + i % 5) * 2.0``, times *treated_cost_factor*
    in the treated arm. So the factor alone separates the arms: 0.6 plants a 40%
    cost cut, ``1 / 0.6`` a loss, and 1.0 identical per-arm multisets.
    """
    treated_ids, control_ids = arm_task_ids(salt, BUCKETS, TREATED_BUCKET, n_per_arm)
    tasks: list[CohortTask] = []
    for treated, ids in ((True, treated_ids), (False, control_ids)):
        for i, task_id in enumerate(ids):
            at = since + timedelta(days=6) * i / n_per_arm
            cost = (1 + i % 5) * 2.0 * (treated_cost_factor if treated else 1.0)
            dispatch = (
                Dispatch(at=at, effort=TREATED_EFFORT, rule_id=TREATED_RULE_ID) if treated
                else Dispatch(at=at, effort=CONTROL_EFFORT)
            )
            tasks.append(CohortTask(
                task_id=task_id,
                dispatches=(dispatch,),
                invocations=(('implementer', cost, at + timedelta(minutes=1)),),
                landed_at=None if i % 5 == 0 else at + timedelta(hours=12 + 24 * (i % 3)),
            ))
    return tasks
