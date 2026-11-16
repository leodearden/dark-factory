"""Stored quality rows for the γ tests: a landing history, quality.db holding it and a marker
derivation, runs.db recording it.

The builders hold no expectations; tests do. Every row passes α's dataclass invariants, so what a
test stores is what read_landing_rows hands back. runs.db is written through
_quality_fixture.write_quality_runs_db, so a test can make runs.db record landings the store lacks.
"""

from __future__ import annotations

from collections.abc import Callable, Iterable, Sequence
from dataclasses import replace
from datetime import datetime, timedelta
from itertools import groupby
from pathlib import Path

from _quality_fixture import write_quality_runs_db
from _runs_db_fixture import EventRow
from shared.quality_series.git_lines import ModuleLines
from shared.quality_series.landings import Landing, LandingSource, LinesBase
from shared.quality_series.rows import Cost, LandingRow, Outcomes, SourceEpochs

from dashboard.data.quality_markers import DerivedState, Marker, PromptCoverage, RoutingFold
from dashboard.data.quality_store import (
    QualitySample,
    insert_sample,
    upsert_landing_rows,
    write_marker_state,
)
from dashboard.loops import _QualityStore

NO_OUTCOMES = Outcomes(
    rejected=None, rejection_events=None, reviewed=None, review_blocking=None,
    flake_confirmed=None, flake_real_red=None, flake_rows=None,
    reverify_count=None, verify_failed_rows=None,
)
NO_COST = Cost(
    all_role_usd=None, by_role=(), implementer_invocations=None, execute_iterations=None,
    turns=None, turns_capped=False, code_lines=None, src_lines=None, test_lines=None,
    review_rounds=None, review_rounds_source=None, charter_class=None,
    charter_class_source=None,
)


def epochs_from(start: datetime) -> SourceEpochs:
    """Every source's first row at *start*, so a history beginning there is fully covered."""
    return SourceEpochs(
        rejections=start, merge_verify=start, review_completed=start, flake_occurrence=start,
        invocation_turns=start, invocations=start,
    )


def _member(
    task_id: str, sha: str, ordinal: int, landed_at: datetime, unit_size: int, *,
    project_id: str,
) -> Landing:
    return Landing(
        project_id=project_id,
        merge_sha=sha,
        task_id=task_id,
        landed_at=landed_at,
        ordinal=ordinal,
        source=LandingSource.MERGE_FINALIZED if unit_size == 1 else LandingSource.TRAIN_MERGED,
        unit_id=sha,
        unit_size=unit_size,
        late_claim=False,
        git_first_parent=True,
        prev_landed_at=None,
        lines_base=LinesBase.FIRST_PARENT,
        base_sha=f'base-of-{sha}',
    )


def history(
    n: int,
    *,
    start: datetime,
    step: timedelta = timedelta(hours=2),
    project_id: str = 'dark_factory',
    epochs: SourceEpochs | None = None,
    outcome: Callable[[int], Outcomes] = lambda i: NO_OUTCOMES,
    cost: Callable[[int], Cost] = lambda i: NO_COST,
    modules: Callable[[int], tuple[ModuleLines, ...]] = lambda i: (),
    units: Sequence[int] | None = None,
) -> list[LandingRow]:
    """*n* landings with ordinals 0..n-1, one *step* apart per verify unit, newest last.

    *units* gives consecutive unit sizes summing to *n* (all units of one when None); a unit of
    k > 1 is a coalesced train whose members share merge_sha, unit_id, unit_size and landed_at.
    *outcome*, *cost* and *modules* take the landing's ordinal. *epochs* defaults to every source
    starting at *start*.
    """
    sizes = list(units) if units is not None else [1] * n
    if sum(sizes) != n or any(size < 1 for size in sizes):
        raise ValueError(f'unit sizes {sizes} do not partition {n} landings')
    vintage = epochs if epochs is not None else epochs_from(start)
    rows: list[LandingRow] = []
    for unit_index, size in enumerate(sizes):
        sha = f'sha-{unit_index:05d}'
        for _ in range(size):
            ordinal = len(rows)
            landing = _member(
                str(1000 + ordinal), sha, ordinal, start + unit_index * step, size,
                project_id=project_id,
            )
            rows.append(LandingRow(
                landing=landing, outcomes=outcome(ordinal), cost=cost(ordinal),
                modules=modules(ordinal), rule_vintage=vintage,
            ))
    return rows


def train(
    task_ids: Sequence[str],
    sha: str,
    ordinal: int,
    landed_at: datetime,
    *,
    project_id: str = 'dark_factory',
    outcome: Outcomes = NO_OUTCOMES,
    cost: Cost = NO_COST,
    modules: tuple[ModuleLines, ...] = (),
    epochs: SourceEpochs | None = None,
) -> list[LandingRow]:
    """One coalesced train: a member per task id from *ordinal* on, all landed at *landed_at*."""
    vintage = epochs if epochs is not None else epochs_from(landed_at)
    return [
        LandingRow(
            landing=_member(task_id, sha, ordinal + index, landed_at, len(task_ids),
                            project_id=project_id),
            outcomes=outcome, cost=cost, modules=modules, rule_vintage=vintage,
        )
        for index, task_id in enumerate(task_ids)
    ]


async def write_quality_db(
    path: Path, rows: Iterable[LandingRow], samples: Iterable[QualitySample] = (),
) -> None:
    """Store *rows* and append *samples* in quality.db at *path*, creating it if absent."""
    stored = list(rows)
    store = _QualityStore(path)
    await store.open()
    try:
        sampled_at = max((row.landing.landed_at for row in stored), default=None)
        if sampled_at is not None:
            await upsert_landing_rows(store.connection, stored, sampled_at=sampled_at)
        for sample in samples:
            await insert_sample(store.connection, sample)
        await store.connection.commit()
    finally:
        await store.close()


async def write_marker_state_db(
    path: Path, project_id: str, markers: Iterable[Marker], *, refreshed_at: datetime,
) -> None:
    """Store *markers* as *project_id*'s complete marker derivation, refreshed at *refreshed_at*,
    in quality.db at *path*; no routing decision is folded."""
    state = DerivedState(
        fold=replace(RoutingFold.empty(project_id), complete=True), markers=tuple(markers),
        skipped=0,
        prompt_coverage=PromptCoverage(digest_since=None, roles_py_landings=None,
                                       counted_since=None,
                                       reason='no routing decision is folded yet'),
        refreshed_at=refreshed_at,
    )
    async with _QualityStore(path) as store:
        await write_marker_state(store.connection, state)
        await store.connection.commit()


def _unit_event(members: list[Landing]) -> EventRow:
    head = members[0]
    if head.source is LandingSource.MERGE_FINALIZED:
        return (head.landed_at, head.task_id, 'merge_finalized',
                {'state': 'done', 'merge_sha': head.merge_sha})
    return (head.landed_at, members[-1].task_id, 'train_merged', {
        'member_task_ids': [member.task_id for member in members],
        'merge_commit_sha': head.merge_sha,
    })


def write_landing_events(
    root: Path, landings: Iterable[LandingRow], extra_events: Iterable[EventRow] = (),
) -> Path:
    """*root*'s runs.db recording each landing: a done merge_finalized per plain landing, one
    train_merged per train, stamped at landed_at; then *extra_events* verbatim.

    runs.db is written once, so a row the landing rule refuses rides in *extra_events*.
    """
    ordered = sorted((row.landing for row in landings), key=lambda landing: landing.ordinal)
    events = [
        _unit_event(list(members))
        for _, members in groupby(ordered, key=lambda landing: landing.unit_id)
    ]
    return write_quality_runs_db(root, [*events, *extra_events])
