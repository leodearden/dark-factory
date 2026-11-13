"""One marker tick over real project roots: a git checkout plus its runs.db, into a real quality.db."""

from __future__ import annotations

import asyncio
import os
import sqlite3
from collections.abc import AsyncIterator, Sequence
from contextlib import closing, suppress
from dataclasses import dataclass
from datetime import UTC, datetime, timedelta
from pathlib import Path

import aiosqlite
import pytest
from _quality_fixture import GitRepo, append_events, build_quality_project, routing_decision
from _runs_db_fixture import EventRow, write_runs_db
from shared.project_id import project_id_for_root
from shared.project_layout import runs_db
from shared.tasks_db_readonly import TaskDbUnreadable, connect_ro

from dashboard.config import DashboardConfig
from dashboard.data.db import DbPool
from dashboard.data.quality_marker_sampler import (
    FACTORY_REPO,
    MARKER_GIT_TIMEOUT_SECONDS,
    MARKER_PER_ROOT_BUDGET_SECONDS,
    ROLES_PY,
    refresh_markers,
)
from dashboard.data.quality_markers import (
    CoverageState,
    DerivedState,
    Marker,
    MarkerKind,
    PromptCoverage,
    RoutingFold,
    StoredMarkerState,
    derived_markers,
    fold_routing_decisions,
)
from dashboard.data.quality_sampler import (
    QUALITY_PER_ROOT_BUDGET_SECONDS,
    QUALITY_SAMPLE_INTERVAL_SECONDS,
)
from dashboard.data.quality_store import read_marker_state
from dashboard.loops import _quality_loop, _QualityStore

T0 = datetime(2026, 10, 27, 13, 0, tzinfo=UTC)
NOW = T0 + timedelta(days=12)
ROLES = 'orchestrator/src/orchestrator/agents/roles.py'


def _at(days: float = 0, hours: float = 0) -> datetime:
    return T0 + timedelta(days=days, hours=hours)


def _route(
    at: datetime, role: str, *, effort: str = 'high', max_turns: int = 60,
    digest: str | None = None,
) -> EventRow:
    return routing_decision(at, role, model='opus', effort=effort, budget_usd=12.0,
                            max_turns=max_turns, prompt_digest=digest)


REVIEWER_FLIP: list[EventRow] = [
    *(_route(_at(hours=hours), 'reviewer_comprehensive') for hours in (0, 0.2, 0.4, 0.6)),
    *(_route(at, 'reviewer_comprehensive', effort='xhigh', max_turns=90)
      for at in (_at(5), _at(5, 1), _at(5, 2), _at(7))),
]
"""Sketch row 7's shape: four high/60 dispatches, then a persisting xhigh/90 step from day 5."""

MERGER_FLIP: list[EventRow] = [
    *(_route(_at(8, hours), 'merger') for hours in (0, 1, 2)),
    *(_route(at, 'merger', effort='max') for at in (_at(9), _at(9, 1), _at(11))),
]


@dataclass(frozen=True)
class Root:
    root: Path
    project_id: str

    @property
    def runs_db(self) -> Path:
        return runs_db(self.root)


def _checkout(path: Path, events: Sequence[EventRow] = REVIEWER_FLIP) -> Root:
    GitRepo(path).init()
    write_runs_db(path, events)
    return Root(path, project_id_for_root(path))


@pytest.fixture
def project(tmp_path: Path) -> Root:
    return _checkout(tmp_path / 'proj')


@pytest.fixture
def factory(tmp_path: Path) -> Path:
    """A factory checkout with no roles.py landing at all."""
    repo = GitRepo(tmp_path / 'factory')
    repo.init()
    return repo.root


@pytest.fixture
async def conn(tmp_path: Path) -> AsyncIterator[aiosqlite.Connection]:
    async with _QualityStore(tmp_path / 'quality.db') as store:
        yield store.connection


@pytest.fixture
def no_curated(tmp_path: Path) -> Path:
    return tmp_path / 'no-curated-markers.yaml'


async def _tick(
    conn: aiosqlite.Connection, roots: Sequence[Path], factory: Path, curated: Path, *,
    now: datetime = NOW, batch: int = 20_000, per_root_budget_s: float = 60.0,
) -> None:
    await refresh_markers(conn, roots, now=now, curated_path=curated, factory_repo=factory,
                          batch=batch, per_root_budget_s=per_root_budget_s)


async def _derived(conn: aiosqlite.Connection, project_id: str) -> DerivedState:
    stored = await read_marker_state(conn, project_id)
    assert stored is not None and stored.derived is not None, f'no derivation stored: {stored}'
    assert stored.failure is None, stored.failure
    return stored.derived


def _full_fold(root: Root) -> RoutingFold:
    with closing(connect_ro(root.runs_db)) as runs:
        return fold_routing_decisions(runs, RoutingFold.empty(root.project_id), batch=None)


def _full_markers(root: Root) -> tuple[Marker, ...]:
    with closing(connect_ro(root.runs_db)) as runs:
        return tuple(derived_markers(runs, root.project_id, ()))


def test_the_factory_repo_is_this_checkout_and_carries_roles_py() -> None:
    assert ROLES_PY == ROLES
    assert (FACTORY_REPO / ROLES_PY).is_file()
    assert (FACTORY_REPO / '.git').exists()


def test_the_marker_tick_fits_after_the_sample_tick_in_half_the_interval() -> None:
    assert MARKER_GIT_TIMEOUT_SECONDS < MARKER_PER_ROOT_BUDGET_SECONDS
    assert (QUALITY_PER_ROOT_BUDGET_SECONDS + MARKER_PER_ROOT_BUDGET_SECONDS
            <= QUALITY_SAMPLE_INTERVAL_SECONDS / 2)


async def test_the_first_tick_stores_the_fold_and_the_row_7_effort_marker(
    conn: aiosqlite.Connection, project: Root, factory: Path, no_curated: Path,
) -> None:
    await _tick(conn, [project.root], factory, no_curated)

    derived = await _derived(conn, project.project_id)
    assert derived.fold == _full_fold(project)
    assert derived.fold.complete
    assert derived.markers == _full_markers(project)
    [effort] = derived.markers
    assert (effort.kind, effort.at, effort.scope) == (MarkerKind.EFFORT, _at(5), project.project_id)
    assert derived.refreshed_at == NOW


async def test_a_later_tick_folds_only_the_new_rows_and_re_derives(
    conn: aiosqlite.Connection, project: Root, factory: Path, no_curated: Path,
) -> None:
    await _tick(conn, [project.root], factory, no_curated)
    first = await _derived(conn, project.project_id)
    append_events(project.runs_db, MERGER_FLIP)
    with closing(sqlite3.connect(project.runs_db)) as runs:
        runs.execute("UPDATE events SET data = json_set(data, '$.model', 'haiku') WHERE id = 1")
        runs.commit()

    await _tick(conn, [project.root], factory, no_curated, now=NOW + timedelta(minutes=10))

    second = await _derived(conn, project.project_id)
    assert first.fold.cursor == len(REVIEWER_FLIP)
    assert second.fold.cursor == len(REVIEWER_FLIP) + len(MERGER_FLIP)
    assert second.fold.steps[0] == first.fold.steps[0]
    assert second.fold.steps[0].model == 'opus'
    assert [(m.kind, m.at, m.source.role) for m in second.markers] == [
        (MarkerKind.EFFORT, _at(5), 'reviewer_comprehensive'),
        (MarkerKind.EFFORT, _at(9), 'merger'),
    ]
    assert second.refreshed_at == NOW + timedelta(minutes=10)


async def test_a_batched_fold_is_stored_incomplete_then_completes_to_the_full_tick(
    conn: aiosqlite.Connection, project: Root, factory: Path, no_curated: Path,
) -> None:
    await _tick(conn, [project.root], factory, no_curated, batch=3)
    first = await _derived(conn, project.project_id)
    assert not first.fold.complete
    assert first.fold.cursor == 3

    for minutes in range(1, 5):
        await _tick(conn, [project.root], factory, no_curated, batch=3,
                    now=NOW + timedelta(minutes=minutes))
    final = await _derived(conn, project.project_id)

    assert final.fold == _full_fold(project)
    assert final.markers == _full_markers(project)


async def test_a_root_that_is_not_a_git_checkout_gets_no_state(
    conn: aiosqlite.Connection, project: Root, factory: Path, no_curated: Path, tmp_path: Path,
) -> None:
    plain = tmp_path / 'not-a-checkout'
    write_runs_db(plain, REVIEWER_FLIP)

    await _tick(conn, [plain, project.root], factory, no_curated)

    assert await read_marker_state(conn, project_id_for_root(plain)) is None
    await _derived(conn, project.project_id)


async def test_an_absent_runs_db_records_the_refusal_and_keeps_the_previous_state(
    conn: aiosqlite.Connection, project: Root, factory: Path, no_curated: Path,
) -> None:
    await _tick(conn, [project.root], factory, no_curated)
    previous = await _derived(conn, project.project_id)
    project.runs_db.unlink()
    with pytest.raises(TaskDbUnreadable) as refusal:
        connect_ro(project.runs_db)
    later = NOW + timedelta(minutes=10)

    await _tick(conn, [project.root], factory, no_curated, now=later)

    assert await read_marker_state(conn, project.project_id) == StoredMarkerState(
        derived=previous, attempted_at=later,
        failure=f'TaskDbUnreadable: {refusal.value.path}: absent',
    )


def _put_git_first_on_path(monkeypatch: pytest.MonkeyPatch, directory: Path, body: str) -> None:
    directory.mkdir()
    fake_git = directory / 'git'
    fake_git.write_text(f'#!/bin/sh\n{body}\n')
    fake_git.chmod(0o755)
    monkeypatch.setenv('PATH', f'{directory}{os.pathsep}{os.environ["PATH"]}')


async def test_a_root_overrunning_its_budget_records_a_failure_naming_it(
    conn: aiosqlite.Connection, project: Root, factory: Path, no_curated: Path, tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    _put_git_first_on_path(monkeypatch, tmp_path / 'slow-git', 'sleep 1\nexit 1')

    await _tick(conn, [project.root], factory, no_curated, per_root_budget_s=0.2)

    assert await read_marker_state(conn, project.project_id) == StoredMarkerState(
        derived=None, attempted_at=NOW,
        failure='marker refresh exceeded the 0.2s per-root budget',
    )


async def test_one_failing_root_does_not_stop_another(
    conn: aiosqlite.Connection, factory: Path, no_curated: Path, tmp_path: Path,
) -> None:
    failing = _checkout(tmp_path / 'failing')
    failing.runs_db.unlink()
    measured = _checkout(tmp_path / 'measured')

    await _tick(conn, [failing.root, measured.root], factory, no_curated)

    stored = await read_marker_state(conn, failing.project_id)
    assert stored is not None and stored.derived is None
    assert (stored.failure or '').startswith('TaskDbUnreadable: ')
    assert (await _derived(conn, measured.project_id)).markers == _full_markers(measured)


async def test_a_write_quality_db_refuses_is_recorded_and_keeps_the_previous_state(
    conn: aiosqlite.Connection, project: Root, factory: Path, no_curated: Path,
) -> None:
    await _tick(conn, [project.root], factory, no_curated)
    previous = await _derived(conn, project.project_id)
    append_events(project.runs_db, MERGER_FLIP)
    await conn.execute("CREATE TRIGGER refuse_markers BEFORE INSERT ON markers_derived"
                       " BEGIN SELECT RAISE(ABORT, 'markers_derived refused'); END")
    await conn.commit()
    later = NOW + timedelta(minutes=10)

    await _tick(conn, [project.root], factory, no_curated, now=later)

    assert await read_marker_state(conn, project.project_id) == StoredMarkerState(
        derived=previous, attempted_at=later,
        failure='marker write failed: IntegrityError: markers_derived refused',
    )


POISONS = {
    'retired-marker-kind': "UPDATE markers_derived SET kind = 'retired_kind' WHERE project_id = ?",
    'retired-routing-layer':
        "UPDATE routing_steps SET source_layer = 'retired_layer' WHERE project_id = ?",
    'unparseable-fold-bound':
        "UPDATE marker_state SET first_decision_at = 'not an instant' WHERE project_id = ?",
    'inconsistent-coverage': 'UPDATE marker_state SET roles_py_landings = -1 WHERE project_id = ?',
}
"""Stored rows a later vocabulary or invariant change could leave undecodable."""


@pytest.mark.parametrize('poison', POISONS.values(), ids=POISONS.keys())
async def test_a_stored_state_that_no_longer_decodes_heals_and_stops_no_other_root(
    conn: aiosqlite.Connection, factory: Path, no_curated: Path, tmp_path: Path, poison: str,
) -> None:
    poisoned = _checkout(tmp_path / 'poisoned')
    other = _checkout(tmp_path / 'other', MERGER_FLIP)
    roots = [poisoned.root, other.root]
    await _tick(conn, roots, factory, no_curated)
    await conn.execute(poison, (poisoned.project_id,))
    await conn.commit()
    with pytest.raises(ValueError):
        await read_marker_state(conn, poisoned.project_id)
    later = NOW + timedelta(minutes=10)

    await _tick(conn, roots, factory, no_curated, now=later)

    healed = await _derived(conn, poisoned.project_id)
    assert (healed.fold, healed.markers) == (_full_fold(poisoned), _full_markers(poisoned))
    assert healed.refreshed_at == later
    assert (await _derived(conn, other.project_id)).refreshed_at == later


def _dated(monkeypatch: pytest.MonkeyPatch, at: str) -> None:
    monkeypatch.setenv('GIT_AUTHOR_DATE', at)
    monkeypatch.setenv('GIT_COMMITTER_DATE', at)


def _factory_with_merges(
    path: Path, monkeypatch: pytest.MonkeyPatch, merges: Sequence[tuple[str, str]],
) -> Path:
    """A factory checkout whose main carries one --no-ff merge per (date, path), dated so."""
    repo = GitRepo(path)
    _dated(monkeypatch, '2026-09-01T00:00:00+00:00')
    repo.init()
    for index, (at, changed) in enumerate(merges):
        _dated(monkeypatch, at)
        repo.merge(f'task/{index}', {changed: f'change {index}\n'}, f'Merge task/{index} into main')
    return repo.root


ROLES_PY_LANDINGS = [
    ('2026-10-20T00:00:00+00:00', ROLES),
    ('2026-10-29T00:00:00+00:00', ROLES),
    ('2026-10-30T00:00:00+00:00', 'orchestrator/src/orchestrator/other.py'),
    ('2026-11-02T00:00:00+00:00', ROLES),
    ('2026-11-04T00:00:00+00:00', ROLES),
]
"""Four roles.py landings: one before the routing history starts (T0), one between T0 and the
curated prompt entry below, and two after it; plus a landing elsewhere that never counts."""

CURATED_PROMPT = """\
- date: 2026-11-01
  kind: prompt
  label: "Reviewer prompt rewritten"
- date: 2026-11-03
  kind: prompt
  project: some_other_project
  label: "Not this project's prompt"
"""


async def _coverage(conn: aiosqlite.Connection, project_id: str) -> PromptCoverage:
    return (await _derived(conn, project_id)).prompt_coverage


async def test_roles_py_landings_count_since_the_last_curated_prompt_marker(
    conn: aiosqlite.Connection, project: Root, tmp_path: Path, monkeypatch: pytest.MonkeyPatch,
) -> None:
    factory = _factory_with_merges(tmp_path / 'factory', monkeypatch, ROLES_PY_LANDINGS)
    curated = tmp_path / 'curated.yaml'
    curated.write_text(CURATED_PROMPT)

    await _tick(conn, [project.root], factory, curated)

    assert await _coverage(conn, project.project_id) == PromptCoverage(
        digest_since=None, roles_py_landings=2,
        counted_since=datetime(2026, 11, 1, tzinfo=UTC), reason=None,
    )


async def test_without_a_curated_prompt_marker_they_count_since_the_first_decision(
    conn: aiosqlite.Connection, project: Root, no_curated: Path, tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    factory = _factory_with_merges(tmp_path / 'factory', monkeypatch, ROLES_PY_LANDINGS)

    await _tick(conn, [project.root], factory, no_curated)

    assert await _coverage(conn, project.project_id) == PromptCoverage(
        digest_since=None, roles_py_landings=3, counted_since=T0, reason=None,
    )


async def test_once_a_decision_carries_a_prompt_digest_git_is_not_consulted(
    conn: aiosqlite.Connection, no_curated: Path, factory: Path, tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    written = _checkout(tmp_path / 'written', [
        *REVIEWER_FLIP, _route(_at(7, 1), 'merger', digest='d' * 64),
    ])
    consulted = tmp_path / 'git-was-run'
    _put_git_first_on_path(monkeypatch, tmp_path / 'tell-tale-git', f'touch {consulted}\nexit 1')

    await _tick(conn, [written.root], factory, no_curated)

    coverage = await _coverage(conn, written.project_id)
    assert coverage == PromptCoverage(digest_since=_at(7, 1), roles_py_landings=None,
                                      counted_since=None, reason=None)
    assert coverage.state is CoverageState.WRITTEN
    assert not consulted.exists()


async def test_a_factory_path_that_is_not_a_repo_is_a_coverage_reason_not_a_failure(
    conn: aiosqlite.Connection, project: Root, no_curated: Path, tmp_path: Path,
) -> None:
    not_a_repo = tmp_path / 'not-a-repo'
    not_a_repo.mkdir()

    await _tick(conn, [project.root], not_a_repo, no_curated)

    assert await _coverage(conn, project.project_id) == PromptCoverage(
        digest_since=None, roles_py_landings=None, counted_since=None,
        reason=f'GitUnavailable: {not_a_repo} is not a git repository',
    )


async def test_the_quality_loop_stores_marker_state_within_its_first_tick(
    tmp_path: Path,
) -> None:
    """The marker tick runs on the loop's first tick, before its first aligned sleep.

    It is read through a DbPool connection of its own, as /api/v2/dashboard/quality reads it,
    because the loop's connection is the loop's alone
    (dashboard/src/dashboard/app.py::lifespan) and a commit on it ends any snapshot read sharing it.
    """
    project = build_quality_project(tmp_path / 'proj',
                                    first_landed_at=datetime(2026, 11, 2, 9, 0, tzinfo=UTC))
    append_events(runs_db(project.root), REVIEWER_FLIP)
    pid = project_id_for_root(project.root)
    path = tmp_path / 'quality.db'
    store = _QualityStore(path)
    await store.open()
    pool = DbPool()
    loop = asyncio.create_task(_quality_loop(store, DashboardConfig(project_root=project.root)))
    try:
        reader = await pool.get(path)
        assert reader is not None
        async with asyncio.timeout(60):
            while (stored := await read_marker_state(reader, pid)) is None:
                await asyncio.sleep(0.05)
    finally:
        loop.cancel()
        with suppress(asyncio.CancelledError):
            await loop
        try:
            await pool.close_all()
        finally:
            await store.close()

    assert stored.failure is None, stored.failure
    assert stored.derived is not None
    assert [(m.kind, m.at) for m in stored.derived.markers] == [(MarkerKind.EFFORT, _at(5))]
