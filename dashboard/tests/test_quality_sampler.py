"""One quality tick over real project roots: a git repo plus its runs.db, into a real quality.db."""

from __future__ import annotations

import asyncio
import os
import sqlite3
from collections.abc import AsyncIterator
from contextlib import closing, suppress
from datetime import UTC, datetime, timedelta
from pathlib import Path
from unittest.mock import AsyncMock, patch

import aiosqlite
import pytest
from _dashboard_helpers import drive_loop_until, yielding_noop_sleep
from _quality_fixture import QualityProject, build_quality_project
from shared.async_sqlite_base import CheckpointResult
from shared.project_id import project_id_for_root
from shared.project_layout import runs_db
from shared.quality_series.landings import GitCrossCheck
from shared.quality_series.rows import FIRST_UNIT_GIT_FLOOR_S
from shared.tasks_db_readonly import TaskDbUnreadable, connect_ro

from dashboard.config import DashboardConfig
from dashboard.data.quality_sampler import (
    QUALITY_GIT_BUDGET_SECONDS,
    QUALITY_PER_ROOT_BUDGET_SECONDS,
    QUALITY_REV_LIST_TIMEOUT_SECONDS,
    QUALITY_SAMPLE_INTERVAL_SECONDS,
    collect_quality,
)
from dashboard.data.quality_store import (
    QualitySample,
    read_landing_rows,
    read_samples,
)
from dashboard.loops import _quality_loop, _QualityStore

FIRST_LANDED_AT = datetime(2026, 11, 2, 9, 0, tzinfo=UTC)


@pytest.fixture
def project(tmp_path: Path) -> QualityProject:
    return build_quality_project(tmp_path / 'proj', first_landed_at=FIRST_LANDED_AT)


@pytest.fixture
async def conn(tmp_path: Path) -> AsyncIterator[aiosqlite.Connection]:
    store = _QualityStore(tmp_path / 'quality.db')
    await store.open()
    try:
        yield store.connection
    finally:
        await store.close()


def _settled(project: QualityProject) -> datetime:
    return project.train_landed_at + timedelta(hours=2)


async def _task_ids(conn: aiosqlite.Connection, project: QualityProject) -> list[str]:
    rows = await read_landing_rows(conn, project_id_for_root(project.root))
    return [row.landing.task_id for row in rows]


async def test_one_cold_tick_stores_every_landing_and_names_the_newest_as_cursor(
    conn: aiosqlite.Connection, project: QualityProject,
) -> None:
    pid = project_id_for_root(project.root)
    now = _settled(project)

    await collect_quality(conn, [project.root], now=now)

    rows = await read_landing_rows(conn, pid)
    assert [row.landing.task_id for row in rows] == ['101', '102', '103']
    plain, *train = rows
    assert plain.landing.merge_sha == project.plain_sha
    assert [row.landing.task_id for row in train] == list(project.train_members)
    for member in train:
        assert member.landing.merge_sha == member.landing.unit_id == project.train_sha
        assert member.landing.unit_size == 2
        assert member.landing.project_id == pid
    assert await read_samples(conn, pid) == [
        QualitySample.value(
            pid, now, cursor_sha=project.train_sha, landings_sampled=3, landings_deferred=0,
            git_check=GitCrossCheck(off_first_parent=0, merges_without_landing=1, late_claims=0),
        ),
    ]


async def test_a_landing_younger_than_the_settle_window_is_re_measured_in_place(
    conn: aiosqlite.Connection, project: QualityProject,
) -> None:
    pid = project_id_for_root(project.root)
    first_tick = project.train_landed_at + timedelta(minutes=5)

    await collect_quality(conn, [project.root], now=first_tick)
    with closing(sqlite3.connect(runs_db(project.root))) as runs:
        runs.execute(
            'INSERT INTO task_results (run_id, task_id, project_id, outcome, completed_at,'
            " execute_iterations) VALUES ('run-102', '102', 'fixture', 'done', ?, 4)",
            ((project.train_landed_at + timedelta(minutes=2)).isoformat(),),
        )
        runs.commit()
    before = {row.landing.task_id: row for row in await read_landing_rows(conn, pid)}
    await collect_quality(conn, [project.root], now=first_tick + timedelta(minutes=10))
    after = {row.landing.task_id: row for row in await read_landing_rows(conn, pid)}

    assert sorted(after) == ['101', '102', '103']
    assert before['102'].cost.execute_iterations is None
    assert after['102'].cost.execute_iterations == 4

    await collect_quality(conn, [project.root], now=_settled(project))

    last = (await read_samples(conn, pid))[-1]
    assert last.landings_sampled == 0
    assert last.cursor_sha == project.train_sha
    assert await _task_ids(conn, project) == ['101', '102', '103']


async def test_a_budget_truncated_backfill_resumes_from_what_was_stored(
    conn: aiosqlite.Connection, project: QualityProject,
) -> None:
    pid = project_id_for_root(project.root)
    now = _settled(project)

    await collect_quality(conn, [project.root], now=now, git_budget_s=0.0)
    assert await _task_ids(conn, project) == ['101']
    await collect_quality(conn, [project.root], now=now + timedelta(minutes=10), git_budget_s=0.0)
    assert await _task_ids(conn, project) == ['101', '102', '103']
    await collect_quality(conn, [project.root], now=now + timedelta(minutes=20), git_budget_s=0.0)

    samples = await read_samples(conn, pid)
    assert [
        (s.cursor_sha, s.landings_sampled, s.landings_deferred) for s in samples
    ] == [
        (project.plain_sha, 1, 2),
        (project.train_sha, 2, 0),
        (project.train_sha, 0, 0),
    ]
    assert await _task_ids(conn, project) == ['101', '102', '103']


async def test_a_root_that_is_not_a_git_checkout_gets_no_sample(
    conn: aiosqlite.Connection, project: QualityProject, tmp_path: Path,
) -> None:
    plain_dir = tmp_path / 'not-a-checkout'
    plain_dir.mkdir()

    await collect_quality(conn, [plain_dir, project.root], now=_settled(project))

    plain_pid = project_id_for_root(plain_dir)
    assert await read_samples(conn, plain_pid) == []
    assert await read_landing_rows(conn, plain_pid) == []
    assert len(await read_samples(conn, project_id_for_root(project.root))) == 1


async def _assert_one_gap(
    conn: aiosqlite.Connection, project: QualityProject, now: datetime, reason: str,
) -> None:
    pid = project_id_for_root(project.root)
    assert await read_samples(conn, pid) == [QualitySample.gap(pid, now, reason)]
    assert await read_landing_rows(conn, pid) == []


async def test_an_absent_runs_db_is_a_gap_naming_the_refusal(
    conn: aiosqlite.Connection, project: QualityProject,
) -> None:
    """The refusal's own fields, not its remedy prose, which is written for tasks.db."""
    runs_db(project.root).unlink()
    with pytest.raises(TaskDbUnreadable) as refusal:
        connect_ro(runs_db(project.root))
    now = _settled(project)

    await collect_quality(conn, [project.root], now=now)

    await _assert_one_gap(conn, project, now, f'TaskDbUnreadable: {refusal.value.path}: absent')


async def test_a_sample_write_that_fails_leaves_a_gap_naming_the_failure(
    conn: aiosqlite.Connection, project: QualityProject,
) -> None:
    await conn.execute(
        "CREATE TRIGGER refuse_landings BEFORE INSERT ON landings"
        " BEGIN SELECT RAISE(ABORT, 'landings refused'); END"
    )
    await conn.commit()
    now = _settled(project)

    await collect_quality(conn, [project.root], now=now)

    await _assert_one_gap(
        conn, project, now, 'sample write failed: IntegrityError: landings refused',
    )


async def test_each_root_gets_its_own_sample_when_one_fails_and_one_measures(
    conn: aiosqlite.Connection, tmp_path: Path,
) -> None:
    measured = build_quality_project(tmp_path / 'measured', first_landed_at=FIRST_LANDED_AT)
    failing = build_quality_project(tmp_path / 'failing', first_landed_at=FIRST_LANDED_AT)
    runs_db(failing.root).unlink()
    now = _settled(measured)

    await collect_quality(conn, [failing.root, measured.root], now=now)

    [gap] = await read_samples(conn, project_id_for_root(failing.root))
    [value] = await read_samples(conn, project_id_for_root(measured.root))
    assert (gap.reason or '').startswith('TaskDbUnreadable: ')
    assert (value.cursor_sha, value.landings_sampled) == (measured.train_sha, 3)
    assert await _task_ids(conn, measured) == ['101', '102', '103']
    assert await _task_ids(conn, failing) == []


async def test_a_missing_git_binary_is_a_gap_naming_it(
    conn: aiosqlite.Connection, project: QualityProject, tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    empty = tmp_path / 'empty-path'
    empty.mkdir()
    monkeypatch.setenv('PATH', str(empty))
    now = _settled(project)

    await collect_quality(conn, [project.root], now=now)

    await _assert_one_gap(
        conn, project, now, f'GitUnavailable: no git binary to run in {os.fspath(project.root)}',
    )


async def test_a_locked_runs_db_is_a_gap_carrying_sqlites_error(
    conn: aiosqlite.Connection, project: QualityProject,
) -> None:
    now = _settled(project)
    with closing(sqlite3.connect(runs_db(project.root), isolation_level=None)) as holder:
        holder.execute('BEGIN EXCLUSIVE')
        try:
            await collect_quality(conn, [project.root], now=now)
        finally:
            holder.execute('ROLLBACK')

    await _assert_one_gap(conn, project, now, 'OperationalError: database is locked')


async def test_a_root_overrunning_the_backstop_is_a_gap_naming_the_budget(
    conn: aiosqlite.Connection, project: QualityProject, tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    slow = tmp_path / 'slow-git'
    slow.mkdir()
    fake_git = slow / 'git'
    fake_git.write_text('#!/bin/sh\nsleep 1\nexit 1\n')
    fake_git.chmod(0o755)
    monkeypatch.setenv('PATH', f'{slow}{os.pathsep}{os.environ["PATH"]}')
    now = _settled(project)

    await collect_quality(conn, [project.root], now=now, per_root_budget_s=0.2)

    await _assert_one_gap(conn, project, now, 'quality sampling exceeded the 0.2s per-root budget')


def test_the_backstop_sits_between_the_producers_own_bound_and_half_the_tick() -> None:
    floor = QUALITY_REV_LIST_TIMEOUT_SECONDS + max(FIRST_UNIT_GIT_FLOOR_S, QUALITY_GIT_BUDGET_SECONDS)

    assert floor < QUALITY_PER_ROOT_BUDGET_SECONDS <= QUALITY_SAMPLE_INTERVAL_SECONDS / 2


async def test_the_quality_loop_samples_a_landing_within_its_first_tick(
    project: QualityProject, tmp_path: Path,
) -> None:
    """PRD sketch row 1 at loop level: the first tick runs before the loop's first aligned sleep."""
    pid = project_id_for_root(project.root)
    store = _QualityStore(tmp_path / 'quality.db')
    await store.open()
    loop = asyncio.create_task(_quality_loop(store, DashboardConfig(project_root=project.root)))
    try:
        async with asyncio.timeout(60):
            while not (samples := await read_samples(store.connection, pid)):
                await asyncio.sleep(0.05)
        rows = await read_landing_rows(store.connection, pid)
    finally:
        loop.cancel()
        with suppress(asyncio.CancelledError):
            await loop
        await store.close()

    train = [row.landing.task_id for row in rows if row.landing.merge_sha == project.train_sha]
    assert train == list(project.train_members)
    first = samples[0]
    assert first == QualitySample.value(
        pid, first.ts, cursor_sha=project.train_sha, landings_sampled=3, landings_deferred=0,
        git_check=GitCrossCheck(off_first_parent=0, merges_without_landing=1, late_claims=0),
    ), f'expected a value sample naming the train merge as cursor, got {first}'


def _stub_checkpoint(store: _QualityStore, on_call: asyncio.Event | None = None) -> AsyncMock:
    async def _checkpoint(*_args: object, **_kwargs: object) -> CheckpointResult:
        if on_call is not None:
            on_call.set()
        return CheckpointResult(0, 0, 0)

    stub = AsyncMock(side_effect=_checkpoint)
    store.checkpoint = stub
    return stub


async def test_the_quality_loop_checkpoints_its_store(tmp_path: Path) -> None:
    """The WAL checkpoint every sampler store owes (the pragma-triad directive), on the quality loop."""
    checkpointed = asyncio.Event()
    async with _QualityStore(tmp_path / 'quality.db') as store:
        stub = _stub_checkpoint(store, checkpointed)
        with (
            patch('dashboard.loops.collect_quality', new=AsyncMock(return_value=None)),
            patch('dashboard.loops._sleep_to_aligned_tick', new=yielding_noop_sleep),
            patch('dashboard.loops._CHECKPOINT_INTERVAL_SECONDS', 0),
        ):
            await drive_loop_until(
                _quality_loop(store, DashboardConfig(project_root=tmp_path)), checkpointed,
            )

    assert stub.called, '_quality_loop never called store.checkpoint()'


async def test_the_quality_loop_checkpoints_at_most_once_per_interval(tmp_path: Path) -> None:
    ticks = 0
    six_ticks = asyncio.Event()

    async def _count_tick(*_args: object, **_kwargs: object) -> None:
        nonlocal ticks
        ticks += 1
        if ticks >= 6:
            six_ticks.set()

    async with _QualityStore(tmp_path / 'quality.db') as store:
        stub = _stub_checkpoint(store)
        with (
            patch('dashboard.loops.collect_quality', new=AsyncMock(side_effect=_count_tick)),
            patch('dashboard.loops._sleep_to_aligned_tick', new=yielding_noop_sleep),
            patch('dashboard.loops._CHECKPOINT_INTERVAL_SECONDS', 3600),
        ):
            await drive_loop_until(
                _quality_loop(store, DashboardConfig(project_root=tmp_path)), six_ticks,
            )

    assert six_ticks.is_set(), f'the loop ran only {ticks} ticks, so the bound below is vacuous'
    assert stub.call_count <= 1, f'{stub.call_count} checkpoints inside one 3600 s interval'
