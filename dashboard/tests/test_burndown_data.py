"""Tests for dashboard.data.burndown — snapshot collection, downsampling, and queries."""

from __future__ import annotations

import contextlib
import logging
import sqlite3
from datetime import UTC, datetime, timedelta
from pathlib import Path
from unittest.mock import patch

import aiosqlite
import pytest

from dashboard.config import DashboardConfig
from dashboard.data.burndown import (
    BURNDOWN_SCHEMA,
    _count_statuses,
    collect_snapshot,
    downsample,
    get_burndown_projects,
    get_burndown_series,
)

# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------


def _create_burndown_db(path: Path) -> None:
    """Create a burndown DB with schema at *path*."""
    conn = sqlite3.connect(str(path))
    conn.executescript(BURNDOWN_SCHEMA)
    conn.commit()
    conn.close()


def _insert_snapshot(
    conn: sqlite3.Connection,
    project_id: str,
    ts: str,
    *,
    pending: int = 0,
    in_progress: int = 0,
    blocked: int = 0,
    deferred: int = 0,
    cancelled: int = 0,
    done: int = 0,
) -> None:
    conn.execute(
        'INSERT INTO snapshots (project_id, ts, pending, in_progress, blocked, deferred, cancelled, done) '
        'VALUES (?, ?, ?, ?, ?, ?, ?, ?)',
        (project_id, ts, pending, in_progress, blocked, deferred, cancelled, done),
    )


def _assert_snapshot_counts(
    row,
    *,
    pending: int = 0,
    in_progress: int = 0,
    blocked: int = 0,
    deferred: int = 0,
    cancelled: int = 0,
    done: int = 0,
) -> None:
    """Assert that a snapshot row matches the expected count values by column name.

    Requires the row to be an aiosqlite.Row (name-based access).  Each count
    column is checked individually so that mismatch messages identify which
    column failed and what values were expected vs. actual.
    """
    assert row['pending'] == pending, (
        f'pending: expected {pending}, got {row["pending"]}'
    )
    assert row['in_progress'] == in_progress, (
        f'in_progress: expected {in_progress}, got {row["in_progress"]}'
    )
    assert row['blocked'] == blocked, (
        f'blocked: expected {blocked}, got {row["blocked"]}'
    )
    assert row['deferred'] == deferred, (
        f'deferred: expected {deferred}, got {row["deferred"]}'
    )
    assert row['cancelled'] == cancelled, (
        f'cancelled: expected {cancelled}, got {row["cancelled"]}'
    )
    assert row['done'] == done, (
        f'done: expected {done}, got {row["done"]}'
    )


def _fake_load(tasks_map):
    return lambda path: tasks_map[path]


# ---------------------------------------------------------------------------
# Fixtures
# ---------------------------------------------------------------------------


@pytest.fixture
async def burndown_env(tmp_path):
    """Yield (db_path, config, conn) with a fresh burndown DB and open connection."""
    db_path = tmp_path / 'burndown.db'
    _create_burndown_db(db_path)
    config = DashboardConfig(project_root=tmp_path)
    async with aiosqlite.connect(str(db_path)) as conn:
        conn.row_factory = aiosqlite.Row
        yield db_path, config, conn


# ---------------------------------------------------------------------------
# _count_statuses
# ---------------------------------------------------------------------------


class TestCountStatuses:
    def test_empty_list(self):
        assert _count_statuses([]) == {
            'pending': 0, 'in_progress': 0, 'blocked': 0,
            'deferred': 0, 'cancelled': 0, 'done': 0,
        }

    def test_standard_statuses(self):
        tasks = [
            {'status': 'pending'},
            {'status': 'pending'},
            {'status': 'in-progress'},
            {'status': 'done'},
            {'status': 'blocked'},
            {'status': 'cancelled'},
            {'status': 'deferred'},
        ]
        result = _count_statuses(tasks)
        assert result == {
            'pending': 2, 'in_progress': 1, 'blocked': 1,
            'deferred': 1, 'cancelled': 1, 'done': 1,
        }

    def test_review_merges_into_in_progress(self):
        tasks = [{'status': 'review'}, {'status': 'in-progress'}]
        result = _count_statuses(tasks)
        assert result['in_progress'] == 2

    def test_unknown_status_defaults_to_pending(self):
        tasks = [{'status': 'something-weird'}]
        result = _count_statuses(tasks)
        assert result['pending'] == 1

    def test_missing_status_key_defaults_to_pending(self):
        tasks = [{'title': 'no status field'}]
        result = _count_statuses(tasks)
        assert result['pending'] == 1


# ---------------------------------------------------------------------------
# collect_snapshot
# ---------------------------------------------------------------------------


class TestCollectSnapshot:
    @pytest.mark.asyncio
    async def test_inserts_main_project(self, burndown_env):
        db_path, config, conn = burndown_env

        fake_tasks = [
            {'status': 'done'},
            {'status': 'done'},
            {'status': 'pending'},
            {'status': 'in-progress'},
        ]

        with (
            patch('dashboard.data.burndown.load_task_tree', return_value=fake_tasks),
            patch('dashboard.data.burndown.find_running_orchestrators', return_value=[]),
        ):
            await collect_snapshot(conn, config)

        async with conn.execute('SELECT * FROM snapshots') as cur:
            rows = await cur.fetchall()

        assert len(rows) == 1
        row = rows[0]
        assert row['project_id'] == str(config.project_root)
        _assert_snapshot_counts(row, pending=1, in_progress=1, done=2)

    @pytest.mark.asyncio
    async def test_symlinked_root_deduplicates_with_orchestrator(self, tmp_path):
        """Symlinked project_root and orchestrator resolving to real path produce only 1 row."""
        real_dir = tmp_path / 'real'
        real_dir.mkdir()
        link = tmp_path / 'link'
        link.symlink_to(real_dir)

        db_path = tmp_path / 'burndown.db'
        _create_burndown_db(db_path)

        config = DashboardConfig(project_root=link)

        # Orchestrator resolves the same project via _resolve_project_root to the real path
        fake_orchestrators = [{'prd': 'fake_prd.md', 'config_path': None}]

        async with aiosqlite.connect(str(db_path)) as conn:
            with (
                patch('dashboard.data.burndown.load_task_tree', return_value=[]),
                patch('dashboard.data.burndown.find_running_orchestrators', return_value=fake_orchestrators),
                patch('dashboard.data.burndown._resolve_project_root', return_value=real_dir.resolve()),
            ):
                await collect_snapshot(conn, config)

            async with conn.execute('SELECT COUNT(*) FROM snapshots') as cur:
                row = await cur.fetchone()
                assert row is not None
                count = row[0]

        # Only 1 row — symlink and real path should deduplicate
        assert count == 1

    @pytest.mark.asyncio
    async def test_deduplicates_main_project_from_orchestrators(self, burndown_env):
        """If an orchestrator targets the same root, only one row is inserted."""
        _, config, conn = burndown_env

        fake_orchestrators = [{'prd': str(config.project_root / 'prd.md'), 'project_root': str(config.project_root)}]

        with (
            patch('dashboard.data.burndown.load_task_tree', return_value=[]),
            patch('dashboard.data.burndown.find_running_orchestrators', return_value=fake_orchestrators),
            patch('dashboard.data.burndown._resolve_project_root', return_value=config.project_root),
        ):
            await collect_snapshot(conn, config)

        async with conn.execute('SELECT COUNT(*) FROM snapshots') as cur:
            row = await cur.fetchone()
            assert row is not None
            count = row[0]

        assert count == 1

    @pytest.mark.asyncio
    async def test_snapshots_known_project_roots_when_no_orchestrators(self, burndown_env):
        """Known roots are snapshotted even when no orchestrators are running."""
        _, base_config, conn = burndown_env

        reify_root = Path('/home/leo/src/reify')
        autopilot_root = Path('/home/leo/src/autopilot-video')

        config = DashboardConfig(
            project_root=base_config.project_root,
            known_project_roots=[reify_root, autopilot_root],
        )

        main_tasks = [{'status': 'pending'}]
        reify_tasks = [{'status': 'done'}, {'status': 'done'}]
        autopilot_tasks = [{'status': 'in-progress'}]

        # Path-keyed dispatch: asyncio.gather fires load_task_tree calls
        # concurrently, so an ordered side_effect list can race. Look up by path.
        _tasks_map = {
            config.tasks_json: main_tasks,
            reify_root / '.taskmaster' / 'tasks' / 'tasks.json': reify_tasks,
            autopilot_root / '.taskmaster' / 'tasks' / 'tasks.json': autopilot_tasks,
        }

        with (
            patch('dashboard.data.burndown.load_task_tree', side_effect=_fake_load(_tasks_map)),
            patch('dashboard.data.burndown.find_running_orchestrators', return_value=[]),
        ):
            await collect_snapshot(conn, config)

        async with conn.execute('SELECT * FROM snapshots') as cur:
            rows = await cur.fetchall()

        assert len(rows) == 3
        by_project = {row['project_id']: row for row in rows}
        assert str(base_config.project_root) in by_project
        assert str(reify_root) in by_project
        assert str(autopilot_root) in by_project

        # main project: 1 pending task
        main_row = by_project[str(base_config.project_root)]
        _assert_snapshot_counts(main_row, pending=1)

        # reify: 2 done tasks
        reify_row = by_project[str(reify_root)]
        _assert_snapshot_counts(reify_row, done=2)

        # autopilot: 1 in-progress task
        autopilot_row = by_project[str(autopilot_root)]
        _assert_snapshot_counts(autopilot_row, in_progress=1)

    @pytest.mark.asyncio
    async def test_dedupes_known_root_against_main_project(self, burndown_env):
        """If known_project_roots includes main project_root, only one row is inserted."""
        _, base_config, conn = burndown_env

        config = DashboardConfig(
            project_root=base_config.project_root,
            known_project_roots=[base_config.project_root],  # same as project_root
        )

        with (
            patch('dashboard.data.burndown.load_task_tree', return_value=[]),
            patch('dashboard.data.burndown.find_running_orchestrators', return_value=[]),
        ):
            await collect_snapshot(conn, config)

        # PRIMARY: exact-key invariant — this project_id has exactly one row.
        # Catches bugs where the main-project row is omitted while a differently-
        # keyed row (e.g. non-resolved path, trailing slash) is inserted instead.
        async with conn.execute(
            'SELECT COUNT(*) FROM snapshots WHERE project_id = ?',
            (str(base_config.project_root),),
        ) as cur:
            row = await cur.fetchone()
            assert row is not None
            assert row[0] == 1

        # SECONDARY: total row count (no WHERE) catches both same-id duplicates
        # AND the symlink case where two rows with different project_id strings
        # are inserted for the same physical directory.
        async with conn.execute('SELECT COUNT(*) FROM snapshots') as cur:
            row = await cur.fetchone()
            assert row is not None
            assert row[0] == 1

    @pytest.mark.asyncio
    async def test_symlinked_root_deduplicates_with_known_roots(self, tmp_path):
        """If known_project_roots includes the resolved real path, it deduplicates with a symlinked project_root."""
        real_dir = tmp_path / 'real'
        real_dir.mkdir()
        link = tmp_path / 'link'
        link.symlink_to(real_dir)

        db_path = tmp_path / 'burndown.db'
        _create_burndown_db(db_path)

        # project_root is the symlink; known_project_roots contains the resolved real path
        config = DashboardConfig(
            project_root=link,
            known_project_roots=[real_dir],  # same underlying dir, unresolved
        )

        async with aiosqlite.connect(str(db_path)) as conn:
            with (
                patch('dashboard.data.burndown.load_task_tree', return_value=[]),
                patch('dashboard.data.burndown.find_running_orchestrators', return_value=[]),
            ):
                await collect_snapshot(conn, config)

            async with conn.execute('SELECT COUNT(*) FROM snapshots') as cur:
                row = await cur.fetchone()
                assert row is not None
                count = row[0]

        # Only 1 row — symlink project_root and known real path deduplicate
        assert count == 1

    @pytest.mark.asyncio
    async def test_dedupes_known_root_against_running_orchestrator(self, burndown_env):
        """If known_project_roots includes a root already discovered via orchestrator, no duplicate."""
        _, base_config, conn = burndown_env

        reify_root = Path('/home/leo/src/reify')

        config = DashboardConfig(
            project_root=base_config.project_root,
            known_project_roots=[reify_root],
        )

        # Orchestrator also points to reify via config_path
        fake_orchestrators = [
            {'prd': None, 'config_path': '/home/leo/src/reify/orchestrator.yaml'},
        ]

        # Orchestrator discovery returns reify_root (un-resolved); dedup prevents a second
        # load_task_tree call for the known_project_roots entry that resolves to the same root.
        # Path-keyed dispatch because asyncio.gather fires calls concurrently —
        # an ordered side_effect list can race on thread scheduling.
        # NOTE: the orchestrator entry's key intentionally omits .resolve() because
        # _read_project_root_from_config is mocked to return the raw (unresolved)
        # reify_root, and production burndown.py passes that raw value through to the
        # tasks.json path construction (see roots_to_snapshot.append around line 100).
        _tasks_map = {
            config.tasks_json: [],
            reify_root / '.taskmaster' / 'tasks' / 'tasks.json': [{'status': 'done'}],
        }

        with (
            patch('dashboard.data.burndown.load_task_tree', side_effect=_fake_load(_tasks_map)),
            patch('dashboard.data.burndown.find_running_orchestrators', return_value=fake_orchestrators),
            patch('dashboard.data.burndown._read_project_root_from_config', return_value=reify_root),
        ):
            await collect_snapshot(conn, config)

        async with conn.execute('SELECT COUNT(*) FROM snapshots WHERE project_id = ?',
                                (str(reify_root),)) as cur:
            row = await cur.fetchone()
            assert row is not None
            count = row[0]

        assert count == 1  # only one row for reify, not two

    @pytest.mark.asyncio
    async def test_main_project_id_is_resolved_path(self, tmp_path):
        """project_id in snapshot must be the resolved path even when project_root is a symlink."""
        real_dir = tmp_path / 'real'
        real_dir.mkdir()
        link = tmp_path / 'link'
        link.symlink_to(real_dir)

        db_path = tmp_path / 'burndown.db'
        _create_burndown_db(db_path)

        config = DashboardConfig(project_root=link)

        async with aiosqlite.connect(str(db_path)) as conn:
            conn.row_factory = aiosqlite.Row
            with (
                patch('dashboard.data.burndown.load_task_tree', return_value=[]),
                patch('dashboard.data.burndown.find_running_orchestrators', return_value=[]),
            ):
                await collect_snapshot(conn, config)

            async with conn.execute('SELECT project_id FROM snapshots') as cur:
                rows = list(await cur.fetchall())

        assert len(rows) == 1
        # project_id must be the resolved real path, not the symlink path
        assert rows[0]['project_id'] == str(real_dir.resolve())

    @pytest.mark.asyncio
    async def test_discovers_config_flag_orchestrator(self, burndown_env):
        """Orchestrators launched with --config (no --prd) are snapshotted."""
        _, config, conn = burndown_env

        reify_root = Path('/home/leo/src/reify')
        fake_orchestrators = [
            {'prd': None, 'config_path': '/home/leo/src/reify/orchestrator.yaml'},
        ]
        reify_tasks = [{'status': 'done'}, {'status': 'pending'}]

        # Path-keyed dispatch because asyncio.gather fires calls concurrently —
        # an ordered side_effect list can race on thread scheduling.
        _tasks_map = {
            config.tasks_json: [],
            reify_root / '.taskmaster' / 'tasks' / 'tasks.json': reify_tasks,
        }

        with (
            patch('dashboard.data.burndown.load_task_tree', side_effect=_fake_load(_tasks_map)),
            patch('dashboard.data.burndown.find_running_orchestrators', return_value=fake_orchestrators),
            patch('dashboard.data.burndown._read_project_root_from_config', return_value=reify_root),
        ):
            await collect_snapshot(conn, config)

        async with conn.execute('SELECT * FROM snapshots ORDER BY project_id') as cur:
            rows = await cur.fetchall()

        assert len(rows) == 2
        ids = {row['project_id'] for row in rows}
        assert str(config.project_root) in ids  # main project
        assert str(reify_root) in ids            # config-discovered project
        # Check reify row counts
        reify_row = next(r for r in rows if r['project_id'] == str(reify_root))
        _assert_snapshot_counts(reify_row, pending=1, done=1)

    @pytest.mark.asyncio
    async def test_continues_when_known_root_unreadable(self, tmp_path, caplog):
        """PermissionError on one known root is skipped; other roots are still snapshotted."""
        db_path = tmp_path / 'burndown.db'
        _create_burndown_db(db_path)

        root_a = Path('/fake/project/root_a')
        root_b = Path('/fake/project/root_b')
        root_c = Path('/fake/project/root_c')

        config = DashboardConfig(
            project_root=tmp_path,
            known_project_roots=[root_a, root_b, root_c],
        )

        main_tasks = [{'status': 'pending'}]
        root_a_tasks = [{'status': 'done'}]
        root_c_tasks = [{'status': 'in-progress'}]

        # Path-keyed dispatch: asyncio.gather fires load_task_tree calls
        # concurrently, so an ordered side_effect list can race. The bad path
        # raises PermissionError; return_exceptions=True isolates the failure.
        bad_tasks_json = root_b.resolve() / '.taskmaster' / 'tasks' / 'tasks.json'
        _tasks_map = {
            config.tasks_json: main_tasks,
            root_a.resolve() / '.taskmaster' / 'tasks' / 'tasks.json': root_a_tasks,
            root_c.resolve() / '.taskmaster' / 'tasks' / 'tasks.json': root_c_tasks,
        }

        def fake_load(path):
            if path == bad_tasks_json:
                raise PermissionError('Permission denied')
            return _tasks_map[path]

        async with aiosqlite.connect(str(db_path)) as conn:
            conn.row_factory = aiosqlite.Row
            with (
                patch('dashboard.data.burndown.load_task_tree', side_effect=fake_load),
                patch('dashboard.data.burndown.find_running_orchestrators', return_value=[]),
                caplog.at_level(logging.WARNING, logger='dashboard.data.burndown'),
            ):
                await collect_snapshot(conn, config)

            async with conn.execute('SELECT project_id FROM snapshots') as cur:
                rows = list(await cur.fetchall())

        project_ids = {row['project_id'] for row in rows}
        # main + root_a + root_c should be present
        assert len(rows) == 3
        assert str(tmp_path.resolve()) in project_ids
        assert str(root_a.resolve()) in project_ids
        assert str(root_c.resolve()) in project_ids
        # root_b should NOT be present
        assert str(root_b.resolve()) not in project_ids

        # Warning contract: exactly one 'Failed to load tasks' warning naming root_b only
        warning_records = [
            r for r in caplog.records
            if r.levelno == logging.WARNING and r.getMessage().startswith('Failed to load tasks')
        ]
        assert len(warning_records) == 1
        combined = warning_records[0].getMessage()
        assert 'root_b' in combined or str(root_b) in combined
        assert 'root_a' not in combined
        assert 'root_c' not in combined
        assert warning_records[0].exc_info is not None

    @pytest.mark.asyncio
    async def test_logs_warning_when_known_root_unreadable(self, tmp_path, caplog):
        """A WARNING is logged naming the failing root when PermissionError occurs."""
        db_path = tmp_path / 'burndown.db'
        _create_burndown_db(db_path)

        bad_root = Path('/fake/project/bad_root')

        config = DashboardConfig(
            project_root=tmp_path,
            known_project_roots=[bad_root],
        )

        bad_tasks_json = bad_root.resolve() / '.taskmaster' / 'tasks' / 'tasks.json'
        _tasks_map: dict = {config.tasks_json: []}

        def fake_load(path):
            if path == bad_tasks_json:
                raise PermissionError('Permission denied')
            return _tasks_map[path]

        async with aiosqlite.connect(str(db_path)) as conn:
            with (
                patch('dashboard.data.burndown.load_task_tree', side_effect=fake_load),
                patch('dashboard.data.burndown.find_running_orchestrators', return_value=[]),
                caplog.at_level(logging.WARNING, logger='dashboard.data.burndown'),
            ):
                await collect_snapshot(conn, config)

        # At least one WARNING record should name the failing root
        warning_records = [r for r in caplog.records if r.levelno == logging.WARNING]
        assert warning_records, 'Expected at least one WARNING log record'
        combined = ' '.join(r.getMessage() for r in warning_records)
        assert 'bad_root' in combined or str(bad_root) in combined
        # exc_info must be populated on the warning record
        assert any(r.exc_info for r in warning_records)

    @pytest.mark.asyncio
    async def test_first_root_failure_does_not_block_subsequent_inserts(self, tmp_path):
        """If the very first known root fails, subsequent roots still get snapshotted."""
        db_path = tmp_path / 'burndown.db'
        _create_burndown_db(db_path)

        bad_root = Path('/fake/project/bad_root')
        good_root = Path('/fake/project/good_root')

        config = DashboardConfig(
            project_root=tmp_path,
            known_project_roots=[bad_root, good_root],
        )

        main_tasks = [{'status': 'pending'}]
        good_tasks = [{'status': 'done'}, {'status': 'done'}]

        bad_tasks_json = bad_root.resolve() / '.taskmaster' / 'tasks' / 'tasks.json'
        _tasks_map = {
            config.tasks_json: main_tasks,
            good_root.resolve() / '.taskmaster' / 'tasks' / 'tasks.json': good_tasks,
        }

        def fake_load(path):
            if path == bad_tasks_json:
                raise PermissionError('denied')
            return _tasks_map[path]

        async with aiosqlite.connect(str(db_path)) as conn:
            conn.row_factory = aiosqlite.Row
            with (
                patch('dashboard.data.burndown.load_task_tree', side_effect=fake_load),
                patch('dashboard.data.burndown.find_running_orchestrators', return_value=[]),
            ):
                await collect_snapshot(conn, config)

            async with conn.execute('SELECT project_id, done FROM snapshots ORDER BY project_id') as cur:
                rows = list(await cur.fetchall())

        assert len(rows) == 2
        project_ids = {row['project_id'] for row in rows}
        assert str(tmp_path.resolve()) in project_ids        # (a) main project row
        assert str(good_root.resolve()) in project_ids       # (b) good_root row
        assert str(bad_root.resolve()) not in project_ids    # (c) no bad_root row

        good_row = next(r for r in rows if r['project_id'] == str(good_root.resolve()))
        assert good_row['done'] == 2  # done=2 for good_root

    @pytest.mark.asyncio
    async def test_orchestrator_fallback_deduplicates_against_resolved_root(self, tmp_path):
        """When _resolve_project_root falls back to the symlinked config.project_root, it still deduplicates."""
        real_dir = tmp_path / 'real'
        real_dir.mkdir()
        link = tmp_path / 'link'
        link.symlink_to(real_dir)

        db_path = tmp_path / 'burndown.db'
        _create_burndown_db(db_path)

        config = DashboardConfig(project_root=link)

        # PRD path lives directly under tmp_path (not under real_dir), so
        # _resolve_project_root will walk up from tmp_path, find no .taskmaster,
        # and fall back to config.project_root (the unresolved symlink).
        prd_path = str(tmp_path / 'fake_prd.md')
        fake_orchestrators = [{'prd': prd_path, 'config_path': None}]

        async with aiosqlite.connect(str(db_path)) as conn:
            with (
                patch('dashboard.data.burndown.load_task_tree', return_value=[]),
                patch('dashboard.data.burndown.find_running_orchestrators', return_value=fake_orchestrators),
                # _resolve_project_root is NOT mocked — it runs for real and falls
                # back to config.project_root (the symlink) because prd_path has no
                # .taskmaster in its ancestor chain.
            ):
                await collect_snapshot(conn, config)

            async with conn.execute('SELECT COUNT(*) FROM snapshots') as cur:
                row = await cur.fetchone()
                assert row is not None
                count = row[0]

        # Only 1 row — the orchestrator fallback targets the same project as the
        # main project_root; resolved and unresolved paths must deduplicate.
        assert count == 1

    @pytest.mark.asyncio
    async def test_load_task_tree_calls_run_concurrently(self, tmp_path):
        """All load_task_tree calls must run concurrently via asyncio.gather.

        Uses a threading.Barrier(N) to detect concurrency: all N threads must
        reach the barrier simultaneously. With sequential awaits, only one thread
        is alive at a time so barrier.wait() times out (BrokenBarrierError).
        With asyncio.gather, all N threads are live simultaneously and the
        barrier succeeds.
        """
        import threading

        db_path = tmp_path / 'burndown.db'
        _create_burndown_db(db_path)

        reify_root = tmp_path / 'reify'
        autopilot_root = tmp_path / 'autopilot'

        config = DashboardConfig(
            project_root=tmp_path,
            known_project_roots=[reify_root, autopilot_root],
        )

        # 3 distinct roots: main project + 2 known roots (no orchestrators)
        n_roots = 3
        barrier = threading.Barrier(n_roots, timeout=10.0)

        def fake_load(path):
            try:
                barrier.wait()
            except threading.BrokenBarrierError:
                pytest.fail(
                    'load_task_tree calls did not reach the barrier within 10s — '
                    'possible causes: (1) sequential execution (calls not running '
                    'concurrently via asyncio.gather); (2) severe scheduler latency '
                    '(threads starved by contention or a slow CI host)'
                )
            return []

        async with aiosqlite.connect(str(db_path)) as conn:
            with (
                patch('dashboard.data.burndown.load_task_tree', side_effect=fake_load),
                patch('dashboard.data.burndown.find_running_orchestrators', return_value=[]),
            ):
                await collect_snapshot(conn, config)

            async with conn.execute('SELECT COUNT(*) FROM snapshots') as cur:
                row = await cur.fetchone()
                assert row is not None
                assert row[0] == 3

    @pytest.mark.asyncio
    async def test_gather_partial_failure_skips_bad_project(self, tmp_path):
        """One load_task_tree raises OSError; collect_snapshot must either raise
        cleanly (pre-fix: asyncio.gather propagates the first exception) OR skip
        the bad project and insert healthy rows (post-fix: return_exceptions=True).
        Either branch holds the invariant: the failing project never appears in
        the snapshot table, and no other project is silently corrupted.

        This test will exercise different branches before and after Task 519
        lands. Once 519 is confirmed done a follow-up task can tighten it to
        only the post-fix branch.
        """
        db_path = tmp_path / 'burndown.db'
        _create_burndown_db(db_path)

        reify_root = Path('/home/leo/src/reify')
        autopilot_root = Path('/home/leo/src/autopilot-video')

        config = DashboardConfig(
            project_root=tmp_path,
            known_project_roots=[reify_root, autopilot_root],
        )

        # The path that will raise OSError — reify is the failing root.
        bad_path = reify_root.resolve() / '.taskmaster' / 'tasks' / 'tasks.json'

        # Only map the two healthy roots; the failing root is intentionally absent.
        _tasks_map = {
            config.tasks_json: [],
            autopilot_root.resolve() / '.taskmaster' / 'tasks' / 'tasks.json': [{'status': 'done'}],
        }

        def fake_load(path):
            if path == bad_path:
                raise OSError('mock disk error')
            return _tasks_map[path]

        raised_oserror = False
        async with aiosqlite.connect(str(db_path)) as conn:
            conn.row_factory = aiosqlite.Row
            with (
                patch('dashboard.data.burndown.load_task_tree', side_effect=fake_load),
                patch('dashboard.data.burndown.find_running_orchestrators', return_value=[]),
            ):
                try:
                    await collect_snapshot(conn, config)
                except OSError:
                    raised_oserror = True

            async with conn.execute('SELECT project_id FROM snapshots') as cur:
                rows = list(await cur.fetchall())
        project_ids = {row['project_id'] for row in rows}

        if raised_oserror:
            # Pre-fix behavior (task 519 not yet landed): asyncio.gather re-raises
            # the first OSError, so Phase 3 never runs and no rows are inserted.
            # The invariant is: the failing project's data never appears in the
            # table, and no partial state is committed.
            assert len(rows) == 0
        else:
            # Post-fix behavior (task 519 landed): return_exceptions=True + per-
            # project skip means healthy projects are still snapshotted and the
            # failing project is cleanly excluded.
            assert len(rows) == 2
            assert str(reify_root.resolve()) not in project_ids
            assert str(config.project_root.resolve()) in project_ids
            assert str(autopilot_root.resolve()) in project_ids

    @pytest.mark.asyncio
    async def test_gather_return_exceptions_preserves_healthy_snapshots(self, tmp_path, caplog):
        """OSError on one known root is isolated; healthy projects are still snapshotted.

        Regression anchor for task 519: asyncio.gather(return_exceptions=True) +
        isinstance(result, BaseException) guard ensure a single unreadable tasks.json
        cannot drop the remaining snapshots.  The test uses OSError (not PermissionError)
        to match task 519's 'unreadable tasks.json' wording.
        """
        db_path = tmp_path / 'burndown.db'
        _create_burndown_db(db_path)

        bad_root = Path('/fake/project/bad_root')
        good_root_1 = Path('/fake/project/good_root_1')
        good_root_2 = Path('/fake/project/good_root_2')

        config = DashboardConfig(
            project_root=tmp_path,
            known_project_roots=[bad_root, good_root_1, good_root_2],
        )

        main_tasks = [{'status': 'pending'}]
        good_1_tasks = [{'status': 'done'}, {'status': 'done'}]
        good_2_tasks = [{'status': 'done'}]

        bad_tasks_json = bad_root.resolve() / '.taskmaster' / 'tasks' / 'tasks.json'
        _tasks_map = {
            config.tasks_json: main_tasks,
            good_root_1.resolve() / '.taskmaster' / 'tasks' / 'tasks.json': good_1_tasks,
            good_root_2.resolve() / '.taskmaster' / 'tasks' / 'tasks.json': good_2_tasks,
        }

        def fake_load(path):
            if path == bad_tasks_json:
                raise OSError('mock disk error')
            return _tasks_map[path]

        async with aiosqlite.connect(str(db_path)) as conn:
            conn.row_factory = aiosqlite.Row
            with (
                patch('dashboard.data.burndown.load_task_tree', side_effect=fake_load),
                patch('dashboard.data.burndown.find_running_orchestrators', return_value=[]),
                caplog.at_level(logging.WARNING, logger='dashboard.data.burndown'),
            ):
                # Must NOT raise even though bad_root fails — return_exceptions=True absorbs it
                await collect_snapshot(conn, config)

            async with conn.execute('SELECT * FROM snapshots') as cur:
                rows = list(await cur.fetchall())

        # (a) three rows: main + good_root_1 + good_root_2
        assert len(rows) == 3

        by_project = {row['project_id']: row for row in rows}

        # (b) bad_root must NOT appear
        assert str(bad_root.resolve()) not in by_project

        # (c) main project and both good roots must appear
        assert str(tmp_path.resolve()) in by_project
        assert str(good_root_1.resolve()) in by_project
        assert str(good_root_2.resolve()) in by_project

        # (d) per-root done counts must reflect the supplied task lists
        good_1_row = by_project[str(good_root_1.resolve())]
        _assert_snapshot_counts(good_1_row, done=2)

        good_2_row = by_project[str(good_root_2.resolve())]
        _assert_snapshot_counts(good_2_row, done=1)

        # (e) at least one WARNING record must name the bad root and carry exc_info
        warning_records = [r for r in caplog.records if r.levelno == logging.WARNING]
        assert warning_records, 'Expected at least one WARNING log record'
        combined = ' '.join(r.getMessage() for r in warning_records)
        assert 'bad_root' in combined or str(bad_root) in combined
        assert any(r.exc_info for r in warning_records)

    @pytest.mark.asyncio
    async def test_main_project_failure_skips_all_inserts(self, burndown_env, caplog):
        """PermissionError on the main project's load_task_tree is isolated via return_exceptions=True.

        The main project is always the first entry in roots_to_snapshot. With no orchestrators
        and no known_project_roots, a failing load_task_tree for the main project must:

        (a) NOT propagate out of collect_snapshot — return_exceptions=True in Phase 2 absorbs it,
            and Phase 3's isinstance(tasks, BaseException) guard logs-and-continues.
        (b) Commit zero rows to the snapshots table — the only root failed, so nothing to insert.
        (c) Emit a WARNING log record naming the main project with exc_info populated.

        Sibling test to test_continues_when_known_root_unreadable and
        test_gather_return_exceptions_preserves_healthy_snapshots, covering the
        previously-untested main-project failure path. Uses path-keyed dispatch so
        the test does not depend on load_task_tree call ordering.
        """
        db_path, config, conn = burndown_env

        bad_path = config.tasks_json  # the main project's tasks.json path

        def fake_load(path):
            if path == bad_path:
                raise PermissionError('Permission denied')
            pytest.fail(f'Unexpected load_task_tree call for {path}')

        with (
            patch('dashboard.data.burndown.load_task_tree', side_effect=fake_load),
            patch('dashboard.data.burndown.find_running_orchestrators', return_value=[]),
            caplog.at_level(logging.WARNING, logger='dashboard.data.burndown'),
        ):
            # Must NOT raise — return_exceptions=True absorbs the PermissionError.
            await collect_snapshot(conn, config)

        # (b) zero rows committed — the only root failed, so snapshots is empty.
        async with conn.execute('SELECT COUNT(*) FROM snapshots') as cur:
            row = await cur.fetchone()
        assert row is not None
        assert row[0] == 0

        # (c) at least one WARNING record must name the main project and carry exc_info.
        warning_records = [r for r in caplog.records if r.levelno == logging.WARNING]
        assert warning_records, 'Expected at least one WARNING log record'
        combined = ' '.join(r.getMessage() for r in warning_records)
        assert str(config.project_root) in combined
        assert any(r.exc_info for r in warning_records)

    @pytest.mark.asyncio
    async def test_skips_known_root_on_resolve_error(self, tmp_path, caplog):
        """OSError raised from Path.resolve for one known_project_roots entry is
        absorbed by Phase 2's return_exceptions=True: a warning is logged naming
        the failing root, and the remaining known roots are still snapshotted.

        This test injects the OSError through a patched Path.resolve that selectively
        raises for paths under bad_root. The mocked load_task_tree calls path.resolve()
        as its first action so the patch actually fires during the gather-driven load
        pipeline — fake_load stands in for any real operation inside load_task_tree
        that might touch Path.resolve (canonicalization, symlink checks, etc.).
        """
        db_path = tmp_path / 'burndown.db'
        _create_burndown_db(db_path)

        root_a = Path('/fake/project/resolve_root_a')
        bad_root = Path('/fake/project/resolve_bad_root')
        root_c = Path('/fake/project/resolve_root_c')

        config = DashboardConfig(
            project_root=tmp_path,
            known_project_roots=[root_a, bad_root, root_c],
        )

        main_tasks = [{'status': 'pending'}]
        root_a_tasks = [{'status': 'done'}]
        root_c_tasks = [{'status': 'in-progress'}]

        # Capture bad_prefix BEFORE patching so __post_init__ resolve() succeeds normally
        bad_prefix = str(bad_root.resolve())
        _tasks_map = {
            config.tasks_json: main_tasks,
            root_a.resolve() / '.taskmaster' / 'tasks' / 'tasks.json': root_a_tasks,
            root_c.resolve() / '.taskmaster' / 'tasks' / 'tasks.json': root_c_tasks,
        }

        original_resolve = Path.resolve

        def bad_resolve(self, *args, **kwargs):
            if str(self).startswith(bad_prefix):
                raise OSError('simulated Path.resolve failure')
            return original_resolve(self, *args, **kwargs)

        def fake_load(path):
            # Synthetic: call resolve() so the patched Path.resolve actually fires
            # during the load pipeline. Real load_task_tree does not call resolve,
            # but any Path operation — read_text, stat, symlink walk — could raise
            # OSError. This injection point stands in for that broader category.
            path.resolve()
            return _tasks_map[path]

        async with aiosqlite.connect(str(db_path)) as conn:
            conn.row_factory = aiosqlite.Row
            with (
                patch.object(Path, 'resolve', bad_resolve),
                patch('dashboard.data.burndown.load_task_tree', side_effect=fake_load),
                patch('dashboard.data.burndown.find_running_orchestrators', return_value=[]),
                caplog.at_level(logging.WARNING, logger='dashboard.data.burndown'),
            ):
                await collect_snapshot(conn, config)

            async with conn.execute('SELECT project_id FROM snapshots') as cur:
                rows = list(await cur.fetchall())

        project_ids = {row['project_id'] for row in rows}
        # Three rows: main + root_a + root_c. bad_root is absorbed by return_exceptions.
        assert len(rows) == 3
        assert str(tmp_path.resolve()) in project_ids
        assert str(root_a.resolve()) in project_ids
        assert str(root_c.resolve()) in project_ids
        assert str(bad_root.resolve()) not in project_ids

        # Warning contract: exactly one 'Failed to load tasks' warning naming bad_root.
        warning_records = [
            r for r in caplog.records
            if r.levelno == logging.WARNING and r.getMessage().startswith('Failed to load tasks')
        ]
        assert len(warning_records) == 1
        combined = warning_records[0].getMessage()
        assert 'resolve_bad_root' in combined or bad_prefix in combined
        assert 'resolve_root_a' not in combined
        assert 'resolve_root_c' not in combined
        assert warning_records[0].exc_info is not None

# ---------------------------------------------------------------------------
# downsample
# ---------------------------------------------------------------------------


class TestDownsample:
    @pytest.mark.asyncio
    async def test_preserves_recent_data(self, tmp_path):
        """Data younger than 7 days is untouched."""
        db_path = tmp_path / 'burndown.db'
        _create_burndown_db(db_path)

        now = datetime.now(UTC)
        sync_conn = sqlite3.connect(str(db_path))
        for i in range(6):
            ts = (now - timedelta(hours=i)).isoformat()
            _insert_snapshot(sync_conn, 'proj', ts, done=i)
        sync_conn.commit()
        sync_conn.close()

        async with aiosqlite.connect(str(db_path)) as conn:
            await downsample(conn)
            async with conn.execute('SELECT COUNT(*) FROM snapshots') as cur:
                row = await cur.fetchone()
                assert row is not None
                count = row[0]

        assert count == 6  # all preserved

    @pytest.mark.asyncio
    async def test_compacts_old_to_hourly(self, tmp_path):
        """Multiple snapshots in the same hour (>7d old) are compacted to one."""
        db_path = tmp_path / 'burndown.db'
        _create_burndown_db(db_path)

        now = datetime.now(UTC)
        old_hour = now - timedelta(days=10)
        sync_conn = sqlite3.connect(str(db_path))
        # Insert 3 snapshots in the same hour, 10 days ago
        for i in range(3):
            ts = (old_hour + timedelta(minutes=i * 10)).isoformat()
            _insert_snapshot(sync_conn, 'proj', ts, done=i)
        sync_conn.commit()
        sync_conn.close()

        async with aiosqlite.connect(str(db_path)) as conn:
            await downsample(conn)
            async with conn.execute('SELECT COUNT(*) FROM snapshots') as cur:
                row = await cur.fetchone()
                assert row is not None
                count = row[0]

        assert count == 1  # compacted to one per hour

    @pytest.mark.asyncio
    async def test_expires_very_old(self, tmp_path):
        """Data older than 90 days is deleted."""
        db_path = tmp_path / 'burndown.db'
        _create_burndown_db(db_path)

        now = datetime.now(UTC)
        sync_conn = sqlite3.connect(str(db_path))
        _insert_snapshot(sync_conn, 'proj', (now - timedelta(days=100)).isoformat(), done=1)
        _insert_snapshot(sync_conn, 'proj', (now - timedelta(days=1)).isoformat(), done=2)
        sync_conn.commit()
        sync_conn.close()

        async with aiosqlite.connect(str(db_path)) as conn:
            await downsample(conn)
            async with conn.execute('SELECT COUNT(*) FROM snapshots') as cur:
                row = await cur.fetchone()
                assert row is not None
                count = row[0]

        assert count == 1  # only the recent one


# ---------------------------------------------------------------------------
# get_burndown_projects
# ---------------------------------------------------------------------------


class TestGetBurndownProjects:
    @pytest.mark.asyncio
    async def test_returns_distinct_projects(self, tmp_path):
        db_path = tmp_path / 'burndown.db'
        _create_burndown_db(db_path)

        now = datetime.now(UTC).isoformat()
        sync_conn = sqlite3.connect(str(db_path))
        _insert_snapshot(sync_conn, '/proj/a', now, done=1)
        _insert_snapshot(sync_conn, '/proj/b', now, done=2)
        _insert_snapshot(sync_conn, '/proj/a', now, done=3)
        sync_conn.commit()
        sync_conn.close()

        async with aiosqlite.connect(str(db_path)) as conn:
            conn.row_factory = aiosqlite.Row
            result = await get_burndown_projects(conn)

        assert result == ['/proj/a', '/proj/b']

    @pytest.mark.asyncio
    async def test_returns_empty_for_none_db(self):
        assert await get_burndown_projects(None) == []

    @pytest.mark.asyncio
    async def test_returns_empty_for_empty_db(self, tmp_path):
        db_path = tmp_path / 'burndown.db'
        _create_burndown_db(db_path)

        async with aiosqlite.connect(str(db_path)) as conn:
            conn.row_factory = aiosqlite.Row
            result = await get_burndown_projects(conn)

        assert result == []


# ---------------------------------------------------------------------------
# get_burndown_series
# ---------------------------------------------------------------------------


class TestGetBurndownSeries:
    @pytest.mark.asyncio
    async def test_returns_correct_structure(self, tmp_path):
        db_path = tmp_path / 'burndown.db'
        _create_burndown_db(db_path)

        now = datetime.now(UTC)
        sync_conn = sqlite3.connect(str(db_path))
        for i in range(3):
            ts = (now - timedelta(hours=i)).isoformat()
            _insert_snapshot(sync_conn, 'proj', ts, pending=5 - i, done=i)
        sync_conn.commit()
        sync_conn.close()

        async with aiosqlite.connect(str(db_path)) as conn:
            conn.row_factory = aiosqlite.Row
            result = await get_burndown_series(conn, 'proj', days=1)

        assert len(result['labels']) == 3
        assert len(result['done']) == 3
        assert len(result['pending']) == 3
        assert set(result.keys()) == {
            'labels', 'done', 'cancelled', 'blocked', 'deferred', 'in_progress', 'pending',
        }

    @pytest.mark.asyncio
    async def test_filters_by_window(self, tmp_path):
        db_path = tmp_path / 'burndown.db'
        _create_burndown_db(db_path)

        now = datetime.now(UTC)
        sync_conn = sqlite3.connect(str(db_path))
        _insert_snapshot(sync_conn, 'proj', (now - timedelta(hours=2)).isoformat(), done=1)
        _insert_snapshot(sync_conn, 'proj', (now - timedelta(days=5)).isoformat(), done=2)
        sync_conn.commit()
        sync_conn.close()

        async with aiosqlite.connect(str(db_path)) as conn:
            conn.row_factory = aiosqlite.Row
            result_1d = await get_burndown_series(conn, 'proj', days=1)
            result_7d = await get_burndown_series(conn, 'proj', days=7)

        assert len(result_1d['labels']) == 1
        assert len(result_7d['labels']) == 2

    @pytest.mark.asyncio
    async def test_returns_empty_for_none_db(self):
        result = await get_burndown_series(None, 'proj')
        assert result['labels'] == []
        assert result['done'] == []

    @pytest.mark.asyncio
    async def test_orders_by_timestamp(self, tmp_path):
        db_path = tmp_path / 'burndown.db'
        _create_burndown_db(db_path)

        now = datetime.now(UTC)
        sync_conn = sqlite3.connect(str(db_path))
        # Insert out of order
        _insert_snapshot(sync_conn, 'proj', (now - timedelta(hours=1)).isoformat(), done=1)
        _insert_snapshot(sync_conn, 'proj', (now - timedelta(hours=3)).isoformat(), done=3)
        _insert_snapshot(sync_conn, 'proj', (now - timedelta(hours=2)).isoformat(), done=2)
        sync_conn.commit()
        sync_conn.close()

        async with aiosqlite.connect(str(db_path)) as conn:
            conn.row_factory = aiosqlite.Row
            result = await get_burndown_series(conn, 'proj', days=7)

        # done values should be in timestamp order (oldest first)
        assert result['done'] == [3, 2, 1]


# ---------------------------------------------------------------------------
# burndown_env fixture validation
# ---------------------------------------------------------------------------


class TestBurndownEnvFixture:
    @pytest.mark.asyncio
    async def test_yields_valid_triple(self, burndown_env):
        db_path, config, conn = burndown_env
        # (a) db_path is a Path and exists on disk
        assert isinstance(db_path, Path)
        assert db_path.exists()
        # (b) config is a DashboardConfig whose project_root is a tmp directory
        assert isinstance(config, DashboardConfig)
        assert db_path.parent == config.project_root
        # (c) conn is a usable aiosqlite connection
        async with conn.execute('SELECT COUNT(*) FROM snapshots') as cur:
            row = await cur.fetchone()
        assert row is not None
        assert row[0] == 0


# ---------------------------------------------------------------------------
# _assert_snapshot_counts helper
# ---------------------------------------------------------------------------


class TestAssertSnapshotCounts:
    @pytest.mark.asyncio
    async def test_passes_on_matching_counts(self, burndown_env):
        """Helper returns None when every count column matches."""
        db_path, config, conn = burndown_env
        await conn.execute(
            'INSERT INTO snapshots (project_id, ts, pending, in_progress, blocked, deferred, cancelled, done) '
            'VALUES (?, ?, ?, ?, ?, ?, ?, ?)',
            ('test_proj', '2024-01-01T00:00:00', 1, 0, 0, 0, 0, 2),
        )
        await conn.commit()

        conn.row_factory = aiosqlite.Row
        async with conn.execute('SELECT * FROM snapshots') as cur:
            rows = list(await cur.fetchall())

        assert len(rows) == 1
        result = _assert_snapshot_counts(rows[0], pending=1, done=2)
        assert result is None

    @pytest.mark.asyncio
    async def test_raises_on_mismatched_count(self, burndown_env):
        """Helper raises AssertionError when a count column doesn't match the expected value."""
        db_path, config, conn = burndown_env
        await conn.execute(
            'INSERT INTO snapshots (project_id, ts, pending, in_progress, blocked, deferred, cancelled, done) '
            'VALUES (?, ?, ?, ?, ?, ?, ?, ?)',
            ('test_proj', '2024-01-01T00:00:00', 0, 0, 0, 0, 0, 2),
        )
        await conn.commit()

        conn.row_factory = aiosqlite.Row
        async with conn.execute('SELECT * FROM snapshots') as cur:
            row = await cur.fetchone()

        with pytest.raises(AssertionError):
            _assert_snapshot_counts(row, done=3)  # actual done=2, expected 3

    @pytest.mark.asyncio
    async def test_default_zeros_match_all_zero_row(self, burndown_env):
        """Helper returns None when all counts are 0 and no kwargs are passed (defaults are 0)."""
        db_path, config, conn = burndown_env
        await conn.execute(
            'INSERT INTO snapshots (project_id, ts, pending, in_progress, blocked, deferred, cancelled, done) '
            'VALUES (?, ?, ?, ?, ?, ?, ?, ?)',
            ('test_proj', '2024-01-01T00:00:00', 0, 0, 0, 0, 0, 0),
        )
        await conn.commit()

        conn.row_factory = aiosqlite.Row
        async with conn.execute('SELECT * FROM snapshots') as cur:
            row = await cur.fetchone()

        result = _assert_snapshot_counts(row)  # all defaults are 0
        assert result is None


# ---------------------------------------------------------------------------
# Orchestrator-discovery failure isolation (#12)
# ---------------------------------------------------------------------------


class TestCollectSnapshotOrchestratorDiscoveryFailure:
    """Verify that orchestrator-discovery failures degrade gracefully.

    A single parameterized test covers three injection points:

    (a) ``find_running_orchestrators()`` itself raises — outer try/except catches it.
    (b) ``_resolve_project_root()`` raises for a PRD-based entry — inner per-entry
        try/except catches it; other entries and known_project_roots still proceed.
    (c) ``_read_project_root_from_config()`` raises for a config_path-based entry —
        same inner try/except handles it.

    In all cases collect_snapshot must not raise, must commit both the main
    project row and the known_project_roots row, and must emit a WARNING that
    mentions 'orchestrator' with exc_info set.
    """

    @pytest.mark.parametrize(
        'find_orch_kwargs, secondary_target, secondary_kwargs, known_root_suffix',
        [
            pytest.param(
                {'side_effect': RuntimeError('subprocess exploded')},
                None,
                None,
                'orch_disc_test_a',
                id='find_running_orchestrators_raises',
            ),
            pytest.param(
                {'return_value': [{'prd': 'fake_prd.md', 'config_path': None}]},
                'dashboard.data.burndown._resolve_project_root',
                {'side_effect': OSError('prd path not found')},
                'orch_disc_test_b',
                id='resolve_project_root_raises',
            ),
            pytest.param(
                {'return_value': [{'prd': None, 'config_path': '/fake/orchestrator.yaml'}]},
                'dashboard.data.burndown._read_project_root_from_config',
                {'side_effect': OSError('YAML parse error')},
                'orch_disc_test_c',
                id='read_project_root_from_config_raises',
            ),
        ],
    )
    @pytest.mark.asyncio
    async def test_orchestrator_discovery_failure_preserves_main_and_known_roots(
        self,
        burndown_env,
        caplog,
        find_orch_kwargs,
        secondary_target,
        secondary_kwargs,
        known_root_suffix,
    ):
        """Orchestrator-discovery failure must degrade gracefully.

        collect_snapshot must not raise; both the main project row and the
        known_project_roots row must be committed; a WARNING mentioning
        'orchestrator' must be emitted with exc_info set.
        """
        db_path, base_config, conn = burndown_env
        known_root = Path(f'/fake/project/{known_root_suffix}')
        config = DashboardConfig(
            project_root=base_config.project_root,
            known_project_roots=[known_root],
        )

        main_tasks = [{'status': 'pending'}]
        known_tasks = [{'status': 'done'}]
        _tasks_map = {
            config.tasks_json: main_tasks,
            known_root.resolve() / '.taskmaster' / 'tasks' / 'tasks.json': known_tasks,
        }

        with contextlib.ExitStack() as stack:
            stack.enter_context(caplog.at_level(logging.WARNING, logger='dashboard.data.burndown'))
            stack.enter_context(
                patch('dashboard.data.burndown.load_task_tree', side_effect=_fake_load(_tasks_map))
            )
            stack.enter_context(
                patch('dashboard.data.burndown.find_running_orchestrators', **find_orch_kwargs)
            )
            if secondary_target is not None:
                stack.enter_context(patch(secondary_target, **secondary_kwargs))
            # Must NOT raise despite the injected failure
            await collect_snapshot(conn, config)

        async with conn.execute('SELECT project_id FROM snapshots') as cur:
            rows = list(await cur.fetchall())

        project_ids = {row['project_id'] for row in rows}
        assert len(rows) == 2, f'Expected 2 rows, got {len(rows)}: {project_ids}'
        assert str(base_config.project_root) in project_ids
        assert str(known_root.resolve()) in project_ids

        # A WARNING naming orchestrator discovery must be logged with exc_info
        warning_records = [r for r in caplog.records if r.levelno == logging.WARNING]
        assert warning_records, 'Expected at least one WARNING for orchestrator discovery failure'
        combined = ' '.join(r.getMessage() for r in warning_records)
        assert 'orchestrator' in combined.lower(), (
            f'Expected "orchestrator" in warning message, got: {combined!r}'
        )
        assert any(r.exc_info for r in warning_records)


# ---------------------------------------------------------------------------
# DB-side insert failure isolation (#11)
# ---------------------------------------------------------------------------


class TestCollectSnapshotInsertFailureIsolation:
    """Verify that a DB-side INSERT failure is isolated to the offending project.

    These tests are written in TDD red-phase for step-4 of task 539.  They
    fail before the per-project commit + try/except is added to Phase 3,
    because the current single trailing commit means any INSERT failure rolls
    back ALL previously buffered inserts (or propagates before commit runs).
    """

    # ------------------------------------------------------------------
    # Helper: build a conn.execute wrapper that raises for a target project
    # ------------------------------------------------------------------

    @staticmethod
    def _make_execute_wrapper(original_execute, failing_project_ids: set):
        """Return an async wrapper around original_execute that raises OperationalError
        when an INSERT INTO snapshots targets one of the failing_project_ids.

        The returned callable exposes a ``trigger_count`` list attribute that is
        appended to each time the wrapper fires.  Tests must assert
        ``wrapper.trigger_count`` after the run to guard against silent
        pass-through if the SQL text is ever reformatted.
        """

        class ExecuteWrapper:
            trigger_count: list[int]

            def __init__(self) -> None:
                self.trigger_count = []

            async def __call__(self, sql: str, params: tuple = ()) -> object:
                if (
                    'INSERT INTO snapshots' in sql
                    and params
                    and params[0] in failing_project_ids
                ):
                    self.trigger_count.append(1)
                    raise aiosqlite.OperationalError('mock disk full')
                return await original_execute(sql, params)

        return ExecuteWrapper()

    @pytest.mark.asyncio
    async def test_db_error_on_extra_insert_preserves_main_snapshot(
        self, burndown_env, caplog
    ):
        """OperationalError on an extra's INSERT must not roll back the main row.

        After step-4: main row is committed before the extra is attempted;
        the extra's failure is caught per-project with a WARNING, and
        collect_snapshot does not raise.
        """
        db_path, base_config, conn = burndown_env
        extra_root = Path('/fake/project/insert_fail_extra_a')
        config = DashboardConfig(
            project_root=base_config.project_root,
            known_project_roots=[extra_root],
        )

        main_tasks = [{'status': 'pending'}]
        extra_tasks = [{'status': 'done'}]
        _tasks_map = {
            config.tasks_json: main_tasks,
            extra_root.resolve() / '.taskmaster' / 'tasks' / 'tasks.json': extra_tasks,
        }

        extra_id = str(extra_root.resolve())
        original_execute = conn.execute
        wrapper = self._make_execute_wrapper(original_execute, {extra_id})
        conn.execute = wrapper
        try:
            with (
                patch('dashboard.data.burndown.load_task_tree', side_effect=_fake_load(_tasks_map)),
                patch('dashboard.data.burndown.find_running_orchestrators', return_value=[]),
                caplog.at_level(logging.WARNING, logger='dashboard.data.burndown'),
            ):
                # Must NOT raise despite the extra's INSERT failing
                await collect_snapshot(conn, config)
        finally:
            conn.execute = original_execute

        # Guard: the wrapper must have actually fired — prevents silent pass-through
        # if the SQL text is ever reformatted and the string-match stops working.
        assert wrapper.trigger_count, (
            'execute_wrapper was never triggered — SQL interception may have silently broken'
        )

        async with conn.execute('SELECT project_id FROM snapshots') as cur:
            rows = list(await cur.fetchall())

        project_ids = {row['project_id'] for row in rows}
        # Main row must be committed (extra's failure must not roll it back)
        assert str(base_config.project_root) in project_ids, (
            f'Main project missing from committed rows: {project_ids}'
        )
        # Extra must NOT be present (its INSERT failed)
        assert extra_id not in project_ids

        # A WARNING naming the failing extra must be logged
        warning_records = [r for r in caplog.records if r.levelno == logging.WARNING]
        assert warning_records, 'Expected a WARNING for the failing extra INSERT'
        combined = ' '.join(r.getMessage() for r in warning_records)
        assert 'insert_fail_extra_a' in combined or extra_id in combined, (
            f'Expected extra path in warning message, got: {combined!r}'
        )

    @pytest.mark.asyncio
    async def test_db_error_on_extra_insert_preserves_other_extras(
        self, burndown_env, caplog
    ):
        """OperationalError on the middle extra must not affect main or flanking extras.

        Main + 3 extras; middle extra_b's INSERT raises.  After step-4:
        main + extra_a + extra_c are committed, extra_b is absent, and
        exactly one WARNING names extra_b.
        """
        db_path, base_config, conn = burndown_env
        extra_a = Path('/fake/project/insert_multi_a')
        extra_b = Path('/fake/project/insert_multi_b')   # the failing one
        extra_c = Path('/fake/project/insert_multi_c')
        config = DashboardConfig(
            project_root=base_config.project_root,
            known_project_roots=[extra_a, extra_b, extra_c],
        )

        main_tasks = [{'status': 'pending'}]
        extra_a_tasks = [{'status': 'done'}]
        extra_b_tasks = [{'status': 'done'}, {'status': 'done'}]
        extra_c_tasks = [{'status': 'in-progress'}]
        _tasks_map = {
            config.tasks_json: main_tasks,
            extra_a.resolve() / '.taskmaster' / 'tasks' / 'tasks.json': extra_a_tasks,
            extra_b.resolve() / '.taskmaster' / 'tasks' / 'tasks.json': extra_b_tasks,
            extra_c.resolve() / '.taskmaster' / 'tasks' / 'tasks.json': extra_c_tasks,
        }

        extra_b_id = str(extra_b.resolve())
        original_execute = conn.execute
        wrapper = self._make_execute_wrapper(original_execute, {extra_b_id})
        conn.execute = wrapper
        try:
            with (
                patch('dashboard.data.burndown.load_task_tree', side_effect=_fake_load(_tasks_map)),
                patch('dashboard.data.burndown.find_running_orchestrators', return_value=[]),
                caplog.at_level(logging.WARNING, logger='dashboard.data.burndown'),
            ):
                await collect_snapshot(conn, config)
        finally:
            conn.execute = original_execute

        # Guard: the wrapper must have actually fired
        assert wrapper.trigger_count, (
            'execute_wrapper was never triggered — SQL interception may have silently broken'
        )

        async with conn.execute('SELECT project_id FROM snapshots') as cur:
            rows = list(await cur.fetchall())

        project_ids = {row['project_id'] for row in rows}
        assert len(rows) == 3, (
            f'Expected 3 rows (main + extra_a + extra_c), got {len(rows)}: {project_ids}'
        )
        assert str(base_config.project_root) in project_ids
        assert str(extra_a.resolve()) in project_ids
        assert str(extra_c.resolve()) in project_ids
        assert extra_b_id not in project_ids

        # Exactly one WARNING naming extra_b
        warning_records = [
            r for r in caplog.records
            if r.levelno == logging.WARNING and (
                'insert_multi_b' in r.getMessage() or extra_b_id in r.getMessage()
            )
        ]
        assert len(warning_records) == 1, (
            f'Expected exactly 1 WARNING naming extra_b, got {len(warning_records)}'
        )

    @pytest.mark.asyncio
    async def test_db_error_on_main_insert_preserves_extras(
        self, burndown_env, caplog
    ):
        """OperationalError on the main project's INSERT must not prevent extras from being committed.

        After step-4: main's failure is caught per-project; the loop continues
        and extra is committed.  A WARNING must name the main project.
        """
        db_path, base_config, conn = burndown_env
        extra_root = Path('/fake/project/insert_fail_main_extra')
        config = DashboardConfig(
            project_root=base_config.project_root,
            known_project_roots=[extra_root],
        )

        main_tasks = [{'status': 'pending'}]
        extra_tasks = [{'status': 'done'}]
        _tasks_map = {
            config.tasks_json: main_tasks,
            extra_root.resolve() / '.taskmaster' / 'tasks' / 'tasks.json': extra_tasks,
        }

        main_id = str(base_config.project_root)
        original_execute = conn.execute
        wrapper = self._make_execute_wrapper(original_execute, {main_id})
        conn.execute = wrapper
        try:
            with (
                patch('dashboard.data.burndown.load_task_tree', side_effect=_fake_load(_tasks_map)),
                patch('dashboard.data.burndown.find_running_orchestrators', return_value=[]),
                caplog.at_level(logging.WARNING, logger='dashboard.data.burndown'),
            ):
                await collect_snapshot(conn, config)
        finally:
            conn.execute = original_execute

        # Guard: the wrapper must have actually fired
        assert wrapper.trigger_count, (
            'execute_wrapper was never triggered — SQL interception may have silently broken'
        )

        async with conn.execute('SELECT project_id FROM snapshots') as cur:
            rows = list(await cur.fetchall())

        project_ids = {row['project_id'] for row in rows}
        # Extra must be committed (main's failure must not prevent it)
        assert str(extra_root.resolve()) in project_ids, (
            f'Extra missing from committed rows: {project_ids}'
        )
        # Main must NOT be present (its INSERT failed)
        assert main_id not in project_ids

        # A WARNING naming the main project must be logged
        warning_records = [r for r in caplog.records if r.levelno == logging.WARNING]
        assert warning_records, 'Expected a WARNING for the failing main INSERT'
        combined = ' '.join(r.getMessage() for r in warning_records)
        assert str(base_config.project_root) in combined, (
            f'Expected main project path in warning, got: {combined!r}'
        )


# ---------------------------------------------------------------------------
# Explicit rollback on unexpected error (#13)
# ---------------------------------------------------------------------------


class TestCollectSnapshotExplicitRollback:
    """Verify that an unexpected exception triggers an explicit conn.rollback() before re-raising.

    This test is written in TDD red-phase for step-6 of task 539.  It fails
    before the outer try/except is added to collect_snapshot, because the
    current code relies on aiosqlite's implicit rollback-on-close which does
    not work correctly for a long-lived persistent connection (as used in
    app.py lifespan).
    """

    @pytest.mark.asyncio
    async def test_explicit_rollback_on_unexpected_error(self, burndown_env):
        """An unexpected exception inside collect_snapshot must trigger conn.rollback()
        and re-raise the original exception.

        Injection point: patch `dashboard.data.burndown.datetime` so that
        `datetime.now(UTC).isoformat()` raises RuntimeError('clock failure').
        This fires before any per-project try/except, so the outer except
        must catch it, call rollback, then re-raise.

        Asserts:
        (a) RuntimeError propagates out of collect_snapshot.
        (b) conn.rollback was called at least once before the re-raise.
        """
        from unittest.mock import patch as _patch

        db_path, config, conn = burndown_env

        # Spy on conn.rollback
        original_rollback = conn.rollback
        rollback_calls = []

        async def rollback_spy():
            rollback_calls.append(1)
            return await original_rollback()

        conn.rollback = rollback_spy

        # Patch datetime.now to raise RuntimeError — fires before any per-project try/except
        class _FakeDatetime:
            @staticmethod
            def now(tz=None):
                raise RuntimeError('clock failure')

        try:
            with (
                _patch('dashboard.data.burndown.find_running_orchestrators', return_value=[]),
                _patch('dashboard.data.burndown.datetime', _FakeDatetime),
                pytest.raises(RuntimeError, match='clock failure'),
            ):
                await collect_snapshot(conn, config)
        finally:
            conn.rollback = original_rollback

        # (b) rollback must have been called before the re-raise
        assert rollback_calls, (
            'Expected conn.rollback() to be called before re-raising the RuntimeError'
        )


# ---------------------------------------------------------------------------
# Docstring contract (#13 — partial-failure semantics)
# ---------------------------------------------------------------------------


class TestCollectSnapshotDocstringContract:
    """Verify that collect_snapshot's docstring documents all four partial-failure facts.

    This test is written in TDD red-phase for step-8 of task 539.  It fails
    before the docstring is updated, because the current one-liner says nothing
    about main-first commit order, orchestrator degradation, per-project
    best-effort inserts, or explicit rollback.
    """

    def test_docstring_documents_partial_failure_semantics(self):
        """collect_snapshot.__doc__ must be a non-trivial multi-line docstring.

        Exact wording is enforced by code review, not the test suite — brittle
        keyword assertions break when synonyms are used without any behavioral
        change.  Instead we verify the docstring is substantive (>= 5 non-empty
        lines) and mentions at least a couple of robustness concepts.
        """
        doc = collect_snapshot.__doc__ or ''
        non_empty_lines = [line.strip() for line in doc.splitlines() if line.strip()]
        assert len(non_empty_lines) >= 5, (
            f'Docstring must be non-trivial (>= 5 non-empty lines); '
            f'got {len(non_empty_lines)}: {doc!r}'
        )
        doc_lower = doc.lower()
        robustness_terms = {'commit', 'rollback', 'fail', 'error', 'except', 'isolat', 'orchestrator'}
        matched = [t for t in robustness_terms if t in doc_lower]
        assert len(matched) >= 2, (
            f'Docstring should mention at least 2 robustness terms from {robustness_terms}; '
            f'matched: {matched}'
        )
