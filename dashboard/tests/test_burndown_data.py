"""Tests for dashboard.data.burndown — snapshot collection, downsampling, and queries."""

from __future__ import annotations

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
            rows = list(await cur.fetchall())

        assert len(rows) == 1
        row = rows[0]
        # row: id, project_id, ts, pending, in_progress, blocked, deferred, cancelled, done
        assert row[1] == str(config.project_root.resolve())  # project_id
        assert row[3] == 1   # pending
        assert row[4] == 1   # in_progress
        assert row[5] == 0   # blocked
        assert row[6] == 0   # deferred
        assert row[7] == 0   # cancelled
        assert row[8] == 2   # done

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
        db_path, config, conn = burndown_env

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
        db_path, base_config, conn = burndown_env

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
            reify_root.resolve() / '.taskmaster' / 'tasks' / 'tasks.json': reify_tasks,
            autopilot_root.resolve() / '.taskmaster' / 'tasks' / 'tasks.json': autopilot_tasks,
        }

        def fake_load(path):
            return _tasks_map[path]

        with (
            patch('dashboard.data.burndown.load_task_tree', side_effect=fake_load),
            patch('dashboard.data.burndown.find_running_orchestrators', return_value=[]),
        ):
            await collect_snapshot(conn, config)

        async with conn.execute('SELECT * FROM snapshots') as cur:
            rows = list(await cur.fetchall())

        # row: id, project_id, ts, pending, in_progress, blocked, deferred, cancelled, done
        assert len(rows) == 3
        by_project = {row[1]: row for row in rows}
        assert str(base_config.project_root.resolve()) in by_project
        assert str(reify_root.resolve()) in by_project
        assert str(autopilot_root.resolve()) in by_project

        # main project: 1 pending task
        main_row = by_project[str(base_config.project_root.resolve())]
        assert main_row[3] == 1  # pending
        assert main_row[4] == 0  # in_progress
        assert main_row[5] == 0  # blocked
        assert main_row[6] == 0  # deferred
        assert main_row[7] == 0  # cancelled
        assert main_row[8] == 0  # done

        # reify: 2 done tasks
        reify_row = by_project[str(reify_root.resolve())]
        assert reify_row[3] == 0  # pending
        assert reify_row[4] == 0  # in_progress
        assert reify_row[5] == 0  # blocked
        assert reify_row[6] == 0  # deferred
        assert reify_row[7] == 0  # cancelled
        assert reify_row[8] == 2  # done

        # autopilot: 1 in-progress task
        autopilot_row = by_project[str(autopilot_root.resolve())]
        assert autopilot_row[3] == 0  # pending
        assert autopilot_row[4] == 1  # in_progress
        assert autopilot_row[5] == 0  # blocked
        assert autopilot_row[6] == 0  # deferred
        assert autopilot_row[7] == 0  # cancelled
        assert autopilot_row[8] == 0  # done

    @pytest.mark.asyncio
    async def test_dedupes_known_root_against_main_project(self, burndown_env):
        """If known_project_roots includes main project_root, only one row is inserted."""
        db_path, base_config, conn = burndown_env

        config = DashboardConfig(
            project_root=base_config.project_root,
            known_project_roots=[base_config.project_root],  # same as project_root
        )

        with (
            patch('dashboard.data.burndown.load_task_tree', return_value=[]),
            patch('dashboard.data.burndown.find_running_orchestrators', return_value=[]),
        ):
            await collect_snapshot(conn, config)

        # Total row count (no WHERE) catches both same-id duplicates AND the
        # symlink case where two rows with different project_id strings are
        # inserted for the same physical directory.
        async with conn.execute('SELECT COUNT(*) FROM snapshots') as cur:
            row = await cur.fetchone()
            assert row is not None
            count = row[0]

        assert count == 1

    @pytest.mark.asyncio
    async def test_symlinked_root_deduplicates_with_known_roots(self, tmp_path):
        """If known_project_roots includes the resolved real path, it deduplicates with a symlinked project_root."""
        real_dir = tmp_path / 'real'
        real_dir.mkdir()
        link = tmp_path / 'link'
        link.symlink_to(real_dir)

        db_path = tmp_path / 'burndown.db'
        _create_burndown_db(db_path)

        from dashboard.config import DashboardConfig
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
        db_path, base_config, conn = burndown_env

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
        _tasks_map = {
            config.tasks_json: [],
            reify_root / '.taskmaster' / 'tasks' / 'tasks.json': [{'status': 'done'}],
        }

        def fake_load(path):
            return _tasks_map[path]

        with (
            patch('dashboard.data.burndown.load_task_tree', side_effect=fake_load),
            patch('dashboard.data.burndown.find_running_orchestrators', return_value=fake_orchestrators),
            patch('dashboard.data.burndown._read_project_root_from_config', return_value=reify_root),
        ):
            await collect_snapshot(conn, config)

        async with conn.execute('SELECT COUNT(*) FROM snapshots WHERE project_id = ?',
                                (str(reify_root.resolve()),)) as cur:
            row = await cur.fetchone()
            assert row is not None
            count = row[0]

        assert count == 1  # only one row for reify, not two

    @pytest.mark.asyncio
    async def test_propagates_load_task_tree_error_for_known_root(self, tmp_path):
        """PermissionError on a known-root load_task_tree call propagates out of collect_snapshot.

        This pins option (a) behaviour: collect_snapshot has no per-root try/except, so
        an error on any load_task_tree call propagates uncaught (asyncio.gather raises
        the first exception). Because Phase 3 (sequential inserts) runs only after gather
        succeeds, and conn.commit() is at the very end of the function, no rows are
        committed — the operation is atomic in the failure case.

        When the deferred robustness task lands (option (b)), this test's assertions should
        flip to: no exception raised, exactly one row for the main project, zero rows for the
        failing known root.
        """
        db_path = tmp_path / 'burndown.db'
        _create_burndown_db(db_path)

        known_root = Path('/home/leo/src/reify')
        config = DashboardConfig(
            project_root=tmp_path,
            known_project_roots=[known_root],
        )

        # Path-keyed dispatch: asyncio.gather fires load_task_tree calls concurrently,
        # so an ordered side_effect list can race. Match by path so the known-root
        # call is the one that raises, regardless of scheduling order.
        def fake_load(path):
            if path == config.tasks_json:
                return []
            raise PermissionError('mocked permission denied')

        async with aiosqlite.connect(str(db_path)) as conn:
            with (
                patch('dashboard.data.burndown.load_task_tree', side_effect=fake_load),
                patch('dashboard.data.burndown.find_running_orchestrators', return_value=[]),
            ):
                with pytest.raises(PermissionError):
                    await collect_snapshot(conn, config)

        # Verify no rows were durably committed. Use a fresh connection so only
        # committed data is visible — the same connection would see its own
        # uncommitted in-flight transaction state (SQLite read-your-own-writes).
        # conn.commit() was never reached, so the implicit transaction is rolled back
        # when the connection closes.
        async with (
            aiosqlite.connect(str(db_path)) as fresh_conn,
            fresh_conn.execute('SELECT COUNT(*) FROM snapshots') as cur,
        ):
            row = await cur.fetchone()
            assert row is not None
            assert row[0] == 0

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
            with (
                patch('dashboard.data.burndown.load_task_tree', return_value=[]),
                patch('dashboard.data.burndown.find_running_orchestrators', return_value=[]),
            ):
                await collect_snapshot(conn, config)

            async with conn.execute('SELECT project_id FROM snapshots') as cur:
                rows = list(await cur.fetchall())

        assert len(rows) == 1
        # project_id must be the resolved real path, not the symlink path
        assert rows[0][0] == str(real_dir.resolve())

    @pytest.mark.asyncio
    async def test_discovers_config_flag_orchestrator(self, burndown_env):
        """Orchestrators launched with --config (no --prd) are snapshotted."""
        db_path, config, conn = burndown_env

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

        def fake_load(path):
            return _tasks_map[path]

        with (
            patch('dashboard.data.burndown.load_task_tree', side_effect=fake_load),
            patch('dashboard.data.burndown.find_running_orchestrators', return_value=fake_orchestrators),
            patch('dashboard.data.burndown._read_project_root_from_config', return_value=reify_root),
        ):
            await collect_snapshot(conn, config)

        async with conn.execute('SELECT * FROM snapshots ORDER BY project_id') as cur:
            rows = list(await cur.fetchall())

        assert len(rows) == 2
        ids = {row[1] for row in rows}
        assert str(config.project_root.resolve()) in ids  # main project
        assert str(reify_root) in ids            # config-discovered project
        # Check reify row counts
        reify_row = next(r for r in rows if r[1] == str(reify_root))
        assert reify_row[3] == 1  # pending
        assert reify_row[8] == 1  # done

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

        reify_root = Path('/home/leo/src/reify')
        autopilot_root = Path('/home/leo/src/autopilot-video')

        config = DashboardConfig(
            project_root=tmp_path,
            known_project_roots=[reify_root, autopilot_root],
        )

        # 3 distinct roots: main project + 2 known roots (no orchestrators)
        n_roots = 3
        barrier = threading.Barrier(n_roots, timeout=2.0)

        def fake_load(path):
            try:
                barrier.wait()
            except threading.BrokenBarrierError:
                pytest.fail(
                    'load_task_tree calls did not run concurrently '
                    '(barrier timed out — calls appear to be sequential)'
                )
            return []

        async with aiosqlite.connect(str(db_path)) as conn:
            with (
                patch('dashboard.data.burndown.load_task_tree', side_effect=fake_load),
                patch('dashboard.data.burndown.find_running_orchestrators', return_value=[]),
            ):
                await collect_snapshot(conn, config)


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
