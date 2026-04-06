"""Tests for dashboard.data.burndown — snapshot collection, downsampling, and queries."""

from __future__ import annotations

import sqlite3
from datetime import UTC, datetime, timedelta
from pathlib import Path
from unittest.mock import patch

import aiosqlite
import pytest

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
    async def test_inserts_main_project(self, tmp_path):
        db_path = tmp_path / 'burndown.db'
        _create_burndown_db(db_path)

        from dashboard.config import DashboardConfig
        config = DashboardConfig(project_root=tmp_path)

        fake_tasks = [
            {'status': 'done'},
            {'status': 'done'},
            {'status': 'pending'},
            {'status': 'in-progress'},
        ]

        async with aiosqlite.connect(str(db_path)) as conn:
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
        assert row[1] == str(tmp_path.resolve())  # project_id
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

        from dashboard.config import DashboardConfig
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
    async def test_deduplicates_main_project_from_orchestrators(self, tmp_path):
        """If an orchestrator targets the same root, only one row is inserted."""
        db_path = tmp_path / 'burndown.db'
        _create_burndown_db(db_path)

        from dashboard.config import DashboardConfig
        config = DashboardConfig(project_root=tmp_path)

        fake_orchestrators = [{'prd': str(tmp_path / 'prd.md'), 'project_root': str(tmp_path)}]

        async with aiosqlite.connect(str(db_path)) as conn:
            with (
                patch('dashboard.data.burndown.load_task_tree', return_value=[]),
                patch('dashboard.data.burndown.find_running_orchestrators', return_value=fake_orchestrators),
                patch('dashboard.data.burndown._resolve_project_root', return_value=tmp_path),
            ):
                await collect_snapshot(conn, config)

            async with conn.execute('SELECT COUNT(*) FROM snapshots') as cur:
                row = await cur.fetchone()
                assert row is not None
                count = row[0]

        assert count == 1

    @pytest.mark.asyncio
    async def test_snapshots_known_project_roots_when_no_orchestrators(self, tmp_path):
        """Known roots are snapshotted even when no orchestrators are running."""
        db_path = tmp_path / 'burndown.db'
        _create_burndown_db(db_path)

        reify_root = Path('/home/leo/src/reify')
        autopilot_root = Path('/home/leo/src/autopilot-video')

        from dashboard.config import DashboardConfig
        config = DashboardConfig(
            project_root=tmp_path,
            known_project_roots=[reify_root, autopilot_root],
        )

        main_tasks = [{'status': 'pending'}]
        reify_tasks = [{'status': 'done'}, {'status': 'done'}]
        autopilot_tasks = [{'status': 'in-progress'}]

        async with aiosqlite.connect(str(db_path)) as conn:
            with (
                patch('dashboard.data.burndown.load_task_tree',
                      side_effect=[main_tasks, reify_tasks, autopilot_tasks]),
                patch('dashboard.data.burndown.find_running_orchestrators', return_value=[]),
            ):
                await collect_snapshot(conn, config)

            async with conn.execute('SELECT project_id FROM snapshots') as cur:
                rows = list(await cur.fetchall())

        assert len(rows) == 3
        project_ids = {row[0] for row in rows}
        assert str(tmp_path.resolve()) in project_ids
        assert str(reify_root.resolve()) in project_ids
        assert str(autopilot_root.resolve()) in project_ids

    @pytest.mark.asyncio
    async def test_dedupes_known_root_against_main_project(self, tmp_path):
        """If known_project_roots includes main project_root, only one row is inserted."""
        db_path = tmp_path / 'burndown.db'
        _create_burndown_db(db_path)

        from dashboard.config import DashboardConfig
        config = DashboardConfig(
            project_root=tmp_path,
            known_project_roots=[tmp_path],  # same as project_root
        )

        async with aiosqlite.connect(str(db_path)) as conn:
            with (
                patch('dashboard.data.burndown.load_task_tree', return_value=[]),
                patch('dashboard.data.burndown.find_running_orchestrators', return_value=[]),
            ):
                await collect_snapshot(conn, config)

            async with conn.execute('SELECT COUNT(*) FROM snapshots WHERE project_id = ?',
                                    (str(tmp_path.resolve()),)) as cur:
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
    async def test_dedupes_known_root_against_running_orchestrator(self, tmp_path):
        """If known_project_roots includes a root already discovered via orchestrator, no duplicate."""
        db_path = tmp_path / 'burndown.db'
        _create_burndown_db(db_path)

        reify_root = Path('/home/leo/src/reify')

        from dashboard.config import DashboardConfig
        config = DashboardConfig(
            project_root=tmp_path,
            known_project_roots=[reify_root],
        )

        # Orchestrator also points to reify via config_path
        fake_orchestrators = [
            {'prd': None, 'config_path': '/home/leo/src/reify/orchestrator.yaml'},
        ]

        async with aiosqlite.connect(str(db_path)) as conn:
            with (
                patch('dashboard.data.burndown.load_task_tree',
                      side_effect=[[], [{'status': 'done'}]]),
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
    async def test_main_project_id_is_resolved_path(self, tmp_path):
        """project_id in snapshot must be the resolved path even when project_root is a symlink."""
        real_dir = tmp_path / 'real'
        real_dir.mkdir()
        link = tmp_path / 'link'
        link.symlink_to(real_dir)

        db_path = tmp_path / 'burndown.db'
        _create_burndown_db(db_path)

        from dashboard.config import DashboardConfig
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
    async def test_discovers_config_flag_orchestrator(self, tmp_path):
        """Orchestrators launched with --config (no --prd) are snapshotted."""
        db_path = tmp_path / 'burndown.db'
        _create_burndown_db(db_path)

        from dashboard.config import DashboardConfig
        config = DashboardConfig(project_root=tmp_path)

        reify_root = Path('/home/leo/src/reify')
        fake_orchestrators = [
            {'prd': None, 'config_path': '/home/leo/src/reify/orchestrator.yaml'},
        ]
        reify_tasks = [{'status': 'done'}, {'status': 'pending'}]

        async with aiosqlite.connect(str(db_path)) as conn:
            with (
                patch('dashboard.data.burndown.load_task_tree', side_effect=[[], reify_tasks]),
                patch('dashboard.data.burndown.find_running_orchestrators', return_value=fake_orchestrators),
                patch('dashboard.data.burndown._read_project_root_from_config', return_value=reify_root),
            ):
                await collect_snapshot(conn, config)

            async with conn.execute('SELECT * FROM snapshots ORDER BY project_id') as cur:
                rows = list(await cur.fetchall())

        assert len(rows) == 2
        ids = {row[1] for row in rows}
        assert str(tmp_path) in ids       # main project
        assert str(reify_root) in ids      # config-discovered project
        # Check reify row counts
        reify_row = next(r for r in rows if r[1] == str(reify_root))
        assert reify_row[3] == 1  # pending
        assert reify_row[8] == 1  # done


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
