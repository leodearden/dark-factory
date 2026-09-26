"""Tests for dashboard.data.burndown — snapshot collection, downsampling, and queries."""

from __future__ import annotations

import asyncio
import contextlib
import logging
import sqlite3
from collections.abc import Callable
from datetime import UTC, datetime, timedelta
from pathlib import Path
from unittest.mock import AsyncMock, patch

import aiosqlite
import httpx
import pytest
from shared.task_statuses import ACTIVE, TaskStatus

import dashboard.data.burndown as burndown_module
from dashboard.config import DashboardConfig
from dashboard.data.burndown import (
    BURNDOWN_SCHEMA,
    aggregate_burndown_projects,
    aggregate_burndown_series,
    collect_snapshot,
    compute_window_completion,
    downsample,
    ensure_snapshot_columns,
    get_burndown_projects,
    get_burndown_series,
)


def _raw_task(task_id, status, **fields) -> dict:
    """One raw MCP ``get_tasks`` row, as fused-memory serves it (string id)."""
    return {'id': str(task_id), 'status': status, 'dependencies': [], 'metadata': {}, **fields}


def _task_rows(tasks_or_dict) -> list[dict]:
    """Coerce a fixture into raw MCP ``get_tasks`` rows.

    A legacy ``{id: status}`` map expands into claimant-less rows, which the
    shared strand predicate reads as stranded when in-progress.  A ``list``
    of dicts passes through, with an ``id`` filled in by position when absent.
    """
    if isinstance(tasks_or_dict, dict):
        return [_raw_task(task_id, status) for task_id, status in tasks_or_dict.items()]
    rows = []
    for position, task in enumerate(tasks_or_dict):
        fields = dict(task)
        rows.append(_raw_task(fields.pop('id', position), **fields))
    return rows


def _root_key(root):
    """Canonicalised key for root-keyed fixture maps."""
    return str(Path(root).resolve())


class _CannedStore:
    """A root-keyed stand-in for the PUBLIC ``dashboard.data.tasks.mcp_tool_call``.

    Modelled on ``dashboard/tests/test_task_snapshot.py::CannedMCP``, keyed by
    project root so one instance serves a whole collector cycle and the REAL
    ``task_snapshot.acquire_snapshot`` runs above it:

    * ``get_statuses`` answers the root's ``{id: status}`` map — the SAME rows
      ``get_tasks`` serves, plus any *census_only* extras — with NO
      ``pagination`` key, the substrate's spelling of COMPLETE.  Census and
      rows therefore agree by construction unless a test says otherwise.
    * ``get_tasks`` applies ``args['statuses']`` as a server-side filter.
    * A root mapped to an exception INSTANCE raises it on every call, and
      every call ``fail_when(root, tool)`` accepts raises ``fail_with``.  Both
      are attributes, so one instance can change behaviour between ticks.
    * An unmapped root is an ``AssertionError``: a fixture that forgot a root
      must not read as an empty project.

    Every call is recorded as ``{'root', 'tool', 'args'}``.
    """

    def __init__(
        self,
        by_root,
        *,
        census_only=None,
        fail_when: Callable[[str, str], bool] = lambda root, tool: False,
        fail_with: BaseException | None = None,
    ) -> None:
        self.by_root = {
            _root_key(root): tasks if isinstance(tasks, BaseException) else _task_rows(tasks)
            for root, tasks in by_root.items()
        }
        self.census_only = {
            _root_key(root): {str(task_id): status for task_id, status in extra.items()}
            for root, extra in (census_only or {}).items()
        }
        self.fail_when = fail_when
        self.fail_with = fail_with or httpx.ReadTimeout('canned read timeout')
        self.calls: list[dict] = []

    def calls_to(self, tool: str, root=None) -> list[dict]:
        """Every recorded call to *tool* (for *root*, when given), in order."""
        return [
            call for call in self.calls
            if call['tool'] == tool and (root is None or call['root'] == _root_key(root))
        ]

    async def __call__(self, client, url, tool, args, **kwargs):
        root = _root_key(args['project_root'])
        self.calls.append({'root': root, 'tool': tool, 'args': dict(args)})
        assert root in self.by_root, f'Unmapped project_root: {root}'
        rows = self.by_root[root]
        if isinstance(rows, BaseException):
            raise rows
        if self.fail_when(root, tool):
            raise self.fail_with
        if tool == 'get_statuses':
            statuses = {str(row['id']): row['status'] for row in rows}
            statuses.update(self.census_only.get(root, {}))
            return {'statuses': statuses}
        if tool == 'get_tasks':
            wanted = args.get('statuses')
            return {'tasks': [
                dict(row) for row in rows if wanted is None or row['status'] in wanted
            ]}
        raise AssertionError(f'unexpected tool {tool!r}')


def _serve(store: _CannedStore):
    """Patch the canned substrate in at the public MCP seam."""
    return patch('dashboard.data.tasks.mcp_tool_call', new=store)


@pytest.fixture(autouse=True)
def _isolate_caches():
    """Neither the snapshot unit cache nor the fetch_tasks cache may cross a test."""
    import dashboard.data.task_snapshot as snapshot_mod
    import dashboard.data.tasks as tasks_mod

    snapshot_mod._snapshot_cache_clear()
    tasks_mod._fetch_tasks_cache_clear()
    yield
    snapshot_mod._snapshot_cache_clear()
    tasks_mod._fetch_tasks_cache_clear()

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
    in_progress_live: int | None = None,
    in_progress_stranded: int = 0,
    concurrency_cap: int | None = None,
    **more,
) -> None:
    """Insert one fixture row by explicit column names.

    *more* names any further column (``review``, ``state``, ``reason``, ...),
    so a fixture can write exactly the row shape the test is about.
    """
    # Default the split to all-live so the conservation invariant
    # (live + stranded == in_progress) holds for fixtures that do not care
    # about it; a NULL cap is the honest "unknown".
    if in_progress_live is None:
        in_progress_live = in_progress - in_progress_stranded
    row = {
        'project_id': project_id, 'ts': ts, 'pending': pending,
        'in_progress': in_progress, 'blocked': blocked, 'deferred': deferred,
        'cancelled': cancelled, 'done': done, 'in_progress_live': in_progress_live,
        'in_progress_stranded': in_progress_stranded, 'concurrency_cap': concurrency_cap,
        **more,
    }
    conn.execute(
        f'INSERT INTO snapshots ({", ".join(row)}) VALUES ({", ".join("?" for _ in row)})',
        tuple(row.values()),
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


# ---------------------------------------------------------------------------
# Fixtures
# ---------------------------------------------------------------------------


@pytest.fixture
def burndown_conn_with_config(tmp_path):
    """Factory fixture: returns a callable that produces an async context manager.

    Each call accepts an optional *project_root* keyword argument plus any
    **config_kwargs forwarded verbatim to DashboardConfig.  The context manager
    yields a ``(db_path, config, conn)`` triple — identical contract to
    ``burndown_env`` — but supports custom known_project_roots and
    project_root overrides that ``burndown_env`` cannot accommodate.

    Usage::

        async with burndown_conn_with_config(known_project_roots=[...]) as (db_path, config, conn):
            ...
    """
    from contextlib import asynccontextmanager

    @asynccontextmanager
    async def _factory(project_root=None, **config_kwargs):
        db_path = tmp_path / 'burndown.db'
        _create_burndown_db(db_path)
        config = DashboardConfig(
            project_root=project_root if project_root is not None else tmp_path,
            **config_kwargs,
        )
        async with aiosqlite.connect(str(db_path)) as conn:
            conn.row_factory = aiosqlite.Row
            yield db_path, config, conn

    return _factory


@pytest.fixture
async def burndown_env(burndown_conn_with_config):
    """Yield (db_path, config, conn) with a fresh burndown DB and open connection."""
    async with burndown_conn_with_config() as triple:
        yield triple


# ---------------------------------------------------------------------------
# Schema + additive migration for the split / cap columns (task 3543)
# ---------------------------------------------------------------------------

# The pre-3543 DDL, verbatim. Every already-deployed burndown.db is on this
# shape, and BURNDOWN_SCHEMA is applied with CREATE TABLE IF NOT EXISTS, so
# editing the DDL string alone would never reach them.
_LEGACY_BURNDOWN_SCHEMA = """\
CREATE TABLE IF NOT EXISTS snapshots (
    id          INTEGER PRIMARY KEY AUTOINCREMENT,
    project_id  TEXT    NOT NULL,
    ts          TEXT    NOT NULL,
    pending     INTEGER NOT NULL DEFAULT 0,
    in_progress INTEGER NOT NULL DEFAULT 0,
    blocked     INTEGER NOT NULL DEFAULT 0,
    deferred    INTEGER NOT NULL DEFAULT 0,
    cancelled   INTEGER NOT NULL DEFAULT 0,
    done        INTEGER NOT NULL DEFAULT 0
);
CREATE INDEX IF NOT EXISTS idx_snapshots_project_ts ON snapshots(project_id, ts);
"""

_NEW_SNAPSHOT_COLUMNS = ('in_progress_live', 'in_progress_stranded', 'concurrency_cap')


def _legacy_burndown_db(path: Path) -> None:
    """Create a pre-3543 burndown DB carrying one legacy snapshot row."""
    conn = sqlite3.connect(str(path))
    conn.executescript(_LEGACY_BURNDOWN_SCHEMA)
    conn.execute(
        'INSERT INTO snapshots (project_id, ts, pending, in_progress, blocked, '
        'deferred, cancelled, done) VALUES (?, ?, ?, ?, ?, ?, ?, ?)',
        ('legacy', '2026-01-01T00:00:00+00:00', 5, 3, 1, 0, 0, 20),
    )
    conn.commit()
    conn.close()


def _columns(path: Path) -> set[str]:
    conn = sqlite3.connect(str(path))
    try:
        return {r[1] for r in conn.execute('PRAGMA table_info(snapshots)')}
    finally:
        conn.close()


def _column_specs(path: Path) -> dict[str, tuple[int, str | None]]:
    """``{column: (notnull, dflt_value)}`` from ``PRAGMA table_info(snapshots)``.

    Asserts against the table the collector actually writes to, rather than
    against the DDL text — the shape is what matters, and it survives any
    reformatting of BURNDOWN_SCHEMA.
    """
    conn = sqlite3.connect(str(path))
    try:
        return {r[1]: (r[3], r[4]) for r in conn.execute('PRAGMA table_info(snapshots)')}
    finally:
        conn.close()


def _column_types(path: Path) -> dict[str, str]:
    """``{column: declared type}`` from ``PRAGMA table_info(snapshots)``."""
    conn = sqlite3.connect(str(path))
    try:
        return {r[1]: r[2] for r in conn.execute('PRAGMA table_info(snapshots)')}
    finally:
        conn.close()


# The pre-δ1 DDL (task 5591), verbatim: the 11-column shape every deployed
# burndown.db is on today — six zones, the NOT NULL DEFAULT 0 split, the
# nullable cap.  Like _LEGACY_BURNDOWN_SCHEMA, it only reaches the new column
# set through ensure_snapshot_columns.
_PRE_DELTA1_BURNDOWN_SCHEMA = """\
CREATE TABLE IF NOT EXISTS snapshots (
    id          INTEGER PRIMARY KEY AUTOINCREMENT,
    project_id  TEXT    NOT NULL,
    ts          TEXT    NOT NULL,
    pending     INTEGER NOT NULL DEFAULT 0,
    in_progress INTEGER NOT NULL DEFAULT 0,
    blocked     INTEGER NOT NULL DEFAULT 0,
    deferred    INTEGER NOT NULL DEFAULT 0,
    cancelled   INTEGER NOT NULL DEFAULT 0,
    done        INTEGER NOT NULL DEFAULT 0,
    in_progress_live     INTEGER NOT NULL DEFAULT 0,
    in_progress_stranded INTEGER NOT NULL DEFAULT 0,
    concurrency_cap      INTEGER
);
CREATE INDEX IF NOT EXISTS idx_snapshots_project_ts ON snapshots(project_id, ts);
"""

_DELTA1_COLUMNS = ('review', 'merge_deferred', 'infra_hold', 'in_progress_rows', 'state', 'reason')
_DELTA1_COUNT_COLUMNS = ('review', 'merge_deferred', 'infra_hold', 'in_progress_rows')
_DELTA1_TEXT_COLUMNS = ('state', 'reason')

# The pre-δ1 row's eleven stored values, keyed by column.
_PRE_DELTA1_ROW = {
    'project_id': 'pre-delta1',
    'ts': '2026-01-01T00:00:00+00:00',
    'pending': 5,
    'in_progress': 3,
    'blocked': 1,
    'deferred': 0,
    'cancelled': 0,
    'done': 20,
    'in_progress_live': 2,
    'in_progress_stranded': 1,
    'concurrency_cap': 24,
}


def _pre_delta1_burndown_db(path: Path, **overrides) -> None:
    """Create a pre-δ1 (11-column) burndown DB carrying ONE measured row.

    *overrides* replace fields of :data:`_PRE_DELTA1_ROW` (e.g. ``ts=`` so a
    read-side test can place the row inside its window).
    """
    row = {**_PRE_DELTA1_ROW, **overrides}
    conn = sqlite3.connect(str(path))
    conn.executescript(_PRE_DELTA1_BURNDOWN_SCHEMA)
    conn.execute(
        f'INSERT INTO snapshots ({", ".join(row)}) VALUES ({", ".join("?" for _ in row)})',
        tuple(row.values()),
    )
    conn.commit()
    conn.close()


class TestSnapshotSchemaColumns:
    def test_created_table_declares_the_new_column_constraints(self, tmp_path):
        """The split columns are NOT NULL DEFAULT 0; ``concurrency_cap`` and the
        six δ1 columns are nullable with no default.

        NULL is the honest "cap unknown" value for the cap — never 0, which
        would read as a cap of zero and alarm on every snapshot.  The δ1
        columns take NULL for "not recorded when the row was written": a
        DEFAULT 0 would fabricate measured zeros across all of history.
        """
        db = tmp_path / 'shape.db'
        _create_burndown_db(db)
        specs = _column_specs(db)

        for col in ('in_progress_live', 'in_progress_stranded'):
            assert col in specs, f'{col} missing from the created snapshots table'
            notnull, default = specs[col]
            assert notnull == 1, f'{col} should be NOT NULL'
            assert default == '0', f'{col} should DEFAULT 0, got {default!r}'

        assert 'concurrency_cap' in specs, 'concurrency_cap missing from snapshots'
        assert specs['concurrency_cap'][0] == 0, 'concurrency_cap must stay nullable'

        for col in _DELTA1_COLUMNS:
            assert col in specs, f'{col} missing from the created snapshots table'
            assert specs[col] == (0, None), (
                f'{col} must be nullable with no default, got (notnull, dflt)={specs[col]!r}'
            )

    def test_fresh_db_is_already_at_the_new_shape(self, tmp_path):
        db = tmp_path / 'fresh.db'
        _create_burndown_db(db)

        assert _columns(db) >= _NEW_SNAPSHOT_COLUMNS_SET

    async def test_fresh_and_migrated_stores_declare_the_delta1_columns_identically(
        self, tmp_path,
    ):
        """A fresh store (BURNDOWN_SCHEMA) and a migrated pre-δ1 store cannot
        drift: the six δ1 columns carry the same (notnull, default, type)."""
        fresh = tmp_path / 'fresh.db'
        _create_burndown_db(fresh)
        migrated = tmp_path / 'migrated.db'
        _pre_delta1_burndown_db(migrated)
        async with aiosqlite.connect(str(migrated)) as conn:
            await ensure_snapshot_columns(conn)
            await conn.commit()

        fresh_specs, migrated_specs = _column_specs(fresh), _column_specs(migrated)
        fresh_types, migrated_types = _column_types(fresh), _column_types(migrated)
        for col in _DELTA1_COLUMNS:
            assert fresh_specs[col] == migrated_specs[col] == (0, None), col
            assert fresh_types[col] == migrated_types[col], col


_NEW_SNAPSHOT_COLUMNS_SET = set(_NEW_SNAPSHOT_COLUMNS)


class TestEnsureSnapshotColumns:
    """``ensure_snapshot_columns(conn)`` — the additive ALTER TABLE migration.

    Mandatory, not cosmetic: BURNDOWN_SCHEMA is applied with CREATE TABLE IF
    NOT EXISTS, so a live burndown.db never gains the new columns from the DDL
    edit alone and the widened INSERT would fail on the first collection cycle
    after deploy — on exactly the machines holding the history this task exists
    to make legible.
    """

    async def test_adds_the_three_columns_to_a_legacy_db(self, tmp_path):
        db = tmp_path / 'legacy.db'
        _legacy_burndown_db(db)
        assert not (_NEW_SNAPSHOT_COLUMNS_SET & _columns(db))

        async with aiosqlite.connect(str(db)) as conn:
            await ensure_snapshot_columns(conn)
            await conn.commit()

        assert _columns(db) >= _NEW_SNAPSHOT_COLUMNS_SET

    async def test_legacy_rows_survive_with_zero_zero_null(self, tmp_path):
        db = tmp_path / 'legacy.db'
        _legacy_burndown_db(db)

        async with aiosqlite.connect(str(db)) as conn:
            await ensure_snapshot_columns(conn)
            await conn.commit()
            cur = await conn.execute(
                'SELECT done, in_progress, in_progress_live, in_progress_stranded, '
                'concurrency_cap FROM snapshots WHERE project_id = ?',
                ('legacy',),
            )
            row = await cur.fetchone()

        assert row is not None
        assert row[0] == 20  # pre-existing data untouched
        assert row[1] == 3
        assert row[2] == 0
        assert row[3] == 0
        assert row[4] is None  # cap genuinely unknown for a historical row

    async def test_is_idempotent(self, tmp_path):
        db = tmp_path / 'legacy.db'
        _legacy_burndown_db(db)

        async with aiosqlite.connect(str(db)) as conn:
            await ensure_snapshot_columns(conn)
            await conn.commit()
            await ensure_snapshot_columns(conn)  # must not raise
            await conn.commit()

        assert _columns(db) >= _NEW_SNAPSHOT_COLUMNS_SET

    async def test_is_a_noop_on_a_fresh_db(self, tmp_path):
        db = tmp_path / 'fresh.db'
        _create_burndown_db(db)

        async with aiosqlite.connect(str(db)) as conn:
            await ensure_snapshot_columns(conn)
            await conn.commit()

        assert _columns(db) >= _NEW_SNAPSHOT_COLUMNS_SET

    async def test_adds_the_six_delta1_columns_nullable_with_no_default(self, tmp_path):
        """The δ1 migration (task 5591) on today's deployed 11-column shape.

        ``DEFAULT 0`` is the thing ruled out: it would stamp a measured zero
        onto every historical row for statuses nobody counted back then.
        """
        db = tmp_path / 'pre_delta1.db'
        _pre_delta1_burndown_db(db)
        assert not (set(_DELTA1_COLUMNS) & _columns(db))

        async with aiosqlite.connect(str(db)) as conn:
            await ensure_snapshot_columns(conn)
            await conn.commit()

        specs, types = _column_specs(db), _column_types(db)
        for col in _DELTA1_COLUMNS:
            assert col in specs, f'{col} not added by the migration'
            notnull, default = specs[col]
            assert notnull == 0, f'{col} must be nullable'
            assert default is None, f'{col} must carry no default, got {default!r}'
        for col in _DELTA1_COUNT_COLUMNS:
            assert types[col] == 'INTEGER', f'{col}: {types[col]!r}'
        for col in _DELTA1_TEXT_COLUMNS:
            assert types[col] == 'TEXT', f'{col}: {types[col]!r}'

    async def test_pre_delta1_row_reads_null_for_every_added_column(self, tmp_path):
        """NULL = "not recorded when the row was written" — never 0 — and the
        row's eleven existing values are untouched."""
        db = tmp_path / 'pre_delta1.db'
        _pre_delta1_burndown_db(db)

        async with aiosqlite.connect(str(db)) as conn:
            await ensure_snapshot_columns(conn)
            await conn.commit()
            conn.row_factory = aiosqlite.Row
            cur = await conn.execute(
                'SELECT * FROM snapshots WHERE project_id = ?', (_PRE_DELTA1_ROW['project_id'],),
            )
            rows = list(await cur.fetchall())

        assert len(rows) == 1
        row = rows[0]
        for col in _DELTA1_COLUMNS:
            assert row[col] is None, f'{col}: expected NULL, got {row[col]!r}'
        for col, value in _PRE_DELTA1_ROW.items():
            assert row[col] == value, f'{col}: expected {value!r}, got {row[col]!r}'

    async def test_delta1_migration_is_idempotent(self, tmp_path):
        db = tmp_path / 'pre_delta1.db'
        _pre_delta1_burndown_db(db)

        async with aiosqlite.connect(str(db)) as conn:
            await ensure_snapshot_columns(conn)
            await conn.commit()
            await ensure_snapshot_columns(conn)  # must not raise
            await conn.commit()

        assert _columns(db) >= set(_DELTA1_COLUMNS)

    async def test_pre_3543_store_reaches_the_full_current_column_set(self, tmp_path):
        """A store two migrations behind lands on exactly the fresh shape."""
        legacy = tmp_path / 'legacy.db'
        _legacy_burndown_db(legacy)
        fresh = tmp_path / 'fresh.db'
        _create_burndown_db(fresh)

        async with aiosqlite.connect(str(legacy)) as conn:
            await ensure_snapshot_columns(conn)
            await conn.commit()

        assert _columns(legacy) == _columns(fresh)
        assert _columns(legacy) >= _NEW_SNAPSHOT_COLUMNS_SET | set(_DELTA1_COLUMNS)

    async def test_migrated_store_carries_all_nine_member_columns(self, tmp_path):
        """One column per ``TaskStatus`` member, written out literally here so
        a renamed member cannot silently rename its column with it."""
        db = tmp_path / 'pre_delta1.db'
        _pre_delta1_burndown_db(db)

        async with aiosqlite.connect(str(db)) as conn:
            await ensure_snapshot_columns(conn)
            await conn.commit()

        assert _columns(db) >= {
            'pending', 'in_progress', 'blocked', 'deferred', 'review',
            'merge_deferred', 'infra_hold', 'done', 'cancelled',
        }

    async def test_widened_insert_succeeds_after_migration(self, tmp_path, dummy_client):
        """``collect_snapshot`` writes its full row into a store migrated in place."""
        db = tmp_path / 'pre_delta1.db'
        _pre_delta1_burndown_db(db)
        project_root = tmp_path / 'project'
        _write_orch_config(project_root, 'max_concurrent_tasks: 24\n')
        config = DashboardConfig(project_root=project_root)
        store = _CannedStore({project_root: [_live_now(id=1), _stranded(id=2)]})

        async with aiosqlite.connect(str(db)) as conn:
            await ensure_snapshot_columns(conn)
            await conn.commit()
            with (
                _serve(store),
                patch('dashboard.data.burndown.find_running_orchestrators', return_value=[]),
            ):
                await collect_snapshot(conn, config, client=dummy_client)
            cur = await conn.execute(
                'SELECT in_progress, in_progress_live, in_progress_stranded, '
                'in_progress_rows, concurrency_cap, state FROM snapshots WHERE project_id = ?',
                (str(config.project_root),),
            )
            row = await cur.fetchone()

        assert row == (2, 1, 1, 2, 24, 'value')


# ---------------------------------------------------------------------------
# collect_snapshot
# ---------------------------------------------------------------------------


class TestCollectSnapshot:
    @pytest.mark.asyncio
    async def test_inserts_main_project(self, burndown_env, dummy_client):
        db_path, config, conn = burndown_env

        fake_tasks = [
            {'status': 'done'},
            {'status': 'done'},
            {'status': 'pending'},
            {'status': 'in-progress'},
        ]

        with (
            _serve(_CannedStore({config.project_root: fake_tasks})),
            patch('dashboard.data.burndown.find_running_orchestrators', return_value=[]),
        ):
            await collect_snapshot(conn, config, client=dummy_client)

        async with conn.execute('SELECT * FROM snapshots') as cur:
            rows = await cur.fetchall()

        assert len(rows) == 1
        row = rows[0]
        assert row['project_id'] == str(config.project_root)
        _assert_snapshot_counts(row, pending=1, in_progress=1, done=2)

    @pytest.mark.asyncio
    async def test_symlinked_root_deduplicates_with_orchestrator(self, tmp_path, burndown_conn_with_config, dummy_client):
        """Symlinked project_root and orchestrator resolving to real path produce only 1 row."""
        real_dir = tmp_path / 'real'
        real_dir.mkdir()
        link = tmp_path / 'link'
        link.symlink_to(real_dir)

        # Orchestrator resolves the same project via _resolve_project_root to the real path
        fake_orchestrators = [{'prd': 'fake_prd.md', 'config_path': None}]

        async with burndown_conn_with_config(project_root=link) as (db_path, config, conn):
            with (
                _serve(_CannedStore({real_dir: []})),
                patch('dashboard.data.burndown.find_running_orchestrators', return_value=fake_orchestrators),
                patch('dashboard.data.burndown._resolve_project_root', return_value=real_dir.resolve()),
            ):
                await collect_snapshot(conn, config, client=dummy_client)

            async with conn.execute('SELECT COUNT(*) FROM snapshots') as cur:
                row = await cur.fetchone()
                assert row is not None
                assert row[0] == 1  # Only 1 row — symlink and real path should deduplicate

            # Distinct from test_main_project_id_is_resolved_path: verifies that the
            # resolved path is used as the project_id specifically in the orchestrator-
            # dedup scenario, where both config.project_root and the mocked
            # _resolve_project_root resolve to the same real_dir.
            async with conn.execute('SELECT project_id FROM snapshots') as cur:
                rows = list(await cur.fetchall())
            assert len(rows) == 1
            assert rows[0]['project_id'] == str(real_dir.resolve())

    @pytest.mark.asyncio
    async def test_deduplicates_main_project_from_orchestrators(self, burndown_env, dummy_client):
        """If an orchestrator targets the same root, only one row is inserted."""
        _, config, conn = burndown_env

        fake_orchestrators = [{'prd': str(config.project_root / 'prd.md'), 'project_root': str(config.project_root)}]

        with (
            _serve(_CannedStore({config.project_root: []})),
            patch('dashboard.data.burndown.find_running_orchestrators', return_value=fake_orchestrators),
            patch('dashboard.data.burndown._resolve_project_root', return_value=config.project_root),
        ):
            await collect_snapshot(conn, config, client=dummy_client)

        async with conn.execute('SELECT COUNT(*) FROM snapshots') as cur:
            row = await cur.fetchone()
            assert row is not None
            assert row[0] == 1

    @pytest.mark.asyncio
    async def test_snapshots_known_project_roots_when_no_orchestrators(self, burndown_env, dummy_client):
        """Known roots are snapshotted even when no orchestrators are running."""
        _, base_config, conn = burndown_env

        reify_root = Path('/nonexistent/known/reify')
        autopilot_root = Path('/nonexistent/known/autopilot')

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
            config.project_root: main_tasks,
            reify_root: reify_tasks,
            autopilot_root: autopilot_tasks,
        }

        with (
            _serve(_CannedStore(_tasks_map)),
            patch('dashboard.data.burndown.find_running_orchestrators', return_value=[]),
        ):
            await collect_snapshot(conn, config, client=dummy_client)

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
    async def test_dedupes_known_root_against_main_project(self, burndown_env, dummy_client):
        """If known_project_roots includes main project_root, only one row is inserted."""
        _, base_config, conn = burndown_env

        config = DashboardConfig(
            project_root=base_config.project_root,
            known_project_roots=[base_config.project_root],  # same as project_root
        )

        with (
            _serve(_CannedStore({base_config.project_root: []})),
            patch('dashboard.data.burndown.find_running_orchestrators', return_value=[]),
        ):
            await collect_snapshot(conn, config, client=dummy_client)

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
    async def test_symlinked_root_deduplicates_with_known_roots(self, tmp_path, burndown_conn_with_config, dummy_client):
        """If known_project_roots includes the resolved real path, it deduplicates with a symlinked project_root."""
        real_dir = tmp_path / 'real'
        real_dir.mkdir()
        link = tmp_path / 'link'
        link.symlink_to(real_dir)

        # project_root is the symlink; known_project_roots contains the resolved real path
        async with burndown_conn_with_config(project_root=link, known_project_roots=[real_dir]) as (db_path, config, conn):
            with (
                _serve(_CannedStore({real_dir: []})),
                patch('dashboard.data.burndown.find_running_orchestrators', return_value=[]),
            ):
                await collect_snapshot(conn, config, client=dummy_client)

            async with conn.execute('SELECT COUNT(*) FROM snapshots') as cur:
                row = await cur.fetchone()
                assert row is not None
                assert row[0] == 1  # Only 1 row — symlink project_root and known real path deduplicate

    @pytest.mark.asyncio
    async def test_dedupes_known_root_against_running_orchestrator(self, burndown_env, dummy_client):
        """If known_project_roots includes a root already discovered via orchestrator, no duplicate."""
        _, base_config, conn = burndown_env

        reify_root = Path('/nonexistent/known/reify')

        config = DashboardConfig(
            project_root=base_config.project_root,
            known_project_roots=[reify_root],
        )

        # Orchestrator also points to reify via config_path
        fake_orchestrators = [
            {'prd': None, 'config_path': '/nonexistent/known/reify/orchestrator.yaml'},
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
            config.project_root: [],
            reify_root: [{'status': 'done'}],
        }

        with (
            _serve(_CannedStore(_tasks_map)),
            patch('dashboard.data.burndown.find_running_orchestrators', return_value=fake_orchestrators),
            patch('dashboard.data.burndown._read_project_root_from_config', return_value=reify_root),
        ):
            await collect_snapshot(conn, config, client=dummy_client)

        async with conn.execute('SELECT COUNT(*) FROM snapshots WHERE project_id = ?',
                                (str(reify_root),)) as cur:
            row = await cur.fetchone()
            assert row is not None
            assert row[0] == 1  # only one row for reify, not two

    @pytest.mark.asyncio
    async def test_main_project_id_is_resolved_path(self, tmp_path, burndown_conn_with_config, dummy_client):
        """project_id in snapshot must be the resolved path even when project_root is a symlink."""
        real_dir = tmp_path / 'real'
        real_dir.mkdir()
        link = tmp_path / 'link'
        link.symlink_to(real_dir)

        async with burndown_conn_with_config(project_root=link) as (db_path, config, conn):
            with (
                _serve(_CannedStore({real_dir: []})),
                patch('dashboard.data.burndown.find_running_orchestrators', return_value=[]),
            ):
                await collect_snapshot(conn, config, client=dummy_client)

            async with conn.execute('SELECT project_id FROM snapshots') as cur:
                rows = list(await cur.fetchall())

        assert len(rows) == 1
        # project_id must be the resolved real path, not the symlink path
        assert rows[0]['project_id'] == str(real_dir.resolve())

    @pytest.mark.asyncio
    async def test_discovers_config_flag_orchestrator(self, burndown_env, dummy_client):
        """Orchestrators launched with --config (no --prd) are snapshotted."""
        _, config, conn = burndown_env

        reify_root = Path('/nonexistent/known/reify')
        fake_orchestrators = [
            {'prd': None, 'config_path': '/nonexistent/known/reify/orchestrator.yaml'},
        ]
        reify_tasks = [{'status': 'done'}, {'status': 'pending'}]

        # Path-keyed dispatch because asyncio.gather fires calls concurrently —
        # an ordered side_effect list can race on thread scheduling.
        _tasks_map = {
            config.project_root: [],
            reify_root: reify_tasks,
        }

        with (
            _serve(_CannedStore(_tasks_map)),
            patch('dashboard.data.burndown.find_running_orchestrators', return_value=fake_orchestrators),
            patch('dashboard.data.burndown._read_project_root_from_config', return_value=reify_root),
        ):
            await collect_snapshot(conn, config, client=dummy_client)

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
    async def test_continues_when_known_root_unreadable(self, burndown_conn_with_config, dummy_client):
        """PermissionError on one known root is skipped; other roots are still snapshotted."""
        root_a = Path('/fake/project/root_a')
        root_b = Path('/fake/project/root_b')
        root_c = Path('/fake/project/root_c')

        main_tasks = [{'status': 'pending'}]
        root_a_tasks = [{'status': 'done'}]
        root_c_tasks = [{'status': 'in-progress'}]

        # The bad root's every substrate call raises PermissionError; the other
        # roots are read concurrently and must be unaffected.
        async with burndown_conn_with_config(known_project_roots=[root_a, root_b, root_c]) as (db_path, config, conn):
            store = _CannedStore({
                config.project_root: main_tasks,
                root_a: root_a_tasks,
                root_b: PermissionError('Permission denied'),
                root_c: root_c_tasks,
            })

            with (
                _serve(store),
                patch('dashboard.data.burndown.find_running_orchestrators', return_value=[]),
            ):
                await collect_snapshot(conn, config, client=dummy_client)

            async with conn.execute('SELECT project_id FROM snapshots') as cur:
                rows = list(await cur.fetchall())

            project_ids = {row['project_id'] for row in rows}
            # main + root_a + root_c should be present
            assert len(rows) == 3
            assert str(config.project_root) in project_ids
            assert str(root_a.resolve()) in project_ids
            assert str(root_c.resolve()) in project_ids
            # root_b should NOT be present
            assert str(root_b.resolve()) not in project_ids

    @pytest.mark.asyncio
    async def test_logs_warning_when_known_root_unreadable(self, burndown_conn_with_config, caplog, dummy_client):
        """A WARNING naming the root, with exc_info, when its acquisition RAISES.

        The real ``acquire_snapshot`` is total by contract — a substrate failure
        comes back as a non-fresh unit, never an exception — so only a fake at
        the collector's public ``acquire_snapshot`` name can exercise this path.
        """
        from dashboard.data.task_snapshot import acquire_snapshot as real_acquire

        bad_root = Path('/fake/project/bad_root')
        bad_root_str = _root_key(bad_root)

        async with burndown_conn_with_config(known_project_roots=[bad_root]) as (db_path, config, conn):
            async def acquire_or_raise(client, cfg, project_root, *, now):
                if _root_key(project_root) == bad_root_str:
                    raise PermissionError('Permission denied')
                return await real_acquire(client, cfg, project_root, now=now)

            with (
                _serve(_CannedStore({config.project_root: []})),
                patch('dashboard.data.burndown.acquire_snapshot', new=acquire_or_raise),
                patch('dashboard.data.burndown.find_running_orchestrators', return_value=[]),
                caplog.at_level(logging.WARNING, logger='dashboard.data.burndown'),
            ):
                await collect_snapshot(conn, config, client=dummy_client)

        # At least one WARNING record should name the failing root
        warning_records = [r for r in caplog.records if r.levelno == logging.WARNING]
        assert warning_records, 'Expected at least one WARNING log record'
        combined = ' '.join(r.getMessage() for r in warning_records)
        assert str(bad_root.resolve()) in combined
        # exc_info must be populated on the warning record
        assert any(r.exc_info for r in warning_records)

    @pytest.mark.asyncio
    async def test_first_root_failure_does_not_block_subsequent_inserts(self, burndown_conn_with_config, dummy_client):
        """If the very first known root fails, subsequent roots still get snapshotted."""
        bad_root = Path('/fake/project/bad_root')
        good_root = Path('/fake/project/good_root')

        main_tasks = [{'status': 'pending'}]
        good_tasks = [{'status': 'done'}, {'status': 'done'}]

        async with burndown_conn_with_config(known_project_roots=[bad_root, good_root]) as (db_path, config, conn):
            store = _CannedStore({
                config.project_root: main_tasks,
                bad_root: PermissionError('denied'),
                good_root: good_tasks,
            })

            with (
                _serve(store),
                patch('dashboard.data.burndown.find_running_orchestrators', return_value=[]),
            ):
                await collect_snapshot(conn, config, client=dummy_client)

            async with conn.execute('SELECT project_id, done FROM snapshots ORDER BY project_id') as cur:
                rows = list(await cur.fetchall())

            assert len(rows) == 2
            project_ids = {row['project_id'] for row in rows}
            assert str(config.project_root) in project_ids       # (a) main project row
            assert str(good_root.resolve()) in project_ids       # (b) good_root row
            assert str(bad_root.resolve()) not in project_ids    # (c) no bad_root row

            good_row = next(r for r in rows if r['project_id'] == str(good_root.resolve()))
            assert good_row['done'] == 2  # done=2 for good_root

    @pytest.mark.asyncio
    async def test_orchestrator_fallback_deduplicates_against_resolved_root(self, tmp_path, burndown_conn_with_config, dummy_client):
        """When _resolve_project_root falls back to the symlinked config.project_root, it still deduplicates."""
        real_dir = tmp_path / 'real'
        real_dir.mkdir()
        link = tmp_path / 'link'
        link.symlink_to(real_dir)

        # Guard: _resolve_project_root walks up from prd_path looking for .taskmaster.
        # If any ancestor of tmp_path has one, the fallback branch never fires and
        # this test silently verifies the wrong code path.  Skip (not fail) when the
        # environment doesn't meet this precondition — a skip is a clearer signal than
        # an unexpected assertion error.
        for ancestor in tmp_path.resolve().parents:
            if (ancestor / '.taskmaster').is_dir():
                pytest.skip(
                    f'{ancestor} contains .taskmaster — cannot exercise '
                    f'_resolve_project_root fallback branch in this environment'
                )

        # PRD path lives directly under tmp_path (not under real_dir), so
        # _resolve_project_root will walk up from tmp_path, find no .taskmaster,
        # and fall back to config.project_root (the unresolved symlink).
        prd_path = str(tmp_path / 'fake_prd.md')
        fake_orchestrators = [{'prd': prd_path, 'config_path': None}]

        async with burndown_conn_with_config(project_root=link) as (db_path, config, conn):
            with (
                _serve(_CannedStore({real_dir: []})),
                patch('dashboard.data.burndown.find_running_orchestrators', return_value=fake_orchestrators),
                # _resolve_project_root is NOT mocked — it runs for real and falls
                # back to config.project_root (the symlink) because prd_path has no
                # .taskmaster in its ancestor chain.
            ):
                await collect_snapshot(conn, config, client=dummy_client)

            async with conn.execute('SELECT COUNT(*) FROM snapshots') as cur:
                row = await cur.fetchone()
                assert row is not None
                # Only 1 row — the orchestrator fallback targets the same project as the
                # main project_root; resolved and unresolved paths must deduplicate.
                assert row[0] == 1

    @pytest.mark.asyncio
    async def test_load_task_tree_calls_run_concurrently(self, burndown_conn_with_config, dummy_client):
        """Every root's read must run concurrently via asyncio.gather.

        Uses a threading.Barrier(N) to detect concurrency: each root's ONE
        ``get_tasks`` call must reach the barrier simultaneously. With
        sequential awaits, only one thread is alive at a time so barrier.wait()
        times out (BrokenBarrierError). With asyncio.gather, all N threads are
        live simultaneously and the barrier succeeds.
        """
        import threading

        reify_root = Path('/fake/project/reify')
        autopilot_root = Path('/fake/project/autopilot')

        # 3 distinct roots: main project + 2 known roots (no orchestrators)
        n_roots = 3
        barrier = threading.Barrier(n_roots, timeout=10.0)

        def _wait_or_fail(b):
            try:
                b.wait()
            except threading.BrokenBarrierError:
                pytest.fail(
                    'get_tasks calls did not reach the barrier within 10s — '
                    'possible causes: (1) sequential awaits (calls not running '
                    'concurrently via asyncio.gather); (2) severe scheduler latency '
                    '(threads starved by contention or a slow CI host)'
                )

        async with burndown_conn_with_config(known_project_roots=[reify_root, autopilot_root]) as (db_path, config, conn):
            store = _CannedStore({config.project_root: [], reify_root: [], autopilot_root: []})

            async def gated(client, url, tool, args, **kwargs):
                if tool == 'get_tasks':
                    await asyncio.to_thread(_wait_or_fail, barrier)
                return await store(client, url, tool, args, **kwargs)

            with (
                patch('dashboard.data.tasks.mcp_tool_call', new=gated),
                patch('dashboard.data.burndown.find_running_orchestrators', return_value=[]),
            ):
                await collect_snapshot(conn, config, client=dummy_client)

            async with conn.execute('SELECT COUNT(*) FROM snapshots') as cur:
                row = await cur.fetchone()
                assert row is not None
                assert row[0] == 3

    @pytest.mark.asyncio
    async def test_gather_partial_failure_skips_bad_project(self, burndown_conn_with_config, dummy_client):
        """One load_task_tree raises OSError; collect_snapshot skips the failing
        project and inserts healthy rows for the remaining projects.

        Invariant (post Task 519): asyncio.gather(return_exceptions=True) plus
        per-project isinstance guard mean collect_snapshot does not raise, the
        failing project is cleanly excluded from the snapshot, and healthy
        projects are snapshotted normally.
        """
        reify_root = Path('/nonexistent/known/reify')
        autopilot_root = Path('/nonexistent/known/autopilot')

        async with burndown_conn_with_config(known_project_roots=[reify_root, autopilot_root]) as (db_path, config, conn):
            # reify is the failing root: every substrate call for it raises OSError.
            store = _CannedStore({
                config.project_root: [],
                reify_root: OSError('mock disk error'),
                autopilot_root: [{'status': 'done'}],
            })

            with (
                _serve(store),
                patch('dashboard.data.burndown.find_running_orchestrators', return_value=[]),
            ):
                await collect_snapshot(conn, config, client=dummy_client)

            async with conn.execute('SELECT project_id FROM snapshots') as cur:
                rows = list(await cur.fetchall())

            project_ids = {row['project_id'] for row in rows}

            assert len(rows) == 2
            assert str(reify_root.resolve()) not in project_ids
            assert str(config.project_root) in project_ids
            assert str(autopilot_root.resolve()) in project_ids

    @pytest.mark.asyncio
    async def test_gather_return_exceptions_preserves_healthy_snapshots(self, burndown_conn_with_config, dummy_client):
        """OSError on one known root is isolated; healthy projects are still snapshotted.

        Regression anchor for task 519: a single unreadable root cannot drop the
        remaining snapshots.  The test uses OSError (not PermissionError) to
        match task 519's 'unreadable tasks.json' wording.  The WARNING-with-
        exc_info half now lives on the tests whose acquisition actually RAISES
        (the real snapshot unit absorbs a substrate failure).
        """
        bad_root = Path('/fake/project/bad_root')
        good_root_1 = Path('/fake/project/good_root_1')
        good_root_2 = Path('/fake/project/good_root_2')

        main_tasks = [{'status': 'pending'}]
        good_1_tasks = [{'status': 'done'}, {'status': 'done'}]
        good_2_tasks = [{'status': 'done'}]

        async with burndown_conn_with_config(known_project_roots=[bad_root, good_root_1, good_root_2]) as (db_path, config, conn):
            store = _CannedStore({
                config.project_root: main_tasks,
                bad_root: OSError('mock disk error'),
                good_root_1: good_1_tasks,
                good_root_2: good_2_tasks,
            })

            with (
                _serve(store),
                patch('dashboard.data.burndown.find_running_orchestrators', return_value=[]),
            ):
                # Must NOT raise even though bad_root fails.
                await collect_snapshot(conn, config, client=dummy_client)

            async with conn.execute('SELECT * FROM snapshots') as cur:
                rows = list(await cur.fetchall())

            # (a) three rows: main + good_root_1 + good_root_2
            assert len(rows) == 3

            by_project = {row['project_id']: row for row in rows}

            # (b) bad_root must NOT appear
            assert str(bad_root.resolve()) not in by_project

            # (c) main project and both good roots must appear
            assert str(config.project_root) in by_project
            assert str(good_root_1.resolve()) in by_project
            assert str(good_root_2.resolve()) in by_project

            # (c') main project row must record the correct pending count
            main_row = by_project[str(config.project_root)]
            _assert_snapshot_counts(main_row, pending=1)

            # (d) per-root done counts must reflect the supplied task lists
            good_1_row = by_project[str(good_root_1.resolve())]
            _assert_snapshot_counts(good_1_row, done=2)

            good_2_row = by_project[str(good_root_2.resolve())]
            _assert_snapshot_counts(good_2_row, done=1)

    @pytest.mark.asyncio
    async def test_main_project_failure_skips_all_inserts(self, burndown_env, dummy_client):
        """A failing main project is isolated: collect_snapshot does not raise.

        The main project is always the first entry in roots_to_snapshot. With no
        orchestrators and no known_project_roots, a main project whose every
        substrate call raises PermissionError must:

        (a) NOT propagate out of collect_snapshot;
        (b) commit zero rows — the only root failed, so nothing to insert;
        (c) never read any other root.

        The WARNING-with-exc_info half lives on the tests whose acquisition
        actually RAISES (the real snapshot unit absorbs a substrate failure).
        """
        db_path, config, conn = burndown_env
        store = _CannedStore({config.project_root: PermissionError('Permission denied')})

        with (
            _serve(store),
            patch('dashboard.data.burndown.find_running_orchestrators', return_value=[]),
        ):
            # Must NOT raise.
            await collect_snapshot(conn, config, client=dummy_client)

        assert {call['root'] for call in store.calls} == {_root_key(config.project_root)}

        # (b) zero rows committed — the only root failed, so snapshots is empty.
        async with conn.execute('SELECT COUNT(*) FROM snapshots') as cur:
            row = await cur.fetchone()
        assert row is not None
        assert row[0] == 0

    @pytest.mark.parametrize(
        'orchestrator_dict,patch_target,canonical_root',
        [
            pytest.param(
                {'prd': None, 'config_path': '/home/leo/src/contract-sentinel/orchestrator.yaml'},
                'dashboard.data.burndown._read_project_root_from_config',
                Path('/home/leo/src/contract-sentinel'),
                id='config_path_branch',
            ),
            pytest.param(
                {'prd': '/home/leo/src/contract-sentinel-prd/prd.md', 'config_path': None},
                'dashboard.data.burndown._resolve_project_root',
                Path('/home/leo/src/contract-sentinel-prd'),
                id='prd_branch',
            ),
        ],
    )
    @pytest.mark.asyncio
    async def test_orchestrator_project_id_matches_helper_return(
        self, burndown_env, orchestrator_dict, patch_target, canonical_root, dummy_client,
    ):
        """project_id stored for orchestrators matches str(helper_return) exactly.

        Contract: both _resolve_project_root and _read_project_root_from_config already
        return canonical (resolved) paths, so root_str must equal str(helper_return)
        without any additional .resolve(). Parametrized across both code branches
        (config_path and prd) to cover the full invariant.
        """
        _, config, conn = burndown_env

        _tasks_map = {
            config.project_root: [],
            canonical_root: [],
        }

        with (
            _serve(_CannedStore(_tasks_map)),
            patch('dashboard.data.burndown.find_running_orchestrators', return_value=[orchestrator_dict]),
            patch(patch_target, return_value=canonical_root),
        ):
            await collect_snapshot(conn, config, client=dummy_client)

        async with conn.execute(
            'SELECT project_id FROM snapshots WHERE project_id = ?', (str(canonical_root),)
        ) as cur:
            row = await cur.fetchone()

        # The stored project_id must equal str(canonical_root) directly — the helper's
        # return value is used as-is, with no additional .resolve() needed.
        assert row is not None, f'No snapshot row found for project_id={str(canonical_root)!r}'
        assert row[0] == str(canonical_root)

    @pytest.mark.asyncio
    async def test_orchestrator_skipped_when_config_helper_returns_none(self, burndown_env, dummy_client):
        """Orchestrators are silently skipped when _read_project_root_from_config returns None.

        Documents the guard at burndown.py lines 129-130: when the helper cannot
        determine the project root (e.g. a relative path in the YAML config that cannot
        be made absolute), the orchestrator entry is skipped via 'continue' and no
        snapshot row is created for it.
        """
        _, config, conn = burndown_env

        fake_orchestrators = [
            {'prd': None, 'config_path': '/some/orchestrator.yaml'},
        ]

        with (
            _serve(_CannedStore({config.project_root: []})),
            patch('dashboard.data.burndown.find_running_orchestrators', return_value=fake_orchestrators),
            patch('dashboard.data.burndown._read_project_root_from_config', return_value=None),
        ):
            await collect_snapshot(conn, config, client=dummy_client)

        # Only the main project's snapshot row should exist; the orchestrator was skipped.
        async with conn.execute('SELECT COUNT(*) FROM snapshots') as cur:
            count_row = await cur.fetchone()

        assert count_row[0] == 1, (
            f'Expected exactly 1 snapshot row (main project only), got {count_row[0]}'
        )

# ---------------------------------------------------------------------------
# collect_snapshot — live/stranded split + concurrency cap (task 3543 / PRD ι)
# ---------------------------------------------------------------------------


def _write_orch_config(root: Path, text: str) -> None:
    root.mkdir(parents=True, exist_ok=True)
    (root / 'dark-factory-orchestrator.yaml').write_text(text)


def _ztask(status='in-progress', **overrides):
    """A task row carrying the claimant columns the strand split reads."""
    task = {
        'id': overrides.pop('id', 1),
        'status': status,
        'metadata': {},
        'claimant_run_id': None,
        'heartbeat_at': None,
    }
    task.update(overrides)
    return task


def _stranded(**overrides):
    """An in-progress row with no claimant at all — missing evidence of life."""
    return _ztask(status='in-progress', **overrides)


def _live_now(**overrides):
    """An in-progress row heartbeating against the collector's live clock."""
    return _ztask(
        status='in-progress',
        claimant_run_id='run-1/sess-1/pid=42',
        heartbeat_at=datetime.now(UTC).isoformat(),
        **overrides,
    )


class TestCollectSnapshotTaskSourceAndCap:
    """The collector persists the rows' live/stranded split and stamps the cap.

    The split needs the claimant columns only ``get_tasks`` rows carry; the
    concurrency cap is read once per root and stored ON the row because
    ``max_concurrent_tasks`` varies across restarts and projects: resolving it
    at render time would compare a historical in-progress census against
    today's cap and mislabel both directions.
    """

    @pytest.mark.asyncio
    async def test_persists_the_live_stranded_split(self, burndown_env, dummy_client):
        """The split comes from the claimant columns on the fetched rows."""
        db_path, config, conn = burndown_env

        tasks = [
            _live_now(id=1),
            _stranded(id=2),
            _stranded(id=3),
            # 'review' is its own census member now, not folded into
            # in_progress, and — having no in-progress status — it is in
            # neither half of the split.
            _ztask(status='review', id=4),
            _ztask(status='pending', id=5),
            _ztask(status='done', id=6),
        ]

        with (
            _serve(_CannedStore({config.project_root: tasks})),
            patch('dashboard.data.burndown.find_running_orchestrators', return_value=[]),
        ):
            await collect_snapshot(conn, config, client=dummy_client)

        async with conn.execute('SELECT * FROM snapshots') as cur:
            rows = list(await cur.fetchall())

        assert len(rows) == 1
        row = rows[0]
        assert row['in_progress'] == 3
        assert row['review'] == 1
        assert row['in_progress_stranded'] == 2
        assert row['in_progress_live'] == 1
        assert row['in_progress_live'] + row['in_progress_stranded'] == row['in_progress_rows'], (
            'conservation invariant must hold on the persisted row'
        )
        assert row['pending'] == 1
        assert row['done'] == 1

    @pytest.mark.asyncio
    async def test_persists_the_cap_in_force_at_snapshot_time(
        self, burndown_env, dummy_client,
    ):
        db_path, config, conn = burndown_env
        _write_orch_config(Path(str(config.project_root)), 'max_concurrent_tasks: 24\n')

        with (
            _serve(_CannedStore({config.project_root: [_live_now(id=1)]})),
            patch('dashboard.data.burndown.find_running_orchestrators', return_value=[]),
        ):
            await collect_snapshot(conn, config, client=dummy_client)

        async with conn.execute('SELECT concurrency_cap FROM snapshots') as cur:
            row = await cur.fetchone()
        assert row is not None
        assert row['concurrency_cap'] == 24

    @pytest.mark.asyncio
    async def test_unknown_cap_persists_null_without_a_breach_warning(
        self, burndown_env, caplog, dummy_client,
    ):
        """No orchestrator config => cap unknown => NULL, and NOT a breach.

        NULL must never be stored as 0: a 0 cap would alarm on every snapshot.
        """
        db_path, config, conn = burndown_env

        with (
            _serve(_CannedStore({config.project_root: [_live_now(id=1), _live_now(id=2)]})),
            patch('dashboard.data.burndown.find_running_orchestrators', return_value=[]),
            caplog.at_level(logging.WARNING, logger='dashboard.data.burndown'),
        ):
            await collect_snapshot(conn, config, client=dummy_client)

        async with conn.execute('SELECT concurrency_cap FROM snapshots') as cur:
            row = await cur.fetchone()
        assert row is not None
        assert row['concurrency_cap'] is None

        breach_warnings = [
            r for r in caplog.records
            if r.levelno == logging.WARNING and 'cap' in r.getMessage().lower()
        ]
        assert not breach_warnings, (
            f'an unknown cap is not a breach; got: {[r.getMessage() for r in breach_warnings]}'
        )

    @pytest.mark.asyncio
    async def test_cap_breach_warns_naming_project_count_and_cap(
        self, burndown_env, caplog, dummy_client,
    ):
        """The E12 'silently' defect: exceeding the cap must be loud.

        The message must carry all three facts — which project, how many
        in-progress, and against what cap — so the log line stands alone.
        """
        db_path, config, conn = burndown_env
        _write_orch_config(Path(str(config.project_root)), 'max_concurrent_tasks: 2\n')

        tasks = [_live_now(id=1), _live_now(id=2), _stranded(id=3)]

        with (
            _serve(_CannedStore({config.project_root: tasks})),
            patch('dashboard.data.burndown.find_running_orchestrators', return_value=[]),
            caplog.at_level(logging.WARNING, logger='dashboard.data.burndown'),
        ):
            await collect_snapshot(conn, config, client=dummy_client)

        warnings = [r.getMessage() for r in caplog.records if r.levelno == logging.WARNING]
        breach = [m for m in warnings if 'cap' in m.lower()]
        assert breach, f'expected a cap-breach WARNING, got: {warnings}'
        message = breach[0]
        assert str(config.project_root) in message, message
        assert '3' in message, f'in-progress count missing from: {message}'
        assert '2' in message, f'cap missing from: {message}'

        # The row is still written — the alarm annotates history, it does not
        # suppress it.
        async with conn.execute('SELECT in_progress, concurrency_cap FROM snapshots') as cur:
            row = await cur.fetchone()
        assert row is not None
        assert row['in_progress'] == 3
        assert row['concurrency_cap'] == 2

    @pytest.mark.asyncio
    async def test_in_progress_equal_to_cap_does_not_warn(
        self, burndown_env, caplog, dummy_client,
    ):
        """The cap is INCLUSIVE — running exactly at capacity is healthy."""
        db_path, config, conn = burndown_env
        _write_orch_config(Path(str(config.project_root)), 'max_concurrent_tasks: 2\n')

        with (
            _serve(_CannedStore({config.project_root: [_live_now(id=1), _live_now(id=2)]})),
            patch('dashboard.data.burndown.find_running_orchestrators', return_value=[]),
            caplog.at_level(logging.WARNING, logger='dashboard.data.burndown'),
        ):
            await collect_snapshot(conn, config, client=dummy_client)

        breach = [
            r.getMessage() for r in caplog.records
            if r.levelno == logging.WARNING and 'cap' in r.getMessage().lower()
        ]
        assert not breach, f'at-capacity must not alarm; got: {breach}'

    @pytest.mark.asyncio
    async def test_cap_is_read_once_per_root_not_once_per_task(
        self, burndown_conn_with_config, dummy_client,
    ):
        """The cap read touches the filesystem; it must not run per row."""
        root_a = Path('/fake/project/root_a')
        root_b = Path('/fake/project/root_b')

        async with burndown_conn_with_config(known_project_roots=[root_a, root_b]) as (
            db_path, config, conn,
        ):
            many = [_ztask(status='pending', id=i) for i in range(25)]
            cap_calls: list[str] = []

            def fake_cap(project_root):
                cap_calls.append(str(project_root))
                return 24

            with (
                _serve(_CannedStore({config.project_root: many, root_a: many, root_b: many})),
                patch('dashboard.data.burndown.read_max_concurrent_tasks', side_effect=fake_cap),
                patch('dashboard.data.burndown.find_running_orchestrators', return_value=[]),
            ):
                await collect_snapshot(conn, config, client=dummy_client)

            assert len(cap_calls) == 3, (
                f'expected one cap read per snapshotted root, got {cap_calls}'
            )
            assert set(cap_calls) == {
                str(config.project_root), str(root_a.resolve()), str(root_b.resolve()),
            }

    @pytest.mark.asyncio
    async def test_offline_marker_still_skips_the_project(
        self, burndown_env, caplog, dummy_client,
    ):
        """An unreachable root is routine: no zeroed row, and no WARNING here.

        The substrate's own fan-out reports the outage under its own logger;
        the collector's record of a known offline root is DEBUG-level.
        """
        db_path, config, conn = burndown_env

        with (
            _serve(_CannedStore({config.project_root: httpx.ConnectError('boom')})),
            patch('dashboard.data.burndown.find_running_orchestrators', return_value=[]),
            caplog.at_level(logging.WARNING, logger='dashboard.data.burndown'),
        ):
            await collect_snapshot(conn, config, client=dummy_client)

        async with conn.execute('SELECT COUNT(*) FROM snapshots') as cur:
            row = await cur.fetchone()
        assert row is not None
        assert row[0] == 0, 'an offline project must not write a zeroed snapshot'
        assert not [
            r for r in caplog.records
            if r.levelno == logging.WARNING and r.name == 'dashboard.data.burndown'
        ], 'a known offline root is a DEBUG-level record, not a WARNING'


# ---------------------------------------------------------------------------
# collect_snapshot — the census through the snapshot unit (task 5591)
# ---------------------------------------------------------------------------

# One column per TaskStatus member, written out literally: the test's own copy
# of the naming, so a production mapping that drifted would disagree with it.
_MEMBER_COLUMN = {
    'pending': 'pending',
    'in-progress': 'in_progress',
    'blocked': 'blocked',
    'deferred': 'deferred',
    'review': 'review',
    'merge-deferred': 'merge_deferred',
    'infra-hold': 'infra_hold',
    'done': 'done',
    'cancelled': 'cancelled',
}


async def _collect(conn, config, client, store: _CannedStore) -> None:
    """One collector cycle over *store*, with no orchestrators running."""
    with (
        _serve(store),
        patch('dashboard.data.burndown.find_running_orchestrators', return_value=[]),
    ):
        await collect_snapshot(conn, config, client=client)


async def _snapshot_rows(conn) -> list:
    """Every stored row, in insert order."""
    async with conn.execute('SELECT * FROM snapshots ORDER BY id') as cur:
        return list(await cur.fetchall())


class TestCollectSnapshotWritesTheCensus:
    """The collector writes a VALUE row read through the task snapshot unit.

    The REAL ``task_snapshot.acquire_snapshot`` runs underneath every test, fed
    by the canned substrate at the public MCP seam: the census gives the nine
    member columns, the rows give the live/stranded split.
    """

    async def test_each_of_the_nine_members_lands_in_its_own_column(
        self, burndown_env, dummy_client,
    ):
        _, config, conn = burndown_env
        fixture = [
            _raw_task(task_id, member.value)
            for task_id, member in enumerate(TaskStatus, start=1)
        ]

        await _collect(conn, config, dummy_client, _CannedStore({config.project_root: fixture}))

        rows = await _snapshot_rows(conn)
        assert len(rows) == 1
        row = rows[0]
        assert row['state'] == 'value'
        assert row['reason'] is None
        columns = [_MEMBER_COLUMN[member] for member in TaskStatus]
        assert {column: row[column] for column in columns} == dict.fromkeys(columns, 1), (
            'every member in its OWN column — review is no longer folded into in_progress'
        )
        assert sum(row[column] for column in columns) == len(fixture) == 9

    async def test_the_split_is_the_rows_and_the_members_are_the_census(
        self, burndown_env, dummy_client,
    ):
        """Two sources, disclosed: a status-map id with no row moves the census only."""
        _, config, conn = burndown_env
        store = _CannedStore(
            {config.project_root: [_live_now(id=1), _stranded(id=2), _ztask(status='pending', id=3)]},
            census_only={config.project_root: {4: 'in-progress'}},
        )

        await _collect(conn, config, dummy_client, store)

        (row,) = await _snapshot_rows(conn)
        assert row['in_progress'] == 3, 'the census counts the row-less id too'
        assert row['in_progress_rows'] == 2
        assert row['in_progress_live'] == 1
        assert row['in_progress_stranded'] == 1
        assert row['in_progress_live'] + row['in_progress_stranded'] == row['in_progress_rows']
        assert row['in_progress'] != row['in_progress_rows']
        assert row['pending'] == 1

    async def test_one_unit_per_root_and_never_the_whole_tree(
        self, burndown_conn_with_config, dummy_client,
    ):
        other = Path('/fake/project/other')
        async with burndown_conn_with_config(known_project_roots=[other]) as (_db, config, conn):
            store = _CannedStore({
                config.project_root: [_ztask(status='pending', id=1)],
                other: [_ztask(status='done', id=1)],
            })
            await _collect(conn, config, dummy_client, store)

        for root in (config.project_root, other):
            row_reads = [call['args'].get('statuses') for call in store.calls_to('get_tasks', root)]
            assert row_reads == [sorted(ACTIVE)], (
                f'{root}: exactly one status-narrowed row read, got {row_reads}'
            )
            assert store.calls_to('get_statuses', root), f'{root}: the census was never read'
        assert all('statuses' in call['args'] for call in store.calls_to('get_tasks')), (
            'a get_tasks call without a statuses filter reads the whole tree'
        )

    async def test_an_empty_project_is_a_measured_zero(self, burndown_env, dummy_client):
        _, config, conn = burndown_env

        await _collect(conn, config, dummy_client, _CannedStore({config.project_root: []}))

        (row,) = await _snapshot_rows(conn)
        assert row['state'] == 'value'
        for column in (*_MEMBER_COLUMN.values(), 'in_progress_rows',
                       'in_progress_live', 'in_progress_stranded'):
            assert row[column] == 0, f'{column}: expected a measured 0, got {row[column]!r}'

    async def test_a_migrated_pre_delta1_store_keeps_its_old_row_unrecorded(
        self, tmp_path, dummy_client,
    ):
        db = tmp_path / 'pre_delta1.db'
        _pre_delta1_burndown_db(db)
        config = DashboardConfig(project_root=tmp_path)
        store = _CannedStore({config.project_root: [_raw_task(1, 'review'), _raw_task(2, 'done')]})

        async with aiosqlite.connect(str(db)) as conn:
            conn.row_factory = aiosqlite.Row
            await ensure_snapshot_columns(conn)
            await conn.commit()
            await _collect(conn, config, dummy_client, store)
            old, new = await _snapshot_rows(conn)

        assert old['project_id'] == _PRE_DELTA1_ROW['project_id']
        for column in (*_DELTA1_COUNT_COLUMNS, 'state'):
            assert old[column] is None, f'old row {column}: expected NULL, got {old[column]!r}'
        assert new['state'] == 'value'
        assert (new['review'], new['merge_deferred'], new['infra_hold'], new['in_progress_rows']) == (
            1, 0, 0, 0,
        )
        assert new['done'] == 1


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
                assert row[0] == 6  # all preserved

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
                assert row[0] == 1  # compacted to one per hour

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
                assert row[0] == 1  # only the recent one


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
        # Exact inventory, so a key can neither vanish nor appear unnoticed.
        # The last three are task 3543's split + cap; see
        # tests/test_burndown_parity_alarm.py for their semantics.
        assert set(result.keys()) == {
            'labels', 'done', 'cancelled', 'blocked', 'deferred', 'in_progress', 'pending',
            'in_progress_live', 'in_progress_stranded', 'concurrency_cap',
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

_COUNT_COLUMNS = ('pending', 'in_progress', 'blocked', 'deferred', 'cancelled', 'done')


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

        async with conn.execute('SELECT * FROM snapshots') as cur:
            rows = list(await cur.fetchall())

        assert len(rows) == 1
        result = _assert_snapshot_counts(rows[0], pending=1, done=2)
        assert result is None

    @pytest.mark.parametrize('column', _COUNT_COLUMNS)
    @pytest.mark.asyncio
    async def test_raises_on_mismatched_count(self, burndown_env, column):
        """Helper raises AssertionError with per-column message when count doesn't match."""
        db_path, config, conn = burndown_env
        values = {col: (2 if col == column else 0) for col in _COUNT_COLUMNS}
        await conn.execute(
            'INSERT INTO snapshots (project_id, ts, pending, in_progress, blocked, deferred, cancelled, done) '
            'VALUES (?, ?, ?, ?, ?, ?, ?, ?)',
            (
                'test_proj', '2024-01-01T00:00:00',
                values['pending'], values['in_progress'], values['blocked'],
                values['deferred'], values['cancelled'], values['done'],
            ),
        )
        await conn.commit()

        async with conn.execute('SELECT * FROM snapshots') as cur:
            row = await cur.fetchone()

        with pytest.raises(AssertionError, match=rf'{column}: expected 3, got 2'):
            _assert_snapshot_counts(row, **{column: 3})  # actual=2, expected=3

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

        async with conn.execute('SELECT * FROM snapshots') as cur:
            row = await cur.fetchone()

        result = _assert_snapshot_counts(row)  # all defaults are 0
        assert result is None


# ---------------------------------------------------------------------------
# burndown_conn_with_config factory fixture validation
# ---------------------------------------------------------------------------


class TestBurndownConnWithConfig:
    @pytest.mark.asyncio
    async def test_factory_yields_valid_triple(self, burndown_conn_with_config, dummy_client):
        """Factory yields (db_path, config, conn) triple with a fresh burndown schema."""
        async with burndown_conn_with_config() as (db_path, config, conn):
            # (a) db_path is a Path and exists on disk
            assert isinstance(db_path, Path)
            assert db_path.exists()
            # (b) config is a DashboardConfig
            assert isinstance(config, DashboardConfig)
            # (c) conn can query snapshots table (empty on creation)
            async with conn.execute('SELECT COUNT(*) FROM snapshots') as cur:
                row = await cur.fetchone()
            assert row is not None
            assert row[0] == 0


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
        dummy_client,
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
            config.project_root: main_tasks,
            known_root.resolve(): known_tasks,
        }

        with contextlib.ExitStack() as stack:
            stack.enter_context(caplog.at_level(logging.WARNING, logger='dashboard.data.burndown'))
            stack.enter_context(
                _serve(_CannedStore(_tasks_map))
            )
            stack.enter_context(
                patch('dashboard.data.burndown.find_running_orchestrators', **find_orch_kwargs)
            )
            if secondary_target is not None:
                stack.enter_context(patch(secondary_target, **secondary_kwargs))
            # Must NOT raise despite the injected failure
            await collect_snapshot(conn, config, client=dummy_client)

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

    @pytest.mark.asyncio
    async def test_oserror_from_post_helper_resolve_emits_specific_warning(
        self, burndown_env, caplog, dummy_client,
    ):
        """An OSError while resolving one orchestrator's project root must emit
        the resolution-specific warning, not the generic 'processing failed'
        one, and must not prevent other entries from being snapshotted.

        The OSError is raised by the resolution helper itself for the bad
        entry, so it is discovery's own ``except OSError`` branch that must
        report it. (An earlier form patched ``Path.resolve`` and passed only
        because a LATER read-failure warning happened to name a root called
        ``orch_resolve_bad``.)
        """
        db_path, base_config, conn = burndown_env

        good_orch_root = Path('/fake/project/orch_good')
        config = DashboardConfig(
            project_root=base_config.project_root,
            known_project_roots=[],
        )
        store = _CannedStore({
            config.project_root: [{'status': 'pending'}],
            good_orch_root: [{'status': 'done'}],
        })

        def fake_resolve_project_root(prd_path, fallback):
            if 'bad' in str(prd_path):
                raise OSError('simulated project-root resolution failure')
            return good_orch_root

        orchestrator_entries = [
            {'prd': '/fake/good_prd.md', 'config_path': None},
            {'prd': '/fake/bad_prd.md', 'config_path': None},
        ]

        with (
            caplog.at_level(logging.WARNING, logger='dashboard.data.burndown'),
            _serve(store),
            patch('dashboard.data.burndown.find_running_orchestrators', return_value=orchestrator_entries),
            patch('dashboard.data.burndown._resolve_project_root', side_effect=fake_resolve_project_root),
        ):
            # Must NOT raise despite the injected OSError.
            await collect_snapshot(conn, config, client=dummy_client)

        async with conn.execute('SELECT project_id FROM snapshots') as cur:
            rows = list(await cur.fetchall())

        project_ids = {row['project_id'] for row in rows}
        # Two rows: main project + good orchestrator entry
        assert len(rows) == 2, f'Expected 2 rows, got {len(rows)}: {project_ids}'
        assert str(base_config.project_root) in project_ids
        assert str(good_orch_root.resolve()) in project_ids

        # Warning contract: the resolution-specific message (NOT the generic
        # 'processing failed' one), with exc_info set.
        warning_records = [
            r for r in caplog.records
            if r.levelno == logging.WARNING and r.name == 'dashboard.data.burndown'
        ]
        resolving = [r for r in warning_records if 'resolv' in r.getMessage().lower()]
        assert resolving, (
            'Expected a WARNING about resolving the project root, got: '
            f'{[r.getMessage() for r in warning_records]!r}'
        )
        assert all(r.exc_info for r in resolving)
        assert not [r for r in warning_records if 'processing failed' in r.getMessage()]


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
        self, burndown_env, caplog, dummy_client,
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
            config.project_root: main_tasks,
            extra_root.resolve(): extra_tasks,
        }

        extra_id = str(extra_root.resolve())
        original_execute = conn.execute
        wrapper = self._make_execute_wrapper(original_execute, {extra_id})
        conn.execute = wrapper
        try:
            with (
                _serve(_CannedStore(_tasks_map)),
                patch('dashboard.data.burndown.find_running_orchestrators', return_value=[]),
                caplog.at_level(logging.WARNING, logger='dashboard.data.burndown'),
            ):
                # Must NOT raise despite the extra's INSERT failing
                await collect_snapshot(conn, config, client=dummy_client)
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
        self, burndown_env, caplog, dummy_client,
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
            config.project_root: main_tasks,
            extra_a.resolve(): extra_a_tasks,
            extra_b.resolve(): extra_b_tasks,
            extra_c.resolve(): extra_c_tasks,
        }

        extra_b_id = str(extra_b.resolve())
        original_execute = conn.execute
        wrapper = self._make_execute_wrapper(original_execute, {extra_b_id})
        conn.execute = wrapper
        try:
            with (
                _serve(_CannedStore(_tasks_map)),
                patch('dashboard.data.burndown.find_running_orchestrators', return_value=[]),
                caplog.at_level(logging.WARNING, logger='dashboard.data.burndown'),
            ):
                await collect_snapshot(conn, config, client=dummy_client)
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
            if r.levelno == logging.WARNING
            and r.name == 'dashboard.data.burndown'
            and (
                'insert_multi_b' in r.getMessage() or extra_b_id in r.getMessage()
            )
        ]
        assert len(warning_records) == 1, (
            f'Expected exactly 1 WARNING naming extra_b, got {len(warning_records)}'
        )

    @pytest.mark.asyncio
    async def test_db_error_on_main_insert_preserves_extras(
        self, burndown_env, caplog, dummy_client,
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
            config.project_root: main_tasks,
            extra_root.resolve(): extra_tasks,
        }

        main_id = str(base_config.project_root)
        original_execute = conn.execute
        wrapper = self._make_execute_wrapper(original_execute, {main_id})
        conn.execute = wrapper
        try:
            with (
                _serve(_CannedStore(_tasks_map)),
                patch('dashboard.data.burndown.find_running_orchestrators', return_value=[]),
                caplog.at_level(logging.WARNING, logger='dashboard.data.burndown'),
            ):
                await collect_snapshot(conn, config, client=dummy_client)
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
    async def test_explicit_rollback_on_unexpected_error(self, burndown_env, dummy_client):
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
                await collect_snapshot(conn, config, client=dummy_client)
        finally:
            conn.rollback = original_rollback

        # (b) rollback must have been called before the re-raise
        assert rollback_calls, (
            'Expected conn.rollback() to be called before re-raising the RuntimeError'
        )


# ---------------------------------------------------------------------------
# Per-root whole-operation budget (task 4884 / #4424)
# ---------------------------------------------------------------------------


# The ~209 s paginated worst case measured for ONE root of this repo's size —
# ``ceil(N/_SNAPSHOT_PAGE_SIZE)`` sequential round trips at ~0.33-0.35 s each.
# The measurement and its derivation live on ``burndown._SNAPSHOT_PAGE_SIZE``;
# this is the test's own copy of the FLOOR it enforces, deliberately spelled
# as a number rather than derived from ``_SNAPSHOT_PER_ROOT_BUDGET`` (deriving
# the bound from the value under test would assert nothing).
_MEASURED_PAGINATED_WORST_CASE = 209.0


class TestCollectSnapshotPerRootBudget:
    """``collect_snapshot``'s Phase-2 fan-out is whole-operation bounded per root.

    Before task 4884 this was the LAST unbounded ``fetch_tasks`` caller in the
    tree: task 4788 gave every route caller a named
    ``DEFAULT_WHOLE_OPERATION_BUDGET`` wrap and deliberately left the
    background collector out of scope, so a single hung MCP fan-out could park
    the collector task forever and silently stop every project's burndown row
    — the 2026-08-27 19.8h wedge shape, one layer down from the routes.

    The hang stub is ``await asyncio.Event().wait()`` — the 4788 idiom — with
    NO duration at all, so no choice of budget value can make an unbounded
    implementation pass these tests.
    """

    @pytest.mark.asyncio
    async def test_hung_root_is_skipped_and_healthy_root_still_snapshots(
        self, tmp_path, burndown_conn_with_config, dummy_client, caplog, monkeypatch,
    ):
        """(a) One root hanging forever costs that root's row, not the cycle."""
        healthy_root = tmp_path / 'healthy'
        healthy_root.mkdir()

        async with burndown_conn_with_config(known_project_roots=[healthy_root]) as (
            _db_path, config, conn,
        ):
            hung_key = _root_key(config.project_root)
            store = _CannedStore({healthy_root: [{'status': 'pending'}, {'status': 'done'}]})

            async def hang_or_serve(client, url, tool, args, **kwargs):
                if _root_key(args['project_root']) == hung_key:
                    # No duration: nothing ever sets this Event, so the only
                    # thing that can end this await is a caller's bound.
                    await asyncio.Event().wait()
                return await store(client, url, tool, args, **kwargs)

            monkeypatch.setattr(burndown_module, '_SNAPSHOT_PER_ROOT_BUDGET', 0.05)

            with (
                patch('dashboard.data.tasks.mcp_tool_call', new=hang_or_serve),
                patch('dashboard.data.burndown.find_running_orchestrators', return_value=[]),
                caplog.at_level(logging.WARNING, logger='dashboard.data.burndown'),
            ):
                # Hard external cap, mirroring test_healthz_deadline._call_healthz:
                # a regression to the unbounded gather must fail fast here rather
                # than wedge the whole suite behind pytest-timeout.
                try:
                    await asyncio.wait_for(
                        collect_snapshot(conn, config, client=dummy_client), timeout=10,
                    )
                except TimeoutError:
                    pytest.fail(
                        'collect_snapshot did not return within 10s against a root '
                        'whose read hangs forever — the Phase-2 fan-out is still '
                        'unbounded (expected a per-root _SNAPSHOT_PER_ROOT_BUDGET '
                        'wrap around the per-root read).'
                    )

            async with conn.execute('SELECT project_id FROM snapshots') as cur:
                rows = list(await cur.fetchall())

            project_ids = {row['project_id'] for row in rows}
            assert str(healthy_root.resolve()) in project_ids, (
                'the healthy root must still get its snapshot row: one hung root '
                'may not sink the cycle'
            )
            assert str(config.project_root) not in project_ids, (
                'the hung root must be SKIPPED, not written with fabricated zero '
                'counts — snapshots is an append-only historical record'
            )
            assert len(rows) == 1

            warnings = [
                r for r in caplog.records
                if r.levelno >= logging.WARNING and str(config.project_root) in r.getMessage()
            ]
            assert warnings, (
                'expected a WARNING naming the hung root; a root that silently '
                'vanishes from an append-only chart is exactly the invisible '
                'failure this bound exists to make visible. Saw: '
                f'{[r.getMessage() for r in caplog.records]}'
            )

    @pytest.mark.asyncio
    async def test_expiry_surfaces_through_the_existing_exception_triage(
        self, tmp_path, burndown_conn_with_config, dummy_client, caplog, monkeypatch,
    ):
        """(b) Task 519's partial-success semantics are preserved.

        Expiry must arrive at Phase 3 as a per-root ``BaseException`` in the
        gather's result list — i.e. ``return_exceptions=True`` is still in
        force — so the exception branch logs-and-continues.  A bound that
        instead let the ``TimeoutError`` escape the gather would abort the
        cycle and roll nothing back but write nothing more either.
        """
        healthy_root = tmp_path / 'healthy'
        healthy_root.mkdir()
        late_root = tmp_path / 'late'
        late_root.mkdir()

        async with burndown_conn_with_config(
            known_project_roots=[healthy_root, late_root],
        ) as (_db_path, config, conn):
            hung_key = _root_key(healthy_root)
            store = _CannedStore({
                config.project_root: [{'status': 'pending'}],
                late_root: [{'status': 'pending'}],
            })

            async def hang_or_serve(client, url, tool, args, **kwargs):
                if _root_key(args['project_root']) == hung_key:
                    await asyncio.Event().wait()
                return await store(client, url, tool, args, **kwargs)

            monkeypatch.setattr(burndown_module, '_SNAPSHOT_PER_ROOT_BUDGET', 0.05)

            with (
                patch('dashboard.data.tasks.mcp_tool_call', new=hang_or_serve),
                patch('dashboard.data.burndown.find_running_orchestrators', return_value=[]),
                caplog.at_level(logging.WARNING, logger='dashboard.data.burndown'),
            ):
                try:
                    await asyncio.wait_for(
                        collect_snapshot(conn, config, client=dummy_client), timeout=10,
                    )
                except TimeoutError:
                    pytest.fail('collect_snapshot did not return within 10s (see (a))')

            async with conn.execute('SELECT project_id FROM snapshots') as cur:
                project_ids = {row['project_id'] for row in await cur.fetchall()}

            # The main project commits FIRST (it is roots_to_snapshot[0]) and a
            # later root's expiry may not roll it back.  The root AFTER the hung
            # one is still reached, which is what "isolated, not aborted" means.
            assert str(config.project_root) in project_ids
            assert str(late_root.resolve()) in project_ids
            assert str(healthy_root.resolve()) not in project_ids

            triage = [
                r for r in caplog.records
                if r.name == 'dashboard.data.burndown'
                and r.levelno == logging.WARNING
                and str(healthy_root.resolve()) in r.getMessage()
                and r.exc_info is not None
            ]
            assert triage, (
                'the timed-out root must reach the exception triage branch '
                '(a WARNING naming it, with exc_info), proving '
                'return_exceptions=True still converts its expiry into a '
                'handled per-root result. Saw: '
                f'{[r.getMessage() for r in caplog.records]}'
            )
            assert triage[0].exc_info is not None
            assert issubclass(triage[0].exc_info[0], TimeoutError), (
                'the exception carried into triage must be the budget expiry '
                f'itself, not a substitute; got {triage[0].exc_info[0]!r}'
            )

    def test_budget_is_sized_between_the_route_default_and_one_collector_cycle(self):
        """(c) The VALUE is derived from the collector's own cycle, not the routes.

        Wrapping at the route convention's 7.0 would be a REGRESSION, not a
        fix: ``_fetch_snapshot_tasks`` probes unpaginated and, on transport
        rejection, falls back to ``fetch_tasks(..., paginate=True)`` — ONE call
        that internally walks ``ceil(N/_SNAPSHOT_PAGE_SIZE)`` SEQUENTIAL round
        trips, MEASURED at ~209 s for one root of this repo's size (the
        measurement and its derivation live on ``_SNAPSHOT_PAGE_SIZE``).  A
        7.0 s bound would time out every big root on every cycle, and because
        ``snapshots`` is APPEND-ONLY and no later cycle backfills, that is a
        permanent unexplained hole in the chart.
        """
        import dashboard.loops as loops_module
        from dashboard.data.tasks import DEFAULT_WHOLE_OPERATION_BUDGET

        budget = burndown_module._SNAPSHOT_PER_ROOT_BUDGET
        one_cycle_half = loops_module._SAMPLE_INTERVAL_SECONDS / 2

        # BOTH bounds are the ones the derivation actually names. An earlier
        # form of this test asserted `>= DEFAULT_WHOLE_OPERATION_BUDGET` (7.0)
        # and `< _SAMPLE_INTERVAL_SECONDS` (600) — two orders of magnitude
        # apart from the stated floor at one end and double the stated ceiling
        # at the other — so a value of 10.0 passed while doing exactly the
        # permanent-hole damage the docstring describes.
        assert budget >= _MEASURED_PAGINATED_WORST_CASE, (
            f'_SNAPSHOT_PER_ROOT_BUDGET ({budget}) is below the MEASURED '
            f'~{_MEASURED_PAGINATED_WORST_CASE} s paginated worst case for one '
            "root of this repo's size (see _SNAPSHOT_PAGE_SIZE). Anything below "
            'it times out every big root on EVERY cycle, and because snapshots '
            'is APPEND-ONLY and no later cycle backfills, that is a permanent '
            'unexplained hole in the chart — not a degraded read. The shared '
            f'route default ({DEFAULT_WHOLE_OPERATION_BUDGET}) is the value '
            'this must NOT be confused with: a BACKGROUND root that walks '
            'ceil(N/P) sequential pages is not a request-path one.'
        )
        assert budget <= one_cycle_half, (
            f'_SNAPSHOT_PER_ROOT_BUDGET ({budget}) is above HALF one collector '
            f'cycle ({one_cycle_half} s = _SAMPLE_INTERVAL_SECONDS / 2 = '
            f'{loops_module._SAMPLE_INTERVAL_SECONDS} / 2). A root that cannot '
            'finish inside one cycle can never finish at all, and the halving '
            'is what guarantees cycle N is done before cycle N+1 starts even '
            'when a root spends its whole budget.'
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


# ---------------------------------------------------------------------------
# Tests: aggregate_burndown_projects
# ---------------------------------------------------------------------------


def _make_burndown_db_with_projects(tmp_path: Path, name: str, project_ids: list[str]) -> Path:
    """Create a burndown DB at tmp_path/name with one snapshot row per project_id."""
    from datetime import UTC, datetime
    tmp_path.mkdir(parents=True, exist_ok=True)
    db_path = tmp_path / name
    _create_burndown_db(db_path)
    conn = sqlite3.connect(str(db_path))
    ts = datetime.now(UTC).isoformat()
    for pid in project_ids:
        _insert_snapshot(conn, pid, ts, done=1)
    conn.commit()
    conn.close()
    return db_path


class TestAggregateBurndownProjects:
    """Tests for aggregate_burndown_projects across multiple DB connections."""

    @pytest.mark.asyncio
    async def test_union_across_two_dbs(self, tmp_path):
        """project_ids from both DBs are unioned."""
        db1_path = _make_burndown_db_with_projects(tmp_path / 'db1', 'b.db', ['proj_a'])
        db2_path = _make_burndown_db_with_projects(tmp_path / 'db2', 'b.db', ['proj_b'])
        async with (
            aiosqlite.connect(str(db1_path)) as c1,
            aiosqlite.connect(str(db2_path)) as c2,
        ):
            result = await aggregate_burndown_projects([c1, c2])
        assert 'proj_a' in result
        assert 'proj_b' in result

    @pytest.mark.asyncio
    async def test_dedup_same_project_in_both_dbs(self, tmp_path):
        """A project_id that appears in both DBs is included only once."""
        db1_path = _make_burndown_db_with_projects(tmp_path / 'db1', 'b.db', ['shared', 'only_1'])
        db2_path = _make_burndown_db_with_projects(tmp_path / 'db2', 'b.db', ['shared', 'only_2'])
        async with (
            aiosqlite.connect(str(db1_path)) as c1,
            aiosqlite.connect(str(db2_path)) as c2,
        ):
            result = await aggregate_burndown_projects([c1, c2])
        assert result == ['only_1', 'only_2', 'shared']

    @pytest.mark.asyncio
    async def test_empty_dbs_list_returns_empty(self):
        """`dbs=[]` returns []."""
        result = await aggregate_burndown_projects([])
        assert result == []

    @pytest.mark.asyncio
    async def test_none_dbs_returns_empty(self):
        """`dbs=[None, None]` returns []."""
        result = await aggregate_burndown_projects([None, None])
        assert result == []


# ---------------------------------------------------------------------------
# Tests: aggregate_burndown_series
# ---------------------------------------------------------------------------


def _make_burndown_db_with_series(
    tmp_path: Path,
    name: str,
    project_id: str,
    rows: list[tuple],
) -> Path:
    """Create a burndown DB with the given (ts, done, cancelled, blocked,
    deferred, in_progress, pending) rows for *project_id*.
    """
    tmp_path.mkdir(parents=True, exist_ok=True)
    db_path = tmp_path / name
    _create_burndown_db(db_path)
    conn = sqlite3.connect(str(db_path))
    for ts, done, cancelled, blocked, deferred, in_progress, pending in rows:
        _insert_snapshot(
            conn, project_id, ts, pending=pending, in_progress=in_progress,
            blocked=blocked, deferred=deferred, cancelled=cancelled, done=done,
        )
    conn.commit()
    conn.close()
    return db_path


def _recent_ts(offset_seconds: int = 0) -> str:
    """Return an ISO timestamp that falls within the last 30 days."""
    return (datetime.now(UTC) - timedelta(seconds=offset_seconds)).isoformat()


class TestAggregateBurndownSeries:
    """Tests for aggregate_burndown_series across multiple DB connections."""

    @pytest.mark.asyncio
    async def test_union_timestamps_sorted(self, tmp_path):
        """Timestamps from both DBs for the same project are unioned and sorted."""
        ts1 = _recent_ts(offset_seconds=200)
        ts2 = _recent_ts(offset_seconds=100)
        db1_path = _make_burndown_db_with_series(
            tmp_path / 'db1', 'b.db', 'proj_a', [(ts1, 1, 0, 0, 0, 0, 0)]
        )
        db2_path = _make_burndown_db_with_series(
            tmp_path / 'db2', 'b.db', 'proj_a', [(ts2, 2, 0, 0, 0, 0, 0)]
        )
        async with (
            aiosqlite.connect(str(db1_path)) as c1,
            aiosqlite.connect(str(db2_path)) as c2,
        ):
            result = await aggregate_burndown_series([c1, c2], 'proj_a', days=30)
        assert ts1 in result['labels']
        assert ts2 in result['labels']
        # labels are sorted
        assert result['labels'] == sorted(result['labels'])

    @pytest.mark.asyncio
    async def test_unique_timestamp_uses_its_values(self, tmp_path):
        """A timestamp present in only one DB uses values from that DB."""
        ts = _recent_ts(offset_seconds=100)
        db1_path = _make_burndown_db_with_series(
            tmp_path / 'db1', 'b.db', 'proj_a', [(ts, 5, 1, 2, 3, 4, 6)]
        )
        db2_path = _make_burndown_db_with_series(
            tmp_path / 'db2', 'b.db', 'proj_a', []
        )
        async with (
            aiosqlite.connect(str(db1_path)) as c1,
            aiosqlite.connect(str(db2_path)) as c2,
        ):
            result = await aggregate_burndown_series([c1, c2], 'proj_a', days=30)
        idx = result['labels'].index(ts)
        assert result['done'][idx] == 5
        assert result['cancelled'][idx] == 1
        assert result['blocked'][idx] == 2
        assert result['deferred'][idx] == 3
        assert result['in_progress'][idx] == 4
        assert result['pending'][idx] == 6

    @pytest.mark.asyncio
    async def test_duplicate_timestamp_last_writer_wins(self, tmp_path):
        """When both DBs have the same timestamp, the last DB in the list wins.

        This documents the last-writer-wins contract: the aggregate function
        iterates DBs in order, and later entries overwrite earlier ones for the
        same timestamp key.
        """
        ts = _recent_ts(offset_seconds=100)
        db1_path = _make_burndown_db_with_series(
            tmp_path / 'db1', 'b.db', 'proj_a', [(ts, 10, 0, 0, 0, 0, 0)]
        )
        db2_path = _make_burndown_db_with_series(
            tmp_path / 'db2', 'b.db', 'proj_a', [(ts, 99, 0, 0, 0, 0, 0)]
        )
        async with (
            aiosqlite.connect(str(db1_path)) as c1,
            aiosqlite.connect(str(db2_path)) as c2,
        ):
            # c2 is last; its done=99 should win over c1's done=10
            result = await aggregate_burndown_series([c1, c2], 'proj_a', days=30)
        idx = result['labels'].index(ts)
        assert result['done'][idx] == 99

    @pytest.mark.asyncio
    async def test_no_data_returns_empty_default(self, tmp_path):
        """project_id with no rows in any DB returns the empty-series default."""
        db1_path = _make_burndown_db_with_series(
            tmp_path / 'db1', 'b.db', 'other_project', [(_recent_ts(), 1, 0, 0, 0, 0, 0)]
        )
        async with aiosqlite.connect(str(db1_path)) as c1:
            result = await aggregate_burndown_series([c1], 'proj_missing', days=30)
        assert result['labels'] == []
        assert result['done'] == []
        assert result['pending'] == []

    @pytest.mark.asyncio
    async def test_days_window_filters(self, tmp_path):
        """Only rows within the days window are included."""
        ts_recent = _recent_ts(offset_seconds=100)
        ts_old = (datetime.now(UTC) - timedelta(days=100)).isoformat()
        db_path = _make_burndown_db_with_series(
            tmp_path / 'db1', 'b.db', 'proj_a',
            [(ts_recent, 3, 0, 0, 0, 0, 0), (ts_old, 99, 0, 0, 0, 0, 0)],
        )
        async with aiosqlite.connect(str(db_path)) as c1:
            result = await aggregate_burndown_series([c1], 'proj_a', days=30)
        assert ts_recent in result['labels']
        assert ts_old not in result['labels']

    @pytest.mark.asyncio
    async def test_empty_dbs_list_returns_empty(self):
        """`dbs=[]` returns the empty-series default."""
        result = await aggregate_burndown_series([], 'proj_a', days=30)
        assert result['labels'] == []
        assert result['done'] == []

    @pytest.mark.asyncio
    async def test_none_dbs_returns_empty(self):
        """`dbs=[None, None]` returns the empty-series default."""
        result = await aggregate_burndown_series([None, None], 'proj_a', days=30)
        assert result['labels'] == []
        assert result['done'] == []

    @pytest.mark.asyncio
    async def test_multiple_collisions_emit_single_summary_warning(self, tmp_path, caplog):
        """Multiple timestamp collisions across two DBs emit exactly ONE WARNING summary.

        Current code (before the fix) emits one warning per collision, so with 3
        colliding timestamps there are 3 WARNING records.  After the fix, one
        aggregated record with the collision count should appear.
        """
        ts1 = _recent_ts(offset_seconds=300)
        ts2 = _recent_ts(offset_seconds=200)
        ts3 = _recent_ts(offset_seconds=100)

        # Both DBs have the same three timestamps for 'proj_a' — 3 collisions.
        db1_path = _make_burndown_db_with_series(
            tmp_path / 'db1', 'b.db', 'proj_a',
            [(ts1, 1, 0, 0, 0, 0, 0), (ts2, 2, 0, 0, 0, 0, 0), (ts3, 3, 0, 0, 0, 0, 0)],
        )
        db2_path = _make_burndown_db_with_series(
            tmp_path / 'db2', 'b.db', 'proj_a',
            [(ts1, 10, 0, 0, 0, 0, 0), (ts2, 20, 0, 0, 0, 0, 0), (ts3, 30, 0, 0, 0, 0, 0)],
        )

        with caplog.at_level(logging.WARNING, logger='dashboard.data.burndown'):
            async with (
                aiosqlite.connect(str(db1_path)) as c1,
                aiosqlite.connect(str(db2_path)) as c2,
            ):
                result = await aggregate_burndown_series([c1, c2], 'proj_a', days=30)

        # (a) Exactly ONE WARNING record mentioning timestamp collisions.
        collision_warnings = [
            r for r in caplog.records
            if r.levelno == logging.WARNING and 'timestamp collisions' in r.getMessage()
        ]
        assert len(collision_warnings) == 1, (
            f'Expected 1 collision summary WARNING, got {len(collision_warnings)}: '
            + str([r.getMessage() for r in collision_warnings])
        )

        # (b) The single record's message contains the collision count and project_id.
        msg = collision_warnings[0].getMessage()
        assert '3 timestamp collisions' in msg, (
            f'Expected "3 timestamp collisions" in warning message: {msg!r}'
        )
        assert 'proj_a' in msg, f'Expected project_id "proj_a" in warning message: {msg!r}'

        # (c) Last-writer-wins: db2 values (10, 20, 30) win over db1 values (1, 2, 3).
        idx1 = result['labels'].index(ts1)
        assert result['done'][idx1] == 10
        idx3 = result['labels'].index(ts3)
        assert result['done'][idx3] == 30


# ---------------------------------------------------------------------------
# TestBurndownNowThreading (step-1)
# ---------------------------------------------------------------------------


_EMPTY_SERIES: dict = {
    'labels': [],
    'done': [],
    'cancelled': [],
    'blocked': [],
    'deferred': [],
    'in_progress': [],
    'pending': [],
}


class TestBurndownNowThreading:
    """get_burndown_series and aggregate_burndown_series must accept an
    injected `now` and, in the aggregate case, resolve it once and thread
    the identical value to every per-DB call — closing the per-DB
    clock-skew race (mirrors costs.py's aggregate_cost_summary pattern).
    """

    @pytest.mark.asyncio
    async def test_get_burndown_series_respects_injected_now(self, tmp_path):
        """now=base anchors the cutoff instead of the live clock."""
        db_path = tmp_path / 'burndown.db'
        _create_burndown_db(db_path)

        base = datetime(2026, 3, 1, tzinfo=UTC)
        sync_conn = sqlite3.connect(str(db_path))
        _insert_snapshot(sync_conn, 'proj', (base - timedelta(days=2)).isoformat(), done=1)
        _insert_snapshot(sync_conn, 'proj', (base - timedelta(days=10)).isoformat(), done=2)
        sync_conn.commit()
        sync_conn.close()

        async with aiosqlite.connect(str(db_path)) as conn:
            conn.row_factory = aiosqlite.Row
            result = await get_burndown_series(conn, 'proj', days=7, now=base)

        assert len(result['labels']) == 1, (
            f"expected only the base-2d row within the 7d window, got {result['labels']!r}"
        )
        assert result['done'] == [1]

    @pytest.mark.asyncio
    async def test_aggregate_burndown_series_threads_shared_now(self):
        """An injected now= is threaded unchanged to every per-DB call."""
        base = datetime(2026, 3, 1, tzinfo=UTC)
        mock = AsyncMock(return_value=_EMPTY_SERIES)
        with patch('dashboard.data.burndown.get_burndown_series', new=mock):
            await aggregate_burndown_series([None, None], 'proj', days=7, now=base)

        assert mock.await_count == 2
        nows = [call.kwargs.get('now') for call in mock.await_args_list]
        assert nows == [base, base], (
            f"expected both per-DB calls to receive now={base!r}, got {nows!r}"
        )

    @pytest.mark.asyncio
    async def test_aggregate_burndown_series_resolves_now_once_when_none(self):
        """Without now=, the aggregator resolves once and shares one value
        across both per-DB calls (no per-DB clock skew)."""
        mock = AsyncMock(return_value=_EMPTY_SERIES)
        with patch('dashboard.data.burndown.get_burndown_series', new=mock):
            await aggregate_burndown_series([None, None], 'proj', days=7)

        assert mock.await_count == 2
        nows = [call.kwargs.get('now') for call in mock.await_args_list]
        for n in nows:
            assert n is not None, f'expected non-None now, got {nows!r}'
            assert n.tzinfo is not None, f'expected tz-aware now, got {nows!r}'
        assert nows[0] == nows[1], (
            f"expected both per-DB calls to share one resolved now, got {nows!r}"
        )


# ---------------------------------------------------------------------------
# compute_window_completion
# ---------------------------------------------------------------------------


class TestComputeWindowCompletion:
    """Tests for the compute_window_completion pure helper."""

    def test_monotonic_done_over_7_distinct_days(self):
        """Delta of [0, 0, 1, 1, 1, 2, 3] across 7 distinct days → completed=3, velocity=3/7."""
        labels = [f'2026-05-{19 + i:02d}T00:00:00' for i in range(7)]
        done = [0, 0, 1, 1, 1, 2, 3]
        result = compute_window_completion({'labels': labels, 'done': done, 'pending': [10] * 7})
        assert result['completed'] == 3
        assert abs(result['velocity'] - 3 / 7) < 1e-9

    def test_reopened_task_clamps_at_zero(self):
        """When done dips at the end [5, 6, 7, 6], completed = max(0, 6-5) = 1."""
        labels = [f'2026-05-{20 + i:02d}T00:00:00' for i in range(4)]
        done = [5, 6, 7, 6]
        result = compute_window_completion({'labels': labels, 'done': done, 'pending': [10] * 4})
        assert result['completed'] == 1
        assert abs(result['velocity'] - 1 / 4) < 1e-9

    def test_empty_labels_returns_zeros(self):
        """Empty series → completed=0, velocity=0.0."""
        result = compute_window_completion({'labels': [], 'done': [], 'pending': []})
        assert result['completed'] == 0
        assert result['velocity'] == 0.0

    def test_single_snapshot_returns_zeros(self):
        """Single snapshot has no delta → completed=0, velocity=0.0."""
        result = compute_window_completion({'labels': ['2026-05-20T00:00:00'], 'done': [42], 'pending': [5]})
        assert result['completed'] == 0
        assert result['velocity'] == 0.0

    def test_flat_series_regression(self):
        """100 snapshots all done=5 in one day → completed=0, velocity=0.0.

        Regression: the buggy frontend code would yield sum(5*100)/100 = 5/day.
        The correct delta-based answer is max(0, 5-5) = 0.
        """
        labels = [f'2026-05-20T{h:02d}:{m:02d}:00' for h in range(0, 10) for m in range(0, 10)][:100]
        done = [5] * 100
        result = compute_window_completion({'labels': labels, 'done': done, 'pending': [10] * 100})
        assert result['completed'] == 0
        assert result['velocity'] == 0.0

    def test_mismatched_lengths_returns_zeros(self):
        """len(labels) != len(done) is a guard branch — must return zeros.

        Mismatched lengths can occur if a series is partially written or truncated.
        The function must not raise; it returns the safe zero dict.
        """
        result = compute_window_completion({
            'labels': ['2026-05-20T00:00:00', '2026-05-21T00:00:00', '2026-05-22T00:00:00'],
            'done': [1, 2],  # one entry short
        })
        assert result['completed'] == 0
        assert result['velocity'] == 0.0
        assert result['window_days'] == 0

    def test_none_values_in_done_coerced(self):
        """None entries in done[] are coerced to 0 via ``x or 0`` guard.

        done[0] = None means the first snapshot had no count recorded.
        done[-1] = 5 → completed = max(0, 5 - 0) = 5.
        """
        result = compute_window_completion({
            'labels': ['2026-05-20T00:00:00', '2026-05-21T00:00:00'],
            'done': [None, 5],
        })
        assert result['completed'] == 5
        assert abs(result['velocity'] - 5 / 2) < 1e-9  # 2 distinct days

    def test_none_at_end_clamps_to_zero(self):
        """None at done[-1] is coerced to 0: max(0, 0 - 3) = 0 (clamped)."""
        result = compute_window_completion({
            'labels': ['2026-05-20T00:00:00', '2026-05-21T00:00:00'],
            'done': [3, None],
        })
        assert result['completed'] == 0
        assert result['velocity'] == 0.0
