"""``scripts/roll_up_journal_reads.py``: the once-only move of history into read_op_counts.

Every test runs the script's ``main`` against a real ``WriteJournal``-initialized
journal under ``tmp_path``, seeded by direct INSERT with history older than the
deploy grace, the way an operator would run it against the live file.
"""

from __future__ import annotations

import asyncio
import json
import sqlite3
import subprocess
import time
from collections import Counter
from datetime import UTC, datetime, timedelta
from pathlib import Path

import pytest
from _fm_helpers import load_script_module

from fused_memory.config.schema import WriteJournalConfig
from fused_memory.services.read_rollup import PREWARM_PRIMARY_KEY_SQL, keeps_read_row
from fused_memory.services.write_journal import SCHEMA_SQL, WriteJournal
from fused_memory.services.write_ops_retention import prune_write_ops_pass

SCRIPT_PATH = Path(__file__).parent.parent / 'scripts' / 'roll_up_journal_reads.py'

_script = load_script_module(SCRIPT_PATH, mod_name='roll_up_journal_reads')

_WINDOW_END = (datetime.now(UTC) - timedelta(days=1)).replace(
    minute=0, second=0, microsecond=0
)
_ROLLUP_HOUR = _WINDOW_END - timedelta(hours=6)
_PRE_EXISTING_ROLLUP = (_ROLLUP_HOUR.isoformat(), 'dark_factory', '', 'get_task', 3)


def _initialized_journal(path: Path) -> Path:
    async def _init() -> Path:
        journal = WriteJournal(path)
        await journal.initialize()
        await journal.close()
        return journal.db_path

    return asyncio.run(_init())


def _history_rows() -> list[dict]:
    """Interleaved history over 30 hours before _WINDOW_END, three rows an hour
    apart in minutes so rowid order is time order."""
    rows = []
    for i in range(90):
        at = _WINDOW_END - timedelta(hours=30) + timedelta(minutes=20 * i + 3)
        shape = i % 6
        if shape == 5:
            kind, operation, success = 'write', 'add_memory', 1
        elif shape == 4:
            kind, operation, success = 'read', 'search', 1
        elif shape == 3 and i % 12 == 3:
            kind, operation, success = 'read', 'get_entity', 0
        else:
            kind, operation, success = 'read', ('get_task', 'get_statuses', None)[i % 3], 1
        rows.append({
            'id': f'op-{i}',
            'operation': operation,
            'project_id': ('dark_factory', 'reify')[(i // 4) % 2],
            'agent_id': (None, 'agent-a')[(i // 7) % 2],
            'kind': kind,
            'success': success,
            'error': None if success else 'boom',
            'params': '{"query": "q"}',
            'created_at': at.isoformat(),
        })
    return rows


def _insert_rows(db_path: Path, rows: list[dict]) -> None:
    conn = sqlite3.connect(db_path)
    try:
        conn.executemany(
            'INSERT INTO write_ops (id, operation, project_id, agent_id, kind, success, error,'
            ' params, created_at) VALUES (:id, :operation, :project_id, :agent_id, :kind,'
            ' :success, :error, :params, :created_at)',
            rows,
        )
        conn.commit()
    finally:
        conn.close()


def _seed_rollup(db_path: Path, *rollup_rows: tuple) -> None:
    conn = sqlite3.connect(db_path)
    try:
        conn.executemany(
            'INSERT INTO read_op_counts (hour_start, project_id, agent_id, operation, count)'
            ' VALUES (?, ?, ?, ?, ?)',
            rollup_rows,
        )
        conn.commit()
    finally:
        conn.close()


@pytest.fixture
def journal(tmp_path) -> Path:
    db_path = _initialized_journal(tmp_path / 'journal')
    _insert_rows(db_path, _history_rows())
    _seed_rollup(db_path, _PRE_EXISTING_ROLLUP)
    return db_path


def _is_count_only(row: dict) -> bool:
    return row['kind'] == 'read' and not keeps_read_row(
        operation=row['operation'], success=bool(row['success'])
    )


def _rollup_key(row: dict) -> tuple:
    at = datetime.fromisoformat(row['created_at']).replace(minute=0, second=0, microsecond=0)
    return (at.isoformat(), row['project_id'] or '', row['agent_id'] or '', row['operation'] or '')


def _dump(db_path: Path) -> list[str]:
    conn = sqlite3.connect(db_path)
    try:
        return list(conn.iterdump())
    finally:
        conn.close()


def _rows(db_path: Path, sql: str) -> list[tuple]:
    conn = sqlite3.connect(db_path)
    try:
        return conn.execute(sql).fetchall()
    finally:
        conn.close()


def _rollup_counts(db_path: Path) -> Counter:
    return Counter({
        tuple(key): count
        for *key, count in _rows(
            db_path,
            'SELECT hour_start, project_id, agent_id, operation, count FROM read_op_counts',
        )
    })


def _write_ops_by_id(db_path: Path) -> dict[str, tuple]:
    return {row[0]: row for row in _rows(db_path, 'SELECT * FROM write_ops')}


def _run(capsys, *argv: str) -> tuple[int, dict | None, str]:
    code = _script.main(list(argv))
    captured = capsys.readouterr()
    report = json.loads(captured.out) if captured.out.strip() else None
    return code, report, captured.err


def _migrate(journal: Path, capsys, *extra: str) -> tuple[int, dict | None, str]:
    return _run(
        capsys, 'migrate', '--journal', str(journal), '--apply',
        '--chunk-rowids', '11', '--pause-seconds', '0', *extra,
    )


class TestDryRun:
    def test_migrate_without_apply_changes_nothing(self, journal, capsys):
        before = _dump(journal)

        code, report, _ = _run(capsys, 'migrate', '--journal', str(journal), '--chunk-rowids', '11')

        assert code == 0
        assert report is not None and report['applied'] is False
        assert report['preview']
        assert _dump(journal) == before


class TestMigrateApply:
    def test_moves_every_count_only_row_into_the_rollup_exactly(self, journal, capsys):
        history = _history_rows()
        before_rows = _write_ops_by_id(journal)
        expected = Counter({_PRE_EXISTING_ROLLUP[:4]: _PRE_EXISTING_ROLLUP[4]})
        expected.update(_rollup_key(row) for row in history if _is_count_only(row))

        code, report, err = _migrate(journal, capsys)

        assert code == 0, err
        assert report is not None and report['applied'] is True
        assert report['rows_moved'] == sum(1 for row in history if _is_count_only(row))
        assert report['chunks'] > 1
        assert report['max_chunk_seconds'] >= report['mean_chunk_seconds'] > 0
        assert 0 <= report['mean_lock_seconds'] <= report['max_lock_seconds']
        assert report['max_lock_seconds'] <= report['max_chunk_seconds']
        assert _rollup_counts(journal) == expected
        after_rows = _write_ops_by_id(journal)
        kept_ids = {row['id'] for row in history if not _is_count_only(row)}
        assert set(after_rows) == kept_ids
        assert all(after_rows[row_id] == before_rows[row_id] for row_id in kept_ids)

    def test_overlapping_pre_existing_rollup_row_is_added_to(self, journal, capsys):
        history = _history_rows()
        raw_in_key = sum(
            1 for row in history
            if _is_count_only(row) and _rollup_key(row) == _PRE_EXISTING_ROLLUP[:4]
        )
        assert raw_in_key > 0

        _migrate(journal, capsys)

        assert _rollup_counts(journal)[_PRE_EXISTING_ROLLUP[:4]] == 3 + raw_in_key

    def test_the_work_queue_scan_is_reported_apart_from_the_chunk_rate(
        self, journal, capsys, monkeypatch,
    ):
        slow_scan = _script.count_only_span

        def scan_taking_time(conn):
            time.sleep(0.4)
            return slow_scan(conn)

        monkeypatch.setattr(_script, 'count_only_span', scan_taking_time)

        code, report, err = _migrate(journal, capsys)

        assert code == 0, err
        assert report is not None
        assert report['setup_seconds'] >= 0.4
        assert report['seconds'] < 0.4


class TestPrewarmOutsideTheWriteLock:
    def test_every_prewarm_runs_before_its_begin_and_never_inside_a_transaction(
        self, journal, capsys, monkeypatch,
    ):
        statements: list[str] = []
        connect = _script.connect_for_migration

        def tracing_connect(path):
            conn = connect(path)
            conn.set_trace_callback(statements.append)
            return conn

        monkeypatch.setattr(_script, 'connect_for_migration', tracing_connect)

        code, report, err = _migrate(journal, capsys)

        assert code == 0, err
        assert report is not None and report['chunks'] > 1
        prewarm_text = PREWARM_PRIMARY_KEY_SQL.split('?')[0]
        kinds = []
        for statement in statements:
            if statement.startswith(prewarm_text):
                kinds.append('prewarm')
            elif statement == 'BEGIN IMMEDIATE':
                kinds.append('begin')
            elif statement == 'COMMIT':
                kinds.append('commit')
        chunk = ['prewarm', 'begin', 'commit']
        assert kinds == chunk * report['chunks']


def _snapshot(journal: Path, capsys, out: Path, *extra: str) -> int:
    code, _, err = _run(capsys, 'snapshot', '--journal', str(journal), '--out', str(out), *extra)
    assert code == 0, err
    return code


class TestPanelInvariance:
    def _snapshots(self, journal, capsys, tmp_path) -> list[Path]:
        wide = tmp_path / 'wide.json'
        _snapshot(
            journal, capsys, wide,
            '--window-hours', '1,24,168', '--window-end', _WINDOW_END.isoformat(),
        )
        straddle = tmp_path / 'straddle.json'
        _snapshot(
            journal, capsys, straddle,
            '--window-hours', '2', '--window-end', (_ROLLUP_HOUR + timedelta(hours=1)).isoformat(),
        )
        return [wide, straddle]

    def test_snapshot_windows_are_unchanged_by_the_migration(self, journal, capsys, tmp_path):
        snapshots = self._snapshots(journal, capsys, tmp_path)
        wide = json.loads(snapshots[0].read_text())
        assert [window['hours'] for window in wide['windows']] == [1, 24, 168]
        assert all(window['counts'] for window in wide['windows'])

        _migrate(journal, capsys)

        for snapshot in snapshots:
            code, report, err = _run(
                capsys, 'verify', '--journal', str(journal), '--snapshot', str(snapshot)
            )
            assert code == 0, (report, err)
            assert report is not None and report['mismatches'] == []

    def test_a_tampered_rollup_count_fails_verify_naming_the_key(
        self, journal, capsys, tmp_path,
    ):
        snapshots = self._snapshots(journal, capsys, tmp_path)
        _migrate(journal, capsys)
        conn = sqlite3.connect(journal)
        conn.execute(
            'UPDATE read_op_counts SET count = count + 1000 WHERE hour_start = ?'
            ' AND project_id = ? AND agent_id = ? AND operation = ?',
            _PRE_EXISTING_ROLLUP[:4],
        )
        conn.commit()
        conn.close()

        code, report, _ = _run(
            capsys, 'verify', '--journal', str(journal), '--snapshot', str(snapshots[1])
        )

        assert code != 0
        assert report is not None
        [mismatch] = report['mismatches']
        assert mismatch['window']['hours'] == 2
        assert mismatch['hour'] == _ROLLUP_HOUR.strftime('%Y-%m-%dT%H:00')
        assert (mismatch['kind'], mismatch['operation']) == ('read', 'get_task')
        assert mismatch['observed'] == mismatch['expected'] + 1000

    def test_a_surplus_rollup_key_absent_from_the_snapshot_fails_verify(
        self, journal, capsys, tmp_path,
    ):
        snapshots = self._snapshots(journal, capsys, tmp_path)
        _migrate(journal, capsys)
        _seed_rollup(
            journal, (_ROLLUP_HOUR.isoformat(), 'dark_factory', '', 'get_double_counted', 7)
        )

        code, report, _ = _run(
            capsys, 'verify', '--journal', str(journal), '--snapshot', str(snapshots[1])
        )

        assert code != 0
        assert report is not None
        [mismatch] = report['mismatches']
        assert mismatch['operation'] == 'get_double_counted'
        assert (mismatch['expected'], mismatch['observed']) == (0, 7)


class TestReadRetentionHorizon:
    """Snapshot windows stop at the raw read horizon fused-memory actually
    prunes to, which its config.yaml reads from WRITE_JOURNAL_READ_RETENTION_DAYS,
    not at WriteJournalConfig's code default."""

    @pytest.fixture(autouse=True)
    def _no_ambient_retention(self, monkeypatch):
        monkeypatch.delenv('WRITE_JOURNAL_READ_RETENTION_DAYS', raising=False)

    def _week_snapshot(self, journal, capsys, out, *extra: str):
        return _run(
            capsys, 'snapshot', '--journal', str(journal), '--out', str(out),
            '--window-hours', '168', '--window-end', _WINDOW_END.isoformat(), *extra,
        )

    def test_the_configured_retention_refuses_a_window_past_it(
        self, journal, capsys, tmp_path, monkeypatch,
    ):
        monkeypatch.setenv('WRITE_JOURNAL_READ_RETENTION_DAYS', '3')
        out = tmp_path / 'refused.json'

        code, report, err = self._week_snapshot(journal, capsys, out)

        assert code == 1
        assert report is None
        assert 'read retention horizon' in err
        assert not out.exists()

    def test_without_config_path_the_checkouts_config_is_read_not_the_cwds(
        self, journal, capsys, tmp_path, monkeypatch,
    ):
        monkeypatch.delenv('CONFIG_PATH', raising=False)
        monkeypatch.chdir(tmp_path)
        monkeypatch.setenv('WRITE_JOURNAL_READ_RETENTION_DAYS', '3')

        code, _, err = self._week_snapshot(journal, capsys, tmp_path / 'refused.json')

        assert code == 1
        assert 'read retention horizon' in err

    def test_an_explicit_retention_overrides_the_configured_one(
        self, journal, capsys, tmp_path, monkeypatch,
    ):
        monkeypatch.setenv('WRITE_JOURNAL_READ_RETENTION_DAYS', '3')

        code, report, err = self._week_snapshot(
            journal, capsys, tmp_path / 'accepted.json', '--read-retention-days', '30',
        )

        assert code == 0, err
        assert report is not None and report['read_retention_days'] == 30.0

    def test_the_report_names_the_configured_retention_it_used(
        self, journal, capsys, tmp_path,
    ):
        code, report, err = self._week_snapshot(journal, capsys, tmp_path / 'default.json')

        assert code == 0, err
        assert report is not None and report['read_retention_days'] == 30.0

    def test_verify_refuses_a_window_the_horizon_has_since_crossed(
        self, journal, capsys, tmp_path,
    ):
        out = tmp_path / 'week.json'
        code, _, err = self._week_snapshot(journal, capsys, out, '--read-retention-days', '30')
        assert code == 0, err
        recorded = json.loads(out.read_text())
        assert recorded['read_retention_days'] == 30.0
        # As if days passed: the recorded horizon now falls inside the window.
        out.write_text(json.dumps({**recorded, 'read_retention_days': 2.0}))

        code, report, err = _run(capsys, 'verify', '--journal', str(journal), '--snapshot', str(out))

        assert code == 1
        assert report is None
        assert 'read retention horizon' in err


def _month_of_history() -> list[dict]:
    """Rows every 4 hours over 40 days ending 2 h ago, in time order, so the
    oldest 10 days lie past the 30-day read horizon."""
    newest = datetime.now(UTC) - timedelta(hours=2)
    rows = []
    for i in range(240):
        shape = i % 5
        kind, operation, success = {
            0: ('read', 'get_task', 1),
            1: ('read', 'get_statuses', 1),
            2: ('read', 'search', 1),
            3: ('read', 'get_task', 0),
            4: ('write', 'add_memory', 1),
        }[shape]
        rows.append({
            'id': f'aged-{i}',
            'operation': operation,
            'project_id': 'dark_factory',
            'agent_id': None,
            'kind': kind,
            'success': success,
            'error': None if success else 'boom',
            'params': '{}',
            'created_at': (newest - timedelta(hours=4 * (239 - i))).isoformat(),
        })
    return rows


def _retention_prune_pass(journal_dir: Path) -> None:
    async def _pass() -> None:
        journal = WriteJournal(journal_dir)
        await journal.initialize()
        try:
            await prune_write_ops_pass(journal, WriteJournalConfig())
        finally:
            await journal.close()

    asyncio.run(_pass())


class TestAlongsideTheRetentionPrune:
    """fused-memory's retention prune keeps running, at startup and every
    prune_interval_seconds, while the migration moves history."""

    @pytest.fixture(autouse=True)
    def _no_ambient_retention(self, monkeypatch):
        monkeypatch.delenv('WRITE_JOURNAL_READ_RETENTION_DAYS', raising=False)

    def test_every_count_only_row_is_counted_or_pruned_once_and_verify_holds(
        self, tmp_path, capsys,
    ):
        journal_dir = tmp_path / 'journal'
        db_path = _initialized_journal(journal_dir)
        history = _month_of_history()
        _insert_rows(db_path, history)
        out = tmp_path / 'before.json'
        _snapshot(db_path, capsys, out)

        _migrate(db_path, capsys, '--max-chunks', '2')
        before_prune = set(_write_ops_by_id(db_path))
        _retention_prune_pass(journal_dir)
        pruned = before_prune - set(_write_ops_by_id(db_path))
        code, report, err = _migrate(db_path, capsys)
        assert code == 0 and report is not None and report['remaining'] is None, err
        _retention_prune_pass(journal_dir)

        horizon = datetime.now(UTC) - timedelta(days=30)
        by_id = {row['id']: row for row in history}
        assert any(_is_count_only(by_id[row_id]) for row_id in pruned), (
            'the prune took no count-only row, so this did not race the migration'
        )
        assert all(datetime.fromisoformat(by_id[row_id]['created_at']) < horizon for row_id in pruned)
        assert _rollup_counts(db_path) == Counter(
            _rollup_key(row) for row in history
            if _is_count_only(row) and row['id'] not in pruned
        )
        code, report, err = _run(capsys, 'verify', '--journal', str(db_path), '--snapshot', str(out))
        assert code == 0, (report, err)
        assert report is not None and report['mismatches'] == []


def _twin_journals(tmp_path) -> tuple[Path, Path]:
    twins = []
    for name in ('interrupted', 'uninterrupted'):
        db_path = _initialized_journal(tmp_path / name)
        _insert_rows(db_path, _history_rows())
        _seed_rollup(db_path, _PRE_EXISTING_ROLLUP)
        twins.append(db_path)
    return twins[0], twins[1]


class TestResumable:
    def test_a_bounded_run_then_a_full_run_equals_one_uninterrupted_run(self, tmp_path, capsys):
        interrupted, uninterrupted = _twin_journals(tmp_path)

        code, partial, _ = _migrate(interrupted, capsys, '--max-chunks', '1')
        assert code == 0
        assert partial is not None and partial['chunks'] == 1
        assert partial['remaining'] is not None
        _migrate(interrupted, capsys)
        _migrate(uninterrupted, capsys)

        assert _rollup_counts(interrupted) == _rollup_counts(uninterrupted)
        assert _write_ops_by_id(interrupted) == _write_ops_by_id(uninterrupted)

    def test_a_further_run_moves_nothing(self, journal, capsys):
        _migrate(journal, capsys)
        settled = _rollup_counts(journal)

        code, report, _ = _migrate(journal, capsys)

        assert code == 0
        assert report is not None and report['rows_moved'] == 0
        assert report['remaining'] is None
        assert _rollup_counts(journal) == settled


def _poison_row(history: list[dict]) -> dict:
    """A count-only row in a LATER chunk, so earlier chunks commit first."""
    return [row for row in history if _is_count_only(row)][-3]


class TestCrashSafety:
    def test_a_failed_chunk_rolls_back_whole_and_a_rerun_lands_exact_totals(
        self, journal, capsys,
    ):
        history = _history_rows()
        poison = _poison_row(history)
        conn = sqlite3.connect(journal)
        conn.execute(
            'CREATE TRIGGER poison BEFORE DELETE ON write_ops '
            f"WHEN OLD.id = '{poison['id']}' BEGIN SELECT RAISE(ABORT, 'boom'); END"
        )
        conn.commit()
        poison_rowid = conn.execute(
            'SELECT rowid FROM write_ops WHERE id = ?', (poison['id'],)
        ).fetchone()[0]
        conn.close()

        code, _, err = _migrate(journal, capsys)

        assert code != 0
        assert 'boom' in err
        surviving = _write_ops_by_id(journal)
        assert poison['id'] in surviving
        first_remaining = _rows(
            journal,
            'SELECT MIN(rowid) FROM write_ops WHERE kind = \'read\' AND success = 1'
            " AND (operation IS NULL OR operation <> 'search')",
        )[0][0]
        assert 1 < first_remaining <= poison_rowid
        moved_so_far = Counter({_PRE_EXISTING_ROLLUP[:4]: _PRE_EXISTING_ROLLUP[4]})
        moved_so_far.update(
            _rollup_key(row) for row in history
            if _is_count_only(row) and row['id'] not in surviving
        )
        assert _rollup_counts(journal) == moved_so_far

        conn = sqlite3.connect(journal)
        conn.execute('DROP TRIGGER poison')
        conn.commit()
        conn.close()
        code, _, err = _migrate(journal, capsys)

        assert code == 0, err
        expected = Counter({_PRE_EXISTING_ROLLUP[:4]: _PRE_EXISTING_ROLLUP[4]})
        expected.update(_rollup_key(row) for row in history if _is_count_only(row))
        assert _rollup_counts(journal) == expected


class TestChunkVerification:
    """The per-chunk oracle: a statement that silently under-applies is not committed."""

    def test_a_rollup_write_that_is_ignored_rolls_the_chunk_back(self, journal, capsys):
        before_rows = _write_ops_by_id(journal)
        before_rollup = _rollup_counts(journal)
        conn = sqlite3.connect(journal)
        conn.execute(
            'CREATE TRIGGER ignore_rollup BEFORE INSERT ON read_op_counts'
            ' BEGIN SELECT RAISE(IGNORE); END'
        )
        conn.commit()
        conn.close()

        code, _, err = _migrate(journal, capsys)

        assert code != 0
        assert 'count is' in err
        assert _write_ops_by_id(journal) == before_rows
        assert _rollup_counts(journal) == before_rollup

    def test_a_raw_delete_that_skips_a_row_rolls_its_chunk_back(self, journal, capsys):
        history = _history_rows()
        skipped = _poison_row(history)
        conn = sqlite3.connect(journal)
        conn.execute(
            'CREATE TRIGGER skip_delete BEFORE DELETE ON write_ops'
            f" WHEN OLD.id = '{skipped['id']}' BEGIN SELECT RAISE(IGNORE); END"
        )
        conn.commit()
        conn.close()

        code, _, err = _migrate(journal, capsys)

        assert code != 0
        assert 'deleted' in err
        surviving = _write_ops_by_id(journal)
        assert skipped['id'] in surviving
        expected = Counter({_PRE_EXISTING_ROLLUP[:4]: _PRE_EXISTING_ROLLUP[4]})
        expected.update(
            _rollup_key(row) for row in history
            if _is_count_only(row) and row['id'] not in surviving
        )
        assert _rollup_counts(journal) == expected


class TestGuards:
    def test_a_count_only_read_inside_the_deploy_grace_refuses(self, journal, capsys):
        recent = (datetime.now(UTC) - timedelta(minutes=2)).isoformat()
        _insert_rows(journal, [{
            'id': 'recent-read', 'operation': 'get_task', 'project_id': 'p', 'agent_id': None,
            'kind': 'read', 'success': 1, 'error': None, 'params': '{}', 'created_at': recent,
        }])
        before = _dump(journal)

        code, _, err = _migrate(journal, capsys)

        assert code != 0
        assert recent in err
        assert _dump(journal) == before

    def test_a_journal_without_read_op_counts_refuses(self, tmp_path, capsys):
        db_path = tmp_path / 'pre_rollup.db'
        conn = sqlite3.connect(db_path)
        conn.executescript(SCHEMA_SQL)
        conn.close()
        _insert_rows(db_path, _history_rows())
        before = _dump(db_path)

        code, _, err = _migrate(db_path, capsys)

        assert code != 0
        assert 'read_op_counts' in err
        assert _dump(db_path) == before


class TestJournalLocation:
    def test_without_journal_the_one_fused_memory_is_configured_to_write_is_used(
        self, journal, capsys, tmp_path, monkeypatch,
    ):
        monkeypatch.setenv('RECONCILIATION_DATA_DIR', str(journal.parent))
        monkeypatch.chdir(tmp_path)
        assert journal.name == 'write_journal.db'

        code, report, err = _run(capsys, 'status')

        assert code == 0, err
        assert report is not None and report['count_only_span'] is not None

    def test_a_temp_dir_with_a_quote_is_rejected_before_anything_runs(
        self, journal, capsys, tmp_path,
    ):
        before = _dump(journal)

        with pytest.raises(SystemExit) as exited:
            _script.main(['reclaim', '--journal', str(journal), '--temp-dir', str(tmp_path / "a'b")])

        assert exited.value.code == 2
        assert 'quote' in capsys.readouterr().err
        assert _dump(journal) == before


class TestStatus:
    def test_status_reports_the_remaining_span_and_rollup_rows_read_only(
        self, journal, capsys,
    ):
        before = _dump(journal)

        code, report, _ = _run(capsys, 'status', '--journal', str(journal))

        assert code == 0
        assert report is not None
        count_only_rowids = [
            rowid for rowid, row in enumerate(_history_rows(), start=1) if _is_count_only(row)
        ]
        assert report['count_only_span'] == {
            'first_rowid': count_only_rowids[0],
            'max_rowid': len(_history_rows()),
        }
        assert report['rollup_rows'] == 1
        assert _dump(journal) == before

    def test_status_after_migration_reports_no_span(self, journal, capsys):
        _migrate(journal, capsys)

        _, report, _ = _run(capsys, 'status', '--journal', str(journal))

        assert report is not None and report['count_only_span'] is None
        assert report['rollup_rows'] == len(_rollup_counts(journal))


def _never_active() -> bool:
    return False


def _always_active() -> bool:
    return True


def _run_reclaim(capsys, journal: Path, *extra: str, unit_active=_never_active):
    code = _script.main(['reclaim', '--journal', str(journal), *extra], unit_active=unit_active)
    captured = capsys.readouterr()
    report = json.loads(captured.out) if captured.out.strip() else None
    return code, report, captured.err


def _db_bytes(db_path: Path) -> bytes:
    """The database file itself; a mode=ro reader may still leave empty WAL sidecars."""
    return db_path.read_bytes()


def _table_row_counts(db_path: Path) -> dict[str, int]:
    names = [row[0] for row in _rows(db_path, "SELECT name FROM sqlite_master WHERE type = 'table'")]
    return {name: _rows(db_path, f'SELECT COUNT(*) FROM "{name}"')[0][0] for name in names}


def _bulky_migrated_journal(tmp_path, capsys) -> Path:
    """A journal whose migration deleted enough wide rows to leave a freelist."""
    db_path = _initialized_journal(tmp_path / 'bulky')
    base = _WINDOW_END - timedelta(hours=40)
    _insert_rows(db_path, [
        {
            'id': f'bulk-{i}', 'operation': 'get_task' if i % 4 else 'search',
            'project_id': 'p', 'agent_id': None, 'kind': 'read', 'success': 1,
            'error': None, 'params': 'x' * 2048,
            'created_at': (base + timedelta(seconds=30 * i)).isoformat(),
        }
        for i in range(2000)
    ])
    code, _, err = _run(
        capsys, 'migrate', '--journal', str(db_path), '--apply',
        '--chunk-rowids', '500', '--pause-seconds', '0',
    )
    assert code == 0, err
    return db_path


class TestReclaim:
    def test_refuses_while_count_only_raw_reads_remain(self, journal, capsys):
        before = _db_bytes(journal)

        code, _, err = _run_reclaim(capsys, journal, '--apply')

        assert code != 0
        assert 'first_rowid=1' in err and f'max_rowid={len(_history_rows())}' in err
        assert _db_bytes(journal) == before

    def test_refuses_while_fused_memory_is_active(self, tmp_path, capsys):
        db_path = _bulky_migrated_journal(tmp_path, capsys)
        before = _db_bytes(db_path)

        code, _, err = _run_reclaim(capsys, db_path, '--apply', unit_active=_always_active)

        assert code != 0
        assert 'fused-memory' in err
        assert _db_bytes(db_path) == before

    def test_dry_run_reports_page_figures_and_changes_nothing(self, tmp_path, capsys):
        db_path = _bulky_migrated_journal(tmp_path, capsys)
        before = _db_bytes(db_path)

        code, report, err = _run_reclaim(capsys, db_path)

        assert code == 0, err
        assert report is not None and report['applied'] is False
        pages = report['before']
        assert pages['freelist_count'] > 0
        assert pages['page_size'] > 0 and pages['page_count'] >= pages['freelist_count']
        assert report['reclaimable_bytes'] == pages['freelist_count'] * pages['page_size']
        assert report['blockers'] == []
        assert _db_bytes(db_path) == before

    def test_apply_vacuums_in_place_and_keeps_every_row(self, tmp_path, capsys):
        db_path = _bulky_migrated_journal(tmp_path, capsys)
        rows_before = _table_row_counts(db_path)
        size_before = db_path.stat().st_size
        inode_before = db_path.stat().st_ino

        code, report, err = _run_reclaim(capsys, db_path, '--apply', '--temp-dir', str(tmp_path))

        assert code == 0, err
        assert report is not None and report['applied'] is True
        assert report['after']['freelist_count'] == 0
        assert report['before']['freelist_count'] > 0
        assert report['after_bytes'] < report['before_bytes']
        assert report['seconds'] >= 0
        assert db_path.stat().st_size < size_before
        assert db_path.stat().st_ino == inode_before
        assert _table_row_counts(db_path) == rows_before

    def test_apply_leaves_no_wal_behind_while_another_connection_stays_open(
        self, tmp_path, capsys,
    ):
        db_path = _bulky_migrated_journal(tmp_path, capsys)
        bystander = sqlite3.connect(db_path)
        try:
            bystander.execute('SELECT COUNT(*) FROM write_ops').fetchone()

            code, report, err = _run_reclaim(capsys, db_path, '--apply')
        finally:
            bystander.close()

        assert code == 0, err
        assert report is not None
        wal = db_path.with_name(db_path.name + '-wal')
        assert not wal.exists() or wal.stat().st_size == 0
        assert report['after_bytes'] < report['before_bytes'] // 2


@pytest.fixture
def systemctl_on_path(tmp_path, monkeypatch):
    """A PATH holding only a fake ``systemctl`` that prints what the test sets."""

    def install(stdout: str) -> None:
        bin_dir = tmp_path / 'fake-bin'
        bin_dir.mkdir(exist_ok=True)
        fake = bin_dir / 'systemctl'
        fake.write_text(f"#!/bin/sh\nprintf '%s' '{stdout}'\n")
        fake.chmod(0o755)
        monkeypatch.setenv('PATH', str(bin_dir))

    return install


class TestUnitProbeFailsClosed:
    """Whatever the probe cannot confirm inactive counts as active, so VACUUM is refused."""

    def _reclaim_with_the_real_probe(self, tmp_path, capsys):
        db_path = _bulky_migrated_journal(tmp_path, capsys)
        before = _db_bytes(db_path)
        code = _script.main(['reclaim', '--journal', str(db_path), '--apply'])
        return code, capsys.readouterr().err, db_path, before

    def test_confirmed_inactive_units_let_the_reclaim_run(
        self, tmp_path, capsys, systemctl_on_path,
    ):
        systemctl_on_path('inactive\ninactive\n')

        code, err, db_path, before = self._reclaim_with_the_real_probe(tmp_path, capsys)

        assert code == 0, err
        assert _db_bytes(db_path) != before

    def test_a_missing_systemctl_refuses(self, tmp_path, capsys, monkeypatch):
        monkeypatch.setenv('PATH', str(tmp_path / 'no-such-bin'))

        code, err, db_path, before = self._reclaim_with_the_real_probe(tmp_path, capsys)

        assert code != 0
        assert 'fused-memory' in err
        assert _db_bytes(db_path) == before

    def test_a_probe_that_times_out_refuses(self, tmp_path, capsys, monkeypatch):
        def hang(*args, **kwargs):
            raise subprocess.TimeoutExpired(cmd='systemctl', timeout=10)

        monkeypatch.setattr(_script.subprocess, 'run', hang)

        code, err, db_path, before = self._reclaim_with_the_real_probe(tmp_path, capsys)

        assert code != 0
        assert 'fused-memory' in err
        assert _db_bytes(db_path) == before

    def test_output_naming_fewer_units_than_asked_refuses(
        self, tmp_path, capsys, systemctl_on_path,
    ):
        systemctl_on_path('inactive\n')

        code, err, db_path, before = self._reclaim_with_the_real_probe(tmp_path, capsys)

        assert code != 0
        assert _db_bytes(db_path) == before
