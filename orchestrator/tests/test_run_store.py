"""Tests for orchestrator.run_store — SQLite persistence of run results."""

from __future__ import annotations

import sqlite3

import pytest

from orchestrator.harness import HarnessReport, TaskReport
from orchestrator.run_store import RunStore
from orchestrator.workflow import WorkflowOutcome


def _sample_report() -> HarnessReport:
    return HarnessReport(
        started_at='2026-03-24T10:00:00+00:00',
        completed_at='2026-03-24T10:30:00+00:00',
        total_tasks=3,
        completed=2,
        blocked=1,
        escalated=0,
        total_cost_usd=1.50,
        task_reports=[
            TaskReport(
                task_id='101',
                title='Implement widget',
                outcome=WorkflowOutcome.DONE,
                cost_usd=0.80,
                duration_ms=120_000,
                agent_invocations=5,
                execute_iterations=2,
                verify_attempts=1,
                review_cycles=0,
                steward_cost_usd=0.0,
                steward_invocations=0,
                completed_at='2026-03-24T10:15:00+00:00',
            ),
            TaskReport(
                task_id='102',
                title='Refactor parser',
                outcome=WorkflowOutcome.DONE,
                cost_usd=0.50,
                duration_ms=90_000,
                agent_invocations=3,
                execute_iterations=1,
                verify_attempts=0,
                review_cycles=1,
                steward_cost_usd=0.20,
                steward_invocations=1,
                completed_at='2026-03-24T10:25:00+00:00',
            ),
            TaskReport(
                task_id='103',
                title='Fix auth bug',
                outcome=WorkflowOutcome.BLOCKED,
                cost_usd=0.20,
                duration_ms=60_000,
                agent_invocations=2,
                execute_iterations=1,
                verify_attempts=3,
                review_cycles=0,
                steward_cost_usd=0.0,
                steward_invocations=0,
                completed_at='2026-03-24T10:20:00+00:00',
            ),
        ],
    )


class TestRunStore:
    def test_schema_creation(self, tmp_path):
        db_path = tmp_path / 'runs.db'
        RunStore(db_path)

        conn = sqlite3.connect(str(db_path))
        tables = {
            row[0]
            for row in conn.execute(
                "SELECT name FROM sqlite_master WHERE type='table'"
            )
        }
        conn.close()
        assert 'runs' in tables
        assert 'task_results' in tables

    def test_schema_idempotent(self, tmp_path):
        db_path = tmp_path / 'runs.db'
        RunStore(db_path)
        RunStore(db_path)  # should not raise

    def test_creates_parent_dirs(self, tmp_path):
        db_path = tmp_path / 'deep' / 'nested' / 'runs.db'
        store = RunStore(db_path)
        assert db_path.exists()
        # Should be usable
        run_id = store.save_run(_sample_report(), 'test_project')
        assert run_id.startswith('run-')

    def test_save_run_roundtrip(self, tmp_path):
        db_path = tmp_path / 'runs.db'
        store = RunStore(db_path)
        report = _sample_report()

        run_id = store.save_run(report, 'dark_factory', '/path/to/prd.md')

        conn = sqlite3.connect(str(db_path))
        conn.row_factory = sqlite3.Row

        # Verify run row
        run = conn.execute(
            'SELECT * FROM runs WHERE run_id = ?', (run_id,)
        ).fetchone()
        assert run is not None
        assert run['project_id'] == 'dark_factory'
        assert run['prd_path'] == '/path/to/prd.md'
        assert run['total_tasks'] == 3
        assert run['completed'] == 2
        assert run['blocked'] == 1
        assert run['escalated'] == 0
        assert run['total_cost_usd'] == pytest.approx(1.50)

        # Verify task result rows
        results = conn.execute(
            'SELECT * FROM task_results WHERE run_id = ? ORDER BY task_id',
            (run_id,),
        ).fetchall()
        assert len(results) == 3

        r0 = results[0]
        assert r0['task_id'] == '101'
        assert r0['outcome'] == 'done'
        assert r0['execute_iterations'] == 2
        assert r0['verify_attempts'] == 1
        assert r0['review_cycles'] == 0
        assert r0['steward_invocations'] == 0

        r1 = results[1]
        assert r1['task_id'] == '102'
        assert r1['outcome'] == 'done'
        assert r1['review_cycles'] == 1
        assert r1['steward_invocations'] == 1

        r2 = results[2]
        assert r2['task_id'] == '103'
        assert r2['outcome'] == 'blocked'

        conn.close()

    def test_save_run_empty_reports(self, tmp_path):
        db_path = tmp_path / 'runs.db'
        store = RunStore(db_path)
        report = HarnessReport(
            started_at='2026-03-24T10:00:00+00:00',
            completed_at='2026-03-24T10:01:00+00:00',
        )

        run_id = store.save_run(report, 'test_project')

        conn = sqlite3.connect(str(db_path))
        run = conn.execute(
            'SELECT * FROM runs WHERE run_id = ?', (run_id,)
        ).fetchone()
        assert run is not None

        results = conn.execute(
            'SELECT * FROM task_results WHERE run_id = ?', (run_id,)
        ).fetchall()
        assert len(results) == 0
        conn.close()

    def test_multiple_runs(self, tmp_path):
        db_path = tmp_path / 'runs.db'
        store = RunStore(db_path)
        report = _sample_report()

        id1 = store.save_run(report, 'project_a')
        id2 = store.save_run(report, 'project_b')

        assert id1 != id2

        conn = sqlite3.connect(str(db_path))
        count = conn.execute('SELECT COUNT(*) FROM runs').fetchone()[0]
        assert count == 2

        task_count = conn.execute(
            'SELECT COUNT(*) FROM task_results'
        ).fetchone()[0]
        assert task_count == 6  # 3 per run
        conn.close()


class TestIncrementalPersistence:
    """Tests for start_run / save_task_result / finish_run lifecycle."""

    def test_start_creates_runs_row(self, tmp_path):
        store = RunStore(tmp_path / 'runs.db')
        store.start_run('run-abc', 'proj', '2026-04-10T00:00:00+00:00', 'prd.md')

        conn = sqlite3.connect(str(tmp_path / 'runs.db'))
        conn.row_factory = sqlite3.Row
        row = conn.execute('SELECT * FROM runs WHERE run_id = ?', ('run-abc',)).fetchone()
        conn.close()
        assert row is not None
        assert row['project_id'] == 'proj'
        assert row['completed_at'] is None
        assert row['total_tasks'] == 0

    def test_save_task_result_persists_immediately(self, tmp_path):
        store = RunStore(tmp_path / 'runs.db')
        store.start_run('run-abc', 'proj', '2026-04-10T00:00:00+00:00')

        tr = TaskReport(
            task_id='7',
            title='Add widget',
            outcome=WorkflowOutcome.DONE,
            cost_usd=1.23,
            duration_ms=5000,
            agent_invocations=3,
            execute_iterations=1,
            verify_attempts=1,
            review_cycles=0,
            completed_at='2026-04-10T00:05:00+00:00',
        )
        store.save_task_result('run-abc', tr, 'proj')

        conn = sqlite3.connect(str(tmp_path / 'runs.db'))
        conn.row_factory = sqlite3.Row
        row = conn.execute(
            'SELECT * FROM task_results WHERE run_id = ? AND task_id = ?',
            ('run-abc', '7'),
        ).fetchone()
        conn.close()
        assert row is not None
        assert row['outcome'] == 'done'
        assert row['cost_usd'] == pytest.approx(1.23)

    def test_finish_run_updates_aggregates(self, tmp_path):
        store = RunStore(tmp_path / 'runs.db')
        store.start_run('run-abc', 'proj', '2026-04-10T00:00:00+00:00')

        report = _sample_report()
        report.completed_at = '2026-04-10T01:00:00+00:00'
        store.finish_run('run-abc', report)

        conn = sqlite3.connect(str(tmp_path / 'runs.db'))
        conn.row_factory = sqlite3.Row
        row = conn.execute('SELECT * FROM runs WHERE run_id = ?', ('run-abc',)).fetchone()
        conn.close()
        assert row['completed_at'] == '2026-04-10T01:00:00+00:00'
        assert row['total_tasks'] == 3
        assert row['completed'] == 2
        assert row['blocked'] == 1
        assert row['total_cost_usd'] == pytest.approx(1.50)

    def test_full_incremental_lifecycle(self, tmp_path):
        """Simulate the real harness flow: start → task results → finish."""
        store = RunStore(tmp_path / 'runs.db')
        store.start_run('run-xyz', 'dark_factory', '2026-04-10T00:00:00+00:00')

        report = _sample_report()

        # Write each task result as it completes
        for tr in report.task_reports:
            store.save_task_result('run-xyz', tr, 'dark_factory')

        # Finalize the run
        report.completed_at = '2026-04-10T00:30:00+00:00'
        store.finish_run('run-xyz', report)

        conn = sqlite3.connect(str(tmp_path / 'runs.db'))
        conn.row_factory = sqlite3.Row

        run_row = conn.execute('SELECT * FROM runs WHERE run_id = ?', ('run-xyz',)).fetchone()
        assert run_row['completed_at'] == '2026-04-10T00:30:00+00:00'
        assert run_row['completed'] == 2

        results = conn.execute(
            'SELECT * FROM task_results WHERE run_id = ? ORDER BY task_id',
            ('run-xyz',),
        ).fetchall()
        assert len(results) == 3
        assert results[0]['task_id'] == '101'
        assert results[1]['task_id'] == '102'
        assert results[2]['task_id'] == '103'
        conn.close()

    def test_task_results_survive_without_finish(self, tmp_path):
        """If the orchestrator crashes before finish_run, task results are still there."""
        store = RunStore(tmp_path / 'runs.db')
        store.start_run('run-crash', 'proj', '2026-04-10T00:00:00+00:00')

        tr = TaskReport(
            task_id='42',
            title='Do thing',
            outcome=WorkflowOutcome.DONE,
            cost_usd=0.50,
            duration_ms=3000,
            completed_at='2026-04-10T00:02:00+00:00',
        )
        store.save_task_result('run-crash', tr, 'proj')

        # Simulate crash — no finish_run called. Open fresh connection.
        conn = sqlite3.connect(str(tmp_path / 'runs.db'))
        conn.row_factory = sqlite3.Row
        row = conn.execute(
            'SELECT * FROM task_results WHERE run_id = ? AND task_id = ?',
            ('run-crash', '42'),
        ).fetchone()
        assert row is not None
        assert row['outcome'] == 'done'

        run_row = conn.execute(
            'SELECT * FROM runs WHERE run_id = ?', ('run-crash',),
        ).fetchone()
        assert run_row['completed_at'] is None  # never finalized
        conn.close()


class TestSchedulerStatePersistence:
    """Tests for save/load/clear_scheduler_pause on RunStore."""

    def test_load_returns_none_when_unset(self, tmp_path):
        """Fresh RunStore has no scheduler pause record."""
        store = RunStore(tmp_path / 'runs.db')
        store.start_run('run-1', 'proj-a', '2026-05-14T10:00:00+00:00')
        result = store.load_scheduler_pause('proj-a')
        assert result is None, f'Expected None for unset pause, got {result!r}'

    def test_save_then_load_round_trip(self, tmp_path):
        """save_scheduler_pause → load_scheduler_pause returns matching dict."""
        store = RunStore(tmp_path / 'runs.db')
        store.start_run('run-x', 'proj-a', '2026-05-14T10:00:00+00:00')
        store.save_scheduler_pause(
            project_id='proj-a',
            reason='park-stop: 5 blocked',
            pause_at_iso='2026-05-14T10:05:00+00:00',
            set_by_run_id='run-x',
        )
        result = store.load_scheduler_pause('proj-a')
        assert result is not None, 'Expected a dict after save'
        assert result['reason'] == 'park-stop: 5 blocked'
        assert result['pause_at'] == '2026-05-14T10:05:00+00:00'
        assert result['set_by_run_id'] == 'run-x'

    def test_save_is_idempotent_upsert(self, tmp_path):
        """Second save with a different reason replaces the first (UPSERT semantics)."""
        store = RunStore(tmp_path / 'runs.db')
        store.start_run('run-1', 'proj-a', '2026-05-14T10:00:00+00:00')
        store.save_scheduler_pause(
            project_id='proj-a',
            reason='first reason',
            pause_at_iso='2026-05-14T10:01:00+00:00',
            set_by_run_id='run-1',
        )
        store.save_scheduler_pause(
            project_id='proj-a',
            reason='second reason',
            pause_at_iso='2026-05-14T10:02:00+00:00',
            set_by_run_id='run-1',
        )
        result = store.load_scheduler_pause('proj-a')
        assert result is not None
        assert result['reason'] == 'second reason', (
            f'Expected most-recent reason; got {result["reason"]!r}'
        )

    def test_clear_removes_row(self, tmp_path):
        """clear_scheduler_pause causes subsequent load to return None."""
        store = RunStore(tmp_path / 'runs.db')
        store.start_run('run-1', 'proj-a', '2026-05-14T10:00:00+00:00')
        store.save_scheduler_pause(
            project_id='proj-a',
            reason='will be cleared',
            pause_at_iso='2026-05-14T10:01:00+00:00',
            set_by_run_id='run-1',
        )
        store.clear_scheduler_pause('proj-a')
        result = store.load_scheduler_pause('proj-a')
        assert result is None, (
            f'Expected None after clear, got {result!r}'
        )


# ---------------------------------------------------------------------------
# Task 3068 / Part B — block_reason + block_phase persistence
# ---------------------------------------------------------------------------
#
# Origin incident (reify esc-5556-1): a 46h warm-lane requeue loop could not be
# reconstructed after the fact because neither events.db nor runs.db recorded
# WHY dispatches requeued.  runs.db's task_results held 14 counter-ish columns
# and no block context at all, even though TaskReport has carried
# block_reason/block_phase in memory since the retry-cap work.


#: The pre-3068 task_results DDL, verbatim, for building a legacy DB by hand.
#: Kept as a literal (rather than derived from _SCHEMA) precisely so it cannot
#: drift with the production schema — the migration tests below need a genuinely
#: OLD table, not whatever _SCHEMA happens to say today.
_LEGACY_TASK_RESULTS_DDL = """\
CREATE TABLE IF NOT EXISTS runs (
    run_id         TEXT PRIMARY KEY,
    project_id     TEXT NOT NULL,
    prd_path       TEXT,
    started_at     TEXT NOT NULL,
    completed_at   TEXT,
    total_tasks    INTEGER DEFAULT 0,
    completed      INTEGER DEFAULT 0,
    blocked        INTEGER DEFAULT 0,
    escalated      INTEGER DEFAULT 0,
    total_cost_usd REAL DEFAULT 0.0,
    paused_for_cap INTEGER DEFAULT 0,
    cap_pause_secs REAL DEFAULT 0.0
);

CREATE TABLE IF NOT EXISTS task_results (
    run_id              TEXT NOT NULL REFERENCES runs(run_id),
    task_id             TEXT NOT NULL,
    project_id          TEXT NOT NULL,
    title               TEXT,
    outcome             TEXT NOT NULL,
    cost_usd            REAL DEFAULT 0.0,
    duration_ms         INTEGER DEFAULT 0,
    agent_invocations   INTEGER DEFAULT 0,
    execute_iterations  INTEGER DEFAULT 0,
    verify_attempts     INTEGER DEFAULT 0,
    review_cycles       INTEGER DEFAULT 0,
    steward_cost_usd    REAL DEFAULT 0.0,
    steward_invocations INTEGER DEFAULT 0,
    completed_at        TEXT,
    PRIMARY KEY (run_id, task_id)
);
"""


def _task_results_columns(db_path) -> list[str]:
    """Return task_results' column names in physical (PRAGMA) order."""
    conn = sqlite3.connect(str(db_path))
    try:
        return [row[1] for row in conn.execute('PRAGMA table_info(task_results)')]
    finally:
        conn.close()


def _build_legacy_db(db_path) -> None:
    """Create a pre-3068 runs.db with one legacy task_results row."""
    conn = sqlite3.connect(str(db_path))
    try:
        conn.executescript(_LEGACY_TASK_RESULTS_DDL)
        conn.execute(
            'INSERT INTO runs (run_id, project_id, started_at) VALUES (?, ?, ?)',
            ('run-legacy', 'proj-a', '2026-07-01T10:00:00+00:00'),
        )
        conn.execute(
            'INSERT INTO task_results '
            '(run_id, task_id, project_id, title, outcome, '
            ' cost_usd, duration_ms, agent_invocations, '
            ' execute_iterations, verify_attempts, review_cycles, '
            ' steward_cost_usd, steward_invocations, completed_at) '
            'VALUES (?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?)',
            (
                'run-legacy', '900', 'proj-a', 'Legacy row', 'done',
                0.5, 1000, 4, 1, 1, 0, 0.0, 0, '2026-07-01T10:05:00+00:00',
            ),
        )
        conn.commit()
    finally:
        conn.close()


class TestBlockContextPersistence:
    """task_results must persist the classified block reason + phase."""

    def test_fresh_schema_has_block_columns_last(self, tmp_path):
        """A fresh DB declares block_reason/block_phase AFTER completed_at.

        SQLite's ALTER TABLE ADD COLUMN can only APPEND.  If _SCHEMA declared
        the new columns anywhere but last, a freshly-created runs.db would have
        a different physical column ORDER than a migrated one — and an operator
        forensic ``SELECT *`` (exactly the use case this task exists to serve)
        would silently read different fields depending on the DB's vintage.
        """
        db_path = tmp_path / 'runs.db'
        RunStore(db_path)

        cols = _task_results_columns(db_path)
        assert {'block_reason', 'block_phase'} <= set(cols), (
            f'missing block-context columns; got {cols}'
        )
        assert cols.index('block_reason') > cols.index('completed_at')
        assert cols.index('block_phase') > cols.index('completed_at')
        # Appended, in this order, and nothing between them.
        assert cols[-2:] == ['block_reason', 'block_phase'], (
            f'block columns must be last for ALTER-parity; got {cols}'
        )

    def test_save_task_result_persists_block_context(self, tmp_path):
        """The incremental writer round-trips block_reason/block_phase."""
        db_path = tmp_path / 'runs.db'
        store = RunStore(db_path)
        store.start_run('run-1', 'proj-a', '2026-07-30T10:00:00+00:00')

        store.save_task_result(
            'run-1',
            TaskReport(
                task_id='3068',
                title='Requeued task',
                outcome=WorkflowOutcome.REQUEUED,
                block_reason='warm_lane_pool_hard_down',
                block_phase='plan',
                completed_at='2026-07-30T10:05:00+00:00',
            ),
            'proj-a',
        )

        conn = sqlite3.connect(str(db_path))
        conn.row_factory = sqlite3.Row
        try:
            row = conn.execute(
                'SELECT block_reason, block_phase FROM task_results '
                'WHERE run_id = ? AND task_id = ?',
                ('run-1', '3068'),
            ).fetchone()
        finally:
            conn.close()
        assert row['block_reason'] == 'warm_lane_pool_hard_down'
        assert row['block_phase'] == 'plan'

    def test_save_task_result_clean_exit_persists_empty_not_null(self, tmp_path):
        """A DONE report persists '' — NOT NULL.

        Same honesty contract as the event payload (Part A): after this lands,
        NULL in these columns means "row predates task 3068" while '' means
        "clean exit, no block".
        """
        db_path = tmp_path / 'runs.db'
        store = RunStore(db_path)
        store.start_run('run-1', 'proj-a', '2026-07-30T10:00:00+00:00')

        store.save_task_result(
            'run-1',
            TaskReport(
                task_id='3069',
                title='Clean task',
                outcome=WorkflowOutcome.DONE,
                completed_at='2026-07-30T10:06:00+00:00',
            ),
            'proj-a',
        )

        conn = sqlite3.connect(str(db_path))
        conn.row_factory = sqlite3.Row
        try:
            row = conn.execute(
                'SELECT block_reason, block_phase FROM task_results '
                'WHERE task_id = ?',
                ('3069',),
            ).fetchone()
        finally:
            conn.close()
        assert row['block_reason'] == '', 'clean exit must persist "" not NULL'
        assert row['block_phase'] == '', 'clean exit must persist "" not NULL'

    def test_save_run_batch_path_persists_block_context(self, tmp_path):
        """The SECOND 14-column writer (save_run) must be extended too.

        The task description named only save_task_result; save_run has its own
        independent per-task INSERT.  Leaving it at 14 columns would silently
        write NULLs on the batch path.
        """
        db_path = tmp_path / 'runs.db'
        store = RunStore(db_path)

        report = _sample_report()
        report.task_reports[0].block_reason = ''
        report.task_reports[0].block_phase = ''
        report.task_reports[1].block_reason = 'warm_lane_pool_hard_down'
        report.task_reports[1].block_phase = 'execute'
        report.task_reports[2].block_reason = 'verify_infra_failure'
        report.task_reports[2].block_phase = 'verify'

        run_id = store.save_run(report, 'proj-a')

        conn = sqlite3.connect(str(db_path))
        conn.row_factory = sqlite3.Row
        try:
            rows = {
                r['task_id']: (r['block_reason'], r['block_phase'])
                for r in conn.execute(
                    'SELECT task_id, block_reason, block_phase FROM task_results '
                    'WHERE run_id = ?',
                    (run_id,),
                )
            }
        finally:
            conn.close()
        assert rows['101'] == ('', '')
        assert rows['102'] == ('warm_lane_pool_hard_down', 'execute')
        assert rows['103'] == ('verify_infra_failure', 'verify')

    # -- Migration of an EXISTING (pre-3068) runs.db --------------------- #
    #
    # _ensure_schema is a bare executescript of CREATE TABLE IF NOT EXISTS, so
    # a column added only to _SCHEMA would silently NOT apply to a runs.db that
    # already exists — every deployed orchestrator would keep writing 14
    # columns and the incident would stay unqueryable in exactly the DBs that
    # matter.  These four tests pin the additive migration.

    def test_migration_adds_columns_to_existing_db(self, tmp_path):
        """Constructing RunStore over a legacy DB adds the two columns."""
        db_path = tmp_path / 'runs.db'
        _build_legacy_db(db_path)
        assert 'block_reason' not in _task_results_columns(db_path), (
            'legacy fixture is not actually legacy'
        )

        RunStore(db_path)

        cols = _task_results_columns(db_path)
        assert {'block_reason', 'block_phase'} <= set(cols), (
            f'migration did not run on the existing DB; got {cols}'
        )
        # ALTER appends, so the migrated order must match the fresh-DB order.
        assert cols[-2:] == ['block_reason', 'block_phase']

    def test_migration_preserves_legacy_rows_as_null(self, tmp_path):
        """The migration is additive: the pre-existing row survives, NULL."""
        db_path = tmp_path / 'runs.db'
        _build_legacy_db(db_path)

        RunStore(db_path)

        conn = sqlite3.connect(str(db_path))
        conn.row_factory = sqlite3.Row
        try:
            row = conn.execute(
                'SELECT title, outcome, block_reason, block_phase '
                'FROM task_results WHERE task_id = ?',
                ('900',),
            ).fetchone()
        finally:
            conn.close()
        assert row is not None, 'migration destroyed the legacy row'
        assert row['title'] == 'Legacy row'
        assert row['outcome'] == 'done'
        # NULL, not '' — this row genuinely predates the fix, and that
        # distinction is the whole point of the always-write-'' contract.
        assert row['block_reason'] is None
        assert row['block_phase'] is None

    def test_writer_works_on_migrated_db(self, tmp_path):
        """save_task_result writes the new columns on a migrated DB."""
        db_path = tmp_path / 'runs.db'
        _build_legacy_db(db_path)
        store = RunStore(db_path)

        store.save_task_result(
            'run-legacy',
            TaskReport(
                task_id='901',
                title='Post-migration task',
                outcome=WorkflowOutcome.REQUEUED,
                block_reason='warm_lane_pool_hard_down',
                block_phase='review',
                completed_at='2026-07-30T11:00:00+00:00',
            ),
            'proj-a',
        )

        conn = sqlite3.connect(str(db_path))
        conn.row_factory = sqlite3.Row
        try:
            row = conn.execute(
                'SELECT block_reason, block_phase FROM task_results '
                'WHERE task_id = ?',
                ('901',),
            ).fetchone()
        finally:
            conn.close()
        assert row['block_reason'] == 'warm_lane_pool_hard_down'
        assert row['block_phase'] == 'review'

    def test_migration_is_idempotent(self, tmp_path):
        """A second construction must not raise "duplicate column name"."""
        db_path = tmp_path / 'runs.db'
        _build_legacy_db(db_path)

        RunStore(db_path)
        cols_after_first = _task_results_columns(db_path)
        RunStore(db_path)  # unguarded ALTER would raise here
        RunStore(db_path)

        assert _task_results_columns(db_path) == cols_after_first

    def test_concurrent_alter_race_is_survivable(self, tmp_path):
        """Losing the table_info->ALTER race must not cost a run its results.

        Two RunStore constructions can interleave between the PRAGMA read and
        the ALTER (the dashboard and a digest reader both open runs.db). The
        loser sees a real sqlite3 "duplicate column name" — which means the
        migration's GOAL is already met. Propagating it would reach
        Harness.run's `except Exception`, leave `_run_store` unset, and drop
        EVERY task_results row for the whole run over a no-op.
        """
        db_path = tmp_path / 'runs.db'
        _build_legacy_db(db_path)
        store = RunStore(db_path)  # migrates for real

        class _StaleReadConn:
            """Reports the PRE-migration column set, ALTERs the migrated DB."""

            def __init__(self, real):
                self._real = real

            def execute(self, sql, *args):
                if sql.startswith('PRAGMA table_info'):
                    # (cid, name, ...) shape, minus the 3068 columns.
                    return iter([(0, 'task_id'), (1, 'completed_at')])
                return self._real.execute(sql, *args)

            def commit(self):
                self._real.commit()

        conn = sqlite3.connect(str(db_path))
        try:
            # Would raise sqlite3.OperationalError without the guard.
            store._migrate_task_results_block_context(
                _StaleReadConn(conn),  # type: ignore[arg-type]
            )
        finally:
            conn.close()

        assert set(_task_results_columns(db_path)) >= {'block_reason', 'block_phase'}

    def test_non_duplicate_alter_failure_still_surfaces(self, tmp_path):
        """Only the benign race is swallowed — a real fault must not be.

        A corrupt DB or an un-takeable lock is exactly the case where a silent
        half-migrated schema would be worse than a loud failure.
        """
        db_path = tmp_path / 'runs.db'
        _build_legacy_db(db_path)
        store = RunStore(db_path)

        class _BrokenConn:
            def execute(self, sql, *args):
                if sql.startswith('PRAGMA table_info'):
                    return iter([(0, 'task_id')])
                raise sqlite3.OperationalError('database is locked')

            def commit(self):  # pragma: no cover - never reached
                raise AssertionError('commit must not be reached')

        with pytest.raises(sqlite3.OperationalError, match='database is locked'):
            store._migrate_task_results_block_context(
                _BrokenConn(),  # type: ignore[arg-type]
            )
