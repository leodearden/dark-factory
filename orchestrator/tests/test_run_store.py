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
