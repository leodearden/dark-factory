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

    def test_save_run_with_provided_run_id(self, tmp_path):
        """When run_id is provided it is used directly instead of auto-generated."""
        db_path = tmp_path / 'runs.db'
        store = RunStore(db_path)
        report = _sample_report()

        preset_id = 'run-abc123def456'
        returned_id = store.save_run(report, 'dark_factory', run_id=preset_id)

        assert returned_id == preset_id

        conn = sqlite3.connect(str(db_path))
        run = conn.execute(
            'SELECT run_id FROM runs WHERE run_id = ?', (preset_id,)
        ).fetchone()
        conn.close()
        assert run is not None
        assert run[0] == preset_id

    def test_save_run_auto_generates_run_id_when_none(self, tmp_path):
        """When run_id is None (default), run_id is auto-generated with 'run-' prefix."""
        db_path = tmp_path / 'runs.db'
        store = RunStore(db_path)
        report = _sample_report()

        run_id = store.save_run(report, 'dark_factory')

        assert run_id is not None
        assert run_id.startswith('run-')
        # Auto-generated format: run-{uuid.hex[:12]}
        suffix = run_id[len('run-'):]
        assert len(suffix) == 12
        assert suffix.isalnum()

    def test_save_run_provided_run_id_preserved_in_task_results(self, tmp_path):
        """task_results rows use the caller-supplied run_id."""
        db_path = tmp_path / 'runs.db'
        store = RunStore(db_path)
        report = _sample_report()

        preset_id = 'run-preset-test'
        store.save_run(report, 'dark_factory', run_id=preset_id)

        conn = sqlite3.connect(str(db_path))
        rows = conn.execute(
            'SELECT DISTINCT run_id FROM task_results'
        ).fetchall()
        conn.close()
        assert len(rows) == 1
        assert rows[0][0] == preset_id
