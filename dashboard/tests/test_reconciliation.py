"""Tests for dashboard.data.reconciliation query functions."""

from __future__ import annotations


class TestGetRecentRuns:
    """Tests for get_recent_runs."""

    async def test_happy_path_returns_runs_ordered_desc(self, reconciliation_db):
        """Returns list of run dicts ordered by started_at DESC with correct fields."""
        from dashboard.data.reconciliation import get_recent_runs

        runs = await get_recent_runs(reconciliation_db)

        assert len(runs) == 2
        # Most recent first (run-002 started 10min ago, run-001 started 2h ago)
        assert runs[0]['id'] == 'run-002'
        assert runs[1]['id'] == 'run-001'

        # Check all expected fields are present
        expected_fields = {
            'id',
            'run_type',
            'trigger_reason',
            'started_at',
            'completed_at',
            'events_processed',
            'status',
            'duration_seconds',
        }
        for run in runs:
            assert set(run.keys()) == expected_fields

    async def test_completed_run_has_duration(self, reconciliation_db):
        """Completed runs should have duration_seconds calculated."""
        from dashboard.data.reconciliation import get_recent_runs

        runs = await get_recent_runs(reconciliation_db)
        completed = [r for r in runs if r['status'] == 'completed'][0]
        assert completed['duration_seconds'] is not None
        # Completed run lasted 5 minutes = 300 seconds
        assert completed['duration_seconds'] == pytest.approx(300.0, abs=1.0)

    async def test_running_run_has_no_duration(self, reconciliation_db):
        """Running (incomplete) runs should have duration_seconds=None."""
        from dashboard.data.reconciliation import get_recent_runs

        runs = await get_recent_runs(reconciliation_db)
        running = [r for r in runs if r['status'] == 'running'][0]
        assert running['duration_seconds'] is None
        assert running['completed_at'] is None

    async def test_respects_limit(self, reconciliation_db):
        """Limit parameter restricts number of results."""
        from dashboard.data.reconciliation import get_recent_runs

        runs = await get_recent_runs(reconciliation_db, limit=1)
        assert len(runs) == 1
        assert runs[0]['id'] == 'run-002'  # Most recent

    async def test_empty_table(self, empty_reconciliation_db):
        """Returns empty list when runs table has no data."""
        from dashboard.data.reconciliation import get_recent_runs

        runs = await get_recent_runs(empty_reconciliation_db)
        assert runs == []

    async def test_missing_db_file(self, missing_db_path):
        """Returns empty list when database file does not exist."""
        from dashboard.data.reconciliation import get_recent_runs

        runs = await get_recent_runs(missing_db_path)
        assert runs == []


import pytest


class TestGetWatermarks:
    """Tests for get_watermarks."""

    async def test_happy_path_returns_watermark_dict(self, reconciliation_db):
        """Returns dict with watermark fields for existing project_id."""
        from dashboard.data.reconciliation import get_watermarks

        result = await get_watermarks(reconciliation_db, project_id='dark_factory')

        assert isinstance(result, dict)
        expected_keys = {
            'last_full_run_completed',
            'last_episode_timestamp',
            'last_memory_timestamp',
            'last_task_change_timestamp',
        }
        assert set(result.keys()) == expected_keys
        # All values should be non-None strings (timestamps were set in fixture)
        for key in expected_keys:
            assert result[key] is not None

    async def test_nonexistent_project_id(self, reconciliation_db):
        """Returns empty dict for a project_id not in the table."""
        from dashboard.data.reconciliation import get_watermarks

        result = await get_watermarks(reconciliation_db, project_id='nonexistent')
        assert result == {}

    async def test_empty_table(self, empty_reconciliation_db):
        """Returns empty dict when watermarks table has no data."""
        from dashboard.data.reconciliation import get_watermarks

        result = await get_watermarks(empty_reconciliation_db, project_id='dark_factory')
        assert result == {}

    async def test_missing_db_file(self, missing_db_path):
        """Returns empty dict when database file does not exist."""
        from dashboard.data.reconciliation import get_watermarks

        result = await get_watermarks(missing_db_path, project_id='dark_factory')
        assert result == {}
