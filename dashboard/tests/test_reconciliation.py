"""Tests for dashboard.data.reconciliation query functions."""

from __future__ import annotations

import logging

import pytest


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
            'project_id',
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
        assert runs[0]['project_id'] == 'dark_factory'

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



class TestGetWatermarks:
    """Tests for get_watermarks."""

    async def test_happy_path_returns_watermark_list(self, reconciliation_db):
        """Returns list of dicts with watermark fields for all projects."""
        from dashboard.data.reconciliation import get_watermarks

        result = await get_watermarks(reconciliation_db)

        assert isinstance(result, list)
        assert len(result) == 1  # fixture only inserts dark_factory
        wm = result[0]
        expected_keys = {
            'project_id',
            'last_full_run_completed',
            'last_episode_timestamp',
            'last_memory_timestamp',
            'last_task_change_timestamp',
        }
        assert set(wm.keys()) == expected_keys
        assert wm['project_id'] == 'dark_factory'
        # All timestamp values should be non-None strings (set in fixture)
        for key in expected_keys - {'project_id'}:
            assert wm[key] is not None

    async def test_empty_table(self, empty_reconciliation_db):
        """Returns empty list when watermarks table has no data."""
        from dashboard.data.reconciliation import get_watermarks

        result = await get_watermarks(empty_reconciliation_db)
        assert result == []

    async def test_missing_db_file(self, missing_db_path):
        """Returns empty list when database file does not exist."""
        from dashboard.data.reconciliation import get_watermarks

        result = await get_watermarks(missing_db_path)
        assert result == []

    async def test_multiple_projects(self, tmp_path):
        """Returns watermarks for all projects ordered by project_id."""
        import sqlite3

        from dashboard.data.reconciliation import get_watermarks
        from tests.conftest import RECONCILIATION_SCHEMA

        db_path = tmp_path / 'multi.db'
        conn = sqlite3.connect(str(db_path))
        conn.executescript(RECONCILIATION_SCHEMA)
        conn.execute(
            "INSERT INTO watermarks (project_id, last_full_run_completed)"
            " VALUES ('alpha', '2026-03-19T10:00:00+00:00')"
        )
        conn.execute(
            "INSERT INTO watermarks (project_id, last_full_run_completed)"
            " VALUES ('beta', '2026-03-19T11:00:00+00:00')"
        )
        conn.commit()
        conn.close()

        result = await get_watermarks(db_path)
        assert len(result) == 2
        assert result[0]['project_id'] == 'alpha'
        assert result[1]['project_id'] == 'beta'


class TestGetLastAttemptedRun:
    """Tests for get_last_attempted_run."""

    async def test_happy_path_returns_per_project_dict(self, reconciliation_db):
        """Returns dict keyed by project_id with most recent run per project."""
        from dashboard.data.reconciliation import get_last_attempted_run

        result = await get_last_attempted_run(reconciliation_db)

        assert isinstance(result, dict)
        assert 'dark_factory' in result
        run = result['dark_factory']
        assert run['id'] == 'run-002'  # Most recent by started_at
        assert set(run.keys()) == {'id', 'status', 'started_at', 'completed_at'}

    async def test_failed_most_recent(self, tmp_path):
        """Returns a failed run when it's the most recent."""
        import sqlite3

        from dashboard.data.reconciliation import get_last_attempted_run

        db_path = tmp_path / 'failed.db'
        conn = sqlite3.connect(str(db_path))
        from tests.conftest import RECONCILIATION_SCHEMA
        conn.executescript(RECONCILIATION_SCHEMA)
        conn.execute(
            "INSERT INTO runs (id, project_id, run_type, trigger_reason, started_at, "
            "completed_at, events_processed, status) "
            "VALUES ('run-f1', 'dark_factory', 'full', 'test', '2026-03-19T12:00:00', "
            "'2026-03-19T12:01:00', 5, 'failed')"
        )
        conn.commit()
        conn.close()

        result = await get_last_attempted_run(db_path)
        assert 'dark_factory' in result
        assert result['dark_factory']['status'] == 'failed'

    async def test_multiple_projects(self, tmp_path):
        """Returns one entry per project, each the most recent run."""
        import sqlite3

        from dashboard.data.reconciliation import get_last_attempted_run

        db_path = tmp_path / 'multi.db'
        conn = sqlite3.connect(str(db_path))
        from tests.conftest import RECONCILIATION_SCHEMA
        conn.executescript(RECONCILIATION_SCHEMA)
        conn.execute(
            "INSERT INTO runs (id, project_id, run_type, trigger_reason, started_at, "
            "completed_at, events_processed, status) "
            "VALUES ('a1', 'alpha', 'full', 'test', '2026-03-19T10:00:00', "
            "'2026-03-19T10:01:00', 2, 'completed')"
        )
        conn.execute(
            "INSERT INTO runs (id, project_id, run_type, trigger_reason, started_at, "
            "completed_at, events_processed, status) "
            "VALUES ('a2', 'alpha', 'full', 'test', '2026-03-19T11:00:00', "
            "'2026-03-19T11:01:00', 3, 'failed')"
        )
        conn.execute(
            "INSERT INTO runs (id, project_id, run_type, trigger_reason, started_at, "
            "completed_at, events_processed, status) "
            "VALUES ('b1', 'beta', 'full', 'test', '2026-03-19T09:00:00', "
            "'2026-03-19T09:01:00', 1, 'completed')"
        )
        conn.commit()
        conn.close()

        result = await get_last_attempted_run(db_path)
        assert len(result) == 2
        assert result['alpha']['id'] == 'a2'  # most recent for alpha
        assert result['beta']['id'] == 'b1'

    async def test_empty_table(self, empty_reconciliation_db):
        """Returns empty dict when runs table has no data."""
        from dashboard.data.reconciliation import get_last_attempted_run

        result = await get_last_attempted_run(empty_reconciliation_db)
        assert result == {}

    async def test_missing_db_file(self, missing_db_path):
        """Returns empty dict when database file does not exist."""
        from dashboard.data.reconciliation import get_last_attempted_run

        result = await get_last_attempted_run(missing_db_path)
        assert result == {}


class TestGetBufferStats:
    """Tests for get_buffer_stats."""

    async def test_happy_path_returns_count_and_age(self, reconciliation_db):
        """Returns dict with buffered_count and oldest_event_age_seconds."""
        from dashboard.data.reconciliation import get_buffer_stats

        result = await get_buffer_stats(reconciliation_db)

        assert isinstance(result, dict)
        assert set(result.keys()) == {'buffered_count', 'oldest_event_age_seconds'}
        # Fixture inserts 3 buffered events
        assert result['buffered_count'] == 3
        # Oldest event is 60 minutes old — age should be roughly 3600 seconds
        assert result['oldest_event_age_seconds'] is not None
        assert result['oldest_event_age_seconds'] >= 3500  # at least ~58 min

    async def test_empty_table(self, empty_reconciliation_db):
        """Returns zero count and None age when no buffered events."""
        from dashboard.data.reconciliation import get_buffer_stats

        result = await get_buffer_stats(empty_reconciliation_db)

        assert result == {'buffered_count': 0, 'oldest_event_age_seconds': None}

    async def test_missing_db_file(self, missing_db_path):
        """Returns default dict when database file does not exist."""
        from dashboard.data.reconciliation import get_buffer_stats

        result = await get_buffer_stats(missing_db_path)

        assert result == {'buffered_count': 0, 'oldest_event_age_seconds': None}


class TestGetBurstState:
    """Tests for get_burst_state."""

    async def test_happy_path_returns_burst_state_list(self, reconciliation_db):
        """Returns list of dicts with correct fields for all agents."""
        from dashboard.data.reconciliation import get_burst_state

        result = await get_burst_state(reconciliation_db)

        assert isinstance(result, list)
        assert len(result) == 2

        expected_fields = {'agent_id', 'state', 'last_write_at', 'burst_started_at'}
        for entry in result:
            assert set(entry.keys()) == expected_fields

        # agent-1: last_write 2min ago, within 150s cooldown → still bursting
        by_agent = {e['agent_id']: e for e in result}
        assert by_agent['agent-1']['state'] == 'bursting'
        assert by_agent['agent-1']['burst_started_at'] is not None
        assert by_agent['agent-2']['state'] == 'idle'
        assert by_agent['agent-2']['burst_started_at'] is None

    async def test_cooldown_expires_stale_burst(self, tmp_path):
        """Agents with last_write older than cooldown are reported as idle."""
        import sqlite3
        from datetime import UTC, datetime, timedelta

        from dashboard.data.reconciliation import get_burst_state
        from tests.conftest import RECONCILIATION_SCHEMA

        db_path = tmp_path / 'stale.db'
        conn = sqlite3.connect(str(db_path))
        conn.executescript(RECONCILIATION_SCHEMA)
        now = datetime.now(UTC)
        # last_write_at is 10 minutes ago — well past default 150s cooldown
        conn.execute(
            "INSERT INTO burst_state (agent_id, state, last_write_at, burst_started_at)"
            " VALUES (?, 'bursting', ?, ?)",
            ('stale-agent', (now - timedelta(minutes=10)).isoformat(),
             (now - timedelta(minutes=15)).isoformat()),
        )
        conn.commit()
        conn.close()

        result = await get_burst_state(db_path)
        assert len(result) == 1
        assert result[0]['state'] == 'idle'
        assert result[0]['burst_started_at'] is None

    async def test_cooldown_preserves_active_burst(self, tmp_path):
        """Agents with recent last_write keep bursting state."""
        import sqlite3
        from datetime import UTC, datetime, timedelta

        from dashboard.data.reconciliation import get_burst_state
        from tests.conftest import RECONCILIATION_SCHEMA

        db_path = tmp_path / 'active.db'
        conn = sqlite3.connect(str(db_path))
        conn.executescript(RECONCILIATION_SCHEMA)
        now = datetime.now(UTC)
        # last_write_at is 30 seconds ago — within 150s cooldown
        conn.execute(
            "INSERT INTO burst_state (agent_id, state, last_write_at, burst_started_at)"
            " VALUES (?, 'bursting', ?, ?)",
            ('active-agent', (now - timedelta(seconds=30)).isoformat(),
             (now - timedelta(seconds=60)).isoformat()),
        )
        conn.commit()
        conn.close()

        result = await get_burst_state(db_path)
        assert len(result) == 1
        assert result[0]['state'] == 'bursting'
        assert result[0]['burst_started_at'] is not None

    async def test_empty_table(self, empty_reconciliation_db):
        """Returns empty list when burst_state table has no data."""
        from dashboard.data.reconciliation import get_burst_state

        result = await get_burst_state(empty_reconciliation_db)
        assert result == []

    async def test_missing_db_file(self, missing_db_path):
        """Returns empty list when database file does not exist."""
        from dashboard.data.reconciliation import get_burst_state

        result = await get_burst_state(missing_db_path)
        assert result == []


class TestGetLatestVerdict:
    """Tests for get_latest_verdict."""

    async def test_happy_path_returns_verdict_dict(self, reconciliation_db):
        """Returns dict with verdict fields for the most recent verdict."""
        from dashboard.data.reconciliation import get_latest_verdict

        result = await get_latest_verdict(reconciliation_db)

        assert result is not None
        assert isinstance(result, dict)
        expected_keys = {'run_id', 'severity', 'action_taken', 'reviewed_at'}
        assert set(result.keys()) == expected_keys
        assert result['run_id'] == 'run-001'
        assert result['severity'] == 'low'
        assert result['action_taken'] == 'logged'

    async def test_empty_table(self, empty_reconciliation_db):
        """Returns None when judge_verdicts table has no data."""
        from dashboard.data.reconciliation import get_latest_verdict

        result = await get_latest_verdict(empty_reconciliation_db)
        assert result is None

    async def test_missing_db_file(self, missing_db_path):
        """Returns None when database file does not exist."""
        from dashboard.data.reconciliation import get_latest_verdict

        result = await get_latest_verdict(missing_db_path)
        assert result is None


class TestExceptionLogging:
    """Tests that reconciliation functions emit DEBUG-level logs on DB unavailability."""

    async def test_missing_db_logs_debug(self, missing_db_path, caplog):
        """get_recent_runs with a missing DB path emits a DEBUG log."""
        from dashboard.data.reconciliation import get_recent_runs

        with caplog.at_level(logging.DEBUG, logger='dashboard.data.reconciliation'):
            result = await get_recent_runs(missing_db_path)

        assert result == []
        assert any(
            r.levelno == logging.DEBUG and 'dashboard.data.reconciliation' in r.name
            for r in caplog.records
        ), f'Expected DEBUG log from dashboard.data.reconciliation, got: {caplog.records}'

    async def test_operational_error_logs_debug(self, tmp_path, caplog):
        """get_watermarks with an empty (no-tables) DB file emits a DEBUG log on OperationalError."""
        from dashboard.data.reconciliation import get_watermarks

        # Create a valid SQLite file but with no tables — causes OperationalError
        db_path = tmp_path / 'empty.db'
        import sqlite3
        conn = sqlite3.connect(str(db_path))
        conn.close()

        with caplog.at_level(logging.DEBUG, logger='dashboard.data.reconciliation'):
            result = await get_watermarks(db_path)

        assert result == []
        assert any(
            r.levelno == logging.DEBUG and 'dashboard.data.reconciliation' in r.name
            for r in caplog.records
        ), f'Expected DEBUG log from dashboard.data.reconciliation, got: {caplog.records}'
