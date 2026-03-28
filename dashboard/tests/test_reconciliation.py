"""Tests for dashboard.data.reconciliation query functions."""

from __future__ import annotations

import logging
from datetime import UTC, datetime, timedelta

import aiosqlite
import pytest


class TestGetRecentRuns:
    """Tests for get_recent_runs."""

    async def test_happy_path_returns_runs_ordered_desc(self, recon_conn):
        """Returns list of run dicts ordered by started_at DESC with correct fields."""
        from dashboard.data.reconciliation import get_recent_runs

        runs = await get_recent_runs(recon_conn)

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
            'journal_entry_count',
        }
        for run in runs:
            assert set(run.keys()) == expected_fields
        assert runs[0]['project_id'] == 'dark_factory'

    async def test_journal_entry_count(self, recon_conn):
        """Runs include journal_entry_count from correlated subquery."""
        from dashboard.data.reconciliation import get_recent_runs

        runs = await get_recent_runs(recon_conn)
        by_id = {r['id']: r for r in runs}
        # run-001 has 2 journal entries in fixture, run-002 has 0
        assert by_id['run-001']['journal_entry_count'] == 2
        assert by_id['run-002']['journal_entry_count'] == 0

    async def test_completed_run_has_duration(self, recon_conn):
        """Completed runs should have duration_seconds calculated."""
        from dashboard.data.reconciliation import get_recent_runs

        runs = await get_recent_runs(recon_conn)
        completed = [r for r in runs if r['status'] == 'completed'][0]
        assert completed['duration_seconds'] is not None
        # Completed run lasted 5 minutes = 300 seconds
        assert completed['duration_seconds'] == pytest.approx(300.0, abs=1.0)

    async def test_running_run_has_no_duration(self, recon_conn):
        """Running (incomplete) runs should have duration_seconds=None."""
        from dashboard.data.reconciliation import get_recent_runs

        runs = await get_recent_runs(recon_conn)
        running = [r for r in runs if r['status'] == 'running'][0]
        assert running['duration_seconds'] is None
        assert running['completed_at'] is None

    async def test_respects_limit(self, recon_conn):
        """Limit parameter restricts number of results."""
        from dashboard.data.reconciliation import get_recent_runs

        runs = await get_recent_runs(recon_conn, limit=1)
        assert len(runs) == 1
        assert runs[0]['id'] == 'run-002'  # Most recent

    async def test_empty_table(self, empty_recon_conn):
        """Returns empty list when runs table has no data."""
        from dashboard.data.reconciliation import get_recent_runs

        runs = await get_recent_runs(empty_recon_conn)
        assert runs == []

    async def test_missing_db_file(self):
        """Returns empty list when database connection is None."""
        from dashboard.data.reconciliation import get_recent_runs

        runs = await get_recent_runs(None)
        assert runs == []

    async def test_duration_with_mixed_naive_aware_timestamps(self, tmp_path):
        """Duration calculated correctly when started_at is naive and completed_at is aware."""
        import sqlite3

        from dashboard.data.reconciliation import get_recent_runs
        from tests.conftest import RECONCILIATION_SCHEMA

        db_path = tmp_path / 'mixed.db'
        conn = sqlite3.connect(str(db_path))
        conn.executescript(RECONCILIATION_SCHEMA)
        conn.execute(
            "INSERT INTO runs (id, project_id, run_type, trigger_reason, started_at, "
            "completed_at, events_processed, status) "
            "VALUES ('run-m1', 'dark_factory', 'full', 'test', '2026-03-28T10:00:00', "
            "'2026-03-28T10:05:00+00:00', 3, 'completed')"
        )
        conn.commit()
        conn.close()

        async with aiosqlite.connect(str(db_path)) as db:
            db.row_factory = aiosqlite.Row
            runs = await get_recent_runs(db)
        assert len(runs) == 1
        assert runs[0]['duration_seconds'] == pytest.approx(300.0, abs=1.0)

    async def test_null_started_at_with_completed_at_sets_duration_none(self, tmp_path, caplog):
        """NULL started_at with valid completed_at yields duration_seconds=None, no exception.

        SQLite enforces NOT NULL on the production schema, but NULLs can appear via
        migrations or schema changes.  We create a relaxed schema (no NOT NULL on
        started_at) to simulate such production data corruption.
        """
        import sqlite3

        from dashboard.data.reconciliation import get_recent_runs
        from tests.conftest import RECONCILIATION_SCHEMA

        # Replace 'started_at TEXT NOT NULL' with 'started_at TEXT' to allow NULL
        relaxed_schema = RECONCILIATION_SCHEMA.replace(
            'started_at TEXT NOT NULL', 'started_at TEXT'
        )

        db_path = tmp_path / 'null_started.db'
        conn = sqlite3.connect(str(db_path))
        conn.executescript(relaxed_schema)
        conn.execute(
            "INSERT INTO runs (id, project_id, run_type, trigger_reason, started_at,"
            " completed_at, events_processed, status)"
            " VALUES ('run-ns1', 'dark_factory', 'full', 'test', NULL,"
            " '2026-03-28T10:05:00+00:00', 0, 'completed')"
        )
        conn.commit()
        conn.close()

        with caplog.at_level(logging.DEBUG, logger='dashboard.data.reconciliation'):
            async with aiosqlite.connect(str(db_path)) as db:
                db.row_factory = aiosqlite.Row
                runs = await get_recent_runs(db)

        assert len(runs) == 1
        assert runs[0]['id'] == 'run-ns1'
        assert runs[0]['duration_seconds'] is None
        assert any('bad timestamps' in record.message for record in caplog.records)

    async def test_malformed_started_at_with_completed_at_sets_duration_none(
        self, tmp_path, caplog
    ):
        """Malformed started_at (non-ISO string) with valid completed_at yields duration_seconds=None.

        This exercises the ValueError branch of the try/except in get_recent_runs.
        """
        import sqlite3

        from dashboard.data.reconciliation import get_recent_runs
        from tests.conftest import RECONCILIATION_SCHEMA

        db_path = tmp_path / 'malformed_started.db'
        conn = sqlite3.connect(str(db_path))
        conn.executescript(RECONCILIATION_SCHEMA)
        conn.execute(
            "INSERT INTO runs (id, project_id, run_type, trigger_reason, started_at,"
            " completed_at, events_processed, status)"
            " VALUES ('run-mal1', 'dark_factory', 'full', 'test', 'not-a-timestamp',"
            " '2026-03-28T10:05:00+00:00', 0, 'completed')"
        )
        conn.commit()
        conn.close()

        with caplog.at_level(logging.DEBUG, logger='dashboard.data.reconciliation'):
            async with aiosqlite.connect(str(db_path)) as db:
                db.row_factory = aiosqlite.Row
                runs = await get_recent_runs(db)

        assert len(runs) == 1
        assert runs[0]['id'] == 'run-mal1'
        assert runs[0]['duration_seconds'] is None
        assert any('bad timestamps' in record.message for record in caplog.records)

    async def test_null_started_at_does_not_affect_other_rows(self, tmp_path):
        """Bad row with NULL started_at does not prevent other rows from being returned correctly.

        Verifies per-row isolation: a bad row yields duration_seconds=None while
        a healthy sibling row still has its duration computed correctly.
        """
        import sqlite3

        from dashboard.data.reconciliation import get_recent_runs
        from tests.conftest import RECONCILIATION_SCHEMA

        # Relaxed schema to allow NULL started_at
        relaxed_schema = RECONCILIATION_SCHEMA.replace(
            'started_at TEXT NOT NULL', 'started_at TEXT'
        )

        db_path = tmp_path / 'mixed_rows.db'
        conn = sqlite3.connect(str(db_path))
        conn.executescript(relaxed_schema)
        # Bad row: NULL started_at, valid completed_at
        conn.execute(
            "INSERT INTO runs (id, project_id, run_type, trigger_reason, started_at,"
            " completed_at, events_processed, status)"
            " VALUES ('run-bad', 'dark_factory', 'full', 'test', NULL,"
            " '2026-03-28T10:05:00+00:00', 0, 'completed')"
        )
        # Good row: valid started_at + completed_at, exactly 5 minutes apart
        conn.execute(
            "INSERT INTO runs (id, project_id, run_type, trigger_reason, started_at,"
            " completed_at, events_processed, status)"
            " VALUES ('run-good', 'dark_factory', 'full', 'test',"
            " '2026-03-28T09:00:00+00:00', '2026-03-28T09:05:00+00:00', 3, 'completed')"
        )
        conn.commit()
        conn.close()

        async with aiosqlite.connect(str(db_path)) as db:
            db.row_factory = aiosqlite.Row
            runs = await get_recent_runs(db)

        assert len(runs) == 2
        by_id = {r['id']: r for r in runs}
        assert by_id['run-bad']['duration_seconds'] is None
        assert by_id['run-good']['duration_seconds'] == pytest.approx(300.0, abs=1.0)


class TestGetWatermarks:
    """Tests for get_watermarks."""

    async def test_happy_path_returns_watermark_list(self, recon_conn):
        """Returns list of dicts with watermark fields for all projects."""
        from dashboard.data.reconciliation import get_watermarks

        result = await get_watermarks(recon_conn)

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

    async def test_empty_table(self, empty_recon_conn):
        """Returns empty list when watermarks table has no data."""
        from dashboard.data.reconciliation import get_watermarks

        result = await get_watermarks(empty_recon_conn)
        assert result == []

    async def test_missing_db_file(self):
        """Returns empty list when database connection is None."""
        from dashboard.data.reconciliation import get_watermarks

        result = await get_watermarks(None)
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

        async with aiosqlite.connect(str(db_path)) as db:
            db.row_factory = aiosqlite.Row
            result = await get_watermarks(db)
        assert len(result) == 2
        assert result[0]['project_id'] == 'alpha'
        assert result[1]['project_id'] == 'beta'


class TestGetLastAttemptedRun:
    """Tests for get_last_attempted_run."""

    async def test_happy_path_returns_per_project_dict(self, recon_conn):
        """Returns dict keyed by project_id with most recent run per project."""
        from dashboard.data.reconciliation import get_last_attempted_run

        result = await get_last_attempted_run(recon_conn)

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

        async with aiosqlite.connect(str(db_path)) as db:
            db.row_factory = aiosqlite.Row
            result = await get_last_attempted_run(db)
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

        async with aiosqlite.connect(str(db_path)) as db:
            db.row_factory = aiosqlite.Row
            result = await get_last_attempted_run(db)
        assert len(result) == 2
        assert result['alpha']['id'] == 'a2'  # most recent for alpha
        assert result['beta']['id'] == 'b1'

    async def test_empty_table(self, empty_recon_conn):
        """Returns empty dict when runs table has no data."""
        from dashboard.data.reconciliation import get_last_attempted_run

        result = await get_last_attempted_run(empty_recon_conn)
        assert result == {}

    async def test_missing_db_file(self):
        """Returns empty dict when database connection is None."""
        from dashboard.data.reconciliation import get_last_attempted_run

        result = await get_last_attempted_run(None)
        assert result == {}


class TestGetBufferStats:
    """Tests for get_buffer_stats."""

    async def test_happy_path_returns_count_and_age(self, recon_conn):
        """Returns dict with buffered_count and oldest_event_age_seconds."""
        from dashboard.data.reconciliation import get_buffer_stats

        result = await get_buffer_stats(recon_conn)

        assert isinstance(result, dict)
        assert set(result.keys()) == {'buffered_count', 'oldest_event_age_seconds'}
        # Fixture inserts 3 buffered events
        assert result['buffered_count'] == 3
        # Oldest event is 60 minutes old — age should be roughly 3600 seconds
        assert result['oldest_event_age_seconds'] is not None
        assert result['oldest_event_age_seconds'] >= 3500  # at least ~58 min

    async def test_empty_table(self, empty_recon_conn):
        """Returns zero count and None age when no buffered events."""
        from dashboard.data.reconciliation import get_buffer_stats

        result = await get_buffer_stats(empty_recon_conn)

        assert result == {'buffered_count': 0, 'oldest_event_age_seconds': None}

    async def test_missing_db_file(self):
        """Returns default dict when database connection is None."""
        from dashboard.data.reconciliation import get_buffer_stats

        result = await get_buffer_stats(None)

        assert result == {'buffered_count': 0, 'oldest_event_age_seconds': None}


class TestGetBurstState:
    """Tests for get_burst_state."""

    async def test_happy_path_returns_burst_state_list(self, recon_conn):
        """Returns list of dicts with correct fields for all agents."""
        from dashboard.data.reconciliation import get_burst_state

        result = await get_burst_state(recon_conn)

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

        async with aiosqlite.connect(str(db_path)) as db:
            db.row_factory = aiosqlite.Row
            result = await get_burst_state(db)
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

        async with aiosqlite.connect(str(db_path)) as db:
            db.row_factory = aiosqlite.Row
            result = await get_burst_state(db)
        assert len(result) == 1
        assert result[0]['state'] == 'bursting'
        assert result[0]['burst_started_at'] is not None

    async def test_empty_table(self, empty_recon_conn):
        """Returns empty list when burst_state table has no data."""
        from dashboard.data.reconciliation import get_burst_state

        result = await get_burst_state(empty_recon_conn)
        assert result == []

    async def test_missing_db_file(self):
        """Returns empty list when database connection is None."""
        from dashboard.data.reconciliation import get_burst_state

        result = await get_burst_state(None)
        assert result == []


class TestGetLatestVerdict:
    """Tests for get_latest_verdict."""

    async def test_happy_path_returns_verdict_dict(self, recon_conn):
        """Returns dict with verdict fields for the most recent verdict."""
        from dashboard.data.reconciliation import get_latest_verdict

        result = await get_latest_verdict(recon_conn)

        assert result is not None
        assert isinstance(result, dict)
        expected_keys = {'run_id', 'severity', 'action_taken', 'reviewed_at'}
        assert set(result.keys()) == expected_keys
        assert result['run_id'] == 'run-001'
        assert result['severity'] == 'low'
        assert result['action_taken'] == 'logged'

    async def test_empty_table(self, empty_recon_conn):
        """Returns None when judge_verdicts table has no data."""
        from dashboard.data.reconciliation import get_latest_verdict

        result = await get_latest_verdict(empty_recon_conn)
        assert result is None

    async def test_missing_db_file(self):
        """Returns None when database connection is None."""
        from dashboard.data.reconciliation import get_latest_verdict

        result = await get_latest_verdict(None)
        assert result is None


class TestExceptionLogging:
    """Tests that with_db emits DEBUG-level logs on DB unavailability."""

    async def test_missing_db_returns_default(self):
        """get_recent_runs with None connection returns default."""
        from dashboard.data.reconciliation import get_recent_runs

        result = await get_recent_runs(None)

        assert result == []

    async def test_operational_error_logs_debug(self, tmp_path, caplog):
        """with_db emits a DEBUG log on OperationalError (no-table DB)."""
        from dashboard.data.reconciliation import get_watermarks

        # Create a valid SQLite file but with no tables — causes OperationalError
        db_path = tmp_path / 'empty.db'
        import sqlite3
        sqlite3.connect(str(db_path)).close()

        async with aiosqlite.connect(str(db_path)) as db:
            db.row_factory = aiosqlite.Row
            with caplog.at_level(logging.DEBUG, logger='dashboard.data.db'):
                result = await get_watermarks(db)

        assert result == []
        assert any(
            r.levelno == logging.DEBUG and 'dashboard.data.db' in r.name
            for r in caplog.records
        ), f'Expected DEBUG log from dashboard.data.db, got: {caplog.records}'


class TestGetJournalEntries:
    """Tests for get_journal_entries."""

    async def test_happy_path_returns_entries(self, recon_conn):
        """Returns list of journal entry dicts for a given run_id."""
        from dashboard.data.reconciliation import get_journal_entries

        entries = await get_journal_entries(recon_conn, 'run-001')

        assert len(entries) == 2
        expected_fields = {
            'id', 'stage', 'timestamp', 'operation', 'target_system',
            'before_state', 'after_state', 'reasoning', 'evidence',
        }
        for entry in entries:
            assert set(entry.keys()) == expected_fields

        # Ordered by timestamp — je-001 first
        assert entries[0]['id'] == 'je-001'
        assert entries[0]['operation'] == 'consolidate'
        assert entries[0]['target_system'] == 'mem0'

    async def test_json_fields_parsed(self, recon_conn):
        """JSON fields (before_state, after_state, evidence) are parsed."""
        from dashboard.data.reconciliation import get_journal_entries

        entries = await get_journal_entries(recon_conn, 'run-001')

        assert entries[0]['before_state'] == {'count': 5}
        assert entries[0]['after_state'] == {'count': 3}
        assert entries[0]['evidence'] == [{'source': 'mem0', 'id': 'm-1'}]

    async def test_null_json_fields(self, recon_conn):
        """NULL JSON fields return None/empty defaults."""
        from dashboard.data.reconciliation import get_journal_entries

        entries = await get_journal_entries(recon_conn, 'run-001')

        # je-002 has NULL before_state
        assert entries[1]['before_state'] is None
        assert entries[1]['after_state'] == {'entities': 2}
        assert entries[1]['evidence'] == []

    async def test_empty_result_for_unknown_run(self, recon_conn):
        """Returns empty list for a run_id with no entries."""
        from dashboard.data.reconciliation import get_journal_entries

        entries = await get_journal_entries(recon_conn, 'nonexistent-run')
        assert entries == []

    async def test_missing_db_file(self):
        """Returns empty list when database connection is None."""
        from dashboard.data.reconciliation import get_journal_entries

        entries = await get_journal_entries(None, 'run-001')
        assert entries == []


class TestParseUtc:
    """Tests for the parse_utc shared utility in dashboard.data.utils."""

    def test_naive_iso_string_gets_utc(self):
        """Naive ISO string (no tzinfo) should be returned with UTC attached."""
        from datetime import UTC as _UTC

        from dashboard.data.utils import parse_utc

        result = parse_utc('2026-03-28T10:00:00')
        assert result.tzinfo is not None
        assert result.tzinfo == _UTC

    def test_aware_iso_string_preserved(self):
        """Aware ISO string (with tzinfo) should be returned unchanged."""
        from dashboard.data.utils import parse_utc

        ts = '2026-03-28T10:00:00+00:00'
        result = parse_utc(ts)
        assert result.tzinfo is not None
        # Value is preserved: tzinfo stays, offset is the same
        assert result.year == 2026
        assert result.hour == 10

    def test_aware_iso_string_with_non_utc_offset_preserved(self):
        """Aware ISO string with non-UTC offset should be returned unchanged (tzinfo preserved)."""
        from datetime import timedelta

        from dashboard.data.utils import parse_utc

        result = parse_utc('2026-03-28T10:00:00+05:30')
        assert result.utcoffset() == timedelta(hours=5, minutes=30)
        assert result.hour == 10

    def test_invalid_string_raises_value_error(self):
        """Invalid ISO string should raise ValueError."""
        import pytest

        from dashboard.data.utils import parse_utc

        with pytest.raises(ValueError):
            parse_utc('not-a-timestamp')

    def test_none_raises_type_error(self):
        """None input should raise TypeError."""
        import pytest

        from dashboard.data.utils import parse_utc

        with pytest.raises(TypeError):
            parse_utc(None)  # type: ignore[arg-type]


class TestPartitionBurstState:
    """Functional tests for partition_burst_state."""

    def test_bursting_agent_is_active(self):
        from dashboard.data.reconciliation import partition_burst_state

        agents = [{'agent_id': 'a1', 'state': 'bursting', 'last_write_at': '2020-01-01T00:00:00+00:00'}]
        active, idle = partition_burst_state(agents)
        assert len(active) == 1
        assert len(idle) == 0

    def test_idle_old_agent_is_idle(self):
        from dashboard.data.reconciliation import partition_burst_state

        agents = [{'agent_id': 'a1', 'state': 'idle', 'last_write_at': '2020-01-01T00:00:00+00:00'}]
        active, idle = partition_burst_state(agents)
        assert len(active) == 0
        assert len(idle) == 1

    def test_idle_recent_agent_is_active(self):
        from dashboard.data.reconciliation import partition_burst_state

        recent_ts = (datetime.now(UTC) - timedelta(minutes=30)).isoformat()
        agents = [{'agent_id': 'a1', 'state': 'idle', 'last_write_at': recent_ts}]
        active, idle = partition_burst_state(agents)
        assert len(active) == 1
        assert len(idle) == 0

    def test_mixed_partition(self):
        from dashboard.data.reconciliation import partition_burst_state

        old_ts = '2020-01-01T00:00:00+00:00'
        recent_ts = (datetime.now(UTC) - timedelta(minutes=10)).isoformat()
        agents = [
            {'agent_id': 'bursting', 'state': 'bursting', 'last_write_at': old_ts},
            {'agent_id': 'idle-old', 'state': 'idle', 'last_write_at': old_ts},
            {'agent_id': 'idle-recent', 'state': 'idle', 'last_write_at': recent_ts},
        ]
        active, idle = partition_burst_state(agents)
        assert [a['agent_id'] for a in active] == ['bursting', 'idle-recent']
        assert [a['agent_id'] for a in idle] == ['idle-old']

    def test_empty_list(self):
        from dashboard.data.reconciliation import partition_burst_state

        active, idle = partition_burst_state([])
        assert active == []
        assert idle == []

    def test_invalid_timestamp_treated_as_idle(self):
        from dashboard.data.reconciliation import partition_burst_state

        agents = [{'agent_id': 'a1', 'state': 'idle', 'last_write_at': 'not-a-date'}]
        active, idle = partition_burst_state(agents)
        assert len(active) == 0
        assert len(idle) == 1

    def test_custom_threshold(self):
        from dashboard.data.reconciliation import partition_burst_state

        ts = (datetime.now(UTC) - timedelta(minutes=5)).isoformat()
        agents = [{'agent_id': 'a1', 'state': 'idle', 'last_write_at': ts}]
        # 5 min old, threshold 3 min → idle
        active, idle = partition_burst_state(agents, active_threshold_seconds=180)
        assert len(active) == 0
        assert len(idle) == 1
        # 5 min old, threshold 10 min → active
        active, idle = partition_burst_state(agents, active_threshold_seconds=600)
        assert len(active) == 1
        assert len(idle) == 0

    def test_missing_state_key_defaults_to_idle(self):
        # Agent with no 'state' key defaults to 'idle' (via .get('state', 'idle')).
        # Combined with a stale 2020 timestamp, the agent is classified as idle.
        from dashboard.data.reconciliation import partition_burst_state

        agents = [{'agent_id': 'a1', 'last_write_at': '2020-01-01T00:00:00+00:00'}]
        active, idle = partition_burst_state(agents)
        assert len(active) == 0
        assert len(idle) == 1

    def test_missing_state_key_recent_write_is_active(self):
        # Agent with no 'state' key defaults to 'idle' (via .get('state', 'idle')),
        # but a recent last_write_at timestamp promotes it to active.
        from dashboard.data.reconciliation import partition_burst_state

        recent_ts = (datetime.now(UTC) - timedelta(minutes=30)).isoformat()
        agents = [{'agent_id': 'a1', 'last_write_at': recent_ts}]
        active, idle = partition_burst_state(agents)
        assert len(active) == 1
        assert len(idle) == 0

    def test_non_idle_state_missing_last_write_at_is_active(self):
        # Agent with state='bursting' and no 'last_write_at' key is classified active.
        # The early return (state != 'idle') fires before the timestamp lookup.
        from dashboard.data.reconciliation import partition_burst_state

        agents = [{'agent_id': 'a1', 'state': 'bursting'}]
        active, idle = partition_burst_state(agents)
        assert len(active) == 1
        assert len(idle) == 0

    def test_bare_agent_dict_no_state_no_last_write_at(self):
        # Agent dict with NEITHER 'state' NOR 'last_write_at' key should land in
        # idle list without raising KeyError.
        from dashboard.data.reconciliation import partition_burst_state

        agents = [{'agent_id': 'x'}]
        active, idle = partition_burst_state(agents)
        assert len(active) == 0
        assert len(idle) == 1

    def test_explicit_idle_missing_last_write_at(self):
        # Agent with explicit state='idle' and NO 'last_write_at' key should land
        # in idle list without raising KeyError.
        from dashboard.data.reconciliation import partition_burst_state

        agents = [{'agent_id': 'x', 'state': 'idle'}]
        active, idle = partition_burst_state(agents)
        assert len(active) == 0
        assert len(idle) == 1

    def test_malformed_timestamp_logs_debug(self, caplog):
        """A malformed last_write_at timestamp should emit a DEBUG log with agent_id."""
        from dashboard.data.reconciliation import partition_burst_state

        agents = [{'agent_id': 'bad-agent', 'state': 'idle', 'last_write_at': 'not-a-date'}]
        with caplog.at_level(logging.DEBUG, logger='dashboard.data.reconciliation'):
            active, idle = partition_burst_state(agents)

        # Agent still lands in idle (existing behaviour unchanged)
        assert len(active) == 0
        assert len(idle) == 1

        # A DEBUG record must be emitted containing 'bad last_write_at' and the agent_id
        debug_records = [
            r for r in caplog.records
            if r.levelno == logging.DEBUG
            and r.name == 'dashboard.data.reconciliation'
        ]
        assert any(
            'bad last_write_at' in r.getMessage() and 'bad-agent' in r.getMessage()
            for r in debug_records
        ), f"Expected debug log with 'bad last_write_at' and 'bad-agent', got: {[r.getMessage() for r in debug_records]}"

    def test_none_last_write_at_handled_consistently(self, caplog):
        """Agent with None last_write_at lands in idle and emits a debug log."""
        from dashboard.data.reconciliation import partition_burst_state

        agents = [{'agent_id': 'a1', 'state': 'idle', 'last_write_at': None}]
        with caplog.at_level(logging.DEBUG, logger='dashboard.data.reconciliation'):
            active, idle = partition_burst_state(agents)

        assert len(idle) == 1
        assert idle[0]['agent_id'] == 'a1'
        assert len(active) == 0
        assert any(
            r.levelno == logging.DEBUG and 'bad last_write_at' in r.message
            for r in caplog.records
        ), f'Expected DEBUG log with "bad last_write_at", got: {caplog.records}'

    def test_state_none_old_timestamp_is_idle(self):
        """Agent with state=None and a stale timestamp must be classified as idle.

        state=None is not a valid burst-state value; it must normalise to 'idle'
        so the agent goes through the timestamp-recency check rather than being
        unconditionally promoted to active (which is the current buggy behaviour
        caused by ``agent.get('state', 'idle')`` returning None when the key
        exists with value None, and ``None != 'idle'`` being True).
        """
        from dashboard.data.reconciliation import partition_burst_state

        agents = [{'agent_id': 'a1', 'state': None, 'last_write_at': '2020-01-01T00:00:00+00:00'}]
        active, idle = partition_burst_state(agents)
        assert len(active) == 0, (
            f'Expected 0 active agents, got {len(active)}: {active}'
        )
        assert len(idle) == 1

    def test_state_none_recent_timestamp_is_active_via_recency(self):
        """Agent with state=None and a recent timestamp must be active via the
        timestamp-recency path (not the ``state != 'idle'`` short-circuit).

        This test verifies the threshold is actually consulted: with a 5-minute-old
        timestamp the agent is active under a 10-minute threshold but idle under a
        3-minute threshold.  If the buggy ``None != 'idle'`` short-circuit were in
        effect the agent would be active under *both* thresholds regardless of the
        timestamp, causing the second assertion to fail.
        """
        from dashboard.data.reconciliation import partition_burst_state

        ts = (datetime.now(UTC) - timedelta(minutes=5)).isoformat()
        agents = [{'agent_id': 'a1', 'state': None, 'last_write_at': ts}]

        # 5-min-old write, 10-min threshold → active (timestamp recent enough)
        active, idle = partition_burst_state(agents, active_threshold_seconds=600)
        assert len(active) == 1, (
            f'Expected 1 active agent (10-min threshold), got {len(active)}'
        )
        assert len(idle) == 0

        # 5-min-old write, 3-min threshold → idle (timestamp too stale)
        active, idle = partition_burst_state(agents, active_threshold_seconds=180)
        assert len(active) == 0, (
            f'Expected 0 active agents (3-min threshold), got {len(active)}: {active}'
        )
        assert len(idle) == 1


class TestWithDb:
    """Unit tests for the with_db helper function."""

    async def test_returns_callback_result_on_success(self, recon_conn):
        """Helper returns the value returned by the callback on success."""
        from dashboard.data.db import with_db

        async def _fn(db):
            async with db.execute('SELECT 1') as cur:
                row = await cur.fetchone()
            return row[0]

        result = await with_db(recon_conn, _fn, default=99)
        assert result == 1

    async def test_returns_default_on_none_connection(self):
        """Helper returns the default value when the connection is None."""
        from dashboard.data.db import with_db

        async def _fn(db):
            return 'should not reach here'

        result = await with_db(None, _fn, default=[])
        assert result == []

    async def test_returns_default_on_operational_error(self, tmp_path):
        """Helper returns the default value on sqlite3.OperationalError (no-table DB)."""
        import sqlite3 as _sqlite3

        from dashboard.data.db import with_db

        # Valid SQLite file but with no tables — querying any table raises OperationalError
        db_path = tmp_path / 'notables.db'
        _sqlite3.connect(str(db_path)).close()

        async def _fn(db):
            async with db.execute('SELECT * FROM nonexistent_table') as cur:
                return await cur.fetchall()

        async with aiosqlite.connect(str(db_path)) as db:
            db.row_factory = aiosqlite.Row
            result = await with_db(db, _fn, default={})
        assert result == {}

    async def test_returns_default_on_none_logs_nothing(self, caplog):
        """Helper returns default for None connection without logging."""
        from dashboard.data.db import with_db

        async def _fn(db):
            return 'noop'

        with caplog.at_level(logging.DEBUG, logger='dashboard.data.db'):
            await with_db(None, _fn, default=None)

        # None connection is a fast path — no log expected
        db_records = [
            r for r in caplog.records
            if 'dashboard.data.db' in r.name
        ]
        assert not db_records, f'Expected no logs for None path, got: {db_records}'

    async def test_emits_debug_log_on_operational_error(self, tmp_path, caplog):
        """Helper emits a DEBUG log on OperationalError."""
        import sqlite3 as _sqlite3

        from dashboard.data.db import with_db

        db_path = tmp_path / 'notables2.db'
        _sqlite3.connect(str(db_path)).close()

        async def _fn(db):
            async with db.execute('SELECT * FROM missing_table') as cur:
                return await cur.fetchall()

        async with aiosqlite.connect(str(db_path)) as db:
            db.row_factory = aiosqlite.Row
            with caplog.at_level(logging.DEBUG, logger='dashboard.data.db'):
                await with_db(db, _fn, default=None)

        debug_records = [
            r for r in caplog.records
            if r.levelno == logging.DEBUG and 'dashboard.data.db' in r.name
        ]
        assert debug_records, f'Expected DEBUG log, got: {caplog.records}'
