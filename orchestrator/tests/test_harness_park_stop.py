"""Tests for Harness.pause_scheduler / resume_scheduler and related wiring.

Task 1322 — AFK hardening: Scheduler park-and-stop with configurable trip conditions.
"""

from __future__ import annotations

import json
import sqlite3
from datetime import datetime, timezone
from pathlib import Path
from unittest.mock import MagicMock, patch

import pytest

from orchestrator.config import OrchestratorConfig
from orchestrator.event_store import EventStore, EventType
from orchestrator.harness import Harness
from orchestrator.run_store import RunStore
from orchestrator.scheduler import Scheduler


def _make_harness_with_mocks(tmp_path: Path) -> tuple[Harness, MagicMock, EventStore]:
    """Build a Harness with a real Scheduler, a MagicMock RunStore, and a real EventStore.

    The Harness is constructed via __new__ to avoid the full startup sequence
    (which requires a running MCP server), then the relevant attributes are
    populated directly — matching the pattern in test_harness_terminal_status_watcher.py.
    """
    config = OrchestratorConfig(project_root=tmp_path)
    harness = Harness(config)

    # Inject a mock RunStore so we can spy on persistence calls.
    mock_run_store = MagicMock(spec=RunStore)
    harness._run_store = mock_run_store
    harness._run_id = 'run-test-0001'

    # Inject a real EventStore pointing at a tmp DB so we can query events.
    db_path = tmp_path / 'events.db'
    event_store = EventStore(db_path, 'run-test-0001')
    harness.event_store = event_store

    return harness, mock_run_store, event_store


def _query_events(event_store: EventStore, event_type_str: str) -> list[dict]:
    """Query all rows matching event_type_str from the EventStore's DB."""
    conn = sqlite3.connect(str(event_store.db_path))
    conn.row_factory = sqlite3.Row
    rows = conn.execute(
        'SELECT * FROM events WHERE event_type = ? ORDER BY id',
        (event_type_str,),
    ).fetchall()
    conn.close()
    return [dict(r) for r in rows]


class TestHarnessPauseScheduler:
    """Tests for Harness.pause_scheduler and resume_scheduler."""

    @pytest.mark.asyncio
    async def test_pause_scheduler_sets_scheduler_paused(self, tmp_path: Path) -> None:
        """pause_scheduler() delegates to scheduler.pause() — is_paused and reason are set."""
        harness, _, _ = _make_harness_with_mocks(tmp_path)

        await harness.pause_scheduler('test reason')

        assert harness.scheduler.is_paused is True
        assert harness.scheduler.pause_reason == 'test reason'

    @pytest.mark.asyncio
    async def test_pause_scheduler_persists_via_runstore(self, tmp_path: Path) -> None:
        """pause_scheduler() calls run_store.save_scheduler_pause with the correct args."""
        harness, mock_run_store, _ = _make_harness_with_mocks(tmp_path)

        await harness.pause_scheduler('test reason')

        mock_run_store.save_scheduler_pause.assert_called_once()
        call_kwargs = mock_run_store.save_scheduler_pause.call_args
        # Accept both positional and keyword call styles.
        kwargs = call_kwargs.kwargs if call_kwargs.kwargs else {}
        args = call_kwargs.args if call_kwargs.args else ()
        # Flatten: save_scheduler_pause(project_id, reason, pause_at_iso, set_by_run_id)
        all_args = dict(zip(
            ['project_id', 'reason', 'pause_at_iso', 'set_by_run_id'],
            args,
        ))
        all_args.update(kwargs)
        assert all_args.get('reason') == 'test reason', (
            f'Expected reason="test reason"; got {all_args!r}'
        )
        assert all_args.get('set_by_run_id') == 'run-test-0001', (
            f'Expected set_by_run_id=run-test-0001; got {all_args!r}'
        )
        # pause_at_iso must be parseable as an ISO timestamp.
        pause_at = all_args.get('pause_at_iso', '')
        datetime.fromisoformat(pause_at)  # raises ValueError if unparseable

    @pytest.mark.asyncio
    async def test_pause_scheduler_emits_event(self, tmp_path: Path) -> None:
        """pause_scheduler() emits EventType.scheduler_paused with reason in data."""
        harness, _, event_store = _make_harness_with_mocks(tmp_path)

        await harness.pause_scheduler('my pause reason')

        rows = _query_events(event_store, 'scheduler_paused')
        assert len(rows) == 1, f'Expected 1 scheduler_paused event; got {len(rows)}'
        data = json.loads(rows[0]['data'] or '{}')
        assert data.get('reason') == 'my pause reason', (
            f'Expected reason in data; got {data!r}'
        )

    @pytest.mark.asyncio
    async def test_resume_scheduler_clears_scheduler_paused(self, tmp_path: Path) -> None:
        """resume_scheduler() delegates to scheduler.resume() and cleans up state."""
        harness, mock_run_store, event_store = _make_harness_with_mocks(tmp_path)

        await harness.pause_scheduler('will be resumed')
        await harness.resume_scheduler()

        assert harness.scheduler.is_paused is False
        assert harness.scheduler.pause_reason is None
        mock_run_store.clear_scheduler_pause.assert_called_once()
        rows = _query_events(event_store, 'scheduler_resumed')
        assert len(rows) == 1, f'Expected 1 scheduler_resumed event; got {len(rows)}'
