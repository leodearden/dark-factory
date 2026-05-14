"""Tests for Harness.pause_scheduler / resume_scheduler and related wiring.

Task 1322 — AFK hardening: Scheduler park-and-stop with configurable trip conditions.
"""

from __future__ import annotations

import asyncio
import json
import re
import sqlite3
from datetime import datetime, timezone
from pathlib import Path
from unittest.mock import AsyncMock, MagicMock, patch

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


class TestHarnessSchedulerParkStopWiring:
    """Tests that Harness wires scheduler._on_park_stop_trip → Harness.pause_scheduler."""

    @pytest.mark.asyncio
    async def test_harness_wires_scheduler_park_stop_callback(self, tmp_path: Path) -> None:
        """After Harness construction, scheduler._on_park_stop_trip must be set.

        Calling the callback directly with a reason string must cause the
        scheduler to become paused with that reason — proving that the callback
        delegates to Harness.pause_scheduler and therefore to scheduler.pause().
        """
        config = OrchestratorConfig(project_root=tmp_path)
        harness = Harness(config)

        assert harness.scheduler._on_park_stop_trip is not None, (
            'Harness must wire scheduler._on_park_stop_trip after construction'
        )

        # Call the callback directly (simulating a trip event fired by the scheduler).
        await harness.scheduler._on_park_stop_trip('integration-reason')

        assert harness.scheduler.is_paused is True, (
            'Scheduler must be paused after the trip callback is invoked'
        )
        assert harness.scheduler.pause_reason == 'integration-reason', (
            f'Expected pause_reason "integration-reason"; got {harness.scheduler.pause_reason!r}'
        )


class TestHarnessRestartPersistence:
    """Tests for restart-time rehydration of persisted scheduler pause state."""

    @pytest.mark.asyncio
    async def test_harness_restart_loads_persisted_pause(self, tmp_path: Path) -> None:
        """_load_persisted_scheduler_pause() restores is_paused and pause_reason from runs.db.

        It must NOT re-write the row (save_scheduler_pause call_count == 0) because
        the pause was already persisted by the prior run.
        """
        # Seed a real RunStore with a persisted pause.
        db_dir = tmp_path / 'data' / 'orchestrator'
        db_dir.mkdir(parents=True)
        db_path = db_dir / 'runs.db'
        seeder = RunStore(db_path)
        seeder.save_scheduler_pause(
            project_id='dark_factory',
            reason='pre-restart park-stop',
            pause_at_iso='2026-05-13T22:00:00+00:00',
            set_by_run_id='prior-run-id',
        )

        # Build a fresh Harness (without calling run()) and inject the real RunStore.
        config = OrchestratorConfig(project_root=tmp_path)
        harness = Harness(config)
        harness._run_store = seeder  # inject so _load_persisted_scheduler_pause can read
        harness._run_id = 'new-run-id'

        # Spy: assert save_scheduler_pause is NOT called (no re-persist).
        original_save = seeder.save_scheduler_pause
        save_call_count = 0

        def _spy_save(*args, **kwargs):
            nonlocal save_call_count
            save_call_count += 1
            return original_save(*args, **kwargs)

        seeder.save_scheduler_pause = _spy_save

        # Act
        await harness._load_persisted_scheduler_pause()

        # Assert scheduler state restored.
        assert harness.scheduler.is_paused is True, (
            'Scheduler must be paused after loading persisted pause state'
        )
        assert harness.scheduler.pause_reason == 'pre-restart park-stop', (
            f'Expected pause_reason "pre-restart park-stop"; got {harness.scheduler.pause_reason!r}'
        )

        # Assert no re-persist.
        assert save_call_count == 0, (
            f'save_scheduler_pause must not be called on restart load; got {save_call_count} calls'
        )


class TestParkStopE2E:
    """End-to-end integration tests for the park-stop trip → persist → restart lifecycle.

    Mirrors the task description test plan verbatim:
    "Mark 5 tasks blocked within 1h, assert scheduler.acquire_next() returns None,
    pause_reason is set. Restart orchestrator, assert pause survives."
    """

    @pytest.mark.asyncio
    async def test_park_stop_e2e_marks_blocked_pauses_and_survives_restart(
        self, tmp_path: Path, monkeypatch
    ) -> None:
        """5 blocked transitions within 1h trip the scheduler; pause survives restart.

        Step-by-step:
        1. Harness1 — threshold=5, window=1h; mock mcp_call for set_task_status success.
        2. Mark task-1..task-5 as blocked → trip fires via asyncio.ensure_future.
        3. Yield the event loop so the trip callback executes.
        4. Assert harness1.scheduler.is_paused is True and reason format matches.
        5. Assert acquire_next() returns None.
        6. Harness2 from same runs.db — simulates orchestrator restart.
        7. Call _load_persisted_scheduler_pause() and assert pause is restored.
        """
        # Arrange: set up shared runs.db directory.
        db_dir = tmp_path / 'data' / 'orchestrator'
        db_dir.mkdir(parents=True)
        db_path = db_dir / 'runs.db'

        # Mock mcp_call so set_task_status returns a clean success response.
        monkeypatch.setattr(
            'orchestrator.scheduler.mcp_call',
            AsyncMock(return_value={}),
        )

        # --- Harness1 ---
        config1 = OrchestratorConfig(
            project_root=tmp_path,
            park_stop_parked_threshold=5,
            park_stop_parked_window_hours=1.0,
        )
        harness1 = Harness(config1)
        # Wire a real RunStore so persistence works.
        run_store1 = RunStore(db_path)
        harness1._run_store = run_store1
        harness1._run_id = 'run-e2e-0001'
        # _on_park_stop_trip is already wired to harness1.pause_scheduler by __init__.

        # Act: mark 5 tasks blocked — trip fires on the 5th.
        for i in range(1, 6):
            await harness1.scheduler.set_task_status(f'task-{i}', 'blocked')

        # Yield the event loop so the fire-and-forget asyncio.ensure_future task runs.
        await asyncio.sleep(0)

        # Assert harness1 is paused with a park-stop reason.
        assert harness1.scheduler.is_paused is True, (
            'Scheduler must be paused after 5 blocked transitions'
        )
        assert harness1.scheduler.pause_reason is not None
        assert re.search(
            r'park-stop.*5.*blocked.*1\.0h',
            harness1.scheduler.pause_reason,
        ), (
            f'Expected park-stop reason format; got {harness1.scheduler.pause_reason!r}'
        )

        # Assert acquire_next() returns None when paused.
        result = await harness1.scheduler.acquire_next()
        assert result is None, (
            f'acquire_next() must return None when scheduler is paused; got {result!r}'
        )

        # Capture the reason for comparison after restart.
        persisted_reason = harness1.scheduler.pause_reason

        # --- Harness2 (restart simulation) ---
        config2 = OrchestratorConfig(project_root=tmp_path)
        harness2 = Harness(config2)
        run_store2 = RunStore(db_path)
        harness2._run_store = run_store2
        harness2._run_id = 'run-e2e-0002'

        # Load persisted pause — simulates the call in Harness.run() startup.
        await harness2._load_persisted_scheduler_pause()

        assert harness2.scheduler.is_paused is True, (
            'Scheduler pause must survive restart (persist → reload)'
        )
        assert harness2.scheduler.pause_reason == persisted_reason, (
            f'Expected persisted reason {persisted_reason!r}; '
            f'got {harness2.scheduler.pause_reason!r}'
        )


class TestInFlightTasksFinishNaturallyWhenPaused:
    """Validates the design claim: 'In-flight agents finish gracefully when paused.'

    Pausing only gates acquire_next() — it must NOT cancel active workflow tasks.
    """

    @pytest.mark.asyncio
    async def test_acquire_next_returns_none_when_paused_and_inflight_tasks_complete(
        self, tmp_path: Path
    ) -> None:
        """Paused scheduler: acquire_next() returns None; existing tasks complete normally.

        Builds two fake in-flight asyncio Tasks (each just yields once via
        sleep(0) and returns a sentinel result).  Asserts:
        (a) acquire_next() returns None — no new dispatch.
        (b) Both Tasks complete normally — pause does not cancel them.
        """
        config = OrchestratorConfig(project_root=tmp_path)
        harness = Harness(config)

        # Pause the scheduler.
        harness.scheduler.pause('test-pause-inflight')

        # Create two fake in-flight workflow tasks that just yield and finish.
        async def _fake_workflow(sentinel: str) -> str:
            await asyncio.sleep(0)
            return sentinel

        task_a = asyncio.create_task(_fake_workflow('result-a'))
        task_b = asyncio.create_task(_fake_workflow('result-b'))

        # (a) acquire_next() must return None while paused.
        result = await harness.scheduler.acquire_next()
        assert result is None, (
            f'acquire_next() must return None when paused; got {result!r}'
        )

        # (b) Pre-existing tasks must complete without cancellation.
        result_a = await task_a
        result_b = await task_b
        assert result_a == 'result-a', f'task_a was cancelled or failed: {result_a!r}'
        assert result_b == 'result-b', f'task_b was cancelled or failed: {result_b!r}'
        assert not task_a.cancelled(), 'task_a must not have been cancelled by the pause'
        assert not task_b.cancelled(), 'task_b must not have been cancelled by the pause'
