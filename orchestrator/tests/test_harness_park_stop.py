"""Tests for Harness.pause_scheduler / resume_scheduler and related wiring.

Task 1322 — AFK hardening: Scheduler park-and-stop with configurable trip conditions.
"""

from __future__ import annotations

import asyncio
import json
import re
import sqlite3
from datetime import UTC, datetime, timedelta
from pathlib import Path
from unittest.mock import AsyncMock, MagicMock

import pytest

from orchestrator.config import OrchestratorConfig
from orchestrator.event_store import EventStore
from orchestrator.harness import Harness
from orchestrator.run_store import RunStore
from shared.cost_store import CostStore


# ---------------------------------------------------------------------------
# Helpers shared by cost-ceiling tests
# ---------------------------------------------------------------------------

async def _make_cost_store(tmp_path: Path, filename: str = 'cost.db') -> CostStore:
    """Open a real CostStore in tmp_path; caller is responsible for close."""
    store = CostStore(tmp_path / filename)
    await store.open()
    return store


def _iso(delta: timedelta) -> str:
    """Return an ISO timestamp offset from *now* by delta (negative = in the past)."""
    return (datetime.now(UTC) + delta).isoformat()


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
            strict=False,
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

        # Act
        await harness._load_persisted_scheduler_pause()

        # Assert scheduler state restored.
        assert harness.scheduler.is_paused is True, (
            'Scheduler must be paused after loading persisted pause state'
        )
        assert harness.scheduler.pause_reason == 'pre-restart park-stop', (
            f'Expected pause_reason "pre-restart park-stop"; got {harness.scheduler.pause_reason!r}'
        )

        # Assert the persisted row was NOT re-written on restart: re-load from
        # disk and check every field still matches the seed values.  This is
        # the load-bearing assertion — a stale spy on save_scheduler_pause
        # would pass vacuously because the load path never invokes save.
        reloaded = RunStore(db_path).load_scheduler_pause('dark_factory')
        assert reloaded is not None, 'Seed row must still exist after load'
        assert reloaded == {
            'reason': 'pre-restart park-stop',
            'pause_at': '2026-05-13T22:00:00+00:00',
            'set_by_run_id': 'prior-run-id',
        }, (
            f'Persisted row must be unchanged after restart load (no re-write '
            f'with new run_id); got {reloaded!r}'
        )

    @pytest.mark.asyncio
    async def test_load_persisted_pause_emits_restored_event(self, tmp_path: Path) -> None:
        """_load_persisted_scheduler_pause() must emit scheduler_pause_restored.

        Operators querying the event log for a run that starts with dispatch
        halted would see no event explaining WHY without this (the scheduler_paused
        event is on the previous run_id).  The restored event carries {reason,
        pause_at, restored_from_run_id} so the timeline self-documents cross-run
        continuity.
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
            set_by_run_id='prior-run-007',
        )

        # Build a Harness with a real EventStore so we can query emitted events.
        config = OrchestratorConfig(project_root=tmp_path)
        harness = Harness(config)
        harness._run_store = seeder
        harness._run_id = 'new-run-001'
        event_db = tmp_path / 'events.db'
        harness.event_store = EventStore(event_db, 'new-run-001')

        # Act
        await harness._load_persisted_scheduler_pause()

        # Assert scheduler state restored.
        assert harness.scheduler.is_paused is True

        # Assert the restored event was emitted with the expected payload.
        rows = _query_events(harness.event_store, 'scheduler_pause_restored')
        assert len(rows) == 1, (
            f'Expected 1 scheduler_pause_restored event; got {len(rows)}'
        )
        import json
        data = json.loads(rows[0]['data'])
        assert data['reason'] == 'pre-restart park-stop', (
            f'Expected reason "pre-restart park-stop"; got {data["reason"]!r}'
        )
        assert data['pause_at'] == '2026-05-13T22:00:00+00:00', (
            f'Expected pause_at to match; got {data["pause_at"]!r}'
        )
        assert data['restored_from_run_id'] == 'prior-run-007', (
            f'Expected restored_from_run_id "prior-run-007"; got {data["restored_from_run_id"]!r}'
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


class TestHarnessCostCeiling:
    """Tests for cost-ceiling config fields, _trailing_24h_cost_usd, and _enforce_cost_ceilings.

    Task 1323 — AFK hardening: Daily cost ceiling with watcher + orch-wide budgets.
    """

    # ------------------------------------------------------------------
    # Step 1 — config fields
    # ------------------------------------------------------------------

    def test_config_defaults(self, tmp_path: Path) -> None:
        """OrchestratorConfig exposes the two ceiling fields with correct defaults."""
        config = OrchestratorConfig(project_root=tmp_path)
        assert config.watcher_daily_cost_ceiling_usd == 50.0, (
            f'Expected 50.0; got {config.watcher_daily_cost_ceiling_usd!r}'
        )
        assert config.orch_daily_cost_ceiling_usd == 200.0, (
            f'Expected 200.0; got {config.orch_daily_cost_ceiling_usd!r}'
        )

    def test_config_overridable(self, tmp_path: Path) -> None:
        """Both ceiling fields accept custom values."""
        config = OrchestratorConfig(
            project_root=tmp_path,
            watcher_daily_cost_ceiling_usd=7.0,
            orch_daily_cost_ceiling_usd=99.5,
        )
        assert config.watcher_daily_cost_ceiling_usd == 7.0
        assert config.orch_daily_cost_ceiling_usd == 99.5

    # ------------------------------------------------------------------
    # Step 3 — _trailing_24h_cost_usd (all roles, watcher_only=False)
    # ------------------------------------------------------------------

    @pytest.mark.asyncio
    async def test_trailing_24h_cost_usd_sums_within_window(self, tmp_path: Path) -> None:
        """_trailing_24h_cost_usd sums rows within the 24h window and excludes older ones."""
        harness, _, _ = _make_harness_with_mocks(tmp_path)
        store = await _make_cost_store(tmp_path)
        harness.cost_store = store

        # Within window: two rows, different roles
        await store.save_invocation(
            run_id='r1', task_id='t1', project_id='dark_factory',
            account_name='acct', model='sonnet', role='orchestrator',
            cost_usd=5.0, input_tokens=None, output_tokens=None,
            cache_read_tokens=None, cache_create_tokens=None,
            duration_ms=100, capped=False,
            started_at=_iso(timedelta(hours=-1)),
            completed_at=_iso(timedelta(hours=-1)),
        )
        await store.save_invocation(
            run_id='r2', task_id='t2', project_id='dark_factory',
            account_name='acct', model='sonnet', role='escalation-watcher-auto',
            cost_usd=3.0, input_tokens=None, output_tokens=None,
            cache_read_tokens=None, cache_create_tokens=None,
            duration_ms=100, capped=False,
            started_at=_iso(timedelta(hours=-2)),
            completed_at=_iso(timedelta(hours=-2)),
        )
        # Outside window (25h ago — must be excluded)
        await store.save_invocation(
            run_id='r3', task_id='t3', project_id='dark_factory',
            account_name='acct', model='sonnet', role='steward',
            cost_usd=100.0, input_tokens=None, output_tokens=None,
            cache_read_tokens=None, cache_create_tokens=None,
            duration_ms=100, capped=False,
            started_at=_iso(timedelta(hours=-25)),
            completed_at=_iso(timedelta(hours=-25)),
        )

        result = await harness._trailing_24h_cost_usd(watcher_only=False)
        assert result == pytest.approx(8.0), (
            f'Expected 5.0 + 3.0 = 8.0 for rows within 24h; got {result!r}'
        )

    @pytest.mark.asyncio
    async def test_trailing_24h_cost_usd_no_cost_store(self, tmp_path: Path) -> None:
        """_trailing_24h_cost_usd returns 0.0 when cost_store is None."""
        harness, _, _ = _make_harness_with_mocks(tmp_path)
        harness.cost_store = None

        result = await harness._trailing_24h_cost_usd(watcher_only=False)
        assert result == 0.0, f'Expected 0.0 with no cost_store; got {result!r}'

    @pytest.mark.asyncio
    async def test_trailing_24h_cost_usd_empty_table(self, tmp_path: Path) -> None:
        """_trailing_24h_cost_usd returns 0.0 when no rows exist."""
        harness, _, _ = _make_harness_with_mocks(tmp_path)
        store = await _make_cost_store(tmp_path)
        harness.cost_store = store

        result = await harness._trailing_24h_cost_usd(watcher_only=False)
        assert result == 0.0, f'Expected 0.0 for empty table; got {result!r}'

    # ------------------------------------------------------------------
    # Step 5 — _trailing_24h_cost_usd(watcher_only=True) role filter
    # ------------------------------------------------------------------

    @pytest.mark.asyncio
    async def test_trailing_24h_cost_usd_watcher_only_filters_role(
        self, tmp_path: Path
    ) -> None:
        """watcher_only=True sums only rows with role matching '%watcher%'.

        Seeds:
          - role='escalation-watcher-auto'  cost=7.0  (within window) — INCLUDED
          - role='terminal-status-watcher'  cost=3.0  (within window) — INCLUDED
          - role='orchestrator'             cost=20.0 (within window) — EXCLUDED
          - role='steward'                  cost=15.0 (within window) — EXCLUDED
        Expects watcher_only=True → 10.0; watcher_only=False → 45.0.
        """
        harness, _, _ = _make_harness_with_mocks(tmp_path)
        store = await _make_cost_store(tmp_path)
        harness.cost_store = store

        rows = [
            ('escalation-watcher-auto', 7.0),
            ('terminal-status-watcher', 3.0),
            ('orchestrator', 20.0),
            ('steward', 15.0),
        ]
        for i, (role, cost) in enumerate(rows):
            await store.save_invocation(
                run_id=f'r{i}', task_id=f't{i}', project_id='dark_factory',
                account_name='acct', model='sonnet', role=role,
                cost_usd=cost, input_tokens=None, output_tokens=None,
                cache_read_tokens=None, cache_create_tokens=None,
                duration_ms=100, capped=False,
                started_at=_iso(timedelta(hours=-(i + 1))),
                completed_at=_iso(timedelta(hours=-(i + 1))),
            )

        watcher_sum = await harness._trailing_24h_cost_usd(watcher_only=True)
        assert watcher_sum == pytest.approx(10.0), (
            f'Expected 7.0+3.0=10.0 for watcher_only=True; got {watcher_sum!r}'
        )

        total_sum = await harness._trailing_24h_cost_usd(watcher_only=False)
        assert total_sum == pytest.approx(45.0), (
            f'Expected 45.0 for watcher_only=False; got {total_sum!r}'
        )

    # ------------------------------------------------------------------
    # Step 7 — _enforce_cost_ceilings: watcher ceiling trip
    # ------------------------------------------------------------------

    @pytest.mark.asyncio
    async def test_enforce_cost_ceilings_watcher_trip(self, tmp_path: Path) -> None:
        """Watcher cost exceeding watcher_daily_cost_ceiling_usd pauses the scheduler.

        Watcher sum > watcher ceiling; total still under orch ceiling.
        Expects: is_paused=True, reason='cost_ceiling_watcher_exceeded',
        RunStore.save_scheduler_pause called with that reason.
        """
        config = OrchestratorConfig(
            project_root=tmp_path,
            watcher_daily_cost_ceiling_usd=10.0,   # will be exceeded
            orch_daily_cost_ceiling_usd=1000.0,    # will NOT be exceeded
        )
        harness = Harness(config)
        mock_run_store = MagicMock(spec=RunStore)
        harness._run_store = mock_run_store
        harness._run_id = 'run-test-ceil-0001'
        store = await _make_cost_store(tmp_path)
        harness.cost_store = store

        # Seed watcher rows summing to 11.0 (> ceiling of 10.0)
        for i in range(2):
            await store.save_invocation(
                run_id=f'r{i}', task_id=f't{i}', project_id='dark_factory',
                account_name='acct', model='opus', role='escalation-watcher-auto',
                cost_usd=5.5, input_tokens=None, output_tokens=None,
                cache_read_tokens=None, cache_create_tokens=None,
                duration_ms=100, capped=False,
                started_at=_iso(timedelta(hours=-(i + 1))),
                completed_at=_iso(timedelta(hours=-(i + 1))),
            )

        await harness._enforce_cost_ceilings()

        assert harness.scheduler.is_paused is True, (
            'Scheduler must be paused when watcher cost exceeds ceiling'
        )
        assert harness.scheduler.pause_reason == 'cost_ceiling_watcher_exceeded', (
            f'Expected cost_ceiling_watcher_exceeded; got {harness.scheduler.pause_reason!r}'
        )
        mock_run_store.save_scheduler_pause.assert_called_once()
        call_args = mock_run_store.save_scheduler_pause.call_args
        kwargs = {**call_args.kwargs}
        args = call_args.args or ()
        all_args = dict(zip(
            ['project_id', 'reason', 'pause_at_iso', 'set_by_run_id'],
            args, strict=False,
        ))
        all_args.update(kwargs)
        assert all_args.get('reason') == 'cost_ceiling_watcher_exceeded', (
            f'Expected reason=cost_ceiling_watcher_exceeded in RunStore call; got {all_args!r}'
        )

    # ------------------------------------------------------------------
    # Step 9 — _enforce_cost_ceilings: orch-wide, ordering, idempotency
    # ------------------------------------------------------------------

    @pytest.mark.asyncio
    async def test_enforce_cost_ceilings_orch_wide_trip(self, tmp_path: Path) -> None:
        """Orch-wide cost exceeding orch ceiling pauses with 'cost_ceiling_orch_exceeded'.

        Watcher cost is under its ceiling; total cost exceeds orch ceiling.
        """
        config = OrchestratorConfig(
            project_root=tmp_path,
            watcher_daily_cost_ceiling_usd=1000.0,  # NOT exceeded
            orch_daily_cost_ceiling_usd=20.0,       # WILL be exceeded
        )
        harness = Harness(config)
        mock_run_store = MagicMock(spec=RunStore)
        harness._run_store = mock_run_store
        harness._run_id = 'run-test-orch-0001'
        store = await _make_cost_store(tmp_path)
        harness.cost_store = store

        # Seed a non-watcher row summing to 25.0 > orch ceiling 20.0
        await store.save_invocation(
            run_id='r1', task_id='t1', project_id='dark_factory',
            account_name='acct', model='sonnet', role='orchestrator',
            cost_usd=25.0, input_tokens=None, output_tokens=None,
            cache_read_tokens=None, cache_create_tokens=None,
            duration_ms=100, capped=False,
            started_at=_iso(timedelta(hours=-1)),
            completed_at=_iso(timedelta(hours=-1)),
        )

        await harness._enforce_cost_ceilings()

        assert harness.scheduler.is_paused is True
        assert harness.scheduler.pause_reason == 'cost_ceiling_orch_exceeded', (
            f'Expected cost_ceiling_orch_exceeded; got {harness.scheduler.pause_reason!r}'
        )

    @pytest.mark.asyncio
    async def test_enforce_cost_ceilings_both_under_no_pause(self, tmp_path: Path) -> None:
        """When both ceilings are NOT exceeded, scheduler is not paused."""
        config = OrchestratorConfig(
            project_root=tmp_path,
            watcher_daily_cost_ceiling_usd=1000.0,
            orch_daily_cost_ceiling_usd=1000.0,
        )
        harness = Harness(config)
        mock_run_store = MagicMock(spec=RunStore)
        harness._run_store = mock_run_store
        harness._run_id = 'run-test-under-0001'
        store = await _make_cost_store(tmp_path)
        harness.cost_store = store

        await store.save_invocation(
            run_id='r1', task_id='t1', project_id='dark_factory',
            account_name='acct', model='sonnet', role='orchestrator',
            cost_usd=1.0, input_tokens=None, output_tokens=None,
            cache_read_tokens=None, cache_create_tokens=None,
            duration_ms=100, capped=False,
            started_at=_iso(timedelta(hours=-1)),
            completed_at=_iso(timedelta(hours=-1)),
        )

        await harness._enforce_cost_ceilings()

        assert harness.scheduler.is_paused is False, (
            'Scheduler must NOT be paused when both costs are under ceilings'
        )
        mock_run_store.save_scheduler_pause.assert_not_called()

        # acquire_next should return None only if paused; here it should proceed
        # (returns None because no tasks, but NOT because paused).
        assert not harness.scheduler.is_paused

    @pytest.mark.asyncio
    async def test_enforce_cost_ceilings_both_exceed_watcher_reason_wins(
        self, tmp_path: Path
    ) -> None:
        """When BOTH ceilings would be exceeded, watcher reason wins (checked first)."""
        config = OrchestratorConfig(
            project_root=tmp_path,
            watcher_daily_cost_ceiling_usd=5.0,   # WILL be exceeded
            orch_daily_cost_ceiling_usd=10.0,     # also exceeded by total
        )
        harness = Harness(config)
        mock_run_store = MagicMock(spec=RunStore)
        harness._run_store = mock_run_store
        harness._run_id = 'run-test-both-0001'
        store = await _make_cost_store(tmp_path)
        harness.cost_store = store

        # Watcher row: 8.0 > watcher ceiling 5.0 AND > orch ceiling 10? No — need
        # total to also exceed orch ceiling. Seed both.
        await store.save_invocation(
            run_id='r1', task_id='t1', project_id='dark_factory',
            account_name='acct', model='opus', role='escalation-watcher-auto',
            cost_usd=8.0, input_tokens=None, output_tokens=None,  # > watcher ceil 5.0
            cache_read_tokens=None, cache_create_tokens=None,
            duration_ms=100, capped=False,
            started_at=_iso(timedelta(hours=-1)),
            completed_at=_iso(timedelta(hours=-1)),
        )
        await store.save_invocation(
            run_id='r2', task_id='t2', project_id='dark_factory',
            account_name='acct', model='sonnet', role='orchestrator',
            cost_usd=5.0, input_tokens=None, output_tokens=None,
            cache_read_tokens=None, cache_create_tokens=None,
            duration_ms=100, capped=False,
            started_at=_iso(timedelta(hours=-2)),
            completed_at=_iso(timedelta(hours=-2)),
        )
        # Total = 13.0 > orch ceiling 10.0; watcher = 8.0 > watcher ceiling 5.0

        await harness._enforce_cost_ceilings()

        assert harness.scheduler.pause_reason == 'cost_ceiling_watcher_exceeded', (
            f'Watcher reason must win when both ceilings trip; '
            f'got {harness.scheduler.pause_reason!r}'
        )
        # save_scheduler_pause called exactly once (watcher branch returns early)
        mock_run_store.save_scheduler_pause.assert_called_once()

    @pytest.mark.asyncio
    async def test_enforce_cost_ceilings_already_paused_no_extra_call(
        self, tmp_path: Path
    ) -> None:
        """When scheduler is already paused, _enforce_cost_ceilings() returns immediately.

        No additional RunStore.save_scheduler_pause call is made.
        """
        harness, mock_run_store, _ = _make_harness_with_mocks(tmp_path)
        store = await _make_cost_store(tmp_path)
        harness.cost_store = store

        # Pre-pause the scheduler.
        await harness.pause_scheduler('pre-existing-pause')
        mock_run_store.reset_mock()  # clear prior call count

        # Even with watcher rows above threshold, no new pause call should be made.
        await store.save_invocation(
            run_id='r1', task_id='t1', project_id='dark_factory',
            account_name='acct', model='opus', role='escalation-watcher-auto',
            cost_usd=999.0, input_tokens=None, output_tokens=None,
            cache_read_tokens=None, cache_create_tokens=None,
            duration_ms=100, capped=False,
            started_at=_iso(timedelta(hours=-1)),
            completed_at=_iso(timedelta(hours=-1)),
        )

        await harness._enforce_cost_ceilings()

        mock_run_store.save_scheduler_pause.assert_not_called(), (
            'Already-paused scheduler: _enforce_cost_ceilings must NOT call '
            'save_scheduler_pause again'
        )
        # Original pause reason must be preserved
        assert harness.scheduler.pause_reason == 'pre-existing-pause'
