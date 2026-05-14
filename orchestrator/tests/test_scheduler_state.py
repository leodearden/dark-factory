"""Tests for scheduler state snapshot, new event types, and related machinery.

Split from test_scheduler.py to keep that file's 4000+ lines from growing
further and to group all state-snapshot / reserve-now / park-tracking tests
together for future maintenance.
"""
from __future__ import annotations

import json
from datetime import UTC, datetime
from pathlib import Path
from unittest.mock import AsyncMock

import pytest

from orchestrator.config import OrchestratorConfig
from orchestrator.event_store import EventType
from orchestrator.overrides import OverrideStore
from orchestrator.scheduler import ModuleLockTable, Scheduler


# ---------------------------------------------------------------------------
# Reuse the _RecordingEventStore test double from test_scheduler.py
# (duplicated here to avoid cross-module import; small enough to justify).
# ---------------------------------------------------------------------------

class _RecordingEventStore:
    """Minimal EventStore stand-in capturing emit() calls in-memory."""

    def __init__(self):
        self.events: list[tuple[str, dict]] = []

    def emit(self, event_type, *, task_id=None, phase=None, role=None,
             data=None, cost_usd=None, duration_ms=None):
        self.events.append((
            str(event_type),
            {
                'task_id': task_id,
                'data': dict(data or {}),
            },
        ))


def _pending_task(task_id: str, *, priority: str = 'medium',
                  files: list[str] | None = None,
                  deps: list[str] | None = None) -> dict:
    return {
        'id': task_id,
        'title': f'Task {task_id}',
        'status': 'pending',
        'dependencies': deps or [],
        'metadata': {'files': files or [f'{task_id}/src']},
        'priority': priority,
    }


# ===========================================================================
# Step-1: EventType new members
# ===========================================================================

class TestEventTypeAdditions:
    """Verify the two new EventType enum members added in step-2."""

    def test_reserve_now_armed_exists(self):
        assert EventType.reserve_now_armed == 'reserve_now_armed'

    def test_reserve_now_consumed_exists(self):
        assert EventType.reserve_now_consumed == 'reserve_now_consumed'


# ===========================================================================
# Step-3: reserve_now_armed diff-emit
# ===========================================================================

class TestReserveNowArmedDiffEmit:
    """_emit_override_diff_events fires reserve_now_armed on False→True transition."""

    @pytest.mark.asyncio
    async def test_reserve_now_armed_emitted_on_false_to_true_transition(self, tmp_path):
        """reserve_now_armed fires when reserve_now transitions False→True between ticks.

        Tick 1: no override → no event (seeds _overrides_initialized).
        Tick 2: set reserve_now=True on X → exactly one reserve_now_armed with task_id='X'.
        """
        from orchestrator.overrides import OverrideStore

        config = OrchestratorConfig(max_per_module=1)
        event_store = _RecordingEventStore()
        store = OverrideStore(tmp_path / 'o.db')

        scheduler = Scheduler(config, override_store=store, event_store=event_store)  # type: ignore[arg-type]
        scheduler._project_root = '/proj'

        task_x = _pending_task('X', files=['mod_x'])
        scheduler.get_tasks = AsyncMock(return_value=[task_x])

        # Tick 1 — seeds _overrides_initialized; no reserve_now yet.
        await scheduler.acquire_next()
        scheduler.release('X')
        armed_after_tick1 = [
            e for e in event_store.events
            if e[0] == EventType.reserve_now_armed.value
        ]
        assert armed_after_tick1 == [], f'No armed events expected after tick 1, got: {armed_after_tick1}'

        # Tick 2 — set reserve_now=True → must emit reserve_now_armed.
        store.set_override('/proj', 'X', reserve_now=True)
        await scheduler.acquire_next()
        armed_after_tick2 = [
            e for e in event_store.events
            if e[0] == EventType.reserve_now_armed.value
        ]
        assert len(armed_after_tick2) == 1, f'Expected 1 armed event, got: {armed_after_tick2}'
        assert armed_after_tick2[0][1]['task_id'] == 'X'

    @pytest.mark.asyncio
    async def test_boost_only_change_does_not_emit_reserve_now_armed(self, tmp_path):
        """A boost_tier change with no reserve_now transition must not emit reserve_now_armed."""
        from orchestrator.overrides import OverrideStore

        config = OrchestratorConfig(max_per_module=1)
        event_store = _RecordingEventStore()
        store = OverrideStore(tmp_path / 'o.db')

        scheduler = Scheduler(config, override_store=store, event_store=event_store)  # type: ignore[arg-type]
        scheduler._project_root = '/proj'

        task_x = _pending_task('X', files=['mod_x'])
        scheduler.get_tasks = AsyncMock(return_value=[task_x])

        # Tick 1 seeds initialized state.
        await scheduler.acquire_next()
        scheduler.release('X')

        # Tick 2: boost only — no reserve_now change.
        store.set_override('/proj', 'X', boost_tier='high')
        await scheduler.acquire_next()
        armed = [e for e in event_store.events if e[0] == EventType.reserve_now_armed.value]
        assert armed == [], f'Expected no armed events for boost-only change, got: {armed}'


# ===========================================================================
# Step-5: reserve_now_consumed short-circuit emit
# ===========================================================================

class TestReserveNowConsumedShortCircuit:
    """reserve_now_consumed is emitted when parks are installed via the short-circuit."""

    @pytest.mark.asyncio
    async def test_reserve_now_consumed_emitted_when_parks_install(self, tmp_path):
        """reserve_now_consumed fires when parks are installed for reserve_now=True task.

        Mirrors TestReserveNowShortCircuit setup: A's modules are pre-held so
        parks survive the tick.  Asserts:
        (a) exactly one reserve_now_consumed event with task_id='A',
            data['modules'] covering A's modules, data['priority'] = A's tier;
        (b) NO reservation_installed event with data.get('reason') == 'reserve_now'
            (semantic upgrade — old event replaced).
        """
        config = OrchestratorConfig(max_per_module=1)
        store = OverrideStore(tmp_path / 'o.db')
        store.set_override('/proj', 'A', reserve_now=True)

        event_store = _RecordingEventStore()
        scheduler = Scheduler(config, override_store=store, event_store=event_store)  # type: ignore[arg-type]
        scheduler._project_root = '/proj'

        # Pre-hold A's modules so parks survive the tick (A cannot acquire them).
        scheduler.lock_table._held['seed'] = {'compiler/src', 'eval/src'}
        scheduler._dispatched.add('seed')

        task_a = _pending_task('A', files=['compiler/src', 'eval/src'], priority='medium')
        task_b = _pending_task('B', files=['other/module'])
        scheduler.get_tasks = AsyncMock(return_value=[task_a, task_b])

        await scheduler.acquire_next()

        consumed = [
            e for e in event_store.events
            if e[0] == EventType.reserve_now_consumed.value
        ]
        # (a) Exactly one consumed event with correct fields.
        assert len(consumed) == 1, f'Expected 1 consumed event, got: {consumed}'
        ev = consumed[0]
        assert ev[1]['task_id'] == 'A'
        assert set(ev[1]['data']['modules']) == {
            'compiler/src', 'eval/src',
        }, f"Expected A's modules in consumed event, got: {ev[1]['data']['modules']}"
        assert ev[1]['data']['priority'] == 'medium', (
            f"Expected priority='medium', got: {ev[1]['data']['priority']}"
        )

        # (b) No legacy reservation_installed with reason='reserve_now'.
        old_style = [
            e for e in event_store.events
            if e[0] == EventType.reservation_installed.value
            and e[1]['data'].get('reason') == 'reserve_now'
        ]
        assert old_style == [], f'Expected no legacy reservation_installed event, got: {old_style}'


# ===========================================================================
# Step-7: _park_install_at tracking in ModuleLockTable
# ===========================================================================

class TestParkInstallAtTracking:
    """ModuleLockTable._park_install_at records ISO8601 install time per owner."""

    def _make_lock_table(self) -> ModuleLockTable:
        return ModuleLockTable(OrchestratorConfig(max_per_module=1))

    def test_install_parks_records_install_at_for_owner(self):
        lt = self._make_lock_table()
        installed, _ = lt.install_parks('T', ['mod/a'], 'medium')
        assert installed, 'Expected at least one module to be installed'
        assert 'T' in lt._park_install_at
        # Must be a parseable ISO8601 string.
        datetime.fromisoformat(lt._park_install_at['T'])

    def test_repeat_install_preserves_first_install_at(self):
        lt = self._make_lock_table()
        lt.install_parks('T', ['mod/a'], 'medium')
        first_ts = lt._park_install_at['T']
        # Install additional modules for the same owner.
        lt.install_parks('T', ['mod/b'], 'medium')
        assert lt._park_install_at['T'] == first_ts, (
            'Second install_parks must not overwrite the first install timestamp'
        )

    def test_clear_parks_for_drops_install_at(self):
        lt = self._make_lock_table()
        lt.install_parks('T', ['mod/a'], 'medium')
        assert 'T' in lt._park_install_at
        lt.clear_parks_for('T')
        assert 'T' not in lt._park_install_at

    def test_prune_owners_drops_install_at(self):
        lt = self._make_lock_table()
        lt.install_parks('T', ['mod/a'], 'medium')
        assert 'T' in lt._park_install_at
        evicted = lt.prune_owners(lambda owner: owner == 'T')
        assert 'T' in evicted
        assert 'T' not in lt._park_install_at

    def test_no_install_at_when_nothing_installed(self):
        lt = self._make_lock_table()
        # Install a conflicting higher-priority park so T's install is blocked.
        lt.install_parks('blocker', ['mod/a'], 'critical')
        lt.install_parks('T', ['mod/a'], 'medium')
        assert 'T' not in lt._park_install_at, (
            '_park_install_at must not be set when install_parks installs nothing'
        )
