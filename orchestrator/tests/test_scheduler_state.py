"""Tests for scheduler state snapshot, new event types, and related machinery.

Split from test_scheduler.py to keep that file's 4000+ lines from growing
further and to group all state-snapshot / reserve-now / park-tracking tests
together for future maintenance.
"""
from __future__ import annotations

import json
from datetime import datetime
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

    def test_park_install_at_bounded_under_preemption_churn(self):
        """_park_install_at must not accumulate stale entries under repeated evictions."""
        lt = self._make_lock_table()
        for i in range(50):
            lt.install_parks(f'low{i}', ['mod/a'], 'medium')
            lt.install_parks('high', ['mod/a'], 'critical')
            lt.clear_parks_for('high')  # free mod/a for next iteration
        # Every low{i} was fully evicted; 'high' was cleared each round.
        assert len(lt._park_install_at) == 0, (
            f'Expected empty _park_install_at after all evictions+clears, '
            f'got {len(lt._park_install_at)} entries — without the fix this would be ~50'
        )

    def test_install_parks_drops_install_at_for_fully_evicted_owner(self):
        """Full eviction must remove the owner from _park_install_at."""
        lt = self._make_lock_table()
        # (1) T1 parks mod/a at medium priority.
        installed, _ = lt.install_parks('T1', ['mod/a'], 'medium')
        assert installed, 'T1 should have installed mod/a'
        old_ts = lt._park_install_at['T1']
        assert 'T1' in lt._park_install_at

        # (2) T2 at critical priority evicts T1 fully.
        _, evicted = lt.install_parks('T2', ['mod/a'], 'critical')
        assert evicted == [('T1', ['mod/a'])], f'Expected T1 evicted, got: {evicted}'
        assert not lt.has_parks('T1'), 'T1 should have no parks after full eviction'

        # (3) Core regression: stale entry must be gone.
        assert 'T1' not in lt._park_install_at, (
            '_park_install_at must drop fully-evicted owner T1'
        )

        # (4) Re-install: fresh timestamp must be recorded.
        lt.clear_parks_for('T2')  # free mod/a
        lt.install_parks('T1', ['mod/a'], 'medium')
        assert 'T1' in lt._park_install_at, 'T1 must have a new entry after re-install'
        new_ts = lt._park_install_at['T1']
        datetime.fromisoformat(new_ts)  # must be valid ISO8601
        assert new_ts >= old_ts, (
            'Re-installed timestamp must be >= original (ISO8601 strings sort chronologically)'
        )


# ===========================================================================
# Step-9: _last_effective_priorities cache
# ===========================================================================

class TestEffectivePrioritiesCache:
    """Scheduler._last_effective_priorities is populated after acquire_next."""

    def test_empty_before_first_tick(self):
        scheduler = Scheduler(OrchestratorConfig(max_per_module=1))
        assert scheduler._last_effective_priorities == {}

    @pytest.mark.asyncio
    async def test_acquire_next_caches_effective_priorities(self):
        """After a tick, _last_effective_priorities maps task_id -> effective tier.

        B (high priority) depends on A (medium).  The reverse index has A as a
        dependency of B, so A's effective priority is upgraded to high via
        priority inheritance (effective = min-rank over dependents).
        """
        config = OrchestratorConfig(max_per_module=1)
        scheduler = Scheduler(config)

        task_a = _pending_task('A', priority='medium', files=['mod_a'])
        task_b = {
            'id': 'B',
            'title': 'Task B',
            'status': 'pending',
            'dependencies': [{'id': 'A'}],
            'metadata': {'files': ['mod_b']},
            'priority': 'high',
        }
        scheduler.get_tasks = AsyncMock(return_value=[task_a, task_b])

        await scheduler.acquire_next()

        assert 'A' in scheduler._last_effective_priorities
        assert 'B' in scheduler._last_effective_priorities
        # A's effective priority inherits high from B (B is A's dependent).
        assert scheduler._last_effective_priorities['A'] == 'high', (
            'A is depended upon by high-priority B; effective priority must be high'
        )
        assert scheduler._last_effective_priorities['B'] == 'high'


# ===========================================================================
# Step-11: get_state_snapshot() shape and deep-copy
# ===========================================================================

_SNAPSHOT_KEYS = frozenset({
    'skip_counts', 'parks', 'effective_priorities',
    'pin_queue', 'overrides', 'current_holders', 'snapshot_at',
})


class TestGetStateSnapshotShape:
    """get_state_snapshot() returns the correct seven-key dict."""

    def test_snapshot_returns_seven_top_level_keys(self):
        scheduler = Scheduler(OrchestratorConfig(max_per_module=1))
        snap = scheduler.get_state_snapshot()
        assert set(snap.keys()) == _SNAPSHOT_KEYS

    def test_snapshot_empty_scheduler_returns_empty_collections(self):
        scheduler = Scheduler(OrchestratorConfig(max_per_module=1))
        snap = scheduler.get_state_snapshot()
        assert snap['skip_counts'] == {}
        assert snap['parks'] == {}
        assert snap['effective_priorities'] == {}
        assert snap['pin_queue'] == []
        assert snap['overrides'] == {}
        assert snap['current_holders'] == {}
        # snapshot_at must be an ISO8601 string.
        datetime.fromisoformat(snap['snapshot_at'])

    def test_snapshot_is_deep_copy_of_internal_state(self):
        """Mutating the returned snapshot must not affect the scheduler's internal state."""
        scheduler = Scheduler(OrchestratorConfig(max_per_module=1))
        scheduler._skip_count['X'] = 3
        snap = scheduler.get_state_snapshot()
        snap['skip_counts']['X'] = 99
        assert scheduler._skip_count['X'] == 3, (
            'Mutating snapshot must not affect internal _skip_count'
        )


# ===========================================================================
# Step-13: get_state_snapshot() with populated state after real ticks
# ===========================================================================

class TestGetStateSnapshotPopulated:
    """get_state_snapshot() reflects real scheduler state after ticks."""

    @pytest.mark.asyncio
    async def test_snapshot_after_real_tick_with_skips_parks_overrides_holders(
        self, tmp_path
    ):
        """Full integration: snapshot reflects real data after two acquire_next ticks.

        Setup:
        - T1 (medium): boost_tier='high' override → effective priority 'high'
        - T2 (medium): pinned=True → pin_queue entry with order 1
        - T3 (critical): pre-hold T3's modules → accumulates skip then park

        Tick 1: T2 dispatches (pinned, bypasses scored loop).
        Tick 2: T3 is top scored (critical > high), can't acquire (modules
                pre-held by 'holder').  T1 (high) dispatches instead.
                T3 skip counter bumped; skip_threshold(critical)=0 so parks
                installed after 1 skip.

        Assertions mirror the step-13 spec exactly.
        """
        config = OrchestratorConfig(max_per_module=1)
        config.fairness.scheduler_v2 = True
        store = OverrideStore(tmp_path / 'o.db')

        # T1 gets a boost override (medium → high).
        store.set_override('/proj', 'T1', boost_tier='high')
        # T2 gets pinned (auto-assigns pin_order=1).
        store.set_override('/proj', 'T2', pinned=True)

        event_store = _RecordingEventStore()
        scheduler = Scheduler(
            config, override_store=store, event_store=event_store  # type: ignore[arg-type]
        )
        scheduler._project_root = '/proj'

        # Pre-hold T3's modules so T3 can never acquire them.
        scheduler.lock_table._held['holder'] = {'t3/src'}
        scheduler._dispatched.add('holder')

        # Three tasks with distinct modules so they don't conflict each other.
        task_t1 = _pending_task('T1', priority='medium', files=['t1/src'])
        task_t2 = _pending_task('T2', priority='medium', files=['t2/src'])
        task_t3 = _pending_task('T3', priority='critical', files=['t3/src'])
        scheduler.get_tasks = AsyncMock(return_value=[task_t1, task_t2, task_t3])

        # Tick 1: T2 dispatches (pinned — bypasses the scored candidate loop).
        await scheduler.acquire_next()

        # Tick 2: T3 is top scored (critical), modules pre-held → skip+park.
        #          T1 (high boosted) acquires its free modules.
        await scheduler.acquire_next()

        snap = scheduler.get_state_snapshot()

        # --- skip_counts ---
        assert snap['skip_counts'].get('T3', 0) >= 1, (
            f"Expected T3 skip_count >= 1, got snap['skip_counts']={snap['skip_counts']}"
        )

        # --- parks ---
        assert 'T3' in snap['parks'], f"Expected T3 in parks, got {snap['parks']}"
        assert snap['parks']['T3']['modules'], (
            'Expected non-empty modules list in parks[T3]'
        )
        datetime.fromisoformat(snap['parks']['T3']['installed_at'])

        # --- effective_priorities ---
        assert snap['effective_priorities'].get('T1') == 'high', (
            f"Expected effective_priorities['T1']='high', "
            f"got {snap['effective_priorities']}"
        )

        # --- pin_queue ---
        assert snap['pin_queue'] == [{'task_id': 'T2', 'order': 1}], (
            f"Expected pin_queue=[{{task_id:'T2',order:1}}], got {snap['pin_queue']}"
        )

        # --- overrides ---
        assert 'T1' in snap['overrides'], f"Expected T1 in overrides, got {snap['overrides']}"
        ov_t1 = snap['overrides']['T1']
        assert ov_t1['boost_tier'] == 'high', f"Expected boost_tier='high', got {ov_t1}"
        assert ov_t1['pinned'] is False, f"Expected pinned=False, got {ov_t1}"
        assert ov_t1['ttl_until'] is None, f"Expected ttl_until=None, got {ov_t1}"

        # --- current_holders ---
        ch = snap['current_holders']
        assert ch, 'Expected non-empty current_holders'
        # Every held module must map to its task_id.
        assert ch.get('t2/src') == 'T2', (
            f"Expected current_holders['t2/src']='T2', got {ch}"
        )
        assert ch.get('t3/src') == 'holder', (
            f"Expected current_holders['t3/src']='holder', got {ch}"
        )
        assert ch.get('t1/src') == 'T1', (
            f"Expected current_holders['t1/src']='T1', got {ch}"
        )


# ===========================================================================
# Step-15: write_state_snapshot()
# ===========================================================================

class TestWriteStateSnapshot:
    """Scheduler.write_state_snapshot() writes a valid JSON file atomically."""

    def test_write_state_snapshot_creates_valid_json(self, tmp_path):
        """write_state_snapshot writes the snapshot to disk as parseable JSON."""
        scheduler = Scheduler(OrchestratorConfig(max_per_module=1))
        path = tmp_path / 'scheduler_state.json'
        scheduler.write_state_snapshot(path)

        assert path.exists(), 'Expected snapshot file to be created'
        data = json.loads(path.read_text())
        assert set(data.keys()) == _SNAPSHOT_KEYS, (
            f'Expected keys {_SNAPSHOT_KEYS}, got {set(data.keys())}'
        )

    def test_write_state_snapshot_atomic_replace(self, tmp_path):
        """Second write overwrites the first; no partial writes visible."""
        scheduler = Scheduler(OrchestratorConfig(max_per_module=1))
        path = tmp_path / 'scheduler_state.json'

        # First write: inject a sentinel skip count.
        scheduler._skip_count['first'] = 1
        scheduler.write_state_snapshot(path)
        # Second write: clear skip_count and add a different sentinel.
        scheduler._skip_count.clear()
        scheduler._skip_count['second'] = 2
        scheduler.write_state_snapshot(path)
        second_content = json.loads(path.read_text())

        assert 'first' not in second_content['skip_counts'], (
            'Second write must have overwritten the first completely'
        )
        assert second_content['skip_counts'].get('second') == 2

    def test_write_state_snapshot_creates_parent_dirs(self, tmp_path):
        """write_state_snapshot creates missing parent directories."""
        scheduler = Scheduler(OrchestratorConfig(max_per_module=1))
        path = tmp_path / 'deep' / 'nested' / 'dir' / 'scheduler_state.json'
        scheduler.write_state_snapshot(path)
        assert path.exists(), (
            'Expected snapshot file to be created even when parent dirs are missing'
        )


# ===========================================================================
# Step-17: acquire_next writes snapshot to default path
# ===========================================================================

class TestAcquireNextWritesSnapshot:
    """acquire_next() writes a scheduler_state.json after each tick."""

    @pytest.mark.asyncio
    async def test_acquire_next_writes_snapshot_to_default_path(self, tmp_path):
        """After acquire_next, scheduler_state.json exists and parses as JSON."""
        config = OrchestratorConfig(max_per_module=1)
        scheduler = Scheduler(config)
        scheduler._project_root = str(tmp_path)

        task_a = _pending_task('A', files=['mod_a'])
        scheduler.get_tasks = AsyncMock(return_value=[task_a])

        await scheduler.acquire_next()

        snap_path = tmp_path / 'data' / 'orchestrator' / 'scheduler_state.json'
        assert snap_path.exists(), (
            f'Expected snapshot at {snap_path} after acquire_next'
        )
        data = json.loads(snap_path.read_text())
        assert 'snapshot_at' in data

    @pytest.mark.asyncio
    async def test_snapshot_write_failure_does_not_break_acquire_next(
        self, tmp_path, monkeypatch
    ):
        """If write_state_snapshot raises, acquire_next still dispatches tasks."""
        config = OrchestratorConfig(max_per_module=1)
        scheduler = Scheduler(config)
        scheduler._project_root = str(tmp_path)

        task_a = _pending_task('A', files=['mod_a'])
        scheduler.get_tasks = AsyncMock(return_value=[task_a])

        # Force write_state_snapshot to raise so we test fault isolation.
        from unittest.mock import MagicMock
        scheduler.write_state_snapshot = MagicMock(side_effect=RuntimeError('disk full'))

        result = await scheduler.acquire_next()
        # The dispatch must still succeed despite the snapshot write failure.
        assert result is not None, (
            'acquire_next must return a TaskAssignment even if snapshot write fails'
        )
        assert result.task_id == 'A'
