"""Tests for scheduler state snapshot, new event types, and related machinery.

Split from test_scheduler.py to keep that file's 4000+ lines from growing
further and to group all state-snapshot / reserve-now / park-tracking tests
together for future maintenance.
"""
from __future__ import annotations

import json
from datetime import datetime
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

    def test_partial_eviction_preserves_install_at(self):
        """Partial eviction (owner retains another park) must keep its _park_install_at entry."""
        lt = self._make_lock_table()
        # (1) T1 parks two independent modules at medium priority.
        installed, _ = lt.install_parks('T1', ['mod/a', 'mod/b'], 'medium')
        assert len(installed) == 2, f'Expected 2 installs, got {installed}'
        ts = lt._park_install_at['T1']

        # (2) T2 at critical priority evicts ONLY mod/a; T1 retains mod/b.
        _, evicted = lt.install_parks('T2', ['mod/a'], 'critical')
        assert evicted == [('T1', ['mod/a'])], f'Expected T1 evicted from mod/a only, got: {evicted}'
        assert lt.has_parks('T1'), 'T1 still owns mod/b — has_parks must return True'

        # (3) Core assertion: original timestamp must survive the partial eviction.
        assert 'T1' in lt._park_install_at, (
            'T1 still has parks (mod/b) — its _park_install_at entry must not be dropped'
        )
        assert lt._park_install_at['T1'] == ts, (
            'Partial eviction must preserve the original installed_at timestamp'
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
        assert datetime.fromisoformat(new_ts) >= datetime.fromisoformat(old_ts), (
            'Re-installed timestamp must be >= original'
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
    'pin_queue', 'overrides', 'current_holders', 'lock_depth', 'snapshot_at',
    'is_paused', 'pause_reason',
})


class TestGetStateSnapshotShape:
    """get_state_snapshot() returns the correct ten-key dict."""

    def test_snapshot_returns_ten_top_level_keys(self):
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
        # Pause state defaults — scheduler is not paused at construction.
        assert snap['is_paused'] is False
        assert snap['pause_reason'] is None

    def test_snapshot_exposes_lock_depth(self):
        # lock_depth lets the dashboard normalize footprints the same way the
        # scheduler does before matching against current_holders.
        scheduler = Scheduler(OrchestratorConfig(max_per_module=1, lock_depth=3))
        snap = scheduler.get_state_snapshot()
        assert snap['lock_depth'] == 3

    def test_bare_config_project_root_isolated_to_tmp(self, tmp_path):
        """Bare OrchestratorConfig() must NOT resolve project_root to the real repo.

        Regression guard for the snapshot-pollution bug: with ORCH_CONFIG_PATH
        unset, the settings source falls back to the relative ``config.yaml``,
        so ``cd orchestrator && pytest`` would load the tracked
        ``orchestrator/config.yaml`` (project_root → real repo) and let
        ``acquire_next`` write ``<repo>/data/orchestrator/scheduler_state.json``
        — which the dashboard then displays as live state.  The autouse
        ``_isolate_orch_config`` fixture pins project_root to this test's
        tmp_path via ORCH_PROJECT_ROOT; assert that isolation holds and that the
        derived snapshot path stays under tmp, never the real tree.
        """
        config = OrchestratorConfig(max_per_module=1)
        assert config.project_root == tmp_path.resolve(), (
            f'project_root leaked outside tmp: {config.project_root!r} — the '
            'autouse config-isolation fixture is not active or was overridden.'
        )
        scheduler = Scheduler(config)
        snapshot_path = (
            Path(scheduler._project_root) / 'data' / 'orchestrator' / 'scheduler_state.json'
        )
        assert snapshot_path.is_relative_to(tmp_path.resolve())

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
# Step-15: _write_state_snapshot_raw()
# ===========================================================================

class TestWriteStateSnapshot:
    """Scheduler._write_state_snapshot_raw() writes a valid JSON file atomically."""

    def test_write_state_snapshot_creates_valid_json(self, tmp_path):
        """_write_state_snapshot_raw writes the snapshot to disk as parseable JSON."""
        scheduler = Scheduler(OrchestratorConfig(max_per_module=1))
        path = tmp_path / 'scheduler_state.json'
        scheduler._write_state_snapshot_raw(path)

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
        scheduler._write_state_snapshot_raw(path)
        # Second write: clear skip_count and add a different sentinel.
        scheduler._skip_count.clear()
        scheduler._skip_count['second'] = 2
        scheduler._write_state_snapshot_raw(path)
        second_content = json.loads(path.read_text())

        assert 'first' not in second_content['skip_counts'], (
            'Second write must have overwritten the first completely'
        )
        assert second_content['skip_counts'].get('second') == 2

    def test_write_state_snapshot_creates_parent_dirs(self, tmp_path):
        """_write_state_snapshot_raw creates missing parent directories."""
        scheduler = Scheduler(OrchestratorConfig(max_per_module=1))
        path = tmp_path / 'deep' / 'nested' / 'dir' / 'scheduler_state.json'
        scheduler._write_state_snapshot_raw(path)
        assert path.exists(), (
            'Expected snapshot file to be created even when parent dirs are missing'
        )

    def test_write_state_snapshot_raw_propagates_disk_errors(
        self, tmp_path, monkeypatch
    ):
        """_write_state_snapshot_raw propagates OSError from os.replace (does not swallow).

        This pins the corrected exception-propagation contract: the private
        primitive must let disk errors bubble up to the caller
        (_write_snapshot_best_effort), which swallows them in its own
        try/except.  Swallowing here would silently advance bookkeeping for a
        write that never persisted.
        """
        import orchestrator.scheduler as scheduler_module

        scheduler = Scheduler(OrchestratorConfig(max_per_module=1))
        path = tmp_path / 'scheduler_state.json'

        # Inject a disk-full error at the os.replace boundary so the test
        # exercises the propagation path inside the primitive itself.
        def _boom(*_args, **_kw):
            raise OSError('disk full')

        monkeypatch.setattr(scheduler_module.os, 'replace', _boom)

        with pytest.raises(OSError):
            scheduler._write_state_snapshot_raw(path)


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
        """If _write_state_snapshot_raw raises, acquire_next still dispatches tasks."""
        config = OrchestratorConfig(max_per_module=1)
        scheduler = Scheduler(config)
        scheduler._project_root = str(tmp_path)

        task_a = _pending_task('A', files=['mod_a'])
        scheduler.get_tasks = AsyncMock(return_value=[task_a])

        # Force _write_state_snapshot_raw to raise so we test fault isolation.
        from unittest.mock import MagicMock
        scheduler._write_state_snapshot_raw = MagicMock(side_effect=RuntimeError('disk full'))

        result = await scheduler.acquire_next()
        # The dispatch must still succeed despite the snapshot write failure.
        assert result is not None, (
            'acquire_next must return a TaskAssignment even if snapshot write fails'
        )
        assert result.task_id == 'A'


# ===========================================================================
# Step-1334: _write_snapshot_best_effort short-circuits when project_root unset
# ===========================================================================

class TestWriteSnapshotBestEffortProjectRootGuard:
    """_write_snapshot_best_effort skips the write when _project_root is unset."""

    @pytest.mark.asyncio
    async def test_write_snapshot_best_effort_skips_when_config_project_root_none(
        self, tmp_path, monkeypatch
    ):
        """Construction-path repro: str(None) produces '_project_root == "None"';
        the snapshot write must be refused rather than creating ./None/ on disk.

        _write_state_snapshot_raw is intentionally NOT mocked here so that the
        directory-existence assertion is a genuine end-to-end regression check:
        if the guard were removed, _write_state_snapshot_raw would call
        path.parent.mkdir(parents=True, exist_ok=True) and create
        tmp_path/None/data/orchestrator/, making the assertion fail.
        """
        config = OrchestratorConfig(max_per_module=1)
        # Bypass pydantic validate_assignment=True: field_validator raises for
        # config.project_root = None, so use object.__setattr__ to inject None
        # directly, faithfully reproducing the str(None) path in scheduler.py:692.
        object.__setattr__(config, 'project_root', None)

        scheduler = Scheduler(config)
        # Sanity: document the exact literal that str(None) produces.
        assert scheduler._project_root == 'None', (
            f"Expected '_project_root' to be the literal string 'None', "
            f"got {scheduler._project_root!r}"
        )

        monkeypatch.chdir(tmp_path)

        await scheduler._write_snapshot_best_effort()

        # Real _write_state_snapshot_raw would create tmp_path/None/data/orchestrator/
        # if the guard were absent; its non-existence proves the guard fired.
        assert not (tmp_path / 'None').exists(), (
            "A directory named 'None' must not be created under the CWD "
            "when project_root is unset"
        )

    @pytest.mark.asyncio
    @pytest.mark.parametrize('bad_root', ['None', '', None])
    async def test_write_snapshot_best_effort_skips_for_falsy_project_root(
        self, bad_root, tmp_path, monkeypatch
    ):
        """Guard covers the literal 'None', empty string, and None itself.

        _write_state_snapshot_raw is mocked so assert_not_called() is the sole
        genuine assertion: it verifies the guard fires before the write is
        attempted for each invalid _project_root value.
        """
        from unittest.mock import MagicMock

        scheduler = Scheduler(OrchestratorConfig(max_per_module=1))
        # Directly override _project_root, matching the test_scheduler_state.py
        # direct-set pattern used at lines 565 and 586.
        scheduler._project_root = bad_root

        scheduler._write_state_snapshot_raw = MagicMock()
        monkeypatch.chdir(tmp_path)

        await scheduler._write_snapshot_best_effort()

        # assert_not_called() is the genuine regression check: the guard must
        # prevent _write_state_snapshot_raw from being invoked for all bad root values.
        scheduler._write_state_snapshot_raw.assert_not_called()

    @pytest.mark.asyncio
    async def test_project_root_guard_logs_once_and_does_not_raise(
        self, caplog: pytest.LogCaptureFixture
    ):
        """Guard emits a WARNING exactly once across repeated calls (dedup).

        The new instance flag ``_snapshot_guard_warned`` must suppress
        duplicate log lines on subsequent guard trips — a misconfigured
        project_root is a static deployment issue, not a per-tick event.
        Best-effort no-raise contract must also hold: the guard returns
        without raising, and _write_state_snapshot_raw is never called.
        """
        import logging
        from unittest.mock import MagicMock

        scheduler = Scheduler(OrchestratorConfig(max_per_module=1))
        scheduler._project_root = 'None'
        scheduler._write_state_snapshot_raw = MagicMock()

        with caplog.at_level(logging.WARNING, logger='orchestrator.scheduler'):
            await scheduler._write_snapshot_best_effort()
            await scheduler._write_snapshot_best_effort()

        # (a) Guard still short-circuits — _write_state_snapshot_raw never called.
        scheduler._write_state_snapshot_raw.assert_not_called()
        # (b) Exactly ONE WARNING across BOTH calls (dedup via instance flag).
        matching = [
            r for r in caplog.records
            if r.levelno == logging.WARNING
            and r.name == 'orchestrator.scheduler'
            and ('project_root' in r.getMessage() or 'snapshot' in r.getMessage())
        ]
        assert len(matching) == 1, (
            f'Expected exactly 1 WARNING (dedup), got {len(matching)}: '
            f'{[r.getMessage() for r in caplog.records]}'
        )


# ===========================================================================
# Task-1332: _write_snapshot_best_effort throttle / coalesce
# ===========================================================================

class TestSnapshotWriteThrottle:
    """Leading-edge time throttle coalesces writes within one interval window."""

    @pytest.mark.asyncio
    async def test_throttle_coalesces_ticks_within_interval(self, tmp_path):
        """K ticks in the same throttle window produce exactly 1 disk write.

        The first-ever write always proceeds (no prior timestamp).  The
        subsequent K-1 ticks all fall within the same throttle window and must
        be coalesced to zero extra writes.

        The lock is released between each tick so task A is eligible every
        time, ensuring acquire_next reaches _write_snapshot_best_effort on
        every tick.  Note: the dedup payload happens to be byte-identical
        across all K ticks (task A is the holder each time), so both the
        time-throttle gate and the content-dedup gate coincide here — both
        mechanisms independently produce the same outcome (1 write).  The
        dedicated ``test_content_identical_payload_skips_disk_write`` isolates
        the content-dedup path; this test exercises the time-throttle gate.
        """
        from unittest.mock import AsyncMock, MagicMock

        clock = {'t': 0.0}
        config = OrchestratorConfig(max_per_module=1)
        config.snapshot_min_write_interval_secs = 1.0  # wide window

        scheduler = Scheduler(config, time_source=lambda: clock['t'])
        scheduler._project_root = str(tmp_path)

        # Provide a single pending task that can be acquired each tick.
        task_a = _pending_task('A', files=['mod_a'])
        scheduler.get_tasks = AsyncMock(return_value=[task_a])

        # Mock the disk-write seam to count real write attempts.
        scheduler._write_state_snapshot_raw = MagicMock()

        # Run K=5 ticks WITHOUT advancing the clock.
        # Release the lock between ticks so A is eligible each time.
        # All ticks fall within [0.0, 0.0 + 1.0) — the first write proceeds
        # (no prior ts), the next 4 must be coalesced.
        K = 5
        for _ in range(K):
            await scheduler.acquire_next()
            scheduler.release('A')  # make A eligible for the next tick

        assert scheduler._write_state_snapshot_raw.call_count == 1, (
            f'Expected 1 disk write for {K} ticks within one throttle window, '
            f'got {scheduler._write_state_snapshot_raw.call_count}'
        )

    @pytest.mark.asyncio
    async def test_acceptance_bound_writes_bounded_under_burst(self, tmp_path):
        """Over N ticks, disk writes == ceil(N * tick_interval / throttle_interval).

        Simulates N=20 ticks at tick_interval=0.05 s with a
        throttle_interval=0.25 s.  The acceptance bound is
        ceil(20 * 0.05 / 0.25) = ceil(4.0) = 4.

        Content-dedup is defeated by mutating a unique _skip_count key per
        tick so every payload differs — making the time-throttle gate the
        SOLE mechanism bounding writes.  The exact-equality assertion guards
        against future time-gate off-by-one regressions that the former
        1 ≤ count ≤ upper_bound assertion would have missed.

        The clock is computed as ``(i+1) * tick_interval`` (not accumulated
        via ``+=``) to eliminate IEEE-754 accumulation drift: repeated
        addition shifts the 3rd/4th boundary write one tick later
        (ticks 12/17 instead of 11/16), causing a mismatch between the
        ``ceil()`` formula and the simulated write positions.  With the
        multiplicative form the drift is removed and the trace is exact:
        writes occur on ticks 1, 6, 11, 16 → exactly 4.
        """
        import math
        from unittest.mock import AsyncMock, MagicMock

        throttle_interval = 0.25
        tick_interval = 0.05
        N = 20

        clock = {'t': 0.0}
        config = OrchestratorConfig(max_per_module=1)
        config.snapshot_min_write_interval_secs = throttle_interval

        scheduler = Scheduler(config, time_source=lambda: clock['t'])
        scheduler._project_root = str(tmp_path)

        task_a = _pending_task('A', files=['mod_a'])
        scheduler.get_tasks = AsyncMock(return_value=[task_a])
        scheduler._write_state_snapshot_raw = MagicMock()

        # Defeat content-dedup via _build_snapshot_payload: acquire_next() cleans
        # up _skip_count entries for task IDs absent from tasks_by_id, so injecting
        # fake _skip_count keys no longer works.  Instead, replace _build_snapshot_payload
        # with a counter that always returns a unique string, making the time-throttle
        # gate the SOLE mechanism bounding writes.
        _dedup_counter = [0]
        def _unique_payload(state=None):  # noqa: E301
            _dedup_counter[0] += 1
            return str(_dedup_counter[0])
        scheduler._build_snapshot_payload = _unique_payload

        for i in range(N):
            clock['t'] = (i + 1) * tick_interval  # multiplicative: no accumulation drift
            await scheduler.acquire_next()
            scheduler.release('A')

        upper_bound = math.ceil(N * tick_interval / throttle_interval)
        count = scheduler._write_state_snapshot_raw.call_count
        assert count == upper_bound, (
            f'Expected exactly {upper_bound} disk writes (time-throttle gate is '
            f'the sole bound; content-dedup defeated by per-tick unique state), '
            f'got {count}'
        )

    @pytest.mark.asyncio
    async def test_flush_state_snapshot_bypasses_throttle(self, tmp_path):
        """flush_state_snapshot() forces a write even within the throttle window.

        After throttled ticks, a flush must produce exactly one additional
        write, proving the explicit final-flush capability persists the most
        recent state regardless of the throttle interval.
        """
        from unittest.mock import AsyncMock, MagicMock

        clock = {'t': 0.0}
        config = OrchestratorConfig(max_per_module=1)
        config.snapshot_min_write_interval_secs = 1.0  # wide window

        scheduler = Scheduler(config, time_source=lambda: clock['t'])
        scheduler._project_root = str(tmp_path)

        task_a = _pending_task('A', files=['mod_a'])
        scheduler.get_tasks = AsyncMock(return_value=[task_a])
        scheduler._write_state_snapshot_raw = MagicMock()

        # First tick writes (no prior timestamp).
        await scheduler.acquire_next()
        scheduler.release('A')
        # Three more ticks — all throttled (clock not advanced).
        for _ in range(3):
            await scheduler.acquire_next()
            scheduler.release('A')
        # Exactly 1 write so far (throttle held the rest).
        assert scheduler._write_state_snapshot_raw.call_count == 1

        # Record count before flush, then flush.  The clock is still at 0.0
        # (well within the 1.0 s window) so only force=True can bypass it.
        count_before = scheduler._write_state_snapshot_raw.call_count
        await scheduler.flush_state_snapshot()

        assert scheduler._write_state_snapshot_raw.call_count == count_before + 1, (
            'flush_state_snapshot() must produce exactly 1 additional write '
            'even when the throttle window has not elapsed'
        )

    @pytest.mark.asyncio
    async def test_content_identical_payload_skips_disk_write(self, tmp_path):
        """With throttle disabled (0.0), identical payload skips the disk write.

        Sequence:
        1. First _write_snapshot_best_effort call → payload captured, count=1.
        2. Second call with no state change → payload byte-identical → skip,
           count stays 1.
        3. Mutate state (scheduler._skip_count['Q'] = 1) and call again →
           payload changed → write, count=2.

        This test isolates the content-diff path from the time-throttle path
        by disabling the throttle (interval=0.0).
        """
        from unittest.mock import MagicMock

        clock = {'t': 0.0}
        config = OrchestratorConfig(max_per_module=1)
        config.snapshot_min_write_interval_secs = 0.0  # throttle disabled

        scheduler = Scheduler(config, time_source=lambda: clock['t'])
        scheduler._project_root = str(tmp_path)
        scheduler._write_state_snapshot_raw = MagicMock()

        # First write: no prior payload → always writes.
        clock['t'] = 1.0
        await scheduler._write_snapshot_best_effort()
        assert scheduler._write_state_snapshot_raw.call_count == 1

        # Second write: state unchanged → payload byte-identical → skip.
        clock['t'] = 2.0
        await scheduler._write_snapshot_best_effort()
        assert scheduler._write_state_snapshot_raw.call_count == 1, (
            'Expected disk write to be skipped for byte-identical payload, '
            f'got call_count={scheduler._write_state_snapshot_raw.call_count}'
        )

        # Third write: state mutated → payload differs → writes.
        scheduler._skip_count['Q'] = 1
        clock['t'] = 3.0
        await scheduler._write_snapshot_best_effort()
        assert scheduler._write_state_snapshot_raw.call_count == 2, (
            'Expected disk write after state mutation (changed payload), '
            f'got call_count={scheduler._write_state_snapshot_raw.call_count}'
        )

    @pytest.mark.asyncio
    async def test_concurrent_flush_and_tick_do_not_overlap_on_disk(self, tmp_path):
        """Concurrent tick and flush invocations must never write to disk in parallel.

        Without a lock, both coroutines can pass the time-gate simultaneously
        (the gate reads _last_snapshot_write_ts before the await, so both see
        the stale value), reach asyncio.to_thread(_write_state_snapshot_raw, …)
        concurrently, and write to the same .json.tmp path from two threads.

        With the lock, only one coroutine holds the gate at a time, so
        _write_state_snapshot_raw is always called serially and max in-flight == 1.
        """
        import threading
        import time as _time
        from unittest.mock import MagicMock

        clock = {'t': 0.0}
        config = OrchestratorConfig(max_per_module=1)
        config.snapshot_min_write_interval_secs = 0.0  # throttle disabled so gate never short-circuits

        scheduler = Scheduler(config, time_source=lambda: clock['t'])
        scheduler._project_root = str(tmp_path)

        # Spy on _write_state_snapshot_raw: tracks how many calls are in-flight simultaneously.
        spy_lock = threading.Lock()
        in_flight = {'current': 0, 'max': 0}

        def spy_write(path, payload):
            with spy_lock:
                in_flight['current'] += 1
                if in_flight['current'] > in_flight['max']:
                    in_flight['max'] = in_flight['current']
            _time.sleep(0.02)  # widen the window so concurrent calls overlap
            with spy_lock:
                in_flight['current'] -= 1

        scheduler._write_state_snapshot_raw = MagicMock(side_effect=spy_write)

        # Launch several concurrent invocations: mix of tick (force=False) and
        # flush (force=True).  Each mutates a unique _skip_count key so payloads
        # differ and content-dedup never coalesces them.
        N = 6
        async def invoke(idx: int):
            scheduler._skip_count[f'X{idx}'] = idx
            force = idx % 2 == 1  # alternate tick / flush
            await scheduler._write_snapshot_best_effort(force=force)

        import asyncio as _asyncio
        await _asyncio.gather(*[invoke(i) for i in range(N)])

        assert in_flight['max'] == 1, (
            f'Expected max in-flight _write_state_snapshot_raw calls == 1 (serialised), '
            f'got {in_flight["max"]} — lock is missing or not covering the critical section'
        )


# ===========================================================================
# Task-1341: override-store swallow sites log warnings on failure
# ===========================================================================

class TestStateSnapshotOverrideStoreLogging:
    """get_state_snapshot() logs a WARNING when the override store call fails."""

    def test_get_pin_queue_failure_logs_warning_and_returns_empty_pin_queue(
        self, caplog: pytest.LogCaptureFixture
    ):
        """pin_queue path: exception must be logged, fallback [] preserved.

        scheduler.py:2342-2343 is currently a bare ``except Exception: pass``
        that swallows failures silently.  This test confirms a WARNING is
        emitted with exc_info and the empty-list fallback is returned.
        """
        import logging
        from unittest.mock import MagicMock

        scheduler = Scheduler(OrchestratorConfig(max_per_module=1))
        mock_store = MagicMock()
        mock_store.get_pin_queue.side_effect = RuntimeError('boom')
        mock_store.get_overrides.return_value = {}
        scheduler._override_store = mock_store

        with caplog.at_level(logging.WARNING, logger='orchestrator.scheduler'):
            snap = scheduler.get_state_snapshot()

        # (a) Fallback preserved: pin_queue is empty list.
        assert snap['pin_queue'] == [], (
            f"Expected pin_queue == [], got {snap['pin_queue']!r}"
        )
        # (b) Overrides fallback also preserved.
        assert snap['overrides'] == {}, (
            f"Expected overrides == {{}}, got {snap['overrides']!r}"
        )
        # (c) Exactly one WARNING record from orchestrator.scheduler mentioning
        # get_pin_queue or pin_queue.
        matching = [
            r for r in caplog.records
            if r.levelno == logging.WARNING
            and r.name == 'orchestrator.scheduler'
            and ('get_pin_queue' in r.getMessage() or 'pin_queue' in r.getMessage())
        ]
        assert len(matching) == 1, (
            f'Expected exactly 1 WARNING mentioning pin_queue, '
            f'got {len(matching)}: {[r.getMessage() for r in caplog.records]}'
        )
        # (d) exc_info must carry the RuntimeError (proves exc_info=True was used).
        record = matching[0]
        assert record.exc_info is not None, 'Expected exc_info to be set on the WARNING record'
        assert record.exc_info[0] is RuntimeError, (
            f'Expected exc_info[0] == RuntimeError, got {record.exc_info[0]!r}'
        )

    def test_get_overrides_failure_logs_warning_and_returns_empty_overrides(
        self, caplog: pytest.LogCaptureFixture
    ):
        """overrides path: exception must be logged, fallback {} preserved.

        scheduler.py:2358-2359 is currently a bare ``except Exception: pass``
        that swallows failures silently.  This test confirms a WARNING is
        emitted with exc_info and the empty-dict fallback is returned.
        """
        import logging
        from unittest.mock import MagicMock

        scheduler = Scheduler(OrchestratorConfig(max_per_module=1))
        mock_store = MagicMock()
        # pin_queue returns an empty iterator (the for-loop consumes it fine).
        mock_store.get_pin_queue.return_value = iter([])
        mock_store.get_overrides.side_effect = RuntimeError('boom')
        scheduler._override_store = mock_store

        with caplog.at_level(logging.WARNING, logger='orchestrator.scheduler'):
            snap = scheduler.get_state_snapshot()

        # (a) Fallback preserved: overrides is empty dict.
        assert snap['overrides'] == {}, (
            f"Expected overrides == {{}}, got {snap['overrides']!r}"
        )
        # (b) pin_queue fallback also preserved.
        assert snap['pin_queue'] == [], (
            f"Expected pin_queue == [], got {snap['pin_queue']!r}"
        )
        # (c) Exactly one WARNING record from orchestrator.scheduler mentioning
        # get_overrides or overrides.
        matching = [
            r for r in caplog.records
            if r.levelno == logging.WARNING
            and r.name == 'orchestrator.scheduler'
            and ('get_overrides' in r.getMessage() or 'overrides' in r.getMessage())
        ]
        assert len(matching) == 1, (
            f'Expected exactly 1 WARNING mentioning overrides, '
            f'got {len(matching)}: {[r.getMessage() for r in caplog.records]}'
        )
        # (d) exc_info must carry the RuntimeError (proves exc_info=True was used).
        record = matching[0]
        assert record.exc_info is not None, 'Expected exc_info to be set on the WARNING record'
        assert record.exc_info[0] is RuntimeError, (
            f'Expected exc_info[0] == RuntimeError, got {record.exc_info[0]!r}'
        )
